import hydra
import torch
import tqdm
from copy import deepcopy
from pathlib import Path

from torchrl.objectives import PPOLoss
from torchrl.envs.libs.gym import GymWrapper
from torchrl.record.loggers import get_logger
from torchrl.modules import ProbabilisticActor
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value.advantages import GAE
from torchrl.envs import (
    Compose,
    ParallelEnv,
    SerialEnv,
    TransformedEnv,
    InitTracker,
    StepCounter,
    RewardSum,
    CatFrames,
    KLRewardTransform,
)
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from env import GenChemEnv
from utils import create_model, create_simple_model, create_rhs_transform
from reward_transform import SMILESReward
from scoring import WrapperScoringClass


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: "DictConfig"):

    device = torch.device(cfg.device) if torch.cuda.is_available() else torch.device("cpu")

    # Environment
    ####################################################################################################################

    vocabulary = torch.load(Path(__file__).resolve().parent / "priors" / "vocabulary.prior")

    # Let's use a basic scoring function that gives a reward of 1.0 if the SMILES is valid and 0.0 otherwise.
    scoring = WrapperScoringClass()
    env_kwargs = {"scoring_function": scoring.get_final_score, "vocabulary": vocabulary}

    def create_transformed_env():
        env = GymWrapper(GenChemEnv(**env_kwargs), categorical_action_encoding=True)
        env = TransformedEnv(env)
        env.append_transform(create_rhs_transform())
        env.append_transform(StepCounter())
        env.append_transform(RewardSum())
        env.append_transform(InitTracker())
        env.append_transform(CatFrames(N=100, dim=-1, in_keys=["observation"], out_keys=["SMILES"]))
        return env

    def create_env_fn(num_workers=cfg.num_env_workers):
        # env = ParallelEnv(  # There is some bug here! When using it hidden states are always zero
        env = SerialEnv(  # This works!
            create_env_fn=create_transformed_env,
            num_workers=num_workers,
        )
        return env

    test_env = GymWrapper(GenChemEnv(**env_kwargs))
    action_spec = test_env.action_spec
    observation_spec = test_env.observation_spec

    # Models
    ####################################################################################################################

    # actor_model = create_model(vocabulary=vocabulary, output_size=action_spec.shape[-1])
    actor_model = create_simple_model(vocabulary=vocabulary, output_size=action_spec.shape[-1])
    # actor_model.load_state_dict(torch.load(Path(__file__).resolve().parent / "priors" / "actor.prior"))
    actor = ProbabilisticActor(
        module=actor_model,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )
    actor_prior = deepcopy(actor)
    actor = actor.to(device)
    # critic = create_model(vocabulary=vocabulary, output_size=1, out_key="state_value")
    critic = create_simple_model(vocabulary=vocabulary, output_size=1, out_key="state_value")
    # critic.load_state_dict(torch.load(Path(__file__).resolve().parent / "priors" / "critic.prior"))
    critic = critic.to(device)

    # Loss modules
    ####################################################################################################################

    adv_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic,
        average_gae=True,
        shifted=True,
    )
    loss_module = PPOLoss(actor, critic, critic_coef=cfg.critic_coef, entropy_coef=cfg.entropy_coef)
    loss_module = loss_module.to(device)

    # Collector
    ####################################################################################################################

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=actor,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        split_trajs=False,
    )

    # Storage
    ####################################################################################################################

    sampler = SamplerWithoutReplacement()
    # rew_transform = SMILESReward(reward_function=scoring.get_final_score, vocabulary=vocabulary)
    kl_transform = KLRewardTransform(actor_prior, coef=0.1, out_keys="reward_kl")
    transforms = Compose(
        # rew_transform,
        kl_transform,
    )
    buffer = TensorDictReplayBuffer(
        # storage=LazyTensorStorage(cfg.num_env_workers),  # TODO: ideally device should be "device"
        storage=LazyTensorStorage(cfg.num_env_workers, device=device),  # TODO: ideally device should be "device"
        sampler=sampler,
        batch_size=cfg.mini_batch_size,
        prefetch=10,
        # transform=transforms,
    )

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.lr,
        weight_decay=0.000,
    )

    # Logger
    ####################################################################################################################

    logger = None
    if cfg.logger_backend:
        logger = get_logger(cfg.logger_backend, logger_name="ppo", experiment_name=cfg.experiment_name)

    # Training loop
    ####################################################################################################################

    total_done = 0
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)

    for data in collector:

        frames_in_batch = data.numel()
        total_done += data.get(("next", "done")).sum()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Log end-of-episode accumulated rewards for training
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if logger is not None and len(episode_rewards) > 0:
            logger.log_scalar(
                "reward_training", episode_rewards.mean().item(), collected_frames
            )
            logger.log_scalar(
                "total_smiles", total_done, collected_frames
            )

        # Score the current batch
        finished_smiles = data["next", "SMILES"][data["next", "done"].squeeze()]
        for smiles in finished_smiles:
            print(vocabulary.decode_smiles(smiles.cpu().numpy()))

        for j in range(cfg.ppo_epochs):

                with torch.no_grad():
                    data = adv_module(data)

                # it is important to pass data that is not flattened
                buffer.extend(data)

                for i, batch in enumerate(buffer):

                    batch = batch.to(device)  # TODO: ideally batch should already be in "device"

                    loss = loss_module(batch)
                    loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

                    # Backward pass
                    loss_sum.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=cfg.max_grad_norm)

                    optim.step()
                    optim.zero_grad()

                    if logger is not None:
                        for key, value in loss.items():
                            logger.log_scalar(key, value.item(), collected_frames)
                        logger.log_scalar("grad_norm", grad_norm.item(), collected_frames)

    collector.shutdown()


if __name__ == "__main__":
    main()