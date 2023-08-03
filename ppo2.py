import tqdm
import hydra
import torch
from pathlib import Path
from copy import deepcopy

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
from torchrl.envs.libs.gym import GymWrapper
from torchrl.record.loggers import get_logger
from torchrl.modules import ProbabilisticActor
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value.advantages import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from env import GenChemEnv
from utils import create_shared_model
from reward_transform import SMILESReward
from scoring import WrapperScoringClass
# from writer import TensorDictMaxValueWriter

# TODO: add fps logging
# TODO: add smiles logging
# TODO: how to combine clipping and KL penalty?
# TODO: add KL penalty to the loss or to the reward
# TODO: add batched scoring as a buffer transform

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):

    seed = 101

    # Set a seed for the random module
    import random
    random.seed(int(seed))

    # Set a seed for the numpy module
    import numpy as np
    np.random.seed(int(seed))

    # Set a seed for the torch module
    torch.manual_seed(int(seed))

    device = torch.device(cfg.device) if torch.cuda.is_available() else torch.device("cpu")

    scoring = WrapperScoringClass()
    vocabulary = torch.load(Path(__file__).resolve().parent / "priors" / "vocabulary.prior")
    env_kwargs = {"scoring_function": scoring.get_final_score, "vocabulary": vocabulary}

# Models
    ####################################################################################################################

    test_env = GymWrapper(GenChemEnv(**env_kwargs))
    action_spec = test_env.action_spec
    observation_spec = test_env.observation_spec

    actor, critic, critic_head, rhs_transform = create_shared_model(vocabulary=vocabulary, output_size=action_spec.shape[-1])

    ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "actor.prior")

    aaa = {
        "module.0.module._embedding.weight": "module.0.module.0.module._embedding.weight",
        "module.1.lstm.weight_ih_l0": "module.0.module.1.lstm.weight_ih_l0",
        "module.1.lstm.weight_hh_l0": "module.0.module.1.lstm.weight_hh_l0",
        "module.1.lstm.bias_ih_l0": "module.0.module.1.lstm.bias_ih_l0",
        "module.1.lstm.bias_hh_l0": "module.0.module.1.lstm.bias_hh_l0",
        "module.1.lstm.weight_ih_l1": "module.0.module.1.lstm.weight_ih_l1",
        "module.1.lstm.weight_hh_l1": "module.0.module.1.lstm.weight_hh_l1",
        "module.1.lstm.bias_ih_l1": "module.0.module.1.lstm.bias_ih_l1",
        "module.1.lstm.bias_hh_l1": "module.0.module.1.lstm.bias_hh_l1",
        "module.1.lstm.weight_ih_l2": "module.0.module.1.lstm.weight_ih_l2",
        "module.1.lstm.weight_hh_l2": "module.0.module.1.lstm.weight_hh_l2",
        "module.1.lstm.bias_ih_l2": "module.0.module.1.lstm.bias_ih_l2",
        "module.1.lstm.bias_hh_l2": "module.0.module.1.lstm.bias_hh_l2",
        "module.2.module.0.weight": "module.1.module.0.weight",
        "module.2.module.0.bias": "module.1.module.0.bias",
    }

    new_ckpt = {}
    for k, v in ckpt.items(): new_ckpt[aaa[k]] = v

    actor.load_state_dict(new_ckpt)

    actor_prior = deepcopy(actor)
    actor_prior = actor_prior.to(device)
    actor = actor.to(device)
    critic = critic.to(device)

    # Environment
    ####################################################################################################################

    # # hack because it is not allowed to have 2 equal transforms
    # for k, v in rhs_transform_critic.primers.items():
    #     rhs_transform_actor.primers[k] = v

    def create_transformed_env():
        env = GymWrapper(GenChemEnv(**env_kwargs), categorical_action_encoding=True, device=device)
        env = TransformedEnv(env)
        env.append_transform(rhs_transform.clone())
        return env

    def create_env_fn(num_workers=cfg.num_env_workers):
        # env = ParallelEnv(  # There is some bug here! When using it hidden states are always zero
        env = SerialEnv(  # This works!
            create_env_fn=create_transformed_env,
            num_workers=num_workers,
        )
        env = TransformedEnv(env)
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        env.append_transform(CatFrames(N=100, dim=-1, in_keys=["observation"], out_keys=["SMILES"]))
        return env

    # Loss modules
    ####################################################################################################################

    adv_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic,
        average_gae=True,
        shifted=True,
    )
    adv_module = adv_module.to(device)
    loss_module = ClipPPOLoss(
        actor, critic,
        critic_coef=cfg.critic_coef,
        entropy_coef=cfg.entropy_coef,
        clip_epsilon=cfg.ppo_clip,
        loss_critic_type="l2",
    )
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
    kl_transform = KLRewardTransform(actor_prior, coef=cfg.kl_coef, out_keys="reward")
    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.num_env_workers, device=device),
        sampler=sampler,
        batch_size=cfg.mini_batch_size,
        prefetch=2,
    )

    # topSMILESBuffer = TensorDictReplayBuffer(
    #     storage=LazyTensorStorage(100, device=device),
    #     sampler=sampler,
    #     batch_size=cfg.mini_batch_size,
    #     prefetch=2,
    # )

    diversity_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(100_000, device=device),
    )

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
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
    repeated_smiles = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)

    for data in collector:

        frames_in_batch = data.numel()
        total_done += data.get(("next", "done")).sum()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Log end-of-episode accumulated rewards for training
        episode_rewards = data["next", "reward"][data["next", "done"]]
        if logger is not None and len(episode_rewards) > 0:
            logger.log_scalar(
                "reward_training", episode_rewards.mean().item(), collected_frames
            )
            logger.log_scalar(
                "min_reward_training", episode_rewards.min().item(), collected_frames
            )
            logger.log_scalar(
                "max_reward_training", episode_rewards.max().item(), collected_frames
            )
            logger.log_scalar(
                "total_smiles", total_done, collected_frames
            )
            logger.log_scalar(
                "repeated_smiles", repeated_smiles, collected_frames
            )

        # Apply reward augmentation
        data = kl_transform(data)

        # Penalize repeated SMILES
        td = data.get("next")
        done = td.get("done").squeeze(-1)
        sub_td = td.get_sub_tensordict(done)
        reward = sub_td.pop("reward")
        finished_smiles = sub_td.get("SMILES")
        finished_smiles_td = sub_td.select("SMILES")
        num_unique_smiles = len(diversity_buffer)
        num_finished_smiles = len(finished_smiles_td)

        if num_finished_smiles > 0 and num_unique_smiles == 0:
            diversity_buffer.extend(finished_smiles_td.clone())

        elif num_finished_smiles > 0:
            for i, smi in enumerate(finished_smiles):
                td_smiles = diversity_buffer._storage._storage
                unique_smiles = td_smiles.get("_data").get("SMILES")[0:num_unique_smiles]
                repeated = (smi == unique_smiles).all(dim=-1).any()
                if repeated:
                    reward[i] = reward[i] * 0.5
                    repeated_smiles += 1
                else:
                    # diversity_buffer.extend(finished_smiles_td[i:i+1].clone())  # TODO: is clone necessary?
                    diversity_buffer.add(finished_smiles_td[i].clone())  # TODO: is clone necessary?
                    num_unique_smiles += 1
        sub_td.set("reward", reward, inplace=True)

        with torch.no_grad():
            data = adv_module(data)

        for j in range(cfg.ppo_epochs):

            # it is important to pass data that is not flattened
            buffer.extend(data)

            for i, batch in enumerate(buffer):

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

        collector.update_policy_weights_()

    collector.shutdown()


if __name__ == "__main__":
    main()
