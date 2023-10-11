import tqdm
import hydra
import torch
from copy import deepcopy
from pathlib import Path

from tensordict import TensorDict
from torchrl.envs import (
    Compose,
    ParallelEnv,
    SerialEnv,
    TransformedEnv,
    InitTracker,
    StepCounter,
    RewardSum,
    CatFrames,
    RewardScaling,
    RewardClipping,
    KLRewardTransform,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import GymWrapper
from torchrl.record.loggers import get_logger
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value.advantages import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torch.distributions.kl import kl_divergence
from env import GenChemEnv, Monitor
from utils import create_shared_model, penalise_repeated_smiles, create_batch_from_replay_smiles
from reward_transform import SMILESReward
from scoring import WrapperScoringClass
from writer import TensorDictMaxValueWriter


# TODO: investigate cat frames reaching max length
# TODO: investigate wrong step count


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
    actor_inference, actor_training, _, critic_training, rhs_transform = create_shared_model(
        vocabulary=vocabulary, output_size=action_spec.shape[-1])
    ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "actor_critic.prior")
    actor_inference.load_state_dict(ckpt)
    actor_training.load_state_dict(ckpt)
    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)
    critic_training = critic_training.to(device)
    prior = deepcopy(actor_training)

    # Environment
    ####################################################################################################################

    def create_base_env():
        env = Monitor(GenChemEnv(**env_kwargs), log_dir=cfg.log_dir)
        env = GymWrapper(env, categorical_action_encoding=True, device=device)
        env = TransformedEnv(env)
        env.append_transform(UnsqueezeTransform(in_keys=["observation"], out_keys=["observation"], unsqueeze_dim=-1))
        env.append_transform(CatFrames(N=100, dim=-1, padding="same", in_keys=["observation"], out_keys=["SMILES"]))
        env.append_transform(CatFrames(N=100, dim=-1, padding="zeros", in_keys=["observation"], out_keys=["SMILES2"]))
        env.append_transform(rhs_transform.clone())
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        return env

    def create_env_fn(num_workers=cfg.num_env_workers):
        env = ParallelEnv(create_env_fn=create_base_env, num_workers=num_workers)
        return env

    # Collector
    ####################################################################################################################

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=actor_inference,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        storing_device=device,
    )

    # Loss modules
    ####################################################################################################################

    adv_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic_training,
        average_gae=False,
        shifted=True,
    )
    adv_module.set_keys(reward="penalised_reward")
    adv_module = adv_module.to(device)
    loss_module = ClipPPOLoss(
        actor_training, critic_training,
        critic_coef=cfg.critic_coef,
        entropy_coef=cfg.entropy_coef,
        clip_epsilon=cfg.ppo_clip,
        loss_critic_type="l2",
        normalize_advantage=True,

    )
    loss_module = loss_module.to(device)
    loss_module.set_keys(reward="penalised_reward")

    # Storage
    ####################################################################################################################

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.num_env_workers, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.mini_batch_size,
        prefetch=2,
    )

    topSMILESBuffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(100, device=device),
        sampler=SamplerWithoutReplacement(),
        prefetch=2,
        batch_size=10,
        writer=TensorDictMaxValueWriter(rank_key="reward"),
    )

    diversity_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(100_000, device=device),
    )

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        eps=cfg.eps,
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
    num_mini_batches = cfg.num_env_workers // cfg.mini_batch_size
    losses = TensorDict({}, batch_size=[cfg.ppo_epochs, num_mini_batches])
    replay_losses = TensorDict({}, batch_size=[cfg.ppo_epochs, num_mini_batches])

    for data in collector:

        # data.pop("recurrent_state_c")
        # data.pop("recurrent_state_h")
        # data.get("next").pop("recurrent_state_c")
        # data.get("next").pop("recurrent_state_h")

        log_info = {}
        frames_in_batch = data.numel()
        total_done += data.get(("next", "terminated")).sum()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        episode_rewards = data["next", "reward"][data["next", "terminated"]]
        episode_length = data["next", "step_count"][data["next", "terminated"]]
        if len(episode_rewards) > 0:
            log_info.update({
                "train/reward": episode_rewards.mean().item(),
                "train/min_reward": episode_rewards.min().item(),
                "train/max_reward": episode_rewards.max().item(),
                "train/total_smiles": total_done,
                "train/repeated_smiles": repeated_smiles,
                "train/episode_length": episode_length.sum().item() / len(episode_length),
            })

        repeated_smiles = penalise_repeated_smiles(
            data, diversity_buffer, repeated_smiles, in_keys="reward", out_keys="penalised_reward")
        episode_rewards = data["next", "penalised_reward"][data["next", "terminated"]]
        log_info.update({
            "train/penalised_reward": episode_rewards.mean().item(),
            "train/penalised_min_reward": episode_rewards.min().item(),
            "train/penalised_max_reward": episode_rewards.max().item(),
        })

        # Add data to the replay buffer
        next_data = data.get("next")
        terminated = next_data.get("terminated").squeeze(-1)
        terminated_smiles = next_data.get_sub_tensordict(idx=terminated).select(
            "SMILES", "SMILES2", "reward", "penalised_reward", "step_count")
        topSMILESBuffer.extend(terminated_smiles)

        for j in range(cfg.ppo_epochs):

            with torch.no_grad():
                data = adv_module(data)

            # it is important to pass data that is not flattened
            buffer.extend(data)

            for i in range(num_mini_batches):

                batch = buffer.sample()
                loss = loss_module(batch)
                loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                losses[j, i] = loss.select("loss_critic", "loss_entropy", "loss_objective").detach()
                kl_div = kl_divergence(actor_training.get_dist(batch), prior.get_dist(batch)).mean()
                loss_sum += kl_div * cfg.kl_coef
                losses[j, i].set("kl_div", kl_div.detach())

                replay_data = topSMILESBuffer.sample()
                replay_batch = create_batch_from_replay_smiles(replay_data, device)
                with torch.no_grad():
                    replay_batch = adv_module(replay_batch)
                replay_loss = loss_module(replay_batch)
                replay_loss_sum = replay_loss["loss_critic"] + replay_loss["loss_objective"] + replay_loss["loss_entropy"]
                kl_div = kl_divergence(actor_training.get_dist(batch), prior.get_dist(batch)).mean()
                replay_loss_sum += kl_div * cfg.kl_coef

                num_batch_smiles = batch.get("next").get("terminated").sum()
                num_replay_smiles = replay_batch.get("next").get("terminated").sum()
                total_smiles = num_batch_smiles + num_replay_smiles
                augmented_loss_sum = loss_sum * (num_batch_smiles / total_smiles) + replay_loss_sum * (num_replay_smiles / total_smiles)

                replay_losses[j, i] = replay_loss.select("loss_critic", "loss_entropy", "loss_objective").detach()
                replay_losses[j, i].set("kl_div", kl_div.detach())

                # Backward pass
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=cfg.max_grad_norm)

                optim.step()
                optim.zero_grad()

        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        replay_losses_mean = replay_losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in replay_losses_mean.items():
            log_info.update({f"train/replay_{key}": value.item()})

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)
        collector.update_policy_weights_()

    collector.shutdown()


if __name__ == "__main__":
    main()
