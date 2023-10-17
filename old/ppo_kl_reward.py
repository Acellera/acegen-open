import os
import tqdm
import yaml
import hydra
import torch
import random
import logging
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from molscore.manager import MolScore

from tensordict import TensorDict
from torchrl.envs import (
    CatFrames,
    SerialEnv,
    InitTracker,
    StepCounter,
    TransformedEnv,
    KLRewardTransform,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import GymWrapper
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value.advantages import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.record.loggers import get_logger

from rl_environments import DeNovoEnv, DeNovoVocabulary
from utils import (
    create_ppo_models,
    penalise_repeated_smiles,
    create_batch_from_replay_smiles,
)
from wip.writer import TensorDictMaxValueWriter
from transforms.reward_transform import SMILESReward


logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path="..", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):

    # Save config
    os.makedirs(cfg.log_dir)
    with open(Path(cfg.log_dir) / "config.yaml", 'w') as yaml_file:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

    # Set seeds
    seed = cfg.seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # Get available device
    device = torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")

    # Create test rl_environments to get action specs
    ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "vocabulary.prior")
    vocabulary = DeNovoVocabulary.from_ckpt(ckpt)
    env_kwargs = {"vocabulary": vocabulary}
    test_env = GymWrapper(DeNovoEnv(**env_kwargs))
    action_spec = test_env.action_spec

    # Models
    ####################################################################################################################

    (actor_inference, actor_training, critic_inference, critic_training, rhs_transform
     ) = create_ppo_models(vocabulary=vocabulary, output_size=action_spec.shape[-1])
    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)
    critic_training = critic_training.to(device)

    # Environment
    ####################################################################################################################

    def create_base_env():
        """Create a single RL rl_environments."""
        env = DeNovoEnv(**env_kwargs)
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
        """Create a vector of parallel environments."""
        env = SerialEnv(create_env_fn=create_base_env, num_workers=num_workers)
        # env = ParallelEnv(create_env_fn=create_base_env, num_workers=num_workers)
        env = TransformedEnv(env)
        env.append_transform(KLRewardTransform(actor_inference, coef=cfg.kl_coef, out_keys="reward-kl"))
        return env

    scoring = MolScore(model_name="ppo", task_config=cfg.molscore).score
    rew_transform = SMILESReward(reward_function=scoring, vocabulary=vocabulary)

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
        actor_training,
        critic_training,
        critic_coef=cfg.critic_coef,
        entropy_coef=cfg.entropy_coef,
        clip_epsilon=cfg.ppo_clip,
        loss_critic_type="l2",
        normalize_advantage=True,
    )
    loss_module = loss_module.to(device)
    loss_module.set_keys(reward="penalised_reward")

    # Buffers
    ####################################################################################################################

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.num_env_workers, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.mini_batch_size,
        prefetch=2,
    )

    top_smiles_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(100, device=device),
        sampler=SamplerWithoutReplacement(),
        prefetch=2,
        batch_size=10,
        writer=TensorDictMaxValueWriter(rank_key="penalised_reward"),
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
        logger = get_logger(
            cfg.logger_backend, logger_name="ppo", experiment_name=cfg.agent_name, project_name=cfg.experiment_name
        )

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
        log_info = {}
        frames_in_batch = data.numel()
        total_done += data.get(("next", "terminated")).sum()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Compute all rewards in a single call
        rew_transform = rew_transform(data)

        # Register smiles lengths and real rewards
        episode_rewards = data["next", "reward"][data["next", "terminated"]]
        episode_length = data["next", "step_count"][data["next", "terminated"]]
        if len(episode_rewards) > 0:
            log_info.update(
                {
                    "train/total_smiles": total_done,
                    "train/repeated_smiles": repeated_smiles,
                    "train/reward": episode_rewards.mean().item(),
                    "train/min_reward": episode_rewards.min().item(),
                    "train/max_reward": episode_rewards.max().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        # Penalise repeated smiles and register penalised rewards
        repeated_smiles = penalise_repeated_smiles(
            data,
            diversity_buffer,
            repeated_smiles,
            in_keys="reward-kl",
            out_keys="penalised_reward",
        )
        episode_rewards = data["next", "penalised_reward"][data["next", "terminated"]]
        log_info.update(
            {
                "train/penalised_reward": episode_rewards.mean().item(),
                "train/penalised_min_reward": episode_rewards.min().item(),
                "train/penalised_max_reward": episode_rewards.max().item(),
            }
        )

        # Add data to the replay buffer
        next_data = data.get("next")
        terminated = next_data.get("terminated").squeeze(-1)
        terminated_smiles = next_data.get_sub_tensordict(idx=terminated).select(
            "SMILES", "SMILES2", "penalised_reward", "step_count")
        top_smiles_buffer.extend(terminated_smiles)

        for j in range(cfg.ppo_epochs):

            with torch.no_grad():
                data = adv_module(data)

            buffer.extend(data)

            for i in range(num_mini_batches):

                # Compute loss for the current mini-batch
                batch = buffer.sample()
                loss = loss_module(batch)
                loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                losses[j, i] = loss.select("loss_critic", "loss_entropy", "loss_objective").detach()

                # Compute loss for the replay buffer batch
                replay_data = top_smiles_buffer.sample()
                cat_replay_data, split_replay_batch = create_batch_from_replay_smiles(
                    replay_data, device, reward_key="penalised_reward")
                replay_batch = cat_replay_data
                with torch.no_grad():
                    replay_batch = actor_training(replay_batch)
                    replay_batch.pop(("next", "recurrent_state_c"))
                    replay_batch.pop(("next", "recurrent_state_h"))
                    replay_batch = adv_module(replay_batch)
                rb_loss = loss_module(replay_batch)
                replay_loss_sum = rb_loss["loss_critic"] + rb_loss["loss_objective"] + rb_loss["loss_entropy"]
                replay_losses[j, i] = rb_loss.select("loss_critic", "loss_entropy", "loss_objective").detach()
                replay_losses[j, i].set(
                    "replay_mean_reward", cat_replay_data["next"][
                        "penalised_reward"][cat_replay_data["next"]["done"]].mean().item())

                # Weighted sum of the losses
                num_batch = batch.numel()
                num_replay = replay_batch.numel()
                total_smiles = num_batch + num_replay
                augmented_loss_sum = loss_sum * (
                        num_batch / total_smiles) + replay_loss_sum * (num_replay / total_smiles)

                # Backward pass
                augmented_loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=cfg.max_grad_norm
                )

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
