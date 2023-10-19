import os
import tqdm
import yaml
import hydra
import random
import logging
import datetime
import numpy as np
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
from molscore.manager import MolScore

import torch
from torch.distributions.kl import kl_divergence
from tensordict import TensorDict

from torchrl.envs import (
    CatFrames,
    InitTracker,
    StepCounter,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.record.loggers import get_logger

from models import get_model_factory
from rl_environments.de_novo_env_td import DeNovoEnv
from vocabulary import DeNovoVocabulary
from utils import create_batch_from_replay_smiles
from wip.writer import TensorDictMaxValueWriter
from transforms import SMILESReward, PenaliseRepeatedSMILES

logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path=".", config_name="ppo_config", version_base="1.2")
def main(cfg: "DictConfig"):

    # Save config
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
    save_dir = f"{cfg.log_dir}_{timestamp_str}"
    os.makedirs(save_dir)
    with open(Path(save_dir) / "ppo_config.yaml", 'w') as yaml_file:
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
    ckpt = torch.load(Path(__file__).resolve().parent / "vocabulary" / "priors" / "zinc_vocabulary.prior")
    vocabulary = DeNovoVocabulary.from_ckpt(ckpt)
    env_kwargs = {
        "start_token": vocabulary.encode_token("^"),
        "end_token": vocabulary.encode_token("$"),
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
    }

    # Models
    ####################################################################################################################

    create_model = get_model_factory(cfg.model)
    ckpt = torch.load(Path(__file__).resolve().parent / "models" / "priors" / "zinc_actor_critic.prior")
    (actor_inference, actor_training, critic_inference, critic_training, *transforms
     ) = create_model(vocabulary_size=len(vocabulary), ckpt=ckpt, batch_size=cfg.num_envs)

    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)
    critic_training = critic_training.to(device)
    prior = deepcopy(actor_training)

    # Environment
    ####################################################################################################################

    def create_env_fn():
        """Create a single RL rl_environments."""
        env = DeNovoEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(UnsqueezeTransform(in_keys=["observation"], out_keys=["observation"], unsqueeze_dim=-1))
        env.append_transform(CatFrames(N=100, dim=-1, padding="same", in_keys=["observation"], out_keys=["SMILES"]))
        env.append_transform(CatFrames(N=100, dim=-1, padding="zeros", in_keys=["observation"], out_keys=["SMILES2"]))
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        for transform in transforms:
            env.append_transform(transform)
        return env

    # TODO: redirect molscore logging to log dir

    scoring = MolScore(model_name="ppo", task_config=cfg.molscore)
    scoring.configs["output_dir"] = save_dir
    scoring_function = scoring.score
    rew_transform = SMILESReward(reward_function=scoring_function, vocabulary=vocabulary)

    # TODO: here check model-environment compatibility

    # Collector
    ####################################################################################################################

    collector = SyncDataCollector(
        policy=actor_inference,
        create_env_fn=create_env_fn,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        storing_device=device,
        device=device,
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
    # adv_module.set_keys(reward="penalised_reward")
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
    # loss_module.set_keys(reward="penalised_reward")

    # Buffers
    ####################################################################################################################

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.num_envs, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.mini_batch_size,
        prefetch=2,
    )

    diversity_buffer1 = TensorDictReplayBuffer(
        storage=LazyTensorStorage(100_000, device=device),
    )
    diversity_buffer2 = TensorDictReplayBuffer(
        storage=LazyTensorStorage(100_000, device=device),
    )

    penalty_transform = PenaliseRepeatedSMILES(
        diversity_buffer=diversity_buffer2,
        duplicate_key="SMILES",
        in_key="reward",
        out_key="penalised_reward",
        penalty=0.5,
    )

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.lr,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
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
    repeated_smiles = 0
    collected_frames = 0
    kl_coef = cfg.kl_coef
    ppo_epochs = cfg.ppo_epochs
    max_grad_norm = cfg.max_grad_norm
    pbar = tqdm.tqdm(total=cfg.total_frames)
    num_mini_batches = cfg.num_envs // cfg.mini_batch_size
    losses = TensorDict({}, batch_size=[cfg.ppo_epochs, num_mini_batches])
    replay_losses = TensorDict({}, batch_size=[cfg.ppo_epochs, num_mini_batches])

    for data in collector:

        log_info = {}
        frames_in_batch = data.numel()
        total_done += data.get(("next", "terminated")).sum()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Compute all rewards in a single call
        data = rew_transform(data)

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
        data = penalty_transform(data)
        unique_smiles = total_done - repeated_smiles

        episode_rewards = data["next", "penalised_reward"][data["next", "terminated"]]
        log_info.update(
            {
                "train/penalised_reward": episode_rewards.mean().item(),
                "train/penalised_min_reward": episode_rewards.min().item(),
                "train/penalised_max_reward": episode_rewards.max().item(),
            }
        )

        for j in range(ppo_epochs):

            with torch.no_grad():
                data = adv_module(data)

            buffer.extend(data)

            for i in range(num_mini_batches):

                # Compute loss for the current mini-batch
                batch = buffer.sample()
                loss = loss_module(batch)
                loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                losses[j, i] = loss.select("loss_critic", "loss_entropy", "loss_objective").detach()
                kl_div = kl_divergence(actor_training.get_dist(batch), prior.get_dist(batch))
                mask = torch.isnan(kl_div) | torch.isinf(kl_div)
                kl_div = kl_div[~mask].mean()
                loss_sum += kl_div * kl_coef
                losses[j, i] = TensorDict({"kl_div": kl_div.detach().item()}, batch_size=[])

                # Backward pass
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=max_grad_norm
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

