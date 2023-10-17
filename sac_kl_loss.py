import os
import tqdm
import yaml
import hydra
import random
import logging
import numpy as np
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
from molscore.manager import MolScore

import torch
from torch.distributions.kl import kl_divergence
from tensordict import TensorDict

from torchrl.envs import (
    SerialEnv,
    CatFrames,
    ParallelEnv,
    InitTracker,
    StepCounter,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import GymWrapper
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import DiscreteSACLoss, SoftUpdate
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.record.loggers import get_logger

from models import get_model_factory
from rl_environments import DeNovoEnv
from vocabulary import DeNovoVocabulary
from utils import penalise_repeated_smiles, create_batch_from_replay_smiles
from wip.writer import TensorDictMaxValueWriter
from transforms.reward_transform import SMILESReward


logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path=".", config_name="sac_config", version_base="1.2")
def main(cfg: "DictConfig"):

    try:
        os.makedirs(cfg.log_dir)
    except FileExistsError:
        raise Exception(f"Log directory {cfg.log_dir} already exists")

    # Save config
    with open(Path(cfg.log_dir) / "ppo_config.yaml", 'w') as yaml_file:
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
    ckpt = torch.load(Path(__file__).resolve().parent / "vocabulary" / "priors" / "vocabulary.prior")
    vocabulary = DeNovoVocabulary.from_ckpt(ckpt)
    env_kwargs = {
        "start_token": vocabulary.encode_token("^"),
        "end_token": vocabulary.encode_token("$"),
        "length_vocabulary": len(vocabulary),
    }
    test_env = GymWrapper(DeNovoEnv(**env_kwargs))
    action_spec = test_env.action_spec

    # Models
    ####################################################################################################################

    (actor_inference, actor_training, critic_inference, critic_training, *transforms
     ) = get_model_factory(cfg.model)(vocabulary_size=action_spec.shape[-1])

    # TODO: check inputs and outputs of models are correct

    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)
    critic_training = critic_training.to(device)
    prior = deepcopy(actor_training)

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
        env.append_transform(CatFrames(N=10, dim=-1, padding="zeros", in_keys=["observation"], out_keys=["burn_in"]))
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        for transform in transforms:
            env.append_transform(transform.clone())
        return env

    def create_env_fn(num_workers=cfg.num_env_workers):
        """Create a vector of parallel environments."""
        env = SerialEnv(create_env_fn=create_base_env, num_workers=num_workers)
        # env = ParallelEnv(create_env_fn=create_base_env, num_workers=num_workers)
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

    loss_module = DiscreteSACLoss(
        actor_network=actor_training,
        action_space=test_env.action_spec,
        qvalue_network=critic_training,
        num_actions=len(vocabulary),
        num_qvalue_nets=2,
        target_entropy_weight=cfg.target_entropy_weight,
        loss_function="smooth_l1",
    )
    loss_module.make_value_estimator(gamma=cfg.gamma)
    loss_module = loss_module.to(device)
    loss_module.set_keys(reward="penalised_reward")
    target_net_updater = SoftUpdate(loss_module, eps=cfg.target_update_polyak)

    # Buffers
    ####################################################################################################################

    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.replay_buffer_size, device=device),
        batch_size=cfg.batch_size,
        prefetch=3,
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
    losses = TensorDict({}, batch_size=[cfg.num_loss_updates])

    kl_coef = cfg.kl_coef
    max_grad_norm = cfg.max_grad_norm

    # TODO: need to pop out current hidden states from batch
    exclude_keys = [
        ("recurrent_state_c_actor"),
        ("recurrent_state_h_actor"),
        ("recurrent_state_c_critic"),
        ("recurrent_state_h_critic"),
        ("next", "recurrent_state_c_actor"),
        ("next", "recurrent_state_h_actor"),
        ("next", "recurrent_state_c_critic"),
        ("next", "recurrent_state_h_critic"),
    ]

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

        data = data.exclude(*exclude_keys).reshape(-1)
        buffer.extend(data)

        batch = buffer.sample()
        with torch.no_grad():
            # Burn in
            import ipdb; ipdb.set_trace()
            batch = actor_training(batch)


if __name__ == "__main__":
    main()
