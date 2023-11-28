import os
import tqdm
import yaml
import json
import hydra
import shutil
import random
import logging
import datetime
import numpy as np
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
from molscore.manager import MolScore

import torch
from tensordict import TensorDict

from torchrl.envs import (
    CatFrames,
    InitTracker,
    StepCounter,
    TransformedEnv,
    UnsqueezeTransform,
    RandomCropTensorDict,
)
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import DiscreteSACLoss, SoftUpdate
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.record.loggers import get_logger

from models import get_model_factory
from rl_environments import DeNovoEnv
from vocabulary import DeNovoVocabulary
from old.writer import TensorDictMaxValueWriter
from transforms.reward_transform import SMILESReward
from transforms.burnin_transform import BurnInTransform


logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path=".", config_name="sac_config", version_base="1.2")
def main(cfg: "DictConfig"):

    # Save config
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
    save_dir = f"{cfg.log_dir}_{timestamp_str}"
    os.makedirs(save_dir)
    with open(Path(save_dir) / "sac_config.yaml", 'w') as yaml_file:
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
    ckpt = torch.load(Path(__file__).resolve().parent / "vocabulary" / "priors" / "chembl_vocabulary.prior")
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

    (actor_inference, actor_training, critic_inference, critic_training, *transforms
     ) = get_model_factory(cfg.model)(vocabulary_size=len(vocabulary), batch_size=cfg.num_envs)

    # TODO: check inputs and outputs of models are correct

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
        env.append_transform(
            CatFrames(
                N=100, dim=-1, padding="constant", in_keys=["observation"], out_keys=["SMILES"], padding_value=-1))
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        for transform in transforms:
            env.append_transform(transform)
        return env

    # Save molscore output. Also redirect output to save_dir
    cfg.molscore = shutil.copy(cfg.molscore, save_dir)
    data = json.load(open(cfg.molscore, 'r'))
    data['output_dir'] = save_dir
    json.dump(data, open(cfg.molscore, 'w'), indent=4)

    # Create scoring function
    scoring = MolScore(model_name="sac", task_config=cfg.molscore)
    scoring.configs["save_dir"] = save_dir
    scoring_function = scoring.score

    # Create reward transform
    rew_transform = SMILESReward(reward_function=scoring_function, vocabulary=vocabulary)

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
        qvalue_network=critic_training,
        num_actions=len(vocabulary),
        num_qvalue_nets=2,
        target_entropy_weight=cfg.target_entropy_weight,
        loss_function="smooth_l1",
    )
    loss_module.make_value_estimator(gamma=cfg.gamma)
    loss_module = loss_module.to(device)
    target_net_updater = SoftUpdate(loss_module, eps=cfg.target_update_polyak)

    # Buffers
    ####################################################################################################################

    crop_seq = RandomCropTensorDict(sub_seq_len=cfg.sampled_sequence_length, sample_dim=-1)
    burn_in = BurnInTransform(lstm_module=actor_training, burn_in=cfg.burn_in)
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.replay_buffer_size),
        batch_size=cfg.batch_size,
        prefetch=3,
        sampler=RandomSampler(),
    )
    buffer.append_transform(crop_seq)
    buffer.append_transform(burn_in)


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

        # data = data.exclude(*exclude_keys).reshape(-1)
        # data = data.exclude(*exclude_keys)
        buffer.extend(data)

        batch = buffer.sample()
        batch = batch.to(device)
        import ipdb; ipdb.set_trace()
        loss_td = loss_module(batch)

        # # Burn in
        # with torch.no_grad():
        #     burn_in_batch = TensorDict({
        #         "observation": batch["observation_burn_in"].unsqueeze(-1),
        #         "is_init": batch["is_init_burn_in"].unsqueeze(-1),
        #         },
        #         device=device,
        #         batch_size=batch["observation_burn_in"].shape)
        #     import ipdb; ipdb.set_trace()
        #     burn_in_batch = actor_training(burn_in_batch)
        #     burn_in_batch = critic_training(burn_in_batch)
        #
        # import ipdb; ipdb.set_trace()
        # batch.set("recurrent_state_c_actor", burn_in_batch[:, -1].get(("next", "recurrent_state_c_actor")))
        # batch.set("recurrent_state_h_actor", burn_in_batch[:, -1].get(("next", "recurrent_state_h_actor")))

        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()

