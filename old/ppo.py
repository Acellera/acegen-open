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
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer, TensorDictPrioritizedReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.writers import TensorDictMaxValueWriter
from torchrl.record.loggers import get_logger

from models import get_model_factory
from rl_environments import MultiStepDeNovoEnv
from vocabulary.vocabulary2 import Vocabulary
from utils import create_batch_from_replay_smiles
from transforms import SMILESReward, PenaliseRepeatedSMILES

logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path="..", config_name="ppo_config", version_base="1.2")
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
    ckpt = Path(__file__).resolve().parent / "vocabulary" / "priors" / "reinvent_vocabulary.txt"
    vocabulary = Vocabulary(ckpt)
    env_kwargs = {
        "start_token": vocabulary.vocab["GO"],
        "end_token": vocabulary.vocab["EOS"],
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
    }

    # Models
    ####################################################################################################################

    create_model = get_model_factory(cfg.model)
    ckpt = torch.load(Path(__file__).resolve().parent / "models" / "priors" / "reinvent.ckpt")
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
        env = MultiStepDeNovoEnv(**env_kwargs)
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
    scoring = MolScore(model_name="ppo", task_config=cfg.molscore)
    scoring.configs["save_dir"] = save_dir
    scoring_function = scoring.score

    # Create reward transform
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

    # Buffers
    ####################################################################################################################

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.num_envs, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.mini_batch_size,
        prefetch=2,
    )

    penalty_transform = None
    if cfg.penalize_repetition is True:
        penalty_transform = PenaliseRepeatedSMILES(
            check_duplicate_key="SMILES",
            in_key="reward",
            out_key="reward",
            penalty=cfg.repetition_penalty,
            device=device,
        )

    experience_replay_buffer = None
    if cfg.experience_replay is True:
        experience_replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(cfg.experience_replay_buffer_size, device=device),
            prefetch=2,
            batch_size=cfg.experience_replay_batch_size,
            writer=TensorDictMaxValueWriter(rank_key="reward"),
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
    model_updates = 1
    collected_frames = 0
    kl_coef = cfg.kl_coef
    ppo_epochs = cfg.ppo_epochs
    max_grad_norm = cfg.max_grad_norm
    pbar = tqdm.tqdm(total=cfg.total_frames)
    replay_frequency = cfg.experience_replay_frequency
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
                    "train/reward": episode_rewards.mean().item(),
                    "train/min_reward": episode_rewards.min().item(),
                    "train/max_reward": episode_rewards.max().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        # Penalise repeated smiles and register penalised rewards
        if penalty_transform is not None:
            data = penalty_transform(data)
            repeated_smiles = penalty_transform.repeated_smiles
            episode_rewards = data["next", "reward"][data["next", "terminated"]]
            log_info.update(
                {
                    "train/repeated_smiles": repeated_smiles,
                    "train/penalised_reward": episode_rewards.mean().item(),
                    "train/penalised_min_reward": episode_rewards.min().item(),
                    "train/penalised_max_reward": episode_rewards.max().item(),
                }
            )

        # Add data to the replay buffer
        if experience_replay_buffer is not None:
            next_data = data.get("next")
            terminated = next_data.get("terminated").squeeze(-1)
            terminated_smiles = next_data.get_sub_tensordict(idx=terminated).select("SMILES", "reward")
            if len(terminated_smiles) > 0:
                experience_replay_buffer.extend(terminated_smiles.cpu())

        for j in range(ppo_epochs):

            with torch.no_grad():

                if cfg.augment_reward:
                    prior_log_probs = prior(data.clone())["sample_log_prob"].unsqueeze(-1)
                    batch_log_probs = actor_training(data.clone())["sample_log_prob"].unsqueeze(-1)
                    kl = (batch_log_probs - prior_log_probs)
                    reward = data.get(("next", "reward"))
                    data.set(("next", "reward"), reward + kl * 0.002)

                data = adv_module(data)

            # buffer.extend(data)
            buffer.extend(data.exclude("recurrent_state", ("next", "recurrent_state")))

            for i in range(num_mini_batches):

                # Compute loss for the current mini-batch
                batch = buffer.sample()

                loss = loss_module(batch)
                loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

                if cfg.augment_loss:

                    # prior_log_probs = prior(batch.clone())["sample_log_prob"].unsqueeze(-1)
                    # batch_log_probs = actor_training(batch.clone())["sample_log_prob"].unsqueeze(-1)
                    # kl = (batch_log_probs - prior_log_probs).mean()

                    dist_actor = actor_training.get_dist(batch.clone())
                    dist_prior = prior.get_dist(batch.clone())
                    kl = kl_divergence(dist_prior, dist_actor)
                    # import ipdb; ipdb.set_trace()

                    # kl = kl.abs()

                    # import ipdb; ipdb.set_trace()
                    # mask = kl < 0.5
                    # kl[mask] = 0
                    kl = kl.mean()

                    loss_sum += kl * kl_coef

                losses[j, i] = loss.select("loss_critic", "loss_entropy", "loss_objective").detach()

                if experience_replay_buffer is not None and model_updates % replay_frequency == 0:
                    replay_batch = experience_replay_buffer.sample()

                    # Create replay batch as a TensorDict
                    cat_replay_data = create_batch_from_replay_smiles(
                        replay_batch,  device, vocabulary=vocabulary)

                    # Compute adv for the replay batch
                    with torch.no_grad():

                        if cfg.augment_reward:
                            prior_log_probs = prior(cat_replay_data.clone())["sample_log_prob"].unsqueeze(-1)
                            batch_log_probs = actor_training(cat_replay_data.clone())["sample_log_prob"].unsqueeze(-1)
                            kl = (batch_log_probs - prior_log_probs)
                            reward = data.get(("next", "reward"))
                            data.set(("next", "reward"), reward + kl * 0.002)

                        replay_batch = adv_module(cat_replay_data)

                    # Compute loss for the replay batch
                    replay_loss = loss_module(replay_batch)
                    replay_loss_sum = replay_loss["loss_critic"] + replay_loss["loss_objective"] + replay_loss["loss_entropy"]

                    if cfg.augment_loss:

                        # prior_log_probs = prior(replay_batch.clone())["sample_log_prob"].unsqueeze(-1)
                        # batch_log_probs = actor_training(replay_batch.clone())["sample_log_prob"].unsqueeze(-1)
                        # kl = (batch_log_probs - prior_log_probs).mean()

                        dist_actor = actor_training.get_dist(batch.clone())
                        dist_prior = prior.get_dist(batch.clone())
                        kl = kl_divergence(dist_prior, dist_actor).mean()

                        replay_loss_sum += kl * kl_coef

                    # Log replay loss
                    replay_losses[j, i] = replay_loss.select("loss_critic", "loss_entropy", "loss_objective").detach()
                    replay_losses[j, i] = TensorDict({
                        "reward": cat_replay_data["next"]["reward"][cat_replay_data["next"]["done"]].mean().item(),
                    }, batch_size=[])

                    # Augment loss
                    bs = batch.numel()
                    rbs = replay_batch.numel()
                    alpha = (rbs / (bs + rbs))
                    loss_sum += (1 - alpha) * loss_sum + alpha * replay_loss_sum

                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=max_grad_norm)
                optim.step()
                optim.zero_grad()
                model_updates += 1

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

