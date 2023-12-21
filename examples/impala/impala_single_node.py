import datetime
import json
import logging
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path

import acegen
import hydra
import numpy as np

import torch
import tqdm
import yaml
from acegen import (
    MultiStepDeNovoEnv as DeNovoEnv,
    PenaliseRepeatedSMILES,
    SMILESReward,
    SMILESVocabulary,
)
from molscore.manager import MolScore
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributions.kl import kl_divergence
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.envs import (
    CatFrames,
    InitTracker,
    StepCounter,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.objectives import A2CLoss
from torchrl.objectives.value.advantages import VTrace
from torchrl.record.loggers import get_logger
from utils import (
    create_batch_from_replay_smiles,
    create_shared_impala_models,
    Experience,
)

logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):

    # Set seeds
    seed = cfg.seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # Save config
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
    save_dir = f"{cfg.log_dir}_{timestamp_str}"
    os.makedirs(save_dir)
    with open(Path(save_dir) / "config.yaml", "w") as yaml_file:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

    # Get available device
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )

    # Load vocabulary
    ckpt = (
        Path(__file__).resolve().parent.parent.parent
        / "priors"
        / "reinvent_vocabulary.txt"
    )
    vocabulary = SMILESVocabulary(ckpt)

    # Models
    ####################################################################################################################

    ckpt = torch.load(
        Path(__file__).resolve().parent.parent.parent / "priors" / "reinvent.ckpt"
    )
    (
        actor_inference,
        actor_training,
        critic_inference,
        critic_training,
        *transforms,
    ) = create_shared_impala_models(
        vocabulary_size=len(vocabulary), ckpt=ckpt, batch_size=1
    )

    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)
    critic_training = critic_training.to(device)
    prior = deepcopy(actor_training)

    # Environment
    ####################################################################################################################

    env_kwargs = {
        "start_token": vocabulary.vocab["GO"],
        "end_token": vocabulary.vocab["EOS"],
        "length_vocabulary": len(vocabulary),
        "batch_size": 1,
        "device": device,
    }

    def create_env_fn():
        """Create a single RL rl_environments."""
        env = DeNovoEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(
            UnsqueezeTransform(
                in_keys=["observation"], out_keys=["observation"], unsqueeze_dim=-1
            )
        )
        env.append_transform(
            CatFrames(
                N=100,
                dim=-1,
                padding="constant",
                in_keys=["observation"],
                out_keys=["SMILES"],
                padding_value=-1,
            )
        )
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        for transform in transforms:
            env.append_transform(transform)
        return env

    # Save molscore output. Also redirect output to save_dir
    cfg.molscore = shutil.copy(cfg.molscore, save_dir)
    data = json.load(open(cfg.molscore, "r"))
    data["output_dir"] = save_dir
    json.dump(data, open(cfg.molscore, "w"), indent=4)

    # Create scoring function
    scoring = MolScore(model_name="impala", task_config=cfg.molscore)
    scoring.configs["save_dir"] = save_dir
    scoring_function = scoring.score

    # Create reward transform
    rew_transform = SMILESReward(
        reward_function=scoring_function, vocabulary=vocabulary
    )

    # Collector
    ####################################################################################################################

    collector = MultiaSyncDataCollector(
        create_env_fn=[create_env_fn()] * cfg.num_workers,
        policy=actor_inference,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        update_at_each_batch=True,
    )

    # Loss modules
    ####################################################################################################################

    adv_module = VTrace(
        gamma=cfg.gamma,
        value_network=critic_training,
        actor_network=actor_training,
        average_adv=False,
        shifted=True,
    )
    adv_module = adv_module.to(device)
    loss_module = A2CLoss(
        actor=actor_training,
        critic=critic_training,
        loss_critic_type=cfg.loss_critic_type,
        entropy_coef=cfg.entropy_coef,
        critic_coef=cfg.critic_coef,
        # normalize_advantage=False,
    )
    loss_module = loss_module.to(device)

    # Buffers
    ####################################################################################################################

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.batch_size, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.batch_size,
        prefetch=4,
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
        experience_replay_buffer = Experience(voc=vocabulary)

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
            cfg.logger_backend,
            logger_name="impala",
            experiment_name=cfg.agent_name,
            project_name=cfg.experiment_name,
        )

    # Training loop
    ####################################################################################################################

    total_done = 0
    num_updates = 0
    collected_frames = 0
    kl_coef = cfg.kl_coef
    sgd_updates = cfg.sgd_updates
    batch_size = cfg.batch_size
    max_grad_norm = cfg.max_grad_norm
    pbar = tqdm.tqdm(total=cfg.total_frames)
    losses = TensorDict({}, batch_size=[sgd_updates])
    accumulator = []

    for data in collector:

        log_info = {}
        frames_in_batch = data.numel()
        total_done += data.get(("next", "done")).sum()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        if len(accumulator) < batch_size:
            accumulator.append(data)
            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, collected_frames)
            continue

        # Create a single batch of trajectories
        stacked_data = torch.cat(accumulator, dim=0).contiguous()
        data = stacked_data.to(device, non_blocking=True)

        # Compute all rewards in a single call
        data = rew_transform(data)

        # Register smiles lengths and real rewards
        episode_rewards = data["next", "reward"][data["next", "done"]]
        episode_length = data["next", "step_count"][data["next", "done"]]
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

        # Get data to be added to the replay buffer later
        replay_data = data.get("next").clone()
        replay_data = replay_data.get_sub_tensordict(
            idx=replay_data.get("terminated").squeeze(-1)
        )

        # Then exclude unnecessary tensors
        data = data.exclude(
            "embed",
            "logits",
            "features",
            "collector",
            "step_count",
            ("next", "step_count"),
            "SMILES",
            ("next", "SMILES"),
        )

        for j in range(sgd_updates):

            buffer.extend(data)

            for batch in buffer:

                batch = batch.to(device, non_blocking=True)

                if num_updates % cfg.replay_frequency == 0:
                    if (
                        experience_replay_buffer is not None
                        and len(experience_replay_buffer) > 10
                    ):
                        batch = batch.exclude(
                            "advantage",
                            "state_value",
                            "value_target",
                            ("next", "state_value"),
                        )
                        for _ in range(cfg.replay_batches):
                            row = random.randint(0, cfg.batch_size - 1)
                            (
                                exp_seqs,
                                exp_reward,
                                exp_prior_likelihood,
                            ) = experience_replay_buffer.sample(6, decode_smiles=False)
                            replay_batch = create_batch_from_replay_smiles(
                                exp_seqs, exp_reward, device, vocabulary=vocabulary
                            )
                            batch[row] = replay_batch[0, 0 : cfg.frames_per_batch]

                # Compute advantage
                with torch.no_grad():
                    batch = adv_module(batch)

                loss = loss_module(batch)
                loss_sum = (
                    loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                )
                losses[j] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                with torch.no_grad():
                    prior_dist = prior.get_dist(batch)
                kl_div = kl_divergence(actor_training.get_dist(batch), prior_dist)
                mask = torch.isnan(kl_div) | torch.isinf(kl_div)
                kl_div = kl_div[~mask].mean()
                loss_sum += kl_div * kl_coef
                losses[j] = TensorDict(
                    {"kl_div": kl_div.detach().item()}, batch_size=[]
                )

                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=max_grad_norm
                )
                optim.step()
                optim.zero_grad()
                num_updates += 1

        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})

        # Add data to the replay buffer
        if experience_replay_buffer is not None:
            smiles_list = []
            for index, seq in enumerate(replay_data.get("SMILES")):
                smiles = vocabulary.decode(seq.cpu().numpy(), ignore_indices=[-1])
                smiles_list.append(smiles)
            rewards = replay_data.get("reward").squeeze(-1).cpu().numpy()
            prior_likelihood = np.zeros_like(rewards)
            new_experience = zip(smiles_list, rewards, rewards, prior_likelihood)
            experience_replay_buffer.add_experience(new_experience)

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)
        collector.update_policy_weights_()
        accumulator = []

    collector.shutdown()


if __name__ == "__main__":
    main()
