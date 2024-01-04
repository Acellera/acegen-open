import datetime
import json
import os
import random
import shutil
from pathlib import Path

import hydra
import numpy as np

import torch
import tqdm
import yaml

from acegen import PenaliseRepeatedSMILES, SMILESEnv, SMILESReward, SMILESVocabulary
from acegen.models import (
    adapt_state_dict,
    create_gru_actor,
    create_gru_actor_critic,
    create_gru_critic,
)
from omegaconf import OmegaConf
from torch.distributions.kl import kl_divergence
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec

from torchrl.envs import (
    BurnInTransform,
    CatFrames,
    InitTracker,
    RandomCropTensorDict,
    StepCounter,
    TensorDictPrimer,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.modules.distributions import OneHotCategorical
from torchrl.objectives import DiscreteSACLoss, SoftUpdate
from torchrl.record.loggers import get_logger


try:
    import molscore
    from molscore.manager import MolScore

    _has_molscore = True
except ImportError as err:
    _has_molscore = False
    MOLSCORE_ERR = err


@hydra.main(config_path=".", config_name="config_pretrain", version_base="1.2")
def main(cfg: "DictConfig"):

    # Save config
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
    save_dir = f"{cfg.log_dir}_{timestamp_str}"
    os.makedirs(save_dir)
    with open(Path(save_dir) / "config.yaml", "w") as yaml_file:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

    # Set seeds
    seed = cfg.seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # Get available device
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )

    # Load Vocabulary
    ckpt = (
        Path(__file__).resolve().parent.parent.parent
        / "priors"
        / "reinvent_vocabulary.txt"
    )
    with open(ckpt, "r") as f:
        tokens = f.read().splitlines()
    vocabulary = SMILESVocabulary.create_from_list_of_chars(tokens)

    # Models
    ####################################################################################################################

    # Create GRU model
    if cfg.shared_nets:
        (
            actor_training,
            actor_inference,
            critic_training,
            critic_inference,
        ) = create_gru_actor_critic(
            vocabulary_size=len(vocabulary),
            distribution_class=OneHotCategorical,
            critic_value_per_action=True,
            python_based=True,
        )
    else:
        actor_training, actor_inference = create_gru_actor(
            len(vocabulary), distribution_class=OneHotCategorical
        )
        critic_training, critic_inference = create_gru_critic(
            len(vocabulary), critic_value_per_action=True
        )

    # Load pretrained weights
    ckpt = torch.load(
        Path(__file__).resolve().parent.parent.parent / "priors" / "reinvent.ckpt"
    )
    actor_inference.load_state_dict(
        adapt_state_dict(ckpt, actor_inference.state_dict())
    )
    actor_training.load_state_dict(adapt_state_dict(ckpt, actor_training.state_dict()))
    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)
    critic_training = critic_training.to(device)

    # Load weights
    actor_inference.load_state_dict(
        adapt_state_dict(ckpt, actor_inference.state_dict())
    )
    actor_training.load_state_dict(adapt_state_dict(ckpt, actor_training.state_dict()))

    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)
    critic_training = critic_training.to(device)

    if not cfg.trainable_actor:
        for param in actor_inference.parameters():
            param.requires_grad = False
        for param in actor_training.parameters():
            param.requires_grad = False

    prior, _ = create_gru_actor(len(vocabulary), distribution_class=OneHotCategorical)
    prior.load_state_dict(adapt_state_dict(ckpt, prior.state_dict()))
    prior = prior.to(device)

    # Environment
    ####################################################################################################################

    # Create transform to populate initial tensordict with recurrent states equal to 0.0
    num_layers = 3
    hidden_size = 512
    if cfg.shared_nets:
        primers = {
            ("recurrent_state",): UnboundedContinuousTensorSpec(
                shape=torch.Size([cfg.num_envs, num_layers, hidden_size]),
                dtype=torch.float32,
            ),
        }
        rhs_primers = [TensorDictPrimer(primers)]
    else:
        actor_primers = {
            ("recurrent_state_actor",): UnboundedContinuousTensorSpec(
                shape=torch.Size([cfg.num_envs, num_layers, hidden_size]),
                dtype=torch.float32,
            ),
        }
        critic_primers = {
            ("recurrent_state_critic",): UnboundedContinuousTensorSpec(
                shape=torch.Size([cfg.num_envs, num_layers, hidden_size]),
                dtype=torch.float32,
            ),
        }
        rhs_primers = [
            TensorDictPrimer(actor_primers),
            TensorDictPrimer(critic_primers),
        ]

    env_kwargs = {
        "start_token": vocabulary.vocab[vocabulary.start_token],
        "end_token": vocabulary.vocab[vocabulary.end_token],
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
        "max_length": 80,
        "one_hot_action_encoding": True,
    }

    def create_env_fn():
        """Create a single RL rl_env."""
        env = SMILESEnv(**env_kwargs)
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
        for rhs_primer in rhs_primers:
            env.append_transform(rhs_primer)
        return env

        # tests rl_env

    test_env = SMILESEnv(**env_kwargs)

    # Scoring transform - more efficient to do it outside the environment
    ####################################################################################################################

    if not _has_molscore:
        raise RuntimeError(
            "MolScore library not found, unable to create a scoring function. "
        ) from MOLSCORE_ERR

    if cfg.molscore is None:
        raise RuntimeError(
            "MolScore config file not provided, unable to create a scoring function. "
            "Please provide a config file,"
            "e.g. ../MolScore/molscore/configs/GuacaMol/Albuterol_similarity.json "
        )

    # Save molscore output. Also redirect output to save_dir
    cfg.molscore = shutil.copy(cfg.molscore, save_dir)
    data = json.load(open(cfg.molscore, "r"))
    data["output_dir"] = save_dir
    json.dump(data, open(cfg.molscore, "w"), indent=4)

    # Create scoring function
    scoring = MolScore(model_name="sac", task_config=cfg.molscore)
    scoring.configs["save_dir"] = save_dir
    scoring_function = scoring.score

    # Create reward transform
    rew_transform = SMILESReward(
        reward_function=scoring_function, vocabulary=vocabulary
    )

    # Collector
    ####################################################################################################################

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=actor_inference,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        storing_device="cpu",
        reset_at_each_iter=True,  # To avoid burn in issues
    )

    # Loss
    ####################################################################################################################

    loss_module = DiscreteSACLoss(
        actor_network=actor_training,
        qvalue_network=critic_training,
        num_actions=len(vocabulary),
        num_qvalue_nets=2,
        target_entropy_weight=cfg.target_entropy_weight,
        target_entropy="auto",
        loss_function=cfg.value_loss_function,
        action_space=test_env.action_spec,
    )
    loss_module.make_value_estimator(gamma=cfg.gamma)
    target_net_updater = SoftUpdate(loss_module, eps=cfg.target_update_polyak)

    # Buffer
    ####################################################################################################################

    crop_seq = RandomCropTensorDict(
        sub_seq_len=cfg.sampled_sequence_length, sample_dim=-1
    )
    burn_in = BurnInTransform(
        modules=(actor_training, critic_training), burn_in=cfg.burn_in
    )
    # buffer = TensorDictReplayBuffer(
    #     storage=LazyMemmapStorage(cfg.replay_buffer_size),
    #     batch_size=cfg.batch_size,
    #     prefetch=3,
    # )
    buffer = TensorDictPrioritizedReplayBuffer(
        storage=LazyMemmapStorage(cfg.replay_buffer_size),
        alpha=0.7,
        beta=0.5,
        pin_memory=False,
        prefetch=3,
        batch_size=cfg.batch_size,
        priority_key="loss_qvalue",
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
            cfg.logger_backend,
            logger_name="sac",
            experiment_name=cfg.agent_name,
            project_name=cfg.experiment_name,
        )

    # Training loop
    ####################################################################################################################

    total_done = 0
    collected_frames = 0
    num_updates = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    kl_coef = cfg.kl_coef
    max_grad_norm = cfg.max_grad_norm

    for data in tqdm.tqdm(collector):

        # Compute all rewards in a single call
        data = rew_transform(data)

        log_info = {}
        frames_in_batch = data.numel()
        total_done += data.get(("next", "done")).sum()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

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
                    "train/episode_length": episode_length.sum().item() / len(
                        episode_length
                    ),
                }
            )
            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, collected_frames)

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

        buffer.extend(data.cpu())

        if total_done < cfg.init_random_smiles:
            continue

        for i in range(cfg.num_loss_updates):

            log_info = {}
            batch = buffer.sample()
            if batch.device != device:
                batch = batch.to(device, non_blocking=True)
            else:
                batch = batch.clone()

            loss = loss_module(batch)
            loss_sum = loss["loss_qvalue"]
            log_info.update({f"train/loss_qvalue": loss["loss_qvalue"].detach().item()})

            if num_updates % cfg.actor_updates_frequency == 0:
                loss_sum += loss["loss_actor"] + loss["loss_alpha"]
                with torch.no_grad():
                    prior_dist = prior.get_dist(batch)
                kl_div = kl_divergence(actor_training.get_dist(batch), prior_dist)
                mask = torch.isnan(kl_div) | torch.isinf(kl_div)
                kl_div = kl_div[~mask].mean()
                loss_sum += kl_div * kl_coef
                log_info.update(
                    {
                        "train/kl_div": kl_div.detach().item(),
                        "train/loss_actor": loss["loss_actor"].detach().item(),
                        "train/loss_alpha": loss["loss_alpha"].detach().item(),
                        "train/alpha": loss["alpha"].detach().item(),
                        "train/entropy": loss["entropy"].detach().item(),
                    }
                )

            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_norm=max_grad_norm
            )
            optim.step()
            optim.zero_grad()

            target_net_updater.step()
            num_updates += 1
            buffer.update_priority(
                index=batch.get("index"), priority=batch.get("td_error")
            )

            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()

    collector.shutdown()
    print("Success!")


if __name__ == "__main__":
    main()