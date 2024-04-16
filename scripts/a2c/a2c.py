#! /usr/bin/python3
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

from acegen.models import adapt_state_dict, models
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.scoring_functions import custom_scoring_functions, Task
from acegen.vocabulary import SMILESVocabulary
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributions.kl import kl_divergence
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.envs import InitTracker, TensorDictPrimer, TransformedEnv
from torchrl.objectives import A2CLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import get_logger

try:
    import molscore
    from molscore import MolScoreBenchmark
    from molscore.manager import MolScore

    _has_molscore = True
except ImportError as err:
    _has_molscore = False
    MOLSCORE_ERR = err

# hydra outputs saved in /tmp
os.chdir("/tmp")


@hydra.main(
    config_path=".",
    config_name="config_denovo",
    version_base="1.2",
)
def main(cfg: "DictConfig"):

    # Set seeds
    seed = cfg.seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # Save config
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
    os.chdir(os.path.dirname(__file__))
    save_dir = f"{cfg.log_dir}/logs_{cfg.agent_name}_{timestamp_str}"
    os.makedirs(save_dir, exist_ok=True)
    with open(Path(save_dir) / "config.yaml", "w") as yaml_file:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

    # Define training task and run
    if cfg.get("molscore", None):

        if not _has_molscore:
            raise RuntimeError(
                "MolScore library not found. Unable to create a scoring function. "
                "To install MolScore, use: `pip install MolScore`"
            ) from MOLSCORE_ERR

        if cfg.molscore in MolScoreBenchmark.presets:
            MSB = MolScoreBenchmark(
                model_name=cfg.agent_name,
                model_parameters=dict(cfg),
                benchmark=cfg.molscore,
                budget=cfg.total_smiles,
                output_dir=os.path.abspath(save_dir),
                add_benchmark_dir=False,
                include=cfg.molscore_include,
            )
            for task in MSB:
                run_a2c(cfg, task)
        else:
            # Save molscore output. Also redirect output to save_dir
            cfg.molscore = shutil.copy(cfg.molscore, save_dir)
            data = json.load(open(cfg.molscore, "r"))
            json.dump(data, open(cfg.molscore, "w"), indent=4)
            task = MolScore(
                model_name=cfg.agent_name,
                task_config=cfg.molscore,
                budget=cfg.total_smiles,
                output_dir=os.path.abspath(save_dir),
                add_run_dir=False,
            )
            run_a2c(cfg, task)
    elif cfg.get("custom_task", None):
        task = Task(custom_scoring_functions[cfg.custom_task], budget=cfg.total_smiles)
        run_a2c(cfg, task)
    else:
        raise ValueError("No scoring function specified.")


def run_a2c(cfg, task):

    # Get available device
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )

    # Get model and vocabulary checkpoints
    if cfg.model in models:
        (
            create_actor,
            create_critic,
            create_shared,
            voc_path,
            ckpt_path,
            tokenizer,
        ) = models[cfg.model]
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")

    # Create vocabulary
    ####################################################################################################################

    vocabulary = SMILESVocabulary.load(voc_path, tokenizer=tokenizer)

    # Create models
    ####################################################################################################################

    # Create actor and critic networks
    if cfg.shared_nets:
        (
            actor_training,
            actor_inference,
            critic_training,
            critic_inference,
        ) = create_shared(vocabulary_size=len(vocabulary))
    else:
        actor_training, actor_inference = create_actor(len(vocabulary))
        critic_training, critic_inference = create_critic(len(vocabulary))

    # Load pretrained weights
    ckpt = torch.load(ckpt_path)
    actor_inference.load_state_dict(
        adapt_state_dict(ckpt, actor_inference.state_dict())
    )
    actor_training.load_state_dict(adapt_state_dict(ckpt, actor_training.state_dict()))
    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)
    critic_training = critic_training.to(device)

    # Define prior
    prior, _ = create_actor(len(vocabulary))
    prior = prior.to(device)
    prior.load_state_dict(adapt_state_dict(ckpt, prior.state_dict()))

    # Create RL environment
    ####################################################################################################################

    rhs_primers = []
    # if rnn's, create a transform to populate initial tensordict with recurrent states equal to 0.0
    if cfg.shared_nets and hasattr(actor_training, "rnn_spec"):
        primers = actor_training.rnn_spec.expand(cfg.num_envs)
        rhs_primers = [TensorDictPrimer(primers)]
    elif hasattr(actor_training, "rnn_spec"):
        actor_primers = actor_training.rnn_spec.expand(cfg.num_envs)
        critic_primers = critic_training.rnn_spec.expand(cfg.num_envs)
        rhs_primers = [
            TensorDictPrimer(actor_primers),
            TensorDictPrimer(critic_primers),
        ]

    # Define environment kwargs
    env_kwargs = {
        "start_token": vocabulary.start_token_index,
        "end_token": vocabulary.end_token_index,
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
    }

    # Define environment creation function
    def create_env_fn():
        """Create a single RL rl_env."""
        env = SMILESEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(InitTracker())
        for rhs_primer in rhs_primers:
            env.append_transform(rhs_primer)
        return env

    env = create_env_fn()

    # Create loss module
    ####################################################################################################################

    adv_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic_training,
        average_gae=True,
        shifted=True,
    )
    loss_module = A2CLoss(
        actor_network=actor_training,
        critic_network=critic_training,
        critic_coef=cfg.critic_coef,
        entropy_coef=cfg.entropy_coef,
        loss_critic_type="l2",
    )

    # Create data storage
    ####################################################################################################################

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.num_envs, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.mini_batch_size,
        prefetch=4,
    )

    # Create optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.lr,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    # Create logger
    ####################################################################################################################

    logger = None
    if cfg.logger_backend:
        logger = get_logger(
            cfg.logger_backend,
            logger_name="a2c",
            experiment_name=f"{cfg.agent_name}_{task.configs.get('task')}",
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.experiment_name,
                "group": cfg.agent_name,
                "reinit": True,
            },
        )

    # Training loop
    ####################################################################################################################

    total_done = 0
    num_updates = 0
    kl_coef = cfg.kl_coef
    max_grad_norm = cfg.max_grad_norm
    pbar = tqdm.tqdm(total=cfg.total_smiles)
    num_mini_batches = cfg.num_envs // cfg.mini_batch_size
    losses = TensorDict({}, batch_size=[num_mini_batches])

    while not task.finished:

        # Generate data
        data = generate_complete_smiles(
            policy_sample=actor_inference,
            policy_evaluate=actor_training,
            vocabulary=vocabulary,
            scoring_function=task,
            environment=env,
            prompt=cfg.get("prompt", None),
            promptsmiles=cfg.get("promptsmiles", None),
            promptsmiles_optimize=cfg.get("promptsmiles_optimize", True),
            promptsmiles_shuffle=cfg.get("promptsmiles_shuffle", True),
            promptsmiles_multi=cfg.get("promptsmiles_multi", False),
            promptsmiles_scan=cfg.get("promptsmiles_scan", False),
            remove_duplicates=True,
        )

        # Update progress bar
        data_next = data.get("next")
        done = data_next.get("done").squeeze(-1)
        pbar.update(done.sum().item())

        # Register smiles lengths and real rewards and total generated smiles
        log_info = {}
        total_done += done.sum().item()
        episode_rewards = data_next["reward"][done]
        episode_length = (data_next["observation"] != 0.0).float().sum(-1).mean()
        if len(episode_rewards) > 0:
            log_info.update(
                {
                    "train/total_smiles": total_done,
                    "train/reward": episode_rewards.mean().item(),
                    "train/min_reward": episode_rewards.min().item(),
                    "train/max_reward": episode_rewards.max().item(),
                    "train/episode_length": episode_length.item(),
                }
            )

        # Compute advantage
        with torch.no_grad():
            data = adv_module(data)

        # Add extended_data to buffer
        buffer.extend(data)

        for j, batch in enumerate(buffer):

            batch = batch.to(device, non_blocking=True)

            # Compute loss
            mask = batch.get("mask").squeeze(-1)
            loss = loss_module(batch)
            loss = loss.named_apply(
                lambda name, value: (
                    # (value * mask).mean() if name.startswith("loss_") else value
                    (value * mask).sum(-1).mean(-1)
                    if name.startswith("loss_")
                    else value
                ),
                batch_size=[],
            )
            loss_sum = (
                loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
            )
            losses[j] = loss.select(
                "loss_critic", "loss_entropy", "loss_objective"
            ).detach()

            # Add KL loss term
            with torch.no_grad():
                prior_dist = prior.get_dist(batch)
                kl_div = kl_divergence(actor_training.get_dist(batch), prior_dist)
            kl_div = (kl_div * mask.squeeze()).sum(-1).mean(-1)
            loss_sum += kl_div * kl_coef
            losses[j] = TensorDict({"kl_div": kl_div.detach().item()}, batch_size=[])

            # Update policy
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

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=total_done)


if __name__ == "__main__":
    main()
