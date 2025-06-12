#! /usr/bin/python3
import datetime
import os
import random
from packaging import version
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np

import torch
import tqdm
import yaml

from acegen.models import adapt_state_dict, models, register_model
from acegen.rl_env import generate_complete_smiles, TokenEnv
from acegen.scoring_functions import (
    custom_scoring_functions,
    register_custom_scoring_function,
    Task,
)
from acegen.vocabulary import Vocabulary
from omegaconf import OmegaConf, open_dict
from tensordict.utils import isin
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    TensorDictMaxValueWriter,
    TensorDictReplayBuffer,
)
from torchrl.envs import InitTracker, TransformedEnv
from torchrl.modules.utils import get_primers_from_module
from torchrl.record.loggers import get_logger

try:
    import molscore
    from molscore import MolScoreBenchmark, MolScoreCurriculum
    from molscore.manager import MolScore

    _has_molscore = True
    if hasattr(molscore, "__version__"):
        _molscore_version = version.parse(molscore.__version__)
    else:
        _molscore_version = version.parse("1.0")
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

    if isinstance(cfg.seed, int):
        seeds = [cfg.seed]
    else:
        seeds = cfg.seed

    for seed in seeds:

        # Set seed
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        cfg.seed = int(seed)

        # Define save_dir and save config
        current_time = datetime.datetime.now()
        timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
        os.chdir(os.path.dirname(__file__))
        save_dir = (
            f"{cfg.log_dir}/{cfg.experiment_name}_{cfg.agent_name}_{timestamp_str}"
        )
        with open_dict(cfg):
            cfg.save_dir = save_dir
            cfg.script = __file__
        os.makedirs(save_dir, exist_ok=True)
        with open(Path(save_dir) / "config.yaml", "w") as yaml_file:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

        # Define training task and run
        if cfg.get("molscore_task", None):

            if not _has_molscore:
                raise RuntimeError(
                    "MolScore library not found. Unable to create a scoring function. "
                    "To install MolScore, use: `pip install MolScore`"
                ) from MOLSCORE_ERR

            if cfg.molscore_mode == "single":
                task = MolScore(
                    model_name=cfg.agent_name,
                    task_config=cfg.molscore_task,
                    budget=cfg.total_smiles,
                    output_dir=os.path.abspath(save_dir),
                    add_run_dir=False,
                    **cfg.get("molscore_kwargs", {}),
                )
                if _molscore_version < version.parse("2.0"):
                    run_hill_climb(cfg, task)
                else:
                    with task as scoring_function:
                        run_hill_climb(cfg, scoring_function)

            if cfg.molscore_mode == "benchmark":
                MSB = MolScoreBenchmark(
                    model_name=cfg.agent_name,
                    model_parameters=dict(cfg),
                    benchmark=cfg.molscore_task,
                    budget=cfg.total_smiles,
                    output_dir=os.path.abspath(save_dir),
                    add_benchmark_dir=False,
                    **cfg.get("molscore_kwargs", {}),
                )
                if _molscore_version < version.parse("2.0"):
                    for task in MSB:
                        run_hill_climb(cfg, task)
                        task.write_scores()
                else:
                    with MSB as benchmark:
                        for task in benchmark:
                            with task as scoring_function:
                                run_hill_climb(cfg, scoring_function)

            if cfg.molscore_mode == "curriculum":
                task = MolScoreCurriculum(
                    model_name=cfg.agent_name,
                    model_parameters=dict(cfg),
                    benchmark=cfg.molscore_task,
                    budget=cfg.total_smiles,
                    output_dir=os.path.abspath(save_dir),
                    **cfg.get("molscore_kwargs", {}),
                )
                if _molscore_version < version.parse("2.0"):
                    run_hill_climb(cfg, task)
                else:
                    with task as scoring_function:
                        run_hill_climb(cfg, scoring_function)

        elif cfg.get("custom_task", None):
            if cfg.custom_task not in custom_scoring_functions:
                register_custom_scoring_function(cfg.custom_task, cfg.custom_task)
            task = Task(
                name=cfg.custom_task,
                scoring_function=custom_scoring_functions[cfg.custom_task],
                budget=cfg.total_smiles,
                output_dir=save_dir,
            )
            run_hill_climb(cfg, task)

        else:
            raise ValueError("No scoring function specified.")


def run_hill_climb(cfg, task):

    # Get available device
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )

    # If custom model, register it
    if cfg.model not in models and cfg.get("custom_model_factory", None) is not None:
        register_model(cfg.model, cfg.model_factory)

    # Check if model is available
    if cfg.model not in models:
        raise ValueError(
            f"Model {cfg.model} not found. For custom models, define a model factory as explain in the tutorials."
        )

    # Get model
    (create_actor, _, _, voc_path, ckpt_path, tokenizer) = models[cfg.model]

    # Create vocabulary
    ####################################################################################################################

    vocabulary = Vocabulary.load(voc_path, tokenizer=tokenizer)

    # Create models
    ####################################################################################################################

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    actor_training, actor_inference = create_actor(vocabulary_size=len(vocabulary))
    actor_inference.load_state_dict(
        adapt_state_dict(deepcopy(ckpt), actor_inference.state_dict())
    )
    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)

    # Create RL environment
    ####################################################################################################################

    env_kwargs = {
        "start_token": vocabulary.start_token_index,
        "end_token": vocabulary.end_token_index,
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
    }

    def create_env_fn():
        """Create a single RL rl_env."""
        env = TokenEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(InitTracker())
        if primers := get_primers_from_module(actor_inference):
            env.append_transform(primers)
        return env

    env = create_env_fn()

    # Create replay buffer
    ####################################################################################################################

    storage = LazyTensorStorage(cfg.replay_buffer_size, device=device)
    experience_replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=PrioritizedSampler(storage.max_size, alpha=1.0, beta=1.0),
        batch_size=cfg.replay_batch_size,
        writer=TensorDictMaxValueWriter(rank_key="priority"),
        priority_key="priority",
    )

    # Create optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        actor_training.parameters(),
        lr=cfg.lr,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    # Create logger
    ####################################################################################################################

    logger = None
    if cfg.logger_backend:
        experiment_name = f"{cfg.agent_name}"
        try:
            experiment_name += f"_{task.cfg.get('task')}"
        except AttributeError:
            experiment_name += "_custom_task"
        logger = get_logger(
            cfg.logger_backend,
            logger_name=cfg.save_dir,
            experiment_name=experiment_name,
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
    pbar = tqdm.tqdm(total=cfg.total_smiles)

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

        log_info = {}
        data_next = data.get("next")
        done = data_next.get("done").squeeze(-1)
        total_done += done.sum().item()
        pbar.update(done.sum().item())

        # Save info about smiles lengths and rewards
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

        # Sort data by reward and keep topk
        _, sscore_idxs = data_next["reward"][done].squeeze(-1).sort(descending=True)
        data = data[sscore_idxs.data[: int(cfg.num_envs * cfg.topk)]]

        for _ in range(cfg.num_epochs):

            # Sampler to create mini-batches
            sampler = BatchSampler(
                SubsetRandomSampler(range(len(data))),
                len(data) // cfg.num_mini_batches,
                drop_last=True,
            )

            for idxs in sampler:

                loss = compute_loss(data[idxs], actor_training)

                # Compute experience replay loss
                if (
                    cfg.experience_replay
                    and len(experience_replay_buffer) > cfg.replay_batch_size
                ):
                    replay_batch = experience_replay_buffer.sample()
                    replay_loss = compute_loss(replay_batch, actor_training)
                    loss = torch.cat((loss, replay_loss), 0)

                # Average loss over the batch
                loss = loss.mean()

                # Calculate gradients and make an update to the network weights
                optim.zero_grad()
                loss.backward()
                optim.step()

        # Then add new experiences to the replay buffer
        if cfg.experience_replay:

            replay_data = data.clone()

            # MaxValueWriter is not compatible with storages of more than one dimension.
            replay_data.batch_size = [replay_data.batch_size[0]]

            # Remove SMILES that are already in the replay buffer
            if len(experience_replay_buffer) > 0:
                is_duplicated = isin(
                    input=replay_data,
                    key="action",
                    reference=experience_replay_buffer[:],
                )
                replay_data = replay_data[~is_duplicated]

            # Add data to the replay buffer
            if len(replay_data) > 0:
                reward = replay_data.get(("next", "reward"))
                replay_data.set("priority", reward)
                experience_replay_buffer.extend(replay_data)

        # Log info
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=total_done)


def get_log_prob(data, model):
    actions = data.get("action")
    model_in = data.select(*model.in_keys, strict=False)
    log_prob = model.get_dist(model_in).log_prob(actions)
    return log_prob


def compute_loss(data, model):
    mask = data.get("mask").squeeze(-1)
    agent_log_prob = get_log_prob(data, model)
    agent_likelihood = (agent_log_prob * mask).sum(-1)
    loss = -agent_likelihood
    return loss


if __name__ == "__main__":
    main()
