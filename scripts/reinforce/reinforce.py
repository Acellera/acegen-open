#! /usr/bin/python3
import datetime
import json
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np

import torch
import tqdm
import yaml

from acegen.models import adapt_state_dict, models, register_model
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.scoring_functions import custom_scoring_functions, Task
from acegen.vocabulary import SMILESVocabulary
from omegaconf import OmegaConf, open_dict
from tensordict.utils import isin

from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    TensorDictMaxValueWriter,
    TensorDictReplayBuffer,
)
from torchrl.envs import InitTracker, TensorDictPrimer, TransformedEnv
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

    if isinstance(cfg.seed, int):
        cfg.seed = [cfg.seed]

    for seed in cfg.seed:

        # Set seed
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

        # Define save_dir and save config
        current_time = datetime.datetime.now()
        timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
        os.chdir(os.path.dirname(__file__))
        save_dir = f"{cfg.log_dir}/logs_{cfg.agent_name}_{timestamp_str}"
        with open_dict(cfg):
            cfg.save_dir = save_dir
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
                    run_reinforce(cfg, task)
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
                run_reinforce(cfg, task)
        elif cfg.get("custom_task", None):
            task = Task(
                name=cfg.custom_task,
                scoring_function=custom_scoring_functions[cfg.custom_task],
                budget=cfg.total_smiles,
                output_dir=save_dir,
            )
            run_reinforce(cfg, task)
        else:
            raise ValueError("No scoring function specified.")


def run_reinforce(cfg, task):

    # Get available device
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )

    # Get model and vocabulary checkpoints
    if cfg.model not in models and cfg.model_factory is not None:
        register_model(cfg.model, cfg.model_factory)
    else:
        raise ValueError(f"Model {cfg.model} not found. For custom models, create and register a model factory.")

    create_actor, _, _, voc_path, ckpt_path, tokenizer = models[cfg.model](cfg)

    # Create vocabulary
    ####################################################################################################################

    vocabulary = SMILESVocabulary.load(voc_path, tokenizer=tokenizer)

    # Create models
    ####################################################################################################################

    ckpt_path = cfg.get("model_weights", ckpt_path)
    ckpt = torch.load(ckpt_path)

    actor_training, actor_inference = create_actor(vocabulary_size=len(vocabulary))
    actor_inference.load_state_dict(
        adapt_state_dict(ckpt, actor_inference.state_dict())
    )
    actor_training.load_state_dict(adapt_state_dict(ckpt, actor_training.state_dict()))
    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)

    # Create RL environment
    ####################################################################################################################

    # For RNNs, create a transform to populate initial tensordict with recurrent states equal to 0.0
    rhs_primers = []
    if hasattr(actor_training, "rnn_spec"):
        primers = actor_training.rnn_spec.expand(cfg.num_envs)
        rhs_primers.append(TensorDictPrimer(primers))

    env_kwargs = {
        "start_token": vocabulary.start_token_index,
        "end_token": vocabulary.end_token_index,
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
    }

    def create_env_fn():
        """Create a single RL rl_env."""
        env = SMILESEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(InitTracker())
        for rhs_primer in rhs_primers:
            env.append_transform(rhs_primer)
        return env

    env = create_env_fn()

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
            experiment_name += f"_{task.configs.get('task')}"
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

        data, loss = compute_loss(data, actor_training)

        # Average loss over the batch
        loss = loss.mean()

        # Calculate gradients and make an update to the network weights
        optim.zero_grad()
        loss.backward()
        optim.step()

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
    reward = data.get(("next", "reward")).squeeze(-1).sum(-1)
    loss = -agent_likelihood * reward

    return data, loss


if __name__ == "__main__":
    main()
