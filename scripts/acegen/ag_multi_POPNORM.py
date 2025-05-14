#! /usr/bin/python3
import datetime
import json
import os
import random
import shutil
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
from acegen.rl_env.baselines import LeaveOneOutBaseline, MovingAverageBaseline
from acegen.scoring_functions import (
    custom_scoring_functions,
    register_custom_scoring_function,
    Task,
)
from acegen.vocabulary import Vocabulary
from omegaconf import OmegaConf, open_dict
from tensordict.utils import isin
from torch.distributions.kl import kl_divergence

from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    RandomSampler,
    TensorDictMaxValueWriter,
    TensorDictPrioritizedReplayBuffer,
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
    config_name="config_multi_denovo",
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
                    run_reinforce(cfg, task)
                else:
                    with task as scoring_function:
                        run_reinforce(cfg, scoring_function)

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
                        run_reinforce(cfg, task)
                        task.write_scores()
                else:
                    with MSB as benchmark:
                        for task in benchmark:
                            with task as scoring_function:
                                run_reinforce(cfg, scoring_function)

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
                    run_reinforce(cfg, task)
                else:
                    with task as scoring_function:
                        run_reinforce(cfg, scoring_function)

        elif cfg.get("custom_task", None):
            if cfg.custom_task not in custom_scoring_functions:
                register_custom_scoring_function(cfg.custom_task, cfg.custom_task)
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

    # Define number of agents in the population
    num_agents = cfg.get("num_agents", 1)

    # Get available device
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )
    buffer_device = deepcopy(device) # Or torch.device("cpu") or deepcopy(device)
    optim_device = deepcopy(device) # Or torch.device("cpu") or deepcopy(device)

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

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    population_inference, population_training = [], []
    for _ in range(num_agents):
        actor_training, actor_inference = create_actor(vocabulary_size=len(vocabulary))
        actor_inference.load_state_dict(
            adapt_state_dict(deepcopy(ckpt), actor_inference.state_dict())
        )
        population_inference.append(actor_inference)
        population_training.append(actor_training)

    prior, _ = create_actor(vocabulary_size=len(vocabulary))
    prior.load_state_dict(
        adapt_state_dict(deepcopy(ckpt), prior.state_dict())
    )
    prior = prior.to(device)

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

    population_experience_replay_buffer = []
    for i in range(num_agents):
        storage = LazyTensorStorage(cfg.replay_buffer_size, device=buffer_device)
        if cfg.get("replay_sampler", "prioritized") in ["random", "uniform"]:
            replay_sampler = RandomSampler()
        elif cfg.get("replay_sampler", "prioritized") == "prioritized":
            replay_sampler = PrioritizedSampler(storage.max_size, alpha=1.0, beta=1.0)
        else:
            raise ValueError(f"Unknown replay sampler: {replay_sampler}")
        experience_replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=replay_sampler,
            batch_size=cfg.replay_batch_size,
            writer=TensorDictMaxValueWriter(rank_key="priority"),
            priority_key="priority",
        )
        population_experience_replay_buffer.append(experience_replay_buffer)

    # Create baseline
    ####################################################################################################################

    # Select baseline
    population_baseline = []
    for _ in range(num_agents):
        baseline = None
        if cfg.get("baseline", False):
            if cfg.baseline == "loo":
                baseline = LeaveOneOutBaseline()
            elif cfg.baseline == "mab":
                baseline = MovingAverageBaseline(device=device)
            else:
                raise ValueError(f"Unknown baseline: {cfg.baseline}")
        population_baseline.append(baseline)

    # Create optimizer
    ####################################################################################################################

    population_optim, population_scheduler = [], []
    for actor in population_training:
        optim = torch.optim.Adam(
            actor.parameters(),
            lr=cfg.lr,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
        population_optim.append(optim)
        if cfg.get("lr_annealing", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=5000 // cfg.num_envs, eta_min=0.0001
            )
            population_scheduler.append(scheduler)
        else:
            population_scheduler.append(None)
            
    def move_optimizer_state(optimizer, device="cpu"):
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

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

        log_info = {}

        # Generate data
        for i, actor_inference, actor_training, experience_replay_buffer, optim, scheduler, baseline in zip(
            range(num_agents),
            population_inference,
            population_training,
            population_experience_replay_buffer,
            population_optim,
            population_scheduler,
            population_baseline,
        ):

            actor_inference = actor_inference.to(device)
            actor_training = actor_training.to(device)
            move_optimizer_state(optim, device)

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

            data = data.to(device)

            # Update pbar
            data_next = data.get("next")
            done = data_next.get("done").squeeze(-1)
            total_done += cfg.num_envs  # done.sum().item()
            pbar.update(done.sum().item())

            # Save info about smiles lengths and rewards
            episode_rewards = data_next["reward"][done]
            episode_length = (data_next["observation"] != 0.0).float().sum(-1).mean()
            if len(episode_rewards) > 0:
                log_info.update(
                    {
                        f"agent{i}/total_smiles": total_done,
                        f"agent{i}/reward": episode_rewards.mean().item(),
                        f"agent{i}/min_reward": episode_rewards.min().item(),
                        f"agent{i}/max_reward": episode_rewards.max().item(),
                        f"agent{i}/episode_length": episode_length.item(),
                    }
                )

            # Apply Hill-Climb
            if cfg.get("topk", False) and cfg.get("topk", False) < 1:
                sscore, sscore_idxs = (
                    data_next["reward"][done].squeeze(-1).sort(descending=True)
                )
                data = data[sscore_idxs.data[: int(cfg.num_envs * cfg.topk)]]

            # Apply experience replay
            if (
                cfg.replay_buffer_size
                and len(experience_replay_buffer) > cfg.replay_batch_size
            ):
                replay_batch = experience_replay_buffer.sample().exclude(
                    "priority", "index", "_weight", "SMILES"
                )
                data = torch.cat((data.exclude("SMILES"), replay_batch.to(device)), 0)

            # Compute loss
            data, loss, agent_likelihood, reward = compute_loss(
                data,
                actor_training,
                prior,
                alpha=cfg.get("alpha", 1),
                sigma=cfg.get("sigma", 0.0),
                baseline=baseline,
                entropy_coef=cfg.get("entropy_coef", 0.0)
            )

            # Add regularizer that penalizes high likelihood for the entire sequence
            if cfg.get("likely_penalty_coef", False):
                loss_p = -(1 / agent_likelihood)
                loss += cfg.likely_penalty_coef * loss_p

            # Add population normalization
            if cfg.get("pop_norm", False):
                pop_likelihoods = []
                for actor in population_training[:i]:
                    actor = actor.to(device)
                    with torch.no_grad():
                        pop_likelihood, _ = get_log_prob(data, actor)
                    pop_likelihoods.append(pop_likelihood)
                    actor = actor.cpu()

                if pop_likelihoods:
                    pop_log_prob = torch.stack(
                        pop_likelihoods, dim=1
                    )
                    if cfg.pop_norm == 'max':
                        pop_log_prob = pop_log_prob.max(dim=1)
                    elif cfg.pop_norm == 'mean':
                        pop_log_prob = pop_log_prob.mean(dim=1)
                    else:
                        raise ValueError("Unknown normalization, please use max or mean")
                    
                    norm_loss = - pop_log_prob * reward
                    log_info[f"agent{i}/norm_loss"] = norm_loss.mean().item()
                    loss -= norm_loss

            # Update
            optim.zero_grad()
            loss = loss.mean()
            loss.backward()
            optim.step()
            
            log_info.update({f"agent{i}/loss": loss.item()})
            
            # Update scheduler
            if cfg.get("lr_annealing", None) & (total_done / cfg.num_agents < 5000):
                log_info.update(
                    {
                        f"agent{i}/lr": scheduler.get_last_lr()[0],
                    }
                )
                scheduler.step()

            # Add new experiences to the replay buffer
            if cfg.get("replay_buffer_size", 100):

                replay_data = data.clone().to(buffer_device)

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

            # Move back to cpu
            actor_inference = actor_inference.cpu()
            actor_training = actor_training.cpu()
            move_optimizer_state(optim, optim_device)
            data = data.cpu()
            del data

        # Log info
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=total_done / cfg.num_agents)


def get_log_prob(data, model):
    mask = data.get("mask").squeeze(-1)
    actions = data.get("action")
    model_in = data.select(*model.in_keys, strict=False)
    dist = model.get_dist(model_in)
    log_prob = dist.log_prob(actions)
    log_prob = (log_prob * mask).sum(-1)
    return log_prob, dist


def compute_loss(
    data, model, prior, alpha=1, sigma=0.0, baseline=None, entropy_coef=0.0
):

    mask = data.get("mask").squeeze(-1)

    with torch.no_grad():
        prior_likelihood, prior_dist = get_log_prob(data, prior)

    agent_likelihood, agent_dist = get_log_prob(data, model)
    reward = data.get(("next", "reward")).squeeze(-1).sum(-1)

    # Reward reshaping
    reward = torch.pow(torch.clamp(reward + sigma * prior_likelihood, min=0.0), alpha)

    # Subtract baselines
    if baseline:
        baseline.update(reward.detach())
        reward = reward - baseline.mean

    # REINFORCE (negative as we minimize the negative log prob to maximize the policy)
    loss = -agent_likelihood * reward

    # Add Entropy loss term
    loss -= entropy_coef * (agent_dist.entropy() * mask.squeeze()).mean(-1)

    return data, loss, agent_likelihood, reward


if __name__ == "__main__":
    main()
