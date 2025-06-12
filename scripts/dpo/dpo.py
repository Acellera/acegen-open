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
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
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
                    run_dpo(cfg, task)
                else:
                    with task as scoring_function:
                        run_dpo(cfg, scoring_function)

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
                        run_dpo(cfg, task)
                        task.write_scores()
                else:
                    with MSB as benchmark:
                        for task in benchmark:
                            with task as scoring_function:
                                run_dpo(cfg, scoring_function)

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
                    run_dpo(cfg, task)
                else:
                    with task as scoring_function:
                        run_dpo(cfg, scoring_function)

        elif cfg.get("custom_task", None):
            if cfg.custom_task not in custom_scoring_functions:
                register_custom_scoring_function(cfg.custom_task, cfg.custom_task)
            task = Task(
                name=cfg.custom_task,
                scoring_function=custom_scoring_functions[cfg.custom_task],
                budget=cfg.total_smiles,
                output_dir=save_dir,
            )
            run_dpo(cfg, task)

        else:
            raise ValueError("No scoring function specified.")


def run_dpo(cfg, task):

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

    prior, _ = create_actor(vocabulary_size=len(vocabulary))
    prior.load_state_dict(adapt_state_dict(deepcopy(ckpt), prior.state_dict()))
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
            remove_duplicates=False,
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

        # Rank the rewards, the top 50% molecules will be considered positive samples
        # The rest will be the negative samples
        _, sscore_idxs = data_next["reward"][done].squeeze(-1).sort(descending=True)
        positive_data = data[sscore_idxs.data[: int(cfg.num_envs * 0.5)]]
        negative_data = data[sscore_idxs.data[int(cfg.num_envs * 0.5) :]]

        for _ in range(cfg.num_epochs):

            # Sampler to create mini-batches
            sampler = BatchSampler(
                SubsetRandomSampler(range(len(positive_data))),
                len(positive_data) // cfg.num_mini_batches,
                drop_last=True,
            )

            for idxs in sampler:

                (
                    loss,
                    prefered_relative_logprob,
                    disprefered_relative_logprob,
                    reward_margins,
                ) = compute_loss(
                    positive_data=positive_data[idxs],
                    negative_data=negative_data[idxs],
                    model=actor_training,
                    prior=prior,
                    beta=cfg.beta,
                )

                log_info.update(
                    {
                        "train/loss": loss.item(),
                        "train/prefered_relative_logprob": prefered_relative_logprob,
                        "train/disprefered_relative_logprob": (
                            disprefered_relative_logprob
                        ),
                        "train/reward_margin": reward_margins,
                    }
                )

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


def compute_loss(positive_data, negative_data, model, prior, beta=0.1):

    # Compute positive log-likelihoods
    pos_mask = positive_data.get("mask").squeeze(-1)
    pos_agent_likelihood = (get_log_prob(positive_data, model) * pos_mask).sum(-1)
    pos_prior_likelihood = (get_log_prob(positive_data, prior) * pos_mask).sum(-1)

    # Compute negative log-likelihoods
    neg_mask = negative_data.get("mask").squeeze(-1)
    neg_agent_likelihood = (get_log_prob(negative_data, model) * pos_mask).sum(-1)
    neg_prior_likelihood = (get_log_prob(negative_data, prior) * pos_mask).sum(-1)

    # Compute loss
    prefered_relative_logprob = pos_agent_likelihood - pos_prior_likelihood
    disprefered_relative_logprob = neg_agent_likelihood - neg_prior_likelihood
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(
        dim=-1
    )
    loss = -torch.nn.functional.logsigmoid(
        beta * (prefered_relative_logprob - disprefered_relative_logprob)
    ).mean(dim=-1)

    return (
        loss,
        prefered_relative_logprob.mean(dim=-1),
        disprefered_relative_logprob.mean(dim=-1),
        reward_margins,
    )


if __name__ == "__main__":
    main()
