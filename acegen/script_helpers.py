import datetime
import os
import random
from pathlib import Path

import numpy as np

import torch
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from packaging import version

from acegen.scoring_functions import (
    custom_scoring_functions,
    register_custom_scoring_function,
    Task,
)

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


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def run_task(cfg: DictConfig, algorithm: callable, script_path: str = None):
    """Shared pre-amble to interpret the config and run the task(s) for the seed(s)."""
    if isinstance(cfg.seed, int):
        seeds = [cfg.seed]
    else:
        seeds = cfg.seed

    for seed in seeds:

        # Set seed
        cfg.seed = int(seed)

        # Define save_dir and save config
        current_time = datetime.datetime.now()
        timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
        os.chdir(os.path.dirname(script_path))
        save_dir = (
            f"{cfg.log_dir}/{cfg.experiment_name}_{cfg.agent_name}_{timestamp_str}"
        )
        with open_dict(cfg):
            cfg.save_dir = save_dir
            cfg.script = script_path
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
                    oracle_budget=cfg.get("oracle_budget", False),
                    output_dir=os.path.abspath(save_dir),
                    add_run_dir=False,
                    **cfg.get("molscore_kwargs", {}),
                )
                if _molscore_version < version.parse("2.0"):
                    algorithm(cfg, task)
                else:
                    with task as scoring_function:
                        algorithm(cfg, scoring_function)

            if cfg.molscore_mode == "benchmark":
                MSB = MolScoreBenchmark(
                    model_name=cfg.agent_name,
                    model_parameters=dict(cfg),
                    benchmark=cfg.molscore_task,
                    budget=cfg.total_smiles,
                    oracle_budget=cfg.get("oracle_budget", False),
                    output_dir=os.path.abspath(save_dir),
                    add_benchmark_dir=False,
                    **cfg.get("molscore_kwargs", {}),
                )
                if _molscore_version < version.parse("2.0"):
                    for task in MSB:
                        algorithm(cfg, task)
                        task.write_scores()
                        torch.cuda.empty_cache()
                else:
                    with MSB as benchmark:
                        for task in benchmark:
                            with task as scoring_function:
                                algorithm(cfg, scoring_function)
                                torch.cuda.empty_cache()

            if cfg.molscore_mode == "curriculum":
                task = MolScoreCurriculum(
                    model_name=cfg.agent_name,
                    model_parameters=dict(cfg),
                    benchmark=cfg.molscore_task,
                    budget=cfg.total_smiles,
                    oracle_budget=cfg.get("oracle_budget", False),
                    output_dir=os.path.abspath(save_dir),
                    **cfg.get("molscore_kwargs", {}),
                )
                if _molscore_version < version.parse("2.0"):
                    algorithm(cfg, task)
                else:
                    with task as scoring_function:
                        algorithm(cfg, scoring_function)

        elif cfg.get("custom_task", None):
            if cfg.custom_task not in custom_scoring_functions:
                register_custom_scoring_function(cfg.custom_task, cfg.custom_task)
            task = Task(
                name=cfg.custom_task,
                scoring_function=custom_scoring_functions[cfg.custom_task],
                budget=cfg.total_smiles,
                output_dir=save_dir,
            )
            algorithm(cfg, task)

        else:
            raise ValueError("No scoring function specified.")
