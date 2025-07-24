#! /usr/bin/python3
import datetime
import os
import random
from pathlib import Path

import hydra
import numpy as np

import torch
import tqdm
import yaml
from acegen.data import load_dataset
from acegen.scoring_functions import (
    custom_scoring_functions,
    register_custom_scoring_function,
    Task,
)

from acegen.script_helpers import run_task, set_seed
from omegaconf import OmegaConf, open_dict
from packaging import version

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
    config_name="config",
    version_base="1.2",
)
def main(cfg: "DictConfig"):
    run_task(cfg, run_screening, __file__)


def run_screening(cfg, task):

    # Set seed
    set_seed(cfg.seed)

    # Load SMILES
    ####################################################################################################################
    all_smiles = load_dataset(cfg.dataset_path)

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

    # Iteratively score SMILES
    ####################################################################################################################
    # Randomise dataset
    random.shuffle(all_smiles)

    total_done = 0
    pbar = tqdm.tqdm(total=cfg.total_smiles)
    batch_size = 100

    while not task.finished:
        # Sample
        smiles = all_smiles[total_done : total_done + batch_size]
        # Score
        scores = task.score(smiles)
        # Update progress
        total_done += batch_size
        pbar.update(batch_size)
        log_info = {
            "screen/total_smiles": total_done,
            "screen/reward": scores.mean(),
            "screen/min_reward": scores.min(),
            "screen/max_reward": scores.max(),
        }
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=total_done)


if __name__ == "__main__":
    main()
