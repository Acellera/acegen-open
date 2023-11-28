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
from pathlib import Path
from omegaconf import OmegaConf
from molscore.manager import MolScore

import torch
from torchrl.record.loggers import get_logger

from examples.models import get_model_factory
from experience_replay import Experience
from acegen.rl_environments import SingleStepDeNovoEnv
from acegen.vocabulary.vocabulary import Vocabulary

logging.basicConfig(level=logging.WARNING)


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


@hydra.main(config_path="../..", config_name="reinvent_config", version_base="1.2")
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

    # Save molscore output. Also redirect output to save_dir
    cfg.molscore = shutil.copy(cfg.molscore, save_dir)
    data = json.load(open(cfg.molscore, 'r'))
    data['output_dir'] = save_dir
    json.dump(data, open(cfg.molscore, 'w'), indent=4)

    # Create scoring function
    scoring = MolScore(model_name="ppo", task_config=cfg.molscore)
    scoring.configs["save_dir"] = save_dir
    scoring_function = scoring.score

    env_kwargs = {
        "start_token": vocabulary.vocab["GO"],
        "end_token": vocabulary.vocab["EOS"],
        "length_vocabulary": len(vocabulary),
        "vocabulary": vocabulary,
        "reward_function": scoring_function,
        "batch_size": cfg.num_envs,
        "device": device,
    }

    # Models
    ####################################################################################################################

    create_model = get_model_factory(cfg.model)
    ckpt = Path(__file__).resolve().parent / "models" / "priors" / "reinvent.ckpt"
    prior = create_model(vocabulary=vocabulary, ckpt_path=ckpt)
    model = create_model(vocabulary=vocabulary, ckpt_path=ckpt)
    prior = prior.to(device)
    model = model.to(device)

    # Environment
    ####################################################################################################################

    def create_env_fn():
        """Create a single RL rl_environments."""
        env = SingleStepDeNovoEnv(**env_kwargs)
        return env

    # Replay buffer
    ####################################################################################################################

    experience = Experience(vocabulary)

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        model.parameters(),
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
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    env = create_env_fn()
    sigma = cfg.sigma

    while collected_frames < cfg.total_frames:

        data = env.step(model(env.reset()))

        log_info = {}
        frames_in_batch = data.numel()
        total_done += data.get(("next", "terminated")).sum()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Calculate reward
        smiles_list = []
        for index, seq in enumerate(data.get("action")):
            smiles = vocabulary.decode(seq.cpu().numpy(), ignore_indices=[-1])
            smiles_list.append(smiles)
        score = torch.tensor(scoring_function(smiles_list), device=device)

        # Identify unique sequences
        arr = data.get("action").cpu().numpy()
        arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
        _, idxs = np.unique(arr_, return_index=True)
        unique_idxs = torch.tensor(np.sort(idxs), dtype=torch.int32, device=device)
        data = data[unique_idxs]
        score = score[unique_idxs]

        # Compute loss
        seqs = data.get("action")
        agent_likelihood = data.get("log_probs")
        with torch.no_grad():
            prior_likelihood = prior.likelihood(seqs)
        augmented_likelihood = prior_likelihood + sigma * score
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Compute experience replay loss
        if cfg.experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_seqs = exp_seqs.to(device)
            exp_score = exp_score.to(device)
            exp_prior_likelihood = exp_prior_likelihood.to(device)
            exp_agent_likelihood = model.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((exp_augmented_likelihood - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Average loss over the batch
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Then add new experience to replay buffer
        if cfg.experience_replay is True:
            new_experience = zip(smiles_list, score.cpu().numpy(), prior_likelihood.cpu().numpy())
            experience.add_experience(new_experience)

        # Log
        if logger:
            log_info.update(
                {
                    "train/total_smiles": total_done,
                    "train/reward": score.cpu().mean().item(),
                    "train/min_reward": score.cpu().min().item(),
                    "train/max_reward": score.cpu().max().item(),
                }
            )
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)


if __name__ == "__main__":
    main()

