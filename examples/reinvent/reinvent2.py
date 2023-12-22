import datetime
import json
import logging
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np

import torch

import torch.nn.functional as F
import tqdm
import yaml
from acegen.models import adapt_state_dict, create_gru_actor
from acegen.rl_env import sample_completed_smiles, SMILESEnv
from acegen.transforms import BurnInTransform, PenaliseRepeatedSMILES, SMILESReward
from acegen.vocabulary import SMILESVocabulary
from molscore.manager import MolScore
from omegaconf import OmegaConf
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatFrames,
    ExplorationType,
    InitTracker,
    StepCounter,
    TensorDictPrimer,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.record.loggers import get_logger
from utils import Experience

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

    ckpt = torch.load(
        Path(__file__).resolve().parent.parent.parent / "priors" / "reinvent.ckpt"
    )
    _, actor = create_gru_actor(len(vocabulary))
    actor.load_state_dict(adapt_state_dict(ckpt, actor.state_dict()))
    actor = actor.to(device)
    prior = deepcopy(actor)

    # Environment
    ####################################################################################################################

    num_layers = 3
    hidden_size = 512
    primers = {
        ("recurrent_state_actor",): UnboundedContinuousTensorSpec(
            shape=torch.Size([cfg.num_envs, num_layers, hidden_size]),
            dtype=torch.float32,
        ),
    }
    rhs_primers = [TensorDictPrimer(primers)]

    env_kwargs = {
        "start_token": vocabulary.vocab[vocabulary.start_token],
        "end_token": vocabulary.vocab[vocabulary.end_token],
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
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

    # Scoring transform - more efficient to do it outside the environment
    ####################################################################################################################

    # Save molscore output. Also redirect output to save_dir
    cfg.molscore = shutil.copy(cfg.molscore, save_dir)
    data = json.load(open(cfg.molscore, "r"))
    data["output_dir"] = save_dir
    json.dump(data, open(cfg.molscore, "w"), indent=4)

    # Create scoring function
    scoring = MolScore(model_name="ppo", task_config=cfg.molscore)
    scoring.configs["save_dir"] = save_dir
    scoring_function = scoring.score

    # Create reward transform
    rew_transform = SMILESReward(
        reward_function=scoring_function,
        vocabulary=vocabulary,
    )

    # Replay buffer
    ####################################################################################################################

    experience = Experience(vocabulary)

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        actor.parameters(),
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
            logger_name="reinvent",
            experiment_name=cfg.agent_name,
            project_name=cfg.experiment_name,
        )

    # Training loop
    ####################################################################################################################

    total_done = 0
    collected_frames = 0
    env = create_env_fn()
    sigma = cfg.sigma
    frames_in_batch = cfg.num_envs

    for _ in tqdm.tqdm(range(0, cfg.total_frames, frames_in_batch)):

        data = sample_completed_smiles(policy=actor, environment=env)

        log_info = {}
        total_done += data.get(("next", "done")).sum()
        collected_frames += frames_in_batch

        # Compute reward
        data = rew_transform(data)

        # Identify unique sequences
        arr = data.get("action").cpu().numpy()
        arr_ = np.ascontiguousarray(arr).view(
            np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
        )
        _, idxs = np.unique(arr_, return_index=True)
        unique_idxs = torch.tensor(np.sort(idxs), dtype=torch.int32, device=device)
        data = data[unique_idxs]

        # Compute prior likelihood
        with torch.no_grad():
            prior_logits = prior(data.select(*prior.in_keys).clone()).get("logits")
            prior_log_prob = F.log_softmax(prior_logits, dim=-1)
            prior_log_prob = prior_log_prob.gather(
                -1, data.get("action").unsqueeze(-1)
            ).squeeze(-1)

        # Compute loss
        agent_likelihood = data.get("sample_log_prob").sum(-1)
        prior_likelihood = prior_log_prob.sum(-1)
        score = data.get(("next", "reward")).squeeze(-1).sum(-1)
        augmented_likelihood = prior_likelihood + sigma * score
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Compute experience replay loss
        if cfg.experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_seqs = exp_seqs.to(device)
            exp_score = exp_score.to(device)
            exp_prior_likelihood = exp_prior_likelihood.to(device)
            exp_agent_likelihood = actor.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((exp_augmented_likelihood - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Average loss over the batch
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = -(1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Then add new experience to replay buffer
        if cfg.experience_replay is True:
            smiles_list = []
            for index, seq in enumerate(data.get("action")):
                smiles = vocabulary.decode(seq.cpu().numpy(), ignore_indices=[-1])
                smiles_list.append(smiles)
            new_experience = zip(
                smiles_list, score.cpu().numpy(), prior_likelihood.cpu().numpy()
            )
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
