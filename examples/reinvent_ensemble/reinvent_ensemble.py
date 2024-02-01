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
from acegen.data import is_in_reference, remove_duplicates
from acegen.models import adapt_state_dict, create_gru_actor, create_lstm_actor
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.vocabulary import SMILESVocabulary
from omegaconf import OmegaConf

from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    TensorDictMaxValueWriter,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import (
    CatFrames,
    InitTracker,
    StepCounter,
    TensorDictPrimer,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.record.loggers import get_logger


try:
    import molscore
    from molscore import MolScoreBenchmark
    from molscore.manager import MolScore

    _has_molscore = True
except ImportError as err:
    _has_molscore = False
    MOLSCORE_ERR = err


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):

    # Set seeds
    seed = cfg.seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # Save the config
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
    save_dir = f"{cfg.log_dir}_{timestamp_str}"
    os.makedirs(save_dir)
    with open(Path(save_dir) / "config.yaml", "w") as yaml_file:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

    if not _has_molscore:
        raise RuntimeError(
            "MolScore library not found, unable to create a scoring function. "
        ) from MOLSCORE_ERR

    if cfg.molscore in MolScoreBenchmark.presets:
        MSB = MolScoreBenchmark(
            model_name=cfg.agent_name,
            model_parameters=dict(cfg),
            benchmark=cfg.molscore,
            budget=cfg.total_smiles,
            output_dir=os.path.abspath(save_dir),
            include=cfg.molscore_include,
        )
        for task in MSB:
            run_reinvent_ensemble(cfg, task)
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
        )
        run_reinvent_ensemble(cfg, task)


def run_reinvent_ensemble(cfg, task):

    # Get available device
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )

    # Load Vocabulary
    ckpt = Path(__file__).resolve().parent.parent.parent / "priors" / cfg.vocabulary
    with open(ckpt, "r") as f:
        tokens = f.read().splitlines()
    tokens_dict = dict(zip(tokens, range(len(tokens))))
    vocabulary = SMILESVocabulary.create_from_dict(
        tokens_dict, start_token="GO", end_token="EOS"
    )

    # Model
    ####################################################################################################################

    ckpt = torch.load(
        Path(__file__).resolve().parent.parent.parent / "priors" / cfg.prior
    )

    if cfg.model == "gru":
        create_actor = create_gru_actor
    elif cfg.model == "lstm":
        create_actor = create_lstm_actor
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")

    actor_training, actor_inference = create_actor(vocabulary_size=len(vocabulary))
    actor_inference.load_state_dict(
        adapt_state_dict(ckpt, actor_inference.state_dict())
    )
    actor_training.load_state_dict(adapt_state_dict(ckpt, actor_training.state_dict()))
    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)

    prior = deepcopy(actor_training)
    actors_training = [actor_training] + [
        deepcopy(actor_training) for _ in range(cfg.num_nets - 1)
    ]
    actors_inference = [actor_inference] + [
        deepcopy(actor_inference) for _ in range(cfg.num_nets - 1)
    ]

    # Environment
    ####################################################################################################################

    # Create a transform to populate initial tensordict with rnn recurrent states equal to 0.0
    primers = actor_training.rnn_spec.expand(cfg.num_envs // cfg.num_nets)
    rhs_primer = TensorDictPrimer(primers)

    env_kwargs = {
        "start_token": vocabulary.vocab[vocabulary.start_token],
        "end_token": vocabulary.vocab[vocabulary.end_token],
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs // cfg.num_nets,
        "device": device,
    }

    def create_env_fn():
        """Create a single RL rl_env."""
        env = SMILESEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        env.append_transform(rhs_primer)
        return env

    # Replay buffer
    ####################################################################################################################

    storage = LazyTensorStorage(cfg.replay_buffer_size, device=device)
    experience_replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=PrioritizedSampler(storage.max_size, alpha=1.0, beta=1.0),
        batch_size=cfg.replay_batch_size,
        writer=TensorDictMaxValueWriter(rank_key="priority"),
        priority_key="priority",
    )

    # Optimizer
    ####################################################################################################################

    optims = []
    for actor_training, lr, eps, weight_decay in zip(
        actors_training, cfg.lr, cfg.eps, cfg.weight_decay
    ):
        optim = torch.optim.Adam(
            actor_training.parameters(),
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
        )
        optims.append(optim)

    # Logger
    ####################################################################################################################

    logger = None
    if cfg.logger_backend:
        logger = get_logger(
            cfg.logger_backend,
            logger_name="reinvent_ensemble",
            experiment_name=cfg.agent_name,
            wandb_kwargs={"config": dict(cfg), "project": cfg.experiment_name},
        )

    # Training loop
    ####################################################################################################################

    total_done = 0
    collected_frames = 0
    env = create_env_fn()
    sigma = cfg.sigma
    frames_in_batch = cfg.num_envs
    pbar = tqdm.tqdm(total=cfg.total_smiles)

    while not task.finished:

        data = []
        for actor_inference in actors_inference:
            data.append(
                generate_complete_smiles(policy=actor_inference, environment=env)
            )
        data = torch.cat(data, dim=0)
        data = remove_duplicates(data, key="action")

        log_info = {}
        total_done += frames_in_batch
        collected_frames += frames_in_batch
        data_next = data.get("next")
        done = data_next.get("done").squeeze(-1)
        smiles = data.select("action").cpu()
        pbar.update(done.sum().item())

        # Compute rewards
        smiles_str = [vocabulary.decode(smi.numpy()) for smi in smiles["action"]]
        data_next["reward"][done] = torch.tensor(
            task(smiles_str), device=device
        ).unsqueeze(-1)

        # Save info about smiles lengths and rewards
        episode_rewards = data_next["reward"][done]
        episode_length = data_next["step_count"][done]
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

        # Select only the necessary tensors
        data = data.select(
            "action",
            "mask",
            "is_init",
            "observation",
            "sample_log_prob",
            ("next", "reward"),
            inplace=True,
        )

        for num, (net_optim, actor_training, sigma) in enumerate(
            zip(optims, actors_training, cfg.sigma)
        ):

            data, loss, agent_likelihood = compute_loss(
                data, actor_training, prior, sigma
            )

            # Compute experience replay loss
            if (
                cfg.experience_replay
                and len(experience_replay_buffer) > cfg.replay_batch_size
            ):
                replay_batch = experience_replay_buffer.sample()
                _, replay_loss, replay_agent_likelihood = compute_loss(
                    replay_batch, actor_training, prior, sigma
                )
                loss = torch.cat((loss, replay_loss), 0)
                agent_likelihood = torch.cat(
                    (agent_likelihood, replay_agent_likelihood), 0
                )

            # Average loss over the batch
            loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = -(1 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p

            # Calculate gradients and make an update to the network weights
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Log info
            if logger:
                log_info.update(
                    {
                        f"train/loss_{num}": loss.item(),
                    }
                )

        # Then add new experiences to the replay buffer
        if cfg.experience_replay is True:

            replay_data = data.clone()

            # Remove SMILES that are already in the replay buffer
            if len(experience_replay_buffer) > 0:
                is_duplicated = is_in_reference(
                    tensordict=replay_data,
                    key="action",
                    reference_tensordict=experience_replay_buffer[:],
                )
                replay_data = replay_data[~is_duplicated]

            # Add data to the replay buffer
            reward = replay_data.get(("next", "reward"))
            replay_data.set("priority", reward)
            experience_replay_buffer.extend(replay_data)

        # Log info
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)


def get_log_prob(data, model):
    actions = data.get("action").clone()
    model_in = data.select(*model.in_keys, strict=False)
    log_prob = model.get_dist(model_in).log_prob(actions)
    return log_prob


def compute_loss(data, model, prior, sigma):

    mask = data.get("mask").squeeze(-1)

    if "prior_log_prob" not in data.keys():
        with torch.no_grad():
            prior_log_prob = get_log_prob(data, prior)
            data.set("prior_log_prob", prior_log_prob)
    else:
        prior_log_prob = data.get("prior_log_prob")

    agent_log_prob = get_log_prob(data, model)
    agent_likelihood = (agent_log_prob * mask).sum(-1)
    prior_likelihood = (prior_log_prob * mask).sum(-1)
    score = data.get(("next", "reward")).squeeze(-1).sum(-1)
    augmented_likelihood = prior_likelihood + sigma * score
    loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

    return data, loss, agent_likelihood


if __name__ == "__main__":
    main()
