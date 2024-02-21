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
from acegen.models import (
    adapt_state_dict,
    create_gpt2_actor,
    create_gpt2_actor_critic,
    create_gpt2_critic,
    create_gru_actor,
    create_gru_actor_critic,
    create_gru_critic,
    create_lstm_actor,
    create_lstm_actor_critic,
    create_lstm_critic,
)
from tensordict.utils import remove_duplicates
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.vocabulary import SMILESVocabulary
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributions.kl import kl_divergence
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from torchrl.envs import (
    CatFrames,
    InitTracker,
    StepCounter,
    TensorDictPrimer,
    TransformedEnv,
    UnsqueezeTransform,
)
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


default_model_map = {
    "gru": (
        create_gru_actor,
        create_gru_critic,
        create_gru_actor_critic,
        "chembl_filtered_vocabulary.txt",
        "gru_chembl_filtered.ckpt",
    ),
    "lstm": (
        create_lstm_actor,
        create_lstm_critic,
        create_lstm_actor_critic,
        "chembl_vocabulary.txt",
        "lstm_chembl.ckpt",
    ),
    "gpt2": (
        create_gpt2_actor,
        create_gpt2_critic,
        create_gpt2_actor_critic,
        "enamine_real_vocabulary.txt",
        "gpt2_enamine_real.ckpt",
    ),
}


@hydra.main(config_path=".", config_name="config_simplified", version_base="1.2")
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
        )
        run_a2c(cfg, task)


def run_a2c(cfg, task):

    # Get available device
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )

    if cfg.model in default_model_map:
        create_actor, create_critic, create_shared, vocab_file, weights_file = (
            default_model_map[cfg.model]
        )
        voc_path = (
            Path(__file__).resolve().parent.parent.parent / "priors" / vocab_file
            if cfg.prior == "default"
            else Path(cfg.prior)
        )
        ckpt_path = (
            Path(__file__).resolve().parent.parent.parent / "priors" / weights_file
            if cfg.prior == "default"
            else Path(cfg.prior)
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")

    # Vocabulary
    ####################################################################################################################

    with open(voc_path, "r") as f:
        tokens = f.read().splitlines()
    tokens_dict = dict(zip(tokens, range(len(tokens))))
    vocabulary = SMILESVocabulary.create_from_dict(
        tokens_dict, start_token="GO", end_token="EOS"
    )

    # Model
    ####################################################################################################################

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

    # Environment
    ####################################################################################################################

    # Create a transform to populate initial tensordict with rnn recurrent states equal to 0.0
    rhs_primers = []
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
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        for rhs_primer in rhs_primers:
            env.append_transform(rhs_primer)
        return env

    # Loss modules
    ####################################################################################################################

    adv_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic_training,
        average_gae=True,
        shifted=True,
    )
    adv_module = adv_module.to(device)
    loss_module = A2CLoss(
        actor_network=actor_training,
        critic_network=critic_training,
        critic_coef=cfg.critic_coef,
        entropy_coef=cfg.entropy_coef,
        loss_critic_type="l2",
    )
    loss_module = loss_module.to(device)

    # A2C data Buffer
    ####################################################################################################################

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.num_envs, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.mini_batch_size,
        prefetch=4,
    )

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        loss_module.parameters(),
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
    env = create_env_fn()
    kl_coef = cfg.kl_coef
    max_grad_norm = cfg.max_grad_norm
    pbar = tqdm.tqdm(total=cfg.total_smiles)
    num_mini_batches = cfg.num_envs // cfg.mini_batch_size
    losses = TensorDict({}, batch_size=[num_mini_batches])

    while not task.finished:

        data = generate_complete_smiles(policy=actor_inference, environment=env)
        data = remove_duplicates(data, key="action")

        log_info = {}
        data_next = data.get("next")
        done = data_next.get("done").squeeze(-1)
        total_done += cfg.num_envs
        smiles = data.select("action").cpu()
        pbar.update(done.sum().item())

        # Compute rewards
        smiles_str = [vocabulary.decode(smi.numpy()) for smi in smiles["action"]]
        data_next["reward"][done] = torch.tensor(
            task(smiles_str), device=device
        ).unsqueeze(-1)

        # Register smiles lengths and real rewards
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
        data_select = [
            "action",
            "done",
            "is_init",
            "observation",
            "sample_log_prob",
            "terminated",
            ("next", "done"),
            ("next", "is_init"),
            ("next", "observation"),
            ("next", "terminated"),
            ("next", "reward"),
        ]
        if hasattr(actor_training, "rnn_spec"):
            data_select += (
                ["recurrent_state"]
                if cfg.shared_nets
                else ["recurrent_state_actor", "recurrent_state_critic"]
            )
            data_select += (
                [("next", "recurrent_state")]
                if cfg.shared_nets
                else [
                    ("next", "recurrent_state_actor"),
                    ("next", "recurrent_state_critic"),
                ]
            )
        data = data.select(*data_select, inplace=True)

        # For transformers-based policies
        data.set("sequence", data.get("observation"))
        data.set(("next", "sequence"), data.get(("next", "observation")))

        # Compute advantage
        with torch.no_grad():
            data = adv_module(data)

        buffer.extend(data)

        for j, batch in enumerate(buffer):

            batch = batch.to(device, non_blocking=True)

            # Compute loss
            mask = batch.get("mask")
            loss = loss_module(batch)
            loss = loss.apply(lambda x: (x * mask).mean(), batch_size=[])
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
            nan_mask = torch.isnan(kl_div) | torch.isinf(kl_div)
            kl_div = (kl_div * mask.squeeze())[~nan_mask].mean()
            loss_sum += kl_div * kl_coef
            losses[j] = TensorDict(
                {"kl_div": kl_div.detach().item()}, batch_size=[]
            )

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
