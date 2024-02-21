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
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.vocabulary import SMILESVocabulary
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.utils import isin, remove_duplicates
from torch.distributions.kl import kl_divergence
from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    SamplerWithoutReplacement,
    TensorDictMaxValueWriter,
    TensorDictReplayBuffer,
)
from torchrl.envs import (
    CatFrames,
    ExplorationType,
    InitTracker,
    StepCounter,
    TensorDictPrimer,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.objectives import ClipPPOLoss
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


@hydra.main(config_path=".", config_name="config2", version_base="1.2")
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
            run_ppo(cfg, task)
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
        run_ppo(cfg, task)


def run_ppo(cfg, task):

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
    if cfg.shared_nets:
        primers = actor_training.rnn_spec.expand(cfg.num_envs)
        rhs_primers = [TensorDictPrimer(primers)]
    else:
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
        env.append_transform(
            UnsqueezeTransform(
                in_keys=["observation"], out_keys=["SMILES"], unsqueeze_dim=-1
            )
        )
        env.append_transform(
            CatFrames(
                N=100,
                dim=-1,
                padding="constant",
                in_keys=["SMILES"],
                out_keys=["SMILES"],
                padding_value=-1,
            )
        )
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
        average_gae=False,
        shifted=True,
    )
    adv_module = adv_module.to(device)
    loss_module = ClipPPOLoss(
        actor=actor_training,
        critic=critic_training,
        critic_coef=cfg.critic_coef,
        entropy_coef=cfg.entropy_coef,
        clip_epsilon=cfg.ppo_clip,
        loss_critic_type="l2",
        normalize_advantage=True,
        reduction="none",
    )
    loss_module = loss_module.to(device)

    # PPO data Buffer
    ####################################################################################################################

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.num_envs + cfg.replay_batch_size, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.mini_batch_size,
        prefetch=4,
    )

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
            logger_name="ppo",
            experiment_name=cfg.agent_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.experiment_name,
                "group": cfg.agent_name,
            },
        )

    # Training loop
    ####################################################################################################################

    total_done = 0
    env = create_env_fn()
    kl_coef = cfg.kl_coef
    ppo_epochs = cfg.ppo_epochs
    max_grad_norm = cfg.max_grad_norm
    pbar = tqdm.tqdm(total=cfg.total_smiles)
    num_mini_batches = (cfg.num_envs + cfg.replay_batch_size) // cfg.mini_batch_size
    losses = TensorDict({}, batch_size=[cfg.ppo_epochs, num_mini_batches])

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
            "mask",
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
        data_select += (
            ["recurrent_state"]
            if cfg.shared_nets
            else ["recurrent_state_actor", "recurrent_state_critic"]
        )
        data_select += (
            [("next", "recurrent_state")]
            if cfg.shared_nets
            else [("next", "recurrent_state_actor"), ("next", "recurrent_state_critic")]
        )
        data = data.select(*data_select, inplace=True)

        # Get data to be potentially added to the replay buffer later
        replay_data = data.clone()

        for j in range(ppo_epochs):

            # Compute experience replay loss
            if (
                cfg.experience_replay
                and len(experience_replay_buffer) > cfg.replay_batch_size
            ):
                replay_batch = experience_replay_buffer.sample()
                replay_batch = replay_batch.exclude(
                    "_weight", "index", "priority", inplace=True
                )
                extended_data = torch.cat([data, replay_batch], dim=0)
            else:
                extended_data = data

            # Compute advantage and prior logits for extended_data
            with torch.no_grad():
                extended_data = adv_module(extended_data)

            # Add extended_data to PPO buffer
            buffer.extend(extended_data)

            for i, batch in enumerate(buffer):

                # PPO loss
                mask = batch.get("mask")

                loss = loss_module(batch)
                loss_sum = (
                    loss["loss_critic"] * mask
                    + loss["loss_objective"] * mask
                    + loss["loss_entropy"] * mask
                ).mean()
                losses[j, i] = TensorDict(
                    {
                        "loss_critic": (loss["loss_critic"] * mask).mean().item(),
                        "loss_objective": (loss["loss_objective"] * mask).mean().item(),
                        "loss_entropy": (loss["loss_entropy"] * mask).mean().item(),
                    },
                    batch_size=[],
                )

                # Add KL loss
                with torch.no_grad():
                    prior_dist = prior.get_dist(batch)

                kl_div = kl_divergence(actor_training.get_dist(batch), prior_dist)
                nan_mask = torch.isnan(kl_div) | torch.isinf(kl_div)
                kl_div = (kl_div[~nan_mask] * mask).mean()
                loss_sum += kl_div * kl_coef
                losses[j, i] = TensorDict(
                    {"kl_div": kl_div.detach().item()}, batch_size=[]
                )

                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=max_grad_norm
                )
                optim.step()
                optim.zero_grad()

        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})

            # Then add new experiences to the replay buffer
        if cfg.experience_replay is True:

            # Remove SMILES that are already in the replay buffer
            if len(experience_replay_buffer) > 0:
                is_duplicated = isin(
                    input=replay_data,
                    key="action",
                    reference=experience_replay_buffer[:],
                )
                replay_data = replay_data[~is_duplicated]

            # Add data to the replay buffer
            reward = replay_data.get(("next", "reward"))
            replay_data.set("priority", reward)
            experience_replay_buffer.extend(replay_data)

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=total_done)


if __name__ == "__main__":
    main()
