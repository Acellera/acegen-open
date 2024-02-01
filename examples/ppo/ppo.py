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
from acegen.data import is_in_reference, remove_duplicates, smiles_to_tensordict
from acegen.models import (
    adapt_state_dict,
    create_gru_actor,
    create_gru_actor_critic,
    create_gru_critic,
    create_lstm_actor,
    create_lstm_actor_critic,
    create_lstm_critic,
)
from acegen.rl_env import SMILESEnv
from acegen.vocabulary import SMILESVocabulary
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributions.kl import kl_divergence
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    SamplerWithoutReplacement,
    TensorDictMaxValueWriter,
    TensorDictReplayBuffer,
)
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

    if cfg.model == "gru":
        create_actor = create_gru_actor
        create_critic = create_gru_critic
        create_shared = create_gru_actor_critic
    elif cfg.model == "lstm":
        create_actor = create_lstm_actor
        create_critic = create_lstm_critic
        create_shared = create_lstm_actor_critic
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")

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
    ckpt = torch.load(
        Path(__file__).resolve().parent.parent.parent / "priors" / cfg.prior
    )
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
        "start_token": vocabulary.vocab[vocabulary.start_token],
        "end_token": vocabulary.vocab[vocabulary.end_token],
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

    # Collector
    ####################################################################################################################

    collector = SyncDataCollector(
        policy=actor_inference,
        create_env_fn=create_env_fn,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=-1,
        storing_device=device,
        device=device,
    )

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
    )
    loss_module = loss_module.to(device)

    # PPO data Buffer
    ####################################################################################################################

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.num_envs + cfg.replay_batches, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.mini_batch_size,
        prefetch=4,
    )

    # Replay data Buffer
    ####################################################################################################################

    experience_replay_buffer = None
    if cfg.experience_replay is True:
        replay_smiles_per_row = (
            8  # Estimation of how many SMILES needed to fill up a row of data
        )
        N = cfg.replay_batches * replay_smiles_per_row
        M = cfg.frames_per_batch // cfg.num_envs

        # Transform to populate recurrent states and is_init in replay batches
        replay_rhs_transform = TensorDictPrimer(
            actor_training.rnn_spec.expand(N, M - 1)
        )
        replay_logp_transform = TensorDictPrimer(
            {"sample_log_prob": UnboundedContinuousTensorSpec(shape=(N, M - 1))}
        )

        # Define a replay buffer to track best SMILES
        storage = LazyTensorStorage(100, device=device)
        experience_replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(100, device=device),
            sampler=PrioritizedSampler(storage.max_size, alpha=0.9, beta=1.0),
            batch_size=N,
            writer=TensorDictMaxValueWriter(rank_key="priority"),
            priority_key="priority",
        )

    def prepare_replay_batch(batch, T):
        """Prepare a batch of replay SMILES for PPO training."""

        # Populate with recurrent states
        replay_rhs_transform(batch)
        replay_rhs_transform(batch.get("next"))

        # Populate with is_init
        batch.set("is_init", batch.get(("next", "done")).roll(1, dims=1))
        batch.set(("next", "is_init"), torch.zeros_like(batch.get("is_init")))

        # Populate with sample log probs
        replay_logp_transform(batch)

        batches = batch.chunk(cfg.replay_batches)
        batches = [b[b.pop("mask")][..., 0:T].expand(1, T) for b in batches]
        return batches

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
    collected_frames = 0
    kl_coef = cfg.kl_coef
    ppo_epochs = cfg.ppo_epochs
    max_grad_norm = cfg.max_grad_norm
    pbar = tqdm.tqdm(total=cfg.total_smiles)
    num_mini_batches = (cfg.num_envs + cfg.replay_batches) // cfg.mini_batch_size
    losses = TensorDict({}, batch_size=[cfg.ppo_epochs, num_mini_batches])

    for data in collector:

        if task.finished:
            break

        log_info = {}
        frames_in_batch = data.numel()
        data_next = data.get("next")
        done = data_next.get("done").squeeze(-1)
        collected_frames += frames_in_batch
        total_done += done.sum().item()
        pbar.update(done.sum().item())

        # Compute rewards
        smiles = data_next.select("SMILES")[done].cpu()
        smiles_list = [
            vocabulary.decode(smi.numpy(), ignore_indices=[-1])
            for smi in smiles["SMILES"]
        ]

        # data_next["reward"][done] = torch.tensor(
        #     task(smiles_list), device=device
        # ).unsqueeze(-1)

        for _ in range(3):
            try:
                rews = task(smiles_list)
                break
            except Exception as e:
                print(f"Attempt failed with error: {e}")

        try:
            data_next["reward"][done] = torch.tensor(rews, device=device).unsqueeze(-1)
        except Exception:
            import ipdb

            ipdb.set_trace()

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

        # Get data to be added to the replay buffer later
        replay_data = (
            data.get("next").get_sub_tensordict(idx=done).select("SMILES", "reward")
        )

        # Select only the necessary tensors
        data = data.select(
            "action",
            "done",
            "is_init",
            "observation",
            "recurrent_state",
            "sample_log_prob",
            "terminated",
            ("next", "done"),
            ("next", "is_init"),
            ("next", "observation"),
            ("next", "recurrent_state"),
            ("next", "terminated"),
            ("next", "reward"),
            inplace=True,
        )

        for j in range(ppo_epochs):

            # Add some data from the replay data to the collected data
            if (
                experience_replay_buffer is not None
                and len(experience_replay_buffer) >= 50
            ):
                replay_batch = experience_replay_buffer.sample()
                replay_batch = replay_batch.select(
                    "mask", *data.keys(include_nested=True), strict=False
                )
                replay_batch.batch_size = torch.Size(
                    [*replay_batch["observation"].shape]
                )
                extended_data = torch.cat(
                    [data.clone(), *prepare_replay_batch(replay_batch, T=data.shape[1])]
                )
            else:
                extended_data = data

            # Compute advantage and prior logits for extended_data
            with torch.no_grad():
                extended_data = adv_module(extended_data)
                # prior_logits = prior(extended_data.select("is_init", "observation")).get("logits")
                # extended_data.set("prior_logits", prior_logits)

            # Add extended_data to PPO buffer
            buffer.extend(extended_data)

            for i, batch in enumerate(buffer):

                # Get next batch
                # batch = buffer.sample()

                # PPO loss
                loss = loss_module(batch)
                loss_sum = (
                    loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                )

                # Add KL loss
                # prior_dist = torch.distributions.Categorical(
                #     logits=batch.get("prior_logits")
                # )
                with torch.no_grad():
                    prior_dist = prior.get_dist(batch)
                # new_logits = prior(batch.select("is_init", "observation"))["logits"]

                # import ipdb; ipdb.set_trace()
                # assert batch["prior_logits"].sum().item() == new_logits.sum().item()

                kl_div = kl_divergence(actor_training.get_dist(batch), prior_dist)
                mask = torch.isnan(kl_div) | torch.isinf(kl_div)
                kl_div = kl_div[~mask].mean()
                loss_sum += kl_div * kl_coef

                # Add regularizer that penalizes high likelihood for the entire sequence
                # import ipdb
                # ipdb.set_trace()
                # loss_p = -(1 / batch.get("sample_log_pron")).mean()
                # loss_sum += 5 * 1e3 * loss_p

                # Register losses
                losses[j, i] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
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

        # Add data to the replay buffer
        if experience_replay_buffer is not None:

            # Remove duplicated SMILES
            replay_data = remove_duplicates(replay_data, key="SMILES")

            # Create a Tensordict with replay data
            replay_data = smiles_to_tensordict(
                replay_data["SMILES"], replay_data["reward"], device=device
            )
            replay_data.set("priority", replay_data.get(("next", "reward")))

            # Remove SMILES that are already in the replay buffer
            if len(experience_replay_buffer) > 0:
                is_duplicated = is_in_reference(
                    tensordict=replay_data,
                    key="action",
                    reference_tensordict=experience_replay_buffer[:],
                )
                replay_data = replay_data[~is_duplicated]

            # Add data to the replay buffer
            if len(replay_data) > 0:
                experience_replay_buffer.extend(replay_data)

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)
        collector.update_policy_weights_()

    collector.shutdown()


if __name__ == "__main__":
    main()
