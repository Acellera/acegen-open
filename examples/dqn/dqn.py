import datetime
import json
import logging
import os
import random
import shutil
from pathlib import Path

import hydra
import numpy as np

import torch
import tqdm
import yaml

from acegen import SMILESEnv, SMILESVocabulary
from acegen.models import adapt_state_dict, create_gru_critic
from omegaconf import OmegaConf
from sampler import SoftmaxSamplingModule
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import (
    BurnInTransform,
    CatFrames,
    InitTracker,
    RandomCropTensorDict,
    StepCounter,
    TensorDictPrimer,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.modules import QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate
from torchrl.record.loggers import get_logger

logging.basicConfig(level=logging.WARNING)

try:
    import molscore
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
    ckpt = Path(__file__).resolve().parent.parent.parent / "priors" / cfg.vocabulary
    with open(ckpt, "r") as f:
        tokens = f.read().splitlines()
    tokens_dict = dict(zip(tokens, range(len(tokens))))
    vocabulary = SMILESVocabulary.create_from_dict(
        tokens_dict, start_token="GO", end_token="EOS"
    )

    # Models
    ####################################################################################################################

    ckpt = torch.load(
        Path(__file__).resolve().parent.parent.parent / "priors" / cfg.prior,
        map_location=device,
    )["critic"]
    model_training, model_inference = create_gru_critic(
        len(vocabulary),
        critic_value_per_action=True,
        python_based=True,
        dropout=0.01,
        layer_norm=True,
    )
    model_inference.load_state_dict(
        adapt_state_dict(ckpt, model_inference.state_dict())
    )
    model_training.load_state_dict(adapt_state_dict(ckpt, model_training.state_dict()))

    # Environment
    ####################################################################################################################

    # Create transform to populate initial tensordict with recurrent states equal to 0.0
    primers = model_training.rnn_spec.expand(cfg.num_envs)
    rhs_primers = [TensorDictPrimer(primers)]

    env_kwargs = {
        "start_token": vocabulary.vocab[vocabulary.start_token],
        "end_token": vocabulary.vocab[vocabulary.end_token],
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
        "max_length": 80,
        "one_hot_action_encoding": True,
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

    # tests env
    test_env = SMILESEnv(**env_kwargs)

    # Scoring transform - more efficient to do it outside the environment
    ####################################################################################################################

    if not _has_molscore:
        raise RuntimeError(
            "MolScore library not found, unable to create a scoring function. "
        ) from MOLSCORE_ERR

    if cfg.molscore is None:
        raise RuntimeError(
            "MolScore config file not provided, unable to create a scoring function. "
            "Please provide a config file,"
            "e.g. ../MolScore/molscore/configs/GuacaMol/Albuterol_similarity.json "
        )

    # Save molscore output. Also redirect output to save_dir
    cfg.molscore = shutil.copy(cfg.molscore, save_dir)
    data = json.load(open(cfg.molscore, "r"))
    data["output_dir"] = save_dir
    json.dump(data, open(cfg.molscore, "w"), indent=4)

    # Create scoring function
    scoring = MolScore(model_name="sac", task_config=cfg.molscore)
    scoring.configs["save_dir"] = save_dir
    scoring_function = scoring.score

    # Collector
    ####################################################################################################################

    model_inference = QValueActor(
        module=model_inference,
        action_space="one-hot",
        in_keys=model_inference.in_keys,
    )

    model_training = QValueActor(
        module=model_training,
        action_space="one-hot",
        in_keys=model_training.in_keys,
    )

    model_training = model_training.to(device)
    sampling_module = SoftmaxSamplingModule()
    model_explore = TensorDictSequential(model_inference, sampling_module).to(device)

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=model_explore,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=-1,
        device=device,
        storing_device=device,
        reset_at_each_iter=True,  # To avoid burn in issues
    )

    # Loss modules
    ####################################################################################################################

    loss_module = DQNLoss(
        value_network=model_training,
        gamma=cfg.gamma,
        loss_function="l2",  # smooth_l1
        delay_value=True,
        action_space="one-hot",
    )
    loss_module.make_value_estimator()
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg.value_network_update_interval
    )

    # Buffers
    ####################################################################################################################

    # crop_seq = RandomCropTensorDict(
    #     sub_seq_len=cfg.sampled_sequence_length, sample_dim=-1
    # )
    # burn_in = BurnInTransform(
    #     modules=(actor_training, critic_training), burn_in=cfg.burn_in
    # )
    # buffer = TensorDictReplayBuffer(
    #     storage=LazyMemmapStorage(cfg.replay_buffer_size),
    #     batch_size=cfg.batch_size,
    #     prefetch=3,
    # )
    buffer = TensorDictPrioritizedReplayBuffer(
        storage=LazyMemmapStorage(cfg.replay_buffer_size),
        alpha=0.7,
        beta=0.5,
        pin_memory=False,
        prefetch=3,
        batch_size=cfg.batch_size,
        priority_key="loss_qvalue",
    )

    # buffer.append_transform(crop_seq)
    # buffer.append_transform(burn_in)

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        eps=cfg.eps,
    )

    # Logger
    ####################################################################################################################

    logger = None
    if cfg.logger_backend:
        logger = get_logger(
            cfg.logger_backend,
            logger_name="dqn",
            experiment_name=cfg.agent_name,
            wandb_kwargs={"config": dict(cfg), "project": cfg.experiment_name},
        )

    # Training loop
    ####################################################################################################################

    total_done = 0
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_smiles)
    num_updates = cfg.num_loss_updates
    max_grad_norm = cfg.max_grad_norm

    for data in collector:

        log_info = {}
        data_next = data.get("next")
        done = data_next.get("done").squeeze(-1)
        frames_in_batch = data.numel()
        total_done += done.sum().item()
        collected_frames += frames_in_batch
        pbar.update(done.sum().item())

        if total_done >= cfg.total_smiles:
            break

        # Compute rewards
        smiles = data_next.select("SMILES")[done].clone().cpu()
        smiles_list = [
            vocabulary.decode(smi.numpy(), ignore_indices=[-1])
            for smi in smiles["SMILES"]
        ]
        data_next["reward"][done] = torch.tensor(
            scoring_function(smiles_list), device=device
        ).unsqueeze(-1)

        # Register smiles lengths and real rewards
        episode_rewards = data["next", "reward"][data["next", "done"]]
        episode_length = data["next", "step_count"][data["next", "done"]]
        if len(episode_rewards) > 0:
            log_info.update(
                {
                    "train/total_smiles": total_done,
                    "train/reward": episode_rewards.mean().item(),
                    "train/min_reward": episode_rewards.min().item(),
                    "train/max_reward": episode_rewards.max().item(),
                    "train/q_values": data["chosen_action_value"].mean().item(),
                    "train/episode_length": episode_length.sum().item() / len(
                        episode_length
                    ),
                }
            )

            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, collected_frames)

        # Update the replay buffer
        data = data.exclude(
            "done" "embed",
            "logits",
            "features",
            "collector",
            "step_count",
            ("next", "step_count"),
            "recurrent_state",
            ("next", "recurrent_state"),
            "SMILES",
            ("next", "SMILES"),
        )

        buffer.extend(data)

        if total_done < cfg.init_random_smiles:
            continue

        for j in range(num_updates):
            sampled_tensordict = buffer.sample()
            sampled_tensordict = sampled_tensordict.to(device)

            if logger:
                logger.log_scalar(
                    "train/chosen_action_value",
                    sampled_tensordict["chosen_action_value"].mean().item(),
                    collected_frames,
                )
                logger.log_scalar(
                    "train/action_value",
                    sampled_tensordict["action_value"].mean().item(),
                    collected_frames,
                )

            loss_td = loss_module(sampled_tensordict)
            q_loss = loss_td["loss"]
            optim.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(loss_module.parameters()), max_norm=max_grad_norm
            )
            optim.step()
            target_net_updater.step()
            if logger:
                logger.log_scalar("train/q_loss", q_loss.item())

        # update weights of the inference policy
        collector.update_policy_weights_()


if __name__ == "__main__":
    main()
