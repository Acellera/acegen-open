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
from tensordict.nn import TensorDictSequential
from torchrl.envs import (
    CatFrames,
    InitTracker,
    StepCounter,
    TransformedEnv,
    UnsqueezeTransform,
    RandomCropTensorDict,
)
from torchrl.data.replay_buffers import RandomSampler
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import DQNLoss, SoftUpdate, HardUpdate
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.record.loggers import get_logger
from torchrl.modules import EGreedyModule

from acegen import SMILESVocabulary, MultiStepDeNovoEnv, SMILESReward, PenaliseRepeatedSMILES, BurnInTransform
from examples.dqn.sampler import CategoricalSamplingModule
from utils import create_dqn_models

logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):

    # Save config
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
    save_dir = f"{cfg.log_dir}_{timestamp_str}"
    os.makedirs(save_dir)
    with open(Path(save_dir) / "config.yaml", 'w') as yaml_file:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

    # Set seeds
    seed = cfg.seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # Get available device
    device = torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")

    # Environment
    ckpt = Path(__file__).resolve().parent.parent.parent / "priors" / "reinvent_vocabulary.txt"
    vocabulary = SMILESVocabulary(ckpt)
    env_kwargs = {
        "start_token": vocabulary.vocab["GO"],
        "end_token": vocabulary.vocab["EOS"],
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
        "one_hot_action_encoding": True,
    }

    # Models
    ####################################################################################################################

    ckpt = torch.load(Path(__file__).resolve().parent.parent.parent / "priors" / "reinvent.ckpt")
    (model_inference, model_training, initial_state_dict,  *transforms
     ) = create_dqn_models(vocabulary_size=len(vocabulary), batch_size=cfg.num_envs, ckpt=ckpt)

    model_training = model_training.to(device)
    model_inference = model_inference.to(device)
    sampling_module = CategoricalSamplingModule()
    greedy_module = EGreedyModule(
        annealing_num_steps=100,
        eps_init=0.1,
        eps_end=0.05,
        spec=model_inference[-1].spec,
    )

    model_explore = TensorDictSequential(
        model_inference,
        sampling_module,
    ).to(device)

    # model_explore = TensorDictSequential(
    #     model_inference,
    #     greedy_module,
    # ).to(device)

    # Environment
    ####################################################################################################################

    def create_env_fn():
        """Create a single RL rl_environments."""
        env = MultiStepDeNovoEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(UnsqueezeTransform(in_keys=["observation"], out_keys=["observation"], unsqueeze_dim=-1))
        env.append_transform(
            CatFrames(
                N=100, dim=-1, padding="constant", in_keys=["observation"], out_keys=["SMILES"], padding_value=-1))
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        for transform in transforms:
            env.append_transform(transform)
        return env

    # Save molscore output. Also redirect output to save_dir
    cfg.molscore = shutil.copy(cfg.molscore, save_dir)
    data = json.load(open(cfg.molscore, 'r'))
    data['output_dir'] = save_dir
    json.dump(data, open(cfg.molscore, 'w'), indent=4)

    # Create scoring function
    scoring = MolScore(model_name="dqn", task_config=cfg.molscore)
    scoring.configs["save_dir"] = save_dir
    scoring_function = scoring.score

    # Create reward transform
    rew_transform = SMILESReward(reward_function=scoring_function, vocabulary=vocabulary)

    # Collector
    ####################################################################################################################

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=model_explore,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        storing_device=device,
    )

    # Loss modules
    ####################################################################################################################

    loss_module = DQNLoss(
        value_network=model_training,
        gamma=cfg.gamma,
        loss_function="l2",
        delay_value=True,
        action_space=model_training[-1].spec,
    )
    loss_module.make_value_estimator()
    # target_net_updater = SoftUpdate(loss_module, eps=cfg.target_update_polyak)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg.value_network_update_interval
    )

    # Buffers
    ####################################################################################################################

    crop_seq = RandomCropTensorDict(sub_seq_len=cfg.sampled_sequence_length, sample_dim=-1)
    burn_in = BurnInTransform(rnn_modules=(model_training,), burn_in=cfg.burn_in)
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.replay_buffer_size),
        batch_size=cfg.batch_size,
        prefetch=3,
        # sampler=RandomSampler(),
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
            cfg.logger_backend, logger_name="dqn", experiment_name=cfg.agent_name, project_name=cfg.experiment_name
        )

    # Training loop
    ####################################################################################################################

    total_done = 0
    collected_frames = 0
    repeated_smiles = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    num_updates = cfg.num_loss_updates
    q_losses = torch.zeros(num_updates, device=device)
    kl_coef = cfg.kl_coef
    max_grad_norm = cfg.max_grad_norm
    loaded_initial_state_dict = False

    for data in collector:

        log_info = {}
        frames_in_batch = data.numel()
        total_done += data.get(("next", "terminated")).sum()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Compute all rewards in a single call
        data = rew_transform(data)

        # Register smiles lengths and real rewards
        episode_rewards = data["next", "reward"][data["next", "terminated"]]
        episode_length = data["next", "step_count"][data["next", "terminated"]]
        if len(episode_rewards) > 0:
            log_info.update(
                {
                    "train/total_smiles": total_done,
                    "train/reward": episode_rewards.mean().item(),
                    "train/min_reward": episode_rewards.min().item(),
                    "train/max_reward": episode_rewards.max().item(),
                    # "train/q_values": torch.gather(data["action_value"],
                    # 2, data["action"].unsqueeze(-1)).mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, collected_frames)

        # Update the replay buffer
        data = data.exclude(
            "embed",
            "logits",
            "features",
            "collector",
            "step_count",
            ("next", "step_count"),
            # "recurrent_state",
            # ("next", "recurrent_state"),
            "SMILES",
            ("next", "SMILES"),
            "action_value",
        )
        buffer.extend(data)

        if collected_frames < cfg.initial_frames:
            continue

        if not loaded_initial_state_dict:
            # Initialize final critic weights
            for layer in model_inference[-2].modules():
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.data.zero_()
                    layer.bias.data.zero_()
            model_training.load_state_dict(model_inference.state_dict())
            loaded_initial_state_dict = True

        for j in range(num_updates):
            log_info = {}
            sampled_tensordict = buffer.sample()

            assert "recurrent_state" in sampled_tensordict.keys()

            sampled_tensordict = sampled_tensordict.to(device)
            loss_td = loss_module(sampled_tensordict)
            q_loss = loss_td["loss"]
            optim.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(loss_module.parameters()), max_norm=cfg.max_grad_norm)
            optim.step()
            target_net_updater.step()
            q_losses[j].copy_(q_loss.detach())

            # Get and log q-values, loss, epsilon, sampling time and training time
            log_info.update(
                {
                    "train/q_loss": q_losses.mean().item(),
                }
            )

            print(j)

            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value) #  collected_frames)

        # update weights of the inference policy
        collector.update_policy_weights_()


if __name__ == "__main__":
    main()

