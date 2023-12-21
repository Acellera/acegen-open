import os
import yaml
import hydra
import random
import logging
import datetime
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from acegen import SMILESVocabulary, MultiStepDeNovoEnv
from sampler import SoftmaxSamplingModule
from utils import create_dqn_models
from torchrl.envs import (
    InitTracker,
    StepCounter,
    TransformedEnv,
)
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
    with open(Path(save_dir) / "config.yaml", 'w') as yaml_file:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

    # Get available device
    device = torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")

    # Vocabulary
    ckpt = Path(__file__).resolve().parent.parent.parent / "priors" / "reinvent_vocabulary.txt"
    vocabulary = SMILESVocabulary(ckpt)

    # Models
    ####################################################################################################################

    ckpt = torch.load(Path(__file__).resolve().parent.parent.parent / "priors" / "reinvent.ckpt", map_location=device)
    (model_inference, model_training, initial_state_dict,  *transforms
     ) = create_dqn_models(vocabulary_size=len(vocabulary), batch_size=cfg.num_envs, ckpt=ckpt)

    model_inference = model_inference.to(device)
    sampling_module = SoftmaxSamplingModule(spec=model_inference[-1].spec)
    model_explore = TensorDictSequential(model_inference, sampling_module).to(device)

    # Environment
    ####################################################################################################################

    env_kwargs = {
        "start_token": vocabulary.vocab["GO"],
        "end_token": vocabulary.vocab["EOS"],
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
        "one_hot_action_encoding": True,
    }

    def create_env_fn():
        """Create a single RL rl_environments."""
        env = MultiStepDeNovoEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        for transform in transforms:
            env.append_transform(transform)
        return env

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

    for data in collector:

        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
