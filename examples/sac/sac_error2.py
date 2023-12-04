import torch
import tqdm
from pathlib import Path
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs.libs.gym import GymEnv
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.modules.distributions import OneHotCategorical
from torchrl.modules import ProbabilisticActor, GRUModule, MLP
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import DiscreteSACLoss
from torchrl.envs import (
    UnsqueezeTransform,
    ParallelEnv,
    TransformedEnv,
    InitTracker,
    StepCounter,
    RewardSum,
)
from acegen import SMILESVocabulary, MultiStepDeNovoEnv
from utils import create_sac_models


def main():

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Create test rl_environments to get action specs
    ckpt = Path(__file__).resolve().parent.parent.parent / "priors" / "reinvent_vocabulary.txt"
    vocabulary = SMILESVocabulary(ckpt)
    env_kwargs = {
        "start_token": vocabulary.vocab["GO"],
        "end_token": vocabulary.vocab["EOS"],
        "length_vocabulary": len(vocabulary),
        "batch_size": 1,
        "device": device,
        "one_hot_action_encoding": True,
    }

    # Models
    ####################################################################################################################

    (actor_inference, actor_training, critic_inference, critic_training, *transforms
     ) = create_sac_models(vocabulary_size=len(vocabulary), batch_size=1)

    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)
    critic_training = critic_training.to(device)

    # Environment
    ####################################################################################################################

    def create_env_fn():
        """Create a single RL rl_environments."""
        env = MultiStepDeNovoEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(UnsqueezeTransform(in_keys=["observation"], out_keys=["observation"], unsqueeze_dim=-1))
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        for transform in transforms:
            env.append_transform(transform)
        return env

    # Collector
    ####################################################################################################################

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=actor_inference,
        frames_per_batch=100,
        total_frames=1000,
        device=device,
        storing_device="cpu",
    )

    # Buffer
    ##################

    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(100),
        batch_size=8,
    )

    # Loss
    ##################

    loss_module = DiscreteSACLoss(
        actor_network=actor_training,
        qvalue_network=critic_training,
        num_actions=len(vocabulary),
        num_qvalue_nets=2,
        loss_function="smooth_l1",
    )
    loss_module.make_value_estimator(gamma=0.99)


    # Collection loop
    ##################

    for data in tqdm.tqdm(collector):
        buffer.extend(data.reshape(-1).cpu())
        batch = buffer.sample()
        loss = loss_module(batch.cuda())

    collector.shutdown()
    print("Success!")


if __name__ == "__main__":
    main()
