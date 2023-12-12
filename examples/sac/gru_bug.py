import tqdm
import torch
import random
import numpy as np
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs.libs.gym import GymEnv
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.modules.distributions import OneHotCategorical
from torchrl.modules import ProbabilisticActor, GRUModule, MLP
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import DiscreteSACLoss
from torchrl.envs import (
    ParallelEnv,
    TransformedEnv,
    InitTracker,
    StepCounter,
    RewardSum,
)


def create_model(input_size, output_size, hidden_size=256, num_layers=3, out_key="logits"):

    embedding_module = TensorDictModule(
        in_keys=["observation"],
        out_keys=["embed"],
        module=torch.nn.Linear(input_size, input_size), # this raises RuntimeError
    )
    lstm_module = GRUModule(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        in_key="embed",
        out_key="features",
        python_based=True,
    )
    mlp = TensorDictModule(
        MLP(
            in_features=hidden_size,
            out_features=output_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=[out_key],
    )

    inference_model = TensorDictSequential(embedding_module, lstm_module, mlp)
    training_model = TensorDictSequential(embedding_module, lstm_module.set_recurrent_mode(), mlp)

    return inference_model, training_model


def create_rhs_transform(input_size, hidden_size=256, num_layers=3):
    lstm_module = GRUModule(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        in_key="observation",
        out_key="features",
    )
    return lstm_module.make_tensordict_primer()


def main():

    # Set seeds
    seed = 2024
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    test_env = GymEnv("CartPole-v1", device=device, categorical_action_encoding=True)
    action_spec = test_env.action_spec.space
    observation_spec = test_env.observation_spec["observation"]

    def create_env_fn():
        env = GymEnv("CartPole-v1", device=device)
        env = TransformedEnv(env)
        env.append_transform(create_rhs_transform(input_size=observation_spec.shape[-1]))
        env.append_transform(InitTracker())
        return env

    # Models
    ##################

    inference_actor, training_actor = create_model(input_size=observation_spec.shape[-1], output_size=action_spec.n)
    inference_actor = ProbabilisticActor(
        module=inference_actor,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    training_actor = ProbabilisticActor(
        module=training_actor,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )
    inference_actor = inference_actor.to(device)
    training_actor = training_actor.to(device)
    _, training_critic = create_model(input_size=observation_spec.shape[-1], output_size=action_spec.n, out_key="action_value")
    training_critic = training_critic.to(device)

    # Collector
    ##################

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=inference_actor,
        frames_per_batch=100,
        total_frames=5000,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        split_trajs=False,
    )

    # Buffer
    ##################

    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(100),
        batch_size=1,
    )

    # Loss
    ##################

    loss_module = DiscreteSACLoss(
        actor_network=training_actor,
        qvalue_network=training_critic,
        num_actions=action_spec.n,
        num_qvalue_nets=2,
        loss_function="smooth_l1",
    )
    loss_module.make_value_estimator(gamma=0.99)

    # Collection loop
    ##################

    for data in tqdm.tqdm(collector):
        buffer.extend(data.cpu())
        batch = buffer.sample()
        loss = loss_module(batch.cuda())
        loss_sum = loss["loss_actor"] + loss["loss_qvalue"] + loss["loss_alpha"]
        loss_sum.backward()

    collector.shutdown()
    print("Success!")


if __name__ == "__main__":
    main()
