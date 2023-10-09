#!/usr/bin/env python3

import torch
from torchrl.modules import LSTMModule, MLP, ActorValueOperator, ProbabilisticActor, ValueOperator
from torchrl.envs import ExplorationType
from tensordict.nn import TensorDictModule, TensorDictSequential


class Embed(torch.nn.Module):
    """Implements a simple embedding layer."""

    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self._embedding = torch.nn.Embedding(input_size, embedding_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, L = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        out = self._embedding(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        out = out.squeeze()  # This is an ugly hack, should not be necessary
        return out


def create_model(vocabulary, output_size, net_name="actor", out_key="logits"):

    embedding_module = TensorDictModule(
        Embed(len(vocabulary), 256),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    lstm_module = LSTMModule(
        input_size=256,
        hidden_size=512,
        num_layers=3,
        in_keys=["embed", f"recurrent_state_h_{net_name}", f"recurrent_state_c_{net_name}"],
        out_keys=["features", ("next", f"recurrent_state_h_{net_name}"), ("next", f"recurrent_state_c_{net_name}")])
    mlp = TensorDictModule(
        MLP(
            in_features=512,
            out_features=output_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=[out_key],
    )

    policy_inference = TensorDictSequential(embedding_module, lstm_module, mlp)
    policy_training = TensorDictSequential(embedding_module, lstm_module.set_recurrent_mode(True), mlp)
    transform = lstm_module.make_tensordict_primer()

    return policy_inference, policy_training, transform


def create_shared_model(vocabulary, output_size, out_key="logits"):

    embedding_module = TensorDictModule(
        Embed(len(vocabulary), 256),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    lstm_module = LSTMModule(
        input_size=256,
        hidden_size=512,
        num_layers=3,
        in_keys=["embed", "recurrent_state_h", "recurrent_state_c"],
        out_keys=["features", ("next", "recurrent_state_h"), ("next", "recurrent_state_c")])
    actor_model = TensorDictModule(
        MLP(
            in_features=512,
            out_features=output_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=[out_key],
    )
    policy_module = ProbabilisticActor(
        module=actor_model,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    critic_module = ValueOperator(
        MLP(
            in_features=512,
            out_features=1,
            num_cells=[],
        ),
        in_keys=["features"],
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic_inference = ActorValueOperator(
        common_operator=TensorDictSequential(embedding_module, lstm_module),
        policy_operator=policy_module,
        value_operator=critic_module,
    )

    actor_critic_training = ActorValueOperator(
        common_operator=TensorDictSequential(embedding_module, lstm_module.set_recurrent_mode(True)),
        policy_operator=policy_module,
        value_operator=critic_module,
    )

    actor_inference = actor_critic_inference.get_policy_operator()
    critic_inference = actor_critic_inference.get_value_operator()
    actor_training = actor_critic_training.get_policy_operator()
    critic_training = actor_critic_training.get_value_operator()
    transform = lstm_module.make_tensordict_primer()

    return actor_inference, actor_training, critic_inference, critic_training, transform
