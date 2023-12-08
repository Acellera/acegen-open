#!/usr/bin/env python3

import copy
from pathlib import Path
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import (
    GRUModule,
    MLP,
    ActorValueOperator,
    ProbabilisticActor,
    QValueActor,
    DistributionalQValueActor,
)
from torchrl.modules.distributions import OneHotCategorical
from torchrl.envs import ExplorationType, TensorDictPrimer
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec
from tensordict.nn import TensorDictSequential
from torchrl.data import DiscreteTensorSpec


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
        inputs = inputs.squeeze(-1)  # Embedding creates an extra dimension
        out = self._embedding(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out


def create_net(vocabulary_size, batch_size):

    embedding_module = TensorDictModule(
        Embed(vocabulary_size, 128),
        in_keys=["observation"],
        out_keys=["embed"],
    )

    gru_module = GRUModule(
        dropout=0.0,
        input_size=128,
        hidden_size=512,
        num_layers=3,
        in_keys=["embed", f"recurrent_state", "is_init"],
        out_keys=[
            "features",
            ("next", f"recurrent_state"),
        ],
        python_based=True,
    )
    mlp = TensorDictModule(
        MLP(
            in_features=512,
            out_features=vocabulary_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=["action_value"],
    )

    model_inference = TensorDictSequential(embedding_module, gru_module, mlp)
    model_training = TensorDictSequential(embedding_module, gru_module.set_recurrent_mode(True), mlp)

    model_inference = QValueActor(
        module=model_inference,
        in_keys=["action_value"],
        spec=DiscreteTensorSpec(vocabulary_size),
        action_space="categorical",
    )
    model_training = QValueActor(
        module=model_training,
        in_keys=["action_value"],
        spec=DiscreteTensorSpec(vocabulary_size),
        action_space="categorical",
    )

    primers = {
        (f"recurrent_state",):
            UnboundedContinuousTensorSpec(
                shape=torch.Size([batch_size, 3, 512]),
                dtype=torch.float32,
            ),
    }
    transform = TensorDictPrimer(primers)

    return model_inference, model_training, transform


def create_dqn_models(vocabulary_size, batch_size, ckpt):

    critic_inference, critic_training, critic_transform = create_net(vocabulary_size, batch_size)
    ckpt = adapt_ckpt(ckpt)
    critic_training.load_state_dict(ckpt)

    # Initialize final critic weights
    for layer in critic_training[0][2].module[0].modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    critic_inference.load_state_dict(critic_training.state_dict())

    return critic_inference, critic_training, critic_transform


def adapt_ckpt(ckpt):

    keys_mapping = {
        'embedding.weight': "module.0.module.0.module._embedding.weight",

        'gru_1.weight_ih': "module.0.module.1.gru.weight_ih_l0",
        'gru_1.weight_hh': "module.0.module.1.gru.weight_hh_l0",
        'gru_1.bias_ih': "module.0.module.1.gru.bias_ih_l0",
        'gru_1.bias_hh': "module.0.module.1.gru.bias_hh_l0",

        'gru_2.weight_ih': "module.0.module.1.gru.weight_ih_l1",
        'gru_2.weight_hh': "module.0.module.1.gru.weight_hh_l1",
        'gru_2.bias_ih': "module.0.module.1.gru.bias_ih_l1",
        'gru_2.bias_hh': "module.0.module.1.gru.bias_hh_l1",

        'gru_3.weight_ih': "module.0.module.1.gru.weight_ih_l2",
        'gru_3.weight_hh': "module.0.module.1.gru.weight_hh_l2",
        'gru_3.bias_ih': "module.0.module.1.gru.bias_ih_l2",
        'gru_3.bias_hh': "module.0.module.1.gru.bias_hh_l2",

        'linear.weight': "module.0.module.2.module.0.weight",
        'linear.bias': "module.0.module.2.module.0.bias",
    }

    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[keys_mapping[k]] = v

    return new_ckpt
