#!/usr/bin/rl_env python3

import copy

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import DiscreteTensorSpec, OneHotDiscreteTensorSpec
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec
from torchrl.envs import ExplorationType, TensorDictPrimer
from torchrl.modules import (
    ActorValueOperator,
    DistributionalQValueActor,
    GRUModule,
    MLP,
    ProbabilisticActor,
    QValueActor,
)


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
    )
    mlp = MLP(
        in_features=512,
        out_features=vocabulary_size,
        num_cells=[],
    )
    predictor = QValueActor(
        module=mlp,
        in_keys=["features"],
        spec=OneHotDiscreteTensorSpec(n=vocabulary_size),
    )

    model_inference = TensorDictSequential(embedding_module, gru_module, predictor)
    model_training = TensorDictSequential(
        embedding_module, gru_module.set_recurrent_mode(True), predictor
    )

    primers = {
        (f"recurrent_state",): UnboundedContinuousTensorSpec(
            shape=torch.Size([batch_size, 3, 512]),
            dtype=torch.float32,
        ),
    }
    transform = TensorDictPrimer(primers)

    return model_inference, model_training, transform


def create_dqn_models(vocabulary_size, batch_size, ckpt):
    critic_inference, critic_training, critic_transform = create_net(
        vocabulary_size, batch_size
    )
    initial_state_dict = copy.deepcopy(critic_training.state_dict())
    ckpt = adapt_ckpt(ckpt)
    critic_training.load_state_dict(ckpt)
    critic_inference.load_state_dict(ckpt)
    return critic_inference, critic_training, initial_state_dict, critic_transform


def adapt_ckpt(ckpt):

    keys_mapping = {
        "embedding.weight": "module.0.module._embedding.weight",
        "gru_1.weight_ih": "module.1.gru.weight_ih_l0",
        "gru_1.weight_hh": "module.1.gru.weight_hh_l0",
        "gru_1.bias_ih": "module.1.gru.bias_ih_l0",
        "gru_1.bias_hh": "module.1.gru.bias_hh_l0",
        "gru_2.weight_ih": "module.1.gru.weight_ih_l1",
        "gru_2.weight_hh": "module.1.gru.weight_hh_l1",
        "gru_2.bias_ih": "module.1.gru.bias_ih_l1",
        "gru_2.bias_hh": "module.1.gru.bias_hh_l1",
        "gru_3.weight_ih": "module.1.gru.weight_ih_l2",
        "gru_3.weight_hh": "module.1.gru.weight_hh_l2",
        "gru_3.bias_ih": "module.1.gru.bias_ih_l2",
        "gru_3.bias_hh": "module.1.gru.bias_hh_l2",
        "linear.weight": "module.2.module.0.module.0.weight",
        "linear.bias": "module.2.module.0.module.0.bias",
    }

    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[keys_mapping[k]] = v

    return new_ckpt
