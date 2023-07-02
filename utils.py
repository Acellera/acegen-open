#!/usr/bin/env python3

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import LSTMModule, MLP


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
        return out


def create_policy(vocabulary, output_size, out_key="logits"):

    embedding_module = TensorDictModule(
        Embed(len(vocabulary), 256),
        in_keys=["obs"],
        out_keys=["embed"],
    )

    lstm_module = LSTMModule(
        input_size=256,
        hidden_size=256,
        num_layers=3,
        in_key="embed",
        out_key="features",
    )
    mlp = TensorDictModule(
        MLP(
            in_features=256,
            out_features=output_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=[out_key],
    )

    return TensorDictSequential(embedding_module, lstm_module, mlp)


def create_critic(vocabulary, output_size, out_key="state_value"):

    embedding_module = TensorDictModule(
        Embed(len(vocabulary), 256),
        in_keys=["obs"],
        out_keys=["embed"],
    )
    mlp = TensorDictModule(
        MLP(
            in_features=256,
            out_features=output_size,
            num_cells=[256, 256, 256],
        ),
        in_keys=["embed"],
        out_keys=[out_key],
    )

    return TensorDictSequential(embedding_module, mlp)


def create_rhs_transform():
    lstm_module = LSTMModule(
        input_size=256,
        hidden_size=256,
        num_layers=3,
        in_key="embed",
        out_key="features",
    )
    return lstm_module.make_tensordict_primer()


