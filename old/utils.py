#!/usr/bin/env python3

from pathlib import Path
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import (
    LSTMModule,
    MLP,
    ActorValueOperator,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.envs import ExplorationType
from torchrl.collectors.utils import split_trajectories


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
        in_keys=[
            "embed",
            f"recurrent_state_h_{net_name}",
            f"recurrent_state_c_{net_name}",
        ],
        out_keys=[
            "features",
            ("next", f"recurrent_state_h_{net_name}"),
            ("next", f"recurrent_state_c_{net_name}"),
        ],
    )
    mlp = TensorDictModule(
        MLP(
            in_features=512,
            out_features=output_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=[out_key],
    )

    model_inference = TensorDictSequential(embedding_module, lstm_module, mlp)
    model_training = TensorDictSequential(
        embedding_module, lstm_module.set_recurrent_mode(True), mlp
    )
    transform = lstm_module.make_tensordict_primer()

    return model_inference, model_training, transform
