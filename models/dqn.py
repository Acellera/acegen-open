#!/usr/bin/env python3

import copy
from pathlib import Path
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import (
    LSTMModule,
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
        out = self._embedding(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        out = out.squeeze(-2)  # This is an ugly hack, should not be necessary
        return out


def create_net(vocabulary_size, batch_size):
    embedding_module = TensorDictModule(
        Embed(vocabulary_size, 256),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    lstm_module = LSTMModule(
        dropout=0.1,
        input_size=256,
        hidden_size=512,
        num_layers=3,
        in_keys=[
            "embed",
            f"recurrent_state_h",
            f"recurrent_state_c",
        ],
        out_keys=[
            "features",
            ("next", f"recurrent_state_h"),
            ("next", f"recurrent_state_c"),
        ],
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
    model_inference = QValueActor(
    # model_inference = DistributionalQValueActor(
        TensorDictSequential(embedding_module, lstm_module, mlp),
        spec=DiscreteTensorSpec(vocabulary_size),
        # support=torch.arange(vocabulary_size)
    )
    model_training = QValueActor(
    # model_training = DistributionalQValueActor(
        TensorDictSequential(embedding_module, lstm_module.set_recurrent_mode(True), mlp),
        spec=DiscreteTensorSpec(vocabulary_size),
        # support=torch.arange(vocabulary_size)
    )

    primers = {
        ('recurrent_state_h',):
            UnboundedContinuousTensorSpec(
                shape=torch.Size([batch_size, 3, 512]),
                dtype=torch.float32,
            ),
        ('recurrent_state_c',):
            UnboundedContinuousTensorSpec(
                shape=torch.Size([batch_size, 3, 512]),
                dtype=torch.float32),
    }
    transform = TensorDictPrimer(primers)

    return model_inference, model_training, transform


def create_dqn_models(vocabulary_size, batch_size):

    critic_inference, critic_training, critic_transform = create_net(vocabulary_size, batch_size)
    initial_state_dict = critic_inference.state_dict()
    ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "chembl_actor.prior")
    ckpt = adapt_ckpt(ckpt)
    critic_inference.load_state_dict(ckpt)
    critic_training.load_state_dict(ckpt)

    return critic_inference, critic_training, initial_state_dict, critic_transform


def adapt_ckpt(ckpt):

    keys_mapping = {
        'module.0.module._embedding.weight': "module.0.module.0.module._embedding.weight",
        'module.1.lstm.weight_ih_l0': "module.0.module.1.lstm.weight_ih_l0",
        'module.1.lstm.weight_hh_l0': "module.0.module.1.lstm.weight_hh_l0",
        'module.1.lstm.bias_ih_l0': "module.0.module.1.lstm.bias_ih_l0",
        'module.1.lstm.bias_hh_l0': "module.0.module.1.lstm.bias_hh_l0",
        'module.1.lstm.weight_ih_l1': "module.0.module.1.lstm.weight_ih_l1",
        'module.1.lstm.weight_hh_l1': "module.0.module.1.lstm.weight_hh_l1",
        'module.1.lstm.bias_ih_l1': "module.0.module.1.lstm.bias_ih_l1",
        'module.1.lstm.bias_hh_l1': "module.0.module.1.lstm.bias_hh_l1",
        'module.1.lstm.weight_ih_l2': "module.0.module.1.lstm.weight_ih_l2",
        'module.1.lstm.weight_hh_l2': "module.0.module.1.lstm.weight_hh_l2",
        'module.1.lstm.bias_ih_l2': "module.0.module.1.lstm.bias_ih_l2",
        'module.1.lstm.bias_hh_l2': "module.0.module.1.lstm.bias_hh_l2",
        'module.2.module.0.module.0.weight': "module.0.module.2.module.0.weight",
        'module.2.module.0.module.0.bias': "module.0.module.2.module.0.bias",
    }

    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[keys_mapping[k]] = v

    return new_ckpt
