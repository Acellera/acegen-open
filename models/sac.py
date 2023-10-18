#!/usr/bin/env python3

from pathlib import Path
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import (
    LSTMModule,
    MLP,
    ActorValueOperator,
    ProbabilisticActor,
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
        out = self._embedding(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        out = out.squeeze()  # This is an ugly hack, should not be necessary
        return out


def create_net(vocabulary_size, net_name="actor"):
    embedding_module = TensorDictModule(
        Embed(vocabulary_size, 256),
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
            out_features=vocabulary_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=["logits"],
    )

    if net_name == "actor":
        mlp = ProbabilisticActor(
            module=mlp,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )
    else:
        mlp = TensorDictModule(
            module=mlp,
            in_keys=["logits"],
            out_keys=["action_value"],
        )

    model_inference = TensorDictSequential(embedding_module, lstm_module, mlp)
    model_training = TensorDictSequential(embedding_module, lstm_module.set_recurrent_mode(True), mlp)
    transform = lstm_module.make_tensordict_primer()

    return model_inference, model_training, transform


def create_sac_models(vocabulary_size):

    actor_inference, actor_training, actor_transform = create_net(vocabulary_size, net_name="actor")
    ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "chembl_actor.prior")
    actor_inference.load_state_dict(ckpt)
    actor_training.load_state_dict(ckpt)

    critic_inference, critic_training, critic_transform = create_net(vocabulary_size, net_name="critic")
    ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "chembl_critic.prior")
    critic_inference.load_state_dict(ckpt)
    critic_training.load_state_dict(ckpt)

    return actor_inference, actor_training, critic_inference, critic_training, actor_transform, critic_transform



