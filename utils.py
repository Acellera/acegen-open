import json
import pandas as pd
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
        out = out.squeeze(-1)  # If time dimension is 1, remove it. Ugly hack, should not be necessary
        out = out.squeeze(-2)  # If time dimension is 1, remove it. Ugly hack, should not be necessary
        return out


def create_ppo_models(vocabulary, output_size, out_key="logits"):
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
        out_keys=[
            "features",
            ("next", "recurrent_state_h"),
            ("next", "recurrent_state_c"),
        ],
    )
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
        common_operator=TensorDictSequential(
            embedding_module, lstm_module.set_recurrent_mode(True)
        ),
        policy_operator=policy_module,
        value_operator=critic_module,
    )

    actor_inference = actor_critic_inference.get_policy_operator()
    critic_inference = actor_critic_inference.get_value_operator()
    actor_training = actor_critic_training.get_policy_operator()
    critic_training = actor_critic_training.get_value_operator()
    transform = lstm_module.make_tensordict_primer()

    ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "actor_critic.prior")
    actor_inference.load_state_dict(ckpt)
    actor_training.load_state_dict(ckpt)

    return actor_inference, actor_training, critic_inference, critic_training, transform


def penalise_repeated_smiles(
    data,
    diversity_buffer,
    repeated_smiles,
    in_keys="reward",
    out_keys="reward",
    penalty=0.0,
):
    """Penalise repeated smiles and add unique smiles to the diversity buffer."""

    td_next = data.get("next")
    done = td_next.get("done").squeeze(-1)  # Get done flags
    terminated = td_next.get("terminated").squeeze(-1)  # Get terminated flags
    assert (done == terminated).all(), "done and terminated flags should be equal"
    sub_td = td_next.get_sub_tensordict(
        idx=terminated
    )  # Get sub-tensordict of done trajectories
    reward = sub_td.get(in_keys)
    finished_smiles = sub_td.get("SMILES")
    finished_smiles_td = sub_td.select("SMILES")
    num_unique_smiles = len(diversity_buffer)
    num_finished_smiles = len(finished_smiles_td)

    if num_finished_smiles > 0 and num_unique_smiles == 0:
        diversity_buffer.extend(finished_smiles_td)

    elif num_finished_smiles > 0:
        for i, smi in enumerate(finished_smiles):
            td_smiles = diversity_buffer._storage._storage
            unique_smiles = td_smiles.get("_data").get("SMILES")[0:num_unique_smiles]
            repeated = (smi == unique_smiles).all(dim=-1).any()
            if repeated:
                reward[i] = reward[i] * penalty
                repeated_smiles += 1
            elif reward[i] > 0:
                diversity_buffer.extend(finished_smiles_td[i : i + 1])
                num_unique_smiles += 1

    sub_td.set(out_keys, reward, inplace=True)

    return repeated_smiles


def create_batch_from_replay_smiles(replay_data, device, reward_key="reward"):
    """Create a TensorDict data batch from replay data."""

    td_list = []

    for smiles in replay_data:
        observation = smiles["SMILES2"][smiles["SMILES2"] != 0].reshape(1, -1)
        tensor_shape = (1, observation.shape[-1], 1)
        reward = torch.zeros(tensor_shape, device=device)
        reward[0, -1] = smiles[reward_key]
        done = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        terminated = done.clone()
        sample_log_prob = reward.clone().reshape(1, -1)
        action = smiles["SMILES"][smiles["SMILES"] != 1].reshape(1, -1)
        is_init = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        is_init[0, 0] = True

        next_observation = action.clone()
        next_done = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        next_done[0, -1] = True
        next_terminated = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        next_terminated[0, -1] = True
        next_is_init = torch.zeros(tensor_shape, device=device, dtype=torch.bool)

        td_list.append(
            TensorDict(
                {
                    "done": done,
                    "action": action,
                    "is_init": is_init,
                    "terminated": terminated,
                    "observation": observation,
                    "sample_log_prob": sample_log_prob,
                    "next": TensorDict(
                        {
                            "observation": next_observation,
                            "terminated": next_terminated,
                            reward_key: reward,
                            "is_init": next_is_init,
                            "done": next_done,
                        },
                        batch_size=tensor_shape[0:2],
                        device=device,
                    ),
                },
                batch_size=tensor_shape[0:2],
                device=device,
            )
        )

    cat_data = torch.cat(td_list, dim=-1)
    split_data = split_trajectories(cat_data)

    return cat_data, split_data

