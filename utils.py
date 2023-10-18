import torch
from tensordict import TensorDict
from torchrl.collectors.utils import split_trajectories


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
