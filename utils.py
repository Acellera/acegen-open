import torch
import random
import warnings
from rdkit import Chem
from tensordict import TensorDict


def create_batch_from_replay_smiles(replay_data, device, vocabulary, reward_key="reward"):
    """Create a TensorDict data batch from replay data."""

    td_list = []

    for data in replay_data:

        smiles = data["SMILES"][data["SMILES"] != -1]

        # RANDOMISE SMILES #########

        if random.random() > 1.0:
            try:
                decoded = vocabulary.decode(smiles.cpu().numpy(), ignore_indices=[-1])
                decoded = Chem.MolToSmiles(Chem.MolFromSmiles(decoded), doRandom=True, canonical=False)
                encoded = vocabulary.encode(vocabulary.tokenize(decoded))
                smiles = torch.tensor(encoded, dtype=torch.int32, device=device)
            except Exception:  # Sometimes a token outside the vocabulary can appear, in this case we just skip
                warnings.warn(f"Skipping {data['SMILES']}")

        ############################

        observation = smiles[:-1].reshape(1, -1, 1).clone()
        action = smiles[1:].reshape(1, -1, 1).clone()
        tensor_shape = (1, observation.shape[1], 1)
        reward = torch.zeros(tensor_shape, device=device)
        reward[0, -1] = data[reward_key]
        done = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        terminated = done.clone()
        sample_log_prob = torch.zeros(tensor_shape, device=device).reshape(1, -1)
        is_init = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        is_init[0, 0] = True

        next_observation = action.clone()
        next_done = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        next_done[0, -1] = True
        next_terminated = next_done.clone()
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
    return cat_data


def create_batch_from_replay_smiles2(replay_smiles, replay_rewards, device, vocabulary):
    """Create a TensorDict data batch from replay data."""

    td_list = []

    for smiles, rew in zip(replay_smiles, replay_rewards):

        encoded = vocabulary.encode(vocabulary.tokenize(smiles))
        smiles = torch.tensor(encoded, dtype=torch.int32, device=device)

        # RANDOMISE SMILES #########

        # if random.random() > 0.5:
        #     try:
        #         decoded = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True, canonical=False)
        #         encoded = vocabulary.encode(vocabulary.tokenize(decoded))
        #         smiles = torch.tensor(encoded, dtype=torch.int32, device=device)
        #     except Exception:  # Sometimes a token outside the vocabulary can appear, in this case we just skip
        #         warnings.warn(f"Skipping {smiles}")
        #         encoded = vocabulary.encode(vocabulary.tokenize(smiles))
        #         smiles = torch.tensor(encoded, dtype=torch.int32, device=device)

        ############################

        observation = smiles[:-1].reshape(1, -1, 1).clone()
        action = smiles[1:].reshape(1, -1).clone()
        tensor_shape = (1, observation.shape[1], 1)
        reward = torch.zeros(tensor_shape, device=device)
        reward[0, -1] = rew
        done = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        terminated = done.clone()
        sample_log_prob = torch.zeros(tensor_shape, device=device).reshape(1, -1)
        is_init = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        is_init[0, 0] = True

        next_observation = smiles[1:].reshape(1, -1, 1).clone()
        next_done = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        next_done[0, -1] = True
        next_terminated = next_done.clone()
        next_is_init = torch.zeros(tensor_shape, device=device, dtype=torch.bool)

        recurrent_states = torch.zeros(*tensor_shape[0:2], 3, 512)
        next_recurrent_states = torch.zeros(*tensor_shape[0:2], 3, 512)

        td_list.append(
            TensorDict(
                {
                    "done": done,
                    "action": action,
                    "is_init": is_init,
                    "terminated": terminated,
                    "observation": observation,
                    "sample_log_prob": sample_log_prob,
                    "recurrent_state": recurrent_states,
                    "next": TensorDict(
                        {
                            "observation": next_observation,
                            "terminated": next_terminated,
                            "reward": reward,
                            "is_init": next_is_init,
                            "done": next_done,
                            "recurrent_state": next_recurrent_states,
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
    return cat_data
