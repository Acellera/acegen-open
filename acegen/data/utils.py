from __future__ import annotations

from typing import Sequence

import torch
from tensordict import TensorDict

from acegen.vocabulary.base import Vocabulary


def remove_duplicated_keys(tensordict: TensorDict, key: str) -> TensorDict:
    """Removes duplicate rows from a PyTorch tensor.

    Args:
    - tensordict (TensorDict): Input tensordict.
    - key (str): Key of the tensor to remove duplicate rows from.

    Returns:
    - TensorDict: Output tensordict with duplicate rows removed.
    """
    tensor = tensordict.get(key)

    _, unique_indices = torch.unique(tensor, dim=0, sorted=True, return_inverse=True)

    # Sort the unique indices
    unique_indices = torch.unique(unique_indices, dim=0)

    # Use torch.sort to ensure the output tensor maintains the order of rows in the input tensor
    unique_tensordict = tensordict[unique_indices]

    return unique_tensordict


def remove_keys_in_reference_tensordict(reference_tensordict, target_tensordict, key):
    """Removes rows from the target tensor that are present in the reference tensor.

    Args:
    - tensordict (TensorDict): Reference TensorDict of shape (N, M).
    - target_tensordict (TensorDict): Target TensorDict of shape (L, M).
    - key (str): Key of the tensor to remove rows from.

    Returns:
    - TensorDict: Filtered target TensorDict containing rows not present in the reference tensor.
    """
    reference_tensor = reference_tensordict.get(key)
    target_tensor = target_tensordict.get(key)
    N = reference_tensor.shape[0]

    cat_data = torch.cat([reference_tensor, target_tensor], dim=0)
    _, unique_indices = torch.unique(cat_data, dim=0, sorted=True, return_inverse=True)

    common_indices = torch.isin(unique_indices[N:], unique_indices[:N])
    filtered_target_tensordict = target_tensordict[~common_indices]

    return filtered_target_tensordict


def smiles_to_tensordict(
    smiles: Sequence[str] | torch.Tensor,
    reward: torch.Tensor,
    vocab: Vocabulary = None,
    device: str | torch.device = "cpu",
):
    """Create an episode Tensordict from a batch of SMILES."""
    if isinstance(smiles, Sequence):
        if vocab is None:
            raise ValueError(
                "If input is a list of SMILES strings, a vocabulary must be provided."
            )
        encoded_smiles = [vocab.encode(s) for s in smiles]
        smiles = torch.ones(len(encoded_smiles), 100, dtype=torch.int32) * -1
        for i, es in enumerate(encoded_smiles):
            smiles[i, -len(es) :] = es

    B, T = smiles.shape
    mask = smiles != -1
    rewards = torch.zeros(B, T, 1)
    rewards[:, -1] = reward
    done = torch.zeros(B, T, 1, dtype=torch.bool)
    done[:, -1] = True

    smiles_tensordict = TensorDict(
        {
            "observation": smiles[:, :-1].int(),
            "action": smiles[:, 1:],
            "done": done[:, :-1],
            "terminated": done[:, :-1],
            "mask": mask[:, :-1],
            "next": TensorDict(
                {
                    "observation": smiles[:, 1:].int(),
                    "reward": rewards[:, 1:],
                    "done": done[:, 1:],
                    "terminated": done[:, 1:],
                },
                batch_size=[B, T - 1],
            ),
        },
        batch_size=[B, T - 1],
    )

    smiles_tensordict = smiles_tensordict.to(device)

    return smiles_tensordict
