from __future__ import annotations

from typing import Sequence

import torch
from tensordict import TensorDict

from acegen.vocabulary.base import Vocabulary


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
