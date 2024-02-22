from __future__ import annotations

import torch
from tensordict import TensorDict


def smiles_to_tensordict(
    smiles: torch.Tensor,
    reward: torch.Tensor = None,
    device: str | torch.device = "cpu",
    mask_value: int = -1,
    replace_mask_value: int = None,
):
    """Create a Tensordict in TorchRL data format from a batch of SMILES.

    Args:
        smiles (torch.Tensor): Batch of SMILES. Shape: (B, T).
        reward (torch.Tensor): Batch of rewards. Shape: (B,) or (B, 1).
        device (str or torch.device): Device to create the Tensordict on.
        mask_value (int): Value to used for padding. Default: -1.
        replace_mask_value (int): Value to replace the mask value with. Default: None.
    """
    B, T = smiles.shape
    mask = smiles != mask_value
    device = torch.device(device) if isinstance(device, str) else device

    if replace_mask_value is not None:
        smiles[~mask] = replace_mask_value
    rewards = torch.zeros(B, T, 1)
    if reward is not None:
        rewards[:, -1] = reward.reshape(-1, 1)
    done = torch.zeros(B, T, 1, dtype=torch.bool)

    lengths = mask.cumsum(dim=1).argmax(dim=1)
    done[torch.arange(B), lengths] = True

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
