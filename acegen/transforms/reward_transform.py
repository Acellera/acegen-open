from __future__ import annotations

import warnings
from typing import Callable, Sequence

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.envs.transforms.transforms import Transform

from acegen.vocabulary.base import Vocabulary


class SMILESReward(Transform):
    """Transform to add a reward to a SMILES.

    This class requires either a reward_function or a reward_function_creator. If both are provided,
    the reward_function will be used. If neither are provided, a ValueError will be raised.

    Args:
        vocabulary (Vocabulary): A vocabulary object with at least encode and decode methods.
        reward_function (callable, optional): A callable that takes a list of SMILES and returns
        a list of rewards.
        reward_function_creator (callable, optional): A callable that creates a reward function.
        in_keys (sequence of NestedKey, optional): keys to be updated.
            default: ["observation", "reward"]
        out_keys (sequence of NestedKey, optional): destination keys.
            Defaults to ``in_keys``.
        reward_scale (int, optional): The scale to apply to the reward.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        reward_function: Callable = None,
        reward_function_creator: Callable = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        reward_scale=1.0,
    ):

        if reward_function is None and reward_function_creator is None:
            raise ValueError(
                "Either reward_function or reward_function_creator must be provided."
            )

        if not reward_function:
            if not isinstance(reward_function_creator, Callable):
                raise ValueError(
                    "A reward_function_creator was provided but it must be a callable"
                    "that returns a reward function, not a {}".format(
                        type(reward_function_creator)
                    )
                )
            reward_function = reward_function_creator()

        if not isinstance(reward_function, Callable):
            raise ValueError(
                "A reward_function was provided but it must be a callable, not a {}".format(
                    type(reward_function)
                )
            )

        if out_keys is None:
            out_keys = ["reward"]
        if in_keys is None:
            in_keys = ["SMILES"]
        self.reward_scale = reward_scale

        super().__init__(in_keys, out_keys)

        self.vocabulary = vocabulary
        self.reward_function = reward_function

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._call(tensordict.get("next"))
        return tensordict

    def _call(self, tensordict: TensorDictBase, _reset=None) -> TensorDictBase:

        # Get steps where trajectories end
        device = tensordict.device
        done = tensordict.get("done").squeeze(-1)

        if done.sum() == 0:
            return tensordict

        # Get reward and smiles
        reward = tensordict.get(self.out_keys[0])
        smiles = tensordict.get(self.in_keys[0])

        # Get smiles as strings
        smiles_list = []
        for smi in smiles:
            smiles_list.append(
                self.vocabulary.decode(smi.cpu().numpy(), ignore_indices=[-1])
            )

        # Calculate reward - try multiple times in case of RuntimeError
        max_attempts = 3
        for i in range(max_attempts):
            try:
                _reward = torch.tensor(self.reward_function(smiles_list), device=device)
                reward += _reward.reshape(reward.shape)
                break
            except RuntimeError:
                if i == max_attempts - 1:
                    raise
                else:
                    warnings.warn(
                        "RuntimeError in reward function. Trying again. Attempt {}/{}".format(
                            i + 1, max_attempts
                        )
                    )
                    continue

        tensordict[self.out_keys[0]][done] = _reward

        return tensordict
