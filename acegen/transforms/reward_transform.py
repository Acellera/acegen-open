import warnings
from typing import Callable

import torch
from tensordict import TensorDictBase
from torchrl.envs.transforms.transforms import Transform

from acegen.vocabulary.base import Vocabulary


class SMILESReward(Transform):
    """Transform to add a reward to a SMILES.

    This class requires either a reward_function or a reward_function_creator. If both are provided,
    the reward_function will be used. If neither are provided, a ValueError will be raised.

    Args:
        vocabulary: A vocabulary object with at least encode and decode methods.
        reward_function: A callable that takes a list of SMILES and returns a list of rewards.
        reward_function_creator: A callable that creates a reward function.
        in_keys: The key in the tensordict that contains the encoded SMILES.
        out_keys: The key in the tensordict to store the reward.
        reward_scale: The scale to apply to the reward.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        reward_function: Callable = None,
        reward_function_creator: Callable = None,
        in_keys: tuple = None,
        out_keys: tuple = None,
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
                    "that returns a reward function."
                )
            reward_function = reward_function_creator()

        if not isinstance(reward_function, Callable):
            raise ValueError(
                "A reward_function was provided but it must be a callable."
            )

        if out_keys is None:
            out_keys = [("next", "reward")]
        if in_keys is None:
            in_keys = [("next", "SMILES")]
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

        sub_tensordict = tensordict.get_sub_tensordict(done)

        if len(sub_tensordict) == 0:
            return tensordict

        # Get reward and smiles
        reward = sub_tensordict.get(self.out_keys[0])
        smiles = sub_tensordict.get(self.in_keys[0])

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
                import ipdb

                ipdb.set_trace()
                reward += torch.tensor(self.reward_function(smiles_list), device=device)
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

        sub_tensordict.set(self.out_keys[0], reward * self.reward_scale, inplace=True)

        return tensordict
