from typing import Optional

import torch
import random
from tensordict.nn import TensorDictModuleBase
from tensordict.tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.envs.utils import exploration_type, ExplorationType

import warnings
from typing import Optional, Union

import numpy as np
import torch

from tensordict.nn import (
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictModuleWrapper,
)
from tensordict.tensordict import TensorDictBase
from tensordict.utils import expand_as_right, expand_right, NestedKey

from torchrl.data.tensor_specs import CompositeSpec, TensorSpec
from torchrl.envs.utils import exploration_type, ExplorationType
from torchrl.modules.tensordict_module.common import _forward_hook_safe_action


class SoftmaxSamplingModule(TensorDictModuleBase):
    """Softmax exploration module.

    This module randomly select the action(s) from a distribution created by applying a softmax transformation to the
    specified tensordict tensors, which are required to have a last dimension size equal to the number of actions.

    Args:
        spec (TensorSpec): the spec used for sampling actions.

    Keyword Args:
        action_key (NestedKey, optional): the key where the action can be found in the input tensordict.
            Default is ``"action"``.
        action_mask_key (NestedKey, optional): the key where the action mask can be found in the input tensordict.
            Default is ``None`` (corresponding to no mask).
    """
    def __init__(
                self,
                spec: Optional[TensorSpec] = None,
                action_key: Optional[NestedKey] = "action",
                action_mask_key: Optional[NestedKey] = None,
        ):
        self.action_key = action_key
        in_keys = [self.action_key]
        self.in_keys = in_keys
        self.out_keys = [self.action_key]
        super().__init__()

        if spec is not None:
            if not isinstance(spec, CompositeSpec) and len(self.out_keys) >= 1:
                spec = CompositeSpec({action_key: spec}, shape=spec.shape[:-1])
        self._spec = spec

    @property
    def spec(self):
        return self._spec
    def forward(self, tensordict: TensorDictBase, temperature = 1.0) -> TensorDictBase:

        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            if isinstance(self.action_key, tuple) and len(self.action_key) > 1:
                action_tensordict = tensordict.get(self.action_key[:-1]) # TODO: is this necessary?
                action_key = self.action_key[-1]
            else:
                action_tensordict = tensordict
                action_key = self.action_key

            # Ensure numeric stability by subtracting the maximum Q-value
            action_values = action_tensordict[action_key]
            max_action_value, _ = torch.max(tensordict[action_key], dim=-1, keepdim=True)
            exp_values = torch.exp((action_values - max_action_value) / temperature)

            # Calculate probabilities using the softmax function
            probabilities = exp_values / torch.sum(exp_values, dim=-1, keepdim=True)

            # Sample an action according to the probabilities
            dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probabilities)
            out = dist.sample()
            tensordict.set(self.action_key, out)
            tensordict.set("action_probs", probabilities)

        return tensordict
