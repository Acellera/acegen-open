import random

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
        action_key (NestedKey, optional): the key where the action can be found in the input tensordict.
            Default is ``"action"``.
    """

    def __init__(
        self,
        action_spec: Optional[TensorSpec] = None,
        logits_key: Optional[NestedKey] = "action_value",
        action_key: Optional[NestedKey] = "action",
    ):
        self.action_key = action_key
        self.logits_key = logits_key
        self.in_keys = [self.logits_key]
        self.out_keys = [self.action_key]
        super().__init__()

        if action_spec is not None:
            if not isinstance(action_spec, CompositeSpec) and len(self.out_keys) >= 1:
                action_spec = CompositeSpec(
                    {action_key: action_spec}, shape=action_spec.shape[:-1]
                )
        self._spec = action_spec

    @property
    def spec(self):
        return self._spec

    def forward(self, tensordict: TensorDictBase, temperature=1.0) -> TensorDictBase:
        """Computes the softmax distribution and samples an action from it."""

        import ipdb

        ipdb.set_trace()
        # TODO: if action spec available, check that logits dimension is equal to the number of actions

        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:

            # Ensure numeric stability by subtracting the maximum Q-value
            logits = tensordict.get(self.logits_key)
            max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
            exp_values = torch.exp((logits - max_logits) / temperature)

            # Calculate probabilities using the softmax function
            probabilities = exp_values / torch.sum(exp_values, dim=-1, keepdim=True)

            # Sample an action according to the probabilities
            dist = torch.distributions.one_hot_categorical.OneHotCategorical(
                probs=probabilities
            )
            out = dist.sample()
            tensordict.set(self.action_key, out)
            tensordict.set("action_probs", probabilities)

        return tensordict
