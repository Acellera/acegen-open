from typing import Optional

import torch
import random
from tensordict.nn import TensorDictModuleBase
from tensordict.tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.envs.utils import exploration_type, ExplorationType


class CategoricalSamplingModule(TensorDictModuleBase):

    def __init__(
            self,
            action_key: Optional[NestedKey] = "action",
    ):
        self.action_key = action_key
        in_keys = [self.action_key]
        self.in_keys = in_keys
        self.out_keys = [self.action_key]
        super().__init__()

    def step(self, frames: int = 1) -> None:
        pass

    def forward(self, tensordict: TensorDictBase, temperature = 1.0) -> TensorDictBase:

        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:

            # Ensure numeric stability by subtracting the maximum Q-value
            action_values = tensordict["action_value"]
            max_action_value, _ = torch.max(tensordict["action_value"], dim=-1, keepdim=True)
            exp_values = torch.exp((action_values - max_action_value) / temperature)

            # Calculate probabilities using the softmax function
            probabilities = exp_values / torch.sum(exp_values, dim=-1, keepdim=True)

            # Sample an action according to the probabilities
            dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probabilities)
            out = dist.sample()
            tensordict.set(self.action_key, out)
            tensordict.set("action_probs", probabilities)

        return tensordict
