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

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:

        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            action_tensordict = tensordict
            action_key = self.action_key
            dist = torch.distributions.one_hot_categorical.OneHotCategorical(logits=tensordict["action_value"])
            out = dist.sample()
            action_tensordict.set(action_key, out)
        return tensordict
