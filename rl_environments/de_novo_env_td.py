import abc

from typing import Dict, List, Optional, Tuple, Type, Union, Callable

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, TensorSpec, DiscreteTensorSpec
from torchrl.envs import EnvBase
from torchrl.data.utils import DEVICE_TYPING


class DeNovoEnv(EnvBase):

    def __init__(
            self,
            start_token: int,
            end_token: int,
            length_vocabulary: int,
            max_length: int = 100,
            device: DEVICE_TYPING = None,
            batch_size: int = 1,
    ):

        super().__init__(
            device=device,
            batch_size=torch.Size([batch_size]),
            run_type_checks=False,  # TODO: review
            dtype=None,  # TODO: review
            allow_done_after_reset=False,  # TODO: review
        )

        self.start_token = start_token
        self.end_token = end_token
        self.length_vocabulary = length_vocabulary
        self.max_length = max_length
        self.num_envs = batch_size

        self._tensordict = TensorDict(
            {
                "done": torch.zeros(self.num_envs, device=self.device),
                "truncated": torch.zeros(self.num_envs, device=self.device),
                "reward": torch.zeros(self.num_envs, device=self.device),
                "observation": torch.ones(self.num_envs, device=self.device) * self.start_token,
            },
            self.batch_size,
        )

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        import ipdb; ipdb.set_trace()
        _reset = tensordict.get("_reset", None)
        if _reset is not None:
            _reset.reshape(self.num_envs)
            self._tensordict[_reset].zero_()
            self._tensordict.get("observations")[_reset] = self.start_token
        return self._tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        import ipdb; ipdb.set_trace()
        actions = tensordict.get("action")
        self._tensordict.update(
            {
                "done": actions == self.end_token,
                "truncated": actions == self.end_token,
                "reward": torch.zeros(self.num_envs, device=self.device),
                "observation": actions,
            },
        )
        return self._tensordict

    def _set_seed(self, seed: Optional[int] = -1):
        torch.manual_seed(seed)


if __name__ == "__main__":



