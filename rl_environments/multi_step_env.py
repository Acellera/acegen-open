from typing import Optional

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data.utils import DEVICE_TYPING
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)


class MultiStepDeNovoEnv(EnvBase):
    def __init__(
            self,
            start_token: int,
            end_token: int,
            length_vocabulary: int,
            max_length: int = 140,
            device: DEVICE_TYPING = None,
            batch_size: int = 1,
    ):
        super().__init__(
            device=device,
            batch_size=torch.Size([batch_size]),
        )

        self.num_envs = batch_size
        self.max_length = max_length
        self.end_token = int(end_token)
        self.start_token = int(start_token)
        self.length_vocabulary = length_vocabulary
        self.episode_length = torch.ones(self.num_envs, device=self.device, dtype=torch.int32)

        self._reset_tensordict = TensorDict(
            {
                "observation": torch.ones(self.num_envs, device=self.device, dtype=torch.int32)
                * self.start_token,
            },
            device=self.device,
            batch_size=self.batch_size,
        )

        self._set_specs()

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            next_tensordict = tensordict
            next_tensordict.update(self._reset_tensordict.clone())
        else:
            next_tensordict = self._reset_tensordict.clone()
        return next_tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        actions = tensordict.get("action")
        if actions.shape[-1] == self.length_vocabulary:  # One-hot encoding
            actions = torch.argmax(actions, dim=-1)
        self.episode_length += 1
        done = (actions == self.end_token) | (self.episode_length == self.max_length)
        self.episode_length[done] = 1
        next_tensordict = TensorDict(
            {
                "done": done,
                "terminated": done.clone(),
                "reward": torch.zeros(self.num_envs, device=self.device),
                "observation": tensordict.get("action").to(torch.int32),
            },
            device=self.device,
            batch_size=self.batch_size,
        )
        next_tensordict.update(tensordict.get("next", {}))
        return next_tensordict

    def _set_seed(self, seed: Optional[int] = -1) -> None:
        torch.manual_seed(seed)

    def _set_specs(self) -> None:
        self.observation_spec = (
            CompositeSpec(
                {
                    "observation": DiscreteTensorSpec(self.length_vocabulary),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.action_spec = (
            CompositeSpec(
                {
                    "action": DiscreteTensorSpec(self.length_vocabulary),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.reward_spec = (
            CompositeSpec({"reward": UnboundedContinuousTensorSpec((1,))})
            .expand(self.num_envs)
            .to(self.device)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}, start_token={self.start_token}," \
               f" end_token={self.end_token} batch_size={self.batch_size})"
