from typing import Optional

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
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
            batch_size=[batch_size],  # TODO: PR to change this
            run_type_checks=False,  # TODO: review
            dtype=None,  # TODO: review
            allow_done_after_reset=False,  # TODO: review
        )

        self.start_token = int(start_token)
        self.end_token = int(end_token)
        self.length_vocabulary = length_vocabulary
        self.max_length = max_length
        self.num_envs = batch_size

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
            next_tensordict = tensordict.clone()
            next_tensordict.update(self._reset_tensordict.clone())
        else:
            next_tensordict = self._reset_tensordict.clone()
        return next_tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict = tensordict.get("next").clone()
        actions = tensordict.get("action")
        next_tensordict.update(
            {
                "done": (actions == self.end_token).to(torch.bool),
                "terminated": (actions == self.end_token).to(torch.bool),
                "reward": torch.zeros(self.num_envs, device=self.device),
                "observation": actions.to(torch.int32),
            },
        )
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
