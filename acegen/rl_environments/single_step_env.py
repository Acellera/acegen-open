from typing import Optional

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data.utils import DEVICE_TYPING
from torchrl.data import (
    CompositeSpec,
    MultiDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)


class SingleStepDeNovoEnv(EnvBase):
    def __init__(
            self,
            start_token: int,
            end_token: int,
            length_vocabulary: int,
            vocabulary,
            reward_function,
            max_length: int = 140,
            device: DEVICE_TYPING = None,
            batch_size: int = 1,
            one_hot_action_encoding: bool = False,
            one_hot_obs_encoding: bool = False,
    ):
        super().__init__(
            device=device,
            batch_size=torch.Size([batch_size]),
        )

        self.vocabulary = vocabulary
        self.reward_function = reward_function

        self.num_envs = batch_size
        self.max_length = max_length
        self.end_token = int(end_token)
        self.start_token = int(start_token)
        self.length_vocabulary = length_vocabulary
        self.one_hot_obs_encoding = one_hot_obs_encoding
        self.one_hot_action_encoding = one_hot_action_encoding
        self.episode_length = torch.ones(self.num_envs, device=self.device, dtype=torch.int32)

        if self.one_hot_obs_encoding:
            start_obs = torch.zeros(batch_size, length_vocabulary, device=self.device, dtype=torch.int32)
            start_obs[:, self.start_token - 1] = 1
        else:
            start_obs = torch.ones(self.num_envs, device=self.device, dtype=torch.int32) * self.start_token

        self._reset_tensordict = TensorDict(
            {
                "observation": start_obs
            },
            device=self.device,
            batch_size=self.batch_size,
        )

        self._set_specs()

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        next_tensordict = self._reset_tensordict.clone()
        return next_tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        actions = tensordict.get("action")
        if self.one_hot_action_encoding:
            actions = torch.argmax(actions, dim=-1)
        reward = torch.zeros(self.num_envs, device=self.device)
        done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        next_tensordict = TensorDict(
            {
                "done": done,
                "terminated": done.clone(),
                "reward": reward,
            },
            device=self.device,
            batch_size=self.batch_size,
        )
        return next_tensordict

    def _set_seed(self, seed: Optional[int] = -1) -> None:
        torch.manual_seed(seed)

    def _set_specs(self) -> None:
        self.observation_spec = (
            CompositeSpec(
                {
                    "observation": MultiDiscreteTensorSpec(
                        nvec=torch.ones(self.num_envs) * self.length_vocabulary,
                        shape=torch.Size([self.num_envs]),
                        device=self.device,
                    ),
                },
                shape=torch.Size([self.num_envs]),
            )
        )
        self.action_spec = (
            CompositeSpec(
                {
                    "action": MultiDiscreteTensorSpec(
                        nvec=self.max_length * [self.length_vocabulary],
                        shape=torch.Size([self.num_envs, self.max_length]),
                        device=self.device,
                    ),
                },
                shape=torch.Size([self.num_envs, self.max_length]),
            )
        )
        self.reward_spec = (
            CompositeSpec({"reward": UnboundedContinuousTensorSpec((1,))})
            .expand(self.num_envs)
            .to(self.device)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}, start_token={self.start_token}," \
               f" end_token={self.end_token} batch_size={self.batch_size})"
