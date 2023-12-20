from typing import Optional

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data.utils import DEVICE_TYPING
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
    OneHotDiscreteTensorSpec,

)


class MultiStepSMILESEnv(EnvBase):
    def __init__(
            self,
            start_token: int,
            end_token: int,
            length_vocabulary: int,
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
            start_obs[:, self.start_token] = 1
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
        if tensordict is not None:
            next_tensordict = tensordict
            next_tensordict.update(self._reset_tensordict.clone())
        else:
            next_tensordict = self._reset_tensordict.clone()
        return next_tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Get actions
        actions = tensordict.get("action")
        if self.one_hot_action_encoding:
            actions = torch.argmax(actions, dim=-1)

        # Update episode length
        self.episode_length += 1

        # Create termination flags
        terminated = actions == self.end_token
        truncated = self.episode_length == self.max_length
        done = terminated | truncated
        self.episode_length[done] = 1

        # Create next_tensordict
        obs = actions.clone().long()
        if self.one_hot_obs_encoding:
            obs = torch.nn.functional.one_hot(obs, num_classes=self.length_vocabulary)
        next_tensordict = TensorDict(
            {
                "done": done,
                "truncated": truncated,
                "terminated": terminated,
                "reward": torch.zeros(self.num_envs, device=self.device),
                "observation": obs
            },
            device=self.device,
            batch_size=self.batch_size,
        )
        next_tensordict.update(tensordict.get("next", {}))
        return next_tensordict

    def _set_seed(self, seed: Optional[int] = -1) -> None:
        torch.manual_seed(seed)

    def _set_specs(self) -> None:
        obs_spec = OneHotDiscreteTensorSpec if self.one_hot_obs_encoding else DiscreteTensorSpec
        self.observation_spec = (
            CompositeSpec(
                {
                    "observation": obs_spec(
                        n=self.length_vocabulary,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                }
            )
            .expand(self.num_envs)
        )
        action_spec = OneHotDiscreteTensorSpec if self.one_hot_action_encoding else DiscreteTensorSpec
        self.action_spec = (
            CompositeSpec(
                {
                    "action": action_spec(
                        n=self.length_vocabulary,
                        dtype=torch.int32,
                        device=self.device,
                    )
                }
            )
            .expand(self.num_envs)
        )
        self.reward_spec = (
            CompositeSpec({"reward": UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
                device=self.device,
            )})
            .expand(self.num_envs)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}, start_token={self.start_token}," \
               f" end_token={self.end_token} batch_size={self.batch_size})"
