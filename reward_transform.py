import rdkit
import torch
from typing import Callable, Any
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import unravel_keys
from torchrl.data.tensor_specs import (
    CompositeSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.transforms.transforms import Transform
from vocabulary import DeNovoVocabulary


def test_scoring(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    return 2.0 if mol else 0.0,


class SMILESReward(Transform):
    def __init__(
            self,
            reward_function: Callable,
            vocabulary: DeNovoVocabulary,
            in_keys=None,
            out_keys=None,
            on_done_only=True,
            truncated_key="truncated",
            use_next: bool = True,
            gradient_mode=False,
    ):
        self.on_done_only = on_done_only
        self.truncated_key = truncated_key
        self.use_next = use_next
        self.gradient_mode = gradient_mode

        if not isinstance(reward_function, Callable):
            raise ValueError("reward_function must be a callable.")

        if out_keys is None:
            out_keys = ["reward"]
        if in_keys is None:
            in_keys = ["SMILES"]

        super().__init__(in_keys, out_keys)

        self.vocabulary = vocabulary
        # self.reward_function = reward_function
        self.reward_function = test_scoring

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        with torch.set_grad_enabled(self.gradient_mode):
            tensordict_save = tensordict
            if self.on_done_only:
                done = tensordict.get("done")
                truncated = tensordict.get(self.truncated_key, None)
                if truncated is not None:
                    done = done | truncated
                done = done.squeeze(-1)
                if done.shape != tensordict.shape:
                    raise ValueError(
                        "the done state shape must match the tensordict shape."
                    )
                sub_tensordict = tensordict.get_sub_tensordict(done)
                reward = sub_tensordict.get("reward")
                smiles = sub_tensordict.get(self.in_keys[0])
                for i, smi in enumerate(smiles):
                    reward[i] *= self.reward_function(self.vocabulary.decode_smiles(smi.cpu().numpy()))
                sub_tensordict.set("reward", reward, inplace=True)
        tensordict_save.update(tensordict, inplace=True)
        return tensordict_save

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        parent = self.parent
        reward_key = parent.reward_key
        reward_spec = UnboundedContinuousTensorSpec(shape=(*parent.batch_size, 1))
        if unravel_keys(reward_key, make_tuple=True) != unravel_keys(
                self.out_keys[0], make_tuple=True
        ):
            # we must change the reward key of the parent
            reward_key = self.out_keys[0]
        reward_spec = CompositeSpec({reward_key: reward_spec}, shape=parent.batch_size)
        return reward_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        parent = self.parent
        reward_key = parent.reward_key
        if unravel_keys(reward_key, make_tuple=True) != unravel_keys(
                self.out_keys[0], make_tuple=True
        ):
            # we should move the parent reward spec to the obs
            reward_spec = parent.reward_spec.clone()
            observation_spec[reward_key] = reward_spec
            raise Exception(f"{reward_key}, {self.out_keys[0]}")
        return observation_spec

    def _call_at_reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.use_next:
            self._inv(tensordict.get("next"))
        else:
            self._inv(tensordict)
        return tensordict
