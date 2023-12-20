import torch
from typing import Callable
from tensordict import TensorDictBase
from torchrl.envs.transforms.transforms import Transform
from acegen.vocabulary.vocabulary_old import DeNovoVocabulary


class SMILESReward(Transform):
    def __init__(
        self,
        reward_function: Callable,
        vocabulary: DeNovoVocabulary,
        in_keys=None,
        out_keys=None,
        reward_scale=1.0,
    ):
        if not isinstance(reward_function, Callable):
            raise ValueError("reward_function must be a callable.")

        if out_keys is None:
            out_keys = ["reward"]
        if in_keys is None:
            in_keys = ["SMILES"]
        self.reward_scale = reward_scale

        super().__init__(in_keys, out_keys)

        self.vocabulary = vocabulary
        self.reward_function = reward_function

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:

        # Get steps where trajectories end
        device = tensordict.device
        td_next = tensordict.get("next")
        done = td_next.get("done").squeeze(-1)
        sub_tensordict = td_next.get_sub_tensordict(done)

        if len(sub_tensordict) == 0:
            return tensordict

        # Get reward and smiles
        reward = sub_tensordict.get("reward")
        smiles = sub_tensordict.get(self.in_keys[0])

        # Get smiles as strings
        smiles_list = []
        for i, smi in enumerate(smiles):
            smiles_list.append(self.vocabulary.decode(smi.cpu().numpy(), ignore_indices=[-1]))

        # Calculate reward
        try:
            reward[:, 0] += torch.tensor(self.reward_function(smiles_list), device=device)
        except RuntimeError:
            reward[:, 0] += torch.tensor(self.reward_function(smiles_list), device=device)

        sub_tensordict.set("real_reward", reward, inplace=True)
        sub_tensordict.set("reward", reward / sub_tensordict.get("step_count").squeeze(), inplace=True)

        return tensordict
