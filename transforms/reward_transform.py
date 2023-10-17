import torch
from typing import Callable
from tensordict import TensorDictBase
from torchrl.envs.transforms.transforms import Transform
from vocabulary.vocabulary import DeNovoVocabulary


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
        self.reward_function = reward_function

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        with torch.set_grad_enabled(self.gradient_mode):
            tensordict_save = tensordict
            if self.on_done_only:
                td_next = tensordict.get("next")
                done = td_next.get("done")
                truncated = td_next.get(self.truncated_key, None)
                if truncated is not None:
                    done = done | truncated
                done = done.squeeze(-1)
                if done.shape != td_next.shape:
                    raise ValueError(
                        "the done state shape must match the tensordict shape."
                    )
                sub_tensordict = td_next.get_sub_tensordict(done)
                reward = sub_tensordict.get("reward")
                smiles = sub_tensordict.get(self.in_keys[0])
                smiles_list = []
                for i, smi in enumerate(smiles):
                    smiles_list.append(self.vocabulary.decode_smiles(smi.cpu().numpy()))
                reward[:, 0].copy_(torch.from_numpy(self.reward_function(smiles_list)))
                sub_tensordict.set("reward", reward, inplace=True)
        tensordict_save.update(tensordict, inplace=True)
        return tensordict_save
