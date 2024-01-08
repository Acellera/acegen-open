from typing import Sequence

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import Transform

from acegen.vocabulary.base import Vocabulary


class SMILESBuffer:
    """Class for prioritized experience replay."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        max_size: int = 100,
        smiles_key: str = "SMILES",
        score_key: str = "reward",
        mask_key: str = "mask",
        transforms: Sequence[Transform] = None,
    ):
        self.memory = []
        self.max_size = max_size
        self.voc = vocabulary
        self.smiles_key = smiles_key
        self.score_key = score_key
        self.mask_key = mask_key
        self.transforms = transforms  # TODO: check that are transforms!

    def add_experience(self, tensordict: TensorDict) -> None:

        # check mask key is in tensordict
        if self.mask_key not in tensordict.keys():
            raise KeyError(
                f"Mask key {self.mask_key} not found in tensordict. Please check that the mask key is correct."
            )

        # check smiles key is in tensordict
        if self.smiles_key not in tensordict.keys():
            raise KeyError(
                f"SMILES key {self.smiles_key} not found in tensordict. Please check that the smiles key is correct."
            )

        # check score key is in tensordict
        if self.score_key not in tensordict.keys():
            raise KeyError(
                f"Score key {self.score_key} not found in tensordict. Please check that the score key is correct."
            )

        if tensordict.batch_size > torch.Size([]):
            smiles = list(tensordict.unbind(0))
        else:
            smiles = [tensordict]

        for s in smiles:
            array = (
                tensordict.get(self.smiles_key)[tensordict.get(self.mask_key)]
                .cpu()
                .numpy()
            )
            smiles_str = self.voc.decode(array)
            s.smiles_str = smiles_str

        self.memory.extend(smiles)

        # Remove duplicates
        seen = set()
        if len(self.memory) > self.max_size:
            idxs, data = [], []
            for i, smiles in enumerate(self.memory):
                if smiles.smiles_str not in seen:
                    seen.add(smiles.smiles_str)
                    idxs.append(i)
                    data.append(smiles)
            self.memory = [self.memory[idx] for idx in idxs]
            self.memory.sort(key=lambda x: x.get(self.score_key).sum(), reverse=True)
            self.memory = self.memory[: self.max_size]

    def sample_smiles(self, n):

        if len(self.memory) < n:
            raise IndexError(
                "Size of memory ({}) is less than requested sample ({})".format(
                    len(self), n
                )
            )
        else:
            scores = [x.get(self.score_key).sum().item() + 1e-10 for x in self.memory]
            sample = np.random.choice(
                len(self), size=n, replace=False, p=scores / np.sum(scores)
            )
            sample = [self.memory[i] for i in sample]
            import ipdb

            ipdb.set_trace()
            # TODO: apply transforms
            sample = [
                x[x.get(self.mask_key) == 1] for x in sample
            ]  # merge with previous loop
            smiles = torch.cat(sample, dim=-1)
        return smiles

    def __len__(self):
        return len(self.memory)
