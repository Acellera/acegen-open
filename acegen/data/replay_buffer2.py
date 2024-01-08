import numpy as np
import torch

from acegen.vocabulary.base import Vocabulary


class SMILESBuffer(object):
    """Class for prioritized experience replay.

    This class remembers the highest scored sequences seen and samples
    from them with probabilities relative to their scores.
    """

    def __init__(self, vocabulary: Vocabulary, max_size: int = 100):
        self.memory = []
        self.max_size = max_size
        self.voc = vocabulary

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples."""
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[: self.max_size]

    def sample_smiles(self, n, decode_smiles=False):
        """Sample a batch size n of experience."""
        if len(self.memory) < n:
            raise IndexError(
                "Size of memory ({}) is less than requested sample ({})".format(
                    len(self), n
                )
            )
        else:
            scores = [x[1].item() + 1e-10 for x in self.memory]
            sample = np.random.choice(
                len(self), size=n, replace=False, p=scores / np.sum(scores)
            )
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        if decode_smiles:
            encoded = [
                torch.tensor(self.voc.encode(smile), dtype=torch.int32)
                for smile in smiles
            ]
            smiles = collate_fn(encoded)
        return smiles, torch.tensor(scores), torch.tensor(prior_likelihood)

    def __len__(self):
        return len(self.memory)


def collate_fn(arr, max_length=100):
    """Function to take a list of encoded sequences and turn them into a batch."""
    collated_arr = torch.ones(len(arr), max_length + 1) * -1
    for i, seq in enumerate(arr):
        collated_arr[i, -seq.size(0) :] = seq
    return collated_arr
