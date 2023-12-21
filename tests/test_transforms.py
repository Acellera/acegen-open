import pytest
import torch
from acegen.transforms import SMILESReward
from acegen.vocabulary import SMILESVocabulary
from tensordict import TensorDict


def dummy_reward_function(smiles):
    return [1 for _ in smiles]


def generate_valid_data_batch(
    vocabulary_size: int, batch_size: int, sequence_length: int
):
    tokens = torch.randint(0, vocabulary_size, (batch_size, sequence_length + 1, 1))
    smiles = torch.randint(0, vocabulary_size, (batch_size, sequence_length + 1, 1))
    done = torch.randint(0, 2, (batch_size, sequence_length + 1, 1))
    batch = TensorDict(
        {
            "observation": tokens[:, :-1],
            "SMILES" "done": torch.zeros(batch_size, sequence_length, 1),
            "is_init": done[:, 1:],
            "next": TensorDict(
                {
                    "observation": tokens[:, 1:],
                    "done": done[:, 0:-1],
                },
                batch_size=[batch_size, sequence_length],
            ),
        },
        batch_size=[batch_size, sequence_length],
    )
    return batch


def test_reward_transform():
    pass
