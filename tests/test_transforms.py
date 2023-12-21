import pytest
import torch
from acegen.transforms import SMILESReward
from acegen.vocabulary import SMILESVocabulary
from tensordict import TensorDict


tokens = ["(", ")", "1", "=", "C", "N", "O"]


def dummy_reward_function(smiles):
    return [1 for _ in smiles]


def generate_valid_data_batch(
    vocabulary_size: int,
    batch_size: int = 2,
    sequence_length: int = 5,
    max_smiles_length: int = 10,
    smiles_key: tuple = ("next", "SMILES"),
    reward_key: tuple = ("next", "reward"),
):
    tokens = torch.randint(0, vocabulary_size, (batch_size, sequence_length + 1))
    smiles = torch.randint(
        0, vocabulary_size, (batch_size, sequence_length, max_smiles_length)
    )
    reward = torch.zeros(batch_size, sequence_length, 1)
    done = torch.randint(0, 2, (batch_size, sequence_length + 1, 1), dtype=torch.bool)
    batch = TensorDict(
        {
            "observation": tokens[:, :-1],
            "is_init": done[:, 1:],
            "next": TensorDict(
                {
                    "observation": tokens[:, 1:],
                    "done": done[:, 1:],
                },
                batch_size=[batch_size, sequence_length],
            ),
        },
        batch_size=[batch_size, sequence_length],
    )
    batch.set(reward_key, reward)
    batch.set(smiles_key, smiles)
    return batch


@pytest.mark.parametrize("in_key", [("next", "SMILES"), "SMILES"])
@pytest.mark.parametrize("out_key", [("next", "reward"), "reward_key2"])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("sequence_length", [5])
@pytest.mark.parametrize("max_smiles_length", [10])
def test_reward_transform(
    batch_size, sequence_length, max_smiles_length, in_key, out_key
):
    vocabulary = SMILESVocabulary.create_from_list_of_chars(tokens)
    data = generate_valid_data_batch(
        len(vocabulary),
        batch_size,
        sequence_length,
        max_smiles_length,
        (in_key,),
        (out_key,),
    )
    reward_transform = SMILESReward(dummy_reward_function, vocabulary)
    data = reward_transform(data)
    assert out_key in data.keys()
    done = data.get(("next", "done")).squeeze(-1)
    assert done[done].get(out_key).sum().item() == done.sum().item()
    assert done[~done].get(out_key).sum().item() == 0.0
