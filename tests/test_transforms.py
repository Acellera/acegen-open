import pytest
import torch
from acegen.transforms import BurnInTransform, SMILESReward
from acegen.vocabulary import SMILESVocabulary
from tensordict import TensorDict
from tests.utils import get_default_devices
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.modules import GRUModule

tokens = ["(", ")", "1", "=", "C", "N", "O"]


def dummy_reward_function(smiles):
    return [1 for _ in smiles]


def generate_valid_data_batch(
    vocabulary_size: int,
    batch_size: int = 2,
    sequence_length: int = 5,
    max_smiles_length: int = 10,
    smiles_key: str = "SMILES",
    reward_key: str = "reward",
):
    tokens = torch.randint(0, vocabulary_size, (batch_size, sequence_length + 1, 1))
    smiles = torch.randint(
        0, vocabulary_size, (batch_size, sequence_length, max_smiles_length)
    )
    reward = torch.zeros(batch_size, sequence_length, 1)
    done = torch.randint(0, 2, (batch_size, sequence_length + 1, 1), dtype=torch.bool)
    is_init = torch.zeros(batch_size, sequence_length, 1, dtype=torch.bool)
    batch = TensorDict(
        {
            "observation": tokens[:, :-1],
            "is_init": is_init,
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
    batch.set(("next", reward_key), reward)
    batch.set(("next", smiles_key), smiles)
    return batch


@pytest.mark.parametrize("smiles_key", ["SMILES"])
@pytest.mark.parametrize("reward_key", ["reward", "reward2"])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("sequence_length", [5])
@pytest.mark.parametrize("max_smiles_length", [10])
def test_reward_transform(
    batch_size, sequence_length, max_smiles_length, smiles_key, reward_key
):
    vocabulary = SMILESVocabulary.create_from_list_of_chars(tokens)
    data = generate_valid_data_batch(
        len(vocabulary),
        batch_size,
        sequence_length,
        max_smiles_length,
        smiles_key,
        reward_key,
    )
    reward_transform = SMILESReward(
        vocabulary=vocabulary,
        reward_function=dummy_reward_function,
        in_keys=[smiles_key],
        out_keys=[reward_key],
    )
    data = reward_transform(data)
    data = data.get("next")
    assert reward_key in data.keys(include_nested=True)
    done = data.get("done").squeeze(-1)
    assert data[done].get(reward_key).sum().item() == done.sum().item()
    assert data[~done].get(reward_key).sum().item() == 0.0


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("sequence_length", [5])
@pytest.mark.parametrize("vocabulary_size", [4])
@pytest.mark.parametrize("max_smiles_length", [10])
@pytest.mark.parametrize("out_keys", [None, ["hidden"]])
@pytest.mark.parametrize("device", get_default_devices())
def test_burn_in_transform(
    vocabulary_size, batch_size, sequence_length, max_smiles_length, out_keys, device
):
    data = generate_valid_data_batch(
        vocabulary_size,
        batch_size,
        sequence_length,
        max_smiles_length,
    )
    gru_module = GRUModule(
        input_size=1,
        hidden_size=10,
        batch_first=True,
        in_keys=["observation", "hidden"],
        out_keys=["intermediate", ("next", "hidden")],
    ).set_recurrent_mode(True)
    hidden_state = torch.zeros(
        batch_size,
        sequence_length,
        gru_module.gru.num_layers,
        gru_module.gru.hidden_size,
    )
    gru_module = gru_module.to(device)
    data.set("hidden", hidden_state)
    data.set("observation", data.get("observation").to(torch.float32))
    burn_in_transform = BurnInTransform(
        modules=[gru_module],
        burn_in=sequence_length - 2,
        out_keys=out_keys,
    )
    data = burn_in_transform(data)

    assert data.shape[-1] == 2
    assert data[:, 0].get("hidden").abs().sum() > 0.0
    assert data[:, 1:].get("hidden").sum() == 0.0


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("sequence_length", [5])
@pytest.mark.parametrize("vocabulary_size", [4])
@pytest.mark.parametrize("max_smiles_length", [10])
@pytest.mark.parametrize("out_keys", [None, ["hidden"]])
@pytest.mark.parametrize("device", get_default_devices())
def test_burn_in_transform_with_buffer(
    vocabulary_size, batch_size, sequence_length, max_smiles_length, out_keys, device
):
    device = torch.device("cuda" if torch.cuda.device_count() > 1 else "cpu")
    data = generate_valid_data_batch(
        vocabulary_size,
        batch_size,
        sequence_length,
        max_smiles_length,
    )
    gru_module = GRUModule(
        input_size=1,
        hidden_size=10,
        batch_first=True,
        in_keys=["observation", "hidden"],
        out_keys=["intermediate", ("next", "hidden")],
    ).set_recurrent_mode(True)
    gru_module = gru_module.to(device)
    hidden_state = torch.zeros(
        batch_size,
        sequence_length,
        gru_module.gru.num_layers,
        gru_module.gru.hidden_size,
    )
    data.set("hidden", hidden_state)
    data.set("observation", data.get("observation").to(torch.float32))
    burn_in_transform = BurnInTransform(
        modules=[gru_module],
        burn_in=sequence_length - 2,
        out_keys=out_keys,
    )
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(batch_size),
        batch_size=1,
    )
    buffer.append_transform(burn_in_transform)
    buffer.extend(data)
    data = buffer.sample(1)
    assert data.shape[-1] == 2
    assert data[:, 0].get("hidden").abs().sum() > 0.0
    assert data[:, 1:].get("hidden").sum() == 0.0
