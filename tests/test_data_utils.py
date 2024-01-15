import pytest
import torch
from acegen.data import is_in_reference, remove_duplicates, smiles_to_tensordict

from tensordict import TensorDict


@pytest.fixture
def sample_tensordict():
    # Create a sample TensorDict for testing
    return TensorDict(
        {
            "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]]),
            "tensor2": torch.tensor([[10, 20], [30, 40], [10, 20], [50, 60]]),
        },
        batch_size=[4],
    )


def test_remove_duplicates(sample_tensordict):
    # Arrange
    input_tensordict = sample_tensordict.copy()
    key = "tensor1"

    # Act
    output_tensordict = remove_duplicates(input_tensordict, key)

    # Assert
    expected_output = TensorDict(
        {
            "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            "tensor2": torch.tensor([[10, 20], [30, 40], [50, 60]]),
        },
        batch_size=[3],
    )
    for key in output_tensordict.keys():
        assert torch.equal(output_tensordict[key], expected_output[key])


def test_is_in_reference(sample_tensordict):
    # Arrange
    reference_tensordict = TensorDict(
        {
            "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [10, 11, 12]]),
            "tensor2": torch.tensor([[10, 20], [30, 40], [50, 60]]),
        },
        batch_size=[3],
    )
    key = "tensor1"

    # Act
    in_reference = is_in_reference(sample_tensordict, reference_tensordict, key)

    # Assert
    expected_in_reference = torch.tensor([True, True, True, False])
    assert torch.equal(in_reference, expected_in_reference)


def test_smiles_to_tensordict():
    # Arrange
    B, T = 3, 4
    smiles = torch.randint(0, 10, (B, T))
    reward = torch.rand(B)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Act
    result = smiles_to_tensordict(smiles, reward, device=device)

    # Assert
    assert isinstance(result, TensorDict)

    # Check keys in the main TensorDict
    keys = result.keys()
    assert "observation" in keys
    assert "action" in keys
    assert "done" in keys
    assert "terminated" in keys
    assert "mask" in keys
    assert "next" in keys

    # Check keys in the nested TensorDict ('next')
    next_tensordict = result["next"]
    keys = next_tensordict.keys()
    assert "observation" in keys
    assert "reward" in keys
    assert "done" in keys
    assert "terminated" in keys

    # Check shapes of tensors
    assert result["observation"].shape == (B, T - 1)
    assert result["action"].shape == (B, T - 1)
    assert result["done"].shape == (B, T - 1, 1)
    assert result["terminated"].shape == (B, T - 1, 1)
    assert result["mask"].shape == (B, T - 1)

    assert next_tensordict["observation"].shape == (B, T - 1)
    assert next_tensordict["reward"].shape == (B, T - 1, 1)
    assert next_tensordict["done"].shape == (B, T - 1, 1)
    assert next_tensordict["terminated"].shape == (B, T - 1, 1)

    # Check if the batch_size attribute is correctly set
    assert result.batch_size == torch.Size([B, T - 1])
