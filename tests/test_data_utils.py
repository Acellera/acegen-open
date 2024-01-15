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

    import ipdb

    ipdb.set_trace()

    for key in output_tensordict.keys():
        assert torch.equal(output_tensordict[key], expected_output[key])

    assert output_tensordict == expected_output
