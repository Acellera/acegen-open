import torch
from acegen.data import smiles_to_tensordict

from tensordict import TensorDict


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

    # Check rewards are in the right position
    assert (result["next"]["reward"][next_tensordict["done"]].cpu() == reward).all()
