import os
import sys
import shutil
import tempfile

import pytest
import torch

from acegen.data import (
    load_dataset,
    MolBloomDataset,
    smiles_to_tensordict,
    SMILESDataset,
)
from acegen.data.chem_utils import fraction_valid
from acegen.vocabulary.tokenizers import SMILESTokenizerChEMBL
from acegen.vocabulary.vocabulary import Vocabulary
from tensordict import TensorDict
from torch.utils.data import DataLoader


try:
    if sys.version_info < (3,10):
        raise ImportError("Molbloom requires Python 3.10 or higher")
    else:
        from molbloom import BloomFilter, CustomFilter
        _has_molbloom = True
except ImportError:
    _has_molbloom = False


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
    assert result["observation"].shape == (B, T)
    assert result["action"].shape == (B, T)
    assert result["done"].shape == (B, T, 1)
    assert result["terminated"].shape == (B, T, 1)
    assert result["mask"].shape == (B, T)

    assert next_tensordict["observation"].shape == (B, T)
    assert next_tensordict["reward"].shape == (B, T, 1)
    assert next_tensordict["done"].shape == (B, T, 1)
    assert next_tensordict["terminated"].shape == (B, T, 1)

    # Check if the batch_size attribute is correctly set
    assert result.batch_size == torch.Size([B, T])

    # Check rewards are in the right position
    assert (result["next"]["reward"][next_tensordict["done"]].cpu() == reward).all()


@pytest.mark.parametrize("randomize_smiles", [False, True])
def test_load_dataset(randomize_smiles):
    dataset_path = os.path.dirname(__file__) + "/data/smiles_test_set"
    dataset_str = load_dataset(dataset_path)
    assert type(dataset_str) == list
    assert len(dataset_str) == 1000
    vocab = Vocabulary.create_from_strings(
        dataset_str, tokenizer=SMILESTokenizerChEMBL()
    )
    temp_dir = tempfile.mkdtemp()
    dataset = SMILESDataset(
        cache_path=temp_dir,
        dataset_path=dataset_path,
        vocabulary=vocab,
        randomize_smiles=randomize_smiles,
    )
    if _has_molbloom:
        molbloom_dataset = MolBloomDataset(dataset_path=dataset_path)
        assert dataset_str[0] in molbloom_dataset
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    data_batch = dataloader.__iter__().__next__()
    assert isinstance(data_batch, TensorDict)
    shutil.rmtree(temp_dir)


def test_fraction_valid():

    multiple_smiles = [
        "CCO",  # Ethanol (C2H5OH)
        "CCN(CC)CC",  # Triethylamine (C6H15N)
        "CC(=O)OC(C)C",  # Diethyl carbonate (C7H14O3)
        "CC(C)C",  # Isobutane (C4H10)
        "CC1=CC=CC=C1",  # Toluene (C7H8)
    ]

    assert fraction_valid(multiple_smiles) == 1.0

    # add an invalid SMILES
    multiple_smiles.append("invalid")

    assert fraction_valid(multiple_smiles) == 5 / 6
