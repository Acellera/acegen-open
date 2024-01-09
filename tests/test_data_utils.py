import pytest
import torch

from acegen.data import (
    remove_duplicated_keys,
    remove_keys_in_reference,
    smiles_to_tensordict,
)
