import pytest
import torch

from acegen.data import (
    remove_duplicated_keys,
    remove_keys_in_reference_tensordict,
    smiles_to_tensordict,
)
