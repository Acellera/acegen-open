import pytest
import torch
from packaging.version import Version


# Expose version constants for conditional skips in tests.
# Usage: @pytest.mark.skipif(TORCHRL_VERSION < Version("0.4"), reason="...")
TORCH_VERSION = Version(torch.__version__.split("+")[0])

try:
    import torchrl
    TORCHRL_VERSION = Version(torchrl.__version__)
except Exception:
    TORCHRL_VERSION = Version("0.0.0")

try:
    import tensordict
    TENSORDICT_VERSION = Version(tensordict.__version__)
except Exception:
    TENSORDICT_VERSION = Version("0.0.0")
