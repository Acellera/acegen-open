"""Compatibility shims for torch/torchrl/tensordict version differences.

Import version-sensitive names from this module rather than directly from
tensordict or torchrl, so that a single try/except handles each API migration.
"""
import inspect
import warnings

import torch
import torch.nn.functional as _F

# ---------------------------------------------------------------------------
# torch.nn.functional.cross_entropy: legacy 'reduce' / 'size_average' kwargs
# deprecated in PyTorch >= 2.x.  torchrl 0.11 still passes reduce=False inside
# MaskedCategorical.log_prob(), which triggers a UserWarning on every log_prob
# call.  Patch cross_entropy once here to silently convert the old-style kwargs
# to the modern 'reduction=' form so the warning never fires.
# Backwards-compatible: the patch is a no-op when neither kwarg is present.
# ---------------------------------------------------------------------------
_orig_cross_entropy = _F.cross_entropy


def _cross_entropy_compat(input, target, *args, **kwargs):
    if "reduce" in kwargs and "reduction" not in kwargs:
        kwargs["reduction"] = "none" if not kwargs.pop("reduce") else "mean"
    if "size_average" in kwargs and "reduction" not in kwargs:
        kwargs["reduction"] = "mean" if kwargs.pop("size_average") else "sum"
    # Strip any remaining legacy kwargs that are now positional-only or removed
    kwargs.pop("reduce", None)
    kwargs.pop("size_average", None)
    return _orig_cross_entropy(input, target, *args, **kwargs)


_F.cross_entropy = _cross_entropy_compat

# ---------------------------------------------------------------------------
# tensordict: TensorDict/TensorDictBase
# Moved from tensordict.tensordict → tensordict at ~0.5→0.6
# ---------------------------------------------------------------------------
try:
    from tensordict.tensordict import TensorDict, TensorDictBase  # <=0.5
except ImportError:
    from tensordict import TensorDict, TensorDictBase  # >=0.6

# ---------------------------------------------------------------------------
# tensordict: set_interaction_type
# Moved from tensordict.nn.probabilistic → torchrl.envs.utils as
# set_exploration_type in newer torchrl/tensordict versions.
# ---------------------------------------------------------------------------
try:
    from tensordict.nn.probabilistic import (
        set_interaction_type as set_exploration_type,
    )
except ImportError:
    from torchrl.envs.utils import set_exploration_type

# ---------------------------------------------------------------------------
# tensordict: remove_duplicates
# Removed from tensordict.utils in >=0.6; provide a fallback implementation.
# ---------------------------------------------------------------------------
try:
    from tensordict.utils import remove_duplicates
except ImportError:

    def remove_duplicates(td, key):
        vals = td.get(key)
        flat = vals.flatten(1) if vals.dim() > 1 else vals.unsqueeze(1)
        seen, keep = set(), []
        for i, v in enumerate(flat.tolist()):
            t = tuple(v)
            if t not in seen:
                seen.add(t)
                keep.append(i)
        return td[keep]

# ---------------------------------------------------------------------------
# tensordict: isin
# Location varies across versions (tensordict.utils vs tensordict).
# ---------------------------------------------------------------------------
try:
    from tensordict.utils import isin
except ImportError:
    try:
        from tensordict import isin
    except ImportError:

        def isin(input, reference, key):
            return torch.isin(input.get(key), reference.get(key))

# ---------------------------------------------------------------------------
# torchrl spec class renames (~0.3→0.4)
#   CompositeSpec      → Composite
#   DiscreteTensorSpec → Categorical
#   UnboundedContinuousTensorSpec → Unbounded
# Current code already uses the new names; shim handles older installations.
# ---------------------------------------------------------------------------
try:
    from torchrl.data import (
        Categorical,
        Composite,
        Unbounded,
    )
except ImportError:
    from torchrl.data import (
        DiscreteTensorSpec as Categorical,
        CompositeSpec as Composite,
        UnboundedContinuousTensorSpec as Unbounded,
    )

# ---------------------------------------------------------------------------
# torchrl spec class rename (~0.11)
#   OneHotDiscreteTensorSpec → OneHot
# ---------------------------------------------------------------------------
try:
    from torchrl.data import OneHotDiscreteTensorSpec  # torchrl < 0.11
except ImportError:
    from torchrl.data import OneHot as OneHotDiscreteTensorSpec  # torchrl >= 0.11

# ---------------------------------------------------------------------------
# torchrl GAE: 'shifted' kwarg was added in ~0.3
# Use make_gae() instead of GAE() directly in scripts that pass shifted=True.
# ---------------------------------------------------------------------------
from torchrl.objectives.value.advantages import GAE as _GAE

_GAE_SUPPORTS_SHIFTED = "shifted" in inspect.signature(_GAE).parameters


def make_gae(**kwargs):
    """Create a GAE instance, dropping 'shifted' if the installed version does not support it."""
    if not _GAE_SUPPORTS_SHIFTED:
        kwargs.pop("shifted", None)
    return _GAE(**kwargs)


# ---------------------------------------------------------------------------
# torchrl GRUModule/LSTMModule: set_recurrent_mode() removed in 0.8
# Old API: rnn_module.set_recurrent_mode(True) returned a weight-sharing copy.
# New API: pass default_recurrent_mode=True to the constructor, share the
#          underlying nn.GRU/nn.LSTM via the gru=/lstm= kwarg.
# ---------------------------------------------------------------------------
def make_recurrent(rnn_module):
    """Return a recurrent-mode copy of a GRUModule/LSTMModule, sharing weights.

    In torchrl < 0.8, delegates to rnn_module.set_recurrent_mode(True).
    In torchrl >= 0.8, reconstructs with the same underlying RNN and
    default_recurrent_mode=True so that the module processes full sequences.
    """
    try:
        return rnn_module.set_recurrent_mode(True)
    except RuntimeError:
        cls = type(rnn_module)
        if hasattr(rnn_module, "gru"):
            inner_kwarg, inner = "gru", rnn_module.gru
        elif hasattr(rnn_module, "lstm"):
            inner_kwarg, inner = "lstm", rnn_module.lstm
        else:
            raise
        return cls(
            **{inner_kwarg: inner},
            in_keys=rnn_module.in_keys,
            out_keys=rnn_module.out_keys,
            default_recurrent_mode=True,
        )
