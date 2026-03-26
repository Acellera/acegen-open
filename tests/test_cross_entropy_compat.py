"""Tests for the F.cross_entropy legacy-kwarg compatibility patch in _compat.

torchrl 0.11 calls F.cross_entropy(..., reduce=False) inside
MaskedCategorical.log_prob().  In PyTorch >= 2.x this triggers:

    UserWarning: size_average and reduce args will be deprecated,
                 please use reduction='none' instead.

acegen._compat patches F.cross_entropy to silently convert the old-style
kwargs, eliminating the warning on both stable and nightly.
"""
import warnings

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper — check whether the installed MaskedCategorical actually uses the
# legacy kwargs (only true for torchrl 0.11+).
# ---------------------------------------------------------------------------
def _masked_categorical_triggers_warning():
    """Return True if MaskedCategorical.log_prob raises the size_average warning."""
    try:
        from torchrl.modules import MaskedCategorical
    except ImportError:
        return False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logits = torch.randn(4, 10)
        mask = torch.ones(4, 10, dtype=torch.bool)
        dist = MaskedCategorical(logits=logits, mask=mask)
        actions = dist.sample()
        dist.log_prob(actions)
    return any("size_average" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Test 1 — after importing acegen._compat the patch is active and the warning
# no longer fires from MaskedCategorical.log_prob().
# ---------------------------------------------------------------------------
def test_masked_categorical_log_prob_no_size_average_warning():
    """MaskedCategorical.log_prob() must not emit the size_average UserWarning."""
    import acegen._compat  # ensure patch is applied  # noqa: F401
    from torchrl.modules import MaskedCategorical

    logits = torch.randn(4, 10)
    mask = torch.ones(4, 10, dtype=torch.bool)
    dist = MaskedCategorical(logits=logits, mask=mask)
    actions = dist.sample()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log_probs = dist.log_prob(actions)

    size_avg_warnings = [w for w in caught if "size_average" in str(w.message)]
    assert len(size_avg_warnings) == 0, (
        f"Unexpected size_average warning after _compat patch: {size_avg_warnings}"
    )
    assert log_probs.shape == (4,)


# ---------------------------------------------------------------------------
# Test 2 — the patched F.cross_entropy still returns correct values when
# called with the legacy reduce=False kwarg.
# ---------------------------------------------------------------------------
def test_cross_entropy_compat_reduce_false_correctness():
    """Patched F.cross_entropy(reduce=False) must match reduction='none'."""
    import acegen._compat  # noqa: F401

    torch.manual_seed(0)
    logits = torch.randn(6, 5)
    targets = torch.randint(0, 5, (6,))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result_legacy = F.cross_entropy(logits, targets, reduce=False)

    size_avg_warnings = [w for w in caught if "size_average" in str(w.message)]
    assert len(size_avg_warnings) == 0, "Patch should silence the warning"

    result_new = F.cross_entropy(logits, targets, reduction="none")
    assert torch.allclose(result_legacy, result_new), (
        "reduce=False result must equal reduction='none'"
    )


# ---------------------------------------------------------------------------
# Test 3 — the patched F.cross_entropy handles size_average=True/False.
# ---------------------------------------------------------------------------
def test_cross_entropy_compat_size_average():
    """Patched F.cross_entropy(size_average=...) must not warn and must be correct."""
    import acegen._compat  # noqa: F401

    torch.manual_seed(1)
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result_mean = F.cross_entropy(logits, targets, size_average=True)
        result_sum = F.cross_entropy(logits, targets, size_average=False)

    size_avg_warnings = [w for w in caught if "size_average" in str(w.message)]
    assert len(size_avg_warnings) == 0

    expected_mean = F.cross_entropy(logits, targets, reduction="mean")
    expected_sum = F.cross_entropy(logits, targets, reduction="sum")
    assert torch.allclose(result_mean, expected_mean)
    assert torch.allclose(result_sum, expected_sum)


# ---------------------------------------------------------------------------
# Test 4 — the patch is a no-op when modern kwargs are used (no regression).
# ---------------------------------------------------------------------------
def test_cross_entropy_compat_modern_kwargs_unchanged():
    """Patched F.cross_entropy with reduction='none' must behave identically."""
    import acegen._compat  # noqa: F401

    torch.manual_seed(2)
    logits = torch.randn(5, 7)
    targets = torch.randint(0, 7, (5,))

    result = F.cross_entropy(logits, targets, reduction="none")
    expected = F.cross_entropy(logits, targets, reduction="none")
    assert torch.allclose(result, expected)
    assert result.shape == (5,)


# ---------------------------------------------------------------------------
# Test 5 — actor training forward pass generates no size_average warnings.
# Exercises the full MaskedCategorical path via the GRU actor.
# ---------------------------------------------------------------------------
def test_actor_log_prob_no_size_average_warning():
    """Running a GRU actor training forward pass must not emit size_average warnings."""
    import acegen._compat  # noqa: F401
    from acegen.models import create_gru_actor
    from acegen.data import smiles_to_tensordict

    torch.manual_seed(0)
    vocab_size = 10
    actor_training, _ = create_gru_actor(vocab_size)

    tokens = torch.randint(1, vocab_size, (4, 9))
    data = smiles_to_tensordict(tokens, reward=torch.rand(4))
    data.set("sequence", data.get("observation"))
    data.set("sequence_mask", data.get("mask"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actor_training(data)

    size_avg_warnings = [w for w in caught if "size_average" in str(w.message)]
    assert len(size_avg_warnings) == 0, (
        f"Unexpected size_average warning during actor forward: {size_avg_warnings}"
    )
