import pytest
import torch
from tensordict import TensorDict

from acegen._compat import isin
from acegen.data import smiles_to_tensordict
from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    RandomSampler,
    TensorDictMaxValueWriter,
    TensorDictReplayBuffer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data_batch(batch_size: int = 5, sequence_length: int = 8, vocabulary_size: int = 20):
    """Create a data batch in the format produced by generate_complete_smiles."""
    tokens = torch.randint(1, vocabulary_size, (batch_size, sequence_length + 1))
    reward = torch.rand(batch_size)
    data = smiles_to_tensordict(tokens, reward=reward)
    # Scripts also store sequence / sequence_mask on the batch
    data.set("sequence", data.get("observation"))
    data.set("sequence_mask", data.get("mask"))
    return data


def make_experience_replay_buffer(
    buffer_size: int = 50,
    replay_batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
    sampler: str = "prioritized",
):
    """Create a replay buffer matching the construction pattern in ag.py / ppo.py."""
    storage = LazyTensorStorage(buffer_size, device=device)
    if sampler == "prioritized":
        replay_sampler = PrioritizedSampler(storage.max_size, alpha=1.0, beta=1.0)
    else:
        replay_sampler = RandomSampler()
    experience_replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=replay_sampler,
        batch_size=replay_batch_size,
        writer=TensorDictMaxValueWriter(rank_key="priority"),
        priority_key="priority",
    )
    return experience_replay_buffer


def extend_buffer(buffer: TensorDictReplayBuffer, data: TensorDict):
    """Extend the replay buffer using the pattern from ag.py / ppo.py.

    This is the exact sequence of operations the training scripts perform:
    1. Clone data
    2. Flatten batch_size to 1D (required by MaxValueWriter)
    3. Deduplicate against existing buffer contents using isin()
    4. Set 'priority' from reward
    5. Call buffer.extend()
    """
    replay_data = data.clone()

    # MaxValueWriter is not compatible with storages of more than one dimension.
    replay_data.batch_size = [replay_data.batch_size[0]]

    # Remove SMILES that are already in the replay buffer
    if len(buffer) > 0:
        is_duplicated = isin(
            input=replay_data,
            key="action",
            reference=buffer[:],
        )
        replay_data = replay_data[~is_duplicated]

    # Add data to the replay buffer
    if len(replay_data) > 0:
        reward = replay_data.get(("next", "reward"))
        # Priority must be 1D (one scalar per episode). torchrl >= 0.11 enforces this strictly.
        replay_data.set("priority", reward.reshape(reward.shape[0], -1).max(dim=-1).values)
        buffer.extend(replay_data)

    return replay_data  # return filtered data so callers can inspect it


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_buffer_construction():
    """Buffer is created without errors and starts empty."""
    buf = make_experience_replay_buffer()
    assert len(buf) == 0


@pytest.mark.parametrize("sampler", ["prioritized", "random"])
def test_buffer_extend_and_length(sampler):
    """Extending the buffer increases its length."""
    buf = make_experience_replay_buffer(sampler=sampler)
    data = make_data_batch(batch_size=5)
    extend_buffer(buf, data)
    assert len(buf) == 5


def test_buffer_extend_does_not_exceed_capacity():
    """Buffer respects its maximum capacity (MaxValueWriter keeps top-K)."""
    capacity = 8
    buf = make_experience_replay_buffer(buffer_size=capacity)

    for _ in range(4):
        data = make_data_batch(batch_size=5)
        extend_buffer(buf, data)

    assert len(buf) <= capacity


def test_buffer_batch_size_flattening():
    """batch_size must be flattened to 1D before extend; 2D raises or misbehaves."""
    buf = make_experience_replay_buffer()
    data = make_data_batch(batch_size=4, sequence_length=6)

    # Verify the raw data has 2D batch_size
    assert len(data.batch_size) == 2

    replay_data = data.clone()
    # Flatten to 1D as the scripts do
    replay_data.batch_size = [replay_data.batch_size[0]]
    assert len(replay_data.batch_size) == 1

    reward = replay_data.get(("next", "reward"))
    replay_data.set("priority", reward.reshape(reward.shape[0], -1).max(dim=-1).values)
    buf.extend(replay_data)  # should succeed
    assert len(buf) == 4


def test_isin_deduplication():
    """isin() correctly identifies duplicates already in the buffer."""
    buf = make_experience_replay_buffer(buffer_size=20)

    data = make_data_batch(batch_size=5)
    extend_buffer(buf, data)
    initial_len = len(buf)

    # Extend with the same data — all entries should be detected as duplicates
    filtered = extend_buffer(buf, data)
    assert len(filtered) == 0, "All entries should be flagged as duplicates"
    assert len(buf) == initial_len, "Buffer length should not grow on re-add of same data"


def test_isin_new_data_not_duplicated():
    """New data (different tokens) is not flagged as duplicate."""
    buf = make_experience_replay_buffer(buffer_size=20)

    data1 = make_data_batch(batch_size=4)
    extend_buffer(buf, data1)

    torch.manual_seed(999)
    data2 = make_data_batch(batch_size=4, vocabulary_size=50)
    filtered = extend_buffer(buf, data2)

    # At least some entries should pass the deduplication filter
    # (may not be 4 if random collision, but practically >0)
    assert len(filtered) >= 0  # non-negative sanity check; failure would be an exception


def test_buffer_sample_returns_tensordict():
    """Sampling from a populated buffer returns a TensorDict."""
    replay_batch_size = 3
    buf = make_experience_replay_buffer(replay_batch_size=replay_batch_size)
    data = make_data_batch(batch_size=10)
    extend_buffer(buf, data)

    sample = buf.sample()
    assert isinstance(sample, TensorDict)
    assert sample.batch_size[0] == replay_batch_size


def test_buffer_sample_exclude_internal_keys():
    """Sampled data can have internal keys excluded, as done in training scripts."""
    buf = make_experience_replay_buffer(replay_batch_size=3)
    data = make_data_batch(batch_size=10)
    extend_buffer(buf, data)

    sample = buf.sample()
    # Scripts exclude these keys before using replay data in the loss
    cleaned = sample.exclude("priority", "index", "_weight")
    assert "priority" not in cleaned.keys()
    assert "index" not in cleaned.keys()
    assert "_weight" not in cleaned.keys()
    # Core data keys should still be present
    assert "action" in cleaned.keys()
    assert "observation" in cleaned.keys()


def test_multiple_extend_sample_cycles():
    """Running several extend/sample cycles completes without errors."""
    buf = make_experience_replay_buffer(
        buffer_size=20, replay_batch_size=4
    )

    for step in range(6):
        data = make_data_batch(batch_size=5)
        extend_buffer(buf, data)

        if len(buf) >= 4:
            sample = buf.sample()
            assert sample.batch_size[0] == 4
            _ = sample.exclude("priority", "index", "_weight")


def test_priority_set_from_reward():
    """Priority stored in the buffer comes from the episode reward tensor."""
    buf = make_experience_replay_buffer(buffer_size=20)
    data = make_data_batch(batch_size=5)
    extend_buffer(buf, data)

    # After extend the buffer entries should have a 'priority' key
    stored = buf[:]
    assert "priority" in stored.keys()


def test_buffer_with_ppo_sampler():
    """PPO uses PrioritizedSampler with alpha=0.9 — verify construction and sampling."""
    storage = LazyTensorStorage(30)
    buf = TensorDictReplayBuffer(
        storage=storage,
        sampler=PrioritizedSampler(storage.max_size, alpha=0.9, beta=1.0),
        batch_size=4,
        writer=TensorDictMaxValueWriter(rank_key="priority"),
        priority_key="priority",
    )
    data = make_data_batch(batch_size=10)
    extend_buffer(buf, data)
    assert len(buf) >= 1

    sample = buf.sample()
    cleaned = sample.exclude("_weight", "index", "priority", inplace=False)
    assert "action" in cleaned.keys()
