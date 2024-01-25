import pytest
import torch
from acegen.models.lstm import (
    create_lstm_actor,
    create_lstm_actor_critic,
    create_lstm_critic,
)
from acegen.models.utils import adapt_state_dict
from tensordict import TensorDict
from tests.utils import get_default_devices
from torchrl.envs import TensorDictPrimer


def generate_valid_data_batch(
    vocabulary_size: int, batch_size: int, sequence_length: int
):
    tokens = torch.randint(0, vocabulary_size, (batch_size, sequence_length + 1))
    done = torch.randint(0, 2, (batch_size, sequence_length + 1, 1), dtype=torch.bool)
    batch = TensorDict(
        {
            "observation": tokens[:, :-1],
            "done": torch.zeros(batch_size, sequence_length, 1),
            "is_init": done[:, 1:],
            "next": TensorDict(
                {
                    "observation": tokens[:, 1:],
                    "done": done[:, :-1],
                },
                batch_size=[batch_size, sequence_length],
            ),
        },
        batch_size=[batch_size, sequence_length],
    )
    return batch


@pytest.mark.parametrize("vocabulary_size", [10])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("python_based", [True, False])
def test_gru_actor(
    vocabulary_size, device, python_based, sequence_length=5, batch_size=10
):
    torch.manual_seed(0)
    # Create the model and a data batch
    training_actor, inference_actor = create_lstm_actor(
        vocabulary_size, python_based=python_based
    )
    training_batch = generate_valid_data_batch(
        vocabulary_size, batch_size, sequence_length
    )
    inference_batch = training_batch[:, 0].clone()

    # Check that the inference model works
    inference_actor = inference_actor.to(device)
    inference_batch = inference_batch.to(device)
    inference_batch = inference_actor(inference_batch)
    assert "logits" in inference_batch.keys()
    assert "action" in inference_batch.keys()
    assert ("next", "recurrent_state_actor_h") in inference_batch.keys(
        include_nested=True
    )
    assert ("next", "recurrent_state_actor_c") in inference_batch.keys(
        include_nested=True
    )

    # Check that the training model works
    training_actor = training_actor.to(device)
    training_batch = training_batch.to(device)
    training_batch = training_actor(training_batch)
    assert "logits" in training_batch.keys()
    assert "action" in training_batch.keys()
    assert ("next", "recurrent_state_actor_h") in training_batch.keys(
        include_nested=True
    )
    assert ("next", "recurrent_state_actor_c") in training_batch.keys(
        include_nested=True
    )


@pytest.mark.parametrize("vocabulary_size", [10])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("critic_value_per_action", [True, False])
@pytest.mark.parametrize("python_based", [True, False])
def test_gru_critic(
    vocabulary_size,
    device,
    critic_value_per_action,
    python_based,
    sequence_length=5,
    batch_size=10,
):
    torch.manual_seed(0)
    # Create the model and a data batch
    training_critic, inference_critic = create_lstm_critic(
        vocabulary_size,
        critic_value_per_action=critic_value_per_action,
        python_based=python_based,
    )
    training_batch = generate_valid_data_batch(
        vocabulary_size, batch_size, sequence_length
    )
    inference_batch = training_batch[:, 0].clone()

    # Check that the inference model works
    inference_critic = inference_critic.to(device)
    inference_batch = inference_batch.to(device)
    inference_batch = inference_critic(inference_batch)
    if critic_value_per_action:
        assert "action_value" in inference_batch.keys()
    else:
        assert "state_value" in inference_batch.keys()
    assert ("next", "recurrent_state_critic_h") in inference_batch.keys(
        include_nested=True
    )
    assert ("next", "recurrent_state_critic_c") in inference_batch.keys(
        include_nested=True
    )

    # Check that the training model works
    training_critic = training_critic.to(device)
    training_batch = training_batch.to(device)
    training_batch = training_critic(training_batch)
    if critic_value_per_action:
        assert "action_value" in training_batch.keys()
    else:
        assert "state_value" in training_batch.keys()
    assert ("next", "recurrent_state_critic_h") in training_batch.keys(
        include_nested=True
    )
    assert ("next", "recurrent_state_critic_c") in training_batch.keys(
        include_nested=True
    )


@pytest.mark.parametrize("vocabulary_size", [10])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("critic_value_per_action", [True, False])
@pytest.mark.parametrize("python_based", [True, False])
def test_gru_actor_critic(
    vocabulary_size,
    device,
    critic_value_per_action,
    python_based,
    sequence_length=5,
    batch_size=10,
):
    torch.manual_seed(0)
    # Create the model and a data batch
    (
        training_actor,
        inference_actor,
        training_critic,
        inference_critic,
    ) = create_lstm_actor_critic(
        vocabulary_size,
        critic_value_per_action=critic_value_per_action,
        python_based=python_based,
    )
    training_batch = generate_valid_data_batch(
        vocabulary_size, batch_size, sequence_length
    )
    inference_batch = training_batch[:, 0].clone()

    # Check that the inference model works
    inference_actor = inference_actor.to(device)
    inference_critic = inference_critic.to(device)
    inference_batch = inference_batch.to(device)
    inference_batch = inference_actor(inference_batch)
    inference_batch = inference_critic(inference_batch)
    assert "logits" in inference_batch.keys()
    assert "action" in inference_batch.keys()
    if critic_value_per_action:
        assert "action_value" in inference_batch.keys()
    else:
        assert "state_value" in inference_batch.keys()
    assert ("next", "recurrent_state_h") in inference_batch.keys(include_nested=True)
    assert ("next", "recurrent_state_c") in inference_batch.keys(include_nested=True)

    # Check that the training model works
    training_actor = training_actor.to(device)
    training_critic = training_critic.to(device)
    training_batch = training_batch.to(device)
    training_batch = training_actor(training_batch)
    training_batch = training_critic(training_batch)
    assert "logits" in training_batch.keys()
    assert "action" in training_batch.keys()
    if critic_value_per_action:
        assert "action_value" in training_batch.keys()
    else:
        assert "state_value" in training_batch.keys()
    assert ("next", "recurrent_state_h") in training_batch.keys(include_nested=True)
    assert ("next", "recurrent_state_c") in training_batch.keys(include_nested=True)


@pytest.mark.parametrize("vocabulary_size", [10])
def test_create_tensordict_primer(
    vocabulary_size,
):
    (
        training_actor,
        _,
        _,
        _,
    ) = create_lstm_actor_critic(
        vocabulary_size,
    )

    primers = training_actor.rnn_spec.expand(10)
    rhs_primer = TensorDictPrimer(primers)
    assert "recurrent_state_c" in rhs_primer.primers.keys()
    assert "recurrent_state_h" in rhs_primer.primers.keys()


def test_adapt_state_dict():
    # Arrange
    source_state_dict = {
        "source_conv1.weight": torch.randn(3, 3, 3),
        "source_fc.weight": torch.randn(10, 5),
    }
    target_state_dict = {
        "target_conv1.weight": torch.randn(3, 3, 3),
        "target_fc.weight": torch.randn(10, 5),
    }

    # Act
    adapted_state_dict = adapt_state_dict(source_state_dict, target_state_dict)

    for key_adapted, key_target in zip(
        adapted_state_dict.keys(), target_state_dict.keys()
    ):
        assert key_adapted == key_target

    for key_source, key_adapted in zip(
        source_state_dict.keys(), adapted_state_dict.keys()
    ):
        assert (source_state_dict[key_source] == adapted_state_dict[key_adapted]).all()

    # Test with state dicts of different lengths
    with pytest.raises(
        ValueError,
        match="The source and target state dicts must have the same number of parameters.",
    ):
        adapt_state_dict(source_state_dict, {"fc.weight": torch.randn(10, 5)})

    # Test with mismatched shapes
    source_state_dict_mismatched = {
        "source_conv1.weight": torch.randn(3, 3, 3),
        "source_fc.weight": torch.randn(5, 10),  # Mismatched shape
    }

    key_source = "source_fc.weight"
    key_target = "target_fc.weight"
    msg = f"The shape of source key {key_source} .* and target key {key_target} .* do not match."
    with pytest.warns(UserWarning, match=msg):
        adapt_state_dict(source_state_dict_mismatched, target_state_dict)
