import pytest
import torch
from acegen.models.gru import (
    create_gru_actor,
    create_gru_actor_critic,
    create_gru_critic,
)
from tensordict import TensorDict
from tests.utils import get_default_devices


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
    training_actor, inference_actor = create_gru_actor(
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
    assert ("next", "recurrent_state_actor") in inference_batch.keys(
        include_nested=True
    )

    # Check that the training model works
    training_actor = training_actor.to(device)
    training_batch = training_batch.to(device)
    training_batch = training_actor(training_batch)
    assert "logits" in training_batch.keys()
    assert "action" in training_batch.keys()
    assert ("next", "recurrent_state_actor") in training_batch.keys(include_nested=True)


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
    training_critic, inference_critic = create_gru_critic(
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
    assert ("next", "recurrent_state_critic") in inference_batch.keys(
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
    assert ("next", "recurrent_state_critic") in training_batch.keys(
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
    ) = create_gru_actor_critic(
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
    assert ("next", "recurrent_state") in inference_batch.keys(include_nested=True)

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
    assert ("next", "recurrent_state") in training_batch.keys(include_nested=True)


def test_adapt_ckpt():
    pass
