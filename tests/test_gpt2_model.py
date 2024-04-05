import pytest
import torch
from acegen.data import smiles_to_tensordict
from acegen.models.gpt2 import (
    create_gpt2_actor,
    create_gpt2_actor_critic,
    create_gpt2_critic,
)
from utils import get_default_devices


def generate_valid_data_batch(
    vocabulary_size: int, batch_size: int, sequence_length: int
):
    tokens = torch.randint(0, vocabulary_size, (batch_size, sequence_length + 1))
    batch = smiles_to_tensordict(
        tokens, replace_mask_value=0
    )  # batch_size, sequence_length)
    batch.set("sequence", batch.get("observation"))
    batch.set("sequence_mask", batch.get("mask"))
    return batch


@pytest.mark.parametrize("vocabulary_size", [10])
@pytest.mark.parametrize("device", get_default_devices())
def test_gpt2_actor(vocabulary_size, device, sequence_length=5, batch_size=10):
    torch.manual_seed(0)
    # Create the model and a data batch
    training_actor, inference_actor = create_gpt2_actor(
        vocabulary_size,
        n_head=2,
        n_layer=2,
    )
    training_batch = generate_valid_data_batch(
        vocabulary_size, batch_size, sequence_length
    )
    inference_batch = training_batch.clone()
    inference_batch.batch_size = [batch_size]

    # Check that the inference model works
    inference_actor = inference_actor.to(device)
    inference_batch = inference_batch.to(device)
    inference_batch = inference_actor(inference_batch)
    assert "logits" in inference_batch.keys()
    assert "action" in inference_batch.keys()

    # Check that the training model works
    training_actor = training_actor.to(device)
    training_batch = training_batch.to(device)
    training_batch = training_actor(training_batch)
    assert "logits" in training_batch.keys()
    assert "action" in training_batch.keys()


@pytest.mark.parametrize("vocabulary_size", [10])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("critic_value_per_action", [True, False])
def test_gpt2_critic(
    vocabulary_size,
    device,
    critic_value_per_action,
    sequence_length=5,
    batch_size=10,
):
    torch.manual_seed(0)
    # Create the model and a data batch
    training_critic, inference_critic = create_gpt2_critic(
        vocabulary_size,
        critic_value_per_action=critic_value_per_action,
        n_head=2,
        n_layer=2,
    )
    training_batch = generate_valid_data_batch(
        vocabulary_size, batch_size, sequence_length
    )
    inference_batch = training_batch.clone()
    inference_batch.batch_size = [batch_size]

    # Check that the inference model works
    inference_critic = inference_critic.to(device)
    inference_batch = inference_batch.to(device)
    inference_batch = inference_critic(inference_batch)
    if critic_value_per_action:
        assert "action_value" in inference_batch.keys()
    else:
        assert "state_value" in inference_batch.keys()

    # Check that the training model works
    training_critic = training_critic.to(device)
    training_batch = training_batch.to(device)
    training_batch = training_critic(training_batch)
    if critic_value_per_action:
        assert "action_value" in training_batch.keys()
    else:
        assert "state_value" in training_batch.keys()


@pytest.mark.parametrize("vocabulary_size", [10])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("critic_value_per_action", [True, False])
def test_gpt2_actor_critic(
    vocabulary_size,
    device,
    critic_value_per_action,
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
    ) = create_gpt2_actor_critic(
        vocabulary_size,
        critic_value_per_action=critic_value_per_action,
        n_head=2,
        n_layer=2,
    )
    training_batch = generate_valid_data_batch(
        vocabulary_size, batch_size, sequence_length
    )
    inference_batch = training_batch.clone()
    inference_batch.batch_size = [batch_size]

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
