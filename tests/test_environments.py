import pytest
import torch
from acegen.env.smiles_env import SMILESEnv
from acegen.env.utils import sample_completed_smiles
from tests.utils import get_default_devices
from torchrl.collectors import RandomPolicy
from torchrl.envs.utils import step_mdp


@pytest.mark.parametrize("start_token", [0])
@pytest.mark.parametrize("end_token", [1])
@pytest.mark.parametrize("length_vocabulary", [3])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("one_hot_action_encoding", [False, True])
@pytest.mark.parametrize("one_hot_obs_encoding", [False, True])
def test_multi_step_smiles_env(
    start_token,
    end_token,
    length_vocabulary,
    device,
    batch_size,
    one_hot_action_encoding,
    one_hot_obs_encoding,
):
    env = SMILESEnv(
        start_token=start_token,
        end_token=end_token,
        length_vocabulary=length_vocabulary,
        device=device,
        batch_size=batch_size,
        one_hot_action_encoding=one_hot_action_encoding,
        one_hot_obs_encoding=one_hot_obs_encoding,
    )
    policy = RandomPolicy(env.action_spec)
    td = env.reset()

    if one_hot_obs_encoding:
        assert (torch.argmax(td.get("observation"), dim=-1) == start_token).all()
    else:
        assert (td.get("observation") == start_token).all()

    for i in range(10):

        if one_hot_obs_encoding:
            assert td.get("observation").shape == (batch_size, length_vocabulary)
        else:
            assert td.get("observation").shape == (batch_size,)

        assert td.get("done").shape == (batch_size, 1)

        td = policy(td)

        if one_hot_action_encoding:
            assert td.get("action").shape == (batch_size, length_vocabulary)
        else:
            assert td.get("action").shape == (batch_size,)

        assert td.device == device

        td = env.step(td)
        td = step_mdp(td)


@pytest.mark.parametrize("start_token", [0])
@pytest.mark.parametrize("end_token", [1])
@pytest.mark.parametrize("length_vocabulary", [3])
@pytest.mark.parametrize("env_device", get_default_devices())
@pytest.mark.parametrize("policy_device", get_default_devices())
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("one_hot_action_encoding", [False, True])
@pytest.mark.parametrize("one_hot_obs_encoding", [False, True])
def test_sample_smiles(
    start_token,
    end_token,
    length_vocabulary,
    env_device,
    policy_device,
    batch_size,
    one_hot_action_encoding,
    one_hot_obs_encoding,
):
    env = SMILESEnv(
        start_token=start_token,
        end_token=end_token,
        length_vocabulary=length_vocabulary,
        device=env_device,
        batch_size=batch_size,
        one_hot_action_encoding=one_hot_action_encoding,
        one_hot_obs_encoding=one_hot_obs_encoding,
    )
    policy = RandomPolicy(env.action_spec)
    policy.device = policy_device
    smiles = sample_completed_smiles(env, policy, max_length=10)
