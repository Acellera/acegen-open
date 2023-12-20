import pytest
import torch
from torchrl.envs.utils import step_mdp
from torchrl.collectors import RandomPolicy
from acegen.smiles_environments.multi_step_smiles_env import MultiStepSMILESEnv
from tests.utils import get_default_devices


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
        one_hot_obs_encoding):
    env = MultiStepSMILESEnv(
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
            # TODO: can it be shape (batch_size, 1)?
            assert td.get("observation").shape == (batch_size,)

        assert td.get("done").shape == (batch_size, 1)

        td = policy(td)

        if one_hot_action_encoding:
            assert td.get("action").shape == (batch_size, length_vocabulary)
        else:
            # TODO: can it be shape (batch_size, 1)?
            assert td.get("action").shape == (batch_size,)

        td = env.step(td)
        td = step_mdp(td)

# @pytest.mark.parametrize("start_token", [0])
# @pytest.mark.parametrize("end_token", [1])
# @pytest.mark.parametrize("length_vocabulary", [3])
# @pytest.mark.parametrize("device", get_default_devices())
# @pytest.mark.parametrize("batch_size", [2, 4])
# @pytest.mark.parametrize("one_hot_action_encoding", [False, True])
# @pytest.mark.parametrize("one_hot_obs_encoding", [False, True])
# def test_single_step_smiles_env(
#         start_token,
#         end_token,
#         length_vocabulary,
#         device,
#         batch_size,
#         one_hot_action_encoding,
#         one_hot_obs_encoding):
#     env = SingleStepSMILESEnv(
#         start_token=start_token,
#         end_token=end_token,
#         length_vocabulary=length_vocabulary,
#         device=device,
#         batch_size=batch_size,
#         one_hot_action_encoding=one_hot_action_encoding,
#         one_hot_obs_encoding=one_hot_obs_encoding,
#     )
#     td = env.reset()
