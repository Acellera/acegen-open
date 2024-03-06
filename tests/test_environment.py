import pytest
import torch
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.vocabulary import SMILESTokenizer, SMILESVocabulary
from tests.utils import get_default_devices
from torchrl.collectors import RandomPolicy
from torchrl.envs.utils import step_mdp

try:
    import promptsmiles

    promptsmiles_available = True
except ImportError:
    promptsmiles_available = False

skip_if_promptsmiles_not_available = pytest.mark.skipif(
    not promptsmiles_available,
    reason="prompsmiles library is not available, skipping this test",
)


@pytest.mark.parametrize("start_token", [0])
@pytest.mark.parametrize("end_token", [1])
@pytest.mark.parametrize("length_vocabulary", [3])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("one_hot_action_encoding", [False, True])
@pytest.mark.parametrize("one_hot_obs_encoding", [False, True])
def test_smiles_env(
    start_token,
    end_token,
    length_vocabulary,
    device,
    batch_size,
    one_hot_action_encoding,
    one_hot_obs_encoding,
):
    torch.manual_seed(0)
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
    torch.manual_seed(0)
    env = SMILESEnv(
        start_token=start_token,
        end_token=end_token,
        length_vocabulary=length_vocabulary,
        device=env_device,
        batch_size=batch_size,
        one_hot_action_encoding=one_hot_action_encoding,
        one_hot_obs_encoding=one_hot_obs_encoding,
    )
    smiles = generate_complete_smiles(
        env, vocabulary=SMILESVocabulary(), policy=None, max_length=10
    )
    terminated = smiles.get(("next", "terminated")).squeeze(
        dim=-1
    )  # if max_length is reached is False
    truncated = smiles.get(("next", "truncated")).squeeze(
        dim=-1
    )  # if max_length is reached is True
    done = smiles.get(("next", "done")).squeeze(dim=-1)
    assert ((terminated | truncated) == done).all()
    finished = done.any(-1)
    obs = smiles.get("observation")
    if one_hot_obs_encoding:
        obs = torch.argmax(obs, dim=-1)
    action = smiles.get("action")
    if one_hot_action_encoding:
        action = torch.argmax(action, dim=-1)
    assert (obs[..., 0] == start_token).all()
    if finished.all():
        assert done.sum() >= batch_size
        assert (done).sum() == batch_size
    assert (action[terminated] == end_token).all()


@skip_if_promptsmiles_not_available
@pytest.mark.parametrize("start_token", [0])
@pytest.mark.parametrize("end_token", [1])
@pytest.mark.parametrize("length_vocabulary", [3])
@pytest.mark.parametrize("env_device", get_default_devices())
@pytest.mark.parametrize("policy_device", get_default_devices())
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("one_hot_action_encoding", [False, True])
@pytest.mark.parametrize("one_hot_obs_encoding", [False, True])
@pytest.mark.parametrize(
    "promptsmiles", ["N1(*)CCN(CC1)CCCCN(*)", "Fc1ccc(*)cc1.C(*)C(O)CC(O)CC(=O)O"]
)
@pytest.mark.parametrize("promptsmiles_optimize", [False, True])
@pytest.mark.parametrize("promptsmiles_shuffle", [False, True])
def test_sample_promptsmiles(
    start_token,
    end_token,
    length_vocabulary,
    env_device,
    policy_device,
    batch_size,
    one_hot_action_encoding,
    one_hot_obs_encoding,
    promptsmiles,
    promptsmiles_optimize,
    promptsmiles_shuffle,
):
    torch.manual_seed(0)
    env = SMILESEnv(
        start_token=start_token,
        end_token=end_token,
        length_vocabulary=length_vocabulary,
        device=env_device,
        batch_size=batch_size,
        one_hot_action_encoding=one_hot_action_encoding,
        one_hot_obs_encoding=one_hot_obs_encoding,
    )
    smiles = generate_complete_smiles(
        env,
        vocabulary=SMILESVocabulary(tokenizer=SMILESTokenizer()),
        policy=None,
        promptsmiles=promptsmiles,
        promptsmiles_optimize=promptsmiles_optimize,
        promptsmiles_shuffle=promptsmiles_shuffle,
        max_length=10,
    )
    terminated = smiles.get(("next", "terminated")).squeeze(
        dim=-1
    )  # if max_length is reached is False
    truncated = smiles.get(("next", "truncated")).squeeze(
        dim=-1
    )  # if max_length is reached is True
    done = smiles.get(("next", "done")).squeeze(dim=-1)
    assert ((terminated | truncated) == done).all()
    finished = done.any(-1)
    obs = smiles.get("observation")
    if one_hot_obs_encoding:
        obs = torch.argmax(obs, dim=-1)
    action = smiles.get("action")
    if one_hot_action_encoding:
        action = torch.argmax(action, dim=-1)
    assert (obs[..., 0] == start_token).all()
    if finished.all():
        assert done.sum() >= batch_size
        assert (done).sum() == batch_size
    assert (action[terminated] == end_token).all()
