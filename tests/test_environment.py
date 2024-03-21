import pytest
import torch
from acegen.models import adapt_state_dict, models
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.vocabulary import SMILESVocabulary
from tests.utils import get_default_devices
from torchrl.collectors import RandomPolicy
from torchrl.envs import InitTracker, TensorDictPrimer, TransformedEnv
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
        max_length=10,
        batch_size=batch_size,
        one_hot_action_encoding=one_hot_action_encoding,
        one_hot_obs_encoding=one_hot_obs_encoding,
    )
    smiles = generate_complete_smiles(env, vocabulary=SMILESVocabulary(), policy=None)
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


@pytest.mark.parametrize("start_token", [0])
@pytest.mark.parametrize("end_token", [1])
@pytest.mark.parametrize("env_device", get_default_devices())
@pytest.mark.parametrize("policy_device", get_default_devices())
@pytest.mark.parametrize(
    "prompt, prompt_error", [("c1ccccc", False), ("c%11ccccc", True)]
)
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize(
    "one_hot_action_encoding", [False]
)  # Fails with one-hot action encoding
@pytest.mark.parametrize("one_hot_obs_encoding", [False, True])
def test_sample_smiles_with_prompt(
    start_token,
    end_token,
    env_device,
    policy_device,
    prompt,
    prompt_error,
    batch_size,
    one_hot_action_encoding,
    one_hot_obs_encoding,
):
    torch.manual_seed(0)
    create_actor, _, _, voc_path, ckpt_path, tokenizer = models["gru"]
    # Create vocabulary
    with open(voc_path, "r") as f:
        tokens = f.read().splitlines()
    tokens_dict = dict(zip(tokens, range(len(tokens))))
    vocabulary = SMILESVocabulary.create_from_dict(
        tokens_dict,
        start_token="GO",
        end_token="EOS",
        tokenizer=tokenizer,
    )
    length_vocabulary = len(vocabulary)
    # Create environment
    env = SMILESEnv(
        start_token=start_token,
        end_token=end_token,
        length_vocabulary=length_vocabulary,
        device=env_device,
        max_length=20,
        batch_size=batch_size,
        one_hot_action_encoding=one_hot_action_encoding,
        one_hot_obs_encoding=one_hot_obs_encoding,
    )
    if prompt_error:
        with pytest.raises(RuntimeError):
            smiles = generate_complete_smiles(
                env, vocabulary=vocabulary, policy=None, prompt=prompt
            )
        pytest.xfail("Correctly raised error for invalid prompt")
    else:
        smiles = generate_complete_smiles(
            env, vocabulary=vocabulary, policy=None, prompt=prompt
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
    # Check they all start with the prompt
    smiles_str = [vocabulary.decode(smi.numpy()) for smi in action.cpu()]
    assert all([smi.startswith(prompt) for smi in smiles_str])

    # Check return_smiles_only
    smiles = generate_complete_smiles(
        env, vocabulary=vocabulary, policy=None, prompt=prompt, return_smiles_only=True
    )
    assert len(smiles) == batch_size
    assert all([smi.startswith(prompt) for smi in smiles])
    if prompt_error:
        assert all(smi == prompt for smi in smiles)


@skip_if_promptsmiles_not_available
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize(
    "promptsmiles", ["N1(*)CCN(CC1)CCCCN(*)", "Fc1ccc(*)cc1.C(*)C(O)CC(O)CC(=O)O"]
)
@pytest.mark.parametrize("promptsmiles_optimize", [False, True])
@pytest.mark.parametrize("promptsmiles_shuffle", [False, True])
@pytest.mark.parametrize("promptsmiles_multi", [False, True])
@pytest.mark.parametrize("device", get_default_devices())
def test_sample_promptsmiles(
    batch_size,
    promptsmiles,
    promptsmiles_optimize,
    promptsmiles_shuffle,
    promptsmiles_multi,
    device,
):
    torch.manual_seed(0)
    create_actor, _, _, voc_path, ckpt_path, tokenizer = models["gru"]

    # Create vocabulary
    with open(voc_path, "r") as f:
        tokens = f.read().splitlines()
    tokens_dict = dict(zip(tokens, range(len(tokens))))
    vocabulary = SMILESVocabulary.create_from_dict(
        tokens_dict,
        start_token="GO",
        end_token="EOS",
        tokenizer=tokenizer,
    )
    length_vocabulary = len(vocabulary)

    # Create policy
    ckpt = torch.load(ckpt_path, map_location=device)
    policy_train, policy_inference = create_actor(length_vocabulary)
    policy_train.load_state_dict(adapt_state_dict(ckpt, policy_train.state_dict()))
    policy_inference.load_state_dict(
        adapt_state_dict(ckpt, policy_inference.state_dict())
    )
    policy_train = policy_train.to(device)
    policy_inference = policy_inference.to(device)

    # Create environment
    env = SMILESEnv(
        start_token=vocabulary.start_token_index,
        end_token=vocabulary.end_token_index,
        length_vocabulary=length_vocabulary,
        device=device,
        batch_size=batch_size,
    )
    env = TransformedEnv(env)
    primers = policy_train.rnn_spec.expand(batch_size)
    env.append_transform(TensorDictPrimer(primers))
    env.append_transform(InitTracker())

    # Sample smiles
    smiles = generate_complete_smiles(
        environment=env,
        vocabulary=vocabulary,
        policy_sample=policy_inference,
        policy_evaluate=policy_train,
        promptsmiles=promptsmiles,
        promptsmiles_optimize=promptsmiles_optimize,
        promptsmiles_shuffle=promptsmiles_shuffle,
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
    if finished.all():
        assert done.sum() >= batch_size
        assert done.sum() == batch_size

    assert (smiles["observation"][:, 0] == env.start_token).all()
    assert (smiles["action"][terminated] == env.end_token).all()
