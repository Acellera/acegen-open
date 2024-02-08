from pathlib import Path

import torch
from acegen.models.gpt import GPT2
from acegen.rl_env import SMILESEnv
from acegen.vocabulary import SMILESVocabulary
from tensordict.nn import TensorDictModule
from transformers import GPT2Config, GPT2Model
from torchrl.modules import ProbabilisticActor
from torchrl.envs import ExplorationType
from acegen.rl_env import generate_complete_smiles
from acegen.models import adapt_state_dict

# Get available device
device = (
    torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
)

# Load Vocabulary
ckpt = (
    Path(__file__).resolve().parent.parent.parent
    / "priors"
    / "enamine_real_vocabulary.txt"
)
with open(ckpt, "r") as f:
    tokens = f.read().splitlines()
tokens_dict = dict(zip(tokens, range(len(tokens))))
vocabulary = SMILESVocabulary.create_from_dict(
    tokens_dict, start_token="GO", end_token="EOS"
)


env_kwargs = {
    "start_token": vocabulary.vocab[vocabulary.start_token],
    "end_token": vocabulary.vocab[vocabulary.end_token],
    "length_vocabulary": len(vocabulary),
    "batch_size": 4,
    "device": device,
}

env = SMILESEnv(**env_kwargs)

# Define model
model_config = GPT2Config()

# Adjust model parameters
model_config.vocab_size = len(vocabulary)
model_config.n_positions = 100
model_config.n_head = 4
model_config.n_layer = 2
model_config.n_embd = 384
model_config.attn_pdrop = 0.1
model_config.embd_pdrop = 0.1
model_config.resid_pdrop = 0.1

policy = TensorDictModule(
    GPT2(model_config),
    in_keys=["context", "context_mask"],
    out_keys=["logits"],
)
probabilistic_policy = ProbabilisticActor(
    module=policy,
    in_keys=["logits"],
    out_keys=["action"],
    distribution_class=torch.distributions.Categorical,
    return_log_prob=True,
    default_interaction_type=ExplorationType.RANDOM,
)

ckpt = torch.load(
    Path(__file__).resolve().parent.parent.parent / "priors" / "gpt2_enamine_real.ckpt"
)

probabilistic_policy.load_state_dict(adapt_state_dict(ckpt, probabilistic_policy.state_dict()))

policy.to(device)
data = generate_complete_smiles(policy=probabilistic_policy, environment=env)
import ipdb; ipdb.set_trace()
