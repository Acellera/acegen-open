from pathlib import Path

import torch
from acegen.models import adapt_state_dict
from acegen.models.gpt2 import create_gpt2_actor
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.vocabulary import SMILESVocabulary
from rdkit.Chem import AllChem, QED
from transformers import GPT2Config, GPT2Model

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
    "start_token": vocabulary.start_token_index,
    "end_token": vocabulary.end_token_index,
    "length_vocabulary": len(vocabulary),
    "batch_size": 4,
    "device": device,
}


env = SMILESEnv(**env_kwargs)

# Define model
probabilistic_policy, _ = create_gpt2_actor(len(vocabulary))

ckpt = torch.load(
    Path(__file__).resolve().parent.parent.parent / "priors" / "gpt2_enamine_real.ckpt"
)

probabilistic_policy.load_state_dict(
    adapt_state_dict(ckpt, probabilistic_policy.state_dict())
)

probabilistic_policy.to(device)
data = generate_complete_smiles(policy=probabilistic_policy, environment=env)
smiles_str = [
    vocabulary.decode(smi.cpu().numpy(), ignore_indices=[0]) for smi in data["action"]
]
print(smiles_str)


def evaluate_mol(smiles: str):
    mol = AllChem.MolFromSmiles(smiles)
    if mol:
        return True
    else:
        return False


valid = [evaluate_mol(s) for s in smiles_str]
print(sum(valid))
