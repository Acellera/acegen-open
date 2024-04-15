import logging
import tarfile
from importlib import resources
from pathlib import Path
from functools import partial

from acegen.models.gpt2 import (
    create_gpt2_actor,
    create_gpt2_actor_critic,
    create_gpt2_critic,
)
from acegen.models.gru import (
    create_gru_actor,
    create_gru_actor_critic,
    create_gru_critic,
)
from acegen.models.lstm import (
    create_lstm_actor,
    create_lstm_actor_critic,
    create_lstm_critic,
)
from acegen.models.utils import adapt_state_dict
from acegen.vocabulary.tokenizers import SMILESTokenizer, SMILESTokenizer2

try:
    from clms.models.vocabulary import AsciiSMILESTokenizer
    from acegen.models.clms import (
        create_clm_actor,
        create_clm_actor_critic,
        create_clm_critic
    )
    clms_available = True
except: 
    print("CLMS models not available.")
    clms_available = False
    pass

def extract(path):
    """Extract tarfile if it exists."""
    if not path.exists():
        tar_path = path.with_suffix(".tar.gz")
        if tar_path.exists():
            logging.info("Extracting model checkpoint...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall()
                return path
        else:
            raise FileNotFoundError(f"File {path} not found.")
    else:
        return path

def gru_model_factory(*args, **kwargs):
    return (
        create_gru_actor, 
        create_gru_critic,
        create_gru_actor_critic,
        resources.files("acegen.priors") / "chembl_filtered_vocabulary.txt",
        resources.files("acegen.priors") / "gru_chembl_filtered.ckpt",
        SMILESTokenizer(),
    )

def lstm_model_factory(*args, **kwargs):
    return (
        create_lstm_actor,
        create_lstm_critic,
        create_lstm_actor_critic,
        resources.files("acegen.priors") / "chembl_vocabulary.txt",
        resources.files("acegen.priors") / "lstm_chembl.ckpt",
        SMILESTokenizer(),
    )

def gpt2_model_factory(*args, **kwargs):
    return (
        create_gpt2_actor,
        create_gpt2_critic,
        create_gpt2_actor_critic,
        resources.files("acegen.priors") / "enamine_real_vocabulary.txt",
        extract(resources.files("acegen.priors") / "gpt2_enamine_real.ckpt"),
        SMILESTokenizer2(),
    )

# Default models
models = {
    "gru": gru_model_factory,
    "lstm": lstm_model_factory,
    "gpt2": gpt2_model_factory,
}

# Add CLM models if available
if clms_available:
    def clm_model_factory(cfg, *args, **kwargs):
        return (
            partial(create_clm_actor, cfg),
            partial(create_clm_critic, cfg),
            partial(create_clm_actor_critic, cfg),
            resources.files("acegen.priors") / "ascii.pt",
            None,
            AsciiSMILESTokenizer(),
        )

    models["clm"] = clm_model_factory

    

