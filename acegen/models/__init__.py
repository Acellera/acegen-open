import logging
import tarfile
from importlib import resources
from pathlib import Path

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


def extract(path):
    """Extract tarfile if it exists."""
    if not Path.exists(path):
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


models = {
    "gru": (
        create_gru_actor,
        create_gru_critic,
        create_gru_actor_critic,
        resources.files("acegen.priors") / "chembl_filtered_vocabulary.txt",
        resources.files("acegen.priors") / "gru_chembl_filtered.ckpt",
        SMILESTokenizer(),
    ),
    "lstm": (
        create_lstm_actor,
        create_lstm_critic,
        create_lstm_actor_critic,
        resources.files("acegen.priors") / "chembl_vocabulary.txt",
        resources.files("acegen.priors") / "lstm_chembl.ckpt",
        SMILESTokenizer(),
    ),
    "gpt2": (
        create_gpt2_actor,
        create_gpt2_critic,
        create_gpt2_actor_critic,
        resources.files("acegen.priors") / "enamine_real_vocabulary.txt",
        extract(resources.files("acegen.priors") / "gpt2_enamine_real.ckpt"),
        SMILESTokenizer2(),
    ),
}
