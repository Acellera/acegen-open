from pathlib import Path

from acegen.models import (
    create_gpt2_actor,
    create_gpt2_actor_critic,
    create_gpt2_critic,
    create_gru_actor,
    create_gru_actor_critic,
    create_gru_critic,
    create_lstm_actor,
    create_lstm_actor_critic,
    create_lstm_critic,
)
from acegen.rl_env.smiles_env import SMILESEnv
from acegen.vocabulary.tokenizers import SMILESTokenizer, SMILESTokenizer2
from acegen.vocabulary.vocabulary import SMILESVocabulary


model_mapping = {
    "gru": (
        create_gru_actor,  # create_actor_method
        create_gru_critic,  # create_critic_method
        create_gru_actor_critic,  # create_actor_critic_method (optional)
        Path(__file__).resolve().parent.parent
        / "priors"
        / "chembl_filtered_vocabulary.txt",  # vocabulary_file
        Path(__file__).resolve().parent.parent
        / "priors"
        / "gru_chembl_filtered.ckpt",  # weights_file
        SMILESTokenizer(),  # tokenizer (optional)
    ),
    "lstm": (
        create_lstm_actor,
        create_lstm_critic,
        create_lstm_actor_critic,
        Path(__file__).resolve().parent.parent / "priors" / "chembl_vocabulary.txt",
        Path(__file__).resolve().parent.parent / "priors" / "lstm_chembl.ckpt",
        SMILESTokenizer(),
    ),
    "gpt2": (
        create_gpt2_actor,
        create_gpt2_critic,
        create_gpt2_actor_critic,
        Path(__file__).resolve().parent.parent
        / "priors"
        / "enamine_real_vocabulary.txt",
        Path(__file__).resolve().parent.parent / "priors" / "gpt2_enamine_real.ckpt",
        SMILESTokenizer2(),
    ),
}
