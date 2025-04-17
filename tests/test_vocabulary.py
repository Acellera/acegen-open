import re
from copy import copy

import pytest

from acegen.vocabulary.vocabulary import Vocabulary

single_smiles = "CC1=CC=CC=C1"
multiple_smiles = [
    "CCO",  # Ethanol (C2H5OH)
    "CCN(CC)CC",  # Triethylamine (C6H15N)
    "CC(=O)OC(C)C",  # Diethyl carbonate (C7H14O3)
    "CC(C)C",  # Isobutane (C4H10)
    "CC1=CC=CC=C1",  # Toluene (C7H8)
]
chars = ["(", ")", "1", "=", "C", "N", "O"]


class Tokenizer:
    def __init__(self, start_token: str = "GO", end_token: str = "EOS"):
        self.start_token = start_token
        self.end_token = end_token

    def tokenize(self, smiles: str) -> list[str]:
        regex = "(\[[^\[\]]{1,6}\])"
        char_list = re.split(regex, smiles)
        tokenized = [self.start_token]
        for char in char_list:
            if char.startswith("["):
                tokenized.append(char)
            else:
                [tokenized.append(unit) for unit in list(char)]
        tokenized.append(self.end_token)
        return tokenized
    
    def untokenize(self, tokens: list[int]) -> str:
        smiles = ""
        for t in tokens:
            if t == self.start_token:
                continue
            elif t == self.end_token:
                break
            else:
                smiles += t
        return smiles


def test_from_smiles():
    tokenizer = Tokenizer()
    vocabulary = Vocabulary.create_from_strings(multiple_smiles, tokenizer=tokenizer)
    assert len(vocabulary) > 0


def create_from_dict():
    tokens_dict = dict(zip(chars + ["EOS", "GO"], range(len(chars) + 2)))
    vocabulary = Vocabulary.create_from_dict(tokens_dict)
    assert len(vocabulary) > 0


def test_create_methods_match():
    tokenizer = Tokenizer()
    vocabulary = Vocabulary.create_from_strings(multiple_smiles, tokenizer=tokenizer)
    tokens_dict = copy(vocabulary.vocab)
    vocabulary2 = Vocabulary.create_from_dict(
        tokens_dict,
        start_token=vocabulary.start_token,
        end_token=vocabulary.end_token,
    )
    for obj1, obj2 in zip(vocabulary.__dict__.items(), vocabulary2.__dict__.items()):
        k1, v1 = obj1
        k2, v2 = obj2
        assert k1 == k2
        if k1 != "tokenizer":
            assert v1 == v2


def test_full_pipeline():
    tokens_dict = dict(zip(chars + ["EOS", "GO"], range(len(chars) + 2)))
    vocabulary = Vocabulary.create_from_dict(tokens_dict)

    with pytest.raises(
        RuntimeError,
        match="Tokenizer not set. Please set a valid tokenizer first."
        "Any class that implements the Tokenizer interface can be used.",
    ):
        tokens = vocabulary.encode(multiple_smiles[0])

    tokenizer = Tokenizer()
    vocabulary.tokenizer = tokenizer

    for smiles in multiple_smiles:
        tokens = vocabulary.encode(smiles)
        smiles2 = vocabulary.decode(tokens)
        assert smiles == smiles2


def test_state_dict():
    tokens_dict = dict(zip(chars + ["STOP", "START"], range(len(chars) + 2)))
    vocabulary = Vocabulary.create_from_dict(
        tokens_dict, start_token="START", end_token="STOP"
    )

    state_dict = vocabulary.state_dict()
    vocabulary2 = Vocabulary()
    vocabulary2.load_state_dict(state_dict)
    for obj1, obj2 in zip(vocabulary.__dict__.items(), vocabulary2.__dict__.items()):
        k1, v1 = obj1
        k2, v2 = obj2
        assert k1 == k2
        if k1 != "tokenizer":
            print(k1)
            assert v1 == v2
