import re
from typing import Protocol

import numpy as np


class SMILESTokenizer:
    """Deals with the tokenization and untokenization of SMILES."""

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)"),
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""

        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            data = re.sub(r"<pad>", "", data)
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for i, token in enumerate(tokens):
            if token == "$":
                break
            if token == "^" and i == 0:
                continue
            smi += token
        return smi


class Vocabulary(Protocol):
    """An interface for handling encoding/decoding from SMILES to an array of indices"""

    def encode(self, smiles: list[str]) -> np.ndarray: ...

    def decode(self, vocab_index: np.ndarray, ignore_indices=[]) -> list[str]: ...


class SMILESVocabulary(Vocabulary):
    """A class for handling encoding/decoding from SMILES to an array of indices"""

    def __init__(self, init_from_file=None, max_length=140):
        self.special_tokens = ["EOS", "GO"]
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file:
            self.init_from_file(init_from_file)

    def encode(self, smiles):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        char_list = self.tokenize(smiles)
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, encoded_smiles, ignore_indices=[]):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in encoded_smiles:
            if i in ignore_indices:
                continue
            if i == self.vocab["GO"]:
                continue
            if i == self.vocab["EOS"]:
                break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = "(\[[^\[\]]{1,6}\])"
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        tokenized.append("GO")
        for char in char_list:
            if char.startswith("["):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append("EOS")
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, "r") as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)

    @classmethod
    def create_from_smiles(cls, smiles_list: list[str], tokenizer=SMILESTokenizer()):
        """Creates a vocabulary for the SMILES syntax."""
        tokens = set()
        for smi in smiles_list:
            tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

        vocabulary = cls()
        vocabulary.add_characters(["<pad>", "$", "^"] + sorted(tokens))
        return vocabulary


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile("Br")
    cl = re.compile("Cl")
    string = br.sub("R", string)
    string = cl.sub("L", string)

    return string
