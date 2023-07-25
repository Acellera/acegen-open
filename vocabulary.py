# coding=utf-8
"""
Vocabulary helper classes.

From https://github.com/MolecularAI/reinvent-models/blob/main/reinvent_models/reinvent_core/models/vocabulary.py
"""

import re
import numpy as np


class Vocabulary:
    """Stores the tokens and allows their conversion to vocabulary indexes."""

    def __init__(self, tokens=None, starting_id=0):
        self._tokens = {}
        self._current_id = starting_id
        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    @property
    def vocab_size(self):
        """Vocabulary size"""
        return len(self._tokens) // 2

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def add(self, token):
        """Adds a token."""
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """Adds many tokens."""
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self._tokens

    def __len__(self):
        return len(self._tokens) // 2

    def encode(self, tokens):
        """Encodes a list of tokens as vocabulary indexes."""
        vocab_index = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            vocab_index[i] = self._tokens[token]
        return vocab_index

    def decode(self, vocab_index):
        """Decodes a vocabulary index matrix to a list of tokens."""
        tokens = []
        for idx in vocab_index:
            tokens.append(self[idx])
        return tokens

    def _add(self, token, idx):
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def tokens(self):
        """Returns the tokens from the vocabulary"""
        return [t for t in self._tokens if isinstance(t, str)]


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
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi


def create_vocabulary(smiles_list, tokenizer):
    """Creates a vocabulary for the SMILES syntax."""
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    vocabulary.update(["<pad>", "$", "^"] + sorted(tokens))
    return vocabulary


class DeNovoVocabulary:
    def __init__(self, vocabulary, tokenizer):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer

    def encode_smiles(self, smile, with_begin_and_end=True):
        """Encodes a SMILE from str to np.array."""
        return self.vocabulary.encode(
            self.tokenizer.tokenize(smile, with_begin_and_end)
        )

    def decode_smiles(self, encoded_smile):
        """Decodes a SMILE from np.array to str."""
        return self.tokenizer.untokenize(self.vocabulary.decode(encoded_smile))

    def encode_token(self, token):
        """Encodes token from str to int"""
        return self.vocabulary.encode([str(token)])[0]

    def decode_token(self, token):
        """Decodes token from int to str"""
        return self.vocabulary.decode([int(token)])[0]

    def remove_start_and_end_tokens(self, smile):
        """Remove start and end tokens from a SMILE"""
        return self.tokenizer.untokenize(smile)

    def count_tokens(self, smile):
        return len(self.tokenizer.tokenize(smile))

    def __len__(self):
        """Returns the length of the vocabulary."""
        return len(self.vocabulary)

    @classmethod
    def from_list(cls, smiles_list):
        """Creates the vocabulary from a list of smiles."""
        tokenizer = SMILESTokenizer()
        vocabulary = create_vocabulary(smiles_list, tokenizer)
        return DeNovoVocabulary(vocabulary, tokenizer)

