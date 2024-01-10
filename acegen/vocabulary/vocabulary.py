import copy
import re

import numpy as np

from acegen.vocabulary.base import Tokenizer, Vocabulary


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters."""
    br = re.compile("Br")
    cl = re.compile("Cl")
    string = br.sub("R", string)
    string = cl.sub("L", string)

    return string


# class SMILESTokenizer(Tokenizer):
#     """One possible implementation of the Tokenizer interface.
#
#     Deals with the tokenization and untokenization of SMILES.
#     """
#
#     def __init__(self, start_token: str = "GO", end_token: str = "EOS"):
#         self.start_token = start_token
#         self.end_token = end_token
#
#     def tokenize(self, smiles: str) -> list[str]:
#         """Takes a SMILES and return a list of characters/tokens."""
#         regex = "(\[[^\[\]]{1,6}\])"
#         smiles = replace_halogen(smiles)
#         char_list = re.split(regex, smiles)
#         tokenized = []
#         tokenized.append(self.start_token)
#         for char in char_list:
#             if char.startswith("["):
#                 tokenized.append(char)
#             else:
#                 chars = list(char)
#                 [tokenized.append(unit) for unit in chars]
#         tokenized.append(self.end_token)
#         return tokenized
#
#     def untokenize(self, tokens: list[str]) -> str:
#         """Untokenizes a SMILES string."""
#         smi = ""
#         for i, token in enumerate(tokens):
#             if token == self.end_token:
#                 break
#             if token == self.start_token and i == 0:
#                 continue
#             smi += token
#         return smi


class SMILESTokenizer(Tokenizer):
    """Deals with the tokenization and untokenization of SMILES."""

    GRAMMAR = "SMILES"
    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)"),
        "atom": re.compile(r"[a-zA-Z]"),
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def __init__(self, start_token: str = "GO", end_token: str = "EOS"):
        self.GRAMMAR = copy.deepcopy(self.GRAMMAR)
        self.REGEXPS = copy.deepcopy(self.REGEXPS)
        self.REGEXP_ORDER = copy.deepcopy(self.REGEXP_ORDER)
        self.start_token = start_token
        self.end_token = end_token

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
            tokens = [self.start_token] + tokens + [self.end_token]
        return tokens

    def untokenize(self, tokens, **kwargs):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == self.end_token:
                break
            if token != self.start_token:
                smi += token
        return smi


class SMILESVocabulary(Vocabulary):
    """A class for handling encoding/decoding from SMILES to an array of indices."""

    def __init__(
        self,
        start_token: str = "GO",
        start_token_index: int = 0,
        end_token: str = "EOS",
        end_token_index: int = 1,
        max_length: int = 140,
    ):
        self.start_token = start_token
        self.end_token = end_token
        self.special_tokens = [end_token, start_token]
        special_indices = [end_token_index, start_token_index]
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, special_indices))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        self.tokenizer = SMILESTokenizer(self.start_token, self.end_token)

    def encode(self, smiles):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices."""
        char_list = self.tokenizer.tokenize(smiles)
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, encoded_smiles, ignore_indices=()):
        """Takes an array of indices and returns the corresponding SMILES."""
        chars = []
        for i in encoded_smiles:
            if i in ignore_indices:
                continue
            if i == self.vocab[self.start_token]:
                continue
            if i == self.vocab[self.end_token]:
                break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def add_characters(self, chars):
        """Adds characters to the vocabulary."""
        for char in chars:
            if char not in self.chars:
                self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)

    @classmethod
    def create_from_smiles(
        cls,
        smiles_list: list[str],
        start_token: str = "GO",
        start_token_index: int = 0,
        end_token: str = "EOS",
        end_token_index: int = 1,
        max_length: int = 140,
    ):
        """Creates a vocabulary for the SMILES syntax."""
        vocabulary = cls(
            start_token=start_token,
            start_token_index=start_token_index,
            end_token=end_token,
            end_token_index=end_token_index,
            max_length=max_length,
        )
        tokens = set()
        for smi in smiles_list:
            tokens.update(vocabulary.tokenizer.tokenize(smi))
        vocabulary = cls()
        vocabulary.add_characters(sorted(tokens))
        return vocabulary

    @classmethod
    def create_from_dict(
        cls,
        vocab: dict[str, int],
        start_token: str = "GO",
        end_token: str = "EOS",
        max_length: int = 140,
    ):
        """Creates a vocabulary from a dictionary.

        The dictionary should map characters to indices and should include the start and end tokens.
        """
        vocabulary = cls(
            start_token=start_token,
            end_token=end_token,
            max_length=max_length,
        )
        vocabulary.vocab_size = len(vocab)
        vocabulary.vocab = vocab
        vocabulary.reversed_vocab = {v: k for k, v in vocabulary.vocab.items()}
        vocabulary.chars = list(vocabulary.vocab.keys())
        vocabulary.special_tokens = [end_token, start_token]
        vocabulary.additional_chars = {
            char for char in vocabulary.chars if char not in vocabulary.special_tokens
        }
        return vocabulary
