from copy import deepcopy

import numpy as np

from acegen.vocabulary.base import Tokenizer, Vocabulary

SMILES_TOKENS = [
    ".",
    "/",
    "\\",
    "@",
    "%",
    "*",
    "=",
    ":",
    "#",
    ">",
    "+",
    "-",
    "<UNK>",
]


class SMILESVocabulary(Vocabulary):
    """A class for handling encoding/decoding from SMILES to an array of indices.

    Args:
        start_token (str, optional): The start token. Defaults to "GO".
        end_token (str, optional): The end token. Defaults to "EOS".
        tokenizer (Tokenizer, optional): A tokenizer to use for tokenizing the SMILES. Defaults to None.
            Any class that implements the tokenize and untokenize methods can be used.

    Examples:
        >>> from acegen.vocabulary import SMILESVocabulary
        >>> chars = ["(", ")", "1", "=", "C", "N", "O"]

        >>> vocabulary = SMILESVocabulary()
        >>> vocabulary.add_characters(chars)

        >>> tokens_dict = dict(zip(chars + ["EOS", "GO"], range(len(chars) + 2)))
        >>> vocabulary = SMILESVocabulary.create_from_dict(tokens_dict)

        >>> state_dict = SMILESVocabulary.state_dict()
        >>> vocabulary2 = SMILESVocabulary()
        >>> vocabulary2.load_state_dict(state_dict)
    """

    def __init__(
        self,
        start_token: str = "GO",
        end_token: str = "EOS",
        tokenizer: Tokenizer = None,
        special_tokens: list = (),
    ):
        self.start_token = start_token
        self.end_token = end_token
        self.special_tokens = [end_token, start_token]
        self.special_tokens += list(set(special_tokens))
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.tokenizer = tokenizer
        self.start_token_index = self.vocab[start_token]
        self.end_token_index = self.vocab[end_token]
        self.special_indices = [self.end_token_index, self.start_token_index]

    def encode(
        self, smiles: str, with_start: bool = True, with_end: bool = True
    ) -> np.ndarray:
        """Takes a list of characters (eg '[NH]') and encodes to array of indices.

        Args:
            smiles (str): The SMILES string to encode.

        Returns:
            np.ndarray: An array of indices corresponding to the SMILES string.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not set. Please set a valid tokenizer first."
                "Any class that implements the Tokenizer interface can be used."
            )

        char_list = self.tokenizer.tokenize(smiles)
        if with_start:
            char_list = [self.start_token] + char_list
        if with_end:
            char_list = char_list + [self.end_token]
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, encoded_smiles, ignore_indices=()):
        """Takes an array of indices and returns the corresponding SMILES.

        Args:
            encoded_smiles (np.ndarray): An array of indices corresponding to a SMILES string.
            ignore_indices (tuple, optional): Indices to ignore. Defaults to ().


        Returns:
            str: The decoded SMILES string.
        """
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
        return smiles

    def add_characters(self, chars):
        """Adds characters to the vocabulary.

        Args:
            chars (list[str]): A list of characters to add to the vocabulary.
        """
        for char in chars:
            if char not in self.chars:
                self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = self.special_tokens + char_list
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
        tokenizer: Tokenizer,
        start_token: str = "GO",
        end_token: str = "EOS",
    ):
        """Creates a vocabulary for the SMILES syntax.

        Args:
            smiles_list (list[str]): A list of SMILES strings to create the vocabulary from.
            tokenizer (Tokenizer): A tokenizer to use for tokenizing the SMILES.
            start_token (str, optional): The start token. Defaults to "GO".
            end_token (str, optional): The end token. Defaults to "EOS".

        Returns:
            SMILESVocabulary: A vocabulary for the SMILES syntax.
        """
        vocabulary = cls(
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer,
        )
        tokens = set()
        for smi in smiles_list:
            tokens.update(vocabulary.tokenizer.tokenize(smi))
        vocabulary.add_characters(sorted(tokens))
        vocabulary.start_token_index = vocabulary.vocab[start_token]
        vocabulary.end_token_index = vocabulary.vocab[end_token]
        return vocabulary

    @classmethod
    def create_from_dict(
        cls,
        vocab: dict[str, int],
        start_token: str = "GO",
        end_token: str = "EOS",
        tokenizer: Tokenizer = None,
    ):
        """Creates a vocabulary from a dictionary mapping characters to indices.

        The dictionary should map characters to indices and should include the start and end tokens.

        Args:
            vocab (dict[str, int]): A dictionary mapping characters to indices.
            start_token (str, optional): The start token. Defaults to "GO".
            end_token (str, optional): The end token. Defaults to "EOS".
            tokenizer (Tokenizer, optional): A tokenizer to use for tokenizing the SMILES. Defaults to None.
                Any class that implements the tokenize and untokenize methods can be used.

        Returns:
            SMILESVocabulary: A vocabulary for the SMILES syntax.
        """
        vocabulary = cls(
            start_token=start_token,
            end_token=end_token,
        )
        vocabulary.start_token_index = vocab[start_token]
        vocabulary.end_token_index = vocab[end_token]
        vocabulary.tokenizer = tokenizer
        vocabulary.vocab_size = len(vocab)
        vocabulary.vocab = vocab
        vocabulary.reversed_vocab = {v: k for k, v in vocabulary.vocab.items()}
        vocabulary.chars = list(vocabulary.vocab.keys())
        vocabulary.special_tokens = [end_token, start_token]
        vocabulary.additional_chars = {
            char for char in vocabulary.chars if char not in vocabulary.special_tokens
        }
        return vocabulary

    def state_dict(self):
        """Returns the state of the vocabulary."""
        state_dict = deepcopy(self.vocab)
        state_dict["start_token"] = self.start_token
        state_dict["start_token_index"] = self.start_token_index
        state_dict["end_token"] = self.end_token
        state_dict["end_token_index"] = self.end_token_index
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the state of the vocabulary."""
        self.start_token = state_dict.pop("start_token")
        self.end_token = state_dict.pop("end_token")
        self.start_token_index = state_dict.pop("start_token_index")
        self.end_token_index = state_dict.pop("end_token_index")
        self.vocab = state_dict
        self.vocab_size = len(self.vocab)
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.chars = list(self.vocab.keys())
        self.special_tokens = [self.end_token, self.start_token]
        self.additional_chars = {
            char for char in self.chars if char not in self.special_tokens
        }
