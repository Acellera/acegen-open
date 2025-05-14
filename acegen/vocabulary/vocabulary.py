from copy import deepcopy
from pathlib import Path

import numpy as np

import torch

from acegen.vocabulary.base import BaseTokenizer, BaseVocabulary


class Vocabulary(BaseVocabulary):
    """A class for handling encoding/decoding from strings to an array of indices.

    Args:
        start_token (str, optional): The start token. Defaults to "GO".
        end_token (str, optional): The end token. Defaults to "EOS".
        tokenizer (BaseTokenizer, optional): A tokenizer to use for tokenizing strings. Defaults to None.
            Any class that implements the tokenize and untokenize methods can be used.

    Examples:
        >>> from acegen.vocabulary import Vocabulary
        >>> chars = ["(", ")", "1", "=", "C", "N", "O"]

        >>> vocabulary = Vocabulary()
        >>> vocabulary.add_characters(chars)

        >>> tokens_dict = dict(zip(chars + ["EOS", "GO"], range(len(chars) + 2)))
        >>> vocabulary = Vocabulary.create_from_dict(tokens_dict)

        >>> state_dict = Vocabulary.state_dict()
        >>> vocabulary2 = Vocabulary()
        >>> vocabulary2.load_state_dict(state_dict)
    """

    def __init__(
        self,
        start_token: str = "GO",
        end_token: str = "EOS",
        tokenizer: BaseTokenizer = None,
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
        self, string: str, with_start: bool = True, with_end: bool = True
    ) -> np.ndarray:
        """Takes a list of characters (eg '[NH]') and encodes to array of indices.

        Args:
            string (str): The string to encode.

        Returns:
            np.ndarray: An array of indices corresponding to the string.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not set. Please set a valid tokenizer first."
                "Any class that implements the Tokenizer interface can be used."
            )

        char_list = self.tokenizer.tokenize(string)
        if with_start:
            char_list = [self.start_token] + char_list
        if with_end:
            char_list = char_list + [self.end_token]
        string_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            string_matrix[i] = self.vocab[char]
        return string_matrix

    def decode(self, encoded_string, ignore_indices=()):
        """Takes an array of indices and returns the corresponding string.

        Args:
            encoded_string (np.ndarray): An array of indices corresponding to a string.
            ignore_indices (tuple, optional): Indices to ignore. Defaults to ().


        Returns:
            str: The decoded string.
        """
        chars = []
        for i in encoded_string:
            if i in ignore_indices:
                continue
            if i == self.vocab[self.start_token]:
                continue
            if i == self.vocab[self.end_token]:
                break
            chars.append(self.reversed_vocab[i])
        
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not set. Please set a valid tokenizer first."
                "Any class that implements the Tokenizer interface can be used."
            )
            
        string = self.tokenizer.untokenize(chars)
        return string

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
    def create_from_strings(
        cls,
        strings_list: list[str],
        tokenizer: BaseTokenizer,
        start_token: str = "GO",
        end_token: str = "EOS",
        special_tokens: list = (),
    ):
        """Creates a vocabulary form a list of strings_list.

        Args:
            strings_list (list[str]): A list of strings to create the vocabulary from.
            tokenizer (BaseTokenizer): A tokenizer to use for tokenizing strings.
            start_token (str, optional): The start token. Defaults to "GO".
            end_token (str, optional): The end token. Defaults to "EOS".
            special_tokens (list, optional): A list of special tokens. Defaults to ().

        Returns:
            Vocabulary: A vocabulary class with the tokens from the strings.
        """
        vocabulary = cls(
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
        )
        tokens = set()
        for string in strings_list:
            tokens.update(vocabulary.tokenizer.tokenize(string))
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
        tokenizer: BaseTokenizer = None,
    ):
        """Creates a vocabulary from a dictionary mapping characters to indices.

        The dictionary should map characters to indices and should include the start and end tokens.

        Args:
            vocab (dict[str, int]): A dictionary mapping characters to indices.
            start_token (str, optional): The start token. Defaults to "GO".
            end_token (str, optional): The end token. Defaults to "EOS".
            tokenizer (BaseTokenizer, optional): A tokenizer to use for tokenizing strings. Defaults to None.
                Any class that implements the tokenize and untokenize methods can be used.

        Returns:
            Vocabulary: A vocabulary created from the dictionary.
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

    @classmethod
    def load(
        cls,
        voc_path: Path,
        start_token: str = "GO",
        end_token: str = "EOS",
        tokenizer: BaseTokenizer = None,
    ):
        """Loads a vocabulary from a file.

        Depending on file extension the vocabulary will be loaded from a text file or a cktp file.

        Args:
            vocab (dict[str, int]): A dictionary mapping characters to indices.
            start_token (str, optional): The start token. Defaults to "GO".
            end_token (str, optional): The end token. Defaults to "EOS".
            tokenizer (BaseTokenizer, optional): A tokenizer to use for tokenizing strings. Defaults to None.
                Any class that implements the tokenize and untokenize methods can be used.

        Returns:
            Vocabulary: A vocabulary loaded from the file.
        """
        if isinstance(voc_path, str):
            voc_path = Path(voc_path)

        if voc_path.suffix in [".ckpt", ".pt"]:
            tokens = torch.load(voc_path)
            vocabulary = cls()
            vocabulary.load_state_dict(tokens)
            vocabulary.tokenizer = tokenizer
            return vocabulary

        elif voc_path.suffix == ".txt":
            with open(voc_path, "r") as f:
                tokens = f.read().splitlines()
            tokens_dict = dict(zip(tokens, range(len(tokens))))
            vocabulary = Vocabulary.create_from_dict(
                tokens_dict,
                start_token=start_token,
                end_token=end_token,
                tokenizer=tokenizer,
            )
            return vocabulary

        elif voc_path.suffix == ".json":
            raise NotImplementedError

        else:
            raise ValueError(
                "File type not supported. Please use a .txt, .ckpt, or .pt file."
            )
