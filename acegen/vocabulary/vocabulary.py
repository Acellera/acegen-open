import numpy as np

from acegen.vocabulary.base import Tokenizer, Vocabulary


class SMILESVocabulary(Vocabulary):
    """A class for handling encoding/decoding from SMILES to an array of indices."""

    def __init__(
        self,
        start_token: str = "GO",
        start_token_index: int = 0,
        end_token: str = "EOS",
        end_token_index: int = 1,
        max_length: int = 140,
        tokenizer: Tokenizer = None,
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
        self.tokenizer = tokenizer

    def encode(self, smiles):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices."""
        if self.tokenizer is None:
            raise ValueError(
                "Tokenizer not set. Please set a valid tokenizer first."
                "Any class that implements the Tokenizer interface can be used."
            )

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
