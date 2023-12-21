from typing import Protocol

import numpy as np


class Tokenizer(Protocol):
    """An interface for handling tokenizing/un-tokenizing SMILES strings.

    Any tokenizer should implement this interface.
    """

    def tokenize(self, smiles: str) -> list[str]: ...

    def untokenize(self, tokens: tuple[str]) -> str: ...


class Vocabulary(Protocol):
    """An interface for handling encoding/decoding from SMILES to an array of indices.

    Any vocabulary should implement this interface.
    """

    def encode(self, smiles: tuple[str]) -> np.ndarray: ...

    def decode(self, vocab_index: np.ndarray, ignore_indices=()) -> list[str]: ...
