from typing import Protocol, Sequence

import numpy as np


class BaseTokenizer(Protocol):
    """An interface for handling tokenizing/un-tokenizing SMILES strings.

    Any tokenizer should implement this interface.
    """

    def tokenize(self, smiles: str) -> list[str]: ...

    def untokenize(self, tokens: Sequence[str]) -> str: ...


class BaseVocabulary(Protocol):
    """An interface for handling encoding/decoding from SMILES to an array of indices.

    Any vocabulary should implement this interface.
    """

    def encode(self, smiles: str) -> np.ndarray: ...

    def decode(self, vocab_index: np.ndarray, ignore_indices=()) -> str: ...
