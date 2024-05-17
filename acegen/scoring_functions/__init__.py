from typing import Callable
from importlib import import_module

from numpy import ndarray
from torch import Tensor

from acegen.scoring_functions.base import Task
from acegen.scoring_functions.chemistry import QED

custom_scoring_functions = {
    "QED": QED,
}


def check_scoring_function(scoring_function):
    """Check if the scoring function is a valid scoring function."""
    # Check it is a callable
    if not isinstance(scoring_function, Callable):
        raise ValueError(
            f"scoring_function must be a callable, got {type(scoring_function)}"
        )

    # Check it accepts a single smiles and returns a number, list, tensor or array
    if not isinstance(scoring_function("CCO"), (float, list, Tensor, ndarray)):
        raise ValueError(
            f"scoring_function must return a float, list, array or tensor, got {type(scoring_function('CCO'))}"
        )

    # Check it accepts multiple smiles and returns a list, a tensor or an array
    scores = scoring_function(["CCO", "CCC"])
    if not isinstance(scores, (list, Tensor, ndarray)):
        raise ValueError(
            f"scoring_function must return a list, array or tensor, got {type(scores)}"
        )

    # If scores is a list, check that each element is a float
    if isinstance(scores, list):
        for score in scores:
            if not isinstance(score, float):
                raise ValueError(
                    f"scoring_function must return a list of floats, got {type(score)}"
                )


def register_custom_scoring_function(name, scoring_function):
    """Register a custom scoring function.

    Example:
        >>> from acegen.scoring_functions import register_custom_scoring_function, custom_scoring_functions
        >>> from my_module import my_scoring_function
        >>> register_custom_scoring_function("my_scoring_function", my_scoring_function)
        >>> custom_scoring_functions["my_scoring_function"]
    """
    if isinstance(scoring_function, str):
        m, f = scoring_function.rsplit(".", 1)
        scoring_function = getattr(import_module(m), f)
    check_scoring_function(scoring_function)
    custom_scoring_functions[name] = scoring_function
