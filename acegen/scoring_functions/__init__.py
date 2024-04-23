from typing import Callable

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

    # Check it accepts a single smiles and returns typing number of a tensor
    if not isinstance(scoring_function("CCO"), (int, float, list, Tensor, ndarray)):
        raise ValueError(
            f"scoring_function must return a number, got {type(scoring_function('CCO'))}"
        )

    # Check it accepts a single smiles and returns a list of number or a tensor
    scores = scoring_function(["CCO", "CCC"])
    if not isinstance(scores, (list, Tensor, ndarray)):
        raise ValueError(
            f"scoring_function must return a list of number, got {type(scoring_function(['CCO', 'CCC']))}"
        )

    # If scores is a list, check that each element is a number
    if isinstance(scores, list):
        for score in scores:
            if not isinstance(score, (int, float)):
                raise ValueError(
                    f"scoring_function must return a list of number, got {type(scoring_function(['CCO', 'CCC']))}"
                )


def register_custom_scoring_function(name, scoring_function):
    """Register a custom scoring function.

    Example:
        >>> from acegen import register_custom_scoring_function, custom_scoring_functions
        >>> from my_module import my_scoring_function
        >>> register_custom_scoring_function("my_scoring_function", my_scoring_function)
        >>> custom_scoring_functions["my_scoring_function"]
    """
    check_scoring_function(scoring_function)
    custom_scoring_functions[name] = scoring_function

