from .ppo import create_shared_ppo_models
from .sac import create_sac_models


def get_model_factory(name):
    """Returns a function that creates a model."""
    if name == "ppo":
        return create_shared_ppo_models
    elif name == "sac":
        return create_sac_models
    else:
        raise ValueError(f"Unknown model name: {name}")