from .ppo_de_novo import create_shared_ppo_models


def get_model_factory(name):
    """Returns a function that creates a model."""
    if name == "ppo_reinvent":
        return create_shared_ppo_models
    else:
        raise ValueError(f"Unknown model name: {name}")