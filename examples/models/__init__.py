from .ppo import create_shared_ppo_models
from .sac import create_sac_models
from .dqn import create_dqn_models
from .reinvent import create_reinvent_model


def get_model_factory(name):
    """Returns a function that creates a model."""
    if name == "ppo":
        return create_shared_ppo_models
    elif name == "sac":
        return create_sac_models
    elif name == "dqn":
        return create_dqn_models
    elif name == "reinvent":
        return create_reinvent_model
    else:
        raise ValueError(f"Unknown model name: {name}")
