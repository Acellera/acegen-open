import warnings

def adapt_state_dict(source_state_dict: dict, target_state_dict: dict):
    """Adapt the source state dict to the target state dict.

    This is useful when loading a model checkpoint from a different model.
    It will only work if the source and target models have the same number of parameters and the
    same parameter shapes.

    Args:
        source_state_dict (dict): The source state dict.
        target_state_dict (dict): The target state dict.
    """
    for key, value in source_state_dict.items():
        if key in target_state_dict:
            target_state_dict[key] = value
        else:
            warnings.warn(f"Warning: {key} not found in target state dict.", RuntimeWarning)

    return target_state_dict
