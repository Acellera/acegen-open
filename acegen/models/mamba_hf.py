import warnings
from copy import deepcopy

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import ActorValueOperator, ProbabilisticActor

try:
    import transformers
    from transformers import MambaConfig, MambaForCausalLM

    _has_transformers = True
except ImportError as err:
    _has_transformers = False
    TRANSFORMERS_ERR = err


class Mamba(nn.Module):
    """GPT2 model for language modeling. This model is a simple wrapper around the HuggingFace Mamba."""

    def __init__(self, config=None):
        if not _has_transformers:
            raise RuntimeError(
                "transformers library not found, please install with pip install transformers."
            ) from TRANSFORMERS_ERR

        super(Mamba, self).__init__()

        # Define model
        self.config = config
        if config is not None:
            self.feature_extractor = MambaForCausalLM(config)
        else:
            self.feature_extractor = None

        # Start in evaluation mode
        self._train_mode = False
        self.forward = self._recurrent_forward

    @property
    def train_mode(self):
        return self._train_mode

    def set_train_mode(self, train_mode: bool = True):
        if train_mode is self._train_mode:
            return self
        # config = deepcopy(self.config)
        # config.use_cache = False
        out = Mamba()  ## NOTE Creating unsynced models???
        out.feature_extractor = self.feature_extractor
        out.feature_extractor.use_cache = False
        out._train_mode = train_mode
        out.forward = out._global_forward
        return out

    def _recurrent_forward(self, observation):
        observation = observation.reshape(-1, 1)
        out = self.feature_extractor(
            input_ids=observation,
        ).logits.squeeze()
        return out

    def _global_forward(self, sequence):
        out = self.feature_extractor(
            input_ids=sequence,
        ).logits
        return out


def define_mamba_configuration(
    vocabulary_size: int,
    hidden_size: int = 768,
    state_size: int = 16,
    num_hidden_layers: int = 32,
    layer_norm_epsilon: float = 1e-5,
    pad_token_id: int = 0,
    bos_token_id: int = 0,
    eos_token_id: int = 0,
    expand: int = 2,
    conv_kernel: int = 4,
    use_bias: bool = False,
    use_conv_bias: bool = True,
    hidden_act: str = "silu",
    initializer_range: float = 0.1,
    residual_in_fp32: bool = True,
    time_step_scale: float = 1.0,
    time_step_min: float = 0.001,
    time_step_max: float = 0.1,
    time_step_init_scheme: str = "random",
    time_step_floor: float = 0.0001,
    rescale_prenorm_residual: bool = False,
    use_cache: bool = True,
):
    """Define a Mamba configuration.

    This function is a simple wrapper around the HuggingFace MambaConfig, allowing to specify relevant parameters.
    """
    # Define model
    config = MambaConfig()

    # Adjust model parameters
    config.vocab_size = vocabulary_size
    config.hidden_size = hidden_size
    config.state_size = state_size
    config.num_hidden_layers = num_hidden_layers
    config.layer_norm_epsilon = layer_norm_epsilon
    config.pad_token_id = pad_token_id
    config.bos_token_id = bos_token_id
    config.eos_token_id = eos_token_id
    config.expand = expand
    config.conv_kernel = conv_kernel
    config.use_bias = use_bias
    config.use_conv_bias = use_conv_bias
    config.hidden_act = hidden_act
    config.initializer_range = initializer_range
    config.residual_in_fp32 = residual_in_fp32
    config.time_step_scale = time_step_scale
    config.time_step_min = time_step_min
    config.time_step_max = time_step_max
    config.time_step_init_scheme = time_step_init_scheme
    config.time_step_floor = time_step_floor
    config.rescale_prenorm_residual = rescale_prenorm_residual
    config.use_cache = use_cache

    return config


def create_mamba_actor(
    vocabulary_size: int,
    hidden_size: int = 128,
    state_size: int = 16,
    num_hidden_layers: int = 24,
    pad_token_index: int = 0,
    start_token_index: int = 0,
    end_token_index: int = 0,
    expand: int = 2,
    conv_kernel: int = 4,
    use_cache: bool = True,
    return_log_prob=True,
    **kwargs
):
    """Create a Mamba actor for language modeling."""
    config = define_mamba_configuration(
        vocabulary_size=vocabulary_size,
        hidden_size=hidden_size,
        state_size=state_size,
        num_hidden_layers=num_hidden_layers,
        pad_token_id=pad_token_index,
        bos_token_id=start_token_index,
        eos_token_id=end_token_index,
        expand=expand,
        conv_kernel=conv_kernel,
        use_cache=use_cache,
    )
    # Define transformer
    lm = Mamba(config)

    # Wrap the transformer in a TensorDictModule to make TensorDict compatible
    policy_training = TensorDictModule(
        lm.set_train_mode(True),
        in_keys=["sequence"],
        out_keys=["logits"],
    )
    policy_inference = TensorDictModule(
        lm,
        in_keys=["observation"],
        out_keys=["logits"],
    )

    # To make the actor probabilities, wrap the policy in a ProbabilisticActor
    # This module will take care of sampling and computing log_probabilities
    probabilistic_policy_training = ProbabilisticActor(
        module=policy_training,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )
    probabilistic_policy_inference = ProbabilisticActor(
        module=policy_inference,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )
    return probabilistic_policy_training, probabilistic_policy_inference
