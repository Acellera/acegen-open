import warnings

import torch
import torch.nn as nn
from packaging.version import Version
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import ActorValueOperator, MaskedCategorical, ProbabilisticActor

try:
    import transformers

    _has_transformers = True
except ImportError as err:
    _has_transformers = False
    TRANSFORMERS_ERR = err


def check_transformers():
    """Check installation and version of transformers."""
    if not _has_transformers:
        raise RuntimeError(
            "transformers library not found, please install with pip install transformers"
        ) from TRANSFORMERS_ERR
    if Version(transformers.__version__) != Version("4.24.0"):
        warnings.warn(
            f"Warning: The current version of transformers library ({transformers.__version__}) "
            f"may not be compatible with the default weights for the GPT-2 model used in AceGen. "
            f"If you intend to use the default weights provided in AceGen, please install transformers "
            f"version 4.24.0 using: `pip install transformers==4.24.0`."
        )


class GPT2(nn.Module):
    """GPT2 model for language modeling. This model is a simple wrapper around the HuggingFace GPT2Model."""

    def __init__(self, config=None):
        check_transformers()
        super(GPT2, self).__init__()

        # Define model
        if config is not None:
            self.feature_extractor = transformers.GPT2Model(config)
        else:
            self.feature_extractor = None

        # Start in evaluation mode
        self._train_mode = False

    @property
    def train_mode(self):
        return self._train_mode

    def set_train_mode(self, train_mode: bool = True):
        if train_mode is self._train_mode:
            return self
        out = GPT2()
        out.feature_extractor = self.feature_extractor
        out._train_mode = train_mode
        return out

    def forward(self, sequence, sequence_mask):

        out = self.feature_extractor(
            input_ids=sequence,
            attention_mask=sequence_mask.long(),
        ).last_hidden_state

        if self.train_mode is False:  # Data collection, return only last token
            obs_length = sequence_mask.sum(-1)
            out = out[torch.arange(len(out)), obs_length.to(torch.int64) - 1]

        return out


def define_gpt2_configuration(
    vocabulary_size: int,
    n_positions: int = 2048,
    n_head: int = 16,
    n_layer: int = 24,
    n_embd: int = 128,
    attn_pdrop: float = 0.1,
    embd_pdrop: float = 0.1,
    resid_pdrop: float = 0.1,
):
    """Define a GPT2 configuration.

    This function is a simple wrapper around the HuggingFace GPT2Config, allowing to specify relevant parameters.
    """
    # Check transformers library and version
    check_transformers()

    # Define model
    config = transformers.GPT2Config()

    # Adjust model parameters
    config.vocab_size = vocabulary_size
    config.n_positions = n_positions
    config.n_head = n_head
    config.n_layer = n_layer
    config.n_embd = n_embd
    config.attn_pdrop = attn_pdrop
    config.embd_pdrop = embd_pdrop
    config.resid_pdrop = resid_pdrop

    return config


def create_gpt2_actor(
    vocabulary_size: int,
    n_positions: int = 2048,
    n_head: int = 16,
    n_layer: int = 24,
    n_embd: int = 128,
    attn_pdrop: float = 0.1,
    embd_pdrop: float = 0.1,
    resid_pdrop: float = 0.1,
    action_mask_key: str = "action_mask",
    return_log_prob=True,
):
    """Create a GPT2 actor for language modeling."""
    config = define_gpt2_configuration(
        vocabulary_size,
        n_positions,
        n_head,
        n_layer,
        n_embd,
        attn_pdrop,
        embd_pdrop,
        resid_pdrop,
    )
    # Define transformer
    lm = GPT2(config)

    # Wrap the transformer in a TensorDictModule to make TensorDict compatible
    lm_training = TensorDictModule(
        lm.set_train_mode(True),
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )
    lm_inference = TensorDictModule(
        lm,
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )

    # Define final layer and also make it a TensorDictModule
    lm_head = TensorDictModule(
        nn.Linear(config.n_embd, vocabulary_size, bias=False),
        in_keys=["features"],
        out_keys=["logits"],
    )

    # Concatenate lm and head, similar to torch.nn.Sequential
    policy_training = TensorDictSequential(lm_training, lm_head)
    policy_inference = TensorDictSequential(lm_inference, lm_head)

    # Create optional mask for inference
    if action_mask_key:
        inf_keys = {"logits": "logits", "mask": action_mask_key}
        inf_dist = MaskedCategorical
    else:
        inf_keys = ["logits"]
        inf_dist = torch.distributions.Categorical

    # To make the actor probabilistic, wrap the policy in a ProbabilisticActor
    # This module will take care of sampling and computing log probabilities
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
        in_keys=inf_keys,
        out_keys=["action"],
        distribution_class=inf_dist,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )
    return probabilistic_policy_training, probabilistic_policy_inference


def create_gpt2_critic(
    vocabulary_size: int,
    n_positions: int = 2048,
    n_head: int = 16,
    n_layer: int = 24,
    n_embd: int = 128,
    attn_pdrop: float = 0.1,
    embd_pdrop: float = 0.1,
    resid_pdrop: float = 0.1,
    critic_value_per_action=False,
):
    """Create a GPT2 critic for language modeling."""
    config = define_gpt2_configuration(
        vocabulary_size,
        n_positions,
        n_head,
        n_layer,
        n_embd,
        attn_pdrop,
        embd_pdrop,
        resid_pdrop,
    )
    # Define transformer
    lm = GPT2(config)

    # Wrap the transformer in a TensorDictModule to make TensorDict compatible
    lm_training = TensorDictModule(
        lm.set_train_mode(True),
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )
    lm_inference = TensorDictModule(
        lm,
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )

    # Define final layer and also make it a TensorDictModule
    lm_head = TensorDictModule(
        nn.Linear(
            config.n_embd,
            vocabulary_size if critic_value_per_action else 1,
            bias=False,
        ),
        in_keys=["features"],
        out_keys=["action_value"] if critic_value_per_action else ["state_value"],
    )

    # Concatenate lm and head, similar to torch.nn.Sequential
    # Critic does not need to be probabilistic, so we can return directly
    critic_training = TensorDictSequential(lm_training, lm_head)
    critic_inference = TensorDictSequential(lm_inference, lm_head)
    return critic_training, critic_inference


def create_gpt2_actor_critic(
    vocabulary_size: int,
    n_positions: int = 2048,
    n_head: int = 16,
    n_layer: int = 24,
    n_embd: int = 128,
    attn_pdrop: float = 0.1,
    embd_pdrop: float = 0.1,
    resid_pdrop: float = 0.1,
    return_log_prob=True,
    critic_value_per_action=False,
):
    """Create a GPT2 shared actor-critic network for language modeling."""
    config = define_gpt2_configuration(
        vocabulary_size,
        n_positions,
        n_head,
        n_layer,
        n_embd,
        attn_pdrop,
        embd_pdrop,
        resid_pdrop,
    )
    # Define transformer
    lm = GPT2(config)

    # Wrap the transformer in a TensorDictModule to make TensorDict compatible
    lm_training = TensorDictModule(
        lm.set_train_mode(True),
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )
    lm_inference = TensorDictModule(
        lm,
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )

    # Define actor head and also make it a TensorDictModule and Probabilistic
    actor_head = TensorDictModule(
        nn.Linear(config.n_embd, vocabulary_size, bias=False),
        in_keys=["features"],
        out_keys=["logits"],
    )
    actor_head = ProbabilisticActor(
        module=actor_head,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define critic head and also make it a TensorDictModule
    critic_head = TensorDictModule(
        nn.Linear(
            config.n_embd,
            vocabulary_size if critic_value_per_action else 1,
            bias=False,
        ),
        in_keys=["features"],
        out_keys=["action_value"] if critic_value_per_action else ["state_value"],
    )

    # Create shared actor-critic TensorDictModule
    actor_critic_train = ActorValueOperator(
        common_operator=lm_training,
        policy_operator=actor_head,
        value_operator=critic_head,
    )
    actor_critic_inference = ActorValueOperator(
        common_operator=lm_inference,
        policy_operator=actor_head,
        value_operator=critic_head,
    )

    # Get individual operators
    actor_training = actor_critic_train.get_policy_operator()
    critic_training = actor_critic_train.get_value_operator()
    actor_inference = actor_critic_inference.get_policy_operator()
    critic_inference = actor_critic_inference.get_value_operator()

    return actor_training, actor_inference, critic_training, critic_inference
