import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import ActorValueOperator, ProbabilisticActor

try:
    from transformers import GPT2Config, GPT2Model

    _has_transformers = True
except ImportError as err:
    _has_transformers = False
    TRANSFORMERS_ERR = err


class GPT2(nn.Module):
    """GPT2 model for language modeling. This model is a simple wrapper around the HuggingFace GPT2Model."""

    def __init__(self, config):
        if not _has_transformers:
            raise RuntimeError(
                "transformers library not found, please install with pip install transformers."
            ) from TRANSFORMERS_ERR

        super(GPT2, self).__init__()

        # Define model
        self.feature_extractor = GPT2Model(config)

    def forward(self, sequence, sequence_mask=None):

        is_inference = True
        if sequence_mask is None:
            #  sequence_mask = (sequence != 0).long()
            sequence_mask = torch.ones_like(sequence, dtype=torch.long)
            is_inference = False

        out = self.feature_extractor(
            input_ids=sequence,
            attention_mask=sequence_mask.long(),
        ).last_hidden_state

        if is_inference:  # Data collection
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
    # Define model
    config = GPT2Config()

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
    lm = TensorDictModule(
        GPT2(config),
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )
    lm_head = TensorDictModule(
        nn.Linear(config.n_embd, vocabulary_size, bias=False),
        in_keys=["features"],
        out_keys=["logits"],
    )
    policy = TensorDictSequential(lm, lm_head)
    probabilistic_policy = ProbabilisticActor(
        module=policy,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )
    return probabilistic_policy, probabilistic_policy


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
    lm = TensorDictModule(
        GPT2(config),
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )
    lm_head = TensorDictModule(
        nn.Linear(
            config.n_embd,
            vocabulary_size if critic_value_per_action else 1,
            bias=False,
        ),
        in_keys=["features"],
        out_keys=["action_value"] if critic_value_per_action else ["state_value"],
    )
    critic = TensorDictSequential(lm, lm_head)
    return critic, critic


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
    lm = TensorDictModule(
        GPT2(config),
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )
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
    critic_head = TensorDictModule(
        nn.Linear(
            config.n_embd,
            vocabulary_size if critic_value_per_action else 1,
            bias=False,
        ),
        in_keys=["features"],
        out_keys=["action_value"] if critic_value_per_action else ["state_value"],
    )
    actor_critic = ActorValueOperator(
        common_operator=lm,
        policy_operator=actor_head,
        value_operator=critic_head,
    )
    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()
    return actor, actor, critic, critic
