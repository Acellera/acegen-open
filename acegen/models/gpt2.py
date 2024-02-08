import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.envs import ExplorationType
from torchrl.modules import ProbabilisticActor
from transformers import GPT2Config, GPT2Model


class GPT2(nn.Module):
    """..."""

    def __init__(self, vocabulary_size, config=None):
        super(GPT2, self).__init__()

        # Define model
        config = GPT2Config()

        # Adjust model parameters
        config.vocab_size = vocabulary_size
        config.n_positions = 2048
        config.n_head = 16
        config.n_layer = 24
        config.n_embd = 128
        config.attn_pdrop = 0.1
        config.embd_pdrop = 0.1
        config.resid_pdrop = 0.1
        config.initializer_range = 0.02

        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Error: GPT2 network configuration should be an instance of GPT2Model"
            )

        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.feature_extractor = GPT2Model(config)

    def forward(self, sequence, sequence_mask=None):

        if sequence_mask is None:
            sequence_mask = torch.ones_like(sequence, dtype=torch.float32)

        out = self.feature_extractor(
            input_ids=sequence,
            attention_mask=sequence_mask.long(),
        ).last_hidden_state

        # Prepare outputs
        has_masked_tokens = (sequence_mask == 0.0).any()
        if has_masked_tokens:  # Data collection
            obs_length = sequence_mask.sum(-1)
            out = out[torch.arange(len(out)), obs_length.to(torch.int64) - 1]

        return self.lm_head(out)


def create_gpt2_actor(vocabulary_size):
    """..."""
    policy = TensorDictModule(
        GPT2(vocabulary_size),
        in_keys=["sequence", "sequence_mask"],
        out_keys=["logits"],
    )
    probabilistic_policy = ProbabilisticActor(
        module=policy,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    return probabilistic_policy, probabilistic_policy
