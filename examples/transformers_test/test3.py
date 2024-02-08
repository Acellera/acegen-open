from pathlib import Path
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class GPT2(nn.Module):
    """..."""

    def __init__(self, transformers_config):
        super(GPT2, self).__init__()

        config = GPT2Config()
        config.vocab_size = transformers_config.vocab_size
        config.n_positions = transformers_config.n_positions
        config.n_head = transformers_config.n_head
        config.n_layer = transformers_config.n_layer
        config.n_embd = transformers_config.n_embd
        config.attn_pdrop = transformers_config.attn_pdrop
        config.embd_pdrop = transformers_config.embd_pdrop
        config.resid_pdrop = transformers_config.resid_pdrop

        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Error: GPT2 network configuration should be an instance of GPT2Model"
            )

        self.config = config
        self.feature_extractor = GPT2Model(config)
        self.lm_head = nn.Linear(transformers_config.n_embd, transformers_config.vocab_size,  bias=False)

    def forward(self, context, context_mask):

        out = self.feature_extractor(
            input_ids=context,
            attention_mask=context_mask.long(),
        ).last_hidden_state

        # Prepare outputs
        has_masked_tokens = (context_mask == 0.0).any()
        if has_masked_tokens:  # Data collection
            obs_length = context_mask.sum(-1)
            out = out[torch.arange(len(out)), obs_length.to(torch.int64) - 1]
        else:  # Gradient computation
            out = out[:, -1]

        return self.lm_head(out)

# Define model
model_config = GPT2Config()

# Adjust model parameters
model_config.vocab_size = 63
model_config.n_positions = 2048
model_config.n_head = 16
model_config.n_layer = 24
model_config.n_embd = 128
model_config.attn_pdrop = 0.1
model_config.embd_pdrop = 0.1
model_config.resid_pdrop = 0.1
model_config.initializer_range = 0.02

policy = GPT2(model_config)
# ckpt = torch.load("pretrained_ckpt_epoch4.prior")
ckpt = torch.load(
    Path(__file__).resolve().parent.parent.parent / "priors" / "gpt2_enamine_real.ckpt"
)

try:
    assert len(policy.state_dict()) == len(ckpt)
except AssertionError:
    print(f"Expected {len(policy.state_dict())} but got {len(ckpt)}")

