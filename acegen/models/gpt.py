import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class GPT2(nn.Module):
    """..."""

    def __init__(self, config):
        super(GPT2, self).__init__()

        if not isinstance(config, GPT2Config):
            raise ValueError(
                "Error: GPT2 network configuration should be an instance of GPT2Model"
            )

        self.config = config
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.feature_extractor = GPT2Model(config)

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
