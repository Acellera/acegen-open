from pathlib import Path
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


class Model(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()

        self.transformer = GPT2(config)
        self.head = LMHead(config)

        self.feature_extractor = self.transformer.feature_extractor
        self.evaluator = self.head.evaluate_pred
        self.device = device

    def forward(self, batch):
        # Prepare batch.
        seqs = batch.long()
        seqs = seqs.to(self.device)

        # Prediction
        features = self.feature_extractor(
            input_ids=seqs.long(), attention_mask=torch.ones_like(seqs).long()
        ).last_hidden_state
        logp_action, _, _ = self.evaluator(features[:, :-1], seqs[:, 1:])

        # Loss
        mask = (seqs[:, 1:] != 0).float()  # Mask padding
        loss = (-logp_action.squeeze(-1) * mask).sum(1).mean()
        #loss = ((-logp_action.squeeze(-1) * mask).sum(1) / mask.sum(1)).mean()
        return loss

class GPT2(nn.Module):
    """
    Wrapper to be able to use GPT2 model from HuggingFace transformers repo.

    Parameters
    ----------
    input_space : gym.Space
        Environment observation space.
    transformers_config : list
        Hidden layers sizes.
    """

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

    def forward(self, inputs):
        """
        Forward pass Neural Network.

        Parameters
        ----------
        inputs : Dict[torch.Tensor]
            Input data.

        Returns
        -------
        out : torch.Tensor
            Output feature map.
        """

        obs = inputs["obs"]
        obs_length = inputs["obs_length"]

        # Forward pass
        out = self.feature_extractor(
            input_ids=obs.long(),
            attention_mask=(obs != 0.0).long(),  # Shape (batch_size, sequence_length)
        ).last_hidden_state

        # Prepare outputs
        has_masked_tokens = (obs == 0.0).any()
        if has_masked_tokens:  # Data collection
            obs_length = inputs["obs_length"]
            out = out[torch.arange(len(out)), obs_length.to(torch.int64) - 1]
        else:  # Gradient computation
            out = out[:, -1]

        return out


class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.lm_head = nn.Linear(self.config.n_embd,
                                 self.config.vocab_size,
                                 bias=False)

        self.apply(self._init_weights)

    def forward(self, x, deterministic=False):
        """
        Predict distribution parameters from x (obs features) and return
        predictions (sampled and clipped), sampled log
        probability and distribution entropy.
        """
        x = self.lm_head(x)

        dist = torch.distributions.Categorical(logits=x)
        if deterministic:
            pred = clipped_pred = dist.probs.argmax(dim=-1, keepdim=True)
        else:
            pred = clipped_pred = dist.sample().unsqueeze(-1)

        # Action log probability
        # logp = dist.log_prob(pred.squeeze(-1)).unsqueeze(-1)
        logp = dist.log_prob(pred.squeeze(-1)).view(pred.size(0), -1).sum(-1).unsqueeze(-1)

        # Distribution entropy
        entropy = dist.entropy().mean()

        return pred, clipped_pred, logp, entropy, dist

    def evaluate_pred(self, x, pred):
        """
        Return log prob of `pred` under the distribution generated from
        x (obs features). Also return entropy of the generated distribution.

        Parameters
        ----------
        x : torch.tensor
            obs feature map obtained from a GPT2.
        pred : torch.tensor
            Prediction to evaluate.

        Returns
        -------
        logp : torch.tensor
            Log probability of `pred` according to the predicted distribution.
        entropy_dist : torch.tensor
            Entropy of the predicted distribution.
        dist : torch.Distribution
            Action probability distribution.
        """

        # Predict distribution parameters
        x = self.lm_head(x)

        # Create distribution
        dist = torch.distributions.Categorical(logits=x)

        # Evaluate log probability of `pred`
        logp = dist.log_prob(pred.squeeze(-1)).unsqueeze(-1).sum(-1, keepdim=True)

        # Distribution entropy
        entropy_dist = dist.entropy().mean()

        return logp, entropy_dist, dist

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def init_weights(self):
        self.apply(self._init_weights)

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

ckpt = torch.load(
    Path(__file__).resolve().parent.parent.parent / "priors" / "gpt2_enamine_real.ckpt"
)
model = Model(model_config)
assert len(model.state_dict()) == len(ckpt)

