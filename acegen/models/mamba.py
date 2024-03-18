from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import ActorValueOperator, ProbabilisticActor

try:
    from mamba_ssm.models.mixer_seq_simple import MixerModel

    _has_mamba = True
except ImportError as err:
    _has_mamba = False
    MAMBA_ERR = err


class Mamba(nn.Module):
    def __init__(self, config=None):
        if not _has_mamba:
            raise RuntimeError(
                "mamba-ssm library not found, please install with pip install mamba-ssm."
            ) from MAMBA_ERR

        super(Mamba, self).__init__()

        # Define model
        self.config = config
        if config:
            self.feature_extractor = MixerModel(
                d_model=config.n_embd,
                n_layer=config.n_layer,
                vocab_size=config.vocab_size,
                ssm_cfg={"use_fast_path": True},
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
            )
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
        out = Mamba()
        out.feature_extractor = self.feature_extractor
        out._train_mode = True
        return out

    def forward(self, sequence, sequence_mask):
        # Reshape observation
        if self.train_mode is False:
            sequence = sequence.reshape(-1, 1)
        # Extract
        out = self.feature_extractor(input_ids=sequence)

        if self.train_mode is False:  # Data collection, return only last token
            out = out.squeeze()
            obs_length = sequence_mask.sum(-1)
            out = out[torch.arange(len(out)), obs_length.to(torch.int64) - 1]

        return out


@dataclass
class MambaConfig:
    vocab_size: int
    n_embd: int = 128
    n_layer: int = 24


def create_mamba_actor(
    vocabulary_size: int,
    n_embd: int = 128,
    n_layer: int = 24,
    return_log_prob: bool = True,
    **kwargs
):
    """Create a Mamba actor for language modelling."""
    # Define mode
    config = MambaConfig(
        vocab_size=vocabulary_size,
        n_embd=n_embd,
        n_layer=n_layer,
    )
    lm = Mamba(config)

    # Transform into TensorDict modules
    lm_training = TensorDictModule(
        lm.set_train_mode(True),
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )

    lm_inference = TensorDictModule(
        lm, in_keys=["sequence", "sequence_mask"], out_keys=["features"]
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
