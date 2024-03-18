import warnings

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import ActorValueOperator, ProbabilisticActor

try:
    from mamba_ssm import mamba

    _has_mamba = True
except ImportError as err:
    _has_mamba = False
    MAMBA_ERR = err

class MambaModule(Mamba):
    # We wrap forward here to compute the mask and keep keys consistent with transformers
    def forward(self, sequence, sequence_mask):
        # Mamba takes a hidden state (batch, seqlen, vocab_size)
        return super().forward(ht)

def create_mamba_actor(
    vocabulary_size: int,
    d_state: int = 16, # SSM state expansion factor
    d_conv: int = 4, # Local convolution width
    expand: int = 2 # Block expansion factor
):
    """Create a Mamba actor for language modelling."""
    lm = MambaModule(
        d_model=vocabulary_size,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand
    )

    lm_training = TensorDictModule(
        lm,
        in_keys=["sequence", "sequence_mask"],
        out_keys=["logits"]
    )