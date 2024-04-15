from clms.models.model import Model
from clms.models.configuration import ModelConfig
from clms.learning.utils.saving import ModelCheckpointDict
import torch

from torch import nn

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import ProbabilisticActor, ActorValueOperator

def load_model_config(config, vocabulary_size):

    if config.model_path is not None:
        ckpt = torch.load(config.model_path, map_location='cpu')
        model_config = ckpt['feature_extractor_kwargs']['transformers_config']
    else:
        model_config = ModelConfig()
        model_config.model_name = config.model_name
        model_config.vocab_size = vocabulary_size
        model_config.n_positions = config.pretrain_max_smile_length
        model_config.n_head = config.n_head
        model_config.n_kv_head = config.n_kv_head
        model_config.n_layer = config.n_layer
        model_config.n_embd = config.n_embd
        model_config.attn_pdrop = config.attn_pdrop
        model_config.embd_pdrop = config.embd_pdrop
        model_config.resid_pdrop = config.resid_pdrop
        model_config.flash_attention = config.flash_attention
    
    return model_config, vocabulary_size

def transform_config(func):
    def wrapper(config, vocabulary_size, *args, **kwargs):
        transformed_config, vocabulary_size = load_model_config(config, vocabulary_size)
        return func(transformed_config, vocabulary_size, *args[1:], **kwargs)
    return wrapper


class AceGenModel(nn.Module):
    def __init__(self, config=None):
        super(AceGenModel, self).__init__()
        self.config = config
        model = Model(config) if config is not None else None
        self.feature_extractor = model.transformer
        self._train_mode = False
    
    @property
    def train_mode(self):
        return self._train_mode
    
    def set_train_mode(self, train_mode: bool = True):
        if train_mode is self._train_mode:
            return self
        out = AceGenModel(self.config)
        out.feature_extractor = self.feature_extractor
        out._train_mode = train_mode
        return out
    
    def forward(self, sequence, sequence_mask):
        out = self.feature_extractor(
            batch = sequence,
        )#attention_mask = sequence_mask.long())
        
        if torch.isnan(out).any():
            raise ValueError("NaN detected in model output.")
        
        if self.train_mode is False:
            obs_length = sequence_mask.sum(-1)
            out = out[torch.arange(len(out)), obs_length.to(torch.int64) - 1]
        
        return out

@transform_config
def create_clm_actor(config, vocabulary_size, return_log_prob=True):
    """ Create a CLM Model actor for language modeling. """
    
    # Define the transformer
    lm = AceGenModel(config)
    
    # Wrap it in a TensorDictModule to make it TensorDictCompatible
    lm_training = TensorDictModule(
        lm.set_train_mode(True),
        in_keys=["sequence", "sequence_mask"],
        out_keys = ["features"]
    )
    
    lm_inference = TensorDictModule(
        lm,
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )

    # Define head layer and make it a TensorDictModule
    lm_head = TensorDictModule(
        nn.Linear(config.n_embd, vocabulary_size, bias=False),
        in_keys = ["features"],
        out_keys = ["logits"]
    )
    
    # Concatenate lm and head, similar to torch.nn.Sequential
    policy_training = TensorDictSequential(lm_training, lm_head)
    policy_inference = TensorDictSequential(lm_inference, lm_head)
    
    # To make actor probabilistic, wrap the policy in a ProbabilisticActor
    # This will take care of sampling and computing log probabilities
    probabilistic_policy_training = ProbabilisticActor(
        module = policy_training,
        in_keys = ["logits"],
        out_keys = ["action"],
        distribution_class = torch.distributions.Categorical,
        return_log_prob = return_log_prob,
        default_interaction_type = ExplorationType.RANDOM,
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

@transform_config
def create_clm_critic(config, vocabulary_size, critic_value_per_action=False):
    
    """ Create CLM critic for language modeling. """
    # Define transformer
    lm = AceGenModel(config)
    
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

    # Define last layer and also make it TensorDictModule
    lm_head = TensorDictModule(
        nn.Linear(
            config.n_embd,
            vocabulary_size if critic_value_per_action else 1,
            bias = False
        ),
        in_keys = ["features"],
        out_keys = ["action_value"] if critic_value_per_action else ["state_value"],
    )
    
    # Concatenate lm and head, similar to troch.nn.Sequential
    # Critic does not need to be probabilistic, so we can return directly
    critic_training = TensorDictSequential(lm_training, lm_head)
    critic_inference = TensorDictSequential(lm_inference, lm_head)
    return critic_training, critic_inference

@transform_config
def create_clm_actor_critic(config, vocabulary_size, return_log_prob = True, critic_value_per_action = False):
    
    # Define the transformer
    lm = AceGenModel(config)
    
    # Wrap it in a TensorDictModule to make it TensorDictCompatible
    lm_training = TensorDictModule(
        lm.set_train_mode(True),
        in_keys=["sequence", "sequence_mask"],
        out_keys = ["features"]
    )
    
    lm_inference = TensorDictModule(
        lm,
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )
    
    actor_head = TensorDictModule(
        nn.Linear(config.n_embd, vocabulary_size ,bias=False),
        in_keys = ["features"],
        out_keys = ["logits"],
    )

    actor_head = ProbabilisticActor(
        module = actor_head,
        in_keys = ["logits"],
        out_keys = ["action"],
        distribution_class = torch.distributions.Categorical,
        return_log_prob = return_log_prob,
        default_interaction_type = ExplorationType.RANDOM,
    )

    # Define critic head and also make it a TensorDictModule
    critic_head = TensorDictModule(
        nn.Linear(
            config.n_embd,
            vocabulary_size if critic_value_per_action else 1,
            bias = False
        ),
        in_keys = ["features"],
        out_keys = ["action_value"] if critic_value_per_action else ["state_value"],
    )

    # Create shared actor-critic TensorDictModule
    actor_critic_train = ActorValueOperator(
        common_operator = lm_training,
        policy_operator = actor_head,
        value_operator = critic_head,
    )
    actor_critic_inference = ActorValueOperator(
        common_operator = lm_inference,
        policy_operator = actor_head,
        value_operator = critic_head,
    )

    # Get individual operators
    actor_training = actor_critic_train.get_policy_operator()
    critic_training = actor_critic_train.get_value_operator()
    actor_inference = actor_critic_inference.get_policy_operator()
    critic_inference = actor_critic_inference.get_value_operator()

    return actor_training, actor_inference, critic_training, critic_inference