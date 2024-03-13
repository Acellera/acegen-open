# Tutorial: Integrating Custom Models in AceGen (WIP)

---

## Prerequisite Knowledge

This tutorial assumes that you are familiar with the AceGen environment. 
If you are not, please refer to the [AceGen environment tutorial](understanding_the_smiles_environment.md).

## Defining a custom model

### Requirement 1

AceGen is built on top of TorchRL, and TorchRL uses Tensordict, a data carrier for managing nested dictionaries of tensors, 
to move around the data between the different components of the reinforcement learning pipeline, such as the environment, 
the model, and the data buffer.

What this means is that when we define a custom model, we need to make it Tensordict-compatible. In other words,
it should accept a Tensordict as input and return a Tensordict as output.

Nonetheless, defining a custom model is straightforward is we know PyTorch. We can define a custom model as a subclass 
of `torch.nn.Module` and wrap it with the `tensordict.nn.TensordictModule` class, which makes sure that the model is 
compatible with Tensordict. We will see how to do it in this tutorial.

### Requirement 2

In reinforcement learning (RL), the model is generally used in 2 different phases, training and inference. 

In each phase, what we want the model to do can be different.

For example, during inference (the data collection phase), we just want the model to generate an action (and sometimes 
additional data like the action log prob) given the current state. Therefore, the received TensorDict 
will not have a temporal dimension, only a batch dimension. i.e. shape = (batch_size, ).

However, during training we want the model to process a sequence of data, and to predict outputs for each element of the
sequence. Therefore, the received TensorDict will have both a batch and a temporal dimension. i.e. shape = (batch_size, 
sequence_length).

## Creating a custom model

The output of the model should simply be the next token to be generated.
We can get a better understanding of its structure by running the following code:

```python
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

class GPT2(nn.Module):

    def __init__(self, config=None):
        super(GPT2, self).__init__()
        self.feature_extractor = GPT2Model(config) if config is not None else None        
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
```

```python
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
```

```python
def create_gpt2_actor(
    vocabulary_size: int,
    return_log_prob=True,
):
    """Create a GPT2 actor for language modeling."""
    config = define_gpt2_configuration(vocabulary_size)
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
    probabilistic_policy_training = ProbabilisticActor(
        module=TensorDictSequential(lm.set_train_mode(True), lm_head),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )
    probabilistic_policy_inference = ProbabilisticActor(
        module=TensorDictSequential(lm, lm_head),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )
    return probabilistic_policy_training, probabilistic_policy_inference
```

## How to make the custom model available in the training scripts

Models are defined in `/acegen/__init__.py` and as a mapping to tuples with the following format:

    model_mapping = {
        "example_model": (
            create_actor_method: Callable # A method to create the actor model
            create_critic_method: Callable # A method to create the critic model (Optional)
            create_actor_critic_method: Callable # A method to create the actor-critic model (Optional)
            vocabulary_file_path: Path # The path to the vocabulary file
            weights_file_path: Path # The path to the weights file
            tokenizer: Tokenizer # The tokenizer to use for the model (Optional)
        )
    }

In the case of our example, it would look like this:

    model_mapping = {
        "gpt2": (
            create_gpt2_actor,
            None,
            None,
            None,
            None, 
            None,
        )
    }

New models can be added by creating a new tuple adding it to the model_mapping dictionary. Then the model can be 
selected in the configuration file by setting the `example_model` parameter to the name of the model.

## Extending the AceGen environment for additional data fields

Therefore, as we will see in the next section, the model can use any of these fields as input. We can even add more fields to the observation if we need to. For example, if we want to use recurrent models, we can add a `recurrent_state` field to the observation.
This a more advanced topic, but it is important to know that we can add more fields to the observation if we need to.
Here is how you would do it, with a something called `Transforms` in TorchRL:

```python

from torchrl.envs.transforms import TensorDictPrimer
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec
from torchrl.envs import TransformedEnv

my_rnn_transform = TensorDictPrimer(
    {
        "recurrent_state": UnboundedContinuousTensorSpec(shape=(1, 10)),
    }
)

env = TransformedEnv(env, my_rnn_transform)
obs = env.reset()
print(obs)
```

Now the output of the above code is:

```python
TensorDict(
    fields={
        done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        observation: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.int32, is_shared=False),
        recurrent_state: Tensor(shape=torch.Size([1, 10]), device=cpu, dtype=torch.float32, is_shared=False),
        sequence: Tensor(shape=torch.Size([1, 100]), device=cpu, dtype=torch.int32, is_shared=False),
        sequence_mask: Tensor(shape=torch.Size([1, 100]), device=cpu, dtype=torch.bool, is_shared=False),
        terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([1]),
    device=None,
    is_shared=False)
```

As we can see, the `recurrent_state` field has been added to the observation.
