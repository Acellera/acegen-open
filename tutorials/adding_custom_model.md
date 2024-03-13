# Tutorial: Integrating Custom Models in AceGen (WIP)

---

## Prerequisite Knowledge

This tutorial assumes that you are familiar with the AceGen environment. 
If you are not, please refer to the [AceGen environment tutorial](understanding_the_smiles_environment.md).

## Defining a custom model

AceGen is built on top of TorchRL, and TorchRL uses Tensordict, a data carrier for managing nested dictionaries of tensors, 
to move around the data between the different components of the reinforcement learning pipeline, such as the environment, 
the model, and the data buffer.

What this means is that when we define a custom model, we need to make it Tensordict-compatible. In other words,
it should accept a Tensordict as input and return a Tensordict as output.

Nonetheless, defining a custom model is straightforward is we know PyTorch. We can define a custom model as a subclass 
of `torch.nn.Module` and wrap it with the `tensordict.nn.TensordictModule` class, which makes sure that the model is 
compatible with Tensordict. We will see how to do it in this tutorial.

## Creating a custom model

The output of the model should simply be the next token to be generated.
We can get a better understanding of its structure by running the following code:

```python
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

class GPT2(nn.Module):
    """GPT2 model for language modeling. This model is a simple wrapper around the HuggingFace GPT2Model."""

    def __init__(self, config):
        super(GPT2, self).__init__()

        # Define model
        self.feature_extractor = GPT2Model(config)

    def forward(self, sequence, sequence_mask=None):

        is_inference = True
        if sequence_mask is None:
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
