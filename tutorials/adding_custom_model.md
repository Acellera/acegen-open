# Tutorial: Integrating Custom Models in AceGen

---

## Prerequisite Knowledge

This tutorial assumes that you are familiar with the AceGen environment. 
If you are not, please refer to the [AceGen environment tutorial](understanding_the_smiles_environment.md).

## Defining a custom model

When integrating custom models into AceGen, it is important to keep in mind certain requirements. Let's delve into them:

### Requirement 1

AceGen is built on top of TorchRL, and TorchRL uses Tensordict, a data carrier for managing nested dictionaries of tensors,
to move around the data between the different components of the reinforcement learning pipeline, such as the environment,
the model, and the data buffer.  Therefore, any custom model must be Tensordict-compatible. In simpler terms, it should accept a Tensordict as input 
and return a Tensordict as output.

To achieve this compatibility, we'll define our custom model as a subclass of `torch.nn.Module` and wrap it with the
`tensordict.nn.TensordictModule` class. This ensures seamless integration with Tensordict. We'll explore this process in 
detail later in the tutorial. It is also important to know that similar to how `torch.nn.Sequential` is used to
concatenate layers, `tensordict.nn.TensordictSequential` can be used to concatenate TensordictModules. 

### Requirement 2

In reinforcement learning (RL), the model serves distinct purposes during training and inference phases. During 
inference (data collection), the model's role is to generate actions (and sometimes  additional data like the action 
log prob) based on the current state. However, during training, it must process sequences of data and predict outputs 
for each sequence element. Consequently, it is fundamental that out model can identify and handle both phases.

Now, the challenge arises in ensuring that our model can effectively handle both these phases. 
One approach might be to design a single model capable of discerning the phase from the shape 
of the input and adapting its behavior accordingly.

For instance:

- Recurrent models, during inference, typically receive a single step of the sequence without a time dimension 
  (shape = (batch_size, )). However, during training, they process sequences with both batch and temporal dimensions
  (shape = (batch_size, sequence_length)).
- While this method may suffice for some models, it doesn't apply universally. Transformer-based models, for instance, 
  consistently expect inputs of shape (batch_size, sequence_length) regardless of the phase. Nonetheless, 
  during training, it is expected to return outcomes for the entire sequence, whereas during inference, it is expected 
  to return outputs only for the first masked token (tokens are generated autoregressively and future tokens are masked).

To address this, it's advisable to define separate models for training and inference, both sharing the same weights. 
This approach ensures consistent behavior across phases, regardless of input shape variations. In the following sections, we'll walk through the process of implementing a custom model using the 
transformers library from HuggingFace that meet these requirements.

---

## Simple Example with a TensorDictModule

We will start by creating a simple example to illustrate how to make a model Tensordict-compatible.

```python
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

data = TensorDict({
    "key_1": torch.ones(3, 4),
    "key_2": torch.zeros(3, 4, dtype=torch.bool),
}, batch_size=[3])

# Define a simple module
module = torch.nn.Linear(4, 5)

# Make it Tensordict-compatible
td_module = TensorDictModule(module, in_keys=["key_1"], out_keys=["key_3"])

# Apply the module to the Tensordict data
data = td_module(data)
print(data)

# Output
TensorDict(
    fields={
        key_1: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
        key_2: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.bool, is_shared=False),
        key_3: Tensor(shape=torch.Size([3, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([3]),
    device=None,
    is_shared=False,
)
```

---

## Creating a custom model

Now we will define a custom model using the transformers library from HuggingFace. We will use the GPT-2 model as an example.
The model will be a `torch.nn.Module`, and will provide different outputs depending on the phase (training or inference),
defined by its `train_mode` attribute.

From all the tensors in the environment TensorDicts, the model will only use the `sequence` and `sequence_mask` tensors, 
so these will be the inputs of the forward method.  As explained in the [AceGen environment tutorial](understanding_the_smiles_environment.md)., the `sequence`
tensor is a tensor of shape (batch_size, sequence_length) that contains all the tokens generated so fat for the current 
SMILES. The `sequence_mask` tensor is a boolean tensor also  of shape (batch_size, sequence_length) that indicates as True 
the tokens that are part of the SMILES and as False the current and future tokens. In other words, masks the future. 
Therefore during inference the model will only return the prediction for the current token and during training it will
return the prediction for all the tokens.

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

        if self.train_mode is False:  # If inference, return only last token set to True by the sequence_mask
            obs_length = sequence_mask.sum(-1)
            out = out[torch.arange(len(out)), obs_length.to(torch.int64) - 1]

        return out
```

Now, we will wrap the model in a `tensordict.nn.TensordictModule` to make it Tensordict-compatible.
Then, use `tensordict.nn.TensordictSequential` to concatenate the model with a final layer that will output the logits.
Finally, we will wrap the model in a `ProbabilisticActor` to handle action sampling and log probability computation.
We will do that for both training and inference versions of the model, obtaining two models that will share the same weights.

```python
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import ProbabilisticActor


def create_gpt2_actor(
    vocabulary_size: int,
):
    # Define transformer
    config = GPT2Config()  # Original GPT2 configuration, can be customized
    config.vocab_size = vocabulary_size
    lm = GPT2(config)    

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

    # Define final layer and also make
    lm_head = TensorDictModule(
        nn.Linear(config.n_embd, vocabulary_size, bias=False),
        in_keys=["features"],
        out_keys=["logits"],
    )

    # Concatenate lm and head, similar to torch.nn.Sequential
    policy_training = TensorDictSequential(lm_training, lm_head)
    policy_inference = TensorDictSequential(lm_inference, lm_head)

    # To make the actor probabilistic, wrap the policy in a ProbabilisticActor
    # This module will take care of sampling and computing log probabilities
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
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )
    return probabilistic_policy_training, probabilistic_policy_inference
```

## How to make the custom model available in the training scripts

Available models to the training scripts are defined in `/acegen/__init__.py` as a mapping to tuples with the following format:

    model_mapping = {
        "example_model": (
            create_actor_method: Callable # A method to create the actor model
            create_critic_method: Callable # A method to create the critic model (Optional)
            create_actor_critic_method: Callable # A method to create the actor-critic model (Optional)
            vocabulary_file_path: Path # The path to the vocabulary file
            weights_file_path: Path # The path to the weights file (Optional)
            tokenizer: Tokenizer # The tokenizer to use for the model (Optional)
        )
    }

New models can be added by creating a new tuple and appending it to the model_mapping dictionary. Then the model can be
selected in any configuration file by setting the `model` parameter to the name of the model. In the case of our example, 
adding the models would look like this:

    model_mapping = {
        "gpt2": (
            create_gpt2_actor,
            None,
            None,
            Path(__file__).resolve().parent.parent.parent / "priors" / "enamine_real_vocabulary.txt",
            Path(__file__).resolve().parent.parent.parent / "priors" / "gpt2_enamine_real.ckpt",
            None,
        )
    }

Here we have assigned vocabulary and weights files from out set of priors to the model. We could, however, use others.  
Now, we can already use the model in the Reinvent and AHC training scripts for de novo molecule generation.
For decorative and linking tasks, we would need to define a tokenizer. We can use, for example, the SMILEStokenizer2()
from AceGen that is compatible with enamine_real_vocabulary.txt.
Finally, the PPO and A2C training scripts require a critic model. It would be similar to the actor model, but without the
ProbabilisticActor wrapper. Let's see how to define it:

```python

def create_gpt2_critic(
    vocabulary_size: int,
):
    """Create a GPT2 critic for language modeling."""
    # Define transformer
    config = GPT2Config()  # Original GPT2 configuration, can be customized
    config.vocab_size = vocabulary_size
    lm = GPT2(config)

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

    # Define final layer and also make it a TensorDictModule
    lm_head = TensorDictModule(
        nn.Linear(
            config.n_embd,
            1,
            bias=False,
        ),
        in_keys=["features"],
        out_keys=["state_value"],
    )

    # Concatenate lm and head, similar to torch.nn.Sequential
    # Critic does not need to be probabilistic, so we can return directly
    critic_training = TensorDictSequential(lm_training, lm_head)
    critic_inference = TensorDictSequential(lm_inference, lm_head)
    return critic_training, critic_inference
```

and then add it to the model_mapping dictionary:

    model_mapping = {
        "gpt2": (
            create_gpt2_actor,
            create_gpt2_critic,
            None,
            Path(__file__).resolve().parent.parent.parent / "priors" / "enamine_real_vocabulary.txt",
            Path(__file__).resolve().parent.parent.parent / "priors" / "gpt2_enamine_real.ckpt",
            SMILEStokenizer2(), # Constratined generation tasks require a tokenizer
        )
    }
