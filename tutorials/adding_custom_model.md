# Tutorial: Integrating Custom Models in AceGen (WIP)

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

One tempting option would be to have a single model that infers the phase from the shape of the input and acts accordingly. 
Recurrent models during inference will receive a single step of the sequence, without a time dimension (i.e. shape = (batch_size, )) 
while during training they will have both a batch and a temporal dimension. i.e. shape = (batch_size, sequence_length).
This can work for some models, but not for all. Transformer-based models always expect an input of shape (batch_size, sequence_length), 
regardless of the phase. The difference between training and inference is that the agent will only return a single token 
which will be determined by the sequence_mask field.

To address this, it's advisable to define separate models for training and inference, both sharing the same weights. 
This approach ensures consistent behavior across phases, regardless of input shape variations.

In the following sections, we'll walk through the process of implementing a custom model using the 
transformers library from HuggingFace that meet these requirements.

---

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

        if self.train_mode is False:  # If inference, return only last token set to True by the sequence_mask
            obs_length = sequence_mask.sum(-1)
            out = out[torch.arange(len(out)), obs_length.to(torch.int64) - 1]

        return out
```

```python
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import ProbabilisticActor


def create_gpt2_actor(
    vocabulary_size: int,
    return_log_prob=True,
):
    # Define transformer
    config = GPT2Config()  # Original GPT2 configuration
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
            weights_file_path: Path # The path to the weights file (Optional)
            tokenizer: Tokenizer # The tokenizer to use for the model (Optional)
        )
    }

New models can be added by creating a new tuple adding it to the model_mapping dictionary. Then the model can be
selected in any configuration file by setting the `model` parameter to the name of the model. In the case of our example, 
adding the models would look like this:

    model_mapping = {
        "gpt2_example": (
            create_gpt2_actor,
            None,
            None,
            Path(__file__).resolve().parent.parent.parent  / "priors" / "chembl_vocabulary.txt",
            None,
            SmilesTokenizer(),
        )
    }

Here we have assigned a vocabulary from out prior to our model, a simple tokenizer and no weights file.
We could, however, add a weights file to the model if we wanted to. Or pretrain the model with AceGen's
retraining scripts and then add the weights file to the model_mapping dictionary. Similarly, we could optimize 
some scoring function with Reinvent or AHC (PPO and A2C would require defining a critic 
model).

---

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
