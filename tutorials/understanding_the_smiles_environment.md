# Tutorial: Understanding the SMILES Environment

---

In this tutorial, we will demonstrate how to create an AceGen environment for smiles generation. We will also explain 
how to interact with it using TensorDicts and how to understand its expected inputs and outputs.

## Prerequisite Knowledge on TorchRL and Tensordict

AceGen is built on top of TorchRL, which is a reinforcement learning library for PyTorch. 
TorchRL uses Tensordict, a specialized data carrier for PyTorch tensors, to achieve compatibility between the 
different components  of the reinforcement learning pipeline, such as the environment, the model, and the data buffer.
Consequently, AceGen environments are also designed to be Tensordict-compatible, meaning that they accept a 
dictionary of tensors as input and return a dictionary of tensors as output.

---

## What is the SMILES environment?

The SMILES environment is a Tensordict-compatible environment for molecule generation with SMILES, and a key component 
of the AceGen library.  In particular, it is the component that manages the segment of the RL loop responsible for 
providing observations in response to the agent's actions. This environment class inherits  from the TorchRL base 
environment component ``EnvBase``, providing a range of advantages that include input and output data transformations, 
compatibility with Gym-based APIs,  efficient vectorized options (enabling the generation of multiple molecules in parallel), 
and the retrieval of clearly specified information attributes regarding expected input and  output data. With this 
environment, all TorchRL components become available for creating potential RL solutions.

---

## How to create a SMILES environment?

### Create a vocabulary

To create a SMILES environment, we first need to create a vocabulary. The vocabulary maps characters to indices and 
vice versa.  There are 3 ways to create a vocabulary in AceGen.

1. Create a vocabulary from a list of characters
```python
from acegen.vocabulary import SMILESVocabulary

chars = ["START", "END", "(", ")", "1", "=", "C", "N", "O"]
chars_dict = {char: index for index, char in enumerate(chars)}
vocab1 = SMILESVocabulary.create_from_dict(chars_dict, start_token="START", end_token="END")
```

2. Create a vocabulary from a list of SMILES strings. This method requires a tokenizer to be know how to split the 
SMILES strings into characters. AceGen already offers some tokenizers, but you can also create your own if you need to.
The only requirement is that it respects the simple interface defined [here](../acegen/vocabulary/base.py).
```python
from acegen.vocabulary import SMILESTokenizer

smiles_list = [
    "CCO",  # Ethanol (C2H5OH)
    "CCN(CC)CC",  # Triethylamine (C6H15N)
    "CC(=O)OC(C)C",  # Diethyl carbonate (C7H14O3)
    "CC(C)C",  # Isobutane (C4H10)
    "CC1=CC=CC=C1",  # Toluene (C7H8)
]

vocab2 = SMILESVocabulary.create_from_smiles(
    smiles_list, start_token="START", end_token="END", tokenizer=SMILESTokenizer(),
)
```

3. Create a vocabulary from a state dictionary. This method is useful when you have a saved vocabulary and you want to
load it. 
```python
state_dict = vocab2.state_dict()
vocab3 = SMILESVocabulary.load_state_dict(state_dict)
```

### Create the environment

Once we have the vocabulary, creating the environment is straightforward.

```python
from acegen.rl_env import SMILESEnv

env =  SMILESEnv(
    start_token=vocab1.start_token_index,
    end_token=vocab1.end_token_index,
    length_vocabulary=len(vocab1),
    batch_size=4, # Number of molecules to generate in parallel
)
```

---

## How to interact with the SMILES environment?

To start exploring how to use the SMILES environment, we can create an initial observation

```python
initial_td = env.reset() 
print(initial_td)

# Output
TensorDict(
    fields={
        done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        observation: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int32, is_shared=False),
        sequence: Tensor(shape=torch.Size([4, 100]), device=cpu, dtype=torch.int32, is_shared=False),
        sequence_mask: Tensor(shape=torch.Size([4, 100]), device=cpu, dtype=torch.bool, is_shared=False),
        terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([1]),
    device=None,
    is_shared=False)
```

Where:
- `observation` is the current token in the sequence (in this case start token).
- `sequence` is the whole sequence of generated tokens up to the maximum number of possible tokens. In this case, the sequence well have the starting token in the first position and zeros in the rest. This is useful for transfomer-based models.
- `sequence_mask` is a mask that indicates which tokens have been generated and which ones are padding tokens. In this case, the mask will have True in the first position and False in the rest. This is also useful for transformer-based models.
- `terminated` is a boolean tensor that indicates if the episode has been terminated because the previous action was the end token action.
- `truncated` is a boolean tensor that indicates if the episode is truncated because the maximum number of tokens has been reached.
- `done` is a boolean tensor that indicates if the episode is done, either because it has been terminated or truncated.

Then, we can create a dummy policy and pass the observation through it to get the next action. The dummy policy is simply
a random policy that selects a random action from the action space and adds it to the TensorDict. While real policies
would be more complex, the input and output data would still be TensorDicts, and the interaction with the environment would
be the same. Therefore, from the point of view of the environment, it does not matter if the policy is random or a
trained model. Thsi is one of the main advantages of using TorchRL, modularity and seamless integration of components.

```python
from torchrl.collectors import RandomPolicy

policy = RandomPolicy(env.full_action_spec)
initial_td = policy(initial_td)
print(initial_td)

# Output
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int32, is_shared=False),
        done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        observation: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int32, is_shared=False),
        sequence: Tensor(shape=torch.Size([4, 100]), device=cpu, dtype=torch.int32, is_shared=False),
        sequence_mask: Tensor(shape=torch.Size([4, 100]), device=cpu, dtype=torch.bool, is_shared=False),
        terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([4]),
    device=None,
    is_shared=False)
```

Now that we have an action, we can take an environment step. 

```python
initial_td = env.step(initial_td)
print(initial_td)
next_td = initial_td.get("next")

# Output
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int32, is_shared=False),
        done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
                reward: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                sequence: Tensor(shape=torch.Size([4, 100]), device=cpu, dtype=torch.int32, is_shared=False),
                sequence_mask: Tensor(shape=torch.Size([4, 100]), device=cpu, dtype=torch.bool, is_shared=False),
                terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([4]),
            device=None,
            is_shared=False),
        observation: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int32, is_shared=False),
        sequence: Tensor(shape=torch.Size([4, 100]), device=cpu, dtype=torch.int32, is_shared=False),
        sequence_mask: Tensor(shape=torch.Size([4, 100]), device=cpu, dtype=torch.bool, is_shared=False),
        terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([4]),
    device=None,
    is_shared=False)
```

As we can see, we are just moving data around coveniently packaged as TensorDicts. Now, the environment has returned the 
next observation, reward, and done information, and we can continue to take steps in the environment by now processing
the `next_td` TensorDict as the input to the policy.

Finally, we can generate a full rollout of the environment.

```python
rollout = env.rollout(max_steps=100, policy=policy)
print(rollout)

# Output
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([4, 9]), device=cpu, dtype=torch.int32, is_shared=False),
        done: Tensor(shape=torch.Size([4, 9, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([4, 9, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([4, 9]), device=cpu, dtype=torch.int64, is_shared=False),
                reward: Tensor(shape=torch.Size([4, 9, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                sequence: Tensor(shape=torch.Size([4, 9, 100]), device=cpu, dtype=torch.int32, is_shared=False),
                sequence_mask: Tensor(shape=torch.Size([4, 9, 100]), device=cpu, dtype=torch.bool, is_shared=False),
                terminated: Tensor(shape=torch.Size([4, 9, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([4, 9, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([4, 9]),
            device=None,
            is_shared=False),
        observation: Tensor(shape=torch.Size([4, 9]), device=cpu, dtype=torch.int64, is_shared=False),
        sequence: Tensor(shape=torch.Size([4, 9, 100]), device=cpu, dtype=torch.int32, is_shared=False),
        sequence_mask: Tensor(shape=torch.Size([4, 9, 100]), device=cpu, dtype=torch.bool, is_shared=False),
        terminated: Tensor(shape=torch.Size([4, 9, 1]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([4, 9, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
    batch_size=torch.Size([4, 9]),
    device=None,
    is_shared=False)
```

We can even decode the generated sequences using the vocabulary, but since the policy is random, the sequences will not
represent valid SMILES strings.

```python
print([vocab1.decode(seq) for seq in rollout["action"].numpy()])

# Output
['1(ONN(1', '()C)())', '=OCN)NNO', 'ONC)N((']
```

---

## What are the exact expected inputs and outputs of the SMILES environment?

We can better understand the expected inputs and outputs of the SMILES environment by running the following code 
snippets, which will print the full action, observation, done, and reward specs.

```python
print(env.full_action_spec)

# Output
CompositeSpec(
    action: DiscreteTensorSpec(
    shape=torch.Size([4]),
    space=DiscreteBox(n=9),
    device=cpu,
    dtype=torch.int32,
    domain=discrete), device=None, shape=torch.Size([4]))
```

```python
print(env.full_observation_spec)

# Output
CompositeSpec(
    observation: DiscreteTensorSpec(
    shape=torch.Size([4]),
    space=DiscreteBox(n=9),
    device=cpu,
    dtype=torch.int32,
    domain=discrete),
sequence: DiscreteTensorSpec(
    shape=torch.Size([4, 100]),
    space=DiscreteBox(n=9),
    device=cpu,
    dtype=torch.int32,
    domain=discrete),
sequence_mask: DiscreteTensorSpec(
    shape=torch.Size([4, 100]),
    space=DiscreteBox(n=2),
    device=cpu,
    dtype=torch.bool,
    domain=discrete), device=None, shape=torch.Size([4]))
```

```python
print(env.full_done_spec)

# Output
CompositeSpec(
    done: DiscreteTensorSpec(
    shape=torch.Size([4, 1]),
    space=DiscreteBox(n=2),
    device=cpu,
    dtype=torch.bool,
    domain=discrete),
truncated: DiscreteTensorSpec(
    shape=torch.Size([4, 1]),
    space=DiscreteBox(n=2),
    device=cpu,
    dtype=torch.bool,
    domain=discrete),
terminated: DiscreteTensorSpec(
    shape=torch.Size([4, 1]),
    space=DiscreteBox(n=2),
    device=cpu,
    dtype=torch.bool,
    domain=discrete), device=None, shape=torch.Size([4, 1]))
```

```python
print(env.full_reward_spec)

# Output
CompositeSpec(
    reward: UnboundedContinuousTensorSpec(
    shape=torch.Size([4, 1]),
    space=None,
    device=cpu,
    dtype=torch.float32,
    domain=continuous), device=None, shape=torch.Size([4]))
```

---

## Extending the AceGen environment for additional data fields

We can even add more fields to the environment TensorDicts if we need to. For example, if we want to use recurrent models,
we can add a `recurrent_state` field to the observation. This a more advanced topic, but it is important to know that we
can add more fields to the observation if we need to.  Here is how you would do it, with a something called `Transforms`
in TorchRL:

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