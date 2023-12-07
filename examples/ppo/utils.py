import torch
import random
import warnings
import numpy as np
from rdkit import Chem
from pathlib import Path

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from torchrl.modules import (
    MLP,
    GRUModule,
    ValueOperator,
    ActorValueOperator,
    ProbabilisticActor,
)
from torchrl.envs import ExplorationType, TensorDictPrimer
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec

## Models #############################################################################################################

class Embed(torch.nn.Module):
    """Implements a simple embedding layer."""

    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self._embedding = torch.nn.Embedding(input_size, embedding_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, L = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        inputs = inputs.squeeze(-1)  # Embedding creates an extra dimension
        out = self._embedding(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out

def create_shared_ppo_models(vocabulary_size, batch_size, ckpt=None):
    """Create a shared PPO model using architecture and weights from the
    "REINVENT 2.0 – an AI tool for de novo drug design" paper.

    The policy component of the model uses the same architecture and weights
    as described in the original paper. The critic component is implemented
    as a simple Multi-Layer Perceptron (MLP) with one output.

    Returns:
    - shared_model: The shared PPO model with both policy and critic components.

    Example:
    ```python
    shared_model = create_shared_ppo_model(10)
    ```
    """

    if ckpt is None:
        ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "reinvent.prior")

    embedding_module = TensorDictModule(
        Embed(vocabulary_size, 128),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    lstm_module = GRUModule(
        dropout=0.0,
        input_size=128,
        hidden_size=512,
        num_layers=3,
        in_keys=["embed", "recurrent_state", "is_init"],
        out_keys=[
            "features",
            ("next", "recurrent_state"),
        ],
    )
    actor_model = TensorDictModule(
        MLP(
            in_features=512,
            out_features=vocabulary_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=["logits"],
    )
    policy_module = ProbabilisticActor(
        module=actor_model,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    critic_module = ValueOperator(
        MLP(
            in_features=512,
            out_features=1,
            num_cells=[256, 256],
        ),
        in_keys=["features"],
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic_inference = ActorValueOperator(
        common_operator=TensorDictSequential(embedding_module, lstm_module),
        policy_operator=policy_module,
        value_operator=critic_module,
    )

    actor_critic_training = ActorValueOperator(
        common_operator=TensorDictSequential(
            embedding_module, lstm_module.set_recurrent_mode(True)
        ),
        policy_operator=policy_module,
        value_operator=critic_module,
    )

    actor_inference = actor_critic_inference.get_policy_operator()
    critic_inference = actor_critic_inference.get_value_operator()
    actor_training = actor_critic_training.get_policy_operator()
    critic_training = actor_critic_training.get_value_operator()

    ckpt = adapt_ppo_ckpt(ckpt)
    actor_inference.load_state_dict(ckpt)
    actor_training.load_state_dict(ckpt)

    primers = {
        ('recurrent_state',):
            UnboundedContinuousTensorSpec(
                shape=torch.Size([batch_size, 3, 512]),
                dtype=torch.float32,
            ),
    }
    transform = TensorDictPrimer(primers)

    return actor_inference, actor_training, critic_inference, critic_training, transform

def adapt_ppo_ckpt(ckpt):
    """Adapt the PPO ckpt from the AceGen ckpt format."""

    keys_mapping = {
        'embedding.weight': "module.0.module.0.module._embedding.weight",
        'gru_1.weight_ih': "module.0.module.1.gru.weight_ih_l0",
        'gru_1.weight_hh': "module.0.module.1.gru.weight_hh_l0",
        'gru_1.bias_ih': "module.0.module.1.gru.bias_ih_l0",
        'gru_1.bias_hh': "module.0.module.1.gru.bias_hh_l0",
        'gru_2.weight_ih': "module.0.module.1.gru.weight_ih_l1",
        'gru_2.weight_hh': "module.0.module.1.gru.weight_hh_l1",
        'gru_2.bias_ih': "module.0.module.1.gru.bias_ih_l1",
        'gru_2.bias_hh': "module.0.module.1.gru.bias_hh_l1",
        'gru_3.weight_ih': "module.0.module.1.gru.weight_ih_l2",
        'gru_3.weight_hh': "module.0.module.1.gru.weight_hh_l2",
        'gru_3.bias_ih': "module.0.module.1.gru.bias_ih_l2",
        'gru_3.bias_hh': "module.0.module.1.gru.bias_hh_l2",
        'linear.weight': "module.1.module.0.weight",
        'linear.bias': "module.1.module.0.bias",
    }

    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[keys_mapping[k]] = v

    return new_ckpt

## Replay buffer #######################################################################################################

class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""

    def __init__(self, voc, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.voc = voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]

    def sample(self, n, decode_smiles=True):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1].item() + 1e-10 for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores/np.sum(scores))
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        if decode_smiles:
            encoded = [torch.tensor(self.voc.encode(smile), dtype=torch.int32) for smile in smiles]
            smiles = collate_fn(encoded)
        return smiles, torch.tensor(scores), torch.tensor(prior_likelihood)

    def __len__(self):
        return len(self.memory)

def collate_fn(arr):
    """Function to take a list of encoded sequences and turn them into a batch"""
    max_length = max([seq.size(0) for seq in arr])
    collated_arr = torch.zeros(len(arr), max_length)
    for i, seq in enumerate(arr):
        collated_arr[i, :seq.size(0)] = seq
    return collated_arr

def create_batch_from_replay_smiles(replay_smiles, replay_rewards, device, vocabulary):
    """Create a TensorDict data batch from replay data."""

    td_list = []

    for smiles, rew in zip(replay_smiles, replay_rewards):

        encoded = vocabulary.encode(smiles)
        smiles = torch.tensor(encoded, dtype=torch.int32, device=device)

        # RANDOMISE SMILES #########

        # if random.random() > 0.5:
        #     try:
        #         decoded = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True, canonical=False)
        #         encoded = vocabulary.encode(decoded)
        #         smiles = torch.tensor(encoded, dtype=torch.int32, device=device)
        #     except Exception:  # Sometimes a token outside the vocabulary can appear, in this case we just skip
        #         warnings.warn(f"Skipping {smiles}")
        #         encoded = vocabulary.encode(smiles)
        #         smiles = torch.tensor(encoded, dtype=torch.int32, device=device)

        ############################

        observation = smiles[:-1].reshape(1, -1, 1).clone()
        action = smiles[1:].reshape(1, -1).clone()
        tensor_shape = (1, observation.shape[1], 1)
        reward = torch.zeros(tensor_shape, device=device)
        reward[0, -1] = rew
        done = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        terminated = done.clone()
        sample_log_prob = torch.zeros(tensor_shape, device=device).reshape(1, -1)
        is_init = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        is_init[0, 0] = True

        next_observation = smiles[1:].reshape(1, -1, 1).clone()
        next_done = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
        next_done[0, -1] = True
        next_terminated = next_done.clone()
        next_is_init = torch.zeros(tensor_shape, device=device, dtype=torch.bool)

        recurrent_states = torch.zeros(*tensor_shape[0:2], 3, 512)
        next_recurrent_states = torch.zeros(*tensor_shape[0:2], 3, 512)

        td_list.append(
            TensorDict(
                {
                    "done": done,
                    "action": action,
                    "is_init": is_init,
                    "terminated": terminated,
                    "observation": observation,
                    "sample_log_prob": sample_log_prob,
                    "recurrent_state": recurrent_states,
                    "next": TensorDict(
                        {
                            "observation": next_observation,
                            "terminated": next_terminated,
                            "reward": reward,
                            "is_init": next_is_init,
                            "done": next_done,
                            "recurrent_state": next_recurrent_states,
                        },
                        batch_size=tensor_shape[0:2],
                        device=device,
                    ),
                },
                batch_size=tensor_shape[0:2],
                device=device,
            )
        )

    cat_data = torch.cat(td_list, dim=-1)
    return cat_data
