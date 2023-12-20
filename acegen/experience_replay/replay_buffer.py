import torch
import numpy as np
from tensordict import TensorDict


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

    def sample_smiles(self, n, decode_smiles=False):
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

    def sample_replay_batch(self, batch_size, decode_smiles=False, device="cpu"):
        """Create a TensorDict data batch from replay data."""

        replay_smiles, replay_rewards, _ = self.sample_smiles(batch_size, decode_smiles)

        td_list = []

        for smiles, rew in zip(replay_smiles, replay_rewards):

            encoded = self.voc.encode(smiles)
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

    def __len__(self):
        return len(self.memory)


def collate_fn(arr):
    """Function to take a list of encoded sequences and turn them into a batch"""
    max_length = max([seq.size(0) for seq in arr])
    collated_arr = torch.zeros(len(arr), max_length)
    for i, seq in enumerate(arr):
        collated_arr[i, :seq.size(0)] = seq
    return collated_arr
