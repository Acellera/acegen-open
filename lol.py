from acegen.rl_env import SMILESEnv
from acegen.vocabulary import SMILESVocabulary

# Create a vocabulary from a list of characters
chars = ["START", "END", "(", ")", "1", "=", "C", "N", "O"]
chars_dict = {char: index for index, char in enumerate(chars)}
vocab = SMILESVocabulary.create_from_dict(
    chars_dict, start_token="START", end_token="END"
)

# Create an environment from the vocabulary
env = SMILESEnv(
    start_token=vocab.start_token_index,
    end_token=vocab.end_token_index,
    length_vocabulary=len(vocab),
    batch_size=4,  # Number of trajectories to collect in parallel
)

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

model = nn.Sequential(
    torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=3),
    torch.nn.LSTM(input_size=3, hidden_size=3, num_layers=1, batch_first=True),
    torch.nn.Linear(in_features=3, out_features=10),
)


class ExamplePolicy(nn.Module):
    """An example policy that takes an observation and returns an action."""

    def __init__(self):
        super(ExamplePolicy, self).__init__()
        self.embed = nn.Embedding(num_embeddings=len(vocab), embedding_dim=3)
        self.rnn = nn.LSTM(
            input_size=3,
            hidden_size=3,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(
            in_features=3,
            out_features=10,
        )

    def forward(self, x, recurrent_state=None):
        x = self.embed(x)
        x, recurrent_state = self.rnn(x, recurrent_state)
        x = self.head(x)
        return x, recurrent_state


policy = TensorDictModule(
    ExamplePolicy(),
    in_keys=["observation", "recurrent_state"],
    out_keys=["action"],
)

print(policy(env.reset()))


import ipdb

ipdb.set_trace()
