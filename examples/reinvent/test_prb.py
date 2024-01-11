import torch
from tensordict import TensorDict
from torchrl.data import (
    LazyTensorStorage,
    TensorDictMaxValueWriter,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import PrioritizedSampler


device = torch.device("cuda")
storage = LazyTensorStorage(100, device=device)
experience_replay_buffer = TensorDictReplayBuffer(
    storage=storage,
    sampler=PrioritizedSampler(storage.max_size, alpha=0.7, beta=0.9),
    batch_size=2,
    writer=TensorDictMaxValueWriter(rank_key="priority"),
    priority_key="priority",
)

N = 4
T = 10
obs = torch.rand(N, T, 3)
act = torch.rand(N, T, 2)
priority = torch.rand(N, T)
data = TensorDict({"obs": obs, "act": act, "priority": priority}, batch_size=[N, T])
data.batch_size = torch.Size([N])
data.set(("next", "obs"), torch.rand(N, T, 3))
indices = experience_replay_buffer.extend(
    data.cpu()
)  # Segmentation fault when combining TensorDictMaxValueWriter and PrioritizedSampler
batch = experience_replay_buffer.sample()
