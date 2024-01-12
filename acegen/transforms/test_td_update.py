import torch
from tensordict import TensorDict

N = 1000
T = 1000

obs = torch.randint(0, 10, (N, T))
actions = torch.randint(0, 10, (N, T))
reward = torch.zeros(N, T)
done = torch.randint(0, 2, (N, T), dtype=torch.bool)

data = TensorDict(
    {"obs": obs, "actions": actions, "reward": reward, "done": done}, batch_size=[N]
)

# print(data["reward"].sum())

done = data.get("done")
data.get("reward")[done] = 1.0

# print(data["reward"].sum())
