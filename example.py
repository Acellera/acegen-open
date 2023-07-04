import torch
from torchrl.envs import TransformedEnv, InitTracker, step_mdp
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, LSTMModule
from tensordict import TensorDict
from torch import nn
from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod

env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
lstm = nn.LSTM(input_size=env.observation_spec["observation"].shape[-1], hidden_size=64, batch_first=True)
lstm_module = LSTMModule(lstm,
            in_keys=["observation", "hidden0", "hidden1"],
            out_keys=["intermediate", ("next", "hidden0"), ("next", "hidden1")]
)
mlp = MLP(num_cells=[64], out_features=1)
 # building two policies with different behaviours:
policy_inference = Seq(lstm_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
policy_training = Seq(lstm_module.set_recurrent_mode(True), Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
traj_td = env.rollout(3)  # some random temporal data
traj_td = policy_training(traj_td)
# let's check that both return the same results
td_inf = TensorDict({}, traj_td.shape[:-1])

for td in traj_td.unbind(-1):

    td_inf = td_inf.update(td.select("is_init", "observation", ("next", "observation")))
    td_inf = policy_inference(td_inf)
    td_inf = step_mdp(td_inf)
    torch.testing.assert_close(td_inf["hidden0"], traj_td[..., -1]["next", "hidden0"])