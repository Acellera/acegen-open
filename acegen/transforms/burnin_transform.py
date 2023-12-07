import torch
from torchrl.envs import Transform


class BurnInTransform(Transform):
    def __init__(self, rnn_modules, burn_in):
        super().__init__()
        self.rnn_modules = rnn_modules
        self.burn_in = burn_in

    def forward(self, td):
        device = td.device or "cpu"
        td_burn_in = td[..., :self.burn_in]
        with torch.no_grad():
            for rnn_module in self.rnn_modules:
                td_burn_in = td_burn_in.to(rnn_module.device)
                td_burn_in = rnn_module(td_burn_in)
        td_burn_in = td_burn_in.to(device)
        td_out = td[..., self.burn_in:]
        td_out[..., 0].update(td_burn_in[..., -1]["next"])
        return td_out
