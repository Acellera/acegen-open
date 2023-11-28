import torch
from torchrl.envs import Transform


class BurnInTransform(Transform):
    def __init__(self, lstm_module, burn_in):
        super().__init__()
        self.lstm = lstm_module  # .set_recurrent_mode()
        self.burn_in = burn_in

    def forward(self, td):
        td_burn_in = td[..., :self.burn_in]
        td_burn_in = td_burn_in.to(self.lstm.device)
        with torch.no_grad():
            self.lstm(td_burn_in)
        td_out = td[..., self.burn_in:]
        td_out[..., 0].update(td_burn_in[..., -1]["next"])
        return td_out
