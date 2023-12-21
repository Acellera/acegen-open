import torch
from torchrl.envs import Transform


class BurnInTransform(Transform):
    """Transform to burn in the recurrent state of an RNN.

    Args:
        rnn_modules: A list of RNN modules to burn in.
        burn_in: The number of steps to burn in.

    Note:
        This transform assumes that all RNN modules are TensorDict-compatible
        modules and that can handle recurrence in the time dimension (recurrent mode).
    """

    def __init__(self, rnn_modules, burn_in):
        super().__init__()
        self.rnn_modules = rnn_modules
        self.burn_in = burn_in

    def forward(self, td):
        device = td.device or "cpu"
        td_burn_in = td[..., : self.burn_in]
        with torch.no_grad():
            for rnn_module in self.rnn_modules:
                td_burn_in = td_burn_in.to(rnn_module.device)
                td_burn_in = rnn_module(td_burn_in)
        td_burn_in = td_burn_in.to(device)
        td_out = td[..., self.burn_in :]

        # TODO: This is a hack to get the recurrent state from the burn in
        # rhs = torch.zeros(td_out.shape[0], td_out.shape[1], 3, 512)
        # rhs[:, 0].copy_(td_burn_in["next"]["recurrent_state"][:, -1])
        # td_out.set("recurrent_state", rhs)

        td_out[..., 0].update(td_burn_in["next"][..., -1])

        return td_out
