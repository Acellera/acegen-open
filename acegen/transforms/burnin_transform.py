from __future__ import annotations

from typing import Sequence

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torchrl.envs import Transform


class BurnInTransform(Transform):
    """Transform to partially burn in data sequences.

    This transform is useful for obtaining up-to-date recurrent states when
    they are not available by burning in a few steps along the time dimension
    from sampled data slices. It is intended to be used as a replay buffer
    transform, not as an environment transform.

    Note:
        This transform expects TensorDicts with its last dimension being the
        time dimension. It also  assumes that the modules can process the
        sequential data.

    Args:
        modules (sequence of TensorDictModule): A list of modules to burn in.
        burn_in (int): The number of time steps to burn in.
        out_keys (sequence of NestedKey, optional): destination keys. defaults to
        all out keys of the modules.

    Examples:
        >>> import torch
        >>> from torchrl.envs import TensorDict
        >>> from torchrl.envs.transforms import BurnInTransform
        >>> from torchrl.modules import GRUModule

        >>> burn_in_transform = BurnInTransform(
        ...     modules=[GRUModule(1, 1, batch_first=True).set_recurrent_mode(True)],
        ...     burn_in=5,
        ... )

    """

    def __init__(
        self,
        modules: Sequence[torch.nn.Module],
        burn_in: int,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        self.modules = modules
        self.burn_in = burn_in

        for module in self.modules:
            if not isinstance(module, TensorDictModuleBase):
                raise ValueError(
                    f"All modules must be TensorDictModules, not {type(module)}."
                )

        in_keys = set()
        for module in self.modules:
            in_keys.update(module.in_keys)

        if out_keys is None:
            out_keys = set()
            for module in self.modules:
                for key in module.out_keys:
                    if key[0] == "next":
                        out_keys.add(key[1])

        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(
            "BurnInTransform can only be used when appended to a ReplayBuffer."
        )

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        raise RuntimeError(
            "BurnInTransform can only be used when appended to a ReplayBuffer."
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_device = tensordict.device or "cpu"
        B, T, *extra_dims = tensordict.batch_size

        # Split the tensor dict into the burn in and the rest.
        td_burn_in = tensordict[..., : self.burn_in]
        td_out = tensordict[..., self.burn_in :]

        # Burn in the recurrent state.
        with torch.no_grad():
            for module in self.modules:
                module_device = next(module.parameters()).device or "cpu"
                td_burn_in = td_burn_in.to(module_device)
                td_burn_in = module(td_burn_in)
        td_burn_in = td_burn_in.to(td_device)

        # Update out TensorDict with the burnt in data.
        for out_key in self.out_keys:
            if out_key not in td_out.keys():
                td_out.set(
                    out_key,
                    torch.zeros(
                        B, T - self.burn_in, *tensordict.get(out_key).shape[2:]
                    ),
                )
            td_out[..., 0][out_key].copy_(td_burn_in["next"][..., -1][out_key])

        return td_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(burn_in={self.burn_in})"
