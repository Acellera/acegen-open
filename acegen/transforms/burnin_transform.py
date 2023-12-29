from __future__ import annotations

from typing import Sequence

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torchrl.envs import Transform


class BurnInTransform(Transform):
    """Transform to partially burn-in data sequences.

    This transform is useful to obtain up-to-date recurrent states when
    they are not available. It burns-in a number of steps along the time dimension
    from sampled sequential data slices and returs the remaining data sequence with
    the burnt in data in its initial time step. It is intended to be used as a
    replay buffer transform, not as an environment transform.

    Args:
        modules (sequence of TensorDictModule): A list of modules to burn in.
        burn_in (int): The number of time steps to burn in.
        out_keys (sequence of NestedKey, optional): destination keys. defaults to
        all the modules out keys that point to the next time step (e.g. ("next", "hidden")).

    .. note::
        This transform expects TensorDicts with its last dimension being the
        time dimension. It also  assumes that all provided modules can process
        sequential data.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import BurnInTransform
        >>> from torchrl.modules import GRUModule

        >>> gru_module = GRUModule(
        ...     input_size=10,
        ...     hidden_size=10,
        ...     in_keys=["observation", "hidden"],
        ...     out_keys=["intermediate", ("next", "hidden")],
        ... ).set_recurrent_mode(True)
        >>> burn_in_transform = BurnInTransform(
        ...     modules=[gru_module],
        ...     burn_in=5,
        ... )
        >>> td = TensorDict({
        ...     "observation": torch.randn(2, 10, 10),
        ...      "hidden": torch.randn(2, 10, gru_module.gru.num_layers, 10),
        ...      "is_init": torch.zeros(2, 10, 1),
        ... }, batch_size=[2, 10])
        >>> td = burn_in_transform(td)
        >>> td.shape
        torch.Size([2, 5])
        >>> td.get("hidden").abs().sum()
        tensor(86.3008)

        >>> from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
        >>> buffer = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(2),
        ...     batch_size=1,
        ... )
        >>> buffer.append_transform(burn_in_transform)
        >>> td = TensorDict({
        ...     "observation": torch.randn(2, 10, 10),
        ...      "hidden": torch.randn(2, 10, gru_module.gru.num_layers, 10),
        ...      "is_init": torch.zeros(2, 10, 1),
        ... }, batch_size=[2, 10])
        >>> buffer.extend(td)
        >>> td = buffer.sample(1)
        >>> td.shape
        torch.Size([1, 5])
        >>> td.get("hidden").abs().sum()
        tensor(37.0344)
    """

    invertible = False

    def __init__(
        self,
        modules: Sequence[TensorDictModuleBase],
        burn_in: int,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        self.modules = modules
        self.burn_in = burn_in

        if not isinstance(modules, Sequence):
            self.modules = [modules]
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

        if self.burn_in == 0:
            return tensordict

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
        return f"{self.__class__.__name__}(burn_in={self.burn_in}, in_keys={self.in_keys}, out_keys={self.out_keys})"
