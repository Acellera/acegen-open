from __future__ import annotations

import warnings
from copy import copy
from typing import Sequence

import torch
from tensordict.tensordict import TensorDictBase
from tensordict.utils import NestedKey

from torchrl.data.tensor_specs import ContinuousBox, TensorSpec
from torchrl.envs.transforms.utils import _get_reset, _set_missing_tolerance
from torchrl.envs.transforms.transforms import ObservationTransform, _apply_to_composite, IMAGE_KEYS


class CatFrames(ObservationTransform):
    """Concatenates successive observation frames into a single tensor.

    This can, for instance, account for movement/velocity of the observed
    feature. Proposed in "Playing Atari with Deep Reinforcement Learning" (
    https://arxiv.org/pdf/1312.5602.pdf).

    When used within a transformed environment,
    :class:`CatFrames` is a stateful class, and it can be reset to its native state by
    calling the :meth:`~.reset` method. This method accepts tensordicts with a
    ``"_reset"`` entry that indicates which buffer to reset.

    Args:
        N (int): number of observation to concatenate.
        dim (int): dimension along which concatenate the
            observations. Should be negative, to ensure that it is compatible
            with environments of different batch_size.
        in_keys (sequence of NestedKey, optional): keys pointing to the frames that have
            to be concatenated. Defaults to ["pixels"].
        out_keys (sequence of NestedKey, optional): keys pointing to where the output
            has to be written. Defaults to the value of `in_keys`.
        padding (str, optional): the padding method. One of ``"same"`` or ``"constant"``.
            Defaults to ``"same"``, ie. the first value is used for padding.
        padding_value (float, optional): the value to use for padding if ``padding="constant"``.
        as_inverse (bool, optional): if ``True``, the transform is applied as an inverse transform. Defaults to ``False``.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(GymEnv('Pendulum-v1'),
        ...     Compose(
        ...         UnsqueezeTransform(-1, in_keys=["observation"]),
        ...         CatFrames(N=4, dim=-1, in_keys=["observation"]),
        ...     )
        ... )
        >>> print(env.rollout(3))

    The :class:`CatFrames` transform can also be used offline to reproduce the
    effect of the online frame concatenation at a different scale (or for the
    purpose of limiting the memory consumption). The followin example
    gives the complete picture, together with the usage of a :class:`torchrl.data.ReplayBuffer`:

    Examples:
        >>> from torchrl.envs import UnsqueezeTransform, CatFrames
        >>> from torchrl.collectors import SyncDataCollector, RandomPolicy
        >>> # Create a transformed environment with CatFrames: notice the usage of UnsqueezeTransform to create an extra dimension
        >>> env = TransformedEnv(
        ...     GymEnv("CartPole-v1", from_pixels=True),
        ...     Compose(
        ...         ToTensorImage(in_keys=["pixels"], out_keys=["pixels_trsf"]),
        ...         Resize(in_keys=["pixels_trsf"], w=64, h=64),
        ...         GrayScale(in_keys=["pixels_trsf"]),
        ...         UnsqueezeTransform(-4, in_keys=["pixels_trsf"]),
        ...         CatFrames(dim=-4, N=4, in_keys=["pixels_trsf"]),
        ...     )
        ... )
        >>> # we design a collector
        >>> collector = SyncDataCollector(
        ...     env,
        ...     RandomPolicy(env.action_spec),
        ...     frames_per_batch=10,
        ...     total_frames=1000,
        ... )
        >>> for data in collector:
        ...     print(data)
        ...     break
        >>> # now let's create a transform for the replay buffer. We don't need to unsqueeze the data here.
        >>> # however, we need to point to both the pixel entry at the root and at the next levels:
        >>> t = Compose(
        ...         ToTensorImage(in_keys=["pixels", ("next", "pixels")], out_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        ...         Resize(in_keys=["pixels_trsf", ("next", "pixels_trsf")], w=64, h=64),
        ...         GrayScale(in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        ...         CatFrames(dim=-4, N=4, in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        ... )
        >>> from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
        >>> rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(1000), transform=t, batch_size=16)
        >>> data_exclude = data.exclude("pixels_trsf", ("next", "pixels_trsf"))
        >>> rb.add(data_exclude)
        >>> s = rb.sample(1) # the buffer has only one element
        >>> # let's check that our sample is the same as the batch collected during inference
        >>> assert (data.exclude("collector")==s.squeeze(0).exclude("index", "collector")).all()

    .. note:: :class:`~CatFrames` currently only supports ``"done"``
        signal at the root. Nested ``done``,
        such as those found in MARL settings, are currently not supported.
        If this feature is needed, please raise an issue on TorchRL repo.

    """

    inplace = False
    _CAT_DIM_ERR = (
        "dim must be < 0 to accomodate for tensordict of "
        "different batch-sizes (since negative dims are batch invariant)."
    )
    ACCEPTED_PADDING = {"same", "constant"}

    def __init__(
            self,
            N: int,
            dim: int,
            in_keys: Sequence[NestedKey] | None = None,
            out_keys: Sequence[NestedKey] | None = None,
            padding="same",
            padding_value=0,
            as_inverse=False,
            reset_key: NestedKey | None = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.N = N
        if dim >= 0:
            raise ValueError(self._CAT_DIM_ERR)
        self.dim = dim
        if padding not in self.ACCEPTED_PADDING:
            raise ValueError(f"padding must be one of {self.ACCEPTED_PADDING}")
        self.padding = padding
        self.padding_value = padding_value
        for in_key in self.in_keys:
            buffer_name = f"_cat_buffers_{in_key}"
            self.register_buffer(
                buffer_name,
                torch.nn.parameter.UninitializedBuffer(
                    device=torch.device("cpu"), dtype=torch.get_default_dtype()
                ),
            )
        # keeps track of calls to _reset since it's only _call that will populate the buffer
        self.as_inverse = as_inverse
        self.reset_key = reset_key

    @property
    def reset_key(self):
        reset_key = self.__dict__.get("_reset_key", None)
        if reset_key is None:
            reset_keys = self.parent.reset_keys
            if len(reset_keys) > 1:
                raise RuntimeError(
                    f"Got more than one reset key in env {self.container}, cannot infer which one to use. "
                    f"Consider providing the reset key in the {type(self)} constructor."
                )
            reset_key = self._reset_key = reset_keys[0]
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    def _reset(
            self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets _buffers."""
        _reset = _get_reset(self.reset_key, tensordict)
        if self.as_inverse and self.parent is not None:
            raise Exception(
                "CatFrames as inverse is not supported as a transform for environments, only for replay buffers."
            )

        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset, _reset=_reset)

        return tensordict_reset

    def _make_missing_buffer(self, data, buffer_name):
        shape = list(data.shape)
        d = shape[self.dim]
        shape[self.dim] = d * self.N
        shape = torch.Size(shape)
        getattr(self, buffer_name).materialize(shape)
        buffer = (
            getattr(self, buffer_name)
            .to(data.dtype)
            .to(data.device)
            .fill_(self.padding_value)
        )
        setattr(self, buffer_name, buffer)
        return buffer

    def _inv_call(self, tensordict: TensorDictBase) -> torch.Tensor:
        if self.as_inverse:
            return self.unfolding(tensordict)
        else:
            return tensordict

    def _call(self, tensordict: TensorDictBase, _reset=None) -> TensorDictBase:
        """Update the episode tensordict with max pooled keys."""
        _just_reset = _reset is not None
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = f"_cat_buffers_{in_key}"
            data = tensordict.get(in_key)
            d = data.size(self.dim)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                buffer = self._make_missing_buffer(data, buffer_name)
            # shift obs 1 position to the right
            if _just_reset:
                if _reset.all():
                    _all = True
                    data_reset = data
                    buffer_reset = buffer
                    dim = self.dim
                else:
                    _all = False
                    data_reset = data[_reset]
                    buffer_reset = buffer[_reset]
                    dim = self.dim - _reset.ndim + 1
                shape = [1 for _ in buffer_reset.shape]
                if _all:
                    shape[dim] = self.N
                else:
                    shape[dim] = self.N

                if self.padding == "same":
                    if _all:
                        buffer.copy_(data_reset.repeat(shape).clone())
                    else:
                        buffer[_reset] = data_reset.repeat(shape).clone()
                elif self.padding == "constant":
                    if _all:
                        buffer.fill_(self.padding_value)
                    else:
                        buffer[_reset] = self.padding_value
                else:
                    # make linter happy. An exception has already been raised
                    raise NotImplementedError

                # # this duplicates the code below, but only for _reset values
                # if _all:
                #     buffer.copy_(torch.roll(buffer_reset, shifts=-d, dims=dim))
                #     buffer_reset = buffer
                # else:
                #     buffer_reset = buffer[_reset] = torch.roll(
                #         buffer_reset, shifts=-d, dims=dim
                #     )
                # add new obs
                if self.dim < 0:
                    n = buffer_reset.ndimension() + self.dim
                else:
                    raise ValueError(self._CAT_DIM_ERR)
                idx = [slice(None, None) for _ in range(n)] + [slice(-d, None)]
                if not _all:
                    buffer_reset = buffer[_reset]
                buffer_reset[idx] = data_reset
                if not _all:
                    buffer[_reset] = buffer_reset
            else:
                buffer.copy_(torch.roll(buffer, shifts=-d, dims=self.dim))
                # add new obs
                if self.dim < 0:
                    n = buffer.ndimension() + self.dim
                else:
                    raise ValueError(self._CAT_DIM_ERR)
                idx = [slice(None, None) for _ in range(n)] + [slice(-d, None)]
                buffer[idx] = buffer[idx].copy_(data)
            # add to tensordict
            tensordict.set(out_key, buffer.clone())
        return tensordict

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = torch.cat([space.low] * self.N, self.dim)
            space.high = torch.cat([space.high] * self.N, self.dim)
            observation_spec.shape = space.low.shape
        else:
            shape = list(observation_spec.shape)
            shape[self.dim] = self.N * shape[self.dim]
            observation_spec.shape = torch.Size(shape)
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.as_inverse:
            return tensordict
        else:
            return self.unfolding(tensordict)

    def unfolding(self, tensordict: TensorDictBase) -> TensorDictBase:
        # it is assumed that the last dimension of the tensordict is the time dimension
        if not tensordict.ndim:
            raise ValueError(
                "CatFrames cannot process unbatched tensordict instances. "
                "Make sure your input has more than one dimension and "
                "the time dimension is marked as 'time', e.g., "
                "`tensordict.refine_names(None, 'time', None)`."
            )
        i = 0
        for i, name in enumerate(tensordict.names):  # noqa: B007
            if name == "time":
                break
        else:
            warnings.warn(
                "The last dimension of the tensordict should be marked as 'time'. "
                "CatFrames will unfold the data along the time dimension assuming that "
                "the time dimension is the last dimension of the input tensordict. "
                "Define a 'time' dimension name (e.g., `tensordict.refine_names(..., 'time')`) to skip this warning. ",
                category=UserWarning,
            )
        tensordict_orig = tensordict
        if i != tensordict.ndim - 1:
            tensordict = tensordict.transpose(tensordict.ndim - 1, i)
        # first sort the in_keys with strings and non-strings
        in_keys = list(
            zip(
                (in_key, out_key)
                for in_key, out_key in zip(self.in_keys, self.out_keys)
                if isinstance(in_key, str) or len(in_key) == 1
            )
        )
        in_keys += list(
            zip(
                (in_key, out_key)
                for in_key, out_key in zip(self.in_keys, self.out_keys)
                if not isinstance(in_key, str) and not len(in_key) == 1
            )
        )
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            # check if we have an obs in "next" that has already been processed.
            # If so, we must add an offset
            data = tensordict.get(in_key)
            if isinstance(in_key, tuple) and in_key[0] == "next":
                # let's get the out_key we have already processed
                prev_out_key = dict(zip(self.in_keys, self.out_keys))[in_key[1]]
                prev_val = tensordict.get(prev_out_key)
                # the first item is located along `dim+1` at the last index of the
                # first time index
                idx = (
                        [slice(None)] * (tensordict.ndim - 1)
                        + [0]
                        + [..., -1]
                        + [slice(None)] * (abs(self.dim) - 1)
                )
                first_val = prev_val[tuple(idx)].unsqueeze(tensordict.ndim - 1)
                data0 = [first_val] * (self.N - 1)
                if self.padding == "constant":
                    data0 = [
                                torch.ones_like(elt) * self.padding_value for elt in data0[:-1]
                            ] + data0[-1:]
                elif self.padding == "same":
                    pass
                else:
                    # make linter happy. An exception has already been raised
                    raise NotImplementedError
            elif self.padding == "same":
                idx = [slice(None)] * (tensordict.ndim - 1) + [0]
                data0 = [data[tuple(idx)].unsqueeze(tensordict.ndim - 1)] * (self.N - 1)
            elif self.padding == "constant":
                idx = [slice(None)] * (tensordict.ndim - 1) + [0]
                data0 = [
                            torch.ones_like(data[tuple(idx)]).unsqueeze(tensordict.ndim - 1)
                            * self.padding_value
                        ] * (self.N - 1)
            else:
                # make linter happy. An exception has already been raised
                raise NotImplementedError

            data = torch.cat(data0 + [data], tensordict.ndim - 1)

            data = data.unfold(tensordict.ndim - 1, self.N, 1)
            data = data.permute(
                *range(0, data.ndim + self.dim),
                -1,
                *range(data.ndim + self.dim, data.ndim - 1),
            )
            tensordict.set(out_key, data)
        return tensordict_orig

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(N={self.N}, dim"
            f"={self.dim}, keys={self.in_keys})"
        )
