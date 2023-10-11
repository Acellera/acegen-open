# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Sequence

import heapq
import torch

from torchrl.data.replay_buffers.writers import Writer


class TensorDictMaxValueWriter(Writer):
    """A Writer class for composable replay buffers that keeps the top elements based on some filter keys."""

    def __init__(self, rank_key, **kw) -> None:
        super().__init__(**kw)
        self._cursor = 0
        self._current_filter_values = []
        self._rank_key = rank_key

    def add(self, data: Any) -> int:

        ret = None

        # Sum the rank key, in case it is a whole trajectory
        rank_data = data.get("_data")[self._rank_key].sum()

        if rank_data is None:
            raise ValueError(f"Rank key {self._rank_key} not found in data.")

        # If the buffer is not full, add the data
        if len(self._storage) < self._storage.max_size:

            ret = self._cursor
            data["index"] = ret
            self._storage[self._cursor] = data
            self._cursor = (self._cursor + 1) % self._storage.max_size

            # Add new reward to the heap
            heapq.heappush(self._current_filter_values, (rank_data, ret))

        # If the buffer is full, check if the new data is better than the worst data in the buffer
        elif rank_data > self._current_filter_values[0][0]:

            # retrieve position of the smallest value
            min_sample = heapq.heappop(self._current_filter_values)
            min_sample_value = min_sample[1]

            # replace the smallest value with the new value
            self._storage[min_sample_value] = data

            # set new data index
            data["index"] = min_sample_value

            # set return value
            ret = min_sample_value

            # Add new reward to the heap
            heapq.heappush(self._current_filter_values, (rank_data, ret))

        return ret

    def extend(self, data: Sequence) -> torch.Tensor:
        for sample in data:
            self.add(sample)

    def _empty(self):
        self._cursor = 0
        self._current_filter_values = []


if __name__ == "__main__":
    pass

