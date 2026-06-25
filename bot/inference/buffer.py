from collections.abc import Sequence

import numpy as np


class RingBuffer:
    """Fixed-size numpy ring buffer for one feature stream."""

    def __init__(
        self,
        capacity: int,
        item_shape: Sequence[int],
        dtype: np.dtype | type = np.float32,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._item_shape = tuple(item_shape)
        self._dtype = np.dtype(dtype)
        self._data = np.zeros((capacity, *self._item_shape), dtype=self._dtype)
        self._next_idx = 0
        self._size = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return self._size == self._capacity

    def append(self, item: np.ndarray) -> None:
        arr = np.asarray(item, dtype=self._dtype)
        if arr.shape != self._item_shape:
            raise ValueError(f"expected item shape {self._item_shape}, got {arr.shape}")

        self._data[self._next_idx] = arr
        self._next_idx = (self._next_idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def window(self) -> np.ndarray:
        if not self.is_full:
            raise ValueError("buffer is not full")
        if self._next_idx == 0:
            return self._data.copy()
        return np.concatenate(
            (self._data[self._next_idx:], self._data[:self._next_idx]),
            axis=0,
        ).copy()
