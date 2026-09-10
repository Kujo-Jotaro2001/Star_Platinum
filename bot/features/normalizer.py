from pathlib import Path

import numpy as np

EXACT_REFRESH_EVERY = 8192
STD_FLOOR = 1e-10


class RollingNormalizer:
    """Incremental rolling z-score normalizer over a fixed window.

    Holds the last `window` raw rows in a ring buffer and carries running sums,
    so each update costs one vector operation instead of rebuilding the window's
    statistics. Rebuilding them per row is O(window) and dominates a pipeline run
    over millions of snapshots; the sums are refreshed exactly every
    `EXACT_REFRESH_EVERY` updates so subtraction never accumulates drift.
    """

    def __init__(self, num_features: int, window: int = 500) -> None:
        self._num_features = num_features
        self._window = window
        self._ring = np.zeros((window, num_features), dtype=np.float64)
        self._size = 0
        self._next = 0
        self._updates = 0
        self._sum = np.zeros(num_features, dtype=np.float64)
        self._sum_sq = np.zeros(num_features, dtype=np.float64)

        self._mean = np.zeros(num_features, dtype=np.float64)
        self._std = np.ones(num_features, dtype=np.float64)

    def update_and_normalize(self, raw: np.ndarray) -> np.ndarray:
        """Push raw values into the window, refresh stats, return the z-score.

        Args:
            raw: shape [num_features] float

        Returns:
            shape [num_features] float32
        """
        values = np.asarray(raw, dtype=np.float64)

        if self._size == self._window:
            evicted = self._ring[self._next]
            self._sum -= evicted
            self._sum_sq -= evicted * evicted
        else:
            self._size += 1

        self._ring[self._next] = values
        self._sum += values
        self._sum_sq += values * values
        self._next = (self._next + 1) % self._window

        self._updates += 1
        if self._updates % EXACT_REFRESH_EVERY == 0:
            self._refresh_sums()
        self._recompute_stats()

        return ((values - self._mean) / self._std).astype(np.float32)

    def normalize_only(self, raw: np.ndarray) -> np.ndarray:
        """Z-score using frozen stats, no window update.

        Args:
            raw: shape [num_features] float

        Returns:
            shape [num_features] float32
        """
        return ((raw - self._mean) / self._std).astype(np.float32)

    def _window_view(self) -> np.ndarray:
        """The window in insertion order, oldest first."""
        if self._size < self._window:
            return self._ring[: self._size]
        return np.concatenate((self._ring[self._next:], self._ring[: self._next]))

    def _refresh_sums(self) -> None:
        data = self._window_view()
        self._sum = data.sum(axis=0)
        self._sum_sq = (data * data).sum(axis=0)

    def _recompute_stats(self) -> None:
        if self._size < 2:
            self._mean = (
                self._ring[0].copy() if self._size else np.zeros(self._num_features)
            )
            self._std = np.ones(self._num_features, dtype=np.float64)
            return

        mean = self._sum / self._size
        var = np.maximum(self._sum_sq / self._size - mean * mean, 0.0)
        std = np.sqrt(var)
        self._mean = mean
        self._std = np.where(std > STD_FLOOR, std, 1.0)

    def save(self, path: Path) -> None:
        """Save normalizer state to .npz file."""
        buf_array = np.full(
            (self._num_features, self._window), np.nan, dtype=np.float64
        )
        data = self._window_view()
        if self._size:
            buf_array[:, : self._size] = data.T

        buf_lengths = np.full(self._num_features, self._size, dtype=np.int32)

        np.savez(
            path,
            mean=self._mean,
            std=self._std,
            window=np.array(self._window),
            num_features=np.array(self._num_features),
            buffers=buf_array,
            buf_lengths=buf_lengths,
        )

    @classmethod
    def load(cls, path: Path) -> "RollingNormalizer":
        """Restore normalizer from .npz file."""
        data = np.load(path)
        num_features = int(data["num_features"])
        window = int(data["window"])

        norm = cls(num_features=num_features, window=window)
        norm._mean = data["mean"].copy()
        norm._std = data["std"].copy()

        size = int(data["buf_lengths"][0]) if num_features else 0
        if size:
            norm._ring[:size] = data["buffers"][:, :size].T
            norm._size = size
            norm._next = size % window
            norm._refresh_sums()

        return norm
