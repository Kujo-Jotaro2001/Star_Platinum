from collections import deque
from pathlib import Path

import numpy as np


class RollingNormalizer:
    """Incremental rolling z-score normalizer with a ring buffer.

    Maintains a deque of the last `window` raw values per feature.
    Computes exact rolling mean and std for z-score normalization.
    """

    def __init__(self, num_features: int, window: int = 500) -> None:
        self._num_features = num_features
        self._window = window
        self._buffers: list[deque[float]] = [
            deque(maxlen=window) for _ in range(num_features)
        ]
        # Cache current mean/std for normalize_only
        self._mean = np.zeros(num_features, dtype=np.float64)
        self._std = np.ones(num_features, dtype=np.float64)

    def update_and_normalize(self, raw: np.ndarray) -> np.ndarray:
        """Push raw values into buffer, update stats, return z-scored.

        Args:
            raw: shape [num_features] float

        Returns:
            shape [num_features] float32
        """
        for i in range(self._num_features):
            self._buffers[i].append(float(raw[i]))

        self._recompute_stats()

        return ((raw - self._mean) / self._std).astype(np.float32)

    def normalize_only(self, raw: np.ndarray) -> np.ndarray:
        """Z-score using frozen stats, no buffer update.

        Args:
            raw: shape [num_features] float

        Returns:
            shape [num_features] float32
        """
        return ((raw - self._mean) / self._std).astype(np.float32)

    def _recompute_stats(self) -> None:
        for i in range(self._num_features):
            buf = self._buffers[i]
            if len(buf) < 2:
                self._mean[i] = buf[0] if buf else 0.0
                self._std[i] = 1.0
            else:
                arr = np.array(buf)
                self._mean[i] = arr.mean()
                s = arr.std()
                self._std[i] = s if s > 1e-10 else 1.0

    def save(self, path: Path) -> None:
        """Save normalizer state to .npz file."""
        # Convert buffers to a 2D array (padded with NaN for unfilled slots)
        buf_array = np.full(
            (self._num_features, self._window), np.nan, dtype=np.float64
        )
        for i, buf in enumerate(self._buffers):
            for j, val in enumerate(buf):
                buf_array[i, j] = val

        buf_lengths = np.array(
            [len(buf) for buf in self._buffers], dtype=np.int32
        )

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

        buf_array = data["buffers"]
        buf_lengths = data["buf_lengths"]
        for i in range(num_features):
            length = int(buf_lengths[i])
            norm._buffers[i] = deque(buf_array[i, :length].tolist(), maxlen=window)

        return norm
