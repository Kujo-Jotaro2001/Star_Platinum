import numpy as np
import pytest

from bot.inference import RingBuffer


class TestRingBuffer:
    def test_preserves_chronological_order(self) -> None:
        buf = RingBuffer(capacity=3, item_shape=(2,), dtype=np.float32)

        buf.append(np.array([1, 10], dtype=np.float32))
        buf.append(np.array([2, 20], dtype=np.float32))
        buf.append(np.array([3, 30], dtype=np.float32))

        expected = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32)
        assert buf.is_full is True
        np.testing.assert_array_equal(buf.window(), expected)

    def test_overwrites_oldest_after_capacity(self) -> None:
        buf = RingBuffer(capacity=3, item_shape=(1,), dtype=np.float32)

        for value in range(5):
            buf.append(np.array([value], dtype=np.float32))

        expected = np.array([[2], [3], [4]], dtype=np.float32)
        np.testing.assert_array_equal(buf.window(), expected)

    def test_window_requires_full_buffer(self) -> None:
        buf = RingBuffer(capacity=2, item_shape=(1,), dtype=np.float32)
        buf.append(np.array([1], dtype=np.float32))

        with pytest.raises(ValueError, match="not full"):
            buf.window()
