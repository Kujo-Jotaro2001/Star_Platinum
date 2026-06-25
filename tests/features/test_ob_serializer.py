import numpy as np

from bot.features.ob_serializer import OB_NUM_COLS, serialize_ob_snapshot


def _levels(pairs: list[tuple[str, str]]) -> list[list[str]]:
    return [[p, q] for p, q in pairs]


class TestSerializeOBSnapshot:
    def test_shape(self) -> None:
        bids = _levels([("100.0", "1.0"), ("99.9", "2.0")])
        asks = _levels([("100.1", "1.5"), ("100.2", "2.5")])
        out = serialize_ob_snapshot(bids, asks, ob_depth=10)
        assert out.shape == (10, OB_NUM_COLS)
        assert out.dtype == np.float32

    def test_price_offsets_nonneg(self) -> None:
        bids = _levels([("100.0", "1.0"), ("99.9", "2.0"), ("99.5", "3.0")])
        asks = _levels([("100.1", "1.5"), ("100.3", "2.5"), ("100.9", "4.0")])
        out = serialize_ob_snapshot(bids, asks, ob_depth=5)
        assert (out[:, 0] >= 0).all()
        assert (out[:, 2] >= 0).all()

    def test_padding(self) -> None:
        bids = _levels([("100.0", "1.0"), ("99.9", "2.0")])
        asks = _levels([("100.1", "1.5")])
        out = serialize_ob_snapshot(bids, asks, ob_depth=5)
        # Rows beyond provided levels must be all zeros
        assert (out[2:, 0] == 0).all()
        assert (out[2:, 1] == 0).all()
        assert (out[1:, 2] == 0).all()
        assert (out[1:, 3] == 0).all()

    def test_best_bid_offset_zero(self) -> None:
        bids = _levels([("100.0", "1.0"), ("99.9", "2.0")])
        asks = _levels([("100.1", "1.5")])
        out = serialize_ob_snapshot(bids, asks, ob_depth=5)
        assert out[0, 0] == 0.0

    def test_best_ask_offset_zero(self) -> None:
        bids = _levels([("100.0", "1.0")])
        asks = _levels([("100.1", "1.5"), ("100.2", "2.5")])
        out = serialize_ob_snapshot(bids, asks, ob_depth=5)
        assert out[0, 2] == 0.0

    def test_size_transform(self) -> None:
        bids = _levels([("100.0", "1.0"), ("99.9", "3.0")])
        asks = _levels([("100.1", "0.5"), ("100.2", "7.0")])
        out = serialize_ob_snapshot(bids, asks, ob_depth=5)
        assert out[0, 1] == np.float32(np.log1p(1.0))
        assert out[1, 1] == np.float32(np.log1p(3.0))
        assert out[0, 3] == np.float32(np.log1p(0.5))
        assert out[1, 3] == np.float32(np.log1p(7.0))

    def test_offset_values(self) -> None:
        bids = _levels([("100.0", "1.0"), ("99.0", "2.0")])
        asks = _levels([("101.0", "1.0"), ("102.0", "2.0")])
        out = serialize_ob_snapshot(bids, asks, ob_depth=3)
        assert out[1, 0] == np.float32((100.0 - 99.0) / 100.0)
        assert out[1, 2] == np.float32((102.0 - 101.0) / 101.0)

    def test_empty_sides(self) -> None:
        out = serialize_ob_snapshot([], [], ob_depth=4)
        assert out.shape == (4, OB_NUM_COLS)
        assert (out == 0).all()
