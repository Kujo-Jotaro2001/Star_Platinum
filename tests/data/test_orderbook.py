from decimal import Decimal

from bot.data.orderbook import OrderBookManager


def _make_book_data(
    bids: list[list[str]],
    asks: list[list[str]],
    seq: int = 1,
) -> dict:
    return {"b": bids, "a": asks, "s": "BTCUSDT", "u": 1, "seq": seq}


class TestOrderBookManager:
    def test_not_ready_initially(self) -> None:
        ob = OrderBookManager()
        assert not ob.is_ready

    def test_ready_after_update(self) -> None:
        ob = OrderBookManager()
        ob.update(_make_book_data(
            bids=[["100.0", "1.0"]],
            asks=[["101.0", "2.0"]],
        ))
        assert ob.is_ready

    def test_snapshot_returns_none_when_empty(self) -> None:
        ob = OrderBookManager()
        assert ob.snapshot(timestamp_ms=1000) is None

    def test_snapshot_converts_to_decimal(self) -> None:
        ob = OrderBookManager()
        ob.update(_make_book_data(
            bids=[["100.50", "1.5"]],
            asks=[["101.25", "2.3"]],
        ))
        snap = ob.snapshot(timestamp_ms=1000)
        assert snap is not None
        assert snap.bids[0].price == Decimal("100.50")
        assert snap.bids[0].qty == Decimal("1.5")
        assert snap.asks[0].price == Decimal("101.25")
        assert snap.asks[0].qty == Decimal("2.3")

    def test_first_snapshot_is_reset(self) -> None:
        ob = OrderBookManager()
        ob.update(_make_book_data(
            bids=[["100.0", "1.0"]],
            asks=[["101.0", "2.0"]],
        ))
        snap = ob.snapshot(timestamp_ms=1000)
        assert snap is not None
        assert snap.is_reset is True

    def test_subsequent_snapshot_not_reset(self) -> None:
        ob = OrderBookManager()
        ob.update(_make_book_data(
            bids=[["100.0", "1.0"]],
            asks=[["101.0", "2.0"]],
            seq=1,
        ))
        ob.snapshot(timestamp_ms=1000)  # consume the reset

        ob.update(_make_book_data(
            bids=[["100.0", "1.1"]],
            asks=[["101.0", "2.1"]],
            seq=2,
        ))
        snap = ob.snapshot(timestamp_ms=2000)
        assert snap is not None
        assert snap.is_reset is False

    def test_seq_regression_marks_reset(self) -> None:
        ob = OrderBookManager()
        ob.update(_make_book_data(
            bids=[["100.0", "1.0"]],
            asks=[["101.0", "2.0"]],
            seq=10,
        ))
        ob.snapshot(timestamp_ms=1000)  # consume the initial reset

        # Simulate reconnect: seq drops back
        ob.update(_make_book_data(
            bids=[["100.0", "1.0"]],
            asks=[["101.0", "2.0"]],
            seq=1,
        ))
        snap = ob.snapshot(timestamp_ms=2000)
        assert snap is not None
        assert snap.is_reset is True

    def test_depth_limit(self) -> None:
        ob = OrderBookManager()
        bids = [[str(100 - i), "1.0"] for i in range(10)]
        asks = [[str(101 + i), "1.0"] for i in range(10)]
        ob.update(_make_book_data(bids=bids, asks=asks))

        snap = ob.snapshot(timestamp_ms=1000, depth=3)
        assert snap is not None
        assert len(snap.bids) == 3
        assert len(snap.asks) == 3

    def test_timestamp_passed_through(self) -> None:
        ob = OrderBookManager()
        ob.update(_make_book_data(
            bids=[["100.0", "1.0"]],
            asks=[["101.0", "2.0"]],
        ))
        snap = ob.snapshot(timestamp_ms=1234567890)
        assert snap is not None
        assert snap.timestamp_ms == 1234567890
