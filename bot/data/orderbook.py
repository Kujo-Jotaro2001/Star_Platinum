import threading
from decimal import Decimal

from bot.data.models import OrderBookLevel, OrderBookSnapshot


class OrderBookManager:
    """Thread-safe container for order book state delivered by pybit.

    pybit applies deltas internally and always delivers the full book
    to the callback. This class holds the latest state and produces
    typed snapshots on demand. Detects reconnections via sequence
    number regression.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict | None = None
        self._last_seq: int = 0
        self._is_reset: bool = True  # first snapshot after init is always a reset

    def update(self, data: dict) -> None:
        """Store latest book data. Called from pybit's WS thread."""
        seq = data.get("seq", 0)
        with self._lock:
            if seq < self._last_seq:
                self._is_reset = True
            self._last_seq = seq
            self._data = data

    def snapshot(self, timestamp_ms: int, depth: int = 50) -> OrderBookSnapshot | None:
        """Produce a typed snapshot. Returns None if no data yet."""
        with self._lock:
            if self._data is None:
                return None
            data = self._data
            is_reset = self._is_reset
            self._is_reset = False

        bids = [
            OrderBookLevel(price=Decimal(p), qty=Decimal(q))
            for p, q in data["b"][:depth]
        ]
        asks = [
            OrderBookLevel(price=Decimal(p), qty=Decimal(q))
            for p, q in data["a"][:depth]
        ]

        return OrderBookSnapshot(
            timestamp_ms=timestamp_ms,
            bids=bids,
            asks=asks,
            is_reset=is_reset,
        )

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._data is not None
