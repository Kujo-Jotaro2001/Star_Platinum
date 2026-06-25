import asyncio
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from bot.data.models import (
    Liquidation,
    LongShortRatio,
    OrderBookLevel,
    OrderBookSnapshot,
    TickerContext,
    Trade,
)
from bot.data.storage import StorageWriter


def _snap(ts: int = 1712300000000, is_reset: bool = False) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp_ms=ts,
        bids=[OrderBookLevel(Decimal("100.0"), Decimal("1.0"))],
        asks=[OrderBookLevel(Decimal("101.0"), Decimal("2.0"))],
        is_reset=is_reset,
    )


def _trade(ts: int = 1712300000000) -> Trade:
    return Trade(
        timestamp_ms=ts,
        trade_id="abc-123",
        side="Buy",
        price=Decimal("100.50"),
        qty=Decimal("0.5"),
    )


def _ticker(ts: int = 1712300000000, partial: bool = False) -> TickerContext:
    return TickerContext(
        timestamp_ms=ts,
        mark_price=Decimal("100.0") if not partial else None,
        index_price=Decimal("99.9") if not partial else None,
        open_interest=Decimal("12345.6"),
        open_interest_value=Decimal("1234560.0"),
        funding_rate=Decimal("0.0001"),
        next_funding_time=1712328800000,
        price_24h_pct=Decimal("0.02"),
        prev_price_1h=Decimal("99.5"),
        volume_24h=Decimal("5000.0"),
        turnover_24h=Decimal("500000.0"),
        collected_at_ms=ts + 1,
    )


def _liquidation(ts: int = 1712300000000) -> Liquidation:
    return Liquidation(
        timestamp_ms=ts,
        side="Sell",
        qty=Decimal("0.3"),
        price=Decimal("99.0"),
        collected_at_ms=ts + 1,
    )


def _ls_ratio(ts: int = 1712300000000) -> LongShortRatio:
    return LongShortRatio(
        timestamp_ms=ts,
        buy_ratio=Decimal("0.55"),
        sell_ratio=Decimal("0.45"),
        collected_at_ms=ts + 1,
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


class TestStorageWriter:
    async def test_writes_snapshot(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT")
        shutdown = asyncio.Event()

        storage.put_nowait(_snap(ts=1712300000000))
        shutdown.set()
        await storage.run(shutdown)

        db_files = list(data_dir.glob("*.db"))
        assert len(db_files) == 1
        assert "BTCUSDT_2024-04-05" in db_files[0].name

        conn = sqlite3.connect(db_files[0])
        rows = conn.execute("SELECT * FROM snapshots").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == 1712300000000
        assert rows[0][1] == 0  # is_reset
        assert json.loads(rows[0][2]) == [["100.0", "1.0"]]

    async def test_writes_trade(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT")
        shutdown = asyncio.Event()

        storage.put_nowait(_trade())
        shutdown.set()
        await storage.run(shutdown)

        db_files = list(data_dir.glob("*.db"))
        conn = sqlite3.connect(db_files[0])
        rows = conn.execute("SELECT * FROM trades").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][1] == "abc-123"
        assert rows[0][2] == "Buy"
        assert rows[0][3] == "100.50"

    async def test_is_reset_flag_stored(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT")
        shutdown = asyncio.Event()

        storage.put_nowait(_snap(is_reset=True))
        shutdown.set()
        await storage.run(shutdown)

        db_files = list(data_dir.glob("*.db"))
        conn = sqlite3.connect(db_files[0])
        row = conn.execute("SELECT is_reset FROM snapshots").fetchone()
        conn.close()
        assert row[0] == 1

    async def test_writes_ticker_context(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT")
        shutdown = asyncio.Event()

        storage.put_nowait(_ticker(ts=1712300000000))
        shutdown.set()
        await storage.run(shutdown)

        db_files = list(data_dir.glob("*.db"))
        conn = sqlite3.connect(db_files[0])
        row = conn.execute(
            "SELECT timestamp_ms, mark_price, funding_rate, next_funding_time, collected_at_ms FROM ticker_context"
        ).fetchone()
        conn.close()
        assert row[0] == 1712300000000
        assert row[1] == "100.0"
        assert row[2] == "0.0001"
        assert row[3] == 1712328800000
        assert row[4] == 1712300000001

    async def test_writes_ticker_context_with_nulls(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT")
        shutdown = asyncio.Event()

        storage.put_nowait(_ticker(ts=1712300000000, partial=True))
        shutdown.set()
        await storage.run(shutdown)

        db_files = list(data_dir.glob("*.db"))
        conn = sqlite3.connect(db_files[0])
        row = conn.execute(
            "SELECT mark_price, index_price, open_interest FROM ticker_context"
        ).fetchone()
        conn.close()
        assert row[0] is None
        assert row[1] is None
        assert row[2] == "12345.6"

    async def test_writes_liquidation(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT")
        shutdown = asyncio.Event()

        storage.put_nowait(_liquidation())
        shutdown.set()
        await storage.run(shutdown)

        db_files = list(data_dir.glob("*.db"))
        conn = sqlite3.connect(db_files[0])
        row = conn.execute(
            "SELECT timestamp_ms, side, qty, price FROM liquidations"
        ).fetchone()
        conn.close()
        assert row == (1712300000000, "Sell", "0.3", "99.0")

    async def test_writes_long_short_ratio(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT")
        shutdown = asyncio.Event()

        storage.put_nowait(_ls_ratio())
        shutdown.set()
        await storage.run(shutdown)

        db_files = list(data_dir.glob("*.db"))
        conn = sqlite3.connect(db_files[0])
        row = conn.execute(
            "SELECT timestamp_ms, buy_ratio, sell_ratio FROM long_short_ratio"
        ).fetchone()
        conn.close()
        assert row == (1712300000000, "0.55", "0.45")

    async def test_long_short_ratio_unique_constraint(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT")
        shutdown = asyncio.Event()

        storage.put_nowait(_ls_ratio(ts=1712300000000))
        storage.put_nowait(_ls_ratio(ts=1712300000000))  # duplicate
        shutdown.set()
        await storage.run(shutdown)

        db_files = list(data_dir.glob("*.db"))
        conn = sqlite3.connect(db_files[0])
        count = conn.execute("SELECT COUNT(*) FROM long_short_ratio").fetchone()[0]
        conn.close()
        assert count == 1

    async def test_daily_rotation(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT")
        shutdown = asyncio.Event()

        storage.put_nowait(_snap(ts=1712300000000))  # 2024-04-05
        storage.put_nowait(_snap(ts=1712400000000))  # 2024-04-06
        shutdown.set()
        await storage.run(shutdown)

        db_files = sorted(data_dir.glob("*.db"))
        assert len(db_files) == 2
        assert "2024-04-05" in db_files[0].name
        assert "2024-04-06" in db_files[1].name

    async def test_backpressure_drops_oldest(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT", queue_maxsize=2)
        shutdown = asyncio.Event()

        storage.put_nowait(_snap(ts=1712300000000))
        storage.put_nowait(_snap(ts=1712300000100))
        storage.put_nowait(_snap(ts=1712300000200))  # drops oldest

        shutdown.set()
        await storage.run(shutdown)

        db_files = list(data_dir.glob("*.db"))
        conn = sqlite3.connect(db_files[0])
        rows = conn.execute(
            "SELECT timestamp_ms FROM snapshots ORDER BY timestamp_ms"
        ).fetchall()
        conn.close()

        timestamps = [r[0] for r in rows]
        assert 1712300000000 not in timestamps
        assert 1712300000100 in timestamps
        assert 1712300000200 in timestamps

    async def test_flush_on_shutdown(self, data_dir: Path) -> None:
        storage = StorageWriter(data_dir=data_dir, symbol="BTCUSDT")
        shutdown = asyncio.Event()

        for i in range(5):
            storage.put_nowait(_snap(ts=1712300000000 + i * 100))

        shutdown.set()
        await storage.run(shutdown)

        db_files = list(data_dir.glob("*.db"))
        conn = sqlite3.connect(db_files[0])
        count = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        conn.close()
        assert count == 5
