import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from bot.data.storage import StorageWriter
from bot.telemetry.types import APPROVED, DecisionRecord, OrderEvent

NOW_MS = 1_800_000_000_000


def _decision(**over) -> DecisionRecord:
    base = dict(
        timestamp_ms=NOW_MS,
        status=APPROVED,
        action="BUY",
        horizon_index=1,
        p_down=0.05,
        p_flat=0.05,
        p_up=0.90,
        predicted_class=2,
        confidence=0.90,
        best_bid=Decimal("30000.5"),
        best_ask=Decimal("30001.0"),
        book_age_ms=12,
        is_reset=False,
        inference_us=3400,
        signal_reason=None,
        risk_reason="approved",
        flow_z_max=2.5,
        flow_z_abs_mean=0.8,
        ctx_ticker_age_ms=250,
        ctx_liq_age_ms=4000,
        ctx_ls_age_ms=900_000,
        class_changed=True,
    )
    base.update(over)
    return DecisionRecord(**base)


def _order(**over) -> OrderEvent:
    base = dict(
        timestamp_ms=NOW_MS,
        order_link_id="sp-e-b-123",
        event="filled",
        side="Buy",
        order_type="PostOnly",
        reduce_only=False,
        qty=Decimal("0.010"),
        limit_price=Decimal("30000.5"),
        fill_price=Decimal("30000.5"),
        decision_price=Decimal("30000.5"),
        latency_ms=84,
        exit_reason=None,
    )
    base.update(over)
    return OrderEvent(**base)


async def _write(tmp_path: Path, items: list) -> Path:
    storage = StorageWriter(data_dir=tmp_path, symbol="BTCUSDT", queue_maxsize=100)
    shutdown = asyncio.Event()
    task = asyncio.create_task(storage.run(shutdown))
    for item in items:
        storage.put_nowait(item)
    await asyncio.sleep(0.2)
    shutdown.set()
    await task
    return tmp_path / "BTCUSDT_2027-01-15.db"


class TestDecisionsTable:
    async def test_a_decision_survives_the_round_trip(self, tmp_path: Path) -> None:
        import sqlite3

        record = _decision()
        await _write(tmp_path, [record])

        db = next(tmp_path.glob("*.db"))
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM decisions").fetchone())
        conn.close()

        assert row["status"] == APPROVED
        assert row["predicted_class"] == 2
        assert row["confidence"] == pytest.approx(0.90)
        assert row["best_bid"] == "30000.5"
        assert row["ctx_ls_age_ms"] == 900_000
        assert row["class_changed"] == 1
        assert row["is_reset"] == 0

    async def test_many_decisions_are_all_kept(self, tmp_path: Path) -> None:
        import sqlite3

        records = [_decision(timestamp_ms=NOW_MS + i) for i in range(25)]
        await _write(tmp_path, records)

        db = next(tmp_path.glob("*.db"))
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        conn.close()
        assert count == 25


class TestOrderEventsTable:
    async def test_an_order_event_survives_the_round_trip(self, tmp_path: Path) -> None:
        import sqlite3

        await _write(tmp_path, [_order()])

        db = next(tmp_path.glob("*.db"))
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM order_events").fetchone())
        conn.close()

        assert row["order_link_id"] == "sp-e-b-123"
        assert row["event"] == "filled"
        assert row["qty"] == "0.010"
        assert row["latency_ms"] == 84
        assert row["reduce_only"] == 0

    async def test_optional_prices_stay_null(self, tmp_path: Path) -> None:
        import sqlite3

        await _write(
            tmp_path,
            [_order(event="submitted", fill_price=None, decision_price=None)],
        )
        db = next(tmp_path.glob("*.db"))
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM order_events").fetchone())
        conn.close()

        assert row["fill_price"] is None
        assert row["decision_price"] is None

    async def test_exit_reason_is_kept(self, tmp_path: Path) -> None:
        import sqlite3

        await _write(tmp_path, [_order(reduce_only=True, exit_reason="STOP_LOSS")])
        db = next(tmp_path.glob("*.db"))
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT exit_reason, reduce_only FROM order_events").fetchone()
        conn.close()
        assert row == ("STOP_LOSS", 1)


class TestCoexistence:
    async def test_telemetry_shares_the_daily_file_with_market_data(
        self, tmp_path: Path
    ) -> None:
        import sqlite3

        from bot.data.models import Trade

        await _write(
            tmp_path,
            [
                Trade(NOW_MS, "t1", "Buy", Decimal("30000"), Decimal("0.5")),
                _decision(),
                _order(),
            ],
        )
        db = next(tmp_path.glob("*.db"))
        conn = sqlite3.connect(db)
        tables = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        counts = {
            name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in ("trades", "decisions", "order_events")
        }
        conn.close()

        assert {"snapshots", "trades", "decisions", "order_events"} <= tables
        assert counts == {"trades": 1, "decisions": 1, "order_events": 1}
