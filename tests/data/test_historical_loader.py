import json
import sqlite3
from pathlib import Path

import pytest

from bot.data.historical_loader import (
    _create_db,
    _parse_orderbook_lines,
    _parse_trade_csv,
)


def _ob_line(ts: int, seq: int, bids: list[list[str]], asks: list[list[str]]) -> str:
    return json.dumps({
        "topic": "orderbook.500.BTCUSDT",
        "type": "snapshot",
        "ts": ts,
        "data": {"s": "BTCUSDT", "b": bids, "a": asks, "u": 1, "seq": seq},
        "cts": ts - 1,
    })


class TestParseOrderbookLines:
    def test_downsample_10ms_to_100ms(self) -> None:
        lines = [
            _ob_line(ts=1000 + i * 10, seq=1000 + i, bids=[["100", "1"]], asks=[["101", "1"]])
            for i in range(20)
        ]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=100, lob_depth=50))
        assert len(out) == 2
        assert out[0][0] == 1000
        assert out[1][0] == 1100

    def test_depth_slice(self) -> None:
        big_bids = [[str(100 - i), "1"] for i in range(500)]
        big_asks = [[str(101 + i), "1"] for i in range(500)]
        lines = [_ob_line(ts=1000, seq=1, bids=big_bids, asks=big_asks)]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=10, lob_depth=50))
        assert len(out) == 1
        bids = json.loads(out[0][2])
        asks = json.loads(out[0][3])
        assert len(bids) == 50
        assert len(asks) == 50

    def test_is_reset_first_snapshot(self) -> None:
        lines = [_ob_line(ts=1000, seq=500, bids=[["100", "1"]], asks=[["101", "1"]])]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=10, lob_depth=50))
        assert out[0][1] is True

    def test_is_reset_on_seq_regression(self) -> None:
        lines = [
            _ob_line(ts=1000, seq=100, bids=[["100", "1"]], asks=[["101", "1"]]),
            _ob_line(ts=1010, seq=200, bids=[["100", "1"]], asks=[["101", "1"]]),
            _ob_line(ts=1020, seq=150, bids=[["100", "1"]], asks=[["101", "1"]]),
        ]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=10, lob_depth=50))
        # (first=True, seq 200>100=False, seq 150<200=True)
        assert [r[1] for r in out] == [True, False, True]

    def test_invalid_interval_raises(self) -> None:
        lines = [_ob_line(ts=1000, seq=1, bids=[["100", "1"]], asks=[["101", "1"]])]
        with pytest.raises(ValueError, match="multiple"):
            list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=15, lob_depth=50))

    def test_price_qty_preserved_as_str(self) -> None:
        lines = [_ob_line(ts=1000, seq=1,
                          bids=[["59116.70", "9.490"]],
                          asks=[["59116.80", "6.618"]])]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=10, lob_depth=50))
        bids = json.loads(out[0][2])
        asks = json.loads(out[0][3])
        assert bids[0] == ["59116.70", "9.490"]
        assert asks[0] == ["59116.80", "6.618"]


class TestParseTradeCSV:
    def test_no_header(self) -> None:
        lines = iter([
            "1725321600000,BTCUSDT,Buy,0.5,59000.0,PlusTick,abc-123",
            "1725321600100,BTCUSDT,Sell,1.0,58999.5,MinusTick,def-456",
        ])
        out = list(_parse_trade_csv(lines))
        assert len(out) == 2
        assert out[0] == (1725321600000, "1725321600000-59000.0-0.5", "Buy", "59000.0", "0.5")

    def test_with_header_and_match_id(self) -> None:
        lines = iter([
            "timestamp,symbol,side,size,price,tickDirection,trdMatchID",
            "1725321600000,BTCUSDT,Buy,0.5,59000.0,PlusTick,abc-123",
            "1725321600100,BTCUSDT,Sell,1.0,58999.5,MinusTick,def-456",
        ])
        out = list(_parse_trade_csv(lines))
        assert len(out) == 2
        assert out[0][1] == "abc-123"
        assert out[1][1] == "def-456"

    def test_with_header_no_match_id_column(self) -> None:
        lines = iter([
            "timestamp,symbol,side,size,price",
            "1725321600000,BTCUSDT,Buy,0.5,59000.0",
        ])
        out = list(_parse_trade_csv(lines))
        assert out[0][1] == "1725321600000-59000.0-0.5"

    def test_price_qty_as_str(self) -> None:
        lines = iter([
            "1725321600000,BTCUSDT,Buy,0.00012345,59000.50001",
        ])
        out = list(_parse_trade_csv(lines))
        _, _, _, price, qty = out[0]
        assert price == "59000.50001"
        assert qty == "0.00012345"
        assert isinstance(price, str)
        assert isinstance(qty, str)


class TestCreateDB:
    def test_creates_tables(self, tmp_path: Path) -> None:
        path = tmp_path / "test.db"
        conn = _create_db(path)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            names = {r[0] for r in rows}
            assert "snapshots" in names
            assert "trades" in names
        finally:
            conn.close()

    def test_trades_price_qty_are_text(self, tmp_path: Path) -> None:
        path = tmp_path / "test.db"
        conn = _create_db(path)
        try:
            conn.execute(
                "INSERT INTO trades (timestamp_ms, trade_id, side, price, qty) VALUES (?, ?, ?, ?, ?)",
                (1000, "abc", "Buy", "59000.0", "0.5"),
            )
            conn.commit()
            row = conn.execute("SELECT price, qty FROM trades").fetchone()
            assert row == ("59000.0", "0.5")
            assert isinstance(row[0], str)
            assert isinstance(row[1], str)
        finally:
            conn.close()
