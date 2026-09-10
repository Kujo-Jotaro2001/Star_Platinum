import json
from pathlib import Path

import pytest

from bot.data.historical_loader import (
    _create_db,
    _parse_orderbook_lines,
    _parse_trade_csv,
)


def _ob_line(
    ts: int,
    seq: int,
    bids: list[list[str]],
    asks: list[list[str]],
    msg_type: str = "delta",
) -> str:
    return json.dumps({
        "topic": "orderbook.500.BTCUSDT",
        "type": msg_type,
        "ts": ts,
        "data": {"s": "BTCUSDT", "b": bids, "a": asks, "u": 1, "seq": seq},
        "cts": ts - 1,
    })


def _snapshot(ts: int, seq: int, bids, asks) -> str:
    return _ob_line(ts, seq, bids, asks, msg_type="snapshot")


class TestParseOrderbookLines:
    """The archive is one snapshot followed by deltas, at a 100 ms cadence."""

    def test_book_is_carried_across_deltas(self) -> None:
        lines = [
            _snapshot(1000, 1, [["100", "1"], ["99", "2"]], [["101", "1"], ["102", "2"]]),
            _ob_line(1100, 2, [["100", "5"]], []),
        ]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=100, lob_depth=50))
        assert json.loads(out[1][2]) == [["100", "5"], ["99", "2"]]
        assert json.loads(out[1][3]) == [["101", "1"], ["102", "2"]]

    def test_zero_quantity_removes_a_level(self) -> None:
        lines = [
            _snapshot(1000, 1, [["100", "1"], ["99", "2"]], [["101", "1"]]),
            _ob_line(1100, 2, [["100", "0"]], []),
        ]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=100, lob_depth=50))
        assert json.loads(out[1][2]) == [["99", "2"]]

    def test_levels_are_ordered_best_first(self) -> None:
        lines = [_snapshot(1000, 1,
                           [["98", "1"], ["100", "1"], ["99", "1"]],
                           [["103", "1"], ["101", "1"], ["102", "1"]])]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=100, lob_depth=50))
        assert [b[0] for b in json.loads(out[0][2])] == ["100", "99", "98"]
        assert [a[0] for a in json.loads(out[0][3])] == ["101", "102", "103"]

    def test_downsample_100ms_to_1s(self) -> None:
        lines = [_snapshot(1000, 1, [["100", "1"]], [["101", "1"]])]
        lines += [
            _ob_line(1000 + i * 100, 1 + i, [["100", str(i)]], [["101", "1"]])
            for i in range(1, 20)
        ]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=1000, lob_depth=50))
        assert [r[0] for r in out] == [1000, 2000]

    def test_native_cadence_keeps_every_message(self) -> None:
        lines = [_snapshot(1000, 1, [["100", "1"]], [["101", "1"]])]
        lines += [_ob_line(1000 + i * 100, 1 + i, [["100", "1"]], []) for i in range(1, 5)]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=100, lob_depth=50))
        assert len(out) == 5

    def test_depth_slice(self) -> None:
        big_bids = [[str(100 - i), "1"] for i in range(500)]
        big_asks = [[str(101 + i), "1"] for i in range(500)]
        out = list(_parse_orderbook_lines(
            iter([_snapshot(1000, 1, big_bids, big_asks)]),
            snapshot_interval_ms=100, lob_depth=50,
        ))
        assert len(json.loads(out[0][2])) == 50
        assert len(json.loads(out[0][3])) == 50

    def test_is_reset_first_snapshot(self) -> None:
        out = list(_parse_orderbook_lines(
            iter([_snapshot(1000, 500, [["100", "1"]], [["101", "1"]])]),
            snapshot_interval_ms=100, lob_depth=50,
        ))
        assert out[0][1] is True

    def test_a_later_snapshot_resets_and_replaces_the_book(self) -> None:
        lines = [
            _snapshot(1000, 100, [["100", "1"], ["99", "1"]], [["101", "1"]]),
            _snapshot(1100, 200, [["50", "1"]], [["51", "1"]]),
        ]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=100, lob_depth=50))
        assert out[1][1] is True
        assert json.loads(out[1][2]) == [["50", "1"]]

    def test_is_reset_on_seq_regression(self) -> None:
        lines = [
            _snapshot(1000, 100, [["100", "1"]], [["101", "1"]]),
            _ob_line(1100, 200, [["100", "1"]], [["101", "1"]]),
            _ob_line(1200, 150, [["100", "1"]], [["101", "1"]]),
        ]
        out = list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=100, lob_depth=50))
        assert [r[1] for r in out] == [True, False, True]

    def test_one_sided_book_is_not_emitted(self) -> None:
        out = list(_parse_orderbook_lines(
            iter([_snapshot(1000, 1, [["100", "1"]], [])]),
            snapshot_interval_ms=100, lob_depth=50,
        ))
        assert out == []

    def test_invalid_interval_raises(self) -> None:
        lines = [_snapshot(1000, 1, [["100", "1"]], [["101", "1"]])]
        with pytest.raises(ValueError, match="multiple"):
            list(_parse_orderbook_lines(iter(lines), snapshot_interval_ms=150, lob_depth=50))

    def test_price_qty_preserved_as_str(self) -> None:
        out = list(_parse_orderbook_lines(
            iter([_snapshot(1000, 1, [["59116.70", "9.490"]], [["59116.80", "6.618"]])]),
            snapshot_interval_ms=100, lob_depth=50,
        ))
        assert json.loads(out[0][2])[0] == ["59116.70", "9.490"]
        assert json.loads(out[0][3])[0] == ["59116.80", "6.618"]


class TestParseTradeCSV:
    """Bybit writes the trade timestamp as float seconds, not integer ms."""

    HEADER = (
        "timestamp,symbol,side,size,price,tickDirection,trdMatchID,"
        "grossValue,homeNotional,foreignNotional,RPI"
    )

    def test_float_seconds_become_milliseconds(self) -> None:
        lines = iter([
            self.HEADER,
            "1755648000.1385,SOLUSDT,Sell,27.7,176.140,ZeroMinusTick,d4bd-1,4.8e+11,27.7,4879.0,0",
        ])
        out = list(_parse_trade_csv(lines))
        assert out[0][0] == 1755648000138

    def test_no_header(self) -> None:
        lines = iter([
            "1725321600.0,BTCUSDT,Buy,0.5,59000.0",
            "1725321600.1,BTCUSDT,Sell,1.0,58999.5",
        ])
        out = list(_parse_trade_csv(lines))
        assert len(out) == 2
        assert out[0] == (1725321600000, "1725321600000-59000.0-0.5", "Buy", "59000.0", "0.5")
        assert out[1][0] == 1725321600100

    def test_with_header_and_match_id(self) -> None:
        lines = iter([
            self.HEADER,
            "1725321600.0,BTCUSDT,Buy,0.5,59000.0,PlusTick,abc-123,0,0,0,0",
            "1725321600.1,BTCUSDT,Sell,1.0,58999.5,MinusTick,def-456,0,0,0,0",
        ])
        out = list(_parse_trade_csv(lines))
        assert [r[1] for r in out] == ["abc-123", "def-456"]

    def test_with_header_no_match_id_column(self) -> None:
        lines = iter([
            "timestamp,symbol,side,size,price",
            "1725321600.0,BTCUSDT,Buy,0.5,59000.0",
        ])
        out = list(_parse_trade_csv(lines))
        assert out[0][1] == "1725321600000-59000.0-0.5"

    def test_side_size_and_price_come_from_their_columns(self) -> None:
        lines = iter([
            self.HEADER,
            "1755648000.1385,SOLUSDT,Sell,27.7,176.140,ZeroMinusTick,x,0,0,0,0",
        ])
        _, _, side, price, qty = list(_parse_trade_csv(lines))[0]
        assert (side, price, qty) == ("Sell", "176.140", "27.7")

    def test_price_qty_as_str(self) -> None:
        lines = iter([
            "1725321600.0,BTCUSDT,Buy,0.00012345,59000.50001",
        ])
        _, _, _, price, qty = list(_parse_trade_csv(lines))[0]
        assert price == "59000.50001"
        assert qty == "0.00012345"
        assert isinstance(price, str) and isinstance(qty, str)


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
