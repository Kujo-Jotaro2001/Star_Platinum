import numpy as np
import pytest

from bot.features.context_features import NUM_CONTEXT_FEATURES, ContextBuilder


def _ticker_row(
    ts: int,
    mark: str | None = None,
    index: str | None = None,
    oi: str | None = None,
    oi_val: str | None = None,
    funding: str | None = None,
    vol24h: str | None = None,
    pct24h: str | None = None,
) -> dict:
    return {
        "timestamp_ms": ts,
        "mark_price": mark,
        "index_price": index,
        "open_interest": oi,
        "open_interest_value": oi_val,
        "funding_rate": funding,
        "next_funding_time": None,
        "price_24h_pct": pct24h,
        "prev_price_1h": None,
        "volume_24h": vol24h,
        "turnover_24h": None,
    }


def _liq_row(ts: int, side: str, qty: str) -> dict:
    return {"timestamp_ms": ts, "side": side, "qty": qty, "price": "100.0"}


def _ls_row(ts: int, buy: str, sell: str) -> dict:
    return {"timestamp_ms": ts, "buy_ratio": buy, "sell_ratio": sell}


class TestContextBuilder:
    def test_output_shape(self) -> None:
        cb = ContextBuilder()
        result = cb.snapshot(1000)
        assert result.shape == (NUM_CONTEXT_FEATURES,)
        assert result.dtype == np.float32

    def test_forward_fill_ticker(self) -> None:
        cb = ContextBuilder()
        cb.update_ticker(_ticker_row(ts=100, mark="50000.0", index="49990.0"))
        # NULL update — should not overwrite
        cb.update_ticker(_ticker_row(ts=200, mark=None, index=None, oi="12345.0"))

        snap = cb.snapshot(300)
        assert snap[0] == pytest.approx(50000.0)  # mark_price (forward-filled)
        assert snap[1] == pytest.approx(49990.0)  # index_price (forward-filled)
        assert snap[3] == pytest.approx(12345.0)  # open_interest (updated)

    def test_basis_bps(self) -> None:
        cb = ContextBuilder()
        cb.update_ticker(_ticker_row(ts=100, mark="50100.0", index="50000.0"))
        snap = cb.snapshot(200)
        expected = (50100.0 - 50000.0) / 50000.0 * 10_000
        assert snap[2] == pytest.approx(expected, rel=1e-4)

    def test_funding_sign(self) -> None:
        cb = ContextBuilder()
        cb.update_ticker(_ticker_row(ts=100, funding="0.0001"))
        snap = cb.snapshot(200)
        assert snap[6] == pytest.approx(1.0)

        cb.update_ticker(_ticker_row(ts=300, funding="-0.0002"))
        snap = cb.snapshot(400)
        assert snap[6] == pytest.approx(-1.0)

    def test_oi_change_pct_1h(self) -> None:
        cb = ContextBuilder()
        # OI = 100 at t=0
        cb.update_ticker(_ticker_row(ts=0, oi="100.0"))
        # OI = 110 at t=30min (still within 1h window)
        cb.update_ticker(_ticker_row(ts=1_800_000, oi="110.0"))

        snap = cb.snapshot(1_800_000)
        # oldest in window is t=0, oi=100. Current = 110. Change = 10%
        assert snap[4] == pytest.approx(0.1)

    def test_oi_change_evicts_old(self) -> None:
        cb = ContextBuilder()
        cb.update_ticker(_ticker_row(ts=0, oi="100.0"))
        cb.update_ticker(_ticker_row(ts=3_600_001, oi="120.0"))

        snap = cb.snapshot(3_600_001)
        # t=0 evicted (older than 1h), only t=3600001 remains → change = 0%
        assert snap[4] == pytest.approx(0.0)

    def test_liquidation_1min_window(self) -> None:
        cb = ContextBuilder()
        cb.update_liquidation(_liq_row(ts=1000, side="Buy", qty="0.5"))
        cb.update_liquidation(_liq_row(ts=2000, side="Sell", qty="0.3"))

        snap = cb.snapshot(50_000)
        assert snap[9] == pytest.approx(0.5)   # liq_buy_vol
        assert snap[10] == pytest.approx(0.3)  # liq_sell_vol
        assert snap[12] == pytest.approx(2.0)  # liq_count

    def test_liquidation_eviction(self) -> None:
        cb = ContextBuilder()
        cb.update_liquidation(_liq_row(ts=0, side="Buy", qty="1.0"))
        cb.update_liquidation(_liq_row(ts=60_001, side="Buy", qty="2.0"))

        snap = cb.snapshot(60_001)
        # t=0 evicted (> 60s ago), only t=60001 remains
        assert snap[9] == pytest.approx(2.0)
        assert snap[12] == pytest.approx(1.0)

    def test_long_short_ratio(self) -> None:
        cb = ContextBuilder()
        cb.update_long_short_ratio(_ls_row(ts=100, buy="0.55", sell="0.45"))

        snap = cb.snapshot(200)
        assert snap[13] == pytest.approx(0.55)
        assert snap[14] == pytest.approx(0.45)
        assert snap[15] == pytest.approx(0.55 / 0.45, rel=1e-4)

    def test_state_carries_over(self) -> None:
        """Simulates cross-day carryover."""
        cb = ContextBuilder()
        cb.update_ticker(_ticker_row(ts=100, mark="50000.0", funding="0.0001"))
        cb.update_long_short_ratio(_ls_row(ts=100, buy="0.6", sell="0.4"))

        # "Next day" — no new updates, but state should persist
        snap = cb.snapshot(86_500_000)
        assert snap[0] == pytest.approx(50000.0)
        assert snap[5] == pytest.approx(0.0001)
        assert snap[13] == pytest.approx(0.6)

    def test_liq_imbalance(self) -> None:
        cb = ContextBuilder()
        cb.update_liquidation(_liq_row(ts=1000, side="Buy", qty="7.0"))
        cb.update_liquidation(_liq_row(ts=2000, side="Sell", qty="3.0"))

        snap = cb.snapshot(50_000)
        expected = (7.0 - 3.0) / (7.0 + 3.0 + 1e-8)
        assert snap[11] == pytest.approx(expected, rel=1e-5)
