import json
from pathlib import Path

import numpy as np
import pytest

from bot.backtest.report import (
    REPORT_FILE,
    RunMeta,
    build_report_data,
    downsample,
    equity_curves,
    load_run,
    render_html,
    save_run,
)
from bot.telemetry.types import PipelineCounters

T0_NS = 1_700_000_000_000_000_000
STEP_NS = 100_000_000

RECORD_DTYPE = np.dtype(
    [
        ("timestamp", "i8"), ("price", "f8"), ("position", "f8"), ("balance", "f8"),
        ("fee", "f8"), ("num_trades", "i8"), ("trading_volume", "f8"),
        ("trading_value", "f8"),
    ],
    align=True,
)


def _record(
    n: int = 200,
    price: np.ndarray | None = None,
    balance: np.ndarray | None = None,
    fee: np.ndarray | None = None,
    position: np.ndarray | None = None,
) -> np.ndarray:
    rec = np.zeros(n, dtype=RECORD_DTYPE)
    rec["timestamp"] = [T0_NS + i * STEP_NS for i in range(n)]
    rec["price"] = np.full(n, 30000.0) if price is None else price
    rec["balance"] = np.zeros(n) if balance is None else balance
    rec["fee"] = np.zeros(n) if fee is None else fee
    rec["position"] = np.zeros(n) if position is None else position
    return rec


def _meta(**over) -> RunMeta:
    base = dict(
        symbol="BTCUSDT", split="test", start="2026-03-01", end="2026-03-01",
        checkpoint="best.ckpt", queue_model="power_prob3", latency_model="intp",
        maker_fee_rate=-0.0001, taker_fee_rate=0.00055, hold_ms=500,
        take_profit_bps=15.0, stop_loss_bps=10.0, order_notional=100.0,
        initial_equity=10000.0, min_confidence=0.55, horizon_index=1,
        feed_latency_ns=10_000_000, latency_is_measured=False,
    )
    base.update(over)
    return RunMeta(**base)


def _stats(**over) -> PipelineCounters:
    stats = PipelineCounters(
        decisions=100, signals_actionable=40, signals_approved=20,
        orders_submitted=30, orders_filled=10, orders_cancelled=15,
        orders_expired=5, exits_by_reason={"HOLD_EXPIRED": 7, "STOP_LOSS": 3},
    )
    for key, value in over.items():
        setattr(stats, key, value)
    return stats


SUMMARY = {
    "start": "2026-03-01T00:00:00", "end": "2026-03-01T23:59:59",
    "SR": 1.42, "Sortino": 2.1, "Return": 0.013, "MaxDrawdown": 0.004,
    "DailyNumberOfTrades": 512.0, "ReturnOverMDD": 3.25,
}


class TestEquityCurves:
    def test_fees_are_subtracted_from_net_only(self) -> None:
        rec = _record(balance=np.full(10, 50.0), fee=np.full(10, 20.0), n=10)
        curves = equity_curves(rec, 10000.0)
        assert curves["gross_pct"][-1] == pytest.approx(0.5)
        assert curves["net_pct"][-1] == pytest.approx(0.3)

    def test_open_position_is_marked_to_market(self) -> None:
        rec = _record(n=10, position=np.full(10, 0.01), balance=np.full(10, -300.0))
        curves = equity_curves(rec, 10000.0)
        # -300 cash + 0.01 * 30000 = 0
        assert curves["gross_pct"][-1] == pytest.approx(0.0)

    def test_benchmark_is_price_change_from_the_start(self) -> None:
        price = np.linspace(30000.0, 30300.0, 10)
        curves = equity_curves(_record(n=10, price=price), 10000.0)
        assert curves["price_pct"][0] == pytest.approx(0.0)
        assert curves["price_pct"][-1] == pytest.approx(1.0)

    def test_drawdown_is_measured_from_the_running_peak(self) -> None:
        balance = np.array([0.0, 100.0, 200.0, 100.0, 150.0])
        curves = equity_curves(_record(n=5, balance=balance), 10000.0)
        # peak 10200, trough 10100 → 0.98%
        assert curves["drawdown_pct"].max() == pytest.approx(100 / 10200 * 100)
        assert curves["drawdown_pct"][0] == 0.0

    def test_drawdown_never_negative(self) -> None:
        balance = np.cumsum(np.full(50, 10.0))
        curves = equity_curves(_record(n=50, balance=balance), 10000.0)
        assert (curves["drawdown_pct"] >= 0).all()


class TestDownsample:
    def _curves(self, n: int) -> dict[str, np.ndarray]:
        return {
            "net_pct": np.zeros(n),
            "drawdown_pct": np.concatenate([np.zeros(n - 1), [99.0]]),
        }

    def test_short_series_passes_through(self) -> None:
        ts = np.array([T0_NS + i * STEP_NS for i in range(50)], dtype=np.int64)
        out = downsample(ts, self._curves(50), max_points=100)
        assert len(out["t"]) == 50

    def test_long_series_is_thinned(self) -> None:
        n = 5000
        ts = np.array([T0_NS + i * STEP_NS for i in range(n)], dtype=np.int64)
        out = downsample(ts, self._curves(n), max_points=200)
        assert len(out["t"]) == 200
        assert len(out["net_pct"]) == 200

    def test_worst_drawdown_survives_thinning(self) -> None:
        n = 5000
        curves = {"drawdown_pct": np.zeros(n)}
        curves["drawdown_pct"][1234] = 42.0  # a spike in the middle of a bucket
        ts = np.array([T0_NS + i * STEP_NS for i in range(n)], dtype=np.int64)
        out = downsample(ts, curves, max_points=100)
        assert max(out["drawdown_pct"]) == pytest.approx(42.0)

    def test_timestamps_are_milliseconds(self) -> None:
        ts = np.array([T0_NS], dtype=np.int64)
        out = downsample(ts, {"net_pct": np.zeros(1)}, max_points=10)
        assert out["t"][0] == T0_NS // 1_000_000


class TestBuildReportData:
    def test_headline_splits_fee_drag_out(self) -> None:
        rec = _record(balance=np.full(10, 100.0), fee=np.full(10, 40.0), n=10)
        data = build_report_data(rec, SUMMARY, _stats(), _meta())
        assert data.headline["gross_return"] == pytest.approx(1.0)
        assert data.headline["net_return"] == pytest.approx(0.6)
        assert data.headline["fee_drag"] == pytest.approx(0.4)

    def test_funnel_follows_the_order_intent_is_lost_in(self) -> None:
        data = build_report_data(_record(), SUMMARY, _stats(), _meta())
        labels = [row["label"] for row in data.funnel]
        assert labels[0] == "Signals actionable"
        assert labels.index("Approved by risk") < labels.index("Orders submitted")
        assert [r["value"] for r in data.funnel][:4] == [40, 20, 30, 10]

    def test_exits_sorted_by_frequency(self) -> None:
        data = build_report_data(_record(), SUMMARY, _stats(), _meta())
        assert [row["label"] for row in data.exits] == ["Hold expired", "Stop loss"]

    def test_assumed_latency_is_flagged(self) -> None:
        data = build_report_data(_record(), SUMMARY, _stats(), _meta())
        assert any("latency" in w.lower() for w in data.warnings)

    def test_measured_latency_is_not_flagged(self) -> None:
        data = build_report_data(
            _record(), SUMMARY, _stats(), _meta(latency_is_measured=True)
        )
        assert data.warnings == []

    def test_a_run_that_never_traded_says_so(self) -> None:
        data = build_report_data(
            _record(), SUMMARY, _stats(orders_submitted=0), _meta(latency_is_measured=True)
        )
        assert any("No orders" in w for w in data.warnings)

    def test_metrics_survive_missing_summary_keys(self) -> None:
        data = build_report_data(_record(), {}, _stats(), _meta())
        assert all(m["value"] for m in data.metrics)

    def test_non_finite_metrics_render_as_a_dash(self) -> None:
        data = build_report_data(
            _record(), {"SR": float("inf"), "Sortino": float("nan")}, _stats(), _meta()
        )
        values = {m["label"]: m["value"] for m in data.metrics}
        assert values["Sharpe"] == "—"
        assert values["Sortino"] == "—"


class TestRenderHtml:
    def test_standalone_is_a_complete_document(self) -> None:
        html = render_html(build_report_data(_record(), SUMMARY, _stats(), _meta()))
        assert html.startswith("<!doctype html>")
        assert "<html lang=\"en\">" in html
        assert 'charset="utf-8"' in html
        assert html.rstrip().endswith("</html>")

    def test_fragment_has_no_document_shell(self) -> None:
        html = render_html(
            build_report_data(_record(), SUMMARY, _stats(), _meta()), standalone=False
        )
        assert "<!doctype" not in html
        assert "<html" not in html
        assert "<style>" in html

    def test_data_is_embedded_as_valid_json(self) -> None:
        html = render_html(build_report_data(_record(), SUMMARY, _stats(), _meta()))
        payload = html.split("const D = ", 1)[1].split(";\n", 1)[0]
        assert json.loads(payload)["meta"]["symbol"] == "BTCUSDT"

    def test_placeholder_is_fully_substituted(self) -> None:
        html = render_html(build_report_data(_record(), SUMMARY, _stats(), _meta()))
        assert "__DATA__" not in html

    def test_both_themes_are_defined(self) -> None:
        html = render_html(build_report_data(_record(), SUMMARY, _stats(), _meta()))
        assert "prefers-color-scheme:dark" in html
        assert ':root[data-theme="dark"]' in html
        assert ':root:not([data-theme="light"])' in html

    def test_no_external_scripts(self) -> None:
        html = render_html(build_report_data(_record(), SUMMARY, _stats(), _meta()))
        assert "<script src=" not in html


class TestSaveAndLoad:
    def test_round_trips_without_replaying(self, tmp_path: Path) -> None:
        record, stats, meta = _record(), _stats(), _meta()
        save_run(tmp_path, record, SUMMARY, stats, meta)

        loaded_record, loaded_summary, loaded_stats, loaded_meta = load_run(tmp_path)
        assert np.array_equal(loaded_record["price"], record["price"])
        assert loaded_summary == SUMMARY
        assert loaded_stats == stats
        assert loaded_meta == meta

    def test_writes_the_three_run_files(self, tmp_path: Path) -> None:
        save_run(tmp_path, _record(), SUMMARY, _stats(), _meta())
        assert {p.name for p in tmp_path.iterdir()} == {
            "record.npz", "run.json", REPORT_FILE
        }

    def test_report_is_regenerated_from_the_saved_run(self, tmp_path: Path) -> None:
        save_run(tmp_path, _record(), SUMMARY, _stats(), _meta())
        first = (tmp_path / REPORT_FILE).read_text(encoding="utf-8")

        record, summary, stats, meta = load_run(tmp_path)
        again = render_html(build_report_data(record, summary, stats, meta))
        assert again == first


class TestMissingPrices:
    """A feed opens before its first snapshot, so the first steps have no price."""

    def _record_with_gap(self, leading: int = 5) -> np.ndarray:
        rec = _record(n=50)
        rec["price"][:leading] = np.nan
        return rec

    def test_leading_nan_prices_do_not_break_the_curves(self) -> None:
        curves = equity_curves(self._record_with_gap(), 10000.0)
        assert np.isfinite(curves["price_pct"]).all()
        assert np.isfinite(curves["net_pct"]).all()

    def test_benchmark_starts_at_zero_despite_the_gap(self) -> None:
        curves = equity_curves(self._record_with_gap(), 10000.0)
        assert curves["price_pct"][0] == pytest.approx(0.0)

    def test_report_serialises_with_a_price_gap(self) -> None:
        data = build_report_data(self._record_with_gap(), SUMMARY, _stats(), _meta())
        html = render_html(data)
        assert "__DATA__" not in html
        payload = html.split("const D = ", 1)[1].split(";", 1)[0]
        assert "NaN" not in payload

    def test_a_gap_in_the_middle_carries_the_last_price(self) -> None:
        rec = _record(n=20)
        rec["price"][10:13] = np.nan
        curves = equity_curves(rec, 10000.0)
        assert np.isfinite(curves["price_pct"]).all()

    def test_an_all_nan_price_series_still_renders(self) -> None:
        rec = _record(n=20)
        rec["price"][:] = np.nan
        curves = equity_curves(rec, 10000.0)
        assert np.isfinite(curves["net_pct"]).all()
