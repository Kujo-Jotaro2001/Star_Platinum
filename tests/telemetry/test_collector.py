from decimal import Decimal

import numpy as np
import pytest

from bot.telemetry.collector import FlowSkew, TelemetryCollector, Window, slippage_bps
from bot.telemetry.types import (
    APPROVED,
    BLOCKED,
    NO_BOOK,
    NOT_READY,
    REJECTED,
    DecisionRecord,
    OrderEvent,
    PipelineCounters,
)

NOW_MS = 1_800_000_000_000


def _decision(
    status: str = APPROVED,
    action: str = "BUY",
    signal_reason: str | None = None,
    risk_reason: str | None = "approved",
    timestamp_ms: int = NOW_MS,
    predicted_class: int = 2,
    confidence: float = 0.9,
    **over,
) -> DecisionRecord:
    return DecisionRecord(
        timestamp_ms=timestamp_ms,
        status=status,
        action=action,
        horizon_index=1,
        p_down=0.05, p_flat=0.05, p_up=0.90,
        predicted_class=predicted_class,
        confidence=confidence,
        signal_reason=signal_reason,
        risk_reason=risk_reason,
        **over,
    )


def _order(event: str = "submitted", **over) -> OrderEvent:
    base = dict(
        timestamp_ms=NOW_MS,
        order_link_id="sp-e-b-1",
        event=event,
        side="Buy",
        order_type="PostOnly",
        reduce_only=False,
        qty=Decimal("0.01"),
    )
    base.update(over)
    return OrderEvent(**base)


class TestWindow:
    def test_empty_window_has_no_summary(self) -> None:
        assert Window().summary() == {}

    def test_quantiles_over_the_samples(self) -> None:
        w = Window(capacity=100)
        for v in range(1, 101):
            w.add(float(v))
        s = w.summary()
        assert s["p50"] == pytest.approx(50.5)
        assert s["max"] == 100.0

    def test_oldest_samples_are_evicted(self) -> None:
        w = Window(capacity=10)
        for v in range(100):
            w.add(float(v))
        assert len(w) == 10
        assert w.summary()["max"] == 99.0


class TestFlowSkew:
    def test_needs_two_samples(self) -> None:
        skew = FlowSkew(3)
        skew.add(np.zeros(3))
        assert skew.summary() == {}

    def test_matching_distribution_reads_as_no_drift(self) -> None:
        rng = np.random.default_rng(0)
        skew = FlowSkew(4)
        for _ in range(4000):
            skew.add(rng.standard_normal(4))
        s = skew.summary()
        assert abs(s["z_mean_worst"]) < 0.1
        assert abs(s["z_std_worst"] - 1.0) < 0.1

    def test_level_shift_is_reported_with_the_feature(self) -> None:
        skew = FlowSkew(3)
        for _ in range(500):
            z = np.zeros(3)
            z[1] = 4.0  # feature 1 has drifted off its training mean
            skew.add(z)
        s = skew.summary()
        assert s["z_mean_worst"] == pytest.approx(4.0)
        assert s["z_mean_worst_feature"] == 1

    def test_variance_blowup_is_reported(self) -> None:
        rng = np.random.default_rng(1)
        skew = FlowSkew(2)
        for _ in range(4000):
            skew.add(np.array([rng.normal(0, 1), rng.normal(0, 6)]))
        s = skew.summary()
        assert s["z_std_worst_feature"] == 1
        assert s["z_std_worst"] > 4

    def test_reset_clears_the_window(self) -> None:
        skew = FlowSkew(2)
        for _ in range(10):
            skew.add(np.ones(2))
        skew.reset()
        assert skew.summary() == {}


class TestCounters:
    def test_approved_signal_counts_once_at_each_stage(self) -> None:
        c = TelemetryCollector()
        c.observe_decision(_decision(), None)
        assert c.session.decisions == 1
        assert c.session.signals_actionable == 1
        assert c.session.signals_approved == 1
        assert c.session.risk_rejected == {}

    def test_risk_rejection_is_attributed_to_its_reason(self) -> None:
        c = TelemetryCollector()
        c.observe_decision(_decision(status=REJECTED, risk_reason="spread_too_wide"), None)
        assert c.session.signals_actionable == 1
        assert c.session.signals_approved == 0
        assert c.session.risk_rejected == {"spread_too_wide": 1}

    def test_signal_block_is_separate_from_a_risk_rejection(self) -> None:
        c = TelemetryCollector()
        c.observe_decision(
            _decision(
                status=BLOCKED, action="NO_TRADE",
                signal_reason="confidence_below_threshold", risk_reason=None,
            ),
            None,
        )
        assert c.session.signal_blocked == {"confidence_below_threshold": 1}
        assert c.session.signals_actionable == 0
        assert c.session.risk_rejected == {}

    def test_not_ready_and_no_book_are_counted_apart(self) -> None:
        c = TelemetryCollector()
        c.observe_decision(_decision(status=NOT_READY, action="NONE", risk_reason=None), None)
        c.observe_decision(_decision(status=NO_BOOK, action="NONE", risk_reason=None), None)
        assert c.session.not_ready == 1
        assert c.session.no_book == 1
        assert c.session.decisions == 2

    def test_window_resets_but_session_accumulates(self) -> None:
        c = TelemetryCollector()
        c.observe_decision(_decision(), None)
        c.rollup(NOW_MS + 1000)
        c.observe_decision(_decision(), None)
        assert c.session.decisions == 2
        assert c.window.decisions == 1


class TestOrders:
    def test_order_lifecycle_counts(self) -> None:
        c = TelemetryCollector()
        c.observe_order(_order("submitted"))
        c.observe_order(_order("filled", latency_ms=120))
        c.observe_order(_order("cancelled", latency_ms=1000))
        c.observe_order(_order("expired", latency_ms=15))

        assert c.session.orders_submitted == 1
        assert c.session.orders_filled == 1
        assert c.session.orders_cancelled == 1
        assert c.session.orders_expired == 1

    def test_exit_reason_recorded(self) -> None:
        c = TelemetryCollector()
        c.observe_order(_order("filled", reduce_only=True, exit_reason="STOP_LOSS"))
        assert c.session.exits_by_reason == {"STOP_LOSS": 1}

    def test_maker_share_distinguishes_order_types(self) -> None:
        c = TelemetryCollector()
        c.observe_order(_order("filled", order_type="PostOnly"))
        c.observe_order(_order("filled", order_type="PostOnly"))
        c.observe_order(_order("filled", order_type="Market"))
        assert c.rollup(NOW_MS + 1000)["maker_share"] == pytest.approx(2 / 3, abs=5e-4)


class TestSlippage:
    def test_buy_filled_above_the_decision_price_is_a_cost(self) -> None:
        event = _order(
            "filled", side="Buy",
            decision_price=Decimal("10000"), fill_price=Decimal("10001"),
        )
        assert slippage_bps(event) == pytest.approx(1.0)

    def test_sell_filled_below_the_decision_price_is_also_a_cost(self) -> None:
        event = _order(
            "filled", side="Sell",
            decision_price=Decimal("10000"), fill_price=Decimal("9999"),
        )
        assert slippage_bps(event) == pytest.approx(1.0)

    def test_price_improvement_is_negative(self) -> None:
        event = _order(
            "filled", side="Buy",
            decision_price=Decimal("10000"), fill_price=Decimal("9999"),
        )
        assert slippage_bps(event) == pytest.approx(-1.0)

    def test_missing_prices_give_nothing(self) -> None:
        assert slippage_bps(_order("filled")) is None


class TestRollup:
    def test_reports_the_funnel_and_the_rates(self) -> None:
        c = TelemetryCollector()
        for _ in range(3):
            c.observe_decision(_decision(), None)
        c.observe_decision(_decision(status=REJECTED, risk_reason="cooldown_active"), None)
        c.observe_order(_order("submitted"))
        c.observe_order(_order("filled", latency_ms=50))

        report = c.rollup(NOW_MS + 60_000)
        assert report["decisions"] == 4
        assert report["actionable"] == 4
        assert report["approved"] == 3
        assert report["approval_rate"] == pytest.approx(0.75)
        assert report["risk_rejected"] == {"cooldown_active": 1}
        assert report["fill_rate"] == 1.0
        assert report["window_s"] == pytest.approx(60.0)

    def test_context_ages_report_the_worst_seen(self) -> None:
        c = TelemetryCollector()
        c.observe_decision(_decision(ctx_ticker_age_ms=200), None)
        c.observe_decision(_decision(ctx_ticker_age_ms=90_000), None)
        assert c.rollup(NOW_MS + 1000)["ctx_age_ms"]["ticker"] == 90_000

    def test_class_churn_measures_flipping(self) -> None:
        c = TelemetryCollector()
        for cls in (2, 0, 2, 0):
            c.observe_decision(_decision(predicted_class=cls), None)
        # three transitions out of four decisions
        assert c.rollup(NOW_MS + 1000)["class_churn"] == pytest.approx(0.75)

    def test_stable_predictions_have_no_churn(self) -> None:
        c = TelemetryCollector()
        for _ in range(5):
            c.observe_decision(_decision(predicted_class=2), None)
        assert c.rollup(NOW_MS + 1000)["class_churn"] == 0.0

    def test_loop_interval_comes_from_decision_timestamps(self) -> None:
        c = TelemetryCollector()
        for k in range(10):
            c.observe_decision(_decision(timestamp_ms=NOW_MS + k * 100), None)
        assert c.rollup(NOW_MS + 1000)["loop_ms"]["p50"] == pytest.approx(100.0)

    def test_flow_skew_appears_when_features_are_seen(self) -> None:
        c = TelemetryCollector(n_flow_features=3)
        for _ in range(50):
            c.observe_decision(_decision(), np.array([0.0, 3.0, 0.0]))
        report = c.rollup(NOW_MS + 1000)
        assert report["flow_skew"]["z_mean_worst_feature"] == 1

    def test_derived_fields_are_written_back_onto_the_record(self) -> None:
        c = TelemetryCollector(n_flow_features=3)
        record = c.observe_decision(_decision(), np.array([-1.0, 4.0, 0.5]))
        assert record.flow_z_max == pytest.approx(4.0)
        assert record.flow_z_abs_mean == pytest.approx((1.0 + 4.0 + 0.5) / 3, abs=5e-5)


class TestPipelineCounters:
    def test_rates_are_zero_when_nothing_happened(self) -> None:
        c = PipelineCounters()
        assert c.fill_rate == 0.0
        assert c.approval_rate == 0.0

    def test_reason_helpers_accumulate(self) -> None:
        c = PipelineCounters()
        c.rejected("spread_too_wide")
        c.rejected("spread_too_wide")
        c.blocked("flat_selected")
        c.exited("HOLD_EXPIRED")
        assert c.risk_rejected == {"spread_too_wide": 2}
        assert c.signal_blocked == {"flat_selected": 1}
        assert c.exits_by_reason == {"HOLD_EXPIRED": 1}
