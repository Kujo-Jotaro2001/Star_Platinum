"""Accumulates what the decision loop sees, and rolls it up for a human.

Every metric here answers "which stage is failing" — the funnel says where
intent is lost, the health gauges say whether the input was real, and the skew
gauges say whether the model is being fed the distribution it trained on.
"""

from dataclasses import replace
from decimal import Decimal

import numpy as np

from bot.telemetry.types import (
    NO_BOOK,
    NOT_READY,
    PREDICTED_STATUSES,
    DecisionRecord,
    OrderEvent,
    PipelineCounters,
)

BPS = Decimal("10000")


class Window:
    """A bounded ring of recent samples, summarised by exact quantiles.

    Bounded because the loop runs at 10 Hz for days; exact because an
    approximate sketch would be harder to trust than the few thousand floats
    this costs.
    """

    def __init__(self, capacity: int = 4096) -> None:
        self._values = np.zeros(capacity, dtype=np.float64)
        self._capacity = capacity
        self._next = 0
        self._size = 0

    def add(self, value: float) -> None:
        self._values[self._next] = value
        self._next = (self._next + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def summary(self, digits: int = 2) -> dict[str, float]:
        if self._size == 0:
            return {}
        data = self._values[: self._size]
        p50, p99 = np.percentile(data, [50, 99])
        return {
            "p50": round(float(p50), digits),
            "p99": round(float(p99), digits),
            "max": round(float(data.max()), digits),
        }

    def __len__(self) -> int:
        return self._size


class FlowSkew:
    """Drift of the live flow features against the frozen training statistics.

    `OnlinePreprocessor` already z-scores against the mean and std saved from the
    train split, so a live stream matching training gives z ~ N(0, 1). A mean far
    from zero is a level shift; a standard deviation far from one is a variance
    blow-up. Either means the model is being asked about a distribution it never
    saw, and no amount of execution tuning will fix that.
    """

    def __init__(self, n_features: int) -> None:
        self._n = n_features
        self._count = 0
        self._sum = np.zeros(n_features, dtype=np.float64)
        self._sum_sq = np.zeros(n_features, dtype=np.float64)

    def add(self, z: np.ndarray) -> None:
        self._count += 1
        self._sum += z
        self._sum_sq += z * z

    def reset(self) -> None:
        self._count = 0
        self._sum[:] = 0.0
        self._sum_sq[:] = 0.0

    def summary(self) -> dict[str, float | int]:
        if self._count < 2:
            return {}
        mean = self._sum / self._count
        var = np.maximum(self._sum_sq / self._count - mean * mean, 0.0)
        std = np.sqrt(var)

        worst_mean = int(np.argmax(np.abs(mean)))
        worst_std = int(np.argmax(np.abs(std - 1.0)))
        return {
            "z_mean_worst": round(float(mean[worst_mean]), 3),
            "z_mean_worst_feature": worst_mean,
            "z_std_worst": round(float(std[worst_std]), 3),
            "z_std_worst_feature": worst_std,
        }


class TelemetryCollector:
    """Session and window counters plus the gauges the rollup reports."""

    def __init__(self, n_flow_features: int = 9, window_capacity: int = 4096) -> None:
        self.session = PipelineCounters()
        self.window = PipelineCounters()

        self._skew = FlowSkew(n_flow_features)
        self._loop_interval = Window(window_capacity)
        self._book_age = Window(window_capacity)
        self._inference_us = Window(window_capacity)
        self._confidence = Window(window_capacity)
        self._time_to_fill = Window(window_capacity)
        self._submit_latency = Window(window_capacity)
        self._slippage_bps = Window(window_capacity)

        self._ctx_ticker_age = 0
        self._ctx_liq_age = 0
        self._ctx_ls_age = 0
        self._resets = 0
        self._class_changes = 0
        self._maker_fills = 0
        self._taker_fills = 0
        self._last_decision_ms: int | None = None
        self._last_class: int | None = None
        self._window_start_ms: int | None = None

    def observe_decision(self, record: DecisionRecord, flow_z: np.ndarray | None) -> DecisionRecord:
        """Fold one decision in. Returns it with the derived fields filled."""
        for counters in (self.session, self.window):
            counters.decisions += 1

        if self._window_start_ms is None:
            self._window_start_ms = record.timestamp_ms
        if self._last_decision_ms is not None:
            self._loop_interval.add(record.timestamp_ms - self._last_decision_ms)
        self._last_decision_ms = record.timestamp_ms

        if record.is_reset:
            self._resets += 1

        changed = (
            self._last_class is not None and record.predicted_class != self._last_class
        )
        self._last_class = record.predicted_class

        if flow_z is not None:
            self._skew.add(flow_z)
            record = replace(
                record,
                flow_z_max=round(float(np.abs(flow_z).max()), 4),
                flow_z_abs_mean=round(float(np.abs(flow_z).mean()), 4),
                class_changed=changed,
            )
        else:
            record = replace(record, class_changed=changed)

        if changed:
            self._class_changes += 1
        self._book_age.add(record.book_age_ms)
        if record.inference_us:
            self._inference_us.add(record.inference_us)
        if record.status in PREDICTED_STATUSES:
            self._confidence.add(record.confidence)

        self._ctx_ticker_age = max(self._ctx_ticker_age, record.ctx_ticker_age_ms)
        self._ctx_liq_age = max(self._ctx_liq_age, record.ctx_liq_age_ms)
        self._ctx_ls_age = max(self._ctx_ls_age, record.ctx_ls_age_ms)

        if record.status == NOT_READY:
            for counters in (self.session, self.window):
                counters.not_ready += 1
        elif record.status == NO_BOOK:
            for counters in (self.session, self.window):
                counters.no_book += 1

        if record.signal_reason is not None:
            for counters in (self.session, self.window):
                counters.blocked(record.signal_reason)
        if record.action in ("BUY", "SELL"):
            for counters in (self.session, self.window):
                counters.signals_actionable += 1
            if record.risk_reason == "approved":
                for counters in (self.session, self.window):
                    counters.signals_approved += 1
            elif record.risk_reason is not None:
                for counters in (self.session, self.window):
                    counters.rejected(record.risk_reason)

        return record

    def observe_order(self, event: OrderEvent) -> None:
        for counters in (self.session, self.window):
            if event.event == "submitted":
                counters.orders_submitted += 1
            elif event.event == "filled":
                counters.orders_filled += 1
            elif event.event == "cancelled":
                counters.orders_cancelled += 1
            elif event.event == "expired":
                counters.orders_expired += 1

        if event.latency_ms is not None:
            if event.event == "filled":
                self._time_to_fill.add(event.latency_ms)
            else:
                self._submit_latency.add(event.latency_ms)

        if event.event == "filled":
            if event.order_type == "PostOnly":
                self._maker_fills += 1
            else:
                self._taker_fills += 1
            slip = slippage_bps(event)
            if slip is not None:
                self._slippage_bps.add(slip)

        if event.exit_reason is not None:
            for counters in (self.session, self.window):
                counters.exited(event.exit_reason)

    def rollup(self, now_ms: int) -> dict:
        """Window summary for structlog, then start a fresh window."""
        window_ms = max(1, now_ms - (self._window_start_ms or now_ms))
        w = self.window
        fills = self._maker_fills + self._taker_fills

        report = {
            "window_s": round(window_ms / 1000, 1),
            "decisions": w.decisions,
            "decisions_per_s": round(w.decisions / (window_ms / 1000), 2),
            "not_ready": w.not_ready,
            "no_book": w.no_book,
            "actionable": w.signals_actionable,
            "approved": w.signals_approved,
            "approval_rate": round(w.approval_rate, 3),
            "submitted": w.orders_submitted,
            "filled": w.orders_filled,
            "cancelled": w.orders_cancelled,
            "expired": w.orders_expired,
            "fill_rate": round(w.fill_rate, 3),
            "blocked": dict(w.signal_blocked),
            "risk_rejected": dict(w.risk_rejected),
            "exits": dict(w.exits_by_reason),
            "resets": self._resets,
            "ctx_age_ms": {
                "ticker": self._ctx_ticker_age,
                "liquidation": self._ctx_liq_age,
                "ls_ratio": self._ctx_ls_age,
            },
            "class_churn": round(self._class_changes / w.decisions, 3) if w.decisions else 0.0,
            "maker_share": round(self._maker_fills / fills, 3) if fills else 0.0,
        }
        for name, window in (
            ("loop_ms", self._loop_interval),
            ("book_age_ms", self._book_age),
            ("inference_us", self._inference_us),
            ("confidence", self._confidence),
            ("time_to_fill_ms", self._time_to_fill),
            ("submit_latency_ms", self._submit_latency),
            ("slippage_bps", self._slippage_bps),
        ):
            summary = window.summary(4 if name in ("confidence", "slippage_bps") else 1)
            if summary:
                report[name] = summary

        skew = self._skew.summary()
        if skew:
            report["flow_skew"] = skew

        self._reset_window(now_ms)
        return report

    def _reset_window(self, now_ms: int) -> None:
        self.window = PipelineCounters()
        self._skew.reset()
        self._resets = 0
        self._class_changes = 0
        self._maker_fills = 0
        self._taker_fills = 0
        self._ctx_ticker_age = 0
        self._ctx_liq_age = 0
        self._ctx_ls_age = 0
        self._window_start_ms = now_ms


def slippage_bps(event: OrderEvent) -> float | None:
    """How far the fill landed from the price the decision was taken at.

    Signed so positive always means worse than intended, whichever side it is:
    a buy that filled above the decision price and a sell that filled below both
    read as a cost.
    """
    if event.fill_price is None or event.decision_price is None:
        return None
    if event.decision_price <= 0:
        return None

    direction = Decimal(1) if event.side == "Buy" else Decimal(-1)
    diff = (event.fill_price - event.decision_price) * direction
    return float(diff / event.decision_price * BPS)
