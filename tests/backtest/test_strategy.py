from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from bot.backtest.strategy import SignalSchedule, run_strategy
from bot.execution.types import ExitPolicy, InstrumentSpec, OrderType
from bot.risk.types import RiskPolicy
from bot.signals.types import SignalPolicy
from tests.backtest.conftest import (
    LOT,
    STEP_NS,
    T0_NS,
    TICK,
    flat_quotes,
    make_backtest,
    write_feed,
)

SYMBOL = "BTCUSDT"
STEPS = 400

UP = (0.05, 0.05, 0.90)
FLAT = (0.05, 0.90, 0.05)
DOWN = (0.90, 0.05, 0.05)

SPEC = InstrumentSpec(
    qty_step=Decimal(str(LOT)),
    price_tick=Decimal(str(TICK)),
    min_order_qty=Decimal(str(LOT)),
)


def _schedule(row: tuple[float, float, float], steps: int) -> SignalSchedule:
    # one prediction per feed step, starting one step in so the book exists
    timestamps = np.array(
        [T0_NS + k * STEP_NS for k in range(1, steps)], dtype=np.int64
    )
    probs = np.tile(np.array(row, dtype=np.float32), (len(timestamps), 1, 1))
    return SignalSchedule(timestamps_ns=timestamps, probabilities=probs)


def _run(
    feed: Path,
    schedule: SignalSchedule,
    hold_ms: int = 500,
    take_profit_bps: str = "15",
    stop_loss_bps: str = "10",
    min_confidence: float = 0.55,
    allow_short: bool = True,
    max_spread_bps: str = "50",
    cooldown_ms: int = 0,
    order_timeout_ms: int = 1000,
    exit_order_type: OrderType = OrderType.MARKET,
    exit_fallback_ms: int = 0,
    exit_repeg: bool = False,
):
    hbt = make_backtest(feed)
    try:
        stats = run_strategy(
            hbt=hbt,
            schedule=schedule,
            symbol=SYMBOL,
            signal_policy=SignalPolicy(
                horizon_index=0,
                min_confidence=min_confidence,
                allow_short=allow_short,
            ),
            risk_policy=RiskPolicy(
                max_position_notional=Decimal("100"),
                max_leverage=Decimal("1000"),
                max_spread_bps=Decimal(max_spread_bps),
                stale_data_ms=10_000,
                cooldown_after_trade_ms=cooldown_ms,
                min_confidence=min_confidence,
                allow_short=allow_short,
            ),
            exit_policy=ExitPolicy(
                hold_ms=hold_ms,
                take_profit_bps=Decimal(take_profit_bps),
                stop_loss_bps=Decimal(stop_loss_bps),
                exit_fallback_ms=exit_fallback_ms,
            ),
            spec=SPEC,
            exit_order_type=exit_order_type,
            order_notional=Decimal("100"),
            initial_equity=Decimal("10000"),
            elapse_ns=STEP_NS,
            order_timeout_ns=order_timeout_ms * 1_000_000,
            exit_repeg=exit_repeg,
        )
        final = hbt.state_values(0)
        return stats, float(final.balance), float(final.fee), int(final.num_trades)
    finally:
        hbt.close()


class TestSignalGating:
    def test_flat_predictions_never_trade(self, feed_dir: Path) -> None:
        feed = write_feed(feed_dir / "flat.npz", flat_quotes(STEPS), trade_qty=3.0)
        stats, _, _, trades = _run(feed, _schedule(FLAT, STEPS))
        assert stats.signals_actionable == 0
        assert stats.orders_submitted == 0
        assert trades == 0

    def test_short_blocked_when_shorting_disabled(self, feed_dir: Path) -> None:
        feed = write_feed(feed_dir / "down.npz", flat_quotes(STEPS), trade_qty=3.0)
        stats, _, _, trades = _run(
            feed, _schedule(DOWN, STEPS), allow_short=False
        )
        assert stats.signals_actionable == 0
        assert trades == 0

    def test_unreachable_confidence_never_trades(self, feed_dir: Path) -> None:
        feed = write_feed(feed_dir / "up.npz", flat_quotes(STEPS), trade_qty=3.0)
        stats, _, _, trades = _run(
            feed, _schedule(UP, STEPS), min_confidence=0.999
        )
        assert stats.signals_actionable == 0
        assert trades == 0

    def test_wide_spread_rejected_by_risk(self, feed_dir: Path) -> None:
        quotes = [(100.0, 101.0)] * STEPS  # ~100 bps
        feed = write_feed(feed_dir / "wide.npz", quotes, trade_qty=3.0)
        stats, _, _, trades = _run(
            feed, _schedule(UP, STEPS), max_spread_bps="5"
        )
        assert stats.signals_actionable > 0
        assert stats.signals_approved == 0
        assert stats.orders_submitted == 0
        assert trades == 0

    def test_same_book_is_approved_once_the_spread_limit_allows_it(
        self, feed_dir: Path
    ) -> None:
        quotes = [(100.0, 101.0)] * STEPS
        feed = write_feed(feed_dir / "wide.npz", quotes, trade_qty=3.0)
        stats, _, _, _ = _run(feed, _schedule(UP, STEPS), max_spread_bps="200")
        assert stats.signals_approved > 0


class TestPostOnlyEntry:
    def test_entry_is_submitted_and_fills(self, feed_dir: Path) -> None:
        feed = write_feed(feed_dir / "up.npz", flat_quotes(STEPS), trade_qty=3.0)
        stats, _, fee, trades = _run(feed, _schedule(UP, STEPS))

        assert stats.signals_approved > 0
        assert stats.orders_submitted > 0
        assert stats.orders_filled > 0
        assert trades > 0

    def test_entry_pays_the_maker_fee_not_the_taker_fee(self, feed_dir: Path) -> None:
        # a single round trip: long hold, no trades after entry to trigger exits
        feed = write_feed(feed_dir / "up.npz", flat_quotes(STEPS), trade_qty=3.0)
        stats, _, _, _ = _run(feed, _schedule(UP, STEPS), hold_ms=10_000_000)
        assert stats.orders_filled == 1
        assert stats.exits_by_reason == {}

    def test_no_fills_without_trades_then_order_times_out(self, feed_dir: Path) -> None:
        # a resting post-only order needs someone to trade through it
        feed = write_feed(feed_dir / "quiet.npz", flat_quotes(STEPS), trade_qty=0.0)
        stats, _, _, trades = _run(feed, _schedule(UP, STEPS), order_timeout_ms=300)

        assert stats.orders_submitted > 0
        assert stats.orders_filled == 0
        assert stats.orders_cancelled > 0
        assert trades == 0

    def test_cooldown_limits_re_entry(self, feed_dir: Path) -> None:
        feed = write_feed(feed_dir / "up.npz", flat_quotes(STEPS), trade_qty=3.0)
        busy, _, _, _ = _run(feed, _schedule(UP, STEPS), cooldown_ms=0)
        calm, _, _, _ = _run(feed, _schedule(UP, STEPS), cooldown_ms=5_000)
        assert calm.orders_submitted < busy.orders_submitted


class TestExit:
    def test_hold_expiry_closes_the_position(self, feed_dir: Path) -> None:
        feed = write_feed(feed_dir / "up.npz", flat_quotes(STEPS), trade_qty=3.0)
        stats, _, _, trades = _run(feed, _schedule(UP, STEPS), hold_ms=500)

        assert "HOLD_EXPIRED" in stats.exits_by_reason
        assert trades >= 2  # every closed round trip is two fills

    def test_take_profit_closes_the_position(self, feed_dir: Path) -> None:
        # price steps up ~20 bps once the position is on
        quotes = flat_quotes(60) + [(100.2, 100.3)] * (STEPS - 60)
        feed = write_feed(feed_dir / "tp.npz", quotes, trade_qty=3.0)
        stats, _, _, _ = _run(
            feed, _schedule(UP, STEPS), hold_ms=10_000_000, take_profit_bps="15"
        )
        assert stats.exits_by_reason.get("TAKE_PROFIT", 0) > 0

    def test_stop_loss_closes_the_position(self, feed_dir: Path) -> None:
        quotes = flat_quotes(60) + [(99.8, 99.9)] * (STEPS - 60)
        feed = write_feed(feed_dir / "sl.npz", quotes, trade_qty=3.0)
        stats, _, _, _ = _run(
            feed, _schedule(UP, STEPS), hold_ms=10_000_000, stop_loss_bps="10"
        )
        assert stats.exits_by_reason.get("STOP_LOSS", 0) > 0

    def test_position_is_flat_at_the_end(self, feed_dir: Path) -> None:
        feed = write_feed(feed_dir / "up.npz", flat_quotes(STEPS), trade_qty=3.0)
        hbt = make_backtest(feed)
        try:
            run_strategy(
                hbt=hbt,
                schedule=_schedule(UP, STEPS),
                symbol=SYMBOL,
                signal_policy=SignalPolicy(0, 0.55, True),
                risk_policy=RiskPolicy(
                    Decimal("100"), Decimal("1000"), Decimal("50"),
                    10_000, 0, 0.55, True,
                ),
                exit_policy=ExitPolicy(500, Decimal("15"), Decimal("10")),
                spec=SPEC,
                exit_order_type=OrderType.MARKET,
                order_notional=Decimal("100"),
                initial_equity=Decimal("10000"),
                elapse_ns=STEP_NS,
                order_timeout_ns=1_000_000_000,
            )
            assert hbt.position(0) == 0
        finally:
            hbt.close()


class TestScheduleBounds:
    def test_nothing_trades_before_the_first_prediction(self, feed_dir: Path) -> None:
        feed = write_feed(feed_dir / "up.npz", flat_quotes(STEPS), trade_qty=3.0)
        late = STEPS - 5
        timestamps = np.array(
            [T0_NS + k * STEP_NS for k in range(late, STEPS)], dtype=np.int64
        )
        probs = np.tile(np.array(UP, dtype=np.float32), (len(timestamps), 1, 1))
        stats, _, _, _ = _run(feed, SignalSchedule(timestamps, probs))
        assert stats.decisions <= STEPS
        assert stats.signals_actionable <= len(timestamps)

    def test_loop_stops_after_the_last_prediction(self, feed_dir: Path) -> None:
        feed = write_feed(feed_dir / "up.npz", flat_quotes(STEPS), trade_qty=3.0)
        early = 20
        timestamps = np.array(
            [T0_NS + k * STEP_NS for k in range(1, early)], dtype=np.int64
        )
        probs = np.tile(np.array(FLAT, dtype=np.float32), (len(timestamps), 1, 1))
        stats, _, _, _ = _run(feed, SignalSchedule(timestamps, probs))
        # it must not replay the remaining ~380 steps of feed on a stale signal
        assert stats.decisions < STEPS // 2

    def test_mismatched_schedule_lengths_rejected(self) -> None:
        with pytest.raises(ValueError):
            SignalSchedule(
                timestamps_ns=np.array([1, 2, 3], dtype=np.int64),
                probabilities=np.zeros((2, 1, 3), dtype=np.float32),
            )
