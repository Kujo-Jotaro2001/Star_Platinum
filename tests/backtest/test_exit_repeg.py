"""The passive exit must follow the book instead of stranding behind it.

A long exits by selling at the ask. When the market falls the ask falls with it,
and an exit posted once sits above the market where it can never fill — which is
precisely the case the position needs out of. Re-pegging keeps the exit reachable
while still paying the smaller maker fee; the crossing fallback is what bounds it.

The fee model in `tests/backtest/conftest.py` is a synthetic maker rebate,
chosen so a fill's sign pins down whether it was maker or taker; it does not
claim anything about the real VIP0 tier this project actually trades at.
"""

from decimal import Decimal
from pathlib import Path

from bot.backtest.strategy import run_strategy
from bot.execution.types import ExitPolicy, OrderType
from bot.risk.types import RiskPolicy
from bot.signals.types import SignalPolicy
from tests.backtest.conftest import STEP_NS, TICK, make_backtest, write_feed
from tests.backtest.test_strategy import SPEC, SYMBOL, UP, _run, _schedule

STEPS = 400


def falling_after_entry(steps: int, bid: float = 100.0, ask: float = 100.1):
    """Flat long enough to fill a long, then a steady one-tick-per-step decline."""
    quotes = [(bid, ask)] * 40
    for k in range(steps - 40):
        drop = k * TICK
        quotes.append((round(bid - drop, 8), round(ask - drop, 8)))
    return quotes


def _final_position(feed: Path, *, exit_repeg: bool, exit_fallback_ms: int) -> float:
    hbt = make_backtest(feed)
    try:
        run_strategy(
            hbt=hbt,
            schedule=_schedule(UP, STEPS),
            symbol=SYMBOL,
            signal_policy=SignalPolicy(0, 0.55, True),
            risk_policy=RiskPolicy(
                max_position_notional=Decimal("100"),
                max_leverage=Decimal("1000"),
                max_spread_bps=Decimal("50"),
                stale_data_ms=10_000,
                cooldown_after_trade_ms=0,
                min_confidence=0.55,
                allow_short=True,
            ),
            exit_policy=ExitPolicy(
                hold_ms=500,
                take_profit_bps=Decimal("15"),
                stop_loss_bps=Decimal("10"),
                exit_fallback_ms=exit_fallback_ms,
            ),
            spec=SPEC,
            exit_order_type=OrderType.POST_ONLY,
            order_notional=Decimal("100"),
            initial_equity=Decimal("10000"),
            elapse_ns=STEP_NS,
            order_timeout_ns=1_000_000_000,
            exit_repeg=exit_repeg,
        )
        return hbt.position(0)
    finally:
        hbt.close()


def test_passive_exit_strands_without_repegging(feed_dir: Path) -> None:
    """Baseline: the exit is posted once, the book leaves, nothing re-posts."""
    feed = write_feed(feed_dir / "fall.npz", falling_after_entry(STEPS), trade_qty=3.0)
    stats, _, _, _ = _run(
        feed, _schedule(UP, STEPS), hold_ms=500,
        exit_order_type=OrderType.POST_ONLY, exit_fallback_ms=0, exit_repeg=False,
    )
    assert stats.orders_filled > 0
    assert stats.exit_repegs == 0


def test_repegging_follows_the_book_down(feed_dir: Path) -> None:
    feed = write_feed(feed_dir / "fall.npz", falling_after_entry(STEPS), trade_qty=3.0)
    stats, _, _, _ = _run(
        feed, _schedule(UP, STEPS), hold_ms=500,
        exit_order_type=OrderType.POST_ONLY, exit_fallback_ms=0, exit_repeg=True,
    )
    assert stats.orders_filled > 0
    assert stats.exit_repegs > 0


def test_repegging_never_crosses_and_so_pays_no_taker_fee(feed_dir: Path) -> None:
    """With no fallback every leg is passive, so the fee total stays negative
    under the synthetic maker-rebate fee model conftest uses for this test."""
    feed = write_feed(feed_dir / "fall.npz", falling_after_entry(STEPS), trade_qty=3.0)
    _, _, fee, trades = _run(
        feed, _schedule(UP, STEPS), hold_ms=500,
        exit_order_type=OrderType.POST_ONLY, exit_fallback_ms=0, exit_repeg=True,
    )
    assert trades > 0
    assert fee < 0, "every fill was a maker fill, under the synthetic rebate fee model"


def test_repegging_alone_chases_a_one_way_market_forever(feed_dir: Path) -> None:
    """Why the crossing fallback cannot simply be switched off.

    Re-pegging quotes at the new top of book every time the old one is left
    behind. In a market that only falls, the ask is gone before anyone lifts it,
    so the chase never completes and the position is still open at the end. Being
    a maker is cheaper per fill but it cannot guarantee a fill at all — the
    fallback is what turns "cheaper" back into "bounded".
    """
    feed = write_feed(feed_dir / "fall.npz", falling_after_entry(STEPS), trade_qty=3.0)
    assert _final_position(feed, exit_repeg=True, exit_fallback_ms=0) != 0


def test_fallback_bounds_the_chase(feed_dir: Path) -> None:
    feed = write_feed(feed_dir / "fall.npz", falling_after_entry(STEPS), trade_qty=3.0)
    assert _final_position(feed, exit_repeg=True, exit_fallback_ms=200) == 0


def test_fallback_still_crosses_when_it_is_configured(feed_dir: Path) -> None:
    """Re-pegging is the default path, not a way to disable the escalation."""
    feed = write_feed(feed_dir / "fall.npz", falling_after_entry(STEPS), trade_qty=3.0)
    _, _, fee, trades = _run(
        feed, _schedule(UP, STEPS), hold_ms=500,
        exit_order_type=OrderType.POST_ONLY, exit_fallback_ms=200, exit_repeg=True,
    )
    assert trades > 0
    assert fee > 0, "a crossed exit pays the taker fee"
