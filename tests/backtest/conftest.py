from pathlib import Path

import numpy as np
import pytest
from hftbacktest import (
    BUY_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
    BacktestAsset,
    HashMapMarketDepthBacktest,
    event_dtype,
)

T0_NS = 1_700_000_000_000_000_000
STEP_NS = 100_000_000  # 100 ms, the feature bucket
FEED_LATENCY_NS = 1_000_000
TICK = 0.1
LOT = 0.001
LEVELS = 5
LEVEL_QTY = 10.0


def _px(value: float) -> float:
    return round(value, 8)


def write_feed(
    path: Path,
    quotes: list[tuple[float, float]],
    trade_qty: float = 0.0,
    t0_ns: int = T0_NS,
    step_ns: int = STEP_NS,
) -> Path:
    """Write a feed whose top of book follows `quotes`, one entry per step.

    Ladders are explicitly zeroed before the new ones are laid down — the depth
    is a map of price to quantity, so a level left unset would linger and the
    best price would never actually move.
    """
    rows: list[tuple] = []

    def add(ev: int, ts: int, px: float, qty: float) -> None:
        rows.append((ev, ts, ts + FEED_LATENCY_NS, px, qty, 0, 0, 0.0))

    def ladder(ev: int, ts: int, best: float, direction: int, qty: float) -> None:
        for i in range(LEVELS):
            add(ev, ts, _px(best + direction * i * TICK), qty)

    bid, ask = quotes[0]
    ladder(EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_SNAPSHOT_EVENT, t0_ns, bid, -1, LEVEL_QTY)
    ladder(EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_SNAPSHOT_EVENT, t0_ns, ask, +1, LEVEL_QTY)

    prev_bid, prev_ask = bid, ask
    for step, (bid, ask) in enumerate(quotes[1:], start=1):
        ts = t0_ns + step * step_ns

        if bid != prev_bid:
            ladder(EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT, ts, prev_bid, -1, 0.0)
            ladder(EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT, ts, bid, -1, LEVEL_QTY)
        if ask != prev_ask:
            ladder(EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT, ts, prev_ask, +1, 0.0)
            ladder(EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT, ts, ask, +1, LEVEL_QTY)

        # Keep time moving even when nothing changes: a feed with no events after
        # the snapshot ends immediately. The heartbeat re-asserts the deepest bid,
        # far from where any test order rests, so queue positions are untouched.
        add(
            EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT,
            ts,
            _px(bid - (LEVELS - 1) * TICK),
            LEVEL_QTY,
        )

        if trade_qty > 0:
            add(EXCH_EVENT | LOCAL_EVENT | SELL_EVENT | TRADE_EVENT, ts, bid, trade_qty)
            add(EXCH_EVENT | LOCAL_EVENT | BUY_EVENT | TRADE_EVENT, ts, ask, trade_qty)

        prev_bid, prev_ask = bid, ask

    np.savez_compressed(path, data=np.array(rows, dtype=event_dtype))
    return path


def make_backtest(feed: Path) -> HashMapMarketDepthBacktest:
    """A deterministic setup: risk-adverse queue, constant latency.

    The fee model here (-1 bps maker, 5.5 bps taker) is a synthetic market-maker
    rate chosen so a test can assert "this fill was maker" by checking the fee
    sign — it isolates execution-path tests from the real VIP0 fee tier used in
    conf/backtest/default.yaml, which has no rebate at all.
    """
    asset = (
        BacktestAsset()
        .data([str(feed)])
        .linear_asset(1.0)
        .constant_order_latency(10_000_000, 10_000_000)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(-0.0001, 0.00055)
        .tick_size(TICK)
        .lot_size(LOT)
    )
    return HashMapMarketDepthBacktest([asset])


def flat_quotes(steps: int, bid: float = 100.0, ask: float = 100.1) -> list[tuple[float, float]]:
    return [(bid, ask)] * steps


@pytest.fixture
def feed_dir(tmp_path: Path) -> Path:
    return tmp_path
