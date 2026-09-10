"""The trading loop hftbacktest drives.

Deliberately plain Python rather than a numba `@njit` kernel: that lets the
backtest call the very same `signals`, `risk` and `execution.lifecycle` functions
`LiveTrader` calls, so the two cannot drift apart. hftbacktest itself does the
heavy work in Rust, and decisions are only taken once per bucket.
"""

import math
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from hftbacktest.order import GTC, GTX, LIMIT, MARKET, NEW, PARTIALLY_FILLED

from bot.execution.lifecycle import (
    build_entry_intent,
    decide_exit,
    open_position,
    passive_exit_price,
    should_cross_exit,
    should_repeg_exit,
)
from bot.execution.types import (
    ExitPolicy,
    InstrumentSpec,
    OpenPosition,
    OrderIntent,
    OrderSide,
    OrderType,
)
from bot.risk.gate import evaluate_risk
from bot.risk.types import MarketState, PositionState, RiskPolicy
from bot.signals.policy import generate_signal
from bot.signals.types import ModelPrediction, SignalAction, SignalPolicy
from bot.telemetry.types import PipelineCounters

NS_PER_MS = 1_000_000
RESTING_STATUSES = frozenset({NEW, PARTIALLY_FILLED})


@dataclass(frozen=True, slots=True)
class SignalSchedule:
    """Model output aligned onto feed time.

    `timestamps_ns[j]` is the snapshot row `probabilities[j]` was produced from,
    so the loop can pick the most recent already-published prediction without
    ever reading one from the future.
    """

    timestamps_ns: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        if len(self.timestamps_ns) != len(self.probabilities):
            raise ValueError(
                f"schedule has {len(self.timestamps_ns)} timestamps but "
                f"{len(self.probabilities)} probability rows"
            )


def run_strategy(
    hbt,
    schedule: SignalSchedule,
    symbol: str,
    signal_policy: SignalPolicy,
    risk_policy: RiskPolicy,
    exit_policy: ExitPolicy,
    spec: InstrumentSpec,
    exit_order_type: OrderType,
    order_notional: Decimal,
    initial_equity: Decimal,
    elapse_ns: int,
    order_timeout_ns: int,
    exit_repeg: bool = False,
    recorder=None,
) -> PipelineCounters:
    """Replay the schedule against the feed, one position at a time."""
    stats = PipelineCounters()
    cursor = -1
    next_order_id = 0

    position: OpenPosition | None = None
    resting: _Resting | None = None
    exiting: _Exiting | None = None
    last_trade_ts_ms: int | None = None

    while hbt.elapse(elapse_ns) == 0:
        stats.decisions += 1
        if recorder is not None:
            recorder.record(hbt)

        now_ns = hbt.current_timestamp
        best_bid, best_ask = _top_of_book(hbt.depth(0))
        if best_bid is None:
            stats.no_book += 1
            hbt.clear_inactive_orders(0)
            continue

        cursor = _advance(schedule.timestamps_ns, cursor, now_ns)
        if cursor < 0:
            hbt.clear_inactive_orders(0)
            continue

        # The feed covers whole days but the split usually does not. Past the last
        # prediction, stop opening positions and leave once the book is flat —
        # otherwise the strategy would keep trading on a stale signal.
        past_end = now_ns > schedule.timestamps_ns[-1]
        if past_end and position is None and resting is None:
            break

        now_ms = now_ns // NS_PER_MS
        market = MarketState(
            symbol=symbol,
            timestamp_ms=now_ms,
            best_bid=best_bid,
            best_ask=best_ask,
            account_equity=_equity(hbt, initial_equity, best_bid, best_ask),
        )

        if resting is not None:
            order = hbt.orders(0).get(resting.order_id)
            filled = _fill_of(order)
            if filled is not None:
                # A fill can land after the cancel was sent: cancelling is not
                # instantaneous, so the order is watched until the exchange
                # confirms it is gone. Dropping it at cancel time leaves the
                # strategy believing it is flat while the position is real.
                position = open_position(
                    resting.intent, filled, now_ms, exit_policy
                )
                last_trade_ts_ms = now_ms
                stats.orders_filled += 1
                resting = None
            elif order is None or order.status not in RESTING_STATUSES:
                if resting.cancelling:
                    stats.orders_cancelled += 1
                else:
                    stats.orders_expired += 1
                resting = None
            elif not resting.cancelling and now_ns >= resting.deadline_ns:
                if order.cancellable:
                    hbt.cancel(0, resting.order_id, False)
                resting.cancelling = True

        elif position is not None:
            if exiting is not None:
                held = hbt.position(0)
                if held == 0:
                    position = None
                    exiting = None
                    last_trade_ts_ms = now_ms
                elif not exiting.escalated:
                    order = hbt.orders(0).get(exiting.order_id)
                    live = order is not None and order.status in RESTING_STATUSES
                    if live and not exiting.cancelling:
                        # Ask for the cancel, then wait for it. Acting while the
                        # passive exit is still live lets both fill and flips the
                        # position through zero into a side the policy never chose.
                        if should_cross_exit(
                            position, market, now_ms, exiting.submitted_ms, exit_policy
                        ):
                            exiting.crossing = True
                        elif exit_repeg and should_repeg_exit(
                            exiting.side, exiting.price, market
                        ):
                            stats.exit_repegs += 1
                        else:
                            hbt.clear_inactive_orders(0)
                            continue
                        if order.cancellable:
                            hbt.cancel(0, exiting.order_id, False)
                        exiting.cancelling = True
                    elif not live:
                        # The passive attempt is gone and there is still a position.
                        # Both paths act on what is actually held, not on what the
                        # strategy believes it holds, so any drift self-corrects.
                        side = OrderSide.SELL if held > 0 else OrderSide.BUY
                        qty = Decimal(str(abs(held)))
                        next_order_id += 1
                        if exiting.crossing or not exit_repeg:
                            _submit(
                                hbt,
                                order_id=next_order_id,
                                side=side,
                                price=best_bid if held > 0 else best_ask,
                                qty=qty,
                                order_type=OrderType.MARKET,
                            )
                            exiting.escalated = True
                        else:
                            price = passive_exit_price(side, market)
                            _submit(
                                hbt,
                                order_id=next_order_id,
                                side=side,
                                price=price,
                                qty=qty,
                                order_type=OrderType.POST_ONLY,
                            )
                            exiting.price = price
                        exiting.order_id = next_order_id
                        exiting.cancelling = False
                        stats.orders_submitted += 1
            else:
                decision = decide_exit(position, market, now_ms)
                if decision.should_exit:
                    side = position.side.opposite
                    # The passive exit quotes on the side it must join to earn the
                    # smaller maker fee; crossing is the fallback, not the first attempt.
                    price = (
                        passive_exit_price(side, market)
                        if exit_order_type is OrderType.POST_ONLY
                        else (best_bid if position.side is OrderSide.BUY else best_ask)
                    )
                    next_order_id += 1
                    _submit(
                        hbt,
                        order_id=next_order_id,
                        side=side,
                        price=price,
                        qty=position.qty,
                        order_type=exit_order_type,
                    )
                    stats.orders_submitted += 1
                    exiting = _Exiting(
                        order_id=next_order_id,
                        side=side,
                        price=price,
                        submitted_ms=now_ms,
                        escalated=exit_order_type is OrderType.MARKET,
                    )
                    stats.exited(decision.reason.value)

        elif not past_end:
            signal = generate_signal(
                symbol=symbol,
                timestamp_ms=now_ms,
                prediction=ModelPrediction(
                    class_probs=_row_to_probs(schedule.probabilities[cursor])
                ),
                policy=signal_policy,
            )
            if signal.action not in (SignalAction.BUY, SignalAction.SELL):
                stats.blocked(signal.reason or "unknown")
            else:
                stats.signals_actionable += 1
                risk = evaluate_risk(
                    signal=signal,
                    position=PositionState(
                        symbol=symbol,
                        quantity=Decimal(0),
                        last_trade_timestamp_ms=last_trade_ts_ms,
                    ),
                    market=market,
                    policy=risk_policy,
                    proposed_order_notional=order_notional,
                    now_ms=now_ms,
                )
                if not risk.approved:
                    stats.rejected(risk.reason)
                else:
                    stats.signals_approved += 1
                    intent = build_entry_intent(
                        signal, market, order_notional, OrderType.POST_ONLY, spec
                    )
                    if intent is not None:
                        next_order_id += 1
                        _submit(
                            hbt,
                            order_id=next_order_id,
                            side=intent.side,
                            price=intent.limit_price,
                            qty=intent.qty,
                            order_type=OrderType.POST_ONLY,
                        )
                        stats.orders_submitted += 1
                        resting = _Resting(
                            order_id=next_order_id,
                            intent=intent,
                            deadline_ns=now_ns + order_timeout_ns,
                        )

        hbt.clear_inactive_orders(0)

    return stats


@dataclass(slots=True)
class _Resting:
    order_id: int
    intent: OrderIntent
    deadline_ns: int
    cancelling: bool = False


@dataclass(slots=True)
class _Exiting:
    """The in-flight exit for the open position.

    `submitted_ms` is when the exit was first decided, not when the current order
    was posted, so re-pegging does not keep resetting the escalation window.
    """

    order_id: int
    side: OrderSide
    price: Decimal
    submitted_ms: int
    cancelling: bool = False
    crossing: bool = False
    escalated: bool = False


def _submit(
    hbt,
    order_id: int,
    side: OrderSide,
    price: Decimal,
    qty: Decimal,
    order_type: OrderType,
) -> None:
    """GTX is hftbacktest's post-only: a GTX order that would cross expires unfilled."""
    time_in_force = GTX if order_type is OrderType.POST_ONLY else GTC
    hft_type = LIMIT if order_type is OrderType.POST_ONLY else MARKET
    submit = (
        hbt.submit_buy_order if side is OrderSide.BUY else hbt.submit_sell_order
    )
    submit(0, order_id, float(price), float(qty), time_in_force, hft_type, False)


def _fill_of(order) -> Decimal | None:
    """The average execution price once an order has traded, else None."""
    if order is None or order.exec_qty <= 0:
        return None
    return Decimal(str(order.exec_price))


def _top_of_book(depth) -> tuple[Decimal, Decimal] | tuple[None, None]:
    """Top of book as exact Decimals, or (None, None) when a side is missing.

    Prices are rebuilt from the tick counts: `depth.best_ask` is a float and
    comes back as e.g. 100.10000000000001, which would poison Decimal maths.
    hftbacktest reports NaN for an empty side.
    """
    if math.isnan(depth.best_bid) or math.isnan(depth.best_ask):
        return None, None

    tick = Decimal(str(depth.tick_size))
    best_bid = Decimal(depth.best_bid_tick) * tick
    best_ask = Decimal(depth.best_ask_tick) * tick
    if best_bid >= best_ask:
        return None, None
    return best_bid, best_ask


def _equity(
    hbt,
    initial_equity: Decimal,
    best_bid: Decimal,
    best_ask: Decimal,
) -> Decimal:
    """Account equity the risk gate sizes against.

    hftbacktest has no notion of a starting balance — `state_values.balance` is
    the cash flow generated by trading and opens at zero — so the configured
    initial equity has to be added back, or every leverage check would divide by
    an empty account.
    """
    state = hbt.state_values(0)
    mid = (best_bid + best_ask) / 2
    return (
        initial_equity
        + Decimal(str(state.balance))
        + Decimal(str(state.position)) * mid
        - Decimal(str(state.fee))
    )


def _advance(timestamps_ns: np.ndarray, cursor: int, now_ns: int) -> int:
    """Index of the newest prediction already published at `now_ns`, or -1."""
    nxt = cursor + 1
    while nxt < len(timestamps_ns) and timestamps_ns[nxt] <= now_ns:
        cursor = nxt
        nxt += 1
    return cursor


def _row_to_probs(row: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(v) for v in horizon) for horizon in row)  # type: ignore[return-value]
