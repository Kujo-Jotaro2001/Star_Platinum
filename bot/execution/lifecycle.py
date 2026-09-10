from decimal import ROUND_FLOOR, Decimal

from bot.execution.types import (
    ExitDecision,
    ExitPolicy,
    ExitReason,
    InstrumentSpec,
    OpenPosition,
    OrderIntent,
    OrderSide,
    OrderType,
)
from bot.risk.types import MarketState
from bot.signals.types import TradingSignal

BPS = Decimal("10000")


def build_entry_intent(
    signal: TradingSignal,
    market: MarketState,
    order_notional: Decimal,
    order_type: OrderType,
    spec: InstrumentSpec,
) -> OrderIntent | None:
    """Turn an approved signal into an entry order.

    Post-only entries join the passive side of the book (buy at best bid) so the
    order rests as a maker; market entries cross it. Quantity is sized off the
    price the order will actually trade at, so the notional the risk gate
    approved is the notional taken on. Returns None when that rounds below the
    instrument's minimum order quantity.
    """
    side = OrderSide.from_signal(signal.action)
    reference_price = _reference_price(side, order_type, market)

    qty = _floor_to_step(order_notional / reference_price, spec.qty_step)
    if qty < spec.min_order_qty:
        return None

    limit_price = reference_price if order_type is OrderType.POST_ONLY else None
    return OrderIntent(
        symbol=signal.symbol,
        side=side,
        order_type=order_type,
        qty=qty,
        limit_price=limit_price,
        reduce_only=False,
        created_ts_ms=signal.timestamp_ms,
    )


def open_position(
    intent: OrderIntent,
    fill_price: Decimal,
    fill_ts_ms: int,
    policy: ExitPolicy,
) -> OpenPosition:
    take_profit_price, stop_loss_price = exit_prices(intent.side, fill_price, policy)
    return OpenPosition(
        symbol=intent.symbol,
        side=intent.side,
        qty=intent.qty,
        entry_price=fill_price,
        entry_ts_ms=fill_ts_ms,
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
        hold_deadline_ms=fill_ts_ms + policy.hold_ms,
    )


def exit_prices(
    side: OrderSide,
    entry_price: Decimal,
    policy: ExitPolicy,
) -> tuple[Decimal, Decimal]:
    """Take-profit and stop-loss trigger levels derived from the entry price.

    These are comparison thresholds, not order prices, so they stay off the tick
    grid — the exit itself is priced from the live book.
    """
    tp_offset = policy.take_profit_bps / BPS
    sl_offset = policy.stop_loss_bps / BPS

    if side is OrderSide.BUY:
        return (
            entry_price * (Decimal(1) + tp_offset),
            entry_price * (Decimal(1) - sl_offset),
        )
    return (
        entry_price * (Decimal(1) - tp_offset),
        entry_price * (Decimal(1) + sl_offset),
    )


def decide_exit(
    position: OpenPosition,
    market: MarketState,
    now_ms: int,
) -> ExitDecision:
    """Evaluate the exit conditions against the price the position can leave at.

    Stop-loss wins over take-profit when a single snapshot straddles both, and
    both win over hold expiry.
    """
    exit_price = market.best_bid if position.side is OrderSide.BUY else market.best_ask

    if _stop_loss_hit(position, exit_price):
        return ExitDecision(True, ExitReason.STOP_LOSS, exit_price)

    if _take_profit_hit(position, exit_price):
        return ExitDecision(True, ExitReason.TAKE_PROFIT, exit_price)

    if now_ms >= position.hold_deadline_ms:
        return ExitDecision(True, ExitReason.HOLD_EXPIRED, exit_price)

    return ExitDecision(False)


def build_exit_intent(
    position: OpenPosition,
    market: MarketState,
    order_type: OrderType,
    now_ms: int,
) -> OrderIntent:
    side = position.side.opposite
    limit_price = (
        _reference_price(side, order_type, market)
        if order_type is OrderType.POST_ONLY
        else None
    )
    return OrderIntent(
        symbol=position.symbol,
        side=side,
        order_type=order_type,
        qty=position.qty,
        limit_price=limit_price,
        reduce_only=True,
        created_ts_ms=now_ms,
    )


def should_cross_exit(
    position: OpenPosition,
    market: MarketState,
    now_ms: int,
    submitted_ms: int,
    policy: ExitPolicy,
) -> bool:
    """Whether a resting passive exit should be abandoned and crossed.

    The passive attempt has stopped being the cheaper option once it has rested
    for its whole window. A stop breach only adds to that when `cross_on_stop`
    is set — see `ExitPolicy.cross_on_stop` for why it is not the default.
    """
    if policy.exit_fallback_ms <= 0:
        return False
    if now_ms - submitted_ms >= policy.exit_fallback_ms:
        return True
    if not policy.cross_on_stop:
        return False

    exit_price = market.best_bid if position.side is OrderSide.BUY else market.best_ask
    return _stop_loss_hit(position, exit_price)


def passive_exit_price(side: OrderSide, market: MarketState) -> Decimal:
    """Where a passive exit has to rest to be the top of its own side."""
    return market.best_ask if side is OrderSide.SELL else market.best_bid


def should_repeg_exit(
    side: OrderSide,
    resting_price: Decimal,
    market: MarketState,
) -> bool:
    """Whether a resting passive exit has been left behind by the book.

    A passive exit is posted once at the top of its side and then the book moves.
    On a favourable move the level it sits at is consumed and it fills; on an
    adverse one it is stranded a tick or more away from the market and cannot
    fill at all — which is exactly the case the position needs out of. Following
    the book down keeps the exit reachable while still paying the smaller maker fee, and is
    the difference between leaving as a maker and paying the taker fee, which on
    these horizons costs more than the move being traded.
    """
    return resting_price != passive_exit_price(side, market)


def _reference_price(
    side: OrderSide,
    order_type: OrderType,
    market: MarketState,
) -> Decimal:
    """The price this order is expected to trade at — passive side for post-only,
    crossing side for market."""
    if order_type is OrderType.POST_ONLY:
        return market.best_bid if side is OrderSide.BUY else market.best_ask
    return market.best_ask if side is OrderSide.BUY else market.best_bid


def _stop_loss_hit(position: OpenPosition, exit_price: Decimal) -> bool:
    if position.side is OrderSide.BUY:
        return exit_price <= position.stop_loss_price
    return exit_price >= position.stop_loss_price


def _take_profit_hit(position: OpenPosition, exit_price: Decimal) -> bool:
    if position.side is OrderSide.BUY:
        return exit_price >= position.take_profit_price
    return exit_price <= position.take_profit_price


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step
