from decimal import Decimal

from bot.risk.types import MarketState, PositionState, RiskDecision, RiskPolicy
from bot.signals.types import SignalAction, TradingSignal

BPS = Decimal("10000")


def evaluate_risk(
    signal: TradingSignal,
    position: PositionState,
    market: MarketState,
    policy: RiskPolicy,
    proposed_order_notional: Decimal,
    now_ms: int,
) -> RiskDecision:
    if signal.action not in (SignalAction.BUY, SignalAction.SELL):
        return RiskDecision(False, "signal_not_actionable", signal)

    if signal.symbol != position.symbol or signal.symbol != market.symbol:
        return RiskDecision(False, "symbol_mismatch", signal)

    if now_ms - market.timestamp_ms > policy.stale_data_ms:
        return RiskDecision(False, "stale_market_data", signal)

    spread_bps = _spread_bps(market)
    if spread_bps > policy.max_spread_bps:
        return RiskDecision(False, "spread_too_wide", signal)

    if signal.confidence < policy.min_confidence:
        return RiskDecision(False, "confidence_below_threshold", signal)

    if signal.action == SignalAction.SELL and not policy.allow_short:
        return RiskDecision(False, "short_disabled", signal)

    if _cooldown_active(position, policy, now_ms):
        return RiskDecision(False, "cooldown_active", signal)

    resulting_notional = _resulting_notional(
        signal.action, position, market, proposed_order_notional
    )
    if resulting_notional > policy.max_position_notional:
        return RiskDecision(False, "max_position_notional_exceeded", signal)

    leverage = _leverage(resulting_notional, market.account_equity)
    if leverage > policy.max_leverage:
        return RiskDecision(False, "max_leverage_exceeded", signal)

    return RiskDecision(True, "approved", signal)


def _spread_bps(market: MarketState) -> Decimal:
    mid = (market.best_bid + market.best_ask) / Decimal("2")
    if mid <= 0:
        return Decimal("Infinity")
    return (market.best_ask - market.best_bid) / mid * BPS


def _cooldown_active(
    position: PositionState,
    policy: RiskPolicy,
    now_ms: int,
) -> bool:
    if position.last_trade_timestamp_ms is None:
        return False
    elapsed_ms = now_ms - position.last_trade_timestamp_ms
    return elapsed_ms < policy.cooldown_after_trade_ms


def _resulting_notional(
    action: SignalAction,
    position: PositionState,
    market: MarketState,
    proposed_order_notional: Decimal,
) -> Decimal:
    signed_position = position.quantity * _mid_price(market)
    signed_delta = proposed_order_notional
    if action == SignalAction.SELL:
        signed_delta = -proposed_order_notional
    return abs(signed_position + signed_delta)


def _leverage(notional: Decimal, account_equity: Decimal) -> Decimal:
    if account_equity <= 0:
        return Decimal("Infinity")
    return notional / account_equity


def _mid_price(market: MarketState) -> Decimal:
    return (market.best_bid + market.best_ask) / Decimal("2")
