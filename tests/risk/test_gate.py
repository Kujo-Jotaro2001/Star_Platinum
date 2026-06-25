from decimal import Decimal

from bot.risk import MarketState, PositionState, RiskPolicy, evaluate_risk
from bot.signals import SignalAction, TradingSignal

SYMBOL = "BTCUSDT"
NOW_MS = 1_800_000_000_000


def _signal(
    action: SignalAction = SignalAction.BUY,
    confidence: float = 0.80,
) -> TradingSignal:
    return TradingSignal(
        symbol=SYMBOL,
        timestamp_ms=NOW_MS,
        action=action,
        confidence=confidence,
        horizon_index=1,
        class_probs=(0.1, 0.1, 0.8),
    )


def _position(
    quantity: Decimal = Decimal("0"),
    last_trade_timestamp_ms: int | None = None,
) -> PositionState:
    return PositionState(
        symbol=SYMBOL,
        quantity=quantity,
        last_trade_timestamp_ms=last_trade_timestamp_ms,
    )


def _market(
    timestamp_ms: int = NOW_MS,
    best_bid: Decimal = Decimal("9999"),
    best_ask: Decimal = Decimal("10001"),
    account_equity: Decimal = Decimal("1000"),
) -> MarketState:
    return MarketState(
        symbol=SYMBOL,
        timestamp_ms=timestamp_ms,
        best_bid=best_bid,
        best_ask=best_ask,
        account_equity=account_equity,
    )


def _policy(
    allow_short: bool = False,
    max_position_notional: Decimal = Decimal("100"),
    max_leverage: Decimal = Decimal("1.0"),
    max_spread_bps: Decimal = Decimal("5.0"),
    stale_data_ms: int = 1000,
    cooldown_after_trade_ms: int = 5000,
    min_confidence: float = 0.55,
) -> RiskPolicy:
    return RiskPolicy(
        max_position_notional=max_position_notional,
        max_leverage=max_leverage,
        max_spread_bps=max_spread_bps,
        stale_data_ms=stale_data_ms,
        cooldown_after_trade_ms=cooldown_after_trade_ms,
        min_confidence=min_confidence,
        allow_short=allow_short,
    )


class TestEvaluateRisk:
    def test_approved_buy_under_normal_conditions(self) -> None:
        result = evaluate_risk(
            signal=_signal(),
            position=_position(),
            market=_market(),
            policy=_policy(),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert result.approved is True
        assert result.reason == "approved"

    def test_rejected_stale_data(self) -> None:
        result = evaluate_risk(
            signal=_signal(),
            position=_position(),
            market=_market(timestamp_ms=NOW_MS - 1001),
            policy=_policy(stale_data_ms=1000),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert result.approved is False
        assert result.reason == "stale_market_data"

    def test_rejected_wide_spread(self) -> None:
        result = evaluate_risk(
            signal=_signal(),
            position=_position(),
            market=_market(best_bid=Decimal("9990"), best_ask=Decimal("10010")),
            policy=_policy(max_spread_bps=Decimal("5.0")),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert result.approved is False
        assert result.reason == "spread_too_wide"

    def test_rejected_low_confidence(self) -> None:
        result = evaluate_risk(
            signal=_signal(confidence=0.54),
            position=_position(),
            market=_market(),
            policy=_policy(min_confidence=0.55),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert result.approved is False
        assert result.reason == "confidence_below_threshold"

    def test_rejected_short_when_short_disabled(self) -> None:
        result = evaluate_risk(
            signal=_signal(action=SignalAction.SELL),
            position=_position(),
            market=_market(),
            policy=_policy(allow_short=False),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert result.approved is False
        assert result.reason == "short_disabled"

    def test_rejected_max_notional_breach(self) -> None:
        result = evaluate_risk(
            signal=_signal(),
            position=_position(quantity=Decimal("0.006")),
            market=_market(),
            policy=_policy(max_position_notional=Decimal("100")),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert result.approved is False
        assert result.reason == "max_position_notional_exceeded"

    def test_rejected_max_leverage_breach(self) -> None:
        result = evaluate_risk(
            signal=_signal(),
            position=_position(),
            market=_market(account_equity=Decimal("40")),
            policy=_policy(max_position_notional=Decimal("1000"), max_leverage=Decimal("1.0")),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert result.approved is False
        assert result.reason == "max_leverage_exceeded"

    def test_rejected_cooldown_violation(self) -> None:
        result = evaluate_risk(
            signal=_signal(),
            position=_position(last_trade_timestamp_ms=NOW_MS - 1000),
            market=_market(),
            policy=_policy(cooldown_after_trade_ms=5000),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert result.approved is False
        assert result.reason == "cooldown_active"
