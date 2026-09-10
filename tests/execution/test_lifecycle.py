from decimal import Decimal

import pytest

from bot.execution.lifecycle import (
    build_entry_intent,
    build_exit_intent,
    decide_exit,
    exit_prices,
    open_position,
    should_cross_exit,
)
from bot.execution.types import (
    ExitPolicy,
    ExitReason,
    InstrumentSpec,
    OrderSide,
    OrderType,
)
from bot.risk import MarketState
from bot.signals import SignalAction, TradingSignal

SYMBOL = "BTCUSDT"
NOW_MS = 1_800_000_000_000

SPEC = InstrumentSpec(
    qty_step=Decimal("0.001"),
    price_tick=Decimal("0.1"),
    min_order_qty=Decimal("0.001"),
)

POLICY = ExitPolicy(
    hold_ms=1000,
    take_profit_bps=Decimal("15"),
    stop_loss_bps=Decimal("10"),
)


def _signal(action: SignalAction = SignalAction.BUY) -> TradingSignal:
    return TradingSignal(
        symbol=SYMBOL,
        timestamp_ms=NOW_MS,
        action=action,
        confidence=0.8,
        horizon_index=1,
        class_probs=(0.1, 0.1, 0.8),
    )


def _market(
    best_bid: Decimal = Decimal("10000"),
    best_ask: Decimal = Decimal("10001"),
    timestamp_ms: int = NOW_MS,
) -> MarketState:
    return MarketState(
        symbol=SYMBOL,
        timestamp_ms=timestamp_ms,
        best_bid=best_bid,
        best_ask=best_ask,
        account_equity=Decimal("1000"),
    )


class TestBuildEntryIntent:
    def test_buy_rests_on_the_bid(self) -> None:
        intent = build_entry_intent(
            _signal(), _market(), Decimal("100"), OrderType.POST_ONLY, SPEC
        )
        assert intent is not None
        assert intent.side is OrderSide.BUY
        assert intent.limit_price == Decimal("10000")
        assert intent.reduce_only is False

    def test_sell_rests_on_the_ask(self) -> None:
        intent = build_entry_intent(
            _signal(SignalAction.SELL), _market(), Decimal("100"),
            OrderType.POST_ONLY, SPEC,
        )
        assert intent is not None
        assert intent.side is OrderSide.SELL
        assert intent.limit_price == Decimal("10001")

    def test_market_order_has_no_limit_price(self) -> None:
        intent = build_entry_intent(
            _signal(), _market(), Decimal("100"), OrderType.MARKET, SPEC
        )
        assert intent is not None
        assert intent.limit_price is None

    def test_market_buy_sized_off_the_crossing_ask(self) -> None:
        intent = build_entry_intent(
            _signal(), _market(), Decimal("100"), OrderType.MARKET, SPEC
        )
        assert intent is not None
        # crosses at 10001, not the 10000 bid: 100 / 10001 → 0.009 after flooring
        assert intent.qty == Decimal("0.009")
        assert intent.qty * Decimal("10001") <= Decimal("100")

    def test_qty_floored_to_step(self) -> None:
        intent = build_entry_intent(
            _signal(), _market(), Decimal("100"), OrderType.POST_ONLY, SPEC
        )
        assert intent is not None
        # 100 / 10000 = 0.01 exactly on a 0.001 step
        assert intent.qty == Decimal("0.010")

    def test_qty_never_rounds_up(self) -> None:
        intent = build_entry_intent(
            _signal(), _market(), Decimal("109"), OrderType.POST_ONLY, SPEC
        )
        assert intent is not None
        # 109 / 10000 = 0.0109 → floored to 0.010
        assert intent.qty == Decimal("0.010")

    def test_below_min_qty_returns_none(self) -> None:
        assert build_entry_intent(
            _signal(), _market(), Decimal("5"), OrderType.POST_ONLY, SPEC
        ) is None

    def test_non_directional_action_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_entry_intent(
                _signal(SignalAction.NO_TRADE), _market(), Decimal("100"),
                OrderType.POST_ONLY, SPEC,
            )


class TestExitPrices:
    def test_long_brackets_entry(self) -> None:
        tp, sl = exit_prices(OrderSide.BUY, Decimal("10000"), POLICY)
        assert tp == Decimal("10015")
        assert sl == Decimal("9990")

    def test_short_brackets_entry_inverted(self) -> None:
        tp, sl = exit_prices(OrderSide.SELL, Decimal("10000"), POLICY)
        assert tp == Decimal("9985")
        assert sl == Decimal("10010")


class TestDecideExit:
    def _long(self) -> object:
        intent = build_entry_intent(
            _signal(), _market(), Decimal("100"), OrderType.POST_ONLY, SPEC
        )
        assert intent is not None
        return open_position(intent, Decimal("10000"), NOW_MS, POLICY)

    def test_holds_inside_the_bracket(self) -> None:
        decision = decide_exit(self._long(), _market(), NOW_MS + 500)
        assert decision.should_exit is False
        assert decision.reason is None

    def test_take_profit_on_the_bid(self) -> None:
        market = _market(best_bid=Decimal("10015"), best_ask=Decimal("10016"))
        decision = decide_exit(self._long(), market, NOW_MS + 500)
        assert decision.reason is ExitReason.TAKE_PROFIT
        assert decision.exit_price == Decimal("10015")

    def test_stop_loss_on_the_bid(self) -> None:
        market = _market(best_bid=Decimal("9990"), best_ask=Decimal("9991"))
        decision = decide_exit(self._long(), market, NOW_MS + 500)
        assert decision.reason is ExitReason.STOP_LOSS

    def test_hold_expiry(self) -> None:
        decision = decide_exit(self._long(), _market(), NOW_MS + 1000)
        assert decision.reason is ExitReason.HOLD_EXPIRED

    def test_stop_loss_wins_over_hold_expiry(self) -> None:
        market = _market(best_bid=Decimal("9000"), best_ask=Decimal("9001"))
        decision = decide_exit(self._long(), market, NOW_MS + 99_999)
        assert decision.reason is ExitReason.STOP_LOSS

    def test_short_take_profit_on_the_ask(self) -> None:
        intent = build_entry_intent(
            _signal(SignalAction.SELL), _market(), Decimal("100"),
            OrderType.POST_ONLY, SPEC,
        )
        assert intent is not None
        position = open_position(intent, Decimal("10000"), NOW_MS, POLICY)
        market = _market(best_bid=Decimal("9984"), best_ask=Decimal("9985"))
        decision = decide_exit(position, market, NOW_MS + 500)
        assert decision.reason is ExitReason.TAKE_PROFIT
        assert decision.exit_price == Decimal("9985")


class TestBuildExitIntent:
    def test_long_exits_by_selling_on_the_ask(self) -> None:
        intent = build_entry_intent(
            _signal(), _market(), Decimal("100"), OrderType.POST_ONLY, SPEC
        )
        assert intent is not None
        position = open_position(intent, Decimal("10000"), NOW_MS, POLICY)

        exit_intent = build_exit_intent(
            position, _market(), OrderType.POST_ONLY, NOW_MS + 1000
        )
        assert exit_intent.side is OrderSide.SELL
        assert exit_intent.reduce_only is True
        assert exit_intent.qty == position.qty
        assert exit_intent.limit_price == Decimal("10001")

    def test_market_exit_has_no_limit_price(self) -> None:
        intent = build_entry_intent(
            _signal(), _market(), Decimal("100"), OrderType.POST_ONLY, SPEC
        )
        assert intent is not None
        position = open_position(intent, Decimal("10000"), NOW_MS, POLICY)
        exit_intent = build_exit_intent(
            position, _market(), OrderType.MARKET, NOW_MS + 1000
        )
        assert exit_intent.limit_price is None


class TestShouldCrossExit:
    """A passive exit is abandoned when waiting stops being the cheaper option."""

    def _position(self, entry: str = "10000"):
        intent = build_entry_intent(
            _signal(), _market(), Decimal("100"), OrderType.POST_ONLY, SPEC
        )
        assert intent is not None
        return open_position(intent, Decimal(entry), NOW_MS, self._policy_only(2000))

    def _policy_only(self, fallback_ms: int, cross_on_stop: bool = False) -> ExitPolicy:
        return ExitPolicy(
            hold_ms=1000,
            take_profit_bps=Decimal("15"),
            stop_loss_bps=Decimal("10"),
            exit_fallback_ms=fallback_ms,
            cross_on_stop=cross_on_stop,
        )

    def test_rests_inside_its_window(self) -> None:
        assert not should_cross_exit(
            self._position(), _market(), NOW_MS + 500, NOW_MS, self._policy_only(2000)
        )

    def test_crossed_once_the_window_elapses(self) -> None:
        assert should_cross_exit(
            self._position(), _market(), NOW_MS + 2000, NOW_MS, self._policy_only(2000)
        )

    def test_stop_breach_crosses_immediately_when_asked_to(self) -> None:
        breached = _market(best_bid=Decimal("9985"), best_ask=Decimal("9986"))
        assert should_cross_exit(
            self._position(), breached, NOW_MS + 1, NOW_MS,
            self._policy_only(2000, cross_on_stop=True),
        )

    def test_stop_breach_alone_does_not_cross_by_default(self) -> None:
        """Crossing costs 5.5 bps taker to cap a stop of comparable width, so the
        breach decides when to leave and the fallback window decides how."""
        breached = _market(best_bid=Decimal("9985"), best_ask=Decimal("9986"))
        assert not should_cross_exit(
            self._position(), breached, NOW_MS + 1, NOW_MS, self._policy_only(2000)
        )

    def test_take_profit_does_not_force_a_cross(self) -> None:
        # a favourable move is no reason to pay the taker fee
        rich = _market(best_bid=Decimal("10050"), best_ask=Decimal("10051"))
        assert not should_cross_exit(
            self._position(), rich, NOW_MS + 1, NOW_MS, self._policy_only(2000)
        )

    def test_zero_fallback_never_escalates(self) -> None:
        breached = _market(best_bid=Decimal("9000"), best_ask=Decimal("9001"))
        assert not should_cross_exit(
            self._position(), breached, NOW_MS + 99_999, NOW_MS, self._policy_only(0)
        )

    def test_short_position_uses_the_ask(self) -> None:
        intent = build_entry_intent(
            _signal(SignalAction.SELL), _market(), Decimal("100"),
            OrderType.POST_ONLY, SPEC,
        )
        assert intent is not None
        position = open_position(intent, Decimal("10000"), NOW_MS, self._policy_only(2000))
        # a short is stopped out when the ask rises through the level
        breached = _market(best_bid=Decimal("10009"), best_ask=Decimal("10011"))
        assert should_cross_exit(
            position, breached, NOW_MS + 1, NOW_MS,
            self._policy_only(2000, cross_on_stop=True),
        )
