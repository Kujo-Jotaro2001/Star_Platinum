from decimal import Decimal

from bot.execution.orders import (
    cancel_order_params,
    format_decimal,
    order_link_id,
    place_order_params,
)
from bot.execution.types import InstrumentSpec, OrderIntent, OrderSide, OrderType

SYMBOL = "BTCUSDT"
NOW_MS = 1_800_000_000_000

SPEC = InstrumentSpec(
    qty_step=Decimal("0.001"),
    price_tick=Decimal("0.1"),
    min_order_qty=Decimal("0.001"),
)


def _intent(
    order_type: OrderType = OrderType.POST_ONLY,
    side: OrderSide = OrderSide.BUY,
    reduce_only: bool = False,
    limit_price: Decimal | None = Decimal("10000"),
) -> OrderIntent:
    return OrderIntent(
        symbol=SYMBOL,
        side=side,
        order_type=order_type,
        qty=Decimal("0.01"),
        limit_price=limit_price if order_type is OrderType.POST_ONLY else None,
        reduce_only=reduce_only,
        created_ts_ms=NOW_MS,
    )


class TestFormatDecimal:
    def test_pads_to_the_step_precision(self) -> None:
        assert format_decimal(Decimal("0.01"), Decimal("0.001")) == "0.010"

    def test_price_at_tick_precision(self) -> None:
        assert format_decimal(Decimal("10000"), Decimal("0.1")) == "10000.0"

    def test_no_scientific_notation_for_small_values(self) -> None:
        assert format_decimal(Decimal("1E-3"), Decimal("0.001")) == "0.001"


class TestPlaceOrderParams:
    def test_post_only_is_a_limit_order_with_time_in_force(self) -> None:
        params = place_order_params(_intent(), SPEC)
        assert params["orderType"] == "Limit"
        assert params["timeInForce"] == "PostOnly"
        assert params["price"] == "10000.0"

    def test_market_order_carries_no_price(self) -> None:
        params = place_order_params(_intent(OrderType.MARKET), SPEC)
        assert params["orderType"] == "Market"
        assert "price" not in params
        assert "timeInForce" not in params

    def test_common_fields(self) -> None:
        params = place_order_params(_intent(), SPEC)
        assert params["category"] == "linear"
        assert params["symbol"] == SYMBOL
        assert params["side"] == "Buy"
        assert params["qty"] == "0.010"
        assert params["reduceOnly"] is False

    def test_reduce_only_propagates(self) -> None:
        params = place_order_params(_intent(reduce_only=True), SPEC)
        assert params["reduceOnly"] is True

    def test_every_value_bybit_receives_is_a_string_or_bool(self) -> None:
        params = place_order_params(_intent(), SPEC)
        assert all(isinstance(v, (str, bool)) for v in params.values())


class TestOrderLinkId:
    def test_entry_and_exit_legs_differ(self) -> None:
        assert order_link_id(_intent()) != order_link_id(_intent(reduce_only=True))

    def test_sides_differ(self) -> None:
        buy = order_link_id(_intent(side=OrderSide.BUY))
        sell = order_link_id(_intent(side=OrderSide.SELL))
        assert buy != sell

    def test_within_bybit_length_limit(self) -> None:
        assert len(order_link_id(_intent())) <= 36

    def test_matches_the_id_sent_to_the_exchange(self) -> None:
        intent = _intent()
        assert place_order_params(intent, SPEC)["orderLinkId"] == order_link_id(intent)


class TestCancelOrderParams:
    def test_targets_the_order_by_link_id(self) -> None:
        params = cancel_order_params(SYMBOL, "sp-e-b-123")
        assert params == {
            "category": "linear",
            "symbol": SYMBOL,
            "orderLinkId": "sp-e-b-123",
        }
