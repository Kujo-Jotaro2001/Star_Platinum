from decimal import Decimal

from bot.execution.types import InstrumentSpec, OrderIntent, OrderType

CATEGORY = "linear"


def place_order_params(intent: OrderIntent, spec: InstrumentSpec) -> dict:
    """Build the kwargs for pybit's HTTP.place_order from an order intent.

    Bybit wants prices and quantities as strings at the instrument's own
    precision; a value with more decimals than the tick or step is rejected.
    """
    params = {
        "category": CATEGORY,
        "symbol": intent.symbol,
        "side": intent.side.value,
        "orderType": _order_type(intent.order_type),
        "qty": format_decimal(intent.qty, spec.qty_step),
        "reduceOnly": intent.reduce_only,
        "orderLinkId": order_link_id(intent),
    }

    if intent.order_type is OrderType.POST_ONLY:
        params["price"] = format_decimal(intent.limit_price, spec.price_tick)
        params["timeInForce"] = "PostOnly"

    return params


def cancel_order_params(symbol: str, order_link_id_value: str) -> dict:
    return {
        "category": CATEGORY,
        "symbol": symbol,
        "orderLinkId": order_link_id_value,
    }


def order_link_id(intent: OrderIntent) -> str:
    """Client-side order id — lets us match WS order updates to our own intents."""
    leg = "x" if intent.reduce_only else "e"
    return f"sp-{leg}-{intent.side.value[0].lower()}-{intent.created_ts_ms}"


def format_decimal(value: Decimal, step: Decimal) -> str:
    """Render at exactly the precision of `step`, without scientific notation."""
    return f"{value.quantize(step):f}"


def _order_type(order_type: OrderType) -> str:
    """Post-only is a Limit order carrying timeInForce=PostOnly, not its own type."""
    return "Market" if order_type is OrderType.MARKET else "Limit"
