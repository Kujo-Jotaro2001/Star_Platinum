from decimal import Decimal

import pytest

from bot.execution.lifecycle import passive_exit_price, should_repeg_exit
from bot.execution.types import OrderSide
from bot.risk.types import MarketState


def market(bid: str, ask: str) -> MarketState:
    return MarketState(
        symbol="SOLUSDT",
        timestamp_ms=1_700_000_000_000,
        best_bid=Decimal(bid),
        best_ask=Decimal(ask),
        account_equity=Decimal("10000"),
    )


def test_passive_exit_price_joins_its_own_side():
    m = market("195.00", "195.01")
    assert passive_exit_price(OrderSide.SELL, m) == Decimal("195.01")
    assert passive_exit_price(OrderSide.BUY, m) == Decimal("195.00")


def test_no_repeg_while_the_exit_is_still_at_the_top():
    m = market("195.00", "195.01")
    assert not should_repeg_exit(OrderSide.SELL, Decimal("195.01"), m)
    assert not should_repeg_exit(OrderSide.BUY, Decimal("195.00"), m)


def test_repeg_when_the_book_falls_away_from_a_sell_exit():
    """The case that strands a long: the ask drops and our sell sits above it."""
    m = market("194.98", "194.99")
    assert should_repeg_exit(OrderSide.SELL, Decimal("195.01"), m)


def test_repeg_when_the_book_rises_away_from_a_buy_exit():
    m = market("195.02", "195.03")
    assert should_repeg_exit(OrderSide.BUY, Decimal("195.00"), m)


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_repegging_is_idempotent(side: OrderSide):
    """Re-posting at the price the rule asks for leaves nothing more to do."""
    m = market("195.00", "195.01")
    assert not should_repeg_exit(side, passive_exit_price(side, m), m)
