from decimal import Decimal

from bot.execution.account import AccountState
from bot.execution.types import OrderSide

SYMBOL = "BTCUSDT"
NOW_MS = 1_800_000_000_000


def _position_row(side: str, size: str, symbol: str = SYMBOL) -> dict:
    return {
        "symbol": symbol,
        "side": side,
        "size": size,
        "entryPrice": "10000",
    }


def _order_row(
    status: str = "New",
    cum_exec_qty: str = "0",
    link_id: str = "sp-e-b-1",
    avg_price: str = "",
    symbol: str = SYMBOL,
    updated_ms: int = NOW_MS,
) -> dict:
    return {
        "symbol": symbol,
        "orderId": "abc",
        "orderLinkId": link_id,
        "orderStatus": status,
        "side": "Buy",
        "qty": "0.01",
        "cumExecQty": cum_exec_qty,
        "avgPrice": avg_price,
        "updatedTime": str(updated_ms),
    }


class TestPosition:
    def test_starts_flat(self) -> None:
        account = AccountState(SYMBOL)
        assert account.is_flat
        assert account.position_qty == 0

    def test_long_is_positive(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_position([_position_row("Buy", "0.01")])
        assert account.position_qty == Decimal("0.01")
        assert not account.is_flat

    def test_short_is_negative(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_position([_position_row("Sell", "0.01")])
        assert account.position_qty == Decimal("-0.01")

    def test_closed_position_reports_flat(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_position([_position_row("Buy", "0.01")])
        account.apply_position([_position_row("", "0")])
        assert account.is_flat

    def test_other_symbols_ignored(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_position([_position_row("Buy", "5", symbol="ETHUSDT")])
        assert account.is_flat

    def test_entry_price_tracked(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_position([_position_row("Buy", "0.01")])
        assert account.entry_price == Decimal("10000")


class TestWallet:
    def test_equity_from_total_equity(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_wallet([{"accountType": "UNIFIED", "totalEquity": "1234.5"}])
        assert account.equity == Decimal("1234.5")

    def test_missing_equity_reads_as_zero(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_wallet([{"accountType": "UNIFIED", "totalEquity": ""}])
        assert account.equity == Decimal(0)


class TestOrders:
    def test_unknown_order_is_none(self) -> None:
        assert AccountState(SYMBOL).order("nope") is None

    def test_new_order_is_open(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_order([_order_row(status="New")])
        order = account.order("sp-e-b-1")
        assert order is not None
        assert order.is_open
        assert not order.is_filled
        assert order.side is OrderSide.BUY

    def test_filled_order_carries_average_price(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_order(
            [_order_row(status="Filled", cum_exec_qty="0.01", avg_price="9999.5")]
        )
        order = account.order("sp-e-b-1")
        assert order is not None
        assert order.is_filled
        assert not order.is_open
        assert order.avg_price == Decimal("9999.5")

    def test_cancelled_order_is_neither_open_nor_filled(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_order([_order_row(status="Cancelled")])
        order = account.order("sp-e-b-1")
        assert order is not None
        assert not order.is_open
        assert not order.is_filled

    def test_updates_replace_earlier_state(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_order([_order_row(status="New")])
        account.apply_order([_order_row(status="Filled", cum_exec_qty="0.01")])
        assert account.order("sp-e-b-1").is_filled

    def test_has_open_orders_tracks_lifecycle(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_order([_order_row(status="New")])
        assert account.has_open_orders()
        account.apply_order([_order_row(status="Filled", cum_exec_qty="0.01")])
        assert not account.has_open_orders()

    def test_forget_closed_orders_keeps_only_open_ones(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_order([_order_row(status="Filled", link_id="done")])
        account.apply_order([_order_row(status="New", link_id="live")])
        account.forget_closed_orders()
        assert account.order("done") is None
        assert account.order("live") is not None

    def test_other_symbols_ignored(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_order([_order_row(symbol="ETHUSDT")])
        assert account.order("sp-e-b-1") is None


class TestPositionState:
    def test_last_trade_timestamp_set_on_first_execution(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_order([_order_row(status="New", updated_ms=NOW_MS)])
        assert account.position_state().last_trade_timestamp_ms == NOW_MS

    def test_last_trade_timestamp_advances_on_further_fills(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_order([_order_row(status="New", updated_ms=NOW_MS)])
        account.apply_order(
            [
                _order_row(
                    status="PartiallyFilled",
                    cum_exec_qty="0.005",
                    updated_ms=NOW_MS + 500,
                )
            ]
        )
        assert account.position_state().last_trade_timestamp_ms == NOW_MS + 500

    def test_status_change_without_a_fill_does_not_advance_it(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_order([_order_row(status="New", updated_ms=NOW_MS)])
        account.apply_order([_order_row(status="Cancelled", updated_ms=NOW_MS + 900)])
        assert account.position_state().last_trade_timestamp_ms == NOW_MS

    def test_carries_the_current_quantity(self) -> None:
        account = AccountState(SYMBOL)
        account.apply_position([_position_row("Sell", "0.02")])
        state = account.position_state()
        assert state.symbol == SYMBOL
        assert state.quantity == Decimal("-0.02")
