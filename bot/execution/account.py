import threading
from dataclasses import dataclass
from decimal import Decimal

from bot.execution.types import OrderSide
from bot.risk.types import PositionState

OPEN_ORDER_STATUSES = frozenset({"Created", "New", "PartiallyFilled", "Untriggered"})


@dataclass(frozen=True, slots=True)
class OrderUpdate:
    order_link_id: str
    order_id: str
    status: str
    side: OrderSide
    qty: Decimal
    cum_exec_qty: Decimal
    avg_price: Decimal | None
    updated_ms: int

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_ORDER_STATUSES

    @property
    def is_filled(self) -> bool:
        return self.status == "Filled"


class AccountState:
    """Position, equity and own-order state fed by the Bybit private WS.

    `apply_*` run on pybit's WS thread; the readers run on the event loop, so
    every field is guarded the same way OrderBookManager guards the book.
    """

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol
        self._lock = threading.Lock()
        self._position_qty = Decimal(0)
        self._entry_price = Decimal(0)
        self._equity = Decimal(0)
        self._orders: dict[str, OrderUpdate] = {}
        self._last_fill_ms: int | None = None

    def apply_position(self, rows: list[dict]) -> None:
        with self._lock:
            for row in rows:
                if row["symbol"] != self._symbol:
                    continue
                self._position_qty = _signed_qty(row["side"], row["size"])
                self._entry_price = _decimal(row.get("entryPrice"))

    def apply_wallet(self, rows: list[dict]) -> None:
        with self._lock:
            for row in rows:
                self._equity = _decimal(row.get("totalEquity"))

    def apply_order(self, rows: list[dict]) -> None:
        with self._lock:
            for row in rows:
                if row["symbol"] != self._symbol:
                    continue
                update = _order_update(row)
                previous = self._orders.get(update.order_link_id)
                if previous is None or update.cum_exec_qty > previous.cum_exec_qty:
                    self._last_fill_ms = update.updated_ms
                self._orders[update.order_link_id] = update

    @property
    def equity(self) -> Decimal:
        with self._lock:
            return self._equity

    @property
    def position_qty(self) -> Decimal:
        with self._lock:
            return self._position_qty

    @property
    def entry_price(self) -> Decimal:
        with self._lock:
            return self._entry_price

    @property
    def is_flat(self) -> bool:
        return self.position_qty == 0

    def position_state(self) -> PositionState:
        with self._lock:
            return PositionState(
                symbol=self._symbol,
                quantity=self._position_qty,
                last_trade_timestamp_ms=self._last_fill_ms,
            )

    def order(self, order_link_id: str) -> OrderUpdate | None:
        with self._lock:
            return self._orders.get(order_link_id)

    def has_open_orders(self) -> bool:
        with self._lock:
            return any(order.is_open for order in self._orders.values())

    def forget_closed_orders(self) -> None:
        """Drop settled orders so the map does not grow for the life of the process."""
        with self._lock:
            self._orders = {
                link_id: order
                for link_id, order in self._orders.items()
                if order.is_open
            }


def _order_update(row: dict) -> OrderUpdate:
    return OrderUpdate(
        order_link_id=row["orderLinkId"],
        order_id=row["orderId"],
        status=row["orderStatus"],
        side=OrderSide(row["side"]),
        qty=_decimal(row.get("qty")),
        cum_exec_qty=_decimal(row.get("cumExecQty")),
        avg_price=Decimal(row["avgPrice"]) if row.get("avgPrice") else None,
        updated_ms=int(row["updatedTime"]),
    )


def _signed_qty(side: str, size: str) -> Decimal:
    """Bybit reports size unsigned with the direction in `side`; "" means flat."""
    if side == OrderSide.SELL.value:
        return -_decimal(size)
    return _decimal(size)


def _decimal(value: str | None) -> Decimal:
    return Decimal(value) if value else Decimal(0)
