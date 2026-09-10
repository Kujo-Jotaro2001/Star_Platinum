from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from bot.signals.types import SignalAction


class OrderSide(str, Enum):
    BUY = "Buy"
    SELL = "Sell"

    @classmethod
    def from_signal(cls, action: SignalAction) -> "OrderSide":
        if action == SignalAction.BUY:
            return cls.BUY
        if action == SignalAction.SELL:
            return cls.SELL
        raise ValueError(f"{action} is not a directional action")

    @property
    def opposite(self) -> "OrderSide":
        return OrderSide.SELL if self is OrderSide.BUY else OrderSide.BUY


class OrderType(str, Enum):
    MARKET = "Market"
    POST_ONLY = "PostOnly"


class ExitReason(str, Enum):
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    HOLD_EXPIRED = "HOLD_EXPIRED"


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    """Hold-until-horizon exit with protective take-profit / stop-loss.

    `hold_ms` is the trained horizon expressed in wall-clock time so the same
    policy drives the backtest replay and the live loop.
    """

    hold_ms: int
    take_profit_bps: Decimal
    stop_loss_bps: Decimal
    exit_fallback_ms: int = 0
    """How long a passive exit may rest before it is crossed with a market order.

    Zero means never escalate. A protective exit that never fills is not
    protective, so the passive attempt is abandoned once it has had its window.
    """

    cross_on_stop: bool = False
    """Whether breaching the stop also crosses immediately, rather than waiting.

    Off by default because the arithmetic does not support it: crossing costs
    5.5 bps of taker fee against 2 bps of maker fee (VIP0, no rebate), so forcing it to cap a
    stop of comparable width pays more than the loss it prevents. The stop
    decides *when* to leave; `exit_fallback_ms` decides *how*.
    """


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Bybit instrument rounding rules — from /v5/market/instruments-info."""

    qty_step: Decimal
    price_tick: Decimal
    min_order_qty: Decimal


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: Decimal
    limit_price: Decimal | None
    reduce_only: bool
    created_ts_ms: int


@dataclass(frozen=True, slots=True)
class OpenPosition:
    symbol: str
    side: OrderSide
    qty: Decimal
    entry_price: Decimal
    entry_ts_ms: int
    take_profit_price: Decimal
    stop_loss_price: Decimal
    hold_deadline_ms: int


@dataclass(frozen=True, slots=True)
class ExitDecision:
    should_exit: bool
    reason: ExitReason | None = None
    exit_price: Decimal | None = None
