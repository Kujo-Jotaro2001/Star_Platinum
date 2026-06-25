from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    price: Decimal
    qty: Decimal


@dataclass(slots=True)
class OrderBookSnapshot:
    timestamp_ms: int
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    is_reset: bool = False


@dataclass(frozen=True, slots=True)
class Trade:
    timestamp_ms: int
    trade_id: str
    side: str  # "Buy" | "Sell"
    price: Decimal
    qty: Decimal


@dataclass(frozen=True, slots=True)
class TickerContext:
    timestamp_ms: int
    mark_price: Decimal | None
    index_price: Decimal | None
    open_interest: Decimal | None        # in BTC
    open_interest_value: Decimal | None  # in USDT
    funding_rate: Decimal | None
    next_funding_time: int | None        # unix ms
    price_24h_pct: Decimal | None
    prev_price_1h: Decimal | None
    volume_24h: Decimal | None
    turnover_24h: Decimal | None
    collected_at_ms: int


@dataclass(frozen=True, slots=True)
class Liquidation:
    timestamp_ms: int   # from field "T"
    side: str           # "Buy" | "Sell"
    qty: Decimal        # from field "v"
    price: Decimal      # from field "p"
    collected_at_ms: int


@dataclass(frozen=True, slots=True)
class LongShortRatio:
    timestamp_ms: int   # from API "timestamp" field
    buy_ratio: Decimal  # from "buyRatio"
    sell_ratio: Decimal  # from "sellRatio"
    collected_at_ms: int
