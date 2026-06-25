from dataclasses import dataclass
from decimal import Decimal

from bot.signals.types import TradingSignal


@dataclass(frozen=True, slots=True)
class PositionState:
    symbol: str
    quantity: Decimal
    last_trade_timestamp_ms: int | None = None


@dataclass(frozen=True, slots=True)
class MarketState:
    symbol: str
    timestamp_ms: int
    best_bid: Decimal
    best_ask: Decimal
    account_equity: Decimal


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    max_position_notional: Decimal
    max_leverage: Decimal
    max_spread_bps: Decimal
    stale_data_ms: int
    cooldown_after_trade_ms: int
    min_confidence: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class RiskDecision:
    approved: bool
    reason: str
    signal: TradingSignal | None = None
