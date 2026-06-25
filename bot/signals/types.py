from dataclasses import dataclass
from enum import Enum


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True, slots=True)
class SignalPolicy:
    horizon_index: int
    min_confidence: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    class_probs: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class TradingSignal:
    symbol: str
    timestamp_ms: int
    action: SignalAction
    confidence: float
    horizon_index: int
    class_probs: tuple[float, float, float]
    reason: str | None = None
