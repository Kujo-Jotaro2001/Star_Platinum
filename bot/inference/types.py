from dataclasses import dataclass
from enum import Enum

from bot.risk.types import RiskDecision
from bot.signals.types import TradingSignal


class InferenceStatus(str, Enum):
    NOT_READY = "NOT_READY"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class InferenceResult:
    timestamp_ms: int
    status: InferenceStatus
    probabilities: tuple[tuple[float, float, float], ...] | None
    signal: TradingSignal | None
    risk_decision: RiskDecision | None
    reason: str
