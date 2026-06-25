import math
from collections.abc import Sequence

from bot.signals.types import (
    ModelPrediction,
    SignalAction,
    SignalPolicy,
    TradingSignal,
)

DOWN_CLASS = 0
FLAT_CLASS = 1
UP_CLASS = 2
CLASS_COUNT = 3


def prediction_from_logits(logits: Sequence[Sequence[float]]) -> ModelPrediction:
    probs = tuple(_softmax(row) for row in logits)
    return ModelPrediction(class_probs=probs)


def generate_signal(
    symbol: str,
    timestamp_ms: int,
    prediction: ModelPrediction,
    policy: SignalPolicy,
) -> TradingSignal:
    if policy.horizon_index < 0 or policy.horizon_index >= len(prediction.class_probs):
        raise ValueError(
            f"horizon_index {policy.horizon_index} is outside prediction horizons "
            f"0..{len(prediction.class_probs) - 1}"
        )

    class_probs = prediction.class_probs[policy.horizon_index]
    if len(class_probs) != CLASS_COUNT:
        raise ValueError(f"class_probs must contain {CLASS_COUNT} class probabilities")

    selected_class = max(range(CLASS_COUNT), key=lambda i: class_probs[i])
    confidence = float(class_probs[selected_class])

    if selected_class == FLAT_CLASS:
        return _signal(
            symbol, timestamp_ms, policy.horizon_index, class_probs,
            SignalAction.NO_TRADE, confidence, "flat_selected",
        )

    if confidence < policy.min_confidence:
        return _signal(
            symbol, timestamp_ms, policy.horizon_index, class_probs,
            SignalAction.NO_TRADE, confidence, "confidence_below_threshold",
        )

    if selected_class == UP_CLASS:
        return _signal(
            symbol, timestamp_ms, policy.horizon_index, class_probs,
            SignalAction.BUY, confidence, None,
        )

    if selected_class == DOWN_CLASS and policy.allow_short:
        return _signal(
            symbol, timestamp_ms, policy.horizon_index, class_probs,
            SignalAction.SELL, confidence, None,
        )

    return _signal(
        symbol, timestamp_ms, policy.horizon_index, class_probs,
        SignalAction.NO_TRADE, confidence, "short_disabled",
    )


def _signal(
    symbol: str,
    timestamp_ms: int,
    horizon_index: int,
    class_probs: tuple[float, float, float],
    action: SignalAction,
    confidence: float,
    reason: str | None,
) -> TradingSignal:
    return TradingSignal(
        symbol=symbol,
        timestamp_ms=timestamp_ms,
        action=action,
        confidence=confidence,
        horizon_index=horizon_index,
        class_probs=class_probs,
        reason=reason,
    )


def _softmax(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != CLASS_COUNT:
        raise ValueError(f"logits must contain {CLASS_COUNT} classes")
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    return tuple(float(v / total) for v in exps)  # type: ignore[return-value]
