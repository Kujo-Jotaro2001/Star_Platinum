import pytest

from bot.signals import (
    ModelPrediction,
    SignalAction,
    SignalPolicy,
    generate_signal,
    prediction_from_logits,
)

SYMBOL = "BTCUSDT"
TS = 1_800_000_000_000


def _policy(allow_short: bool = False, min_confidence: float = 0.55) -> SignalPolicy:
    return SignalPolicy(
        horizon_index=1,
        min_confidence=min_confidence,
        allow_short=allow_short,
    )


class TestGenerateSignal:
    def test_high_up_probability_produces_buy(self) -> None:
        prediction = ModelPrediction(class_probs=((0.2, 0.7, 0.1), (0.1, 0.2, 0.7)))

        signal = generate_signal(SYMBOL, TS, prediction, _policy())

        assert signal.action == SignalAction.BUY
        assert signal.confidence == 0.7
        assert signal.horizon_index == 1
        assert signal.reason is None

    def test_high_down_probability_produces_sell_when_short_allowed(self) -> None:
        prediction = ModelPrediction(class_probs=((0.2, 0.7, 0.1), (0.8, 0.1, 0.1)))

        signal = generate_signal(SYMBOL, TS, prediction, _policy(allow_short=True))

        assert signal.action == SignalAction.SELL
        assert signal.confidence == 0.8
        assert signal.reason is None

    def test_high_down_probability_no_trade_when_short_disabled(self) -> None:
        prediction = ModelPrediction(class_probs=((0.2, 0.7, 0.1), (0.8, 0.1, 0.1)))

        signal = generate_signal(SYMBOL, TS, prediction, _policy(allow_short=False))

        assert signal.action == SignalAction.NO_TRADE
        assert signal.reason == "short_disabled"

    def test_high_flat_probability_produces_no_trade(self) -> None:
        prediction = ModelPrediction(class_probs=((0.2, 0.7, 0.1), (0.1, 0.8, 0.1)))

        signal = generate_signal(SYMBOL, TS, prediction, _policy())

        assert signal.action == SignalAction.NO_TRADE
        assert signal.confidence == 0.8
        assert signal.reason == "flat_selected"

    def test_low_confidence_produces_no_trade(self) -> None:
        prediction = ModelPrediction(class_probs=((0.2, 0.7, 0.1), (0.2, 0.25, 0.54)))

        signal = generate_signal(SYMBOL, TS, prediction, _policy(min_confidence=0.55))

        assert signal.action == SignalAction.NO_TRADE
        assert signal.confidence == 0.54
        assert signal.reason == "confidence_below_threshold"

    def test_invalid_horizon_index_raises_value_error(self) -> None:
        prediction = ModelPrediction(class_probs=((0.2, 0.7, 0.1),))
        policy = SignalPolicy(horizon_index=3, min_confidence=0.55, allow_short=False)

        with pytest.raises(ValueError, match="horizon_index"):
            generate_signal(SYMBOL, TS, prediction, policy)

    def test_prediction_from_logits_converts_to_probabilities(self) -> None:
        prediction = prediction_from_logits(((0.0, 0.0, 2.0),))

        probs = prediction.class_probs[0]
        assert probs[2] > probs[0]
        assert probs[2] > probs[1]
        assert sum(probs) == pytest.approx(1.0)
