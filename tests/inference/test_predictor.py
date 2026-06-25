import sys
from decimal import Decimal
from pathlib import Path

import pytest
import torch

from bot.features.normalizer import RollingNormalizer
from bot.inference import (
    InferencePredictor,
    InferenceStatus,
    OnlinePreprocessor,
    load_hybrid_model_from_checkpoint,
)
from bot.models.hybrid import HybridSignalModel
from bot.risk import MarketState, PositionState, RiskPolicy
from bot.signals import SignalPolicy

SYMBOL = "BTCUSDT"
NOW_MS = 1_800_000_000_000


class RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.called = False
        self.shapes: tuple[torch.Size, torch.Size, torch.Size | None] | None = None

    def forward(
        self,
        ob: torch.Tensor,
        flow: torch.Tensor,
        ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.called = True
        self.shapes = (ob.shape, flow.shape, ctx.shape if ctx is not None else None)
        return torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]],
            dtype=torch.float32,
            device=ob.device,
        )


def _predictor(seq_len: int = 3) -> tuple[InferencePredictor, RecordingModel]:
    model = RecordingModel()
    preprocessor = OnlinePreprocessor(
        seq_len=seq_len,
        ob_depth=2,
        flow_normalizer=RollingNormalizer(num_features=9, window=5),
    )
    predictor = InferencePredictor(
        model=model,
        preprocessor=preprocessor,
        symbol=SYMBOL,
        signal_policy=SignalPolicy(
            horizon_index=1,
            min_confidence=0.55,
            allow_short=False,
        ),
        risk_policy=RiskPolicy(
            max_position_notional=Decimal("1000"),
            max_leverage=Decimal("2"),
            max_spread_bps=Decimal("10"),
            stale_data_ms=1000,
            cooldown_after_trade_ms=5000,
            min_confidence=0.55,
            allow_short=False,
        ),
        device="cpu",
        use_ctx=True,
    )
    return predictor, model


def _position() -> PositionState:
    return PositionState(symbol=SYMBOL, quantity=Decimal("0"))


def _market() -> MarketState:
    return MarketState(
        symbol=SYMBOL,
        timestamp_ms=NOW_MS,
        best_bid=Decimal("9999"),
        best_ask=Decimal("10001"),
        account_equity=Decimal("1000"),
    )


def _append_ready(preprocessor: OnlinePreprocessor, seq_len: int = 3) -> None:
    for i in range(seq_len):
        preprocessor.append_snapshot(
            timestamp_ms=NOW_MS + i * 100,
            bids=[["100.0", "1.0"], ["99.9", "2.0"]],
            asks=[["100.1", "1.5"], ["100.2", "2.5"]],
            trades=[{"side": "Buy", "price": "100.0", "qty": "0.5"}],
        )


class TestInferencePredictor:
    def test_returns_not_ready_before_warmup(self) -> None:
        predictor, model = _predictor()

        result = predictor.predict(
            timestamp_ms=NOW_MS,
            position=_position(),
            market=_market(),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert result.status == InferenceStatus.NOT_READY
        assert result.reason == "buffers_not_ready"
        assert result.signal is None
        assert model.called is False

    def test_builds_tensors_with_correct_shapes(self) -> None:
        predictor, _ = _predictor(seq_len=3)
        _append_ready(predictor.preprocessor, seq_len=3)

        ob, flow, ctx = predictor.build_tensors()

        assert ob.shape == (1, 3, 2, 4)
        assert flow.shape == (1, 3, 9)
        assert ctx.shape == (1, 3, 17)

    def test_uses_softmax_probabilities(self) -> None:
        predictor, model = _predictor(seq_len=3)
        _append_ready(predictor.preprocessor, seq_len=3)

        result = predictor.predict(
            timestamp_ms=NOW_MS,
            position=_position(),
            market=_market(),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert result.status == InferenceStatus.APPROVED
        assert result.probabilities is not None
        assert result.probabilities[1][2] == pytest.approx(0.909443, rel=1e-5)
        assert sum(result.probabilities[1]) == pytest.approx(1.0)
        assert model.shapes == (
            torch.Size([1, 3, 2, 4]),
            torch.Size([1, 3, 9]),
            torch.Size([1, 3, 17]),
        )

    def test_does_not_call_execution(self) -> None:
        sys.modules.pop("bot.execution", None)
        predictor, _ = _predictor(seq_len=3)
        _append_ready(predictor.preprocessor, seq_len=3)

        predictor.predict(
            timestamp_ms=NOW_MS,
            position=_position(),
            market=_market(),
            proposed_order_notional=Decimal("50"),
            now_ms=NOW_MS,
        )

        assert "bot.execution" not in sys.modules

    def test_loads_lightning_style_hybrid_checkpoint(self, tmp_path: Path) -> None:
        model = HybridSignalModel(
            ob_depth=2,
            d_model=6,
            d_ctx=4,
            n_lob_blocks=1,
            in_features_flow=9,
            in_features_ctx=17,
            flow_encoder="gru",
            gru_layers=1,
            n_classes=3,
            n_horizons=2,
            dropout=0.0,
            use_ctx=True,
        )
        checkpoint_path = tmp_path / "model.ckpt"
        torch.save(
            {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}},
            checkpoint_path,
        )

        loaded = load_hybrid_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device="cpu",
            ob_depth=2,
            d_model=6,
            d_ctx=4,
            n_lob_blocks=1,
            in_features_flow=9,
            in_features_ctx=17,
            flow_encoder="gru",
            gru_layers=1,
            n_classes=3,
            n_horizons=2,
            dropout=0.0,
            use_ctx=True,
        )

        assert isinstance(loaded, HybridSignalModel)
        assert loaded.training is False
