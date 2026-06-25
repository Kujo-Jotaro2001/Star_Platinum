from decimal import Decimal
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from bot.features.normalizer import RollingNormalizer
from bot.inference.preprocessor import OnlinePreprocessor
from bot.inference.types import InferenceResult, InferenceStatus
from bot.models.hybrid import HybridSignalModel
from bot.risk import MarketState, PositionState, RiskPolicy, evaluate_risk
from bot.signals import ModelPrediction, SignalPolicy, generate_signal


class InferencePredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor: OnlinePreprocessor,
        symbol: str,
        signal_policy: SignalPolicy,
        risk_policy: RiskPolicy,
        device: str | torch.device = "auto",
        use_ctx: bool = True,
    ) -> None:
        self._device = _resolve_device(device)
        self._model = model.to(self._device)
        self._model.eval()
        self._preprocessor = preprocessor
        self._symbol = symbol
        self._signal_policy = signal_policy
        self._risk_policy = risk_policy
        self._use_ctx = use_ctx

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        normalizer_stats_path: Path,
        symbol: str,
        signal_policy: SignalPolicy,
        risk_policy: RiskPolicy,
        seq_len: int,
        ob_depth: int,
        d_model: int,
        d_ctx: int,
        n_lob_blocks: int,
        in_features_flow: int,
        in_features_ctx: int,
        flow_encoder: str,
        gru_layers: int,
        n_classes: int,
        n_horizons: int,
        dropout: float,
        use_ctx: bool,
        device: str | torch.device = "auto",
        oi_history_window_ms: int = 3_600_000,
        liq_window_ms: int = 60_000,
    ) -> "InferencePredictor":
        model = load_hybrid_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            ob_depth=ob_depth,
            d_model=d_model,
            d_ctx=d_ctx,
            n_lob_blocks=n_lob_blocks,
            in_features_flow=in_features_flow,
            in_features_ctx=in_features_ctx,
            flow_encoder=flow_encoder,
            gru_layers=gru_layers,
            n_classes=n_classes,
            n_horizons=n_horizons,
            dropout=dropout,
            use_ctx=use_ctx,
        )
        normalizer = RollingNormalizer.load(normalizer_stats_path)
        preprocessor = OnlinePreprocessor(
            seq_len=seq_len,
            ob_depth=ob_depth,
            flow_normalizer=normalizer,
            oi_history_window_ms=oi_history_window_ms,
            liq_window_ms=liq_window_ms,
        )
        return cls(
            model=model,
            preprocessor=preprocessor,
            symbol=symbol,
            signal_policy=signal_policy,
            risk_policy=risk_policy,
            device=device,
            use_ctx=use_ctx,
        )

    @property
    def preprocessor(self) -> OnlinePreprocessor:
        return self._preprocessor

    def build_tensors(self) -> tuple[Tensor, Tensor, Tensor]:
        ob, flow, ctx = self._preprocessor.windows()
        return (
            torch.from_numpy(ob).unsqueeze(0).to(self._device),
            torch.from_numpy(flow).unsqueeze(0).to(self._device),
            torch.from_numpy(ctx).unsqueeze(0).to(self._device),
        )

    def predict(
        self,
        timestamp_ms: int,
        position: PositionState,
        market: MarketState,
        proposed_order_notional: Decimal,
        now_ms: int,
    ) -> InferenceResult:
        if not self._preprocessor.is_ready:
            return InferenceResult(
                timestamp_ms=timestamp_ms,
                status=InferenceStatus.NOT_READY,
                probabilities=None,
                signal=None,
                risk_decision=None,
                reason="buffers_not_ready",
            )

        ob, flow, ctx = self.build_tensors()
        with torch.inference_mode():
            logits = self._model(ob, flow, ctx if self._use_ctx else None)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        probabilities = _probabilities_to_tuple(probs)
        signal = generate_signal(
            symbol=self._symbol,
            timestamp_ms=timestamp_ms,
            prediction=ModelPrediction(class_probs=probabilities),
            policy=self._signal_policy,
        )
        risk_decision = evaluate_risk(
            signal=signal,
            position=position,
            market=market,
            policy=self._risk_policy,
            proposed_order_notional=proposed_order_notional,
            now_ms=now_ms,
        )
        status = (
            InferenceStatus.APPROVED
            if risk_decision.approved
            else InferenceStatus.REJECTED
        )
        return InferenceResult(
            timestamp_ms=timestamp_ms,
            status=status,
            probabilities=probabilities,
            signal=signal,
            risk_decision=risk_decision,
            reason=risk_decision.reason,
        )


def load_hybrid_model_from_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device,
    ob_depth: int,
    d_model: int,
    d_ctx: int,
    n_lob_blocks: int,
    in_features_flow: int,
    in_features_ctx: int,
    flow_encoder: str,
    gru_layers: int,
    n_classes: int,
    n_horizons: int,
    dropout: float,
    use_ctx: bool,
) -> HybridSignalModel:
    model = HybridSignalModel(
        ob_depth=ob_depth,
        d_model=d_model,
        d_ctx=d_ctx,
        n_lob_blocks=n_lob_blocks,
        in_features_flow=in_features_flow,
        in_features_ctx=in_features_ctx,
        flow_encoder=flow_encoder,
        gru_layers=gru_layers,
        n_classes=n_classes,
        n_horizons=n_horizons,
        dropout=dropout,
        use_ctx=use_ctx,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=_resolve_device(device),
        weights_only=False,
    )
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(_model_state_dict(state_dict))
    model.eval()
    return model


def _model_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    prefix = "model."
    if any(key.startswith(prefix) for key in state_dict):
        return {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
    return state_dict


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _probabilities_to_tuple(
    probs: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in probs)  # type: ignore[return-value]
