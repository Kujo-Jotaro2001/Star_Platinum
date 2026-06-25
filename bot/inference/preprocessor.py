from dataclasses import dataclass

import numpy as np

from bot.features.context_features import NUM_CONTEXT_FEATURES, ContextBuilder
from bot.features.flow_features import NUM_FLOW_FEATURES, extract_flow_features
from bot.features.normalizer import RollingNormalizer
from bot.features.ob_serializer import OB_NUM_COLS, serialize_ob_snapshot
from bot.inference.buffer import RingBuffer


@dataclass(frozen=True, slots=True)
class OnlineFeatureFrame:
    timestamp_ms: int
    ob: np.ndarray
    flow: np.ndarray
    ctx: np.ndarray


class OnlinePreprocessor:
    """Maintains online feature state and sliding model windows."""

    def __init__(
        self,
        seq_len: int,
        ob_depth: int,
        flow_normalizer: RollingNormalizer,
        oi_history_window_ms: int = 3_600_000,
        liq_window_ms: int = 60_000,
    ) -> None:
        self._ob_depth = ob_depth
        self._ctx_builder = ContextBuilder(
            oi_history_window_ms=oi_history_window_ms,
            liq_window_ms=liq_window_ms,
        )
        self._flow_normalizer = flow_normalizer
        self._ob_buffer = RingBuffer(seq_len, (ob_depth, OB_NUM_COLS), np.float32)
        self._flow_buffer = RingBuffer(seq_len, (NUM_FLOW_FEATURES,), np.float32)
        self._ctx_buffer = RingBuffer(seq_len, (NUM_CONTEXT_FEATURES,), np.float32)

    @property
    def is_ready(self) -> bool:
        return (
            self._ob_buffer.is_full
            and self._flow_buffer.is_full
            and self._ctx_buffer.is_full
        )

    def update_ticker(self, row: dict) -> None:
        self._ctx_builder.update_ticker(row)

    def update_liquidation(self, row: dict) -> None:
        self._ctx_builder.update_liquidation(row)

    def update_long_short_ratio(self, row: dict) -> None:
        self._ctx_builder.update_long_short_ratio(row)

    def append_snapshot(
        self,
        timestamp_ms: int,
        bids: list[list[str]],
        asks: list[list[str]],
        trades: list[dict],
    ) -> OnlineFeatureFrame:
        ob = serialize_ob_snapshot(
            bids=bids,
            asks=asks,
            ob_depth=self._ob_depth,
        )
        flow_raw = extract_flow_features(trades)
        flow = self._flow_normalizer.normalize_only(flow_raw)
        ctx = self._ctx_builder.snapshot(timestamp_ms)

        self._ob_buffer.append(ob)
        self._flow_buffer.append(flow)
        self._ctx_buffer.append(ctx)

        return OnlineFeatureFrame(
            timestamp_ms=timestamp_ms,
            ob=ob,
            flow=flow,
            ctx=ctx,
        )

    def windows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._ob_buffer.window(),
            self._flow_buffer.window(),
            self._ctx_buffer.window(),
        )
