from decimal import Decimal
from pathlib import Path

import numpy as np
import torch
from hftbacktest import Recorder
from hftbacktest.stats import LinearAssetRecord

from bot.backtest.run import build_schedule, predict_probabilities, split_bounds
from bot.backtest.strategy import run_strategy
from bot.execution.types import ExitPolicy, InstrumentSpec, OrderType
from bot.models.hybrid import HybridSignalModel
from bot.risk.types import RiskPolicy
from bot.signals.types import SignalPolicy
from bot.training.dataset import LOBDataset
from tests.backtest.conftest import (
    LOT,
    STEP_NS,
    T0_NS,
    TICK,
    flat_quotes,
    make_backtest,
    write_feed,
)

SYMBOL = "BTCUSDT"
N, K, H, C = 400, 4, 3, 3
# The feed covers whole days, the split does not: give the run room past the last
# prediction so the strategy can close out and stop, as it would on real data.
FEED_STEPS = N + 100
SEQ_LEN = 10
NS_PER_MS = 1_000_000


def _features(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A synthetic Phase 2 output on the same 100 ms grid as the feed."""
    return {
        "ob": rng.standard_normal((N, K, 4)).astype(np.float32),
        "flow": rng.standard_normal((N, 9)).astype(np.float32),
        "ctx": rng.standard_normal((N, 17)).astype(np.float32),
        "labels": rng.integers(0, 3, (N, H)).astype(np.int8),
        "flat_mask": np.zeros((N, H), dtype=bool),
        "timestamps_ms": np.array(
            [(T0_NS + k * STEP_NS) // NS_PER_MS for k in range(N)], dtype=np.int64
        ),
    }


def _model() -> HybridSignalModel:
    """A real model, head biased towards "up".

    An untrained head lands on the same argmax for every window; when that class
    is flat the run trades nothing and the test proves nothing. The bias
    guarantees directional signals while the encoders still shape the output.
    """
    torch.manual_seed(0)
    model = HybridSignalModel(
        ob_depth=K, d_model=16, d_ctx=8, n_lob_blocks=2,
        in_features_flow=9, in_features_ctx=17, flow_encoder="gru",
        gru_layers=1, n_classes=C, n_horizons=H, dropout=0.0, use_ctx=True,
    )
    with torch.no_grad():
        model.head.bias.view(H, C)[:, 2] += 5.0
    model.eval()
    return model


def _run(feed: Path, features: dict[str, np.ndarray], lo: int, hi: int, min_confidence: float):
    dataset = LOBDataset(
        features["ob"][lo:hi], features["flow"][lo:hi], features["ctx"][lo:hi],
        features["labels"][lo:hi], features["flat_mask"][lo:hi], SEQ_LEN, True,
    )
    schedule = build_schedule(
        predict_probabilities(_model(), dataset, 64, torch.device("cpu"), True),
        features["timestamps_ms"][lo:hi],
        SEQ_LEN,
    )

    hbt = make_backtest(feed)
    recorder = Recorder(hbt.num_assets, 100_000)
    try:
        stats = run_strategy(
            hbt=hbt,
            schedule=schedule,
            symbol=SYMBOL,
            signal_policy=SignalPolicy(
                horizon_index=1, min_confidence=min_confidence, allow_short=True
            ),
            risk_policy=RiskPolicy(
                max_position_notional=Decimal("100"),
                max_leverage=Decimal("1000"),
                max_spread_bps=Decimal("50"),
                stale_data_ms=10_000,
                cooldown_after_trade_ms=0,
                min_confidence=min_confidence,
                allow_short=True,
            ),
            exit_policy=ExitPolicy(500, Decimal("15"), Decimal("10")),
            spec=InstrumentSpec(
                qty_step=Decimal(str(LOT)),
                price_tick=Decimal(str(TICK)),
                min_order_qty=Decimal(str(LOT)),
            ),
            exit_order_type=OrderType.MARKET,
            order_notional=Decimal("100"),
            initial_equity=Decimal("10000"),
            elapse_ns=STEP_NS,
            order_timeout_ns=1_000_000_000,
            recorder=recorder.recorder,
        )
        final_position = hbt.position(0)
        num_trades = int(hbt.state_values(0).num_trades)
    finally:
        hbt.close()
    return schedule, stats, recorder, final_position, num_trades


class TestEndToEnd:
    def test_model_output_drives_a_complete_hftbacktest_run(self, feed_dir: Path) -> None:
        features = _features(np.random.default_rng(0))
        feed = write_feed(feed_dir / "feed.npz", flat_quotes(FEED_STEPS), trade_qty=3.0)
        lo, hi = split_bounds(N, "test", 0.8, 0.1)

        schedule, stats, _, final_position, num_trades = _run(
            feed, features, lo, hi, min_confidence=0.30
        )

        assert len(schedule.probabilities) == (hi - lo) - SEQ_LEN + 1
        assert stats.signals_approved > 0
        assert stats.orders_filled > 0
        assert num_trades > 0
        assert final_position == 0

    def test_schedule_stays_inside_the_split(self, feed_dir: Path) -> None:
        features = _features(np.random.default_rng(1))
        feed = write_feed(feed_dir / "feed.npz", flat_quotes(FEED_STEPS), trade_qty=3.0)
        lo, hi = split_bounds(N, "test", 0.8, 0.1)

        schedule, stats, _, _, _ = _run(feed, features, lo, hi, min_confidence=0.30)

        split_timestamps = features["timestamps_ms"][lo:hi] * NS_PER_MS
        assert schedule.timestamps_ns[0] >= split_timestamps[0]
        assert schedule.timestamps_ns[-1] == split_timestamps[-1]

        # Decisions are confined to the split even though the feed is longer: the
        # loop still elapses through the earlier steps to build the book, but it
        # takes no signal there and stops once past the last prediction.
        assert stats.signals_actionable <= len(schedule.probabilities)
        assert stats.decisions < FEED_STEPS

    def test_stats_are_computable_from_the_recording(self, feed_dir: Path) -> None:
        features = _features(np.random.default_rng(2))
        feed = write_feed(feed_dir / "feed.npz", flat_quotes(FEED_STEPS), trade_qty=3.0)
        lo, hi = split_bounds(N, "test", 0.8, 0.1)

        _, _, recorder, _, _ = _run(feed, features, lo, hi, min_confidence=0.30)

        record = LinearAssetRecord(recorder.get(0))
        summary = record.stats()
        assert summary is not None

    def test_an_unreachable_confidence_threshold_trades_nothing(
        self, feed_dir: Path
    ) -> None:
        features = _features(np.random.default_rng(3))
        feed = write_feed(feed_dir / "feed.npz", flat_quotes(FEED_STEPS), trade_qty=3.0)
        lo, hi = split_bounds(N, "test", 0.8, 0.1)

        _, stats, _, _, num_trades = _run(feed, features, lo, hi, min_confidence=0.999)

        assert stats.signals_actionable == 0
        assert stats.orders_submitted == 0
        assert num_trades == 0
