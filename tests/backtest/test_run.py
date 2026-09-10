from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from bot.backtest.run import (
    build_asset,
    build_schedule,
    covered_dates,
    predict_probabilities,
    split_bounds,
)
from bot.training.dataset import LOBDataset

N, K, H, C = 40, 4, 3, 3
SEQ_LEN = 5
NS_PER_MS = 1_000_000


class ConstantModel(torch.nn.Module):
    def forward(
        self,
        ob: torch.Tensor,
        flow: torch.Tensor,
        ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.zeros(ob.shape[0], H, C, dtype=torch.float32)


def _dataset() -> LOBDataset:
    return LOBDataset(
        np.zeros((N, K, 4), dtype=np.float32),
        np.zeros((N, 9), dtype=np.float32),
        np.zeros((N, 17), dtype=np.float32),
        np.zeros((N, H), dtype=np.int8),
        np.zeros((N, H), dtype=bool),
        SEQ_LEN,
        use_ctx=True,
    )


def _cfg(**backtest) -> OmegaConf:
    defaults = {
        "maker_fee_rate": -0.0001,
        "taker_fee_rate": 0.00055,
        "queue_model": "risk_adverse",
        "queue_power": 3.0,
        "latency_model": "constant",
        "entry_latency_ns": 10_000_000,
        "response_latency_ns": 10_000_000,
    }
    defaults.update(backtest)
    return OmegaConf.create(
        {
            "backtest": defaults,
            "execution": {"price_tick": 0.1, "qty_step": 0.001},
        }
    )


class TestSplitBounds:
    def test_chronological_and_contiguous(self) -> None:
        train = split_bounds(1000, "train", 0.8, 0.1)
        val = split_bounds(1000, "val", 0.8, 0.1)
        test = split_bounds(1000, "test", 0.8, 0.1)

        assert train == (0, 800)
        assert val == (800, 900)
        assert test == (900, 1000)
        assert train[1] == val[0] and val[1] == test[0]

    def test_matches_the_training_split(self) -> None:
        n, train_ratio, val_ratio = 12_345, 0.8, 0.1
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        assert split_bounds(n, "train", train_ratio, val_ratio) == (0, train_end)
        assert split_bounds(n, "val", train_ratio, val_ratio) == (train_end, val_end)
        assert split_bounds(n, "test", train_ratio, val_ratio) == (val_end, n)

    def test_unknown_split_rejected(self) -> None:
        with pytest.raises(ValueError):
            split_bounds(100, "holdout", 0.8, 0.1)


class TestPredictProbabilities:
    def test_one_row_per_window(self) -> None:
        dataset = _dataset()
        probs = predict_probabilities(
            ConstantModel(), dataset, 8, torch.device("cpu"), True
        )
        assert probs.shape == (len(dataset), H, C)
        assert len(dataset) == N - SEQ_LEN + 1

    def test_rows_are_probability_distributions(self) -> None:
        probs = predict_probabilities(
            ConstantModel(), _dataset(), 8, torch.device("cpu"), True
        )
        assert np.allclose(probs.sum(axis=-1), 1.0)
        assert (probs >= 0).all()


class TestBuildSchedule:
    def _timestamps(self) -> np.ndarray:
        return np.array([1_700_000_000_000 + i * 100 for i in range(N)], dtype=np.int64)

    def test_each_window_maps_onto_its_last_snapshot(self) -> None:
        timestamps = self._timestamps()
        probs = np.zeros((N - SEQ_LEN + 1, H, C), dtype=np.float32)
        schedule = build_schedule(probs, timestamps, SEQ_LEN)

        assert len(schedule.timestamps_ns) == len(probs)
        # window 0 spans [0, seq_len) and is scored at index seq_len - 1
        assert schedule.timestamps_ns[0] == timestamps[SEQ_LEN - 1] * NS_PER_MS
        assert schedule.timestamps_ns[-1] == timestamps[-1] * NS_PER_MS

    def test_nanoseconds_not_milliseconds(self) -> None:
        timestamps = self._timestamps()
        probs = np.zeros((N - SEQ_LEN + 1, H, C), dtype=np.float32)
        schedule = build_schedule(probs, timestamps, SEQ_LEN)
        assert schedule.timestamps_ns.dtype == np.int64
        assert schedule.timestamps_ns[0] > 1_600_000_000_000_000_000


class TestCoveredDates:
    def test_single_day(self) -> None:
        # 2026-03-01T00:00:00Z and a second later
        base = 1772323200_000
        start, end = covered_dates(np.array([base, base + 1000], dtype=np.int64))
        assert start == end == date(2026, 3, 1)

    def test_spans_midnight(self) -> None:
        base = 1772323200_000
        start, end = covered_dates(
            np.array([base, base + 36 * 3600 * 1000], dtype=np.int64)
        )
        assert start == date(2026, 3, 1)
        assert end == date(2026, 3, 2)


class TestBuildAsset:
    def test_builds_with_a_constant_latency_model(self, tmp_path: Path) -> None:
        feed = tmp_path / "f.npz"
        feed.touch()
        # construction must not touch the file; loading happens in the backtest
        assert build_asset([feed], None, [], _cfg()) is not None

    @pytest.mark.parametrize(
        "queue_model",
        ["risk_adverse", "log_prob", "power_prob", "power_prob2", "power_prob3"],
    )
    def test_every_documented_queue_model_is_wired(
        self, tmp_path: Path, queue_model: str
    ) -> None:
        feed = tmp_path / "f.npz"
        feed.touch()
        assert build_asset([feed], None, [], _cfg(queue_model=queue_model)) is not None

    def test_unknown_queue_model_rejected(self, tmp_path: Path) -> None:
        feed = tmp_path / "f.npz"
        feed.touch()
        with pytest.raises(ValueError, match="queue_model"):
            build_asset([feed], None, [], _cfg(queue_model="nope"))

    def test_unknown_latency_model_rejected(self, tmp_path: Path) -> None:
        feed = tmp_path / "f.npz"
        feed.touch()
        with pytest.raises(ValueError, match="latency_model"):
            build_asset([feed], None, [], _cfg(latency_model="nope"))
