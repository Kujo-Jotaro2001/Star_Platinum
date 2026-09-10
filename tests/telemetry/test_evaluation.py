import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from bot.data.storage import DECISIONS_DDL, SNAPSHOTS_DDL
from bot.telemetry.evaluation import load_decisions, load_mid_prices, score
from bot.telemetry.types import APPROVED, BLOCKED, HOLDING, NOT_READY

T0 = 1_800_000_000_000
STEP_MS = 100
HORIZON = 5
ALPHA = 0.001


def _db(tmp_path: Path, mids: list[float], decisions: list[tuple[int, int, float, str]]) -> Path:
    """A daily DB holding a mid-price path and the decisions taken along it."""
    path = tmp_path / "BTCUSDT_2026-03-01.db"
    conn = sqlite3.connect(path)
    conn.execute(SNAPSHOTS_DDL)
    conn.execute(DECISIONS_DDL)

    for i, mid in enumerate(mids):
        conn.execute(
            "INSERT INTO snapshots (timestamp_ms, is_reset, bids, asks) VALUES (?, ?, ?, ?)",
            (
                T0 + i * STEP_MS,
                int(i == 0),
                json.dumps([[str(mid - 0.5), "1"]]),
                json.dumps([[str(mid + 0.5), "1"]]),
            ),
        )

    for ts, predicted, confidence, status in decisions:
        conn.execute(
            """INSERT INTO decisions (
                timestamp_ms, status, action, horizon_index, p_down, p_flat, p_up,
                predicted_class, confidence, book_age_ms, is_reset, inference_us,
                flow_z_max, flow_z_abs_mean, ctx_ticker_age_ms, ctx_liq_age_ms,
                ctx_ls_age_ms, class_changed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, status, "BUY", 1, 0.1, 0.1, 0.8, predicted, confidence,
             0, 0, 0, 0.0, 0.0, 0, 0, 0, 0),
        )
    conn.commit()
    conn.close()
    return path


def _rising(n: int = 60) -> list[float]:
    return [30000.0 * (1 + 0.002 * i) for i in range(n)]


def _flat(n: int = 60) -> list[float]:
    return [30000.0] * n


class TestLoading:
    def test_only_decisions_carrying_a_prediction_are_loaded(self, tmp_path: Path) -> None:
        path = _db(
            tmp_path, _flat(),
            [
                (T0 + 1000, 2, 0.9, APPROVED),
                (T0 + 1100, 1, 0.5, HOLDING),
                (T0 + 1200, 0, 0.7, NOT_READY),
                (T0 + 1300, 2, 0.8, BLOCKED),
            ],
        )
        rows = load_decisions([path])
        assert len(rows) == 2
        assert {r["status"] for r in rows} == {APPROVED, BLOCKED}

    def test_mid_prices_come_from_the_book(self, tmp_path: Path) -> None:
        path = _db(tmp_path, [30000.0, 30010.0], [])
        timestamps, mids, resets = load_mid_prices([path])
        assert list(mids) == [30000.0, 30010.0]
        assert timestamps[0] == T0
        assert resets[0]  # first snapshot of the day is a reset


class TestScore:
    def _score(self, mids: list[float], decisions: list[tuple[int, int, float, str]]):
        timestamps = np.array([T0 + i * STEP_MS for i in range(len(mids))], dtype=np.int64)
        is_reset = np.zeros(len(mids), dtype=bool)
        rows = [
            {"timestamp_ms": ts, "predicted_class": c, "confidence": conf, "status": st}
            for ts, c, conf, st in decisions
        ]
        return score(
            rows, timestamps, np.array(mids), is_reset, horizon=HORIZON, alpha=ALPHA
        )

    def test_a_correct_call_on_a_rising_market_scores(self) -> None:
        mids = _rising()
        # index 30 sits well clear of both reset zones
        result = self._score(mids, [(T0 + 30 * STEP_MS, 2, 0.9, APPROVED)])
        assert result.n_scored == 1
        assert result.accuracy == 1.0
        assert result.realised_mix["up"] == 1.0

    def test_a_wrong_call_scores_zero(self) -> None:
        result = self._score(_rising(), [(T0 + 30 * STEP_MS, 0, 0.9, APPROVED)])
        assert result.n_scored == 1
        assert result.accuracy == 0.0

    def test_flat_market_realises_flat(self) -> None:
        result = self._score(_flat(), [(T0 + 30 * STEP_MS, 1, 0.9, APPROVED)])
        assert result.realised_mix["flat"] == 1.0
        assert result.accuracy == 1.0

    def test_decisions_without_a_matching_snapshot_are_skipped(self) -> None:
        result = self._score(_rising(), [(T0 + 7, 2, 0.9, APPROVED)])
        assert result.n_decisions == 1
        assert result.n_scored == 0

    def test_reset_zones_are_excluded(self) -> None:
        mids = _rising()
        timestamps = np.array([T0 + i * STEP_MS for i in range(len(mids))], dtype=np.int64)
        is_reset = np.zeros(len(mids), dtype=bool)
        is_reset[30] = True
        rows = [{"timestamp_ms": T0 + 30 * STEP_MS, "predicted_class": 2, "confidence": 0.9}]
        result = score(rows, timestamps, np.array(mids), is_reset, HORIZON, ALPHA)
        assert result.n_scored == 0

    def test_empty_input_is_a_zero_score_not_a_crash(self) -> None:
        result = self._score(_rising(), [])
        assert result.n_scored == 0
        assert result.f1_macro == 0.0
        assert result.confusion.shape == (3, 3)

    def test_confusion_matrix_is_actual_by_predicted(self) -> None:
        mids = _rising()
        decisions = [
            (T0 + 20 * STEP_MS, 0, 0.9, APPROVED),  # says down, market rose
            (T0 + 30 * STEP_MS, 2, 0.9, APPROVED),  # says up, market rose
        ]
        result = self._score(mids, decisions)
        assert result.confusion[2, 0] == 1
        assert result.confusion[2, 2] == 1

    def test_f1_is_macro_over_the_three_classes(self) -> None:
        result = self._score(_rising(), [(T0 + 30 * STEP_MS, 2, 0.9, APPROVED)])
        assert result.f1_macro == pytest.approx(sum(result.f1_per_class) / 3)


class TestHitRateByConfidence:
    def test_bands_are_reported_separately(self) -> None:
        mids = _rising()
        timestamps = np.array([T0 + i * STEP_MS for i in range(len(mids))], dtype=np.int64)
        is_reset = np.zeros(len(mids), dtype=bool)
        rows = [
            # low confidence, wrong
            {"timestamp_ms": T0 + 20 * STEP_MS, "predicted_class": 0, "confidence": 0.55},
            # high confidence, right
            {"timestamp_ms": T0 + 30 * STEP_MS, "predicted_class": 2, "confidence": 0.95},
        ]
        result = score(rows, timestamps, np.array(mids), is_reset, HORIZON, ALPHA)
        assert result.hit_rate_by_confidence["0.50+"] == 0.0
        assert result.hit_rate_by_confidence["0.90+"] == 1.0

    def test_empty_bands_are_omitted(self) -> None:
        mids = _rising()
        timestamps = np.array([T0 + i * STEP_MS for i in range(len(mids))], dtype=np.int64)
        rows = [{"timestamp_ms": T0 + 30 * STEP_MS, "predicted_class": 2, "confidence": 0.95}]
        result = score(
            rows, timestamps, np.array(mids), np.zeros(len(mids), dtype=bool), HORIZON, ALPHA
        )
        assert list(result.hit_rate_by_confidence) == ["0.90+"]


class TestMix:
    def test_predicted_and_realised_mixes_are_comparable(self) -> None:
        mids = _rising()
        timestamps = np.array([T0 + i * STEP_MS for i in range(len(mids))], dtype=np.int64)
        rows = [
            {"timestamp_ms": T0 + i * STEP_MS, "predicted_class": 1, "confidence": 0.6}
            for i in (20, 25, 30, 35)
        ]
        result = score(
            rows, timestamps, np.array(mids), np.zeros(len(mids), dtype=bool), HORIZON, ALPHA
        )
        # the model says flat everywhere while the market only went up
        assert result.predicted_mix["flat"] == 1.0
        assert result.realised_mix["up"] == 1.0
        assert result.accuracy == 0.0
