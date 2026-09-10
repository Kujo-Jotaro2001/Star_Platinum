"""Score live predictions against what the market actually did.

Deliberately reuses `compute_labels`: the live prediction is graded by the very
smooth mid-price rule the training set was labelled with, so the number that
comes out is directly comparable to the validation F1 the checkpoint was chosen
on. Any other formula would produce a score that looks like F1 and cannot be
compared to it.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bot.features.labels import compute_labels
from bot.telemetry.types import PREDICTED_STATUSES

N_CLASSES = 3
CLASS_NAMES = ("down", "flat", "up")


@dataclass(frozen=True, slots=True)
class LiveScore:
    n_decisions: int
    n_scored: int
    accuracy: float
    f1_macro: float
    f1_per_class: tuple[float, ...]
    confusion: np.ndarray
    hit_rate_by_confidence: dict[str, float]
    predicted_mix: dict[str, float]
    realised_mix: dict[str, float]


def load_decisions(db_paths: list[Path]) -> list[dict]:
    """Decisions that carry a real prediction, in time order."""
    placeholders = ",".join("?" for _ in PREDICTED_STATUSES)
    rows: list[dict] = []
    for path in db_paths:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        query = (
            "SELECT timestamp_ms, predicted_class, confidence, action, status "
            f"FROM decisions WHERE status IN ({placeholders}) ORDER BY timestamp_ms"
        )
        rows.extend(dict(r) for r in conn.execute(query, tuple(PREDICTED_STATUSES)))
        conn.close()
    return rows


def load_mid_prices(db_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mid price per snapshot, with timestamps and the reset flag."""
    timestamps: list[int] = []
    mids: list[float] = []
    resets: list[bool] = []

    for path in db_paths:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT timestamp_ms, is_reset, bids, asks FROM snapshots ORDER BY timestamp_ms"
        ):
            bids = json.loads(row["bids"])
            asks = json.loads(row["asks"])
            if not bids or not asks:
                continue
            timestamps.append(row["timestamp_ms"])
            mids.append((float(bids[0][0]) + float(asks[0][0])) / 2)
            resets.append(bool(row["is_reset"]))

    return (
        np.array(timestamps, dtype=np.int64),
        np.array(mids, dtype=np.float64),
        np.array(resets, dtype=bool),
    )


def score(
    decisions: list[dict],
    timestamps_ms: np.ndarray,
    mid_prices: np.ndarray,
    is_reset: np.ndarray,
    horizon: int,
    alpha: float,
    confidence_buckets: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> LiveScore:
    """Grade each prediction against the label its own timestamp earns."""
    labels, _, valid = compute_labels(
        mid_prices, is_reset, horizons=[horizon], alpha=alpha
    )
    truth = labels[:, 0]

    predicted: list[int] = []
    realised: list[int] = []
    confidences: list[float] = []

    for row in decisions:
        idx = int(np.searchsorted(timestamps_ms, row["timestamp_ms"]))
        if idx >= len(timestamps_ms) or timestamps_ms[idx] != row["timestamp_ms"]:
            continue
        if not valid[idx]:
            continue
        predicted.append(int(row["predicted_class"]))
        realised.append(int(truth[idx]))
        confidences.append(float(row["confidence"]))

    if not predicted:
        return LiveScore(
            n_decisions=len(decisions), n_scored=0, accuracy=0.0, f1_macro=0.0,
            f1_per_class=(0.0,) * N_CLASSES,
            confusion=np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64),
            hit_rate_by_confidence={}, predicted_mix={}, realised_mix={},
        )

    p = np.array(predicted, dtype=np.int64)
    t = np.array(realised, dtype=np.int64)
    c = np.array(confidences, dtype=np.float64)

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for actual, guess in zip(t, p):
        confusion[actual, guess] += 1

    f1 = tuple(_f1(p, t, cls) for cls in range(N_CLASSES))

    return LiveScore(
        n_decisions=len(decisions),
        n_scored=len(p),
        accuracy=float((p == t).mean()),
        f1_macro=float(sum(f1) / N_CLASSES),
        f1_per_class=f1,
        confusion=confusion,
        hit_rate_by_confidence=_hit_rate_by_confidence(p, t, c, confidence_buckets),
        predicted_mix=_mix(p),
        realised_mix=_mix(t),
    )


def _f1(preds: np.ndarray, truth: np.ndarray, cls: int) -> float:
    tp = int(((preds == cls) & (truth == cls)).sum())
    if tp == 0:
        return 0.0
    fp = int(((preds == cls) & (truth != cls)).sum())
    fn = int(((preds != cls) & (truth == cls)).sum())
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _hit_rate_by_confidence(
    preds: np.ndarray,
    truth: np.ndarray,
    confidence: np.ndarray,
    buckets: tuple[float, ...],
) -> dict[str, float]:
    """Accuracy per confidence band.

    If this does not rise with confidence the threshold is not buying anything,
    and `min_confidence` is filtering on noise.
    """
    out: dict[str, float] = {}
    edges = list(buckets) + [1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.any():
            out[f"{lo:.2f}+"] = float((preds[mask] == truth[mask]).mean())
    return out


def _mix(values: np.ndarray) -> dict[str, float]:
    total = len(values)
    return {
        CLASS_NAMES[cls]: float((values == cls).sum() / total) for cls in range(N_CLASSES)
    }
