import bisect
from pathlib import Path

import hydra
import numpy as np
import structlog
from omegaconf import DictConfig

from bot.features.context_features import NUM_CONTEXT_FEATURES, ContextBuilder
from bot.features.flow_features import NUM_FLOW_FEATURES, extract_flow_features
from bot.features.labels import compute_labels
from bot.features.normalizer import RollingNormalizer
from bot.features.ob_serializer import OB_NUM_COLS, serialize_ob_snapshot
from bot.features.replay import (
    find_db_files,
    load_liquidations,
    load_long_short_ratio,
    load_snapshots,
    load_ticker_context,
    load_trades,
)

logger = structlog.get_logger()


def _bucket_trades(
    trades: list[dict],
    snapshot_timestamps: list[int],
    bucket_ms: int,
) -> list[list[dict]]:
    """Assign trades into buckets aligned to snapshot timestamps.

    Bucket at snapshot T contains trades where T-bucket_ms <= ts < T.
    Returns one list of trades per snapshot, in snapshot order.
    """
    trade_ts = [t["timestamp_ms"] for t in trades]
    buckets: list[list[dict]] = []

    for snap_ts in snapshot_timestamps:
        lo = snap_ts - bucket_ms
        hi = snap_ts
        i_start = bisect.bisect_left(trade_ts, lo)
        i_end = bisect.bisect_left(trade_ts, hi)
        buckets.append(trades[i_start:i_end])

    return buckets


def _advance_context(
    ctx: ContextBuilder,
    ticker_rows: list[dict],
    liq_rows: list[dict],
    ls_rows: list[dict],
    ticker_idx: int,
    liq_idx: int,
    ls_idx: int,
    up_to_ts: int,
) -> tuple[int, int, int]:
    """Consume context rows up to (and including) up_to_ts."""
    while ticker_idx < len(ticker_rows) and ticker_rows[ticker_idx]["timestamp_ms"] <= up_to_ts:
        ctx.update_ticker(ticker_rows[ticker_idx])
        ticker_idx += 1
    while liq_idx < len(liq_rows) and liq_rows[liq_idx]["timestamp_ms"] <= up_to_ts:
        ctx.update_liquidation(liq_rows[liq_idx])
        liq_idx += 1
    while ls_idx < len(ls_rows) and ls_rows[ls_idx]["timestamp_ms"] <= up_to_ts:
        ctx.update_long_short_ratio(ls_rows[ls_idx])
        ls_idx += 1
    return ticker_idx, liq_idx, ls_idx


def run_pipeline(
    data_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
    ob_depth: int = 50,
    bucket_ms: int = 100,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    normalizer_window: int = 500,
    label_alpha: float = 0.001,
    label_horizons: list[int] | None = None,
    oi_history_window_ms: int = 3_600_000,
    liq_window_ms: int = 60_000,
    output_dir: Path | None = None,
) -> None:
    """Run the full offline feature extraction pipeline.

    Reads daily SQLite files, extracts features, computes labels,
    normalizes, and saves numpy arrays.
    """
    if label_horizons is None:
        label_horizons = [1, 5, 10]
    if output_dir is None:
        output_dir = Path("data/features") / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find and load data
    db_paths = find_db_files(data_dir, symbol, start_date, end_date)
    logger.info("pipeline.loading", db_count=len(db_paths))

    snapshots = load_snapshots(db_paths)
    trades = load_trades(db_paths)
    ticker_rows = load_ticker_context(db_paths)
    liq_rows = load_liquidations(db_paths)
    ls_rows = load_long_short_ratio(db_paths)

    n = len(snapshots)
    logger.info("pipeline.loaded", snapshots=n, trades=len(trades),
                ticker=len(ticker_rows), liquidations=len(liq_rows),
                ls_ratio=len(ls_rows))

    # 2. Pre-allocate arrays
    ob_raw = np.zeros((n, ob_depth, OB_NUM_COLS), dtype=np.float32)
    flow_raw = np.zeros((n, NUM_FLOW_FEATURES), dtype=np.float32)
    ctx_features = np.zeros((n, NUM_CONTEXT_FEATURES), dtype=np.float32)
    mid_prices = np.zeros(n, dtype=np.float64)
    timestamps_ms = np.zeros(n, dtype=np.int64)
    is_reset = np.zeros(n, dtype=bool)

    # 3. Extract raw features
    snapshot_timestamps = [s["timestamp_ms"] for s in snapshots]
    trade_buckets = _bucket_trades(trades, snapshot_timestamps, bucket_ms)
    ctx_builder = ContextBuilder(
        oi_history_window_ms=oi_history_window_ms,
        liq_window_ms=liq_window_ms,
    )

    ticker_idx, liq_idx, ls_idx = 0, 0, 0

    for i, snap in enumerate(snapshots):
        ts = snap["timestamp_ms"]
        timestamps_ms[i] = ts
        is_reset[i] = bool(snap["is_reset"])

        ticker_idx, liq_idx, ls_idx = _advance_context(
            ctx_builder, ticker_rows, liq_rows, ls_rows,
            ticker_idx, liq_idx, ls_idx, ts,
        )

        bids = snap["bids"]
        asks = snap["asks"]
        ob_raw[i] = serialize_ob_snapshot(bids, asks, ob_depth)
        best_bid = float(bids[0][0]) if bids else 0.0
        best_ask = float(asks[0][0]) if asks else 0.0
        mid_prices[i] = (best_bid + best_ask) / 2

        flow_raw[i] = extract_flow_features(trade_buckets[i])
        ctx_features[i] = ctx_builder.snapshot(ts)

    logger.info("pipeline.features_extracted")

    # 4. Compute labels from mid_price series
    labels, flat_mask, valid_mask = compute_labels(
        mid_prices, is_reset, horizons=label_horizons, alpha=label_alpha
    )

    # 5. Normalize flow features only (LOB tensor is self-normalizing)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    flow_norm = RollingNormalizer(NUM_FLOW_FEATURES, window=normalizer_window)
    flow_out = np.zeros_like(flow_raw)

    for i in range(train_end):
        flow_out[i] = flow_norm.update_and_normalize(flow_raw[i])

    for i in range(train_end, n):
        flow_out[i] = flow_norm.normalize_only(flow_raw[i])

    logger.info("pipeline.normalized", train=train_end, val=val_end - train_end, test=n - val_end)

    # 6. Save outputs
    np.save(output_dir / "ob_raw.npy", ob_raw)
    np.save(output_dir / "flow_features.npy", flow_out)
    np.save(output_dir / "ctx_features.npy", ctx_features)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "flat_mask.npy", flat_mask)
    np.save(output_dir / "timestamps_ms.npy", timestamps_ms)
    np.save(output_dir / "is_reset.npy", is_reset)
    flow_norm.save(output_dir / "normalizer_stats.npz")

    logger.info("pipeline.done", output_dir=str(output_dir), n=n)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_pipeline(
        data_dir=Path(cfg.ingestion.data_dir),
        symbol=cfg.ingestion.symbol,
        ob_depth=cfg.ingestion.lob_depth,
        start_date=cfg.features.start_date,
        end_date=cfg.features.end_date,
        bucket_ms=cfg.features.bucket_ms,
        train_ratio=cfg.features.train_ratio,
        val_ratio=cfg.features.val_ratio,
        normalizer_window=cfg.features.normalizer_window,
        label_alpha=cfg.features.label_alpha,
        label_horizons=list(cfg.features.label_horizons),
        oi_history_window_ms=cfg.features.oi_history_window_ms,
        liq_window_ms=cfg.features.liq_window_ms,
    )


if __name__ == "__main__":
    main()
