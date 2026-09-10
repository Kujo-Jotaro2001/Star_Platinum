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
    count_snapshots,
    find_db_files,
    iter_snapshots,
    load_liquidations,
    load_long_short_ratio,
    load_snapshot_timestamps,
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


def _memmap(path: Path, shape: tuple[int, ...], dtype) -> np.ndarray:
    """Open an .npy on disk for writing without holding it in memory.

    The LOB tensor alone is 800 bytes per snapshot, so two weeks of data is
    ~10 GB — more than fits alongside everything else. Writing straight through
    to the file keeps the pipeline's footprint to one day at a time.
    """
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


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
    """Run the offline feature extraction pipeline, one day at a time.

    Days are processed in order and written straight into memory-mapped outputs;
    context state carries across day boundaries, which is why the days cannot be
    processed independently.
    """
    if label_horizons is None:
        label_horizons = [1, 5, 10]
    if output_dir is None:
        output_dir = Path("data/features") / symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    db_paths = find_db_files(data_dir, symbol, start_date, end_date)
    n = count_snapshots(db_paths)
    logger.info("pipeline.loading", db_count=len(db_paths), snapshots=n)
    if n == 0:
        raise ValueError(f"no snapshots found for {symbol} in {start_date}..{end_date}")

    ob_raw = _memmap(output_dir / "ob_raw.npy", (n, ob_depth, OB_NUM_COLS), np.float32)
    flow_raw = _memmap(output_dir / "flow_features.npy", (n, NUM_FLOW_FEATURES), np.float32)
    ctx_features = _memmap(output_dir / "ctx_features.npy", (n, NUM_CONTEXT_FEATURES), np.float32)
    top_of_book = _memmap(output_dir / "top_of_book.npy", (n, 2), np.float64)
    mid_prices = np.zeros(n, dtype=np.float64)
    timestamps_ms = np.zeros(n, dtype=np.int64)
    is_reset = np.zeros(n, dtype=bool)

    ctx_builder = ContextBuilder(
        oi_history_window_ms=oi_history_window_ms,
        liq_window_ms=liq_window_ms,
    )

    offset = 0
    for db_path in db_paths:
        day_ts = load_snapshot_timestamps(db_path)
        trade_buckets = _bucket_trades(load_trades([db_path]), day_ts, bucket_ms)

        ticker_rows = load_ticker_context([db_path])
        liq_rows = load_liquidations([db_path])
        ls_rows = load_long_short_ratio([db_path])
        ticker_idx = liq_idx = ls_idx = 0

        for i, snap in enumerate(iter_snapshots(db_path)):
            j = offset + i
            ts = snap["timestamp_ms"]
            timestamps_ms[j] = ts
            is_reset[j] = bool(snap["is_reset"])

            ticker_idx, liq_idx, ls_idx = _advance_context(
                ctx_builder, ticker_rows, liq_rows, ls_rows,
                ticker_idx, liq_idx, ls_idx, ts,
            )

            bids = snap["bids"]
            asks = snap["asks"]
            ob_raw[j] = serialize_ob_snapshot(bids, asks, ob_depth)
            best_bid = float(bids[0][0]) if bids else 0.0
            best_ask = float(asks[0][0]) if asks else 0.0
            top_of_book[j] = (best_bid, best_ask)
            mid_prices[j] = (best_bid + best_ask) / 2

            flow_raw[j] = extract_flow_features(trade_buckets[i])
            ctx_features[j] = ctx_builder.snapshot(ts)

        offset += len(day_ts)
        logger.info("pipeline.day_done", db=db_path.name, snapshots=len(day_ts), total=offset)

    logger.info("pipeline.features_extracted", n=offset)

    # Saved before the remaining steps: extraction is the expensive part of the
    # run, and losing it to a failure further down costs half an hour.
    np.save(output_dir / "timestamps_ms.npy", timestamps_ms)
    np.save(output_dir / "is_reset.npy", is_reset)
    ob_raw.flush()
    top_of_book.flush()

    labels, flat_mask, valid_mask = compute_labels(
        mid_prices, is_reset, horizons=label_horizons, alpha=label_alpha
    )

    # Rows outside valid_mask — reset zones and the edges where a horizon has no
    # room — carry label 0 by default, which the loss would otherwise train on as
    # a confident "down". They are given the flat index so the tensor holds a real
    # class, and `valid_mask` is what keeps them out of the loss. `flat_mask` is
    # deliberately left alone: it means "this label is flat", which is a class to
    # be learned, not a row to be skipped.
    labels[~valid_mask] = 1
    logger.info(
        "pipeline.labels",
        valid=int(valid_mask.sum()),
        excluded=int((~valid_mask).sum()),
    )

    # Normalize flow features in place: fit on the train split, freeze for the rest.
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    flow_norm = RollingNormalizer(NUM_FLOW_FEATURES, window=normalizer_window)
    for i in range(train_end):
        flow_raw[i] = flow_norm.update_and_normalize(flow_raw[i])
    for i in range(train_end, n):
        flow_raw[i] = flow_norm.normalize_only(flow_raw[i])

    logger.info("pipeline.normalized", train=train_end, val=val_end - train_end, test=n - val_end)

    flow_raw.flush()
    ctx_features.flush()
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "flat_mask.npy", flat_mask)
    np.save(output_dir / "valid_mask.npy", valid_mask)
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
