from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import hydra
import numpy as np
import structlog
import torch
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, Recorder
from hftbacktest.stats import LinearAssetRecord
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bot.backtest.prepare import prepare_range
from bot.backtest.report import RunMeta, save_run
from bot.backtest.strategy import SignalSchedule, run_strategy
from bot.execution.types import ExitPolicy, InstrumentSpec, OrderType
from bot.inference.predictor import load_hybrid_model_from_checkpoint
from bot.risk.types import RiskPolicy
from bot.signals.types import SignalPolicy
from bot.telemetry.types import PipelineCounters
from bot.training.dataset import LOBDataset

logger = structlog.get_logger()

SPLIT_NAMES = ("train", "val", "test")
NS_PER_MS = 1_000_000


def split_bounds(n: int, split: str, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if split == "train":
        return 0, train_end
    if split == "val":
        return train_end, val_end
    if split == "test":
        return val_end, n
    raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")


def predict_probabilities(
    model: torch.nn.Module,
    dataset: LOBDataset,
    batch_size: int,
    device: torch.device,
    use_ctx: bool,
) -> np.ndarray:
    """Score every window in the dataset, returning [M, H, C] softmax outputs."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["ob"].to(device),
                batch["flow"].to(device),
                batch["ctx"].to(device) if use_ctx else None,
            )
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())

    return np.concatenate(chunks, axis=0)


def cached_probabilities(
    cache_dir: Path,
    checkpoint: Path,
    split: str,
    n_windows: int,
    compute,
) -> np.ndarray:
    """Score the split, or reuse the scores of an identical earlier pass.

    A config sweep varies only signals/risk/execution, which the model never
    sees, so every run of it would otherwise repeat the same GPU pass verbatim.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{checkpoint.stem}_{split}_{n_windows}.npy"
    if path.exists():
        logger.info("backtest.predictions_cached", path=str(path))
        return np.load(path)

    probabilities = compute()
    np.save(path, probabilities)
    return probabilities


def build_schedule(
    probabilities: np.ndarray,
    timestamps_ms: np.ndarray,
    seq_len: int,
) -> SignalSchedule:
    """Attach each scored window to the snapshot it was scored at.

    `LOBDataset` window `j` spans `[j, j+seq_len)` and is labelled at its last
    timestep, so row `j` belongs to snapshot `j + seq_len - 1`.
    """
    start = seq_len - 1
    aligned = timestamps_ms[start:start + len(probabilities)]
    return SignalSchedule(
        timestamps_ns=aligned.astype(np.int64) * NS_PER_MS,
        probabilities=probabilities,
    )


def covered_dates(timestamps_ms: np.ndarray) -> tuple[date, date]:
    """UTC dates the timestamps span, inclusive — the feed days that must exist."""
    return (_utc_date(timestamps_ms[0]), _utc_date(timestamps_ms[-1]))


def build_asset(
    feeds: list[Path],
    initial_snapshot: Path | None,
    latencies: list[Path],
    cfg: DictConfig,
) -> BacktestAsset:
    asset = (
        BacktestAsset()
        .data([str(p) for p in feeds])
        .linear_asset(1.0)
        .no_partial_fill_exchange()
        .trading_value_fee_model(
            cfg.backtest.maker_fee_rate, cfg.backtest.taker_fee_rate
        )
        .tick_size(cfg.execution.price_tick)
        .lot_size(cfg.execution.qty_step)
    )
    if initial_snapshot is not None:
        asset = asset.initial_snapshot(str(initial_snapshot))

    asset = _apply_queue_model(asset, cfg)
    return _apply_latency_model(asset, latencies, cfg)


def _apply_queue_model(asset: BacktestAsset, cfg: DictConfig) -> BacktestAsset:
    model = cfg.backtest.queue_model
    if model == "risk_adverse":
        return asset.risk_adverse_queue_model()
    if model == "log_prob":
        return asset.log_prob_queue_model()
    if model == "power_prob":
        return asset.power_prob_queue_model(cfg.backtest.queue_power)
    if model == "power_prob2":
        return asset.power_prob_queue_model2(cfg.backtest.queue_power)
    if model == "power_prob3":
        return asset.power_prob_queue_model3(cfg.backtest.queue_power)
    raise ValueError(f"unknown queue_model {model!r}")


def _apply_latency_model(
    asset: BacktestAsset,
    latencies: list[Path],
    cfg: DictConfig,
) -> BacktestAsset:
    if cfg.backtest.latency_model == "constant":
        return asset.constant_order_latency(
            cfg.backtest.entry_latency_ns, cfg.backtest.response_latency_ns
        )
    if cfg.backtest.latency_model == "intp":
        return asset.intp_order_latency([str(p) for p in latencies])
    raise ValueError(f"unknown latency_model {cfg.backtest.latency_model!r}")


def summary_row(record: LinearAssetRecord, book_size: float) -> dict:
    """The metric table as plain values.

    `Stats.summary()` returns a polars DataFrame rather than printing one, and
    `book_size` is what turns the metrics from absolute currency into returns.
    """
    df = record.stats(book_size=book_size).summary()
    row = df.row(0, named=True)
    return {key: (value.isoformat() if hasattr(value, "isoformat") else value)
            for key, value in row.items()}


def log_stats(stats: PipelineCounters, summary: dict) -> None:
    logger.info(
        "backtest.metrics",
        **{
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in summary.items()
        },
    )
    logger.info(
        "backtest.strategy",
        decisions=stats.decisions,
        signals_actionable=stats.signals_actionable,
        signals_approved=stats.signals_approved,
        orders_submitted=stats.orders_submitted,
        orders_filled=stats.orders_filled,
        orders_cancelled=stats.orders_cancelled,
        orders_expired=stats.orders_expired,
        fill_rate=round(stats.fill_rate, 4),
        approval_rate=round(stats.approval_rate, 4),
        blocked=stats.signal_blocked,
        risk_rejected=stats.risk_rejected,
        exit_repegs=stats.exit_repegs,
        exits=stats.exits_by_reason,
    )


def _utc_date(timestamp_ms: int) -> date:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).date()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    symbol: str = cfg.ingestion.symbol
    features_dir = Path("data/features") / symbol
    seq_len: int = cfg.model.seq_len
    use_ctx: bool = cfg.model.use_ctx

    ob_raw = np.load(features_dir / "ob_raw.npy", mmap_mode="r")
    flow = np.load(features_dir / "flow_features.npy", mmap_mode="r")
    ctx = np.load(features_dir / "ctx_features.npy", mmap_mode="r")
    labels = np.load(features_dir / "labels.npy", mmap_mode="r")
    flat_mask = np.load(features_dir / "flat_mask.npy", mmap_mode="r")
    valid_mask = np.load(features_dir / "valid_mask.npy", mmap_mode="r")
    timestamps_ms = np.load(features_dir / "timestamps_ms.npy")

    lo, hi = split_bounds(
        len(ob_raw), cfg.backtest.split, cfg.features.train_ratio, cfg.features.val_ratio
    )
    logger.info("backtest.split", split=cfg.backtest.split, start=lo, end=hi, n=hi - lo)

    dataset = LOBDataset(
        ob_raw[lo:hi], flow[lo:hi], ctx[lo:hi], labels[lo:hi], flat_mask[lo:hi],
        seq_len, use_ctx, valid_mask=valid_mask[lo:hi],
    )
    device = torch.device(
        "cuda" if cfg.backtest.device == "auto" and torch.cuda.is_available()
        else ("cpu" if cfg.backtest.device == "auto" else cfg.backtest.device)
    )
    model = load_hybrid_model_from_checkpoint(
        checkpoint_path=Path(cfg.backtest.checkpoint_path),
        device=device,
        ob_depth=cfg.model.ob_depth,
        d_model=cfg.model.d_model,
        d_ctx=cfg.model.d_ctx,
        n_lob_blocks=cfg.model.n_lob_blocks,
        in_features_flow=cfg.model.in_features_flow,
        in_features_ctx=cfg.model.in_features_ctx,
        flow_encoder=cfg.model.flow_encoder,
        gru_layers=cfg.model.gru_layers,
        n_classes=cfg.model.n_classes,
        n_horizons=len(cfg.features.label_horizons),
        dropout=cfg.model.dropout,
        use_ctx=use_ctx,
    ).to(device)

    logger.info("backtest.scoring", windows=len(dataset))
    probabilities = cached_probabilities(
        cache_dir=Path(cfg.backtest.prediction_cache_dir),
        checkpoint=Path(cfg.backtest.checkpoint_path),
        split=cfg.backtest.split,
        n_windows=len(dataset),
        compute=lambda: predict_probabilities(
            model, dataset, cfg.backtest.batch_size, device, use_ctx
        ),
    )
    schedule = build_schedule(probabilities, timestamps_ms[lo:hi], seq_len)

    start_date, end_date = covered_dates(schedule.timestamps_ns // NS_PER_MS)
    logger.info("backtest.preparing_feed", start=str(start_date), end=str(end_date))

    feeds, initial_snapshot, latencies = prepare_range(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        raw_dir=Path(cfg.historical.raw_dir),
        out_dir=Path(cfg.backtest.feed_dir),
        base_url=cfg.historical.base_url,
        tick_size=cfg.execution.price_tick,
        lot_size=cfg.execution.qty_step,
        feed_latency_ns=cfg.backtest.feed_latency_ns,
        base_latency_ns=cfg.backtest.base_latency_ns,
        mul_entry=cfg.backtest.mul_entry,
        offset_entry_ns=cfg.backtest.offset_entry_ns,
        mul_resp=cfg.backtest.mul_resp,
        offset_resp_ns=cfg.backtest.offset_resp_ns,
    )

    horizon_index: int = cfg.signals.horizon_index
    hold_snapshots = (
        cfg.execution.hold_snapshots
        if cfg.execution.hold_snapshots is not None
        else cfg.features.label_horizons[horizon_index]
    )

    hbt = HashMapMarketDepthBacktest(
        [build_asset(feeds, initial_snapshot, latencies, cfg)]
    )
    recorder = Recorder(hbt.num_assets, cfg.backtest.recorder_buffer_size)

    logger.info("backtest.running", feeds=len(feeds), predictions=len(schedule.probabilities))
    stats = run_strategy(
        hbt=hbt,
        schedule=schedule,
        symbol=symbol,
        signal_policy=SignalPolicy(
            horizon_index=horizon_index,
            min_confidence=cfg.signals.min_confidence,
            allow_short=cfg.signals.allow_short,
        ),
        risk_policy=RiskPolicy(
            max_position_notional=Decimal(str(cfg.risk.max_position_notional)),
            max_leverage=Decimal(str(cfg.risk.max_leverage)),
            max_spread_bps=Decimal(str(cfg.risk.max_spread_bps)),
            stale_data_ms=cfg.risk.stale_data_ms,
            cooldown_after_trade_ms=cfg.risk.cooldown_after_trade_ms,
            min_confidence=cfg.risk.min_confidence,
            allow_short=cfg.risk.allow_short,
        ),
        exit_policy=ExitPolicy(
            hold_ms=hold_snapshots * cfg.features.bucket_ms,
            take_profit_bps=Decimal(str(cfg.execution.take_profit_bps)),
            stop_loss_bps=Decimal(str(cfg.execution.stop_loss_bps)),
            exit_fallback_ms=cfg.execution.exit_fallback_ms,
            cross_on_stop=cfg.execution.cross_on_stop,
        ),
        spec=InstrumentSpec(
            qty_step=Decimal(str(cfg.execution.qty_step)),
            price_tick=Decimal(str(cfg.execution.price_tick)),
            min_order_qty=Decimal(str(cfg.execution.min_order_qty)),
        ),
        exit_order_type=OrderType(cfg.execution.exit_order_type),
        exit_repeg=cfg.execution.exit_repeg,
        order_notional=Decimal(str(cfg.execution.order_notional)),
        initial_equity=Decimal(str(cfg.backtest.initial_equity)),
        elapse_ns=cfg.features.bucket_ms * NS_PER_MS,
        order_timeout_ns=cfg.execution.order_timeout_ms * NS_PER_MS,
        recorder=recorder.recorder,
    )
    hbt.close()

    raw_record = recorder.get(0)
    initial_equity = float(cfg.backtest.initial_equity)
    summary = summary_row(LinearAssetRecord(raw_record), initial_equity)
    log_stats(stats, summary)

    tag = cfg.backtest.tag
    name = f"{symbol}_{cfg.backtest.split}" + (f"_{tag}" if tag else "")
    run_dir = Path(cfg.backtest.report_dir) / name
    report = save_run(
        run_dir=run_dir,
        record=raw_record,
        summary=summary,
        strategy=stats,
        meta=RunMeta(
            symbol=symbol,
            split=cfg.backtest.split,
            start=str(start_date),
            end=str(end_date),
            checkpoint=Path(cfg.backtest.checkpoint_path).name,
            queue_model=cfg.backtest.queue_model,
            latency_model=cfg.backtest.latency_model,
            maker_fee_rate=cfg.backtest.maker_fee_rate,
            taker_fee_rate=cfg.backtest.taker_fee_rate,
            hold_ms=hold_snapshots * cfg.features.bucket_ms,
            take_profit_bps=cfg.execution.take_profit_bps,
            stop_loss_bps=cfg.execution.stop_loss_bps,
            order_notional=cfg.execution.order_notional,
            initial_equity=initial_equity,
            min_confidence=cfg.signals.min_confidence,
            horizon_index=horizon_index,
            feed_latency_ns=cfg.backtest.feed_latency_ns,
            latency_is_measured=False,
        ),
    )
    logger.info("backtest.report", path=str(report))


if __name__ == "__main__":
    main()
