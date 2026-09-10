import asyncio
import os
import signal
from decimal import Decimal
from pathlib import Path

import hydra
import structlog
import uvloop
from dotenv import load_dotenv
from omegaconf import DictConfig
from pybit.unified_trading import HTTP, WebSocket

from bot.data.context_poller import poll_long_short_ratio
from bot.data.orderbook import OrderBookManager
from bot.data.storage import StorageWriter
from bot.data.ws_client import DataIngestion
from bot.execution.account import AccountState
from bot.execution.live import LiveTrader
from bot.execution.types import ExitPolicy, InstrumentSpec, OrderType
from bot.inference.predictor import InferencePredictor
from bot.risk.types import RiskPolicy
from bot.signals.types import SignalPolicy
from bot.telemetry.collector import TelemetryCollector

logger = structlog.get_logger()


async def _run(cfg: DictConfig) -> None:
    load_dotenv()
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    symbol: str = cfg.ingestion.symbol
    demo: bool = cfg.execution.demo
    api_key = os.environ["BYBIT_API_KEY"]
    api_secret = os.environ["BYBIT_API_SECRET"]

    horizon_index: int = cfg.signals.horizon_index
    hold_snapshots = (
        cfg.execution.hold_snapshots
        if cfg.execution.hold_snapshots is not None
        else cfg.features.label_horizons[horizon_index]
    )

    predictor = InferencePredictor.from_checkpoint(
        checkpoint_path=Path(cfg.inference.checkpoint_path),
        normalizer_stats_path=Path(cfg.inference.normalizer_stats_path),
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
        seq_len=cfg.model.seq_len,
        ob_depth=cfg.ingestion.lob_depth,
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
        use_ctx=cfg.model.use_ctx,
        device=cfg.inference.device,
        oi_history_window_ms=cfg.features.oi_history_window_ms,
        liq_window_ms=cfg.features.liq_window_ms,
    )

    account = AccountState(symbol)
    storage = StorageWriter(
        data_dir=Path(cfg.ingestion.data_dir),
        symbol=symbol,
        queue_maxsize=cfg.ingestion.queue_maxsize,
    )
    trader = LiveTrader(
        symbol=symbol,
        http=HTTP(api_key=api_key, api_secret=api_secret, demo=demo),
        predictor=predictor,
        account=account,
        spec=InstrumentSpec(
            qty_step=Decimal(str(cfg.execution.qty_step)),
            price_tick=Decimal(str(cfg.execution.price_tick)),
            min_order_qty=Decimal(str(cfg.execution.min_order_qty)),
        ),
        exit_policy=ExitPolicy(
            hold_ms=hold_snapshots * cfg.features.bucket_ms,
            take_profit_bps=Decimal(str(cfg.execution.take_profit_bps)),
            stop_loss_bps=Decimal(str(cfg.execution.stop_loss_bps)),
            exit_fallback_ms=cfg.execution.exit_fallback_ms,
            cross_on_stop=cfg.execution.cross_on_stop,
        ),
        order_notional=Decimal(str(cfg.execution.order_notional)),
        entry_order_type=OrderType(cfg.execution.entry_order_type),
        exit_order_type=OrderType(cfg.execution.exit_order_type),
        exit_repeg=cfg.execution.exit_repeg,
        order_timeout_ms=cfg.execution.order_timeout_ms,
        horizon_index=horizon_index,
        collector=TelemetryCollector(
            n_flow_features=cfg.model.in_features_flow,
            window_capacity=cfg.telemetry.window_capacity,
        ),
        storage=storage,
        rollup_interval_s=cfg.telemetry.rollup_interval_s,
        dry_run=cfg.execution.dry_run,
    )

    private_ws = WebSocket(
        channel_type="private",
        api_key=api_key,
        api_secret=api_secret,
        demo=demo,
    )
    trader.subscribe_private(private_ws)

    orderbook = OrderBookManager()
    ingestion = DataIngestion(
        symbol=symbol,
        storage=storage,
        orderbook=orderbook,
        lob_depth=cfg.ingestion.lob_depth,
        snapshot_interval_s=cfg.ingestion.snapshot_interval_s,
        observer=trader,
    )

    storage_task = asyncio.create_task(storage.run(shutdown))
    ingestion_task = asyncio.create_task(ingestion.run(shutdown))
    trader_task = asyncio.create_task(trader.run(shutdown))
    ls_ratio_task = asyncio.create_task(
        poll_long_short_ratio(
            HTTP(), storage, symbol, shutdown,
            interval_s=cfg.ingestion.ls_ratio_interval_s,
            observer=trader,
        )
    )

    logger.info("live.bot_started", symbol=symbol, demo=demo)
    await shutdown.wait()
    logger.info("live.bot_shutting_down")

    ingestion.stop()
    private_ws.exit()
    ingestion_task.cancel()
    trader_task.cancel()

    await asyncio.gather(ls_ratio_task, trader_task, return_exceptions=True)
    await storage_task
    logger.info("live.bot_stopped")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    uvloop.run(_run(cfg))


if __name__ == "__main__":
    main()
