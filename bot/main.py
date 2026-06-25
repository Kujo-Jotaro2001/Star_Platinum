import asyncio
import signal
from pathlib import Path

import hydra
import structlog
import uvloop
from dotenv import load_dotenv
from omegaconf import DictConfig
from pybit.unified_trading import HTTP

from bot.data.context_poller import poll_long_short_ratio
from bot.data.orderbook import OrderBookManager
from bot.data.storage import StorageWriter
from bot.data.ws_client import DataIngestion

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
    data_dir = Path(cfg.ingestion.data_dir)

    orderbook = OrderBookManager()
    storage = StorageWriter(
        data_dir=data_dir,
        symbol=symbol,
        queue_maxsize=cfg.ingestion.queue_maxsize,
    )
    ingestion = DataIngestion(
        symbol=symbol,
        storage=storage,
        orderbook=orderbook,
        lob_depth=cfg.ingestion.lob_depth,
        snapshot_interval_s=cfg.ingestion.snapshot_interval_s,
    )
    http = HTTP()

    storage_task = asyncio.create_task(storage.run(shutdown))
    ingestion_task = asyncio.create_task(ingestion.run(shutdown))
    ls_ratio_task = asyncio.create_task(
        poll_long_short_ratio(
            http, storage, symbol, shutdown,
            interval_s=cfg.ingestion.ls_ratio_interval_s,
        )
    )

    logger.info("bot.started", symbol=symbol)
    await shutdown.wait()
    logger.info("bot.shutting_down")

    ingestion.stop()
    ingestion_task.cancel()

    await asyncio.gather(ls_ratio_task, return_exceptions=True)
    await storage_task
    logger.info("bot.stopped")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    uvloop.run(_run(cfg))


if __name__ == "__main__":
    main()
