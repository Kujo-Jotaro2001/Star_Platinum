import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import structlog

from bot.data.models import (
    Liquidation,
    LongShortRatio,
    OrderBookSnapshot,
    TickerContext,
    Trade,
)

logger = structlog.get_logger()

SNAPSHOTS_DDL = """
CREATE TABLE IF NOT EXISTS snapshots (
    timestamp_ms INTEGER NOT NULL,
    is_reset     INTEGER NOT NULL DEFAULT 0,
    bids         TEXT    NOT NULL,
    asks         TEXT    NOT NULL
)
"""

TRADES_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    timestamp_ms INTEGER NOT NULL,
    trade_id     TEXT    NOT NULL,
    side         TEXT    NOT NULL,
    price        TEXT    NOT NULL,
    qty          TEXT    NOT NULL
)
"""

_TICKER_CONTEXT_DDL = """
CREATE TABLE IF NOT EXISTS ticker_context (
    timestamp_ms        INTEGER NOT NULL,
    mark_price          TEXT,
    index_price         TEXT,
    open_interest       TEXT,
    open_interest_value TEXT,
    funding_rate        TEXT,
    next_funding_time   INTEGER,
    price_24h_pct       TEXT,
    prev_price_1h       TEXT,
    volume_24h          TEXT,
    turnover_24h        TEXT,
    collected_at_ms     INTEGER NOT NULL
)
"""

_LIQUIDATIONS_DDL = """
CREATE TABLE IF NOT EXISTS liquidations (
    timestamp_ms    INTEGER NOT NULL,
    side            TEXT    NOT NULL,
    qty             TEXT    NOT NULL,
    price           TEXT    NOT NULL,
    collected_at_ms INTEGER NOT NULL
)
"""

_LONG_SHORT_RATIO_DDL = """
CREATE TABLE IF NOT EXISTS long_short_ratio (
    timestamp_ms    INTEGER NOT NULL UNIQUE,
    buy_ratio       TEXT    NOT NULL,
    sell_ratio      TEXT    NOT NULL,
    collected_at_ms INTEGER NOT NULL
)
"""

SNAPSHOTS_INDEX = "CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(timestamp_ms)"
TRADES_INDEX = "CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp_ms)"
_TICKER_CONTEXT_INDEX = "CREATE INDEX IF NOT EXISTS idx_ticker_context_ts ON ticker_context(timestamp_ms)"
_LIQUIDATIONS_INDEX = "CREATE INDEX IF NOT EXISTS idx_liquidations_ts ON liquidations(timestamp_ms)"


def _levels_to_json(levels: list) -> str:
    return json.dumps([[str(lv.price), str(lv.qty)] for lv in levels])


def _dec(value: object) -> str | None:
    return str(value) if value is not None else None


def _db_path(data_dir: Path, symbol: str, dt: datetime) -> Path:
    return data_dir / f"{symbol}_{dt.strftime('%Y-%m-%d')}.db"


class StorageWriter:
    """Async consumer that writes all data types to daily SQLite files."""

    def __init__(
        self,
        data_dir: Path,
        symbol: str,
        queue_maxsize: int = 10_000,
    ) -> None:
        self._data_dir = data_dir
        self._symbol = symbol
        self._queue: asyncio.Queue[
            OrderBookSnapshot | Trade | TickerContext | Liquidation | LongShortRatio
        ] = asyncio.Queue(maxsize=queue_maxsize)
        self._db: aiosqlite.Connection | None = None
        self._current_date: str = ""
        self._dropped: int = 0

    def put_nowait(
        self,
        item: OrderBookSnapshot | Trade | TickerContext | Liquidation | LongShortRatio,
    ) -> None:
        """Non-blocking enqueue. Drops oldest item on backpressure."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._dropped += 1
        self._queue.put_nowait(item)

    async def run(self, shutdown: asyncio.Event) -> None:
        """Main consumer loop. Runs until shutdown is set and queue is drained."""
        self._data_dir.mkdir(parents=True, exist_ok=True)

        while not shutdown.is_set():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            await self._write(item)

        # Flush remaining items on shutdown
        await self._flush()

        if self._db is not None:
            await self._db.close()

    async def _flush(self) -> None:
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self._write(item)

    async def _write(
        self,
        item: OrderBookSnapshot | Trade | TickerContext | Liquidation | LongShortRatio,
    ) -> None:
        db = await self._ensure_db(item.timestamp_ms)
        if isinstance(item, OrderBookSnapshot):
            await db.execute(
                "INSERT INTO snapshots (timestamp_ms, is_reset, bids, asks) VALUES (?, ?, ?, ?)",
                (
                    item.timestamp_ms,
                    int(item.is_reset),
                    _levels_to_json(item.bids),
                    _levels_to_json(item.asks),
                ),
            )
        elif isinstance(item, Trade):
            await db.execute(
                "INSERT INTO trades (timestamp_ms, trade_id, side, price, qty) VALUES (?, ?, ?, ?, ?)",
                (
                    item.timestamp_ms,
                    item.trade_id,
                    item.side,
                    str(item.price),
                    str(item.qty),
                ),
            )
        elif isinstance(item, TickerContext):
            await db.execute(
                """INSERT INTO ticker_context (
                    timestamp_ms, mark_price, index_price, open_interest,
                    open_interest_value, funding_rate, next_funding_time,
                    price_24h_pct, prev_price_1h, volume_24h, turnover_24h,
                    collected_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item.timestamp_ms,
                    _dec(item.mark_price),
                    _dec(item.index_price),
                    _dec(item.open_interest),
                    _dec(item.open_interest_value),
                    _dec(item.funding_rate),
                    item.next_funding_time,
                    _dec(item.price_24h_pct),
                    _dec(item.prev_price_1h),
                    _dec(item.volume_24h),
                    _dec(item.turnover_24h),
                    item.collected_at_ms,
                ),
            )
        elif isinstance(item, Liquidation):
            await db.execute(
                "INSERT INTO liquidations (timestamp_ms, side, qty, price, collected_at_ms) VALUES (?, ?, ?, ?, ?)",
                (
                    item.timestamp_ms,
                    item.side,
                    str(item.qty),
                    str(item.price),
                    item.collected_at_ms,
                ),
            )
        elif isinstance(item, LongShortRatio):
            await db.execute(
                "INSERT OR IGNORE INTO long_short_ratio (timestamp_ms, buy_ratio, sell_ratio, collected_at_ms) VALUES (?, ?, ?, ?)",
                (
                    item.timestamp_ms,
                    str(item.buy_ratio),
                    str(item.sell_ratio),
                    item.collected_at_ms,
                ),
            )
        await db.commit()

        if self._dropped > 0:
            logger.warning("storage.backpressure", dropped=self._dropped)
            self._dropped = 0

    async def _ensure_db(self, timestamp_ms: int) -> aiosqlite.Connection:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")

        if date_str != self._current_date:
            if self._db is not None:
                await self._db.close()
            path = _db_path(self._data_dir, self._symbol, dt)
            self._db = await aiosqlite.connect(path)
            await self._db.execute(SNAPSHOTS_DDL)
            await self._db.execute(TRADES_DDL)
            await self._db.execute(_TICKER_CONTEXT_DDL)
            await self._db.execute(_LIQUIDATIONS_DDL)
            await self._db.execute(_LONG_SHORT_RATIO_DDL)
            await self._db.execute(SNAPSHOTS_INDEX)
            await self._db.execute(TRADES_INDEX)
            await self._db.execute(_TICKER_CONTEXT_INDEX)
            await self._db.execute(_LIQUIDATIONS_INDEX)
            await self._db.commit()
            self._current_date = date_str
            logger.info("storage.db_opened", path=str(path))

        return self._db  # type: ignore[return-value]
