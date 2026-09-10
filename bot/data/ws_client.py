import asyncio
import time
from decimal import Decimal

import structlog
from pybit.unified_trading import WebSocket

from bot.data.models import Liquidation, TickerContext, Trade
from bot.data.observer import MarketObserver
from bot.data.orderbook import OrderBookManager
from bot.data.storage import StorageWriter

logger = structlog.get_logger()


def _dec(value: str | None) -> Decimal | None:
    return Decimal(value) if value else None


class DataIngestion:
    """Bridges pybit WebSocket (thread-based) to asyncio pipeline.

    - Subscribes to orderbook, trade, ticker, and liquidation channels via pybit
    - Snapshots the order book every snapshot_interval_s on an asyncio timer
    - Pushes all data into StorageWriter's queue
    """

    def __init__(
        self,
        symbol: str,
        storage: StorageWriter,
        orderbook: OrderBookManager,
        lob_depth: int = 50,
        snapshot_interval_s: float = 0.1,
        observer: MarketObserver | None = None,
    ) -> None:
        self._symbol = symbol
        self._storage = storage
        self._ob = orderbook
        self._lob_depth = lob_depth
        self._snapshot_interval_s = snapshot_interval_s
        self._observer = observer
        self._ws: WebSocket | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self, shutdown: asyncio.Event) -> None:
        self._loop = asyncio.get_running_loop()

        self._ws = WebSocket(channel_type="linear")
        self._ws.orderbook_stream(
            depth=self._lob_depth,
            symbol=self._symbol,
            callback=self._on_orderbook,
        )
        self._ws.trade_stream(
            symbol=self._symbol,
            callback=self._on_trade,
        )
        self._ws.ticker_stream(
            symbol=self._symbol,
            callback=self._on_ticker,
        )
        self._ws.all_liquidation_stream(
            symbol=self._symbol,
            callback=self._on_liquidation,
        )
        logger.info("ws.subscribed", symbol=self._symbol)

        while not shutdown.is_set():
            snap = self._ob.snapshot(
                timestamp_ms=_now_ms(), depth=self._lob_depth
            )
            if snap is not None:
                self._storage.put_nowait(snap)
                if self._observer is not None:
                    self._observer.on_snapshot(snap)
            await asyncio.sleep(self._snapshot_interval_s)

    def stop(self) -> None:
        if self._ws is not None:
            self._ws.exit()

    # -- pybit callbacks (run in WS thread) --

    def _on_orderbook(self, message: dict) -> None:
        self._ob.update(message["data"])

    def _on_trade(self, message: dict) -> None:
        loop = self._loop
        if loop is None:
            return
        for t in message["data"]:
            trade = Trade(
                timestamp_ms=t["T"],
                trade_id=t["i"],
                side=t["S"],
                price=Decimal(t["p"]),
                qty=Decimal(t["v"]),
            )
            loop.call_soon_threadsafe(self._handle_trade, trade)

    def _on_ticker(self, message: dict) -> None:
        loop = self._loop
        if loop is None:
            return
        d = message["data"]
        ticker = TickerContext(
            timestamp_ms=message["ts"],
            mark_price=_dec(d.get("markPrice")),
            index_price=_dec(d.get("indexPrice")),
            open_interest=_dec(d.get("openInterest")),
            open_interest_value=_dec(d.get("openInterestValue")),
            funding_rate=_dec(d.get("fundingRate")),
            next_funding_time=int(d["nextFundingTime"]) if d.get("nextFundingTime") else None,
            price_24h_pct=_dec(d.get("price24hPcnt")),
            prev_price_1h=_dec(d.get("prevPrice1h")),
            volume_24h=_dec(d.get("volume24h")),
            turnover_24h=_dec(d.get("turnover24h")),
            collected_at_ms=_now_ms(),
        )
        loop.call_soon_threadsafe(self._handle_ticker, ticker)

    def _on_liquidation(self, message: dict) -> None:
        loop = self._loop
        if loop is None:
            return
        now = _now_ms()
        for event in message["data"]:
            liq = Liquidation(
                timestamp_ms=event["T"],
                side=event["S"],
                qty=Decimal(event["v"]),
                price=Decimal(event["p"]),
                collected_at_ms=now,
            )
            loop.call_soon_threadsafe(self._handle_liquidation, liq)

    # -- event-loop delivery: storage first, then the optional live consumer --

    def _handle_trade(self, trade: Trade) -> None:
        self._storage.put_nowait(trade)
        if self._observer is not None:
            self._observer.on_trade(trade)

    def _handle_ticker(self, ticker: TickerContext) -> None:
        self._storage.put_nowait(ticker)
        if self._observer is not None:
            self._observer.on_ticker(ticker)

    def _handle_liquidation(self, liquidation: Liquidation) -> None:
        self._storage.put_nowait(liquidation)
        if self._observer is not None:
            self._observer.on_liquidation(liquidation)


def _now_ms() -> int:
    return int(time.time() * 1000)
