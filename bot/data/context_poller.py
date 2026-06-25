import asyncio
import time

from pybit.unified_trading import HTTP

from bot.data.models import LongShortRatio
from bot.data.storage import StorageWriter


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _sleep_or_stop(shutdown: asyncio.Event, timeout: float) -> bool:
    """Sleep until timeout elapses or shutdown is set. Returns True if shutdown."""
    try:
        await asyncio.wait_for(shutdown.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


async def poll_long_short_ratio(
    http: HTTP,
    storage: StorageWriter,
    symbol: str,
    shutdown: asyncio.Event,
    interval_s: float = 900.0,
) -> None:
    last_ts: int | None = None

    while not shutdown.is_set():
        response = await asyncio.to_thread(
            http.get_long_short_ratio,
            category="linear",
            symbol=symbol,
            period="15min",
            limit=1,
        )
        items = response["result"]["list"]
        if items:
            latest = items[0]
            ts = int(latest["timestamp"])
            if last_ts is None or ts > last_ts:
                storage.put_nowait(LongShortRatio(
                    timestamp_ms=ts,
                    buy_ratio=latest["buyRatio"],
                    sell_ratio=latest["sellRatio"],
                    collected_at_ms=_now_ms(),
                ))
                last_ts = ts

        if await _sleep_or_stop(shutdown, interval_s):
            return
