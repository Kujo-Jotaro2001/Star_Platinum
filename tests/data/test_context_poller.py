import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bot.data.context_poller import poll_long_short_ratio
from bot.data.models import LongShortRatio
from bot.data.storage import StorageWriter

_FAST = 0.005  # interval_s for tests — short enough to get multiple cycles


def _ls_response(ts: str, buy: str, sell: str) -> dict:
    return {
        "retCode": 0,
        "result": {
            "list": [{"timestamp": ts, "buyRatio": buy, "sellRatio": sell}],
        },
    }


@pytest.fixture
def storage(tmp_path: Path) -> StorageWriter:
    return StorageWriter(data_dir=tmp_path / "data", symbol="BTCUSDT")


async def _drain(storage: StorageWriter) -> list:
    items = []
    while not storage._queue.empty():
        items.append(storage._queue.get_nowait())
    return items


class TestPollLongShortRatio:
    async def test_polls_and_enqueues(
        self,
        storage: StorageWriter,
    ) -> None:
        http = MagicMock()
        http.get_long_short_ratio.return_value = _ls_response(
            ts="1712300000000", buy="0.55", sell="0.45"
        )

        shutdown = asyncio.Event()
        task = asyncio.create_task(
            poll_long_short_ratio(http, storage, "BTCUSDT", shutdown, interval_s=_FAST)
        )
        await asyncio.sleep(0.005)
        shutdown.set()
        await asyncio.wait_for(task, timeout=1.0)

        http.get_long_short_ratio.assert_called_with(
            category="linear",
            symbol="BTCUSDT",
            period="15min",
            limit=1,
        )
        items = await _drain(storage)
        assert len(items) >= 1
        assert isinstance(items[0], LongShortRatio)
        assert items[0].timestamp_ms == 1712300000000
        assert str(items[0].buy_ratio) == "0.55"
        assert str(items[0].sell_ratio) == "0.45"

    async def test_dedup_same_timestamp(
        self,
        storage: StorageWriter,
    ) -> None:
        http = MagicMock()
        http.get_long_short_ratio.return_value = _ls_response(
            ts="1712300000000", buy="0.55", sell="0.45"
        )

        shutdown = asyncio.Event()
        task = asyncio.create_task(
            poll_long_short_ratio(http, storage, "BTCUSDT", shutdown, interval_s=_FAST)
        )
        await asyncio.sleep(0.03)  # several cycles
        shutdown.set()
        await asyncio.wait_for(task, timeout=1.0)

        items = await _drain(storage)
        assert len(items) == 1

    async def test_enqueues_when_timestamp_advances(
        self,
        storage: StorageWriter,
    ) -> None:
        responses = [
            _ls_response(ts="1712300000000", buy="0.55", sell="0.45"),
            _ls_response(ts="1712300900000", buy="0.52", sell="0.48"),
        ]
        call_count = {"n": 0}

        def fake_get(**_kwargs: object) -> dict:
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            return responses[idx]

        http = MagicMock()
        http.get_long_short_ratio.side_effect = fake_get

        shutdown = asyncio.Event()
        task = asyncio.create_task(
            poll_long_short_ratio(http, storage, "BTCUSDT", shutdown, interval_s=_FAST)
        )
        await asyncio.sleep(0.03)
        shutdown.set()
        await asyncio.wait_for(task, timeout=1.0)

        items = await _drain(storage)
        assert len(items) == 2
        assert items[0].timestamp_ms == 1712300000000
        assert items[1].timestamp_ms == 1712300900000

    async def test_empty_list_is_safe(
        self,
        storage: StorageWriter,
    ) -> None:
        http = MagicMock()
        http.get_long_short_ratio.return_value = {"result": {"list": []}}

        shutdown = asyncio.Event()
        task = asyncio.create_task(
            poll_long_short_ratio(http, storage, "BTCUSDT", shutdown, interval_s=_FAST)
        )
        await asyncio.sleep(0.005)
        shutdown.set()
        await asyncio.wait_for(task, timeout=1.0)

        items = await _drain(storage)
        assert items == []

    async def test_stops_on_shutdown(self, storage: StorageWriter) -> None:
        http = MagicMock()
        http.get_long_short_ratio.return_value = _ls_response(
            ts="1712300000000", buy="0.55", sell="0.45"
        )

        shutdown = asyncio.Event()
        shutdown.set()
        task = asyncio.create_task(
            poll_long_short_ratio(http, storage, "BTCUSDT", shutdown)
        )
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done()
