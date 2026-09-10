from decimal import Decimal

import numpy as np

from bot.data.models import (
    Liquidation,
    LongShortRatio,
    OrderBookLevel,
    OrderBookSnapshot,
    TickerContext,
    Trade,
)
from bot.features.context_features import ContextBuilder
from bot.features.flow_features import extract_flow_features
from bot.features.ob_serializer import serialize_ob_snapshot
from bot.inference.accumulator import TradeAccumulator
from bot.inference.adapters import (
    book_levels,
    liquidation_row,
    long_short_row,
    ticker_row,
    trade_row,
)

NOW_MS = 1_800_000_000_000


def _ticker() -> TickerContext:
    return TickerContext(
        timestamp_ms=NOW_MS,
        mark_price=Decimal("10000"),
        index_price=Decimal("9998"),
        open_interest=Decimal("5"),
        open_interest_value=Decimal("50000"),
        funding_rate=Decimal("0.0001"),
        next_funding_time=NOW_MS + 3_600_000,
        price_24h_pct=Decimal("0.02"),
        prev_price_1h=Decimal("9990"),
        volume_24h=Decimal("1234"),
        turnover_24h=Decimal("999"),
        collected_at_ms=NOW_MS,
    )


class TestTickerAdapter:
    def test_feeds_context_builder(self) -> None:
        builder = ContextBuilder()
        builder.update_ticker(ticker_row(_ticker()))
        ctx = builder.snapshot(NOW_MS)

        assert ctx[0] == 10000.0                    # mark_price
        assert ctx[1] == 9998.0                     # index_price
        assert ctx[2] > 0                           # basis_bps, mark above index
        assert ctx[5] == np.float32(0.0001)         # funding_rate
        assert ctx[16] == 50000.0                   # open_interest_value

    def test_none_fields_survive_forward_fill(self) -> None:
        builder = ContextBuilder()
        builder.update_ticker(ticker_row(_ticker()))

        partial = TickerContext(
            timestamp_ms=NOW_MS + 100, mark_price=None, index_price=None,
            open_interest=None, open_interest_value=None, funding_rate=None,
            next_funding_time=None, price_24h_pct=None, prev_price_1h=None,
            volume_24h=None, turnover_24h=None, collected_at_ms=NOW_MS + 100,
        )
        builder.update_ticker(ticker_row(partial))

        assert builder.snapshot(NOW_MS + 100)[0] == 10000.0


class TestLiquidationAdapter:
    def test_lands_in_the_one_minute_window(self) -> None:
        builder = ContextBuilder()
        builder.update_liquidation(
            liquidation_row(
                Liquidation(NOW_MS, "Buy", Decimal("2.5"), Decimal("10000"), NOW_MS)
            )
        )
        ctx = builder.snapshot(NOW_MS)
        assert ctx[9] == 2.5     # liq_buy_vol_1m
        assert ctx[12] == 1.0    # liq_count_1m


class TestLongShortAdapter:
    def test_feeds_the_ratio_features(self) -> None:
        builder = ContextBuilder()
        builder.update_long_short_ratio(
            long_short_row(
                LongShortRatio(NOW_MS, Decimal("0.6"), Decimal("0.4"), NOW_MS)
            )
        )
        ctx = builder.snapshot(NOW_MS)
        assert ctx[13] == np.float32(0.6)
        assert ctx[14] == np.float32(0.4)
        assert ctx[15] == np.float32(1.5)


class TestTradeAdapter:
    def test_feeds_flow_extraction(self) -> None:
        trades = [
            trade_row(Trade(NOW_MS, "1", "Buy", Decimal("10000"), Decimal("1"))),
            trade_row(Trade(NOW_MS, "2", "Sell", Decimal("10000"), Decimal("3"))),
        ]
        flow = extract_flow_features(trades)
        assert flow[3] == 2.0     # trade_count
        assert flow[7] == 1.0     # buy_count
        assert flow[8] == 1.0     # sell_count
        assert flow[4] < 0        # ofi, sell-heavy


class TestBookLevels:
    def test_round_trips_through_the_serializer(self) -> None:
        snapshot = OrderBookSnapshot(
            timestamp_ms=NOW_MS,
            bids=[
                OrderBookLevel(Decimal("10000"), Decimal("1")),
                OrderBookLevel(Decimal("9999"), Decimal("2")),
            ],
            asks=[
                OrderBookLevel(Decimal("10001"), Decimal("1")),
                OrderBookLevel(Decimal("10002"), Decimal("2")),
            ],
        )
        bids, asks = book_levels(snapshot)
        assert bids == [["10000", "1"], ["9999", "2"]]

        ob = serialize_ob_snapshot(bids, asks, ob_depth=2)
        assert ob.shape == (2, 4)
        assert ob[0, 0] == 0.0    # best bid offset from itself
        assert ob[1, 0] > 0       # second level sits below the best bid


class TestTradeAccumulator:
    def test_drain_empties_the_bucket(self) -> None:
        accumulator = TradeAccumulator()
        accumulator.add(Trade(NOW_MS, "1", "Buy", Decimal("10000"), Decimal("1")))
        accumulator.add(Trade(NOW_MS, "2", "Sell", Decimal("10000"), Decimal("1")))

        assert len(accumulator) == 2
        assert len(accumulator.drain()) == 2
        assert len(accumulator) == 0

    def test_drain_on_an_empty_bucket(self) -> None:
        assert TradeAccumulator().drain() == []

    def test_buckets_do_not_leak_into_each_other(self) -> None:
        accumulator = TradeAccumulator()
        accumulator.add(Trade(NOW_MS, "1", "Buy", Decimal("10000"), Decimal("1")))
        first = accumulator.drain()
        accumulator.add(Trade(NOW_MS, "2", "Sell", Decimal("10000"), Decimal("1")))
        assert len(first) == 1
        assert len(accumulator.drain()) == 1
