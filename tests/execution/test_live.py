from decimal import Decimal

import pytest

from bot.data.models import (
    Liquidation,
    LongShortRatio,
    OrderBookLevel,
    OrderBookSnapshot,
    TickerContext,
    Trade,
)
from bot.execution.account import AccountState
from bot.execution.live import LiveTrader
from bot.execution.types import ExitPolicy, InstrumentSpec, OrderType
from bot.features.normalizer import RollingNormalizer
from bot.inference.preprocessor import OnlinePreprocessor
from bot.inference.types import InferenceResult, InferenceStatus
from bot.risk.types import RiskDecision
from bot.signals.types import SignalAction, TradingSignal
from bot.telemetry.collector import TelemetryCollector
from bot.telemetry.types import DecisionRecord, OrderEvent

SYMBOL = "BTCUSDT"
NOW_MS = 1_800_000_000_000

SPEC = InstrumentSpec(
    qty_step=Decimal("0.001"),
    price_tick=Decimal("0.1"),
    min_order_qty=Decimal("0.001"),
)
EXIT_POLICY = ExitPolicy(
    hold_ms=1000,
    take_profit_bps=Decimal("15"),
    stop_loss_bps=Decimal("10"),
)


class FakeStorage:
    """Captures what the trader would persist, without touching a database."""

    def __init__(self) -> None:
        self.items: list[object] = []

    def put_nowait(self, item: object) -> None:
        self.items.append(item)

    def of_type(self, cls) -> list:
        return [i for i in self.items if isinstance(i, cls)]


class FakeHTTP:
    def __init__(self) -> None:
        self.placed: list[dict] = []
        self.cancelled: list[dict] = []

    def place_order(self, **kwargs) -> dict:
        self.placed.append(kwargs)
        return {"retCode": 0}

    def cancel_order(self, **kwargs) -> dict:
        self.cancelled.append(kwargs)
        return {"retCode": 0}


class FakePredictor:
    def __init__(self) -> None:
        self.preprocessor = OnlinePreprocessor(
            seq_len=2,
            ob_depth=2,
            flow_normalizer=RollingNormalizer(num_features=9, window=5),
        )
        self.result = _approved_result()
        self.calls = 0

    def predict(self, **kwargs) -> InferenceResult:
        self.calls += 1
        return self.result


def _signal(action: SignalAction = SignalAction.BUY) -> TradingSignal:
    return TradingSignal(
        symbol=SYMBOL,
        timestamp_ms=NOW_MS,
        action=action,
        confidence=0.9,
        horizon_index=0,
        class_probs=(0.05, 0.05, 0.90),
    )


def _approved_result() -> InferenceResult:
    signal = _signal()
    return InferenceResult(
        timestamp_ms=NOW_MS,
        status=InferenceStatus.APPROVED,
        probabilities=((0.05, 0.05, 0.90),),
        signal=signal,
        risk_decision=RiskDecision(True, "approved", signal),
        reason="approved",
    )


def _rejected_result() -> InferenceResult:
    signal = _signal(SignalAction.NO_TRADE)
    return InferenceResult(
        timestamp_ms=NOW_MS,
        status=InferenceStatus.REJECTED,
        probabilities=((0.05, 0.90, 0.05),),
        signal=signal,
        risk_decision=RiskDecision(False, "flat_selected", signal),
        reason="flat_selected",
    )


def _snapshot(
    bid: str = "10000",
    ask: str = "10001",
    timestamp_ms: int = NOW_MS,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp_ms=timestamp_ms,
        bids=[OrderBookLevel(Decimal(bid), Decimal("1"))],
        asks=[OrderBookLevel(Decimal(ask), Decimal("1"))],
    )


def _order_row(status: str, link_id: str, avg_price: str = "10000", ms: int = NOW_MS) -> dict:
    return {
        "symbol": SYMBOL,
        "orderId": "id",
        "orderLinkId": link_id,
        "orderStatus": status,
        "side": "Buy",
        "qty": "0.01",
        "cumExecQty": "0.01" if status == "Filled" else "0",
        "avgPrice": avg_price,
        "updatedTime": str(ms),
    }


@pytest.fixture
def clock(monkeypatch) -> dict:
    state = {"now": NOW_MS}
    monkeypatch.setattr("bot.execution.live._now_ms", lambda: state["now"])
    return state


def _trader(
    http: FakeHTTP,
    predictor: FakePredictor,
    account: AccountState,
    entry_order_type: OrderType = OrderType.POST_ONLY,
    dry_run: bool = False,
    storage: FakeStorage | None = None,
) -> LiveTrader:
    return LiveTrader(
        symbol=SYMBOL,
        http=http,
        predictor=predictor,
        account=account,
        spec=SPEC,
        exit_policy=EXIT_POLICY,
        order_notional=Decimal("100"),
        entry_order_type=entry_order_type,
        exit_order_type=OrderType.MARKET,
        order_timeout_ms=1000,
        horizon_index=0,
        collector=TelemetryCollector(n_flow_features=9),
        storage=storage if storage is not None else FakeStorage(),
        rollup_interval_s=3600.0,
        dry_run=dry_run,
    )


def _parts() -> tuple[FakeHTTP, FakePredictor, AccountState]:
    return FakeHTTP(), FakePredictor(), AccountState(SYMBOL)


class TestObserver:
    def test_snapshots_are_queued(self) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        trader.on_snapshot(_snapshot())
        assert trader._queue.qsize() == 1

    def test_queue_drops_oldest_when_full(self) -> None:
        http, predictor, account = _parts()
        trader = LiveTrader(
            symbol=SYMBOL, http=http, predictor=predictor, account=account,
            spec=SPEC, exit_policy=EXIT_POLICY, order_notional=Decimal("100"),
            entry_order_type=OrderType.POST_ONLY, exit_order_type=OrderType.MARKET,
            order_timeout_ms=1000, queue_maxsize=2, horizon_index=0,
            collector=TelemetryCollector(n_flow_features=9), storage=FakeStorage(),
        )
        for i in range(5):
            trader.on_snapshot(_snapshot(timestamp_ms=NOW_MS + i))
        assert trader._queue.qsize() == 2
        assert trader._queue.get_nowait().timestamp_ms == NOW_MS + 3

    def test_context_updates_reach_the_preprocessor(self) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)

        trader.on_ticker(
            TickerContext(
                timestamp_ms=NOW_MS, mark_price=Decimal("10000"),
                index_price=Decimal("9999"), open_interest=Decimal("5"),
                open_interest_value=Decimal("50000"), funding_rate=Decimal("0.0001"),
                next_funding_time=None, price_24h_pct=Decimal("0.01"),
                prev_price_1h=None, volume_24h=Decimal("100"),
                turnover_24h=None, collected_at_ms=NOW_MS,
            )
        )
        trader.on_liquidation(
            Liquidation(NOW_MS, "Buy", Decimal("1"), Decimal("10000"), NOW_MS)
        )
        trader.on_long_short_ratio(
            LongShortRatio(NOW_MS, Decimal("0.6"), Decimal("0.4"), NOW_MS)
        )

        ctx = predictor.preprocessor._ctx_builder.snapshot(NOW_MS)
        assert ctx[0] == 10000.0     # mark_price
        assert ctx[9] == 1.0         # liq_buy_vol_1m
        assert ctx[13] == pytest.approx(0.6)  # ls_buy_ratio

    def test_trades_accumulate_until_the_next_snapshot(self) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        trader.on_trade(Trade(NOW_MS, "1", "Buy", Decimal("10000"), Decimal("0.5")))
        trader.on_trade(Trade(NOW_MS, "2", "Sell", Decimal("10000"), Decimal("0.5")))
        assert len(trader._accumulator) == 2


class TestEntry:
    async def test_approved_signal_places_a_post_only_order(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())

        assert len(http.placed) == 1
        order = http.placed[0]
        assert order["side"] == "Buy"
        assert order["timeInForce"] == "PostOnly"
        assert order["price"] == "10000.0"
        assert order["reduceOnly"] is False

    async def test_rejected_signal_places_nothing(self, clock) -> None:
        http, predictor, account = _parts()
        predictor.result = _rejected_result()
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())
        assert http.placed == []

    async def test_empty_book_side_is_skipped(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        await trader.step(OrderBookSnapshot(NOW_MS, bids=[], asks=[]))
        assert predictor.calls == 0
        assert http.placed == []

    async def test_dry_run_decides_but_sends_nothing(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account, dry_run=True)
        await trader.step(_snapshot())
        assert predictor.calls == 1
        assert http.placed == []

    async def test_no_second_order_while_one_rests(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())
        await trader.step(_snapshot())
        assert len(http.placed) == 1

    async def test_market_entry_still_waits_for_the_reported_fill(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account, entry_order_type=OrderType.MARKET)
        await trader.step(_snapshot())
        assert http.placed[0]["orderType"] == "Market"
        assert trader._position is None
        assert trader._resting is not None


class TestRestingOrderReconciliation:
    async def test_fill_opens_a_position_at_the_average_price(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())
        link_id = http.placed[0]["orderLinkId"]

        account.apply_order([_order_row("Filled", link_id, avg_price="9999.5")])
        await trader.step(_snapshot())

        assert trader._resting is None
        assert trader._position is not None
        assert trader._position.entry_price == Decimal("9999.5")
        assert trader._position.take_profit_price == Decimal("9999.5") * Decimal("1.0015")

    async def test_cancelled_order_clears_the_resting_state(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())
        link_id = http.placed[0]["orderLinkId"]

        account.apply_order([_order_row("Cancelled", link_id)])
        await trader.step(_snapshot())

        assert trader._resting is None
        assert trader._position is None

    async def test_timeout_sends_a_cancel_but_keeps_watching(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())

        clock["now"] = NOW_MS + 1000
        await trader.step(_snapshot())

        assert http.cancelled[0]["orderLinkId"] == http.placed[0]["orderLinkId"]
        # cancelling is a request, not an event — the order is still live
        assert trader._resting is not None
        assert trader._resting.cancelling is True

    async def test_cancel_is_sent_once(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())

        clock["now"] = NOW_MS + 1000
        await trader.step(_snapshot())
        clock["now"] = NOW_MS + 2000
        await trader.step(_snapshot())

        assert len(http.cancelled) == 1

    async def test_a_fill_after_the_cancel_still_opens_the_position(self, clock) -> None:
        """The exchange can trade an order between the cancel request and its arrival.

        Missing that fill leaves the loop believing it is flat while holding a
        real position with no protective exit attached to it.
        """
        http, predictor, account = _parts()
        storage = FakeStorage()
        trader = _trader(http, predictor, account, storage=storage)
        await trader.step(_snapshot())
        link_id = http.placed[0]["orderLinkId"]

        clock["now"] = NOW_MS + 1000
        await trader.step(_snapshot())
        assert trader._resting is not None

        # the cancel is in flight; the order trades anyway
        account.apply_order([_order_row("Filled", link_id, avg_price="10000")])
        clock["now"] = NOW_MS + 1100
        await trader.step(_snapshot())

        assert trader._position is not None
        assert trader._position.qty == Decimal("0.010")
        assert trader._resting is None

    async def test_a_confirmed_cancel_clears_the_resting_state(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())
        link_id = http.placed[0]["orderLinkId"]

        clock["now"] = NOW_MS + 1000
        await trader.step(_snapshot())
        account.apply_order([_order_row("Cancelled", link_id)])
        clock["now"] = NOW_MS + 1100
        await trader.step(_snapshot())

        assert trader._resting is None
        assert trader._position is None

    async def test_no_second_entry_while_a_cancel_is_in_flight(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())

        clock["now"] = NOW_MS + 1000
        await trader.step(_snapshot())
        clock["now"] = NOW_MS + 1100
        await trader.step(_snapshot())

        # exactly one entry order exists until the first one is resolved
        assert len([o for o in http.placed if not o["reduceOnly"]]) == 1

    async def test_still_resting_before_the_deadline(self, clock) -> None:
        http, predictor, account = _parts()
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())

        clock["now"] = NOW_MS + 999
        await trader.step(_snapshot())

        assert trader._resting is not None
        assert http.cancelled == []


class TestExit:
    async def _open_position(self, clock, http, predictor, account) -> LiveTrader:
        trader = _trader(http, predictor, account)
        await trader.step(_snapshot())
        account.apply_order([_order_row("Filled", http.placed[0]["orderLinkId"])])
        await trader.step(_snapshot())
        account.apply_position(
            [{"symbol": SYMBOL, "side": "Buy", "size": "0.01", "entryPrice": "10000"}]
        )
        return trader

    async def test_holds_inside_the_bracket(self, clock) -> None:
        http, predictor, account = _parts()
        trader = await self._open_position(clock, http, predictor, account)
        clock["now"] = NOW_MS + 500
        await trader.step(_snapshot())
        assert len(http.placed) == 1

    async def test_take_profit_sends_a_reduce_only_market_order(self, clock) -> None:
        http, predictor, account = _parts()
        trader = await self._open_position(clock, http, predictor, account)

        clock["now"] = NOW_MS + 500
        await trader.step(_snapshot(bid="10020", ask="10021"))

        exit_order = http.placed[-1]
        assert exit_order["side"] == "Sell"
        assert exit_order["reduceOnly"] is True
        assert exit_order["orderType"] == "Market"

    async def test_stop_loss_exits(self, clock) -> None:
        http, predictor, account = _parts()
        trader = await self._open_position(clock, http, predictor, account)

        clock["now"] = NOW_MS + 500
        await trader.step(_snapshot(bid="9980", ask="9981"))

        assert http.placed[-1]["reduceOnly"] is True

    async def test_hold_expiry_exits(self, clock) -> None:
        http, predictor, account = _parts()
        trader = await self._open_position(clock, http, predictor, account)

        clock["now"] = NOW_MS + 1000
        await trader.step(_snapshot())

        assert http.placed[-1]["reduceOnly"] is True

    async def test_only_one_exit_is_sent_while_it_settles(self, clock) -> None:
        http, predictor, account = _parts()
        trader = await self._open_position(clock, http, predictor, account)

        clock["now"] = NOW_MS + 1000
        await trader.step(_snapshot())
        await trader.step(_snapshot())

        assert sum(1 for o in http.placed if o["reduceOnly"]) == 1

    async def test_position_clears_once_the_account_reports_flat(self, clock) -> None:
        http, predictor, account = _parts()
        trader = await self._open_position(clock, http, predictor, account)

        clock["now"] = NOW_MS + 1000
        await trader.step(_snapshot())
        account.apply_position(
            [{"symbol": SYMBOL, "side": "", "size": "0", "entryPrice": ""}]
        )
        await trader.step(_snapshot())

        assert trader._position is None
        assert trader._exit_pending is False


class TestTelemetry:
    async def test_every_pass_records_a_decision(self, clock) -> None:
        http, predictor, account = _parts()
        storage = FakeStorage()
        trader = _trader(http, predictor, account, storage=storage)

        await trader.step(_snapshot())
        await trader.step(_snapshot())
        assert len(storage.of_type(DecisionRecord)) == 2

    async def test_an_incomplete_book_is_still_recorded(self, clock) -> None:
        http, predictor, account = _parts()
        storage = FakeStorage()
        trader = _trader(http, predictor, account, storage=storage)

        await trader.step(OrderBookSnapshot(NOW_MS, bids=[], asks=[]))
        records = storage.of_type(DecisionRecord)
        assert len(records) == 1
        assert records[0].status == "NO_BOOK"
        assert predictor.calls == 0

    async def test_a_rejection_keeps_its_reason(self, clock) -> None:
        http, predictor, account = _parts()
        predictor.result = _rejected_result()
        storage = FakeStorage()
        trader = _trader(http, predictor, account, storage=storage)

        await trader.step(_snapshot())
        record = storage.of_type(DecisionRecord)[0]
        assert record.risk_reason == "flat_selected"
        assert record.status in ("BLOCKED", "REJECTED")
        assert trader._collector.session.decisions == 1

    async def test_holding_a_position_is_recorded_without_a_prediction(self, clock) -> None:
        http, predictor, account = _parts()
        storage = FakeStorage()
        trader = _trader(http, predictor, account, storage=storage)

        await trader.step(_snapshot())
        account.apply_order([_order_row("Filled", http.placed[0]["orderLinkId"])])
        await trader.step(_snapshot())
        calls_before = predictor.calls
        await trader.step(_snapshot())

        assert predictor.calls == calls_before  # no prediction while holding
        assert storage.of_type(DecisionRecord)[-1].status == "HOLDING"

    async def test_order_events_carry_measured_latency(self, clock) -> None:
        http, predictor, account = _parts()
        storage = FakeStorage()
        trader = _trader(http, predictor, account, storage=storage)

        await trader.step(_snapshot())
        clock["now"] = NOW_MS + 250
        account.apply_order([_order_row("Filled", http.placed[0]["orderLinkId"])])
        await trader.step(_snapshot())

        fills = [e for e in storage.of_type(OrderEvent) if e.event == "filled"]
        assert fills[0].latency_ms == 250
        assert fills[0].decision_price == Decimal("10000")

    async def test_counters_match_what_was_persisted(self, clock) -> None:
        http, predictor, account = _parts()
        storage = FakeStorage()
        trader = _trader(http, predictor, account, storage=storage)

        await trader.step(_snapshot())
        counters = trader._collector.session
        assert counters.signals_actionable == 1
        assert counters.signals_approved == 1
        assert counters.orders_submitted == 1

    async def test_context_ages_reach_the_record(self, clock) -> None:
        http, predictor, account = _parts()
        storage = FakeStorage()
        trader = _trader(http, predictor, account, storage=storage)

        trader.on_long_short_ratio(
            LongShortRatio(NOW_MS - 5000, Decimal("0.6"), Decimal("0.4"), NOW_MS)
        )
        await trader.step(_snapshot())

        record = storage.of_type(DecisionRecord)[0]
        assert record.ctx_ls_age_ms == 5000
        assert record.ctx_ticker_age_ms == -1  # never seen
