import asyncio
import time
from dataclasses import dataclass, replace
from decimal import Decimal

import structlog
from pybit.unified_trading import HTTP, WebSocket

from bot.data.models import (
    Liquidation,
    LongShortRatio,
    OrderBookSnapshot,
    TickerContext,
    Trade,
)
from bot.data.storage import StorageWriter
from bot.execution.account import AccountState
from bot.execution.lifecycle import (
    build_entry_intent,
    build_exit_intent,
    decide_exit,
    passive_exit_price,
    should_cross_exit,
    should_repeg_exit,
)
from bot.execution.lifecycle import open_position as build_open_position
from bot.execution.orders import cancel_order_params, place_order_params
from bot.execution.types import (
    ExitPolicy,
    InstrumentSpec,
    OpenPosition,
    OrderIntent,
    OrderSide,
    OrderType,
)
from bot.inference.accumulator import TradeAccumulator
from bot.inference.adapters import (
    book_levels,
    liquidation_row,
    long_short_row,
    ticker_row,
    trade_row,
)
from bot.inference.predictor import InferencePredictor
from bot.inference.types import InferenceStatus
from bot.risk.types import MarketState
from bot.telemetry.collector import TelemetryCollector
from bot.telemetry.types import (
    APPROVED,
    BLOCKED,
    HOLDING,
    NO_BOOK,
    NOT_READY,
    REJECTED,
    WAITING,
    DecisionRecord,
    OrderEvent,
)

logger = structlog.get_logger()


@dataclass(slots=True)
class _RestingOrder:
    intent: OrderIntent
    link_id: str
    deadline_ms: int
    submitted_ms: int
    decision_price: Decimal
    cancelling: bool = False


class LiveTrader:
    """Drives one symbol: model signal in, Bybit orders out.

    Implements MarketObserver, so it consumes the very stream StorageWriter is
    recording. Snapshots are queued rather than acted on inline — the trading
    step awaits blocking pybit REST calls and must not stall ingestion.
    """

    def __init__(
        self,
        symbol: str,
        http: HTTP,
        predictor: InferencePredictor,
        account: AccountState,
        spec: InstrumentSpec,
        exit_policy: ExitPolicy,
        order_notional: Decimal,
        entry_order_type: OrderType,
        exit_order_type: OrderType,
        order_timeout_ms: int,
        horizon_index: int,
        collector: TelemetryCollector,
        storage: StorageWriter,
        exit_repeg: bool = False,
        rollup_interval_s: float = 60.0,
        queue_maxsize: int = 100,
        dry_run: bool = False,
    ) -> None:
        self._symbol = symbol
        self._http = http
        self._predictor = predictor
        self._account = account
        self._spec = spec
        self._exit_policy = exit_policy
        self._order_notional = order_notional
        self._entry_order_type = entry_order_type
        self._exit_order_type = exit_order_type
        self._exit_repeg = exit_repeg
        self._order_timeout_ms = order_timeout_ms
        self._signal_horizon = horizon_index
        self._collector = collector
        self._storage = storage
        self._rollup_interval_ms = int(rollup_interval_s * 1000)
        self._dry_run = dry_run
        self._last_rollup_ms = 0
        self._exit_decision_price: Decimal | None = None
        self._exit_reason: str | None = None
        self._exit_submitted_ms = 0
        self._exit_link_id: str | None = None
        self._exit_price: Decimal | None = None
        self._exit_escalated = False

        self._accumulator = TradeAccumulator()
        self._queue: asyncio.Queue[OrderBookSnapshot] = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._position: OpenPosition | None = None
        self._resting: _RestingOrder | None = None
        self._exit_pending = False

    # -- MarketObserver, all on the event loop --

    def on_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(snapshot)

    def on_trade(self, trade: Trade) -> None:
        self._accumulator.add(trade)

    def on_ticker(self, ticker: TickerContext) -> None:
        self._predictor.preprocessor.update_ticker(ticker_row(ticker))

    def on_liquidation(self, liquidation: Liquidation) -> None:
        self._predictor.preprocessor.update_liquidation(liquidation_row(liquidation))

    def on_long_short_ratio(self, ratio: LongShortRatio) -> None:
        self._predictor.preprocessor.update_long_short_ratio(long_short_row(ratio))

    # -- private WS callbacks (pybit thread) --

    def subscribe_private(self, ws: WebSocket) -> None:
        ws.position_stream(callback=lambda m: self._account.apply_position(m["data"]))
        ws.wallet_stream(callback=lambda m: self._account.apply_wallet(m["data"]))
        ws.order_stream(callback=lambda m: self._account.apply_order(m["data"]))

    async def run(self, shutdown: asyncio.Event) -> None:
        logger.info(
            "live.started",
            symbol=self._symbol,
            entry_order_type=self._entry_order_type.value,
            dry_run=self._dry_run,
        )
        while not shutdown.is_set():
            snapshot = await self._queue.get()
            await self.step(snapshot)

    async def step(self, snapshot: OrderBookSnapshot) -> None:
        """One decision cycle: fold the snapshot into features, then act on it.

        Every pass emits a DecisionRecord, including the passes where no
        prediction was possible. How often the loop cannot act at all is itself a
        diagnostic, and it is invisible when only trades are logged.
        """
        now_ms = _now_ms()

        if not snapshot.bids or not snapshot.asks:
            self._record(snapshot, NO_BOOK, now_ms, None)
            await self._maybe_rollup(now_ms)
            return

        bids, asks = book_levels(snapshot)
        frame = self._predictor.preprocessor.append_snapshot(
            timestamp_ms=snapshot.timestamp_ms,
            bids=bids,
            asks=asks,
            trades=[trade_row(t) for t in self._accumulator.drain()],
        )

        market = MarketState(
            symbol=self._symbol,
            timestamp_ms=snapshot.timestamp_ms,
            best_bid=snapshot.bids[0].price,
            best_ask=snapshot.asks[0].price,
            account_equity=self._account.equity,
        )

        if self._position is not None:
            await self._manage_position(market, now_ms)
            self._record(snapshot, HOLDING, now_ms, frame.flow)
        elif self._resting is not None:
            await self._reconcile_resting(now_ms)
            self._record(snapshot, WAITING, now_ms, frame.flow)
        else:
            await self._maybe_enter(snapshot, market, now_ms, frame.flow)

        await self._maybe_rollup(now_ms)

    async def _manage_position(self, market: MarketState, now_ms: int) -> None:
        if self._exit_pending:
            if self._account.is_flat:
                self._emit_order(
                    now_ms,
                    link_id=f"exit-{self._exit_submitted_ms}",
                    event="filled",
                    side=self._position.side.opposite.value,
                    order_type=self._exit_order_type.value,
                    reduce_only=True,
                    qty=self._position.qty,
                    fill_price=market.best_bid
                    if self._position.side is OrderSide.BUY
                    else market.best_ask,
                    decision_price=self._exit_decision_price,
                    latency_ms=now_ms - self._exit_submitted_ms,
                    exit_reason=self._exit_reason,
                )
                logger.info("live.position_closed", symbol=self._symbol)
                self._position = None
                self._exit_pending = False
                self._exit_escalated = False
                self._exit_link_id = None
                self._exit_price = None
                self._exit_reason = None
                self._exit_decision_price = None
            elif not self._exit_escalated:
                await self._reconcile_exit(market, now_ms)
            return

        decision = decide_exit(self._position, market, now_ms)
        if not decision.should_exit:
            return

        intent = build_exit_intent(
            self._position, market, self._exit_order_type, now_ms
        )
        self._exit_reason = decision.reason.value
        self._exit_decision_price = decision.exit_price
        self._exit_submitted_ms = now_ms
        self._exit_escalated = self._exit_order_type is OrderType.MARKET
        self._exit_price = intent.limit_price
        self._exit_link_id = await self._submit(intent, now_ms, decision.exit_price)
        self._exit_pending = True
        logger.info(
            "live.exit_submitted",
            symbol=self._symbol,
            reason=decision.reason.value,
            exit_price=str(decision.exit_price),
        )

    async def _reconcile_exit(self, market: MarketState, now_ms: int) -> None:
        """Keep a resting passive exit reachable, and cross it once it must be.

        A passive exit is posted at the top of its own side and then the book
        moves. Left alone it strands a tick or more behind the market exactly
        when the position needs out, which is what forces the crossing that costs
        more than the move being traded. Re-pegging follows the book while still
        paying the smaller maker fee; crossing stays the bounded fallback.
        """
        order = (
            self._account.order(self._exit_link_id)
            if self._exit_link_id is not None
            else None
        )
        still_resting = order is not None and order.is_open
        cross = should_cross_exit(
            self._position, market, now_ms, self._exit_submitted_ms, self._exit_policy
        )
        if still_resting and not cross:
            if not (
                self._exit_repeg
                and self._exit_price is not None
                and should_repeg_exit(
                    self._position.side.opposite, self._exit_price, market
                )
            ):
                return
            await self._cancel(self._exit_link_id)
            await self._repost_exit(market, now_ms)
            return

        if still_resting:
            await self._cancel(self._exit_link_id)

        # Size the crossing order off the position the exchange reports, not the
        # one this loop believes in: a passive exit can partially fill while the
        # cancel is in flight, and exiting the remembered quantity would flip the
        # position through zero into a side the policy never chose.
        held = self._account.position_qty
        if held == 0:
            return
        intent = replace(
            build_exit_intent(self._position, market, OrderType.MARKET, now_ms),
            qty=abs(held),
            side=OrderSide.SELL if held > 0 else OrderSide.BUY,
        )
        self._exit_link_id = await self._submit(
            intent, now_ms, self._exit_decision_price
        )
        self._exit_price = None
        self._exit_escalated = True
        logger.info(
            "live.exit_escalated",
            symbol=self._symbol,
            reason=self._exit_reason,
            waited_ms=now_ms - self._exit_submitted_ms,
            qty=str(intent.qty),
        )

    async def _repost_exit(self, market: MarketState, now_ms: int) -> None:
        """Re-post the passive exit at the side of the book it must join."""
        held = self._account.position_qty
        if held == 0:
            return
        side = OrderSide.SELL if held > 0 else OrderSide.BUY
        price = passive_exit_price(side, market)
        intent = replace(
            build_exit_intent(self._position, market, OrderType.POST_ONLY, now_ms),
            qty=abs(held),
            side=side,
            limit_price=price,
        )
        self._exit_price = price
        self._exit_link_id = await self._submit(
            intent, now_ms, self._exit_decision_price
        )
        logger.info(
            "live.exit_repegged",
            symbol=self._symbol,
            reason=self._exit_reason,
            price=str(price),
        )

    async def _reconcile_resting(self, now_ms: int) -> None:
        order = self._account.order(self._resting.link_id)

        if order is not None and order.is_filled:
            self._emit_order(
                now_ms,
                link_id=self._resting.link_id,
                event="filled",
                side=self._resting.intent.side.value,
                order_type=self._resting.intent.order_type.value,
                reduce_only=False,
                qty=self._resting.intent.qty,
                limit_price=self._resting.intent.limit_price,
                fill_price=order.avg_price,
                decision_price=self._resting.decision_price,
                latency_ms=now_ms - self._resting.submitted_ms,
            )
            self._position = build_open_position(
                self._resting.intent,
                order.avg_price,
                order.updated_ms,
                self._exit_policy,
            )
            logger.info(
                "live.entry_filled",
                symbol=self._symbol,
                side=self._resting.intent.side.value,
                entry_price=str(order.avg_price),
                qty=str(self._position.qty),
            )
            self._resting = None
            return

        if order is not None and not order.is_open:
            self._emit_order(
                now_ms,
                link_id=self._resting.link_id,
                event="cancelled" if self._resting.cancelling else "expired",
                side=self._resting.intent.side.value,
                order_type=self._resting.intent.order_type.value,
                reduce_only=False,
                qty=self._resting.intent.qty,
                limit_price=self._resting.intent.limit_price,
                latency_ms=now_ms - self._resting.submitted_ms,
            )
            logger.info(
                "live.entry_closed_unfilled",
                symbol=self._symbol,
                status=order.status,
                cancelled=self._resting.cancelling,
            )
            self._resting = None
            return

        # Cancelling is a request, not an event: the order can still trade before
        # it lands. Keep watching until the order stream says it is gone, or the
        # loop would believe it is flat while holding a real, unprotected position.
        if not self._resting.cancelling and now_ms >= self._resting.deadline_ms:
            await self._cancel(self._resting.link_id)
            self._resting.cancelling = True
            logger.info("live.entry_cancel_sent", symbol=self._symbol)

    async def _maybe_enter(
        self,
        snapshot: OrderBookSnapshot,
        market: MarketState,
        now_ms: int,
        flow_z,
    ) -> None:
        started_us = time.perf_counter_ns() // 1000
        result = self._predictor.predict(
            timestamp_ms=snapshot.timestamp_ms,
            position=self._account.position_state(),
            market=market,
            proposed_order_notional=self._order_notional,
            now_ms=now_ms,
        )
        inference_us = time.perf_counter_ns() // 1000 - started_us

        self._record(
            snapshot, _status_of(result), now_ms, flow_z,
            result=result, inference_us=inference_us,
        )
        if result.status is not InferenceStatus.APPROVED:
            return

        intent = build_entry_intent(
            result.signal, market, self._order_notional, self._entry_order_type,
            self._spec,
        )
        if intent is None:
            return

        decision_price = (
            market.best_bid if intent.side is OrderSide.BUY else market.best_ask
        )
        link_id = await self._submit(intent, now_ms, decision_price)
        # Market entries go through the same path: the order stream reports the
        # real average fill price, which is what the exit bracket must be built on.
        self._resting = _RestingOrder(
            intent=intent,
            link_id=link_id,
            deadline_ms=now_ms + self._order_timeout_ms,
            submitted_ms=now_ms,
            decision_price=decision_price,
        )

        logger.info(
            "live.entry_submitted",
            symbol=self._symbol,
            side=intent.side.value,
            qty=str(intent.qty),
            limit_price=str(intent.limit_price),
            confidence=round(result.signal.confidence, 4),
            link_id=link_id,
        )

    async def _submit(
        self,
        intent: OrderIntent,
        now_ms: int,
        decision_price: Decimal | None,
    ) -> str:
        params = place_order_params(intent, self._spec)
        if not self._dry_run:
            await asyncio.to_thread(self._http.place_order, **params)
        self._emit_order(
            now_ms,
            link_id=params["orderLinkId"],
            event="submitted",
            side=intent.side.value,
            order_type=intent.order_type.value,
            reduce_only=intent.reduce_only,
            qty=intent.qty,
            limit_price=intent.limit_price,
            decision_price=decision_price,
            latency_ms=_now_ms() - now_ms,
        )
        return params["orderLinkId"]

    # -- telemetry --

    def _record(
        self,
        snapshot: OrderBookSnapshot,
        status: str,
        now_ms: int,
        flow_z,
        result=None,
        inference_us: int = 0,
    ) -> None:
        ticker_age, liq_age, ls_age = self._predictor.preprocessor.context_ages_ms(
            now_ms
        )
        probs = _probabilities_of(result, self._signal_horizon)

        record = DecisionRecord(
            timestamp_ms=snapshot.timestamp_ms,
            status=status,
            action=result.signal.action.value if result and result.signal else "NONE",
            horizon_index=self._signal_horizon,
            p_down=probs[0],
            p_flat=probs[1],
            p_up=probs[2],
            predicted_class=max(range(3), key=lambda i: probs[i]),
            confidence=max(probs),
            best_bid=snapshot.bids[0].price if snapshot.bids else None,
            best_ask=snapshot.asks[0].price if snapshot.asks else None,
            book_age_ms=now_ms - snapshot.timestamp_ms,
            is_reset=snapshot.is_reset,
            inference_us=inference_us,
            signal_reason=result.signal.reason if result and result.signal else None,
            risk_reason=result.risk_decision.reason
            if result and result.risk_decision
            else None,
            ctx_ticker_age_ms=ticker_age,
            ctx_liq_age_ms=liq_age,
            ctx_ls_age_ms=ls_age,
        )
        self._storage.put_nowait(self._collector.observe_decision(record, flow_z))

    def _emit_order(self, now_ms: int, link_id: str, **fields) -> None:
        event = OrderEvent(timestamp_ms=now_ms, order_link_id=link_id, **fields)
        self._collector.observe_order(event)
        self._storage.put_nowait(event)

    async def _maybe_rollup(self, now_ms: int) -> None:
        if now_ms - self._last_rollup_ms < self._rollup_interval_ms:
            return
        if self._last_rollup_ms:
            logger.info("live.rollup", symbol=self._symbol, **self._collector.rollup(now_ms))
        else:
            self._collector.rollup(now_ms)
        self._last_rollup_ms = now_ms

    async def _cancel(self, link_id: str) -> None:
        if self._dry_run:
            return
        await asyncio.to_thread(
            self._http.cancel_order, **cancel_order_params(self._symbol, link_id)
        )


def _status_of(result) -> str:
    if result.status is InferenceStatus.NOT_READY:
        return NOT_READY
    if result.status is InferenceStatus.APPROVED:
        return APPROVED
    # A rejection before the risk gate is the signal policy declining to act;
    # the two are different failures and must not share a bucket.
    return BLOCKED if result.signal and result.signal.reason else REJECTED


def _probabilities_of(result, horizon_index: int) -> tuple[float, float, float]:
    if result is None or result.probabilities is None:
        return (0.0, 0.0, 0.0)
    return result.probabilities[horizon_index]


def _now_ms() -> int:
    return int(time.time() * 1000)
