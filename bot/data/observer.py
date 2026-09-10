from typing import Protocol

from bot.data.models import (
    Liquidation,
    LongShortRatio,
    OrderBookSnapshot,
    TickerContext,
    Trade,
)


class MarketObserver(Protocol):
    """Optional live consumer of the same stream StorageWriter records.

    Every method runs on the event loop, never on the pybit WS thread.
    """

    def on_snapshot(self, snapshot: OrderBookSnapshot) -> None: ...

    def on_trade(self, trade: Trade) -> None: ...

    def on_ticker(self, ticker: TickerContext) -> None: ...

    def on_liquidation(self, liquidation: Liquidation) -> None: ...

    def on_long_short_ratio(self, ratio: LongShortRatio) -> None: ...
