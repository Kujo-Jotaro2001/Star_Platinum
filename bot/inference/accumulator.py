from bot.data.models import Trade


class TradeAccumulator:
    """Collects trades arriving between two snapshots into one flow bucket.

    Both `add` and `drain` run on the event loop — DataIngestion hands trades
    over with call_soon_threadsafe — so no locking is needed here.
    """

    def __init__(self) -> None:
        self._trades: list[Trade] = []

    def add(self, trade: Trade) -> None:
        self._trades.append(trade)

    def drain(self) -> list[Trade]:
        trades = self._trades
        self._trades = []
        return trades

    def __len__(self) -> int:
        return len(self._trades)
