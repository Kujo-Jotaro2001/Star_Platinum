"""Adapters from the live WS dataclasses to the row shapes the Phase 2 feature
extractors expect, so online and offline feed identical inputs into the same code.
"""

from bot.data.models import (
    Liquidation,
    LongShortRatio,
    OrderBookSnapshot,
    TickerContext,
    Trade,
)


def ticker_row(ticker: TickerContext) -> dict:
    return {
        "timestamp_ms": ticker.timestamp_ms,
        "mark_price": ticker.mark_price,
        "index_price": ticker.index_price,
        "open_interest": ticker.open_interest,
        "open_interest_value": ticker.open_interest_value,
        "funding_rate": ticker.funding_rate,
        "volume_24h": ticker.volume_24h,
        "price_24h_pct": ticker.price_24h_pct,
    }


def liquidation_row(liquidation: Liquidation) -> dict:
    return {
        "timestamp_ms": liquidation.timestamp_ms,
        "side": liquidation.side,
        "qty": liquidation.qty,
    }


def long_short_row(ratio: LongShortRatio) -> dict:
    return {
        "timestamp_ms": ratio.timestamp_ms,
        "buy_ratio": ratio.buy_ratio,
        "sell_ratio": ratio.sell_ratio,
    }


def trade_row(trade: Trade) -> dict:
    return {
        "side": trade.side,
        "price": trade.price,
        "qty": trade.qty,
    }


def book_levels(snapshot: OrderBookSnapshot) -> tuple[list[list[str]], list[list[str]]]:
    """Levels in the [price, qty] string form serialize_ob_snapshot parses."""
    return (
        [[str(level.price), str(level.qty)] for level in snapshot.bids],
        [[str(level.price), str(level.qty)] for level in snapshot.asks],
    )
