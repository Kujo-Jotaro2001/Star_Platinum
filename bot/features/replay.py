import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path


def find_db_files(
    data_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
) -> list[Path]:
    """Find daily DB files for symbol in [start_date, end_date] range.

    Dates are inclusive, format YYYY-MM-DD.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    paths = []
    d = start
    while d <= end:
        p = data_dir / f"{symbol}_{d.isoformat()}.db"
        if p.exists():
            paths.append(p)
        d += timedelta(days=1)
    return paths


def _query_all(db_paths: list[Path], sql: str) -> list[dict]:
    """Run a query across multiple DB files, return rows as dicts in order."""
    rows: list[dict] = []
    for path in db_paths:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        for row in conn.execute(sql):
            rows.append(dict(row))
        conn.close()
    return rows


def load_snapshots(db_paths: list[Path]) -> list[dict]:
    rows = _query_all(
        db_paths,
        "SELECT timestamp_ms, is_reset, bids, asks FROM snapshots ORDER BY timestamp_ms",
    )
    for r in rows:
        r["bids"] = json.loads(r["bids"])
        r["asks"] = json.loads(r["asks"])
    return rows


def load_trades(db_paths: list[Path]) -> list[dict]:
    return _query_all(
        db_paths,
        "SELECT timestamp_ms, side, price, qty FROM trades ORDER BY timestamp_ms",
    )


def load_ticker_context(db_paths: list[Path]) -> list[dict]:
    return _query_all(
        db_paths,
        """SELECT timestamp_ms, mark_price, index_price, open_interest,
                  open_interest_value, funding_rate, next_funding_time,
                  price_24h_pct, prev_price_1h, volume_24h, turnover_24h
           FROM ticker_context ORDER BY timestamp_ms""",
    )


def load_liquidations(db_paths: list[Path]) -> list[dict]:
    return _query_all(
        db_paths,
        "SELECT timestamp_ms, side, qty, price FROM liquidations ORDER BY timestamp_ms",
    )


def load_long_short_ratio(db_paths: list[Path]) -> list[dict]:
    return _query_all(
        db_paths,
        "SELECT timestamp_ms, buy_ratio, sell_ratio FROM long_short_ratio ORDER BY timestamp_ms",
    )
