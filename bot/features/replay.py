import json
import sqlite3
from collections.abc import Iterator
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


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _query_all(db_paths: list[Path], sql: str, table: str) -> list[dict]:
    """Run a query across multiple DB files, return rows as dicts in order.

    Files written by the historical loader hold only `snapshots` and `trades` —
    Bybit's archives carry no ticker, liquidation or ratio feed — so a missing
    table means that source simply has nothing to say for those days, not that
    the file is broken.
    """
    rows: list[dict] = []
    for path in db_paths:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            if not _table_exists(conn, table):
                continue
            for row in conn.execute(sql):
                rows.append(dict(row))
        finally:
            conn.close()
    return rows


def count_snapshots(db_paths: list[Path]) -> int:
    """Total snapshots across the files, so output arrays can be sized up front."""
    total = 0
    for path in db_paths:
        conn = sqlite3.connect(path)
        total += conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        conn.close()
    return total


def load_snapshot_timestamps(db_path: Path) -> list[int]:
    """Just the timestamps for one day — enough to bucket that day's trades."""
    conn = sqlite3.connect(db_path)
    rows = [r[0] for r in conn.execute(
        "SELECT timestamp_ms FROM snapshots ORDER BY timestamp_ms"
    )]
    conn.close()
    return rows


def iter_snapshots(db_path: Path) -> Iterator[dict]:
    """Stream one day's snapshots.

    A day is ~860k snapshots and each one costs ~20 KB once its levels are parsed
    into Python objects, so materialising even a single day exceeds the memory of
    an ordinary machine. The pipeline consumes these one at a time.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute(
            "SELECT timestamp_ms, is_reset, bids, asks FROM snapshots ORDER BY timestamp_ms"
        ):
            yield {
                "timestamp_ms": row["timestamp_ms"],
                "is_reset": row["is_reset"],
                "bids": json.loads(row["bids"]),
                "asks": json.loads(row["asks"]),
            }
    finally:
        conn.close()


def load_snapshots(db_paths: list[Path]) -> list[dict]:
    rows = _query_all(
        db_paths,
        "SELECT timestamp_ms, is_reset, bids, asks FROM snapshots ORDER BY timestamp_ms",
        "snapshots",
    )
    for r in rows:
        r["bids"] = json.loads(r["bids"])
        r["asks"] = json.loads(r["asks"])
    return rows


def load_trades(db_paths: list[Path]) -> list[dict]:
    return _query_all(
        db_paths,
        "SELECT timestamp_ms, side, price, qty FROM trades ORDER BY timestamp_ms",
        "trades",
    )


def load_ticker_context(db_paths: list[Path]) -> list[dict]:
    return _query_all(
        db_paths,
        """SELECT timestamp_ms, mark_price, index_price, open_interest,
                  open_interest_value, funding_rate, next_funding_time,
                  price_24h_pct, prev_price_1h, volume_24h, turnover_24h
           FROM ticker_context ORDER BY timestamp_ms""",
        "ticker_context",
    )


def load_liquidations(db_paths: list[Path]) -> list[dict]:
    return _query_all(
        db_paths,
        "SELECT timestamp_ms, side, qty, price FROM liquidations ORDER BY timestamp_ms",
        "liquidations",
    )


def load_long_short_ratio(db_paths: list[Path]) -> list[dict]:
    return _query_all(
        db_paths,
        "SELECT timestamp_ms, buy_ratio, sell_ratio FROM long_short_ratio ORDER BY timestamp_ms",
        "long_short_ratio",
    )
