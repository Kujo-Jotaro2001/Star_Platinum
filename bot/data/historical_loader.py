import csv
import gzip
import io
import json
import sqlite3
import zipfile
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import hydra
import requests
import structlog
from omegaconf import DictConfig

from bot.data.storage import (
    SNAPSHOTS_DDL,
    SNAPSHOTS_INDEX,
    TRADES_DDL,
    TRADES_INDEX,
)

logger = structlog.get_logger()

SOURCE_INTERVAL_MS = 10  # quote-saver native snapshot cadence


def _parse_orderbook_lines(
    lines: Iterator[str],
    snapshot_interval_ms: int,
    lob_depth: int,
) -> Iterator[tuple[int, bool, str, str]]:
    """Parse JSON lines from ob500 file → (ts, is_reset, bids_json, asks_json).

    - Keeps every Nth snapshot where N = snapshot_interval_ms // SOURCE_INTERVAL_MS.
    - First kept snapshot has is_reset=True.
    - Seq regression (seq < last_seq) on a kept snapshot sets is_reset=True.
    - Slices bids/asks to lob_depth levels.
    """
    if snapshot_interval_ms % SOURCE_INTERVAL_MS != 0:
        raise ValueError(
            f"snapshot_interval_ms ({snapshot_interval_ms}) must be a multiple "
            f"of source cadence ({SOURCE_INTERVAL_MS})"
        )
    stride = snapshot_interval_ms // SOURCE_INTERVAL_MS

    last_seq = 0
    first = True
    for i, line in enumerate(lines):
        if i % stride != 0:
            continue
        msg = json.loads(line)
        data = msg["data"]
        ts = int(msg["ts"])
        seq = int(data.get("seq", 0))

        is_reset = first or seq < last_seq
        first = False
        last_seq = seq

        bids = [[str(p), str(q)] for p, q in data["b"][:lob_depth]]
        asks = [[str(p), str(q)] for p, q in data["a"][:lob_depth]]
        yield ts, is_reset, json.dumps(bids), json.dumps(asks)


def _parse_trade_csv(
    lines: Iterator[str],
) -> Iterator[tuple[int, str, str, str, str]]:
    """Parse Bybit trade CSV → (ts, trade_id, side, price, qty).

    Auto-detects whether the first row is a header.
    Columns (positional): timestamp, symbol, side, size, price, [tickDirection, trdMatchID, ...]
    trade_id taken from trdMatchID if present, else synthesized as f"{ts}-{price}-{qty}".
    """
    reader = csv.reader(lines)
    header: list[str] | None = None
    rows_iter = iter(reader)

    try:
        first_row = next(rows_iter)
    except StopIteration:
        return

    # Header detection: try to parse column 0 as int
    try:
        int(first_row[0])
        # No header — this is a data row, rewind
        data_rows: Iterator[list[str]] = _chain_one(first_row, rows_iter)
    except ValueError:
        header = [c.strip() for c in first_row]
        data_rows = rows_iter

    match_id_idx: int | None = None
    if header is not None:
        for i, col in enumerate(header):
            if col in ("trdMatchID", "trade_id", "tradeId"):
                match_id_idx = i
                break

    for row in data_rows:
        ts = int(row[0])
        side = row[2]
        qty = row[3]
        price = row[4]
        if match_id_idx is not None and match_id_idx < len(row) and row[match_id_idx]:
            trade_id = row[match_id_idx]
        else:
            trade_id = f"{ts}-{price}-{qty}"
        yield ts, trade_id, side, price, qty


def _chain_one(first: list[str], rest: Iterator[list[str]]) -> Iterator[list[str]]:
    yield first
    yield from rest


def _create_db(path: Path) -> sqlite3.Connection:
    """Open SQLite connection with snapshots + trades schema (historical only)."""
    conn = sqlite3.connect(path)
    conn.execute(SNAPSHOTS_DDL)
    conn.execute(TRADES_DDL)
    conn.execute(SNAPSHOTS_INDEX)
    conn.execute(TRADES_INDEX)
    conn.commit()
    return conn


def _download_orderbook(base_url: str, symbol: str, date_str: str) -> Iterator[str]:
    url = f"{base_url}/orderbook/linear/{symbol}/{date_str}_{symbol}_ob500.data.zip"
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    buf = io.BytesIO(resp.content)
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError(f"Empty zip at {url}")
        with zf.open(names[0]) as f:
            for raw in f:
                line = raw.decode("utf-8").strip()
                if line:
                    yield line


def _download_trades(base_url: str, symbol: str, date_str: str) -> Iterator[str]:
    url = f"{base_url}/trade/linear/{symbol}/{date_str}_{symbol}.csv.gz"
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
        for raw in gz:
            line = raw.decode("utf-8").rstrip("\n\r")
            if line:
                yield line


def _load_day(
    symbol: str,
    day: date,
    db_dir: Path,
    snapshot_interval_ms: int,
    lob_depth: int,
    base_url: str,
) -> None:
    date_str = day.isoformat()
    db_path = db_dir / f"{symbol}_{date_str}.db"
    if db_path.exists():
        logger.info("historical.skip_existing", date=date_str, path=str(db_path))
        return

    logger.info("historical.loading_day", date=date_str, symbol=symbol)

    ob_rows = list(_parse_orderbook_lines(
        _download_orderbook(base_url, symbol, date_str),
        snapshot_interval_ms=snapshot_interval_ms,
        lob_depth=lob_depth,
    ))
    trade_rows = list(_parse_trade_csv(_download_trades(base_url, symbol, date_str)))

    db_dir.mkdir(parents=True, exist_ok=True)
    conn = _create_db(db_path)
    try:
        conn.executemany(
            "INSERT INTO snapshots (timestamp_ms, is_reset, bids, asks) VALUES (?, ?, ?, ?)",
            [(ts, int(is_reset), bids, asks) for ts, is_reset, bids, asks in ob_rows],
        )
        conn.executemany(
            "INSERT INTO trades (timestamp_ms, trade_id, side, price, qty) VALUES (?, ?, ?, ?, ?)",
            trade_rows,
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "historical.day_done",
        date=date_str,
        snapshots=len(ob_rows),
        trades=len(trade_rows),
        path=str(db_path),
    )


def load_historical_data(
    symbol: str,
    start_date: date,
    end_date: date,
    db_dir: str,
    snapshot_interval_ms: int,
    lob_depth: int,
    base_url: str = "https://quote-saver.bycsi.com",
) -> None:
    """Download Bybit historical OB + trades into daily SQLite files.

    Resumable: days whose DB file already exists are skipped.
    """
    db_dir_path = Path(db_dir)
    day = start_date
    while day <= end_date:
        _load_day(
            symbol=symbol,
            day=day,
            db_dir=db_dir_path,
            snapshot_interval_ms=snapshot_interval_ms,
            lob_depth=lob_depth,
            base_url=base_url,
        )
        day += timedelta(days=1)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    load_historical_data(
        symbol=cfg.ingestion.symbol,
        start_date=date.fromisoformat(cfg.historical.start_date),
        end_date=date.fromisoformat(cfg.historical.end_date),
        db_dir=cfg.ingestion.data_dir,
        snapshot_interval_ms=cfg.historical.snapshot_interval_ms,
        lob_depth=cfg.ingestion.lob_depth,
        base_url=cfg.historical.base_url,
    )


if __name__ == "__main__":
    main()
