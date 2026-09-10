import csv
import gzip
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
from sortedcontainers import SortedDict

from bot.data.storage import (
    SNAPSHOTS_DDL,
    SNAPSHOTS_INDEX,
    TRADES_DDL,
    TRADES_INDEX,
)

logger = structlog.get_logger()

SOURCE_INTERVAL_MS = 100  # quote-saver ob500 cadence, measured from the archives
TRADES_BASE_URL = "https://public.bybit.com"


def _parse_orderbook_lines(
    lines: Iterator[str],
    snapshot_interval_ms: int,
    lob_depth: int,
) -> Iterator[tuple[int, bool, str, str]]:
    """Rebuild the book from the snapshot/delta stream → (ts, is_reset, bids, asks).

    Only the first message of a file is a full `snapshot`; every later line is a
    `delta` carrying just the levels that changed, with quantity "0" meaning the
    level is gone. Storing a delta as if it were a book would fill the database
    with fragments, so state is carried the same way `OrderBookManager` carries
    it live — which is also what keeps the offline and online books identical.

    Emits one row per `snapshot_interval_ms`; the source itself runs at
    `SOURCE_INTERVAL_MS`.
    """
    if snapshot_interval_ms % SOURCE_INTERVAL_MS != 0:
        raise ValueError(
            f"snapshot_interval_ms ({snapshot_interval_ms}) must be a multiple "
            f"of source cadence ({SOURCE_INTERVAL_MS})"
        )
    stride = snapshot_interval_ms // SOURCE_INTERVAL_MS

    bids: SortedDict = SortedDict()
    asks: SortedDict = SortedDict()
    last_seq = 0
    first = True
    kept = 0

    for i, line in enumerate(lines):
        msg = json.loads(line)
        data = msg["data"]
        seq = int(data.get("seq", 0))

        reset_here = msg.get("type") == "snapshot" or seq < last_seq
        if reset_here:
            bids.clear()
            asks.clear()
        last_seq = seq

        _apply_levels(bids, data.get("b", ()))
        _apply_levels(asks, data.get("a", ()))

        if i % stride != 0 or not bids or not asks:
            continue

        is_reset = first or reset_here
        first = False
        kept += 1

        yield (
            int(msg["ts"]),
            is_reset,
            json.dumps(_top(bids, lob_depth, descending=True)),
            json.dumps(_top(asks, lob_depth, descending=False)),
        )


def _apply_levels(side: SortedDict, levels) -> None:
    """Apply one side of a delta. A quantity of zero removes the level."""
    for price, qty in levels:
        key = float(price)
        if float(qty) == 0.0:
            side.pop(key, None)
        else:
            side[key] = (price, qty)


def _top(side: SortedDict, depth: int, descending: bool) -> list[list[str]]:
    """Best `depth` levels as [price, qty] strings, best first."""
    keys = side.keys()
    chosen = reversed(keys[-depth:]) if descending else keys[:depth]
    return [list(side[k]) for k in chosen]


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

    # Header detection: a data row starts with a numeric timestamp. It is float
    # seconds in Bybit's own files, so parsing it as an int would misread every
    # data row as a header and silently drop the first trade.
    try:
        float(first_row[0])
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
        # Bybit writes the trade timestamp as float seconds ("1755648000.1385"),
        # not integer milliseconds.
        ts = int(float(row[0]) * 1000)
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


def raw_paths(raw_dir: Path, symbol: str, date_str: str) -> tuple[Path, Path]:
    """Cache locations of the day's two source archives: (order book, trades).

    The names match the upstream ones, which is also what
    `hftbacktest.data.utils.bybithistmktdata.convert` expects to be handed.
    """
    return (
        raw_dir / f"{date_str}_{symbol}_ob500.data.zip",
        raw_dir / f"{symbol}{date_str}.csv.gz",
    )


def ensure_raw_files(
    base_url: str,
    raw_dir: Path,
    symbol: str,
    date_str: str,
    trades_url: str = TRADES_BASE_URL,
) -> tuple[Path, Path]:
    """Download the day's archives unless already cached. Returns their paths."""
    ob_path, trades_path = raw_paths(raw_dir, symbol, date_str)
    _download_if_missing(
        f"{base_url}/orderbook/linear/{symbol}/{date_str}_{symbol}_ob500.data.zip",
        ob_path,
    )
    # Order book and trades come from different hosts: quote-saver serves the
    # book archives, while trades live on Bybit's own public data site.
    _download_if_missing(
        f"{trades_url}/trading/{symbol}/{symbol}{date_str}.csv.gz",
        trades_path,
    )
    return ob_path, trades_path


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    # Download to a partial name and rename: an interrupted transfer must not be
    # left behind looking like a complete cache entry on the next run.
    partial = path.with_name(path.name + ".part")
    with partial.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    partial.replace(path)


def _read_orderbook(path: Path) -> Iterator[str]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError(f"Empty zip: {path}")
        with zf.open(names[0]) as f:
            for raw in f:
                line = raw.decode("utf-8").strip()
                if line:
                    yield line


def _read_trades(path: Path) -> Iterator[str]:
    with gzip.open(path, "rb") as gz:
        for raw in gz:
            line = raw.decode("utf-8").strip()
            if line:
                yield line


def _load_day(
    symbol: str,
    day: date,
    db_dir: Path,
    raw_dir: Path,
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

    ob_path, trades_path = ensure_raw_files(base_url, raw_dir, symbol, date_str)

    ob_rows = list(_parse_orderbook_lines(
        _read_orderbook(ob_path),
        snapshot_interval_ms=snapshot_interval_ms,
        lob_depth=lob_depth,
    ))
    trade_rows = list(_parse_trade_csv(_read_trades(trades_path)))

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
    raw_dir: str,
    snapshot_interval_ms: int,
    lob_depth: int,
    base_url: str = "https://quote-saver.bycsi.com",
) -> None:
    """Download Bybit historical OB + trades into daily SQLite files.

    Resumable: days whose DB file already exists are skipped. The source archives
    are kept under `raw_dir` — the feature pipeline reads the downsampled SQLite,
    while the backtest feeds the untouched archives straight to hftbacktest.
    """
    db_dir_path = Path(db_dir)
    raw_dir_path = Path(raw_dir)
    day = start_date
    while day <= end_date:
        _load_day(
            symbol=symbol,
            day=day,
            db_dir=db_dir_path,
            raw_dir=raw_dir_path,
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
        raw_dir=cfg.historical.raw_dir,
        snapshot_interval_ms=cfg.historical.snapshot_interval_ms,
        lob_depth=cfg.ingestion.lob_depth,
        base_url=cfg.historical.base_url,
    )


if __name__ == "__main__":
    main()
