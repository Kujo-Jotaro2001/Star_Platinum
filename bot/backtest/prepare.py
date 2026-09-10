"""Turn cached Bybit source archives into hftbacktest feed files.

The archives `historical_loader` caches under `cfg.historical.raw_dir` are exactly
what `hftbacktest.data.utils.bybithistmktdata` consumes, so the backtest replays
the untouched 10 ms / 500-level feed rather than the 100 ms / 50-level slice the
feature pipeline stores.

Caveat carried by the source itself: Bybit's public history has no local
timestamp, so feed latency is synthesised by adding `feed_latency_ns` to every
exchange timestamp. Order latency derived from it is therefore structured but not
measured — replace it with a latency file collected on Demo before trusting fill
rates.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import structlog
from hftbacktest.data.utils import bybithistmktdata
from hftbacktest.data.utils.feed_order_latency import generate_order_latency
from hftbacktest.data.utils.snapshot import create_last_snapshot

from bot.data.historical_loader import ensure_raw_files, raw_paths

logger = structlog.get_logger()

FEED_SUFFIX = ".npz"


def feed_path(out_dir: Path, symbol: str, date_str: str) -> Path:
    return out_dir / f"{symbol}_{date_str}{FEED_SUFFIX}"


def snapshot_path(out_dir: Path, symbol: str, date_str: str) -> Path:
    return out_dir / f"{symbol}_{date_str}_eod{FEED_SUFFIX}"


def latency_path(out_dir: Path, symbol: str, date_str: str) -> Path:
    return out_dir / f"{symbol}_{date_str}_latency{FEED_SUFFIX}"


def convert_day(
    symbol: str,
    date_str: str,
    raw_dir: Path,
    out_dir: Path,
    base_url: str,
    feed_latency_ns: int,
    base_latency_ns: int,
) -> Path:
    """Download the day's archives if needed, then convert them to a feed file."""
    out_path = feed_path(out_dir, symbol, date_str)
    if out_path.exists():
        return out_path

    depth_file, trades_file = ensure_raw_files(base_url, raw_dir, symbol, date_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert to a partial name and rename. A conversion killed part-way leaves a
    # truncated npz, and every later run would treat it as a finished cache entry
    # and fail deep inside hftbacktest with an unreadable-archive error. The
    # partial name has to keep the .npz suffix: numpy appends one otherwise, and
    # the rename would then look for a file that was never written.
    partial = out_path.with_name(f"{out_path.stem}.part{out_path.suffix}")
    bybithistmktdata.convert(
        depth_filename=str(depth_file),
        trades_filename=str(trades_file),
        output_filename=str(partial),
        feed_latency=feed_latency_ns,
        base_latency=base_latency_ns,
    )
    partial.replace(out_path)
    return out_path


def convert_range(
    symbol: str,
    start_date: date,
    end_date: date,
    raw_dir: Path,
    out_dir: Path,
    base_url: str,
    feed_latency_ns: int,
    base_latency_ns: int,
) -> list[Path]:
    """Convert every day in the inclusive range, in chronological order."""
    paths: list[Path] = []
    day = start_date
    while day <= end_date:
        paths.append(
            convert_day(
                symbol=symbol,
                date_str=day.isoformat(),
                raw_dir=raw_dir,
                out_dir=out_dir,
                base_url=base_url,
                feed_latency_ns=feed_latency_ns,
                base_latency_ns=base_latency_ns,
            )
        )
        day += timedelta(days=1)
    return paths


def build_eod_snapshot(
    feed_file: Path,
    out_dir: Path,
    symbol: str,
    date_str: str,
    tick_size: float,
    lot_size: float,
    initial_snapshot: Path | None = None,
) -> Path:
    """End-of-day book state, used to seed the next day without a warm-up gap."""
    out_path = snapshot_path(out_dir, symbol, date_str)
    if out_path.exists():
        return out_path

    create_last_snapshot(
        [str(feed_file)],
        tick_size=tick_size,
        lot_size=lot_size,
        initial_snapshot=str(initial_snapshot) if initial_snapshot else None,
        output_snapshot_filename=str(out_path),
    )
    return out_path


def build_order_latency(
    feed_file: Path,
    out_dir: Path,
    symbol: str,
    date_str: str,
    mul_entry: float,
    offset_entry_ns: float,
    mul_resp: float,
    offset_resp_ns: float,
) -> Path:
    """Order latency as a linear function of the feed latency in the same file."""
    out_path = latency_path(out_dir, symbol, date_str)
    if out_path.exists():
        return out_path

    generate_order_latency(
        feed_file=str(feed_file),
        output_file=str(out_path),
        mul_entry=mul_entry,
        offset_entry=offset_entry_ns,
        mul_resp=mul_resp,
        offset_resp=offset_resp_ns,
    )
    return out_path


def prepare_range(
    symbol: str,
    start_date: date,
    end_date: date,
    raw_dir: Path,
    out_dir: Path,
    base_url: str,
    tick_size: float,
    lot_size: float,
    feed_latency_ns: int,
    base_latency_ns: int,
    mul_entry: float,
    offset_entry_ns: float,
    mul_resp: float,
    offset_resp_ns: float,
) -> tuple[list[Path], Path | None, list[Path]]:
    """Full preparation for a date range.

    Returns the feed files, the snapshot seeding the first day (None when the
    range starts at the beginning of the available history) and the latency files.
    """
    feeds = convert_range(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        raw_dir=raw_dir,
        out_dir=out_dir,
        base_url=base_url,
        feed_latency_ns=feed_latency_ns,
        base_latency_ns=base_latency_ns,
    )

    latencies = [
        build_order_latency(
            feed_file=feed,
            out_dir=out_dir,
            symbol=symbol,
            date_str=_date_str_of(feed, symbol),
            mul_entry=mul_entry,
            offset_entry_ns=offset_entry_ns,
            mul_resp=mul_resp,
            offset_resp_ns=offset_resp_ns,
        )
        for feed in feeds
    ]

    initial_snapshot = _previous_day_snapshot(
        symbol=symbol,
        day=start_date,
        raw_dir=raw_dir,
        out_dir=out_dir,
        base_url=base_url,
        tick_size=tick_size,
        lot_size=lot_size,
        feed_latency_ns=feed_latency_ns,
        base_latency_ns=base_latency_ns,
    )
    return feeds, initial_snapshot, latencies


def _previous_day_snapshot(
    symbol: str,
    day: date,
    raw_dir: Path,
    out_dir: Path,
    base_url: str,
    tick_size: float,
    lot_size: float,
    feed_latency_ns: int,
    base_latency_ns: int,
) -> Path | None:
    """Build the previous day's EOD snapshot when its archives are reachable.

    Without it the first day opens on an empty book and the strategy sits idle
    until the feed has rebuilt enough depth.
    """
    previous = day - timedelta(days=1)
    date_str = previous.isoformat()

    existing = snapshot_path(out_dir, symbol, date_str)
    if existing.exists():
        return existing

    if not _raw_is_cached(raw_dir, symbol, date_str):
        return None

    feed = convert_day(
        symbol=symbol,
        date_str=date_str,
        raw_dir=raw_dir,
        out_dir=out_dir,
        base_url=base_url,
        feed_latency_ns=feed_latency_ns,
        base_latency_ns=base_latency_ns,
    )
    try:
        return build_eod_snapshot(
            feed_file=feed,
            out_dir=out_dir,
            symbol=symbol,
            date_str=date_str,
            tick_size=tick_size,
            lot_size=lot_size,
        )
    except RuntimeError:
        # create_last_snapshot replays the whole previous day to reach its final
        # book and raises a bare RuntimeError if that replay does not complete.
        # The seed only saves the first minutes of warm-up — each day's feed opens
        # with a full snapshot of its own — so losing it must not abort the run.
        logger.warning("backtest.seed_snapshot_unavailable", date=date_str)
        return None


def _raw_is_cached(raw_dir: Path, symbol: str, date_str: str) -> bool:
    """Whether a day's archives are already on disk.

    The seeding day is only built from what has been downloaded already: the day
    before the requested range may predate the available history, and pulling an
    extra day unasked would be a surprise download.
    """
    return all(path.exists() for path in raw_paths(raw_dir, symbol, date_str))


def _date_str_of(feed: Path, symbol: str) -> str:
    return feed.stem[len(symbol) + 1:]


def load_feed(feed_file: Path) -> np.ndarray:
    """Read a converted feed back as an event array — for inspection and tests."""
    with np.load(feed_file) as npz:
        return npz["data"]
