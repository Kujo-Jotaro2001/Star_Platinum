from datetime import date
from pathlib import Path

import numpy as np
import pytest

from bot.backtest import prepare
from bot.backtest.prepare import (
    convert_day,
    convert_range,
    feed_path,
    latency_path,
    prepare_range,
    snapshot_path,
)

SYMBOL = "BTCUSDT"
DATE = "2026-03-01"
BASE_URL = "https://quote-saver.bycsi.com"


def _raw_files(raw_dir: Path, symbol: str, date_str: str) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    ob = raw_dir / f"{date_str}_{symbol}_ob500.data.zip"
    trades = raw_dir / f"{symbol}{date_str}.csv.gz"
    ob.write_bytes(b"zip")
    trades.write_bytes(b"gz")
    return ob, trades


class FakeConverter:
    """Stands in for bybithistmktdata.convert — no network, no real archives."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def convert(self, **kwargs) -> np.ndarray:
        self.calls.append(kwargs)
        # numpy appends .npz when the name lacks it; mimic that so the caller's
        # rename is tested against the filename that would really be written.
        out = Path(kwargs["output_filename"])
        if out.suffix != ".npz":
            out = out.with_name(out.name + ".npz")
        out.write_bytes(b"npz")
        return np.zeros(0)


class TestPaths:
    def test_feed_snapshot_and_latency_are_distinct(self, tmp_path: Path) -> None:
        paths = {
            feed_path(tmp_path, SYMBOL, DATE),
            snapshot_path(tmp_path, SYMBOL, DATE),
            latency_path(tmp_path, SYMBOL, DATE),
        }
        assert len(paths) == 3

    def test_names_carry_symbol_and_date(self, tmp_path: Path) -> None:
        assert feed_path(tmp_path, SYMBOL, DATE).name == f"{SYMBOL}_{DATE}.npz"
        assert snapshot_path(tmp_path, SYMBOL, DATE).name == f"{SYMBOL}_{DATE}_eod.npz"


class TestConvertDay:
    def test_converts_cached_archives(self, tmp_path: Path, monkeypatch) -> None:
        raw_dir, out_dir = tmp_path / "raw", tmp_path / "hft"
        ob, trades = _raw_files(raw_dir, SYMBOL, DATE)
        fake = FakeConverter()
        monkeypatch.setattr(prepare, "bybithistmktdata", fake)

        out = convert_day(
            symbol=SYMBOL, date_str=DATE, raw_dir=raw_dir, out_dir=out_dir,
            base_url=BASE_URL, feed_latency_ns=10_000_000, base_latency_ns=0,
        )

        assert out == feed_path(out_dir, SYMBOL, DATE)
        assert out.exists()
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["depth_filename"] == str(ob)
        assert call["trades_filename"] == str(trades)
        assert call["feed_latency"] == 10_000_000

    def test_existing_feed_is_not_reconverted(self, tmp_path: Path, monkeypatch) -> None:
        raw_dir, out_dir = tmp_path / "raw", tmp_path / "hft"
        _raw_files(raw_dir, SYMBOL, DATE)
        out_dir.mkdir(parents=True)
        feed_path(out_dir, SYMBOL, DATE).write_bytes(b"already here")

        fake = FakeConverter()
        monkeypatch.setattr(prepare, "bybithistmktdata", fake)
        convert_day(
            symbol=SYMBOL, date_str=DATE, raw_dir=raw_dir, out_dir=out_dir,
            base_url=BASE_URL, feed_latency_ns=0, base_latency_ns=0,
        )
        assert fake.calls == []


class TestConvertRange:
    def test_one_feed_per_day_in_order(self, tmp_path: Path, monkeypatch) -> None:
        raw_dir, out_dir = tmp_path / "raw", tmp_path / "hft"
        for day in ("2026-03-01", "2026-03-02", "2026-03-03"):
            _raw_files(raw_dir, SYMBOL, day)
        monkeypatch.setattr(prepare, "bybithistmktdata", FakeConverter())

        feeds = convert_range(
            symbol=SYMBOL, start_date=date(2026, 3, 1), end_date=date(2026, 3, 3),
            raw_dir=raw_dir, out_dir=out_dir, base_url=BASE_URL,
            feed_latency_ns=0, base_latency_ns=0,
        )
        assert [f.name for f in feeds] == [
            f"{SYMBOL}_2026-03-01.npz",
            f"{SYMBOL}_2026-03-02.npz",
            f"{SYMBOL}_2026-03-03.npz",
        ]

    def test_inclusive_of_both_ends(self, tmp_path: Path, monkeypatch) -> None:
        raw_dir, out_dir = tmp_path / "raw", tmp_path / "hft"
        _raw_files(raw_dir, SYMBOL, "2026-03-01")
        monkeypatch.setattr(prepare, "bybithistmktdata", FakeConverter())

        feeds = convert_range(
            symbol=SYMBOL, start_date=date(2026, 3, 1), end_date=date(2026, 3, 1),
            raw_dir=raw_dir, out_dir=out_dir, base_url=BASE_URL,
            feed_latency_ns=0, base_latency_ns=0,
        )
        assert len(feeds) == 1


class TestPrepareRange:
    def _patch(self, monkeypatch) -> list[str]:
        built: list[str] = []
        monkeypatch.setattr(prepare, "bybithistmktdata", FakeConverter())
        monkeypatch.setattr(
            prepare, "generate_order_latency",
            lambda **kw: (Path(kw["output_file"]).write_bytes(b"lat"), built.append("latency"))[1],
        )
        monkeypatch.setattr(
            prepare, "create_last_snapshot",
            lambda *a, **kw: (
                Path(kw["output_snapshot_filename"]).write_bytes(b"snap"),
                built.append("snapshot"),
            )[1],
        )
        return built

    def test_builds_a_latency_file_per_feed(self, tmp_path: Path, monkeypatch) -> None:
        raw_dir, out_dir = tmp_path / "raw", tmp_path / "hft"
        for day in ("2026-03-01", "2026-03-02"):
            _raw_files(raw_dir, SYMBOL, day)
        self._patch(monkeypatch)

        feeds, _, latencies = prepare_range(
            symbol=SYMBOL, start_date=date(2026, 3, 1), end_date=date(2026, 3, 2),
            raw_dir=raw_dir, out_dir=out_dir, base_url=BASE_URL,
            tick_size=0.1, lot_size=0.001,
            feed_latency_ns=0, base_latency_ns=0,
            mul_entry=1.0, offset_entry_ns=0, mul_resp=1.0, offset_resp_ns=0,
        )
        assert len(latencies) == len(feeds) == 2
        assert latencies[0].name == f"{SYMBOL}_2026-03-01_latency.npz"

    def test_no_seed_snapshot_when_the_previous_day_is_absent(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        raw_dir, out_dir = tmp_path / "raw", tmp_path / "hft"
        _raw_files(raw_dir, SYMBOL, "2026-03-01")
        self._patch(monkeypatch)

        _, initial_snapshot, _ = prepare_range(
            symbol=SYMBOL, start_date=date(2026, 3, 1), end_date=date(2026, 3, 1),
            raw_dir=raw_dir, out_dir=out_dir, base_url=BASE_URL,
            tick_size=0.1, lot_size=0.001,
            feed_latency_ns=0, base_latency_ns=0,
            mul_entry=1.0, offset_entry_ns=0, mul_resp=1.0, offset_resp_ns=0,
        )
        assert initial_snapshot is None

    def test_seeds_from_the_previous_day_when_it_is_cached(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        raw_dir, out_dir = tmp_path / "raw", tmp_path / "hft"
        _raw_files(raw_dir, SYMBOL, "2026-02-28")
        _raw_files(raw_dir, SYMBOL, "2026-03-01")
        self._patch(monkeypatch)

        _, initial_snapshot, _ = prepare_range(
            symbol=SYMBOL, start_date=date(2026, 3, 1), end_date=date(2026, 3, 1),
            raw_dir=raw_dir, out_dir=out_dir, base_url=BASE_URL,
            tick_size=0.1, lot_size=0.001,
            feed_latency_ns=0, base_latency_ns=0,
            mul_entry=1.0, offset_entry_ns=0, mul_resp=1.0, offset_resp_ns=0,
        )
        assert initial_snapshot is not None
        assert initial_snapshot.name == f"{SYMBOL}_2026-02-28_eod.npz"


class TestAtomicConversion:
    """A killed conversion must not leave something that looks like a valid cache."""

    def test_partial_name_keeps_the_npz_suffix(self, tmp_path: Path, monkeypatch) -> None:
        raw_dir, out_dir = tmp_path / "raw", tmp_path / "hft"
        _raw_files(raw_dir, SYMBOL, DATE)
        fake = FakeConverter()
        monkeypatch.setattr(prepare, "bybithistmktdata", fake)

        convert_day(
            symbol=SYMBOL, date_str=DATE, raw_dir=raw_dir, out_dir=out_dir,
            base_url=BASE_URL, feed_latency_ns=0, base_latency_ns=0,
        )
        written = Path(fake.calls[0]["output_filename"])
        assert written.suffix == ".npz", "numpy would append another .npz otherwise"
        assert ".part" in written.name

    def test_final_file_is_in_place_and_no_partial_remains(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        raw_dir, out_dir = tmp_path / "raw", tmp_path / "hft"
        _raw_files(raw_dir, SYMBOL, DATE)
        monkeypatch.setattr(prepare, "bybithistmktdata", FakeConverter())

        out = convert_day(
            symbol=SYMBOL, date_str=DATE, raw_dir=raw_dir, out_dir=out_dir,
            base_url=BASE_URL, feed_latency_ns=0, base_latency_ns=0,
        )
        assert out.exists()
        assert list(out_dir.glob("*.part*")) == []

    def test_a_failed_conversion_leaves_no_feed_behind(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        raw_dir, out_dir = tmp_path / "raw", tmp_path / "hft"
        _raw_files(raw_dir, SYMBOL, DATE)

        class Exploding:
            def convert(self, **kwargs):
                Path(kwargs["output_filename"]).write_bytes(b"trunc")
                raise RuntimeError("killed mid-write")

        monkeypatch.setattr(prepare, "bybithistmktdata", Exploding())
        with pytest.raises(RuntimeError):
            convert_day(
                symbol=SYMBOL, date_str=DATE, raw_dir=raw_dir, out_dir=out_dir,
                base_url=BASE_URL, feed_latency_ns=0, base_latency_ns=0,
            )
        assert not feed_path(out_dir, SYMBOL, DATE).exists()
