import gzip
import zipfile
from pathlib import Path

import pytest

from bot.data import historical_loader
from bot.data.historical_loader import (
    _read_orderbook,
    _read_trades,
    ensure_raw_files,
    raw_paths,
)

SYMBOL = "BTCUSDT"
DATE = "2026-03-01"
BASE_URL = "https://quote-saver.bycsi.com"


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self._payload


class FakeRequests:
    def __init__(self, payload: bytes = b"payload") -> None:
        self.payload = payload
        self.urls: list[str] = []

    def get(self, url: str, stream: bool = False, timeout: int = 0) -> FakeResponse:
        self.urls.append(url)
        return FakeResponse(self.payload)


@pytest.fixture
def fake_requests(monkeypatch) -> FakeRequests:
    fake = FakeRequests()
    monkeypatch.setattr(historical_loader, "requests", fake)
    return fake


class TestRawPaths:
    def test_names_match_the_upstream_archives(self, tmp_path: Path) -> None:
        ob, trades = raw_paths(tmp_path, SYMBOL, DATE)
        assert ob.name == f"{DATE}_{SYMBOL}_ob500.data.zip"
        assert trades.name == f"{SYMBOL}{DATE}.csv.gz"


class TestEnsureRawFiles:
    def test_downloads_both_archives(self, tmp_path: Path, fake_requests) -> None:
        ob, trades = ensure_raw_files(BASE_URL, tmp_path, SYMBOL, DATE)

        assert ob.exists() and trades.exists()
        assert len(fake_requests.urls) == 2
        assert fake_requests.urls[0].endswith(f"{DATE}_{SYMBOL}_ob500.data.zip")
        assert "/orderbook/linear/" in fake_requests.urls[0]
        assert "public.bybit.com/trading/" in fake_requests.urls[1]

    def test_cached_files_are_not_downloaded_again(
        self, tmp_path: Path, fake_requests
    ) -> None:
        ensure_raw_files(BASE_URL, tmp_path, SYMBOL, DATE)
        ensure_raw_files(BASE_URL, tmp_path, SYMBOL, DATE)
        assert len(fake_requests.urls) == 2

    def test_only_the_missing_archive_is_fetched(
        self, tmp_path: Path, fake_requests
    ) -> None:
        ob, _ = raw_paths(tmp_path, SYMBOL, DATE)
        ob.parent.mkdir(parents=True, exist_ok=True)
        ob.write_bytes(b"already here")

        ensure_raw_files(BASE_URL, tmp_path, SYMBOL, DATE)
        assert len(fake_requests.urls) == 1
        assert "public.bybit.com/trading/" in fake_requests.urls[0]

    def test_no_partial_file_is_left_behind(self, tmp_path: Path, fake_requests) -> None:
        ensure_raw_files(BASE_URL, tmp_path, SYMBOL, DATE)
        assert list(tmp_path.glob("*.part")) == []

    def test_an_interrupted_download_is_not_treated_as_cached(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        class Exploding(FakeRequests):
            def get(self, url: str, stream: bool = False, timeout: int = 0):
                self.urls.append(url)
                raise ConnectionError("dropped")

        monkeypatch.setattr(historical_loader, "requests", Exploding())
        with pytest.raises(ConnectionError):
            ensure_raw_files(BASE_URL, tmp_path, SYMBOL, DATE)

        ob, _ = raw_paths(tmp_path, SYMBOL, DATE)
        assert not ob.exists()


class TestReaders:
    def test_reads_lines_out_of_the_zip(self, tmp_path: Path) -> None:
        path = tmp_path / "ob.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("data", '{"a":1}\n\n{"a":2}\n')

        assert list(_read_orderbook(path)) == ['{"a":1}', '{"a":2}']

    def test_empty_zip_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.zip"
        with zipfile.ZipFile(path, "w"):
            pass
        with pytest.raises(RuntimeError, match="Empty zip"):
            list(_read_orderbook(path))

    def test_reads_lines_out_of_the_gzip(self, tmp_path: Path) -> None:
        path = tmp_path / "trades.csv.gz"
        with gzip.open(path, "wb") as f:
            f.write(b"a,b,c\n1,2,3\n\n4,5,6\n")

        assert list(_read_trades(path)) == ["a,b,c", "1,2,3", "4,5,6"]
