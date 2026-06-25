import json
import sqlite3
from pathlib import Path

import numpy as np

from bot.features.context_features import NUM_CONTEXT_FEATURES
from bot.features.flow_features import NUM_FLOW_FEATURES
from bot.features.ob_serializer import OB_NUM_COLS
from bot.features.pipeline import run_pipeline

OB_DEPTH = 50


def _create_test_db(path: Path, n_snapshots: int = 100) -> None:
    """Create a synthetic daily DB with all required tables."""
    conn = sqlite3.connect(path)

    conn.execute("""
        CREATE TABLE snapshots (
            timestamp_ms INTEGER, is_reset INTEGER, bids TEXT, asks TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE trades (
            timestamp_ms INTEGER, trade_id TEXT, side TEXT, price TEXT, qty TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE ticker_context (
            timestamp_ms INTEGER, mark_price TEXT, index_price TEXT,
            open_interest TEXT, open_interest_value TEXT, funding_rate TEXT,
            next_funding_time INTEGER, price_24h_pct TEXT, prev_price_1h TEXT,
            volume_24h TEXT, turnover_24h TEXT, collected_at_ms INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE liquidations (
            timestamp_ms INTEGER, side TEXT, qty TEXT, price TEXT, collected_at_ms INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE long_short_ratio (
            timestamp_ms INTEGER UNIQUE, buy_ratio TEXT, sell_ratio TEXT, collected_at_ms INTEGER
        )
    """)

    base_ts = 1712300000000  # 2024-04-05
    base_price = 50000.0

    for i in range(n_snapshots):
        ts = base_ts + i * 100
        price = base_price + i * 0.5
        is_reset = 1 if i == 0 else 0
        bids = json.dumps([[str(price - 0.5), "1.0"], [str(price - 1.0), "2.0"]])
        asks = json.dumps([[str(price + 0.5), "1.5"], [str(price + 1.0), "2.5"]])
        conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?)",
            (ts, is_reset, bids, asks),
        )

    for i in range(n_snapshots * 2):
        ts = base_ts + i * 50
        side = "Buy" if i % 3 != 0 else "Sell"
        price = str(base_price + i * 0.25)
        conn.execute(
            "INSERT INTO trades VALUES (?, ?, ?, ?, ?)",
            (ts, f"trade-{i}", side, price, "0.1"),
        )

    for i in range(0, n_snapshots, 10):
        ts = base_ts + i * 100
        mark = str(base_price + i * 0.5 + 1.0)
        index = str(base_price + i * 0.5)
        conn.execute(
            "INSERT INTO ticker_context VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, mark, index, "12345.0", "617250000.0", "0.0001", None,
             "0.02", str(base_price - 100), "50000.0", "2500000000.0", ts + 1),
        )

    for i in range(5):
        ts = base_ts + i * 2000
        side = "Buy" if i % 2 == 0 else "Sell"
        conn.execute(
            "INSERT INTO liquidations VALUES (?, ?, ?, ?, ?)",
            (ts, side, "0.5", "50000.0", ts + 1),
        )

    conn.execute(
        "INSERT INTO long_short_ratio VALUES (?, ?, ?, ?)",
        (base_ts, "0.55", "0.45", base_ts + 1),
    )

    conn.commit()
    conn.close()


class TestPipeline:
    def test_end_to_end(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"

        _create_test_db(data_dir / "BTCUSDT_2024-04-05.db", n_snapshots=100)

        run_pipeline(
            data_dir=data_dir,
            symbol="BTCUSDT",
            start_date="2024-04-05",
            end_date="2024-04-05",
            ob_depth=OB_DEPTH,
            output_dir=output_dir,
        )

        assert (output_dir / "ob_raw.npy").exists()
        assert not (output_dir / "lob_features.npy").exists()
        assert (output_dir / "flow_features.npy").exists()
        assert (output_dir / "ctx_features.npy").exists()
        assert (output_dir / "labels.npy").exists()
        assert (output_dir / "flat_mask.npy").exists()
        assert (output_dir / "timestamps_ms.npy").exists()
        assert (output_dir / "is_reset.npy").exists()
        assert (output_dir / "normalizer_stats.npz").exists()

        n = 100
        ob = np.load(output_dir / "ob_raw.npy")
        assert ob.shape == (n, OB_DEPTH, OB_NUM_COLS)
        assert ob.dtype == np.float32

        flow = np.load(output_dir / "flow_features.npy")
        assert flow.shape == (n, NUM_FLOW_FEATURES)
        assert flow.dtype == np.float32

        ctx = np.load(output_dir / "ctx_features.npy")
        assert ctx.shape == (n, NUM_CONTEXT_FEATURES)
        assert ctx.dtype == np.float32

        labels = np.load(output_dir / "labels.npy")
        assert labels.shape == (n, 3)
        assert labels.dtype == np.int8

        flat_mask = np.load(output_dir / "flat_mask.npy")
        assert flat_mask.shape == (n, 3)
        assert flat_mask.dtype == bool

        timestamps = np.load(output_dir / "timestamps_ms.npy")
        assert timestamps.shape == (n,)
        assert timestamps.dtype == np.int64

        is_reset = np.load(output_dir / "is_reset.npy")
        assert is_reset.shape == (n,)
        assert is_reset.dtype == bool
        assert is_reset[0] is np.bool_(True)

    def test_multi_day(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"

        _create_test_db(data_dir / "BTCUSDT_2024-04-05.db", n_snapshots=50)
        _create_test_db(data_dir / "BTCUSDT_2024-04-06.db", n_snapshots=50)

        run_pipeline(
            data_dir=data_dir,
            symbol="BTCUSDT",
            start_date="2024-04-05",
            end_date="2024-04-06",
            ob_depth=OB_DEPTH,
            output_dir=output_dir,
        )

        ob = np.load(output_dir / "ob_raw.npy")
        assert ob.shape[0] == 100

    def test_ob_raw_values_sane(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"

        _create_test_db(data_dir / "BTCUSDT_2024-04-05.db", n_snapshots=30)

        run_pipeline(
            data_dir=data_dir,
            symbol="BTCUSDT",
            start_date="2024-04-05",
            end_date="2024-04-05",
            ob_depth=OB_DEPTH,
            output_dir=output_dir,
        )

        ob = np.load(output_dir / "ob_raw.npy")
        assert np.all(np.isfinite(ob))
        # Best bid/ask rows: price offset always 0
        assert (ob[:, 0, 0] == 0).all()
        assert (ob[:, 0, 2] == 0).all()
        # Price offsets never negative
        assert (ob[:, :, 0] >= 0).all()
        assert (ob[:, :, 2] >= 0).all()

    def test_normalizer_stats_loadable(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "output"

        _create_test_db(data_dir / "BTCUSDT_2024-04-05.db", n_snapshots=50)

        run_pipeline(
            data_dir=data_dir,
            symbol="BTCUSDT",
            start_date="2024-04-05",
            end_date="2024-04-05",
            ob_depth=OB_DEPTH,
            output_dir=output_dir,
        )

        from bot.features.normalizer import RollingNormalizer
        norm = RollingNormalizer.load(output_dir / "normalizer_stats.npz")

        raw = np.ones(NUM_FLOW_FEATURES, dtype=np.float32)
        result = norm.normalize_only(raw)
        assert result.shape == (NUM_FLOW_FEATURES,)
        assert np.all(np.isfinite(result))
