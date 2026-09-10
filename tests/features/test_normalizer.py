import numpy as np
import pytest

from bot.features.normalizer import RollingNormalizer


class TestRollingNormalizer:
    def test_output_shape(self) -> None:
        norm = RollingNormalizer(num_features=3, window=5)
        raw = np.array([1.0, 2.0, 3.0])
        result = norm.update_and_normalize(raw)
        assert result.shape == (3,)
        assert result.dtype == np.float32

    def test_first_sample_zero(self) -> None:
        """First sample: mean=value, std=1 → z-score=0."""
        norm = RollingNormalizer(num_features=2, window=5)
        result = norm.update_and_normalize(np.array([10.0, 20.0]))
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])

    def test_known_zscore(self) -> None:
        norm = RollingNormalizer(num_features=1, window=100)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        results = []
        for v in values:
            results.append(norm.update_and_normalize(np.array([v])))

        # After 5 values [1,2,3,4,5]: mean=3, std≈1.414
        last = results[-1][0]
        expected = (5.0 - 3.0) / np.std([1, 2, 3, 4, 5])
        assert last == pytest.approx(expected, rel=1e-5)

    def test_rolling_window_eviction(self) -> None:
        norm = RollingNormalizer(num_features=1, window=3)
        for v in [10.0, 20.0, 30.0]:
            norm.update_and_normalize(np.array([v]))

        # Buffer is [10, 20, 30]. Now add 40 — evicts 10.
        result = norm.update_and_normalize(np.array([40.0]))
        # Buffer is [20, 30, 40]. mean=30, std≈8.165
        expected = (40.0 - 30.0) / np.std([20, 30, 40])
        assert result[0] == pytest.approx(expected, rel=1e-5)

    def test_normalize_only_frozen(self) -> None:
        norm = RollingNormalizer(num_features=1, window=100)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            norm.update_and_normalize(np.array([v]))

        # normalize_only should use current stats without updating
        result1 = norm.normalize_only(np.array([6.0]))
        result2 = norm.normalize_only(np.array([6.0]))
        np.testing.assert_array_equal(result1, result2)

        # Stats should still be from [1,2,3,4,5]
        expected = (6.0 - 3.0) / np.std([1, 2, 3, 4, 5])
        assert result1[0] == pytest.approx(expected, rel=1e-5)

    def test_save_load_roundtrip(self, tmp_path) -> None:
        norm = RollingNormalizer(num_features=2, window=5)
        for i in range(7):
            norm.update_and_normalize(np.array([float(i), float(i * 2)]))

        path = tmp_path / "normalizer_stats.npz"
        norm.save(path)

        loaded = RollingNormalizer.load(path)

        raw = np.array([10.0, 20.0])
        r1 = norm.normalize_only(raw)
        r2 = loaded.normalize_only(raw)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_save_load_preserves_buffer(self, tmp_path) -> None:
        norm = RollingNormalizer(num_features=1, window=3)
        for v in [1.0, 2.0, 3.0]:
            norm.update_and_normalize(np.array([v]))

        path = tmp_path / "stats.npz"
        norm.save(path)
        loaded = RollingNormalizer.load(path)

        # Continue updating from loaded — should evict oldest
        r_orig = norm.update_and_normalize(np.array([4.0]))
        r_loaded = loaded.update_and_normalize(np.array([4.0]))
        np.testing.assert_array_almost_equal(r_orig, r_loaded)

    def test_constant_values_std_one(self) -> None:
        """Constant input should have std clamped to 1.0, not 0."""
        norm = RollingNormalizer(num_features=1, window=5)
        for _ in range(5):
            result = norm.update_and_normalize(np.array([42.0]))
        # mean=42, std clamped to 1.0 → z-score=0
        assert result[0] == pytest.approx(0.0)


class TestIncrementalMatchesExact:
    """The running sums must agree with recomputing the window from scratch."""

    def _exact(self, rows: np.ndarray, window: int, i: int) -> tuple[float, float]:
        w = rows[max(0, i - window + 1):i + 1]
        if len(w) < 2:
            return (w[0][0] if len(w) else 0.0), 1.0
        std = w[:, 0].std()
        return w[:, 0].mean(), (std if std > 1e-10 else 1.0)

    def test_matches_a_full_recompute_at_every_step(self) -> None:
        rng = np.random.default_rng(0)
        window = 50
        rows = rng.standard_normal((400, 1)) * 5 + 100
        norm = RollingNormalizer(num_features=1, window=window)
        for i, row in enumerate(rows):
            got = norm.update_and_normalize(row)
            mean, std = self._exact(rows, window, i)
            assert got[0] == pytest.approx((row[0] - mean) / std, rel=1e-4, abs=1e-4)

    def test_survives_an_exact_refresh(self) -> None:
        # cross the refresh boundary and confirm nothing shifts
        rng = np.random.default_rng(1)
        window = 64
        n = 9000
        rows = rng.standard_normal((n, 2)) * 3
        norm = RollingNormalizer(num_features=2, window=window)
        for row in rows:
            norm.update_and_normalize(row)
        tail = rows[-window:]
        np.testing.assert_allclose(norm._mean, tail.mean(axis=0), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(norm._std, tail.std(axis=0), rtol=1e-9, atol=1e-9)

    def test_multi_feature_windows_are_independent(self) -> None:
        norm = RollingNormalizer(num_features=2, window=3)
        for a, b in ((1.0, 100.0), (2.0, 200.0), (3.0, 300.0), (4.0, 400.0)):
            norm.update_and_normalize(np.array([a, b]))
        np.testing.assert_allclose(norm._mean, [3.0, 300.0])

    def test_save_load_after_many_updates(self, tmp_path) -> None:
        rng = np.random.default_rng(2)
        norm = RollingNormalizer(num_features=3, window=32)
        for _ in range(500):
            norm.update_and_normalize(rng.standard_normal(3))
        path = tmp_path / "s.npz"
        norm.save(path)
        loaded = RollingNormalizer.load(path)

        probe = rng.standard_normal(3)
        np.testing.assert_allclose(
            norm.normalize_only(probe), loaded.normalize_only(probe)
        )
        # and continuing from the restored state must track the original
        step = rng.standard_normal(3)
        np.testing.assert_allclose(
            norm.update_and_normalize(step), loaded.update_and_normalize(step),
            rtol=1e-5, atol=1e-5,
        )
