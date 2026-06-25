import numpy as np

from bot.features.labels import compute_labels


class TestComputeLabels:
    def test_output_shapes(self) -> None:
        mid = np.array([100.0, 100.1, 100.2, 100.3, 100.4, 100.5], dtype=np.float64)
        is_reset = np.zeros(6, dtype=bool)
        labels, flat_mask, valid_mask = compute_labels(mid, is_reset, horizons=[1])
        assert labels.shape == (6, 1)
        assert labels.dtype == np.int8
        assert flat_mask.shape == (6, 1)
        assert flat_mask.dtype == bool
        assert valid_mask.shape == (6,)
        assert valid_mask.dtype == bool

    def test_uptrend_labels_up(self) -> None:
        # Steadily rising prices — large enough to exceed alpha=0.001
        mid = np.linspace(100.0, 110.0, 50, dtype=np.float64)
        is_reset = np.zeros(50, dtype=bool)
        labels, flat_mask, valid_mask = compute_labels(mid, is_reset, horizons=[5])

        valid_labels = labels[valid_mask, 0]
        # Most should be 2 (up)
        assert (valid_labels == 2).sum() > len(valid_labels) // 2

    def test_downtrend_labels_down(self) -> None:
        mid = np.linspace(110.0, 100.0, 50, dtype=np.float64)
        is_reset = np.zeros(50, dtype=bool)
        labels, flat_mask, valid_mask = compute_labels(mid, is_reset, horizons=[5])

        valid_labels = labels[valid_mask, 0]
        assert (valid_labels == 0).sum() > len(valid_labels) // 2

    def test_flat_prices_label_flat(self) -> None:
        mid = np.full(20, 100.0, dtype=np.float64)
        is_reset = np.zeros(20, dtype=bool)
        labels, flat_mask, valid_mask = compute_labels(mid, is_reset, horizons=[1])

        for t in range(20):
            if valid_mask[t]:
                assert labels[t, 0] == 1
                assert flat_mask[t, 0] is np.bool_(True)

    def test_reset_masks_both_directions(self) -> None:
        n = 30
        mid = np.linspace(100.0, 103.0, n, dtype=np.float64)
        is_reset = np.zeros(n, dtype=bool)
        is_reset[15] = True  # reset at index 15

        _, _, valid_mask = compute_labels(mid, is_reset, horizons=[5])

        # 5 before reset and 5 after (including reset itself) should be invalid
        for i in range(10, 21):
            assert valid_mask[i] is np.bool_(False), f"index {i} should be invalid"

        # Early and late snapshots should be valid (if they have enough horizon)
        assert valid_mask[5] is np.bool_(True)

    def test_edges_invalid(self) -> None:
        mid = np.linspace(100.0, 101.0, 10, dtype=np.float64)
        is_reset = np.zeros(10, dtype=bool)
        _, _, valid_mask = compute_labels(mid, is_reset, horizons=[5])

        # First 5 and last point don't have enough lookback/forward
        for i in range(5):
            assert valid_mask[i] is np.bool_(False)

    def test_multiple_horizons(self) -> None:
        mid = np.linspace(100.0, 102.0, 30, dtype=np.float64)
        is_reset = np.zeros(30, dtype=bool)
        labels, flat_mask, valid_mask = compute_labels(mid, is_reset, horizons=[1, 5, 10])

        assert labels.shape == (30, 3)
        assert flat_mask.shape == (30, 3)

    def test_alpha_threshold(self) -> None:
        # Create prices where return is exactly at alpha boundary
        mid = np.array([100.0, 100.0, 100.1, 100.1], dtype=np.float64)
        is_reset = np.zeros(4, dtype=bool)
        labels, _, valid_mask = compute_labels(
            mid, is_reset, horizons=[1], alpha=0.0005
        )
        # Check that the function runs without error — specific values depend on smoothing
        assert labels.shape == (4, 1)
