import numpy as np


def compute_labels(
    mid_prices: np.ndarray,
    is_reset: np.ndarray,
    horizons: list[int] | None = None,
    alpha: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute multi-horizon labels from mid-price series.

    Method: smooth mid-price.
      prev = mean(mid_price[t-h : t])
      fut  = mean(mid_price[t : t+h])
      ret  = (fut - prev) / prev

    Args:
        mid_prices: shape [N] float64
        is_reset: shape [N] bool — True on reconnect snapshots
        horizons: list of horizon lengths in snapshots, default [1, 5, 10]
        alpha: threshold for flat classification

    Returns:
        labels:     shape [N, H] int8 — 0=down, 1=flat, 2=up
        flat_mask:  shape [N, H] bool — True where label is flat
        valid_mask: shape [N] bool — False for reset zones
    """
    if horizons is None:
        horizons = [1, 5, 10]

    n = len(mid_prices)
    h_count = len(horizons)
    max_h = max(horizons)

    labels = np.zeros((n, h_count), dtype=np.int8)
    flat_mask = np.zeros((n, h_count), dtype=bool)
    valid_mask = np.ones(n, dtype=bool)

    # Build reset zone mask: horizon snapshots before AND after each reset
    reset_indices = np.where(is_reset)[0]
    for idx in reset_indices:
        start_before = max(0, idx - max_h)
        end_after = min(n, idx + max_h + 1)
        valid_mask[start_before:end_after] = False

    for hi, h in enumerate(horizons):
        for t in range(n):
            if not valid_mask[t]:
                continue
            # Need h points before and h points after
            if t < h or t + h > n:
                valid_mask[t] = False
                continue

            prev = mid_prices[t - h : t].mean()
            fut = mid_prices[t : t + h].mean()

            if prev == 0:
                valid_mask[t] = False
                continue

            ret = (fut - prev) / prev

            if ret > alpha:
                labels[t, hi] = 2
            elif ret < -alpha:
                labels[t, hi] = 0
            else:
                labels[t, hi] = 1
                flat_mask[t, hi] = True

    return labels, flat_mask, valid_mask
