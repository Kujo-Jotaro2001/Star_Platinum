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

    Window means come from a prefix sum: a day is ~860k snapshots and the
    horizons run to thousands of samples, so evaluating the windows one at a
    time is quadratic enough to dominate the whole pipeline.

    Args:
        mid_prices: shape [N] float64
        is_reset: shape [N] bool — True on reconnect snapshots
        horizons: list of horizon lengths in snapshots, default [1, 5, 10]
        alpha: threshold for flat classification

    Returns:
        labels:     shape [N, H] int8 — 0=down, 1=flat, 2=up
        flat_mask:  shape [N, H] bool — True where label is flat
        valid_mask: shape [N] bool — False for reset zones and unusable edges
    """
    if horizons is None:
        horizons = [1, 5, 10]

    n = len(mid_prices)
    h_count = len(horizons)
    max_h = max(horizons)

    labels = np.zeros((n, h_count), dtype=np.int8)
    flat_mask = np.zeros((n, h_count), dtype=bool)
    valid_mask = np.ones(n, dtype=bool)

    # Reset zone: max_h snapshots either side of a reconnect.
    for idx in np.where(is_reset)[0]:
        valid_mask[max(0, idx - max_h):min(n, idx + max_h + 1)] = False

    # The edges where the widest horizon has no room, matching the per-horizon
    # checks the scalar form applied cumulatively.
    valid_mask[:max_h] = False
    if max_h > 0:
        valid_mask[n - max_h + 1:] = False

    prefix = np.concatenate(([0.0], np.cumsum(mid_prices, dtype=np.float64)))
    t = np.arange(n)

    for hi, h in enumerate(horizons):
        prev = np.zeros(n, dtype=np.float64)
        fut = np.zeros(n, dtype=np.float64)

        lo, hi_end = h, n - h + 1
        if lo >= hi_end:
            valid_mask[:] = False
            continue

        idx = t[lo:hi_end]
        prev[idx] = (prefix[idx] - prefix[idx - h]) / h
        fut[idx] = (prefix[idx + h] - prefix[idx]) / h

        usable = valid_mask.copy()
        usable[:lo] = False
        usable[hi_end:] = False
        usable &= prev != 0
        valid_mask &= usable

        ret = np.zeros(n, dtype=np.float64)
        np.divide(fut - prev, prev, out=ret, where=usable)

        labels[usable & (ret > alpha), hi] = 2
        labels[usable & (ret < -alpha), hi] = 0
        flat = usable & (ret <= alpha) & (ret >= -alpha)
        labels[flat, hi] = 1
        flat_mask[flat, hi] = True

    return labels, flat_mask, valid_mask
