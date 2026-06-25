import numpy as np

# Feature indices for reference:
# 0: buy_volume, 1: sell_volume, 2: signed_volume,
# 3: trade_count, 4: ofi, 5: max_trade_size, 6: mean_trade_size,
# 7: buy_count, 8: sell_count
NUM_FLOW_FEATURES = 9

_ZEROS = np.zeros(NUM_FLOW_FEATURES, dtype=np.float32)


def extract_flow_features(trades: list[dict]) -> np.ndarray:
    """Extract 9 trade flow features from trades in one 100ms bucket.

    Each trade dict has keys: side, price, qty (all str).
    Volumes have log1p applied.
    Returns shape [9] float32.
    """
    if not trades:
        return _ZEROS.copy()

    buy_vol = 0.0
    sell_vol = 0.0
    buy_count = 0
    sell_count = 0
    max_size = 0.0

    for t in trades:
        qty = float(t["qty"])
        if t["side"] == "Buy":
            buy_vol += qty
            buy_count += 1
        else:
            sell_vol += qty
            sell_count += 1
        if qty > max_size:
            max_size = qty

    total_count = buy_count + sell_count
    signed_vol = buy_vol - sell_vol
    total_vol = buy_vol + sell_vol
    ofi = signed_vol / (total_vol + 1e-8)
    mean_size = total_vol / total_count if total_count > 0 else 0.0

    return np.array(
        [
            np.log1p(buy_vol),
            np.log1p(sell_vol),
            np.log1p(abs(signed_vol)) * np.sign(signed_vol),
            float(total_count),
            ofi,
            np.log1p(max_size),
            np.log1p(mean_size),
            float(buy_count),
            float(sell_count),
        ],
        dtype=np.float32,
    )
