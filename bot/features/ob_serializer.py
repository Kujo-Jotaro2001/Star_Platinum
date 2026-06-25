import numpy as np

OB_NUM_COLS = 4  # bid_offset, bid_size, ask_offset, ask_size


def serialize_ob_snapshot(
    bids: list[list[str]],
    asks: list[list[str]],
    ob_depth: int,
) -> np.ndarray:
    """Encode one order book snapshot as a raw tensor [K, 4] float32.

    Columns per row k:
      0: bid_price_offset = (best_bid - bids[k].price) / best_bid
      1: log1p(bid_qty)
      2: ask_price_offset = (asks[k].price - best_ask) / best_ask
      3: log1p(ask_qty)

    Fewer than ob_depth levels on a side → remaining rows padded with zeros.
    Encoding is self-normalizing; no z-score applied.
    """
    out = np.zeros((ob_depth, OB_NUM_COLS), dtype=np.float32)

    if bids:
        bid_arr = np.array(bids[:ob_depth], dtype=np.float64)
        bid_prices = bid_arr[:, 0]
        bid_qtys = bid_arr[:, 1]
        best_bid = bid_prices[0]
        if best_bid > 0:
            offsets = (best_bid - bid_prices) / best_bid
            out[: len(bid_prices), 0] = offsets.astype(np.float32)
        out[: len(bid_qtys), 1] = np.log1p(bid_qtys).astype(np.float32)

    if asks:
        ask_arr = np.array(asks[:ob_depth], dtype=np.float64)
        ask_prices = ask_arr[:, 0]
        ask_qtys = ask_arr[:, 1]
        best_ask = ask_prices[0]
        if best_ask > 0:
            offsets = (ask_prices - best_ask) / best_ask
            out[: len(ask_prices), 2] = offsets.astype(np.float32)
        out[: len(ask_qtys), 3] = np.log1p(ask_qtys).astype(np.float32)

    return out
