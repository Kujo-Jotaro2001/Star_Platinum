from collections import deque

import numpy as np

# Feature indices for reference:
# 0: mark_price, 1: index_price, 2: basis_bps,
# 3: open_interest, 4: oi_change_pct_1h,
# 5: funding_rate, 6: funding_sign,
# 7: volume_24h, 8: price_24h_pct,
# 9: liq_buy_vol_1m, 10: liq_sell_vol_1m, 11: liq_imbalance_1m, 12: liq_count_1m,
# 13: ls_buy_ratio, 14: ls_sell_ratio, 15: ls_ratio,
# 16: open_interest_value
NUM_CONTEXT_FEATURES = 17

class ContextBuilder:
    """Stateful builder that forward-fills ticker/liquidation/LS ratio data
    and produces a 17-element context feature vector per snapshot.

    State carries over across daily DB boundaries.
    """

    def __init__(
        self,
        oi_history_window_ms: int = 3_600_000,
        liq_window_ms: int = 60_000,
    ) -> None:
        self._oi_history_window_ms = oi_history_window_ms
        self._liq_window_ms = liq_window_ms
        # Forward-fill state for ticker fields
        self._mark_price: float = 0.0
        self._index_price: float = 0.0
        self._open_interest: float = 0.0
        self._open_interest_value: float = 0.0
        self._funding_rate: float = 0.0
        self._volume_24h: float = 0.0
        self._price_24h_pct: float = 0.0

        # OI history for oi_change_pct_1h: (timestamp_ms, oi_value)
        self._oi_history: deque[tuple[int, float]] = deque()

        # Liquidation sliding window: (timestamp_ms, side, qty)
        self._liq_window: deque[tuple[int, str, float]] = deque()

        # Forward-fill state for long/short ratio
        self._ls_buy_ratio: float = 0.0
        self._ls_sell_ratio: float = 0.0

    def update_ticker(self, row: dict) -> None:
        """Update forward-fill state from a ticker_context DB row.

        NULL/None fields are skipped (forward-fill preserves previous value).
        """
        if row["mark_price"] is not None:
            self._mark_price = float(row["mark_price"])
        if row["index_price"] is not None:
            self._index_price = float(row["index_price"])
        if row["open_interest"] is not None:
            oi = float(row["open_interest"])
            self._open_interest = oi
            self._oi_history.append((row["timestamp_ms"], oi))
        if row["open_interest_value"] is not None:
            self._open_interest_value = float(row["open_interest_value"])
        if row["funding_rate"] is not None:
            self._funding_rate = float(row["funding_rate"])
        if row["volume_24h"] is not None:
            self._volume_24h = float(row["volume_24h"])
        if row["price_24h_pct"] is not None:
            self._price_24h_pct = float(row["price_24h_pct"])

    def update_liquidation(self, row: dict) -> None:
        """Push a liquidation event into the 1-minute sliding window."""
        self._liq_window.append((
            row["timestamp_ms"],
            row["side"],
            float(row["qty"]),
        ))

    def update_long_short_ratio(self, row: dict) -> None:
        """Update forward-fill state for long/short ratio."""
        self._ls_buy_ratio = float(row["buy_ratio"])
        self._ls_sell_ratio = float(row["sell_ratio"])

    def snapshot(self, timestamp_ms: int) -> np.ndarray:
        """Produce 17 context features at the given snapshot timestamp.

        Returns shape [17] float32.
        """
        # Evict stale OI history
        cutoff_oi = timestamp_ms - self._oi_history_window_ms
        while self._oi_history and self._oi_history[0][0] < cutoff_oi:
            self._oi_history.popleft()

        # OI change % over 1h
        if self._oi_history and self._oi_history[0][1] > 0:
            oi_1h_ago = self._oi_history[0][1]
            oi_change_pct = (self._open_interest - oi_1h_ago) / oi_1h_ago
        else:
            oi_change_pct = 0.0

        # Basis bps
        if self._index_price > 0:
            basis_bps = (self._mark_price - self._index_price) / self._index_price * 10_000
        else:
            basis_bps = 0.0

        # Funding sign
        funding_sign = 1.0 if self._funding_rate >= 0 else -1.0

        # Evict stale liquidations
        cutoff_liq = timestamp_ms - self._liq_window_ms
        while self._liq_window and self._liq_window[0][0] < cutoff_liq:
            self._liq_window.popleft()

        # Aggregate liquidations in window
        liq_buy_vol = 0.0
        liq_sell_vol = 0.0
        liq_count = 0
        for _, side, qty in self._liq_window:
            if side == "Buy":
                liq_buy_vol += qty
            else:
                liq_sell_vol += qty
            liq_count += 1
        liq_total = liq_buy_vol + liq_sell_vol
        liq_imbalance = (liq_buy_vol - liq_sell_vol) / (liq_total + 1e-8)

        # LS ratio
        ls_ratio = (
            self._ls_buy_ratio / self._ls_sell_ratio
            if self._ls_sell_ratio > 0
            else 1.0
        )

        return np.array(
            [
                self._mark_price,
                self._index_price,
                basis_bps,
                self._open_interest,
                oi_change_pct,
                self._funding_rate,
                funding_sign,
                self._volume_24h,
                self._price_24h_pct,
                liq_buy_vol,
                liq_sell_vol,
                liq_imbalance,
                float(liq_count),
                self._ls_buy_ratio,
                self._ls_sell_ratio,
                ls_ratio,
                self._open_interest_value,
            ],
            dtype=np.float32,
        )
