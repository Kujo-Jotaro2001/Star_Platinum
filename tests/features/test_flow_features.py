import numpy as np
import pytest

from bot.features.flow_features import NUM_FLOW_FEATURES, extract_flow_features


def _trade(side: str, qty: str) -> dict:
    return {"side": side, "price": "100.0", "qty": qty}


class TestFlowFeatures:
    def test_output_shape(self) -> None:
        result = extract_flow_features([_trade("Buy", "1.0")])
        assert result.shape == (NUM_FLOW_FEATURES,)
        assert result.dtype == np.float32

    def test_empty_bucket_all_zeros(self) -> None:
        result = extract_flow_features([])
        np.testing.assert_array_equal(result, np.zeros(NUM_FLOW_FEATURES, dtype=np.float32))

    def test_buy_volume_log1p(self) -> None:
        result = extract_flow_features([_trade("Buy", "2.0")])
        assert result[0] == pytest.approx(np.log1p(2.0))  # buy_volume
        assert result[1] == pytest.approx(0.0)  # sell_volume

    def test_sell_volume(self) -> None:
        result = extract_flow_features([_trade("Sell", "3.0")])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(np.log1p(3.0))

    def test_signed_volume_positive(self) -> None:
        result = extract_flow_features([
            _trade("Buy", "5.0"),
            _trade("Sell", "2.0"),
        ])
        # signed = 5 - 2 = 3, log1p(3) * sign(3) = log1p(3)
        assert result[2] == pytest.approx(np.log1p(3.0))

    def test_signed_volume_negative(self) -> None:
        result = extract_flow_features([
            _trade("Buy", "1.0"),
            _trade("Sell", "4.0"),
        ])
        # signed = 1 - 4 = -3, log1p(3) * -1
        assert result[2] == pytest.approx(-np.log1p(3.0))

    def test_trade_count(self) -> None:
        result = extract_flow_features([
            _trade("Buy", "1.0"),
            _trade("Buy", "2.0"),
            _trade("Sell", "3.0"),
        ])
        assert result[3] == pytest.approx(3.0)
        assert result[7] == pytest.approx(2.0)  # buy_count
        assert result[8] == pytest.approx(1.0)  # sell_count

    def test_ofi(self) -> None:
        result = extract_flow_features([
            _trade("Buy", "7.0"),
            _trade("Sell", "3.0"),
        ])
        # ofi = (7-3) / (7+3 + 1e-8) ≈ 0.4
        assert result[4] == pytest.approx(4.0 / (10.0 + 1e-8), rel=1e-6)

    def test_max_trade_size(self) -> None:
        result = extract_flow_features([
            _trade("Buy", "1.0"),
            _trade("Buy", "5.0"),
            _trade("Sell", "3.0"),
        ])
        assert result[5] == pytest.approx(np.log1p(5.0))

    def test_mean_trade_size(self) -> None:
        result = extract_flow_features([
            _trade("Buy", "2.0"),
            _trade("Sell", "4.0"),
        ])
        # mean = (2+4)/2 = 3
        assert result[6] == pytest.approx(np.log1p(3.0))
