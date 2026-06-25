import numpy as np

from bot.features.normalizer import RollingNormalizer
from bot.inference import OnlinePreprocessor


def _bids() -> list[list[str]]:
    return [["100.0", "1.0"], ["99.9", "2.0"]]


def _asks() -> list[list[str]]:
    return [["100.1", "1.5"], ["100.2", "2.5"]]


class TestOnlinePreprocessor:
    def test_output_shapes_match_contracts(self) -> None:
        preprocessor = OnlinePreprocessor(
            seq_len=3,
            ob_depth=2,
            flow_normalizer=RollingNormalizer(num_features=9, window=5),
        )
        trades = [
            {"side": "Buy", "price": "100.0", "qty": "0.5"},
            {"side": "Sell", "price": "100.1", "qty": "0.25"},
        ]

        frame = preprocessor.append_snapshot(
            timestamp_ms=1_800_000_000_000,
            bids=_bids(),
            asks=_asks(),
            trades=trades,
        )

        assert frame.ob.shape == (2, 4)
        assert frame.ob.dtype == np.float32
        assert frame.flow.shape == (9,)
        assert frame.flow.dtype == np.float32
        assert frame.ctx.shape == (17,)
        assert frame.ctx.dtype == np.float32

    def test_windows_match_model_contracts_after_warmup(self) -> None:
        preprocessor = OnlinePreprocessor(
            seq_len=3,
            ob_depth=2,
            flow_normalizer=RollingNormalizer(num_features=9, window=5),
        )

        for i in range(3):
            preprocessor.append_snapshot(
                timestamp_ms=1_800_000_000_000 + i * 100,
                bids=_bids(),
                asks=_asks(),
                trades=[],
            )

        ob, flow, ctx = preprocessor.windows()
        assert ob.shape == (3, 2, 4)
        assert flow.shape == (3, 9)
        assert ctx.shape == (3, 17)
