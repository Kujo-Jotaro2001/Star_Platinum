import numpy as np
import torch

from bot.training.dataset import LOBDataset

N, K, H = 200, 50, 3
SEQ_LEN = 10


def _make_dataset(use_ctx: bool = True) -> LOBDataset:
    ob = np.random.randn(N, K, 4).astype(np.float32)
    flow = np.random.randn(N, 9).astype(np.float32)
    ctx = np.random.randn(N, 17).astype(np.float32)
    labels = np.random.randint(0, 3, (N, H)).astype(np.int8)
    flat_mask = np.random.choice([True, False], (N, H))
    return LOBDataset(ob, flow, ctx, labels, flat_mask, SEQ_LEN, use_ctx)


class TestLOBDataset:
    def test_length(self) -> None:
        ds = _make_dataset()
        assert len(ds) == N - SEQ_LEN + 1

    def test_item_keys(self) -> None:
        ds = _make_dataset()
        item = ds[0]
        assert set(item.keys()) == {
            "ob", "flow", "ctx", "labels", "flat_mask", "valid",
        }

    def test_ob_shape(self) -> None:
        ds = _make_dataset()
        item = ds[0]
        assert item["ob"].shape == (SEQ_LEN, K, 4)
        assert item["ob"].dtype == torch.float32

    def test_flow_shape(self) -> None:
        ds = _make_dataset()
        item = ds[0]
        assert item["flow"].shape == (SEQ_LEN, 9)

    def test_ctx_shape_with_ctx(self) -> None:
        ds = _make_dataset(use_ctx=True)
        item = ds[0]
        assert item["ctx"].shape == (SEQ_LEN, 17)

    def test_ctx_zeros_without_ctx(self) -> None:
        ds = _make_dataset(use_ctx=False)
        item = ds[0]
        assert item["ctx"].shape == (SEQ_LEN, 17)
        assert (item["ctx"] == 0).all()

    def test_labels_shape(self) -> None:
        ds = _make_dataset()
        item = ds[0]
        assert item["labels"].shape == (H,)
        assert item["labels"].dtype == torch.int64

    def test_flat_mask_shape(self) -> None:
        ds = _make_dataset()
        item = ds[0]
        assert item["flat_mask"].shape == (H,)
        assert item["flat_mask"].dtype == torch.bool

    def test_ctx_is_actual_slice(self) -> None:
        """ctx at each timestep in the window comes from the actual data (option a)."""
        ob = np.arange(N * K * 4, dtype=np.float32).reshape(N, K, 4)
        flow = np.zeros((N, 9), dtype=np.float32)
        ctx = np.arange(N * 17, dtype=np.float32).reshape(N, 17)
        labels = np.zeros((N, H), dtype=np.int8)
        flat_mask = np.zeros((N, H), dtype=bool)
        ds = LOBDataset(ob, flow, ctx, labels, flat_mask, SEQ_LEN, use_ctx=True)

        item = ds[5]
        expected_ctx = torch.from_numpy(ctx[5:5 + SEQ_LEN].copy())
        assert torch.equal(item["ctx"], expected_ctx)

    def test_label_from_last_timestep(self) -> None:
        """Labels correspond to the last timestep of the window."""
        ob = np.zeros((N, K, 4), dtype=np.float32)
        flow = np.zeros((N, 9), dtype=np.float32)
        ctx = np.zeros((N, 17), dtype=np.float32)
        labels = np.arange(N * H, dtype=np.int8).reshape(N, H) % 3
        flat_mask = np.zeros((N, H), dtype=bool)
        ds = LOBDataset(ob, flow, ctx, labels, flat_mask, SEQ_LEN, use_ctx=True)

        idx = 7
        item = ds[idx]
        label_idx = idx + SEQ_LEN - 1
        expected = torch.from_numpy(labels[label_idx].copy()).long()
        assert torch.equal(item["labels"], expected)


class TestStride:
    def _arrays(self) -> tuple:
        rng = np.random.default_rng(0)
        return (
            rng.standard_normal((N, K, 4)).astype(np.float32),
            rng.standard_normal((N, 9)).astype(np.float32),
            rng.standard_normal((N, 17)).astype(np.float32),
            (np.arange(N * H).reshape(N, H) % 3).astype(np.int8),
            np.zeros((N, H), dtype=bool),
        )

    def _ds(self, stride: int, arrays: tuple | None = None) -> LOBDataset:
        ob, flow, ctx, labels, flat_mask = arrays or self._arrays()
        return LOBDataset(ob, flow, ctx, labels, flat_mask, SEQ_LEN, True, stride=stride)

    def test_stride_one_matches_the_default(self) -> None:
        assert len(self._ds(1)) == N - SEQ_LEN + 1

    def test_stride_reduces_the_window_count(self) -> None:
        assert len(self._ds(10)) == (N - SEQ_LEN + 1 + 9) // 10

    def test_windows_start_stride_apart(self) -> None:
        arrays = self._arrays()
        strided = self._ds(10, arrays)
        dense = self._ds(1, arrays)
        # sample j of the strided set is sample j*10 of the dense one
        assert torch.equal(strided[3]["ob"], dense[30]["ob"])
        assert torch.equal(strided[3]["labels"], dense[30]["labels"])

    def test_last_window_stays_in_bounds(self) -> None:
        ds = self._ds(7)
        item = ds[len(ds) - 1]
        assert item["ob"].shape[0] == SEQ_LEN

    def test_zero_stride_rejected(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="stride"):
            self._ds(0)
