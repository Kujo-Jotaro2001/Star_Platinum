import numpy as np
import torch
from torch.utils.data import Dataset


class LOBDataset(Dataset):
    """Windowed dataset for LOB + flow + ctx features.

    Each sample is a window of seq_len consecutive snapshots.
    Labels, flat_mask and valid correspond to the LAST timestep of the window.

    `stride` skips window starts. Consecutive windows at a 100 ms cadence overlap
    by more than 98%, so training on every one costs an order of magnitude of
    compute for almost no extra information.

    Returns dict:
        ob:        [T, K, 4]  float32
        flow:      [T, 9]     float32
        ctx:       [T, 17]    float32  (zeros if use_ctx=False)
        labels:    [H]        int64
        flat_mask: [H]        bool  — the label is flat; a class, not an exclusion
        valid:     []         bool  — the row has a usable label at all
    """

    def __init__(
        self,
        ob_raw: np.ndarray,
        flow: np.ndarray,
        ctx: np.ndarray,
        labels: np.ndarray,
        flat_mask: np.ndarray,
        seq_len: int,
        use_ctx: bool,
        stride: int = 1,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")
        self._ob = ob_raw
        self._flow = flow
        self._ctx = ctx
        self._labels = labels
        self._flat_mask = flat_mask
        self._valid_mask = (
            valid_mask if valid_mask is not None
            else np.ones(len(labels), dtype=bool)
        )
        self._seq_len = seq_len
        self._use_ctx = use_ctx
        self._stride = stride
        windows = len(ob_raw) - seq_len + 1
        self._n = max(0, (windows + stride - 1) // stride)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self._stride
        end = start + self._seq_len
        label_idx = end - 1  # last timestep of window

        ob = torch.from_numpy(self._ob[start:end].copy())
        flow = torch.from_numpy(self._flow[start:end].copy())

        if self._use_ctx:
            ctx = torch.from_numpy(self._ctx[start:end].copy())
        else:
            ctx = torch.zeros(self._seq_len, self._ctx.shape[1], dtype=torch.float32)

        labels = torch.from_numpy(self._labels[label_idx].copy()).long()
        flat_mask = torch.from_numpy(self._flat_mask[label_idx].copy())

        return {
            "ob": ob,
            "flow": flow,
            "ctx": ctx,
            "labels": labels,
            "flat_mask": flat_mask,
            "valid": torch.tensor(bool(self._valid_mask[label_idx])),
        }
