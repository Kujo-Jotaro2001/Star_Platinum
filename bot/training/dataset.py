import numpy as np
import torch
from torch.utils.data import Dataset


class LOBDataset(Dataset):
    """Windowed dataset for LOB + flow + ctx features.

    Each sample is a window of seq_len consecutive snapshots.
    Labels and flat_mask correspond to the LAST timestep of the window.

    Returns dict:
        ob:        [T, K, 4]  float32
        flow:      [T, 9]     float32
        ctx:       [T, 17]    float32  (zeros if use_ctx=False)
        labels:    [H]        int64
        flat_mask: [H]        bool
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
    ) -> None:
        self._ob = ob_raw
        self._flow = flow
        self._ctx = ctx
        self._labels = labels
        self._flat_mask = flat_mask
        self._seq_len = seq_len
        self._use_ctx = use_ctx
        # First valid index: seq_len (window starts at idx 0, label at idx seq_len-1)
        # But we need seq_len items, so valid label indices: [seq_len-1, N-1]
        self._n = len(ob_raw) - seq_len

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx
        end = idx + self._seq_len
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
        }
