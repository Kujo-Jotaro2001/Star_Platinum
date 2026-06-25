import torch
from torch import Tensor


def compute_metrics(
    logits: Tensor,
    targets: Tensor,
    flat_mask: Tensor,
    n_classes: int = 3,
) -> dict[str, float]:
    """Per-horizon F1 (macro) and accuracy, excluding flat-masked samples.

    Args:
        logits:    [B, H, C]
        targets:   [B, H] int64
        flat_mask: [B, H] bool — True = flat, exclude
        n_classes: number of classes

    Returns:
        dict with keys: f1_h0, f1_h1, ..., acc_h0, acc_h1, ..., f1_macro
    """
    preds = logits.argmax(dim=-1)  # [B, H]
    H = targets.shape[1]
    mask = ~flat_mask  # True = keep

    metrics: dict[str, float] = {}
    f1_sum = 0.0

    for h in range(H):
        m = mask[:, h]
        if m.sum() == 0:
            metrics[f"f1_h{h}"] = 0.0
            metrics[f"acc_h{h}"] = 0.0
            continue

        p = preds[:, h][m]
        t = targets[:, h][m]

        # Accuracy
        acc = (p == t).float().mean().item()
        metrics[f"acc_h{h}"] = acc

        # Macro F1
        f1 = _macro_f1(p, t, n_classes)
        metrics[f"f1_h{h}"] = f1
        f1_sum += f1

    metrics["f1_macro"] = f1_sum / H if H > 0 else 0.0
    return metrics


def _macro_f1(preds: Tensor, targets: Tensor, n_classes: int) -> float:
    """Compute macro F1 from flat preds and targets tensors."""
    f1_per_class: list[float] = []
    for c in range(n_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        precision = tp / (tp + fp).clamp(min=1e-8)
        recall = tp / (tp + fn).clamp(min=1e-8)
        if tp == 0:
            f1_per_class.append(0.0)
        else:
            f1_per_class.append((2 * precision * recall / (precision + recall)).item())
    return sum(f1_per_class) / n_classes
