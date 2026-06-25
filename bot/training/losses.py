import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHorizonLoss(nn.Module):
    """Cross-entropy loss across multiple prediction horizons with flat_mask.

    flat_mask positions are zeroed out before averaging.
    Optional focal weighting: (1 - p_t)^gamma * CE.
    """

    def __init__(
        self,
        n_classes: int,
        use_focal: bool,
        focal_gamma: float,
        class_weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        self._n_classes = n_classes
        self._use_focal = use_focal
        self._focal_gamma = focal_gamma
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights: Tensor | None = None

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        flat_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            logits:    [B, H, C]
            targets:   [B, H] int64
            flat_mask: [B, H] bool — True = flat, exclude from loss
        Returns:
            scalar loss
        """
        B, H, C = logits.shape
        # Reshape for cross_entropy: [B*H, C] vs [B*H]
        logits_flat = logits.reshape(B * H, C)
        targets_flat = targets.reshape(B * H)

        ce = F.cross_entropy(
            logits_flat, targets_flat,
            weight=self.class_weights,
            reduction="none",
        )  # [B*H]

        if self._use_focal:
            probs = F.softmax(logits_flat.detach(), dim=-1)
            p_t = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - p_t) ** self._focal_gamma
            ce = ce * focal_weight

        ce = ce.view(B, H)  # [B, H]
        mask = ~flat_mask  # True = keep
        return (ce * mask).sum() / mask.sum().clamp(min=1)
