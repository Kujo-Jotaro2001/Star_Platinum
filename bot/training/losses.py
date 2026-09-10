import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHorizonLoss(nn.Module):
    """Cross-entropy loss across multiple prediction horizons.

    Rows without a usable label — reset zones and the array edges — are zeroed
    out before averaging. Flat is *not* excluded: it is one of the three classes
    the model is asked to predict, and a model given no flat gradient cannot
    learn to abstain, which leaves confidence thresholding as the only way to
    decline a trade.
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
        valid: Tensor,
    ) -> Tensor:
        """
        Args:
            logits:  [B, H, C]
            targets: [B, H] int64
            valid:   [B] or [B, H] bool — False = no usable label, exclude
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
        mask = valid.view(B, 1).expand(B, H) if valid.dim() == 1 else valid
        return (ce * mask).sum() / mask.sum().clamp(min=1)
