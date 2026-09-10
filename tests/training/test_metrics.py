import torch

from bot.training.metrics import compute_metrics

B, H, C = 8, 3, 3


class TestComputeMetrics:
    def test_keys_present(self) -> None:
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        valid = torch.ones(B, H, dtype=torch.bool)
        m = compute_metrics(logits, targets, valid, n_classes=C)
        for h in range(H):
            assert f"f1_h{h}" in m
            assert f"acc_h{h}" in m
        assert "f1_macro" in m

    def test_perfect_predictions(self) -> None:
        targets = torch.randint(0, C, (B, H))
        # One-hot logits that perfectly match targets
        logits = torch.zeros(B, H, C)
        for b in range(B):
            for h in range(H):
                logits[b, h, targets[b, h]] = 10.0
        valid = torch.ones(B, H, dtype=torch.bool)

        m = compute_metrics(logits, targets, valid, n_classes=C)
        for h in range(H):
            assert m[f"acc_h{h}"] == 1.0

    def test_flat_predictions_are_scored_not_skipped(self) -> None:
        """A correct flat call counts as correct — it is a class, not a gap."""
        targets = torch.ones(B, H, dtype=torch.long)  # every row genuinely flat
        logits = torch.zeros(B, H, C)
        logits[:, :, 1] = 10.0
        valid = torch.ones(B, H, dtype=torch.bool)
        m = compute_metrics(logits, targets, valid, n_classes=C)
        for h in range(H):
            assert m[f"acc_h{h}"] == 1.0

    def test_invalid_rows_are_excluded(self) -> None:
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        none_valid = torch.zeros(B, H, dtype=torch.bool)
        m = compute_metrics(logits, targets, none_valid, n_classes=C)
        for h in range(H):
            assert m[f"f1_h{h}"] == 0.0
            assert m[f"acc_h{h}"] == 0.0

    def test_f1_macro_is_mean(self) -> None:
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        valid = torch.ones(B, H, dtype=torch.bool)
        m = compute_metrics(logits, targets, valid, n_classes=C)
        expected = sum(m[f"f1_h{h}"] for h in range(H)) / H
        assert abs(m["f1_macro"] - expected) < 1e-6
