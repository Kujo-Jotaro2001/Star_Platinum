import torch

from bot.training.losses import MultiHorizonLoss

B, H, C = 4, 3, 3


class TestMultiHorizonLoss:
    def test_output_is_scalar(self) -> None:
        loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        flat_mask = torch.zeros(B, H, dtype=torch.bool)
        loss = loss_fn(logits, targets, flat_mask)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_flat_mask_excludes(self) -> None:
        loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))

        # All non-flat
        all_keep = torch.zeros(B, H, dtype=torch.bool)
        loss_all = loss_fn(logits, targets, all_keep)

        # Mask some out — loss should differ
        partial = torch.zeros(B, H, dtype=torch.bool)
        partial[0, 0] = True
        partial[1, 2] = True
        loss_partial = loss_fn(logits, targets, partial)

        assert loss_all.item() != loss_partial.item()

    def test_all_flat_returns_zero(self) -> None:
        loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        all_flat = torch.ones(B, H, dtype=torch.bool)
        loss = loss_fn(logits, targets, all_flat)
        assert loss.item() == 0.0

    def test_focal_loss_differs_from_vanilla(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        flat_mask = torch.zeros(B, H, dtype=torch.bool)

        vanilla = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        focal = MultiHorizonLoss(n_classes=C, use_focal=True, focal_gamma=2.0)

        loss_v = vanilla(logits, targets, flat_mask)
        loss_f = focal(logits, targets, flat_mask)
        # Focal should always be <= vanilla (down-weights confident predictions)
        assert loss_f.item() <= loss_v.item() + 1e-6

    def test_class_weights_applied(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        flat_mask = torch.zeros(B, H, dtype=torch.bool)

        no_w = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        with_w = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0,
                                  class_weights=torch.tensor([2.0, 1.0, 1.0]))

        loss_nw = no_w(logits, targets, flat_mask)
        loss_ww = with_w(logits, targets, flat_mask)
        assert loss_nw.item() != loss_ww.item()

    def test_gradient_flows(self) -> None:
        loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        logits = torch.randn(B, H, C, requires_grad=True)
        targets = torch.randint(0, C, (B, H))
        flat_mask = torch.zeros(B, H, dtype=torch.bool)
        loss = loss_fn(logits, targets, flat_mask)
        loss.backward()
        assert logits.grad is not None
