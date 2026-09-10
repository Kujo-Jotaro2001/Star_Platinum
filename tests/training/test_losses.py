import torch

from bot.training.losses import MultiHorizonLoss

B, H, C = 4, 3, 3


class TestMultiHorizonLoss:
    def test_output_is_scalar(self) -> None:
        loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        valid = torch.ones(B, H, dtype=torch.bool)
        loss = loss_fn(logits, targets, valid)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_invalid_rows_are_excluded(self) -> None:
        loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))

        all_keep = torch.ones(B, H, dtype=torch.bool)
        loss_all = loss_fn(logits, targets, all_keep)

        partial = torch.ones(B, H, dtype=torch.bool)
        partial[0, 0] = False
        partial[1, 2] = False
        loss_partial = loss_fn(logits, targets, partial)

        assert loss_all.item() != loss_partial.item()

    def test_nothing_valid_returns_zero(self) -> None:
        loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        none_valid = torch.zeros(B, H, dtype=torch.bool)
        loss = loss_fn(logits, targets, none_valid)
        assert loss.item() == 0.0

    def test_a_per_row_mask_covers_every_horizon(self) -> None:
        """`valid` is per row, not per horizon: a row without a usable label has
        none at any horizon."""
        loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        per_row = torch.tensor([True, False, True, True])
        expanded = per_row.view(B, 1).expand(B, H)
        assert loss_fn(logits, targets, per_row).item() == loss_fn(
            logits, targets, expanded
        ).item()

    def test_flat_labels_reach_the_loss(self) -> None:
        """The whole point of the change: a flat-labelled row is trained on.

        Excluding it leaves the model no gradient towards predicting flat, so it
        can only ever pick a direction and has no way to decline a trade.
        """
        loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        torch.manual_seed(0)
        logits = torch.randn(B, H, C, requires_grad=True)
        all_flat = torch.ones(B, H, dtype=torch.long)  # class 1 = flat
        valid = torch.ones(B, H, dtype=torch.bool)
        loss_fn(logits, all_flat, valid).backward()
        assert logits.grad.abs().sum().item() > 0

    def test_focal_loss_differs_from_vanilla(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        valid = torch.ones(B, H, dtype=torch.bool)

        vanilla = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        focal = MultiHorizonLoss(n_classes=C, use_focal=True, focal_gamma=2.0)

        loss_v = vanilla(logits, targets, valid)
        loss_f = focal(logits, targets, valid)
        # Focal should always be <= vanilla (down-weights confident predictions)
        assert loss_f.item() <= loss_v.item() + 1e-6

    def test_class_weights_applied(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(B, H, C)
        targets = torch.randint(0, C, (B, H))
        valid = torch.ones(B, H, dtype=torch.bool)

        no_w = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        with_w = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0,
                                  class_weights=torch.tensor([2.0, 1.0, 1.0]))

        loss_nw = no_w(logits, targets, valid)
        loss_ww = with_w(logits, targets, valid)
        assert loss_nw.item() != loss_ww.item()

    def test_gradient_flows(self) -> None:
        loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
        logits = torch.randn(B, H, C, requires_grad=True)
        targets = torch.randint(0, C, (B, H))
        flat_mask = torch.zeros(B, H, dtype=torch.bool)
        loss = loss_fn(logits, targets, flat_mask)
        loss.backward()
        assert logits.grad is not None
