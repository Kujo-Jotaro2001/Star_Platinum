import numpy as np
import pytest
import torch

from bot.models.hybrid import HybridSignalModel
from bot.training.losses import MultiHorizonLoss
from bot.training.train import SignalLitModule, _compute_class_weights

B, T, K = 2, 10, 50
H, C = 3, 3


def _make_batch(use_ctx: bool = True) -> dict[str, torch.Tensor]:
    return {
        "ob": torch.randn(B, T, K, 4),
        "flow": torch.randn(B, T, 9),
        "ctx": torch.randn(B, T, 17) if use_ctx else torch.zeros(B, T, 17),
        "labels": torch.randint(0, C, (B, H)),
        "flat_mask": torch.zeros(B, H, dtype=torch.bool),
        "valid": torch.ones(B, dtype=torch.bool),
    }


def _make_module(use_ctx: bool = True) -> SignalLitModule:
    model = HybridSignalModel(
        ob_depth=K, d_model=32, d_ctx=16, n_lob_blocks=2,
        in_features_flow=9, in_features_ctx=17,
        flow_encoder="gru", gru_layers=1,
        n_classes=C, n_horizons=H, dropout=0.0, use_ctx=use_ctx,
    )
    loss_fn = MultiHorizonLoss(n_classes=C, use_focal=False, focal_gamma=2.0)
    return SignalLitModule(
        model=model, loss_fn=loss_fn,
        lr=1e-3, weight_decay=1e-4, use_ctx=use_ctx, n_classes=C,
    )


class TestSignalLitModule:
    def test_training_step_returns_loss(self) -> None:
        module = _make_module()
        batch = _make_batch()
        loss = module.training_step(batch, 0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_validation_step_runs(self) -> None:
        module = _make_module()
        batch = _make_batch()
        module.validation_step(batch, 0)

    def test_training_step_without_ctx(self) -> None:
        module = _make_module(use_ctx=False)
        batch = _make_batch(use_ctx=False)
        loss = module.training_step(batch, 0)
        assert loss.shape == ()

    def test_configure_optimizers(self) -> None:
        module = _make_module()
        optim = module.configure_optimizers()
        assert isinstance(optim, torch.optim.AdamW)


class TestClassWeights:
    """Weighting runs over the rows that carry a label, across all three classes."""

    def test_flat_is_weighted_like_any_other_class(self) -> None:
        """It is the majority class at every horizon, so leaving it unweighted
        while it is in the loss would let it dominate."""
        labels = np.array([[0], [2], [1], [1], [1]], dtype=np.int8)
        valid = np.ones(5, dtype=bool)
        w = _compute_class_weights(labels, valid, 3)
        assert w[1].item() < w[0].item()

    def test_absent_class_gets_a_neutral_weight(self) -> None:
        labels = np.array([[0], [2], [0], [2]], dtype=np.int8)
        valid = np.ones(4, dtype=bool)
        w = _compute_class_weights(labels, valid, 3)
        assert w[1].item() == pytest.approx(1.0)

    def test_balanced_classes_get_equal_weights(self) -> None:
        labels = np.array([[0], [2], [0], [2]], dtype=np.int8)
        valid = np.ones(4, dtype=bool)
        w = _compute_class_weights(labels, valid, 3)
        assert w[0].item() == pytest.approx(w[2].item())
        assert w[0].item() == pytest.approx(1.0)

    def test_rarer_class_gets_the_larger_weight(self) -> None:
        labels = np.array([[0]] * 9 + [[2]], dtype=np.int8)
        valid = np.ones(10, dtype=bool)
        w = _compute_class_weights(labels, valid, 3)
        assert w[2].item() > w[0].item()

    def test_invalid_rows_do_not_count(self) -> None:
        labels = np.array([[0]] * 9 + [[2]], dtype=np.int8)
        valid = np.ones(10, dtype=bool)
        valid[:8] = False  # one 0 and one 2 left, so the two balance
        w = _compute_class_weights(labels, valid, 3)
        assert w[0].item() == pytest.approx(w[2].item())

    def test_active_weights_average_to_one(self) -> None:
        labels = np.array([[0]] * 7 + [[2]] * 3, dtype=np.int8)
        valid = np.ones(10, dtype=bool)
        w = _compute_class_weights(labels, valid, 3).numpy()
        counts = np.array([7, 0, 3])
        present = counts > 0
        assert float((w[present] * counts[present]).sum() / counts.sum()) == pytest.approx(1.0, rel=1e-5)

    def test_no_usable_labels_leaves_every_weight_neutral(self) -> None:
        labels = np.ones((5, 1), dtype=np.int8)
        valid = np.zeros(5, dtype=bool)
        w = _compute_class_weights(labels, valid, 3)
        assert np.allclose(w.numpy(), 1.0)
