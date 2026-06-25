import torch

from bot.models.hybrid import HybridSignalModel
from bot.training.losses import MultiHorizonLoss
from bot.training.train import SignalLitModule

B, T, K = 2, 10, 50
H, C = 3, 3


def _make_batch(use_ctx: bool = True) -> dict[str, torch.Tensor]:
    return {
        "ob": torch.randn(B, T, K, 4),
        "flow": torch.randn(B, T, 9),
        "ctx": torch.randn(B, T, 17) if use_ctx else torch.zeros(B, T, 17),
        "labels": torch.randint(0, C, (B, H)),
        "flat_mask": torch.zeros(B, H, dtype=torch.bool),
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
