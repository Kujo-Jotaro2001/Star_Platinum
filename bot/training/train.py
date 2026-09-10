from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import structlog
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from bot.models.hybrid import HybridSignalModel
from bot.training.dataset import LOBDataset
from bot.training.losses import MultiHorizonLoss
from bot.training.metrics import compute_metrics

logger = structlog.get_logger()


class SignalLitModule(pl.LightningModule):
    def __init__(
        self,
        model: HybridSignalModel,
        loss_fn: MultiHorizonLoss,
        lr: float,
        weight_decay: float,
        use_ctx: bool,
        n_classes: int,
        arch: dict | None = None,
    ) -> None:
        super().__init__()
        # The architecture travels with the weights, so a checkpoint can be
        # loaded without also being told how it was built. `model` and `loss_fn`
        # are live modules and are saved by Lightning as weights already.
        self.save_hyperparameters({"arch": arch or {}})
        self.model = model
        self.loss_fn = loss_fn
        self._lr = lr
        self._weight_decay = weight_decay
        self._use_ctx = use_ctx
        self._n_classes = n_classes

    def forward(self, ob: Tensor, flow: Tensor, ctx: Tensor | None = None) -> Tensor:
        return self.model(ob, flow, ctx)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits = self.model(
            batch["ob"],
            batch["flow"],
            batch["ctx"] if self._use_ctx else None,
        )
        loss = self.loss_fn(logits, batch["labels"], batch["valid"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        logits = self.model(
            batch["ob"],
            batch["flow"],
            batch["ctx"] if self._use_ctx else None,
        )
        loss = self.loss_fn(logits, batch["labels"], batch["valid"])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        metrics = compute_metrics(
            logits, batch["labels"], batch["valid"], self._n_classes,
        )
        for k, v in metrics.items():
            self.log(f"val_{k}", v, prog_bar=(k == "f1_macro"), sync_dist=True)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        logits = self.model(
            batch["ob"],
            batch["flow"],
            batch["ctx"] if self._use_ctx else None,
        )
        metrics = compute_metrics(
            logits, batch["labels"], batch["valid"], self._n_classes,
        )
        for k, v in metrics.items():
            self.log(f"test_{k}", v, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )


def _compute_class_weights(
    labels: np.ndarray, valid_mask: np.ndarray, n_classes: int
) -> Tensor:
    """Inverse-frequency class weights over the rows that carry a label.

    All three classes are counted, flat included — it is the majority class at
    every horizon, and leaving it out of the weighting while it is in the loss
    would let it dominate. A class with no examples at all keeps a weight of 1
    rather than a derived one: dividing by a zero count yields a multiplier in
    the millions. Normalisation runs over the classes that are present, so the
    active weights average to one.
    """
    valid_labels = labels[valid_mask].ravel()
    counts = np.bincount(valid_labels, minlength=n_classes).astype(np.float64)
    present = counts > 0

    weights = np.ones(n_classes, dtype=np.float64)
    if present.any():
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.from_numpy(weights).float()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.train.seed, workers=True)

    symbol: str = cfg.ingestion.symbol
    features_dir = Path("data/features") / symbol
    n_horizons = len(cfg.features.label_horizons)

    logger.info("train.loading_data", features_dir=str(features_dir))

    # Memory-mapped: the LOB tensor alone is ~10 GB for a two-week range.
    ob_raw = np.load(features_dir / "ob_raw.npy", mmap_mode="r")
    flow = np.load(features_dir / "flow_features.npy", mmap_mode="r")
    ctx = np.load(features_dir / "ctx_features.npy", mmap_mode="r")
    labels = np.load(features_dir / "labels.npy")
    flat_mask = np.load(features_dir / "flat_mask.npy")
    valid_mask = np.load(features_dir / "valid_mask.npy")

    N = len(ob_raw)
    train_end = int(N * cfg.features.train_ratio)
    val_end = int(N * (cfg.features.train_ratio + cfg.features.val_ratio))

    logger.info("train.split", N=N, train=train_end, val=val_end - train_end, test=N - val_end)

    use_ctx: bool = cfg.model.use_ctx
    seq_len: int = cfg.model.seq_len

    stride: int = cfg.train.window_stride
    train_ds = LOBDataset(ob_raw[:train_end], flow[:train_end], ctx[:train_end],
                          labels[:train_end], flat_mask[:train_end], seq_len, use_ctx,
                          stride=stride, valid_mask=valid_mask[:train_end])
    val_ds = LOBDataset(ob_raw[train_end:val_end], flow[train_end:val_end],
                        ctx[train_end:val_end], labels[train_end:val_end],
                        flat_mask[train_end:val_end], seq_len, use_ctx, stride=stride,
                        valid_mask=valid_mask[train_end:val_end])
    test_ds = LOBDataset(ob_raw[val_end:], flow[val_end:], ctx[val_end:],
                         labels[val_end:], flat_mask[val_end:], seq_len, use_ctx,
                         stride=stride, valid_mask=valid_mask[val_end:])
    logger.info("train.windows", train=len(train_ds), val=len(val_ds), test=len(test_ds),
                stride=stride)

    batch_size: int = cfg.train.batch_size
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=cfg.train.num_workers, persistent_workers=cfg.train.num_workers > 0,
                          pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers, persistent_workers=cfg.train.num_workers > 0,
                          pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=cfg.train.num_workers, persistent_workers=cfg.train.num_workers > 0,
                          pin_memory=True)

    class_weights = _compute_class_weights(
        labels[:train_end], valid_mask[:train_end], cfg.model.n_classes,
    )
    logger.info("train.class_weights", weights=class_weights.tolist())

    model = HybridSignalModel(
        ob_depth=cfg.model.ob_depth,
        d_model=cfg.model.d_model,
        d_ctx=cfg.model.d_ctx,
        n_lob_blocks=cfg.model.n_lob_blocks,
        in_features_flow=cfg.model.in_features_flow,
        in_features_ctx=cfg.model.in_features_ctx,
        flow_encoder=cfg.model.flow_encoder,
        gru_layers=cfg.model.gru_layers,
        n_classes=cfg.model.n_classes,
        n_horizons=n_horizons,
        dropout=cfg.model.dropout,
        use_ctx=use_ctx,
    )

    loss_fn = MultiHorizonLoss(
        n_classes=cfg.model.n_classes,
        use_focal=cfg.train.use_focal_loss,
        focal_gamma=cfg.train.focal_gamma,
        class_weights=class_weights,
    )

    lit_module = SignalLitModule(
        model=model,
        loss_fn=loss_fn,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        use_ctx=use_ctx,
        n_classes=cfg.model.n_classes,
        arch={
            "ob_depth": cfg.model.ob_depth,
            "d_model": cfg.model.d_model,
            "d_ctx": cfg.model.d_ctx,
            "n_lob_blocks": cfg.model.n_lob_blocks,
            "in_features_flow": cfg.model.in_features_flow,
            "in_features_ctx": cfg.model.in_features_ctx,
            "flow_encoder": cfg.model.flow_encoder,
            "gru_layers": cfg.model.gru_layers,
            "n_classes": cfg.model.n_classes,
            "n_horizons": n_horizons,
            "dropout": cfg.model.dropout,
            "use_ctx": use_ctx,
        },
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=cfg.train.val_metric,
            mode="max",
            save_top_k=1,
            # No "=" in the name: it is a valid filename but Hydra reads it as
            # override syntax, so the checkpoint cannot be passed back on the CLI.
            filename="best-epoch{epoch:02d}-f1{val_f1_macro:.4f}",
        ),
        pl.callbacks.EarlyStopping(
            monitor=cfg.train.val_metric,
            mode="max",
            patience=cfg.train.patience,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        gradient_clip_val=cfg.train.grad_clip_val,
        callbacks=callbacks,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        accelerator="auto",
        deterministic=True,
    )

    logger.info("train.starting")
    trainer.fit(lit_module, train_dl, val_dl)

    logger.info("train.testing")
    trainer.test(lit_module, test_dl, ckpt_path="best")

    logger.info("train.done", best_ckpt=trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
