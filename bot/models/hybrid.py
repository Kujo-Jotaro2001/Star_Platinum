import torch.nn as nn
from torch import Tensor

from bot.models.encoders import CtxEncoder, FlowEncoder, LOBEncoder
from bot.models.fusion import GatedFusion


class HybridSignalModel(nn.Module):
    """Full model: LOB + Flow + (optional) Ctx encoders → GatedFusion → Head.

    Input:
        ob:   [B, T, K, 4]
        flow: [B, T, 9]
        ctx:  [B, T, 17] or None
    Output:
        logits: [B, H, C]
    """

    def __init__(
        self,
        ob_depth: int,
        d_model: int,
        d_ctx: int,
        n_lob_blocks: int,
        in_features_flow: int,
        in_features_ctx: int,
        flow_encoder: str,
        gru_layers: int,
        n_classes: int,
        n_horizons: int,
        dropout: float,
        use_ctx: bool,
    ) -> None:
        super().__init__()
        self._use_ctx = use_ctx
        self._n_horizons = n_horizons
        self._n_classes = n_classes

        self.lob_encoder = LOBEncoder(
            ob_depth=ob_depth,
            d_model=d_model,
            n_blocks=n_lob_blocks,
            dropout=dropout,
        )
        self.flow_encoder = FlowEncoder(
            in_features=in_features_flow,
            d_model=d_model,
            encoder_type=flow_encoder,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        self.ctx_encoder: CtxEncoder | None = None
        if use_ctx:
            self.ctx_encoder = CtxEncoder(
                in_features=in_features_ctx,
                d_ctx=d_ctx,
            )

        self.fusion = GatedFusion(
            d_model=d_model,
            d_ctx=d_ctx,
            use_ctx=use_ctx,
        )
        self.head = nn.Linear(d_model, n_horizons * n_classes)

    def forward(
        self,
        ob: Tensor,
        flow: Tensor,
        ctx: Tensor | None = None,
    ) -> Tensor:
        z_lob = self.lob_encoder(ob)
        z_flow = self.flow_encoder(flow)
        z_ctx = self.ctx_encoder(ctx) if self._use_ctx and ctx is not None else None
        fused = self.fusion(z_lob, z_flow, z_ctx)  # [B, T, d_model]
        pooled = fused.mean(dim=1)  # [B, d_model]
        logits = self.head(pooled)  # [B, H*C]
        return logits.view(-1, self._n_horizons, self._n_classes)
