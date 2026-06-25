import torch
import torch.nn as nn
from torch import Tensor


class GatedFusion(nn.Module):
    """Gated fusion of LOB, flow, and (optional) context encodings.

    Input:  z_lob [B,T,d_model], z_flow [B,T,d_model], z_ctx [B,T,d_ctx] (optional)
    Output: [B, T, d_model]

    Concat → candidate + gate projections → sigmoid gating.
    use_ctx is fixed at init time for clean ONNX export.
    """

    def __init__(self, d_model: int, d_ctx: int, use_ctx: bool) -> None:
        super().__init__()
        self._use_ctx = use_ctx
        input_dim = 2 * d_model + (d_ctx if use_ctx else 0)
        self.candidate = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_lob: Tensor,
        z_flow: Tensor,
        z_ctx: Tensor | None = None,
    ) -> Tensor:
        parts = [z_lob, z_flow]
        if self._use_ctx:
            parts.append(z_ctx)
        h = torch.cat(parts, dim=-1)
        return self.gate(h) * self.candidate(h)
