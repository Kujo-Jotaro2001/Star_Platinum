import torch.nn as nn
from torch import Tensor


class LOBEncoder(nn.Module):
    """CNN2d encoder for raw order book tensor.

    Input:  [B, T, K, 4]
    Output: [B, T, d_model]

    3 Conv2d blocks with asymmetric kernels:
      block 0: (1, 3) — spatial patterns across price levels only
      block 1-2: (3, 3) — spatio-temporal patterns
    The K dimension is collapsed by a mean at the end.
    """

    def __init__(
        self,
        ob_depth: int,
        d_model: int,
        n_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        channels = self._channel_schedule(d_model, n_blocks)
        blocks: list[nn.Module] = []
        for i in range(n_blocks):
            in_c = channels[i]
            out_c = channels[i + 1]
            kernel = (1, 3) if i == 0 else (3, 3)
            # padding 'same' equivalent: pad T and K to preserve dims
            pad_t = (kernel[0] - 1) // 2
            pad_k = (kernel[1] - 1) // 2
            blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel, padding=(pad_t, pad_k)),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ))
        self.blocks = nn.ModuleList(blocks)

    @staticmethod
    def _channel_schedule(d_model: int, n_blocks: int) -> list[int]:
        """4 → ... → d_model in n_blocks steps."""
        if n_blocks == 1:
            return [4, d_model]
        step = d_model / n_blocks
        channels = [4]
        for i in range(1, n_blocks):
            channels.append(int(step * i))
        channels.append(d_model)
        return channels

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, K, 4]
        x = x.permute(0, 3, 1, 2)  # [B, 4, T, K]
        for block in self.blocks:
            x = block(x)
        # x: [B, d_model, T, K]
        # Mean over K rather than AdaptiveAvgPool2d: identical result, but the
        # pooling op has no deterministic CUDA backward and the trainer runs with
        # deterministic algorithms on.
        x = x.mean(dim=3)  # [B, d_model, T]
        x = x.permute(0, 2, 1)  # [B, T, d_model]
        return x


class FlowEncoder(nn.Module):
    """Sequence encoder for trade-flow features.

    Input:  [B, T, in_features]
    Output: [B, T, d_model]

    Default: GRU. Opt-in: Mamba (requires mamba_ssm).
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        encoder_type: str,
        gru_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self._type = encoder_type

        if encoder_type == "gru":
            self.rnn = nn.GRU(
                input_size=in_features,
                hidden_size=d_model,
                num_layers=gru_layers,
                batch_first=True,
                dropout=dropout if gru_layers > 1 else 0.0,
            )
        elif encoder_type == "mamba":
            try:
                from mamba_ssm import Mamba
            except ImportError as e:
                raise ImportError(
                    "Mamba encoder requires mamba_ssm: "
                    "pip install mamba-ssm"
                ) from e
            self.proj_in = nn.Linear(in_features, d_model)
            self.mamba = Mamba(d_model=d_model)
        else:
            raise ValueError(f"Unknown flow_encoder type: {encoder_type!r}")

    def forward(self, x: Tensor) -> Tensor:
        if self._type == "gru":
            out, _ = self.rnn(x)
            return out
        # mamba
        x = self.proj_in(x)
        return self.mamba(x)


class CtxEncoder(nn.Module):
    """Per-timestep MLP for context features.

    Input:  [B, T, in_features]
    Output: [B, T, d_ctx]
    """

    def __init__(self, in_features: int, d_ctx: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, d_ctx),
            nn.GELU(),
            nn.LayerNorm(d_ctx),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
