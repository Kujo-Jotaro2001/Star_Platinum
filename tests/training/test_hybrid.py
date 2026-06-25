import torch

from bot.models.hybrid import HybridSignalModel

B, T, K = 2, 10, 50
H, C = 3, 3


class TestHybridSignalModel:
    def _make_model(self, use_ctx: bool = True) -> HybridSignalModel:
        return HybridSignalModel(
            ob_depth=K, d_model=32, d_ctx=16, n_lob_blocks=2,
            in_features_flow=9, in_features_ctx=17,
            flow_encoder="gru", gru_layers=1,
            n_classes=C, n_horizons=H, dropout=0.0, use_ctx=use_ctx,
        )

    def test_output_shape_with_ctx(self) -> None:
        model = self._make_model(use_ctx=True)
        ob = torch.randn(B, T, K, 4)
        flow = torch.randn(B, T, 9)
        ctx = torch.randn(B, T, 17)
        out = model(ob, flow, ctx)
        assert out.shape == (B, H, C)

    def test_output_shape_without_ctx(self) -> None:
        model = self._make_model(use_ctx=False)
        ob = torch.randn(B, T, K, 4)
        flow = torch.randn(B, T, 9)
        out = model(ob, flow)
        assert out.shape == (B, H, C)

    def test_gradient_flows(self) -> None:
        model = self._make_model(use_ctx=True)
        ob = torch.randn(B, T, K, 4, requires_grad=True)
        flow = torch.randn(B, T, 9, requires_grad=True)
        ctx = torch.randn(B, T, 17, requires_grad=True)
        out = model(ob, flow, ctx)
        out.sum().backward()
        assert ob.grad is not None
        assert flow.grad is not None
        assert ctx.grad is not None
