import torch

from bot.models.fusion import GatedFusion

B, T = 2, 10
D_MODEL, D_CTX = 128, 64


class TestGatedFusion:
    def test_with_ctx(self) -> None:
        fuse = GatedFusion(d_model=D_MODEL, d_ctx=D_CTX, use_ctx=True)
        z_lob = torch.randn(B, T, D_MODEL)
        z_flow = torch.randn(B, T, D_MODEL)
        z_ctx = torch.randn(B, T, D_CTX)
        out = fuse(z_lob, z_flow, z_ctx)
        assert out.shape == (B, T, D_MODEL)

    def test_without_ctx(self) -> None:
        fuse = GatedFusion(d_model=D_MODEL, d_ctx=D_CTX, use_ctx=False)
        z_lob = torch.randn(B, T, D_MODEL)
        z_flow = torch.randn(B, T, D_MODEL)
        out = fuse(z_lob, z_flow)
        assert out.shape == (B, T, D_MODEL)

    def test_gradient_flows(self) -> None:
        fuse = GatedFusion(d_model=32, d_ctx=16, use_ctx=True)
        z_lob = torch.randn(B, T, 32, requires_grad=True)
        z_flow = torch.randn(B, T, 32, requires_grad=True)
        z_ctx = torch.randn(B, T, 16, requires_grad=True)
        out = fuse(z_lob, z_flow, z_ctx)
        out.sum().backward()
        assert z_lob.grad is not None
        assert z_flow.grad is not None
        assert z_ctx.grad is not None
