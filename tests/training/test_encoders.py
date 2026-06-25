import torch

from bot.models.encoders import CtxEncoder, FlowEncoder, LOBEncoder

B, T, K = 2, 10, 50


class TestLOBEncoder:
    def test_output_shape(self) -> None:
        enc = LOBEncoder(ob_depth=K, d_model=128, n_blocks=3, dropout=0.0)
        x = torch.randn(B, T, K, 4)
        out = enc(x)
        assert out.shape == (B, T, 128)

    def test_single_block(self) -> None:
        enc = LOBEncoder(ob_depth=K, d_model=64, n_blocks=1, dropout=0.0)
        x = torch.randn(B, T, K, 4)
        out = enc(x)
        assert out.shape == (B, T, 64)

    def test_gradient_flows(self) -> None:
        enc = LOBEncoder(ob_depth=K, d_model=32, n_blocks=2, dropout=0.0)
        x = torch.randn(B, T, K, 4, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None


class TestFlowEncoder:
    def test_gru_shape(self) -> None:
        enc = FlowEncoder(in_features=9, d_model=128, encoder_type="gru",
                          gru_layers=2, dropout=0.0)
        x = torch.randn(B, T, 9)
        out = enc(x)
        assert out.shape == (B, T, 128)

    def test_gradient_flows(self) -> None:
        enc = FlowEncoder(in_features=9, d_model=32, encoder_type="gru",
                          gru_layers=1, dropout=0.0)
        x = torch.randn(B, T, 9, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None

    def test_unknown_type_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="Unknown flow_encoder"):
            FlowEncoder(in_features=9, d_model=32, encoder_type="transformer",
                        gru_layers=1, dropout=0.0)


class TestCtxEncoder:
    def test_output_shape(self) -> None:
        enc = CtxEncoder(in_features=17, d_ctx=64)
        x = torch.randn(B, T, 17)
        out = enc(x)
        assert out.shape == (B, T, 64)

    def test_gradient_flows(self) -> None:
        enc = CtxEncoder(in_features=17, d_ctx=32)
        x = torch.randn(B, T, 17, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None
