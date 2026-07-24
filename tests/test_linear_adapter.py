"""Unit tests for the full-rank LinearAdapter (FAccT resubmission Ablation 1)."""

from __future__ import annotations

import pytest
import torch

from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import LinearAdapter, PerPurposeLoRAEncoder


@pytest.fixture
def backbone() -> StandardEncoder:
    torch.manual_seed(0)
    return StandardEncoder(
        input_dim=20,
        hidden_dims=[32, 32],
        repr_dim=16,
        dropout=0.0,
    ).eval()


@pytest.fixture
def linear_encoder(backbone: StandardEncoder) -> PerPurposeLoRAEncoder:
    return PerPurposeLoRAEncoder(
        backbone=backbone,
        n_purposes=3,
        adapter_type="linear",
        lora_target="repr_proj_only",
    ).eval()


def test_linear_adapter_zero_init_is_identity() -> None:
    adapter = LinearAdapter(in_features=10, out_features=8)
    x = torch.randn(7, 10)
    out = adapter(x)
    assert out.shape == (7, 8)
    assert torch.equal(out, torch.zeros_like(out))


def test_linear_adapter_zero_out() -> None:
    adapter = LinearAdapter(in_features=10, out_features=8)
    with torch.no_grad():
        adapter.delta.weight.copy_(torch.randn_like(adapter.delta.weight))
        adapter.bias.copy_(torch.randn_like(adapter.bias))
    adapter.zero_out()
    x = torch.randn(5, 10)
    assert torch.equal(adapter(x), torch.zeros(5, 8))


def test_adapter_type_validation(backbone: StandardEncoder) -> None:
    with pytest.raises(ValueError, match="adapter_type"):
        PerPurposeLoRAEncoder(backbone=backbone, n_purposes=2, adapter_type="mlp")


def test_linear_encoder_zero_init_matches_backbone(
    backbone: StandardEncoder, linear_encoder: PerPurposeLoRAEncoder
) -> None:
    x = torch.randn(11, 20)
    base = backbone(x)
    for p in range(3):
        assert torch.allclose(linear_encoder(x, p), base, atol=1e-6)


def test_linear_encoder_gradients_flow_to_adapters_only(
    linear_encoder: PerPurposeLoRAEncoder,
) -> None:
    x = torch.randn(11, 20)
    out = linear_encoder(x, 1)
    out.sum().backward()
    for p_idx, per_purpose in enumerate(linear_encoder.adapters):
        for adapter in per_purpose:
            grad = adapter.delta.weight.grad
            if p_idx == 1:
                assert grad is not None
            else:
                assert grad is None or torch.equal(grad, torch.zeros_like(grad))
    for p in linear_encoder.backbone.parameters():
        assert p.grad is None


def test_linear_init_from_affine_is_exact(
    linear_encoder: PerPurposeLoRAEncoder,
) -> None:
    """Full-rank adapter realises z' = Q z + c exactly (no SVD truncation)."""
    torch.manual_seed(1)
    d = 16
    # Random projection-like Q (not low-rank-compatible: rank(Q - I) = d/2 > any small r)
    M = torch.randn(d, d // 2)
    Qsub, _ = torch.linalg.qr(M)
    Q = torch.eye(d) - Qsub @ Qsub.T
    c = torch.randn(d)

    linear_encoder.init_last_layer_from_affine(0, Q, c)

    x = torch.randn(9, 20)
    base = linear_encoder.backbone(x)
    got = linear_encoder(x, 0)
    want = base @ Q.T + c
    assert torch.allclose(got, want, atol=1e-5)
    # Other purposes untouched.
    assert torch.allclose(linear_encoder(x, 1), base, atol=1e-6)


def test_linear_encoder_repr_proj_only_targets_last_linear(
    linear_encoder: PerPurposeLoRAEncoder,
) -> None:
    assert len(linear_encoder._linear_modules) == 1
    host = linear_encoder._linear_modules[0]
    adapter = linear_encoder.adapters[0][0]
    assert isinstance(adapter, LinearAdapter)
    assert adapter.delta.weight.shape == host.weight.shape


def test_linear_encoder_n_trainable() -> None:
    torch.manual_seed(0)
    backbone = StandardEncoder(
        input_dim=20, hidden_dims=[32, 32], repr_dim=16, dropout=0.0,
    )
    enc = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=2,
        adapter_type="linear", lora_target="repr_proj_only",
    )
    # Per purpose: delta (16×32) + bias (16)
    assert enc.n_trainable() == 2 * (16 * 32 + 16)
