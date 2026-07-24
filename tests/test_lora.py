"""Unit tests for LoRA per-purpose adapters."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import LoRAAdapter, PerPurposeLoRAEncoder


# ───────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────


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
def encoder(backbone: StandardEncoder) -> PerPurposeLoRAEncoder:
    return PerPurposeLoRAEncoder(
        backbone=backbone,
        n_purposes=3,
        rank=4,
    ).eval()


# ───────────────────────────────────────────────────────────────────────────
# LoRAAdapter unit
# ───────────────────────────────────────────────────────────────────────────


def test_lora_adapter_zero_init_is_identity() -> None:
    """At init, B=0 so the adapter contribution is exactly zero."""
    torch.manual_seed(0)
    adapter = LoRAAdapter(in_features=10, out_features=8, rank=4)
    x = torch.randn(7, 10)
    out = adapter(x)
    assert out.shape == (7, 8)
    # Strict zero — B is initialised to all zeros and matmul preserves that.
    assert torch.equal(out, torch.zeros_like(out))


def test_lora_adapter_nonzero_after_training() -> None:
    """If B is forced nonzero, adapter contribution becomes nonzero."""
    torch.manual_seed(0)
    adapter = LoRAAdapter(in_features=10, out_features=8, rank=4)
    # Manually break the identity init by writing a nonzero B.
    with torch.no_grad():
        adapter.B.weight.copy_(torch.randn_like(adapter.B.weight))
    x = torch.randn(7, 10)
    out = adapter(x)
    assert not torch.allclose(out, torch.zeros_like(out)), \
        "adapter should produce nonzero output after B becomes nonzero"


# ───────────────────────────────────────────────────────────────────────────
# PerPurposeLoRAEncoder
# ───────────────────────────────────────────────────────────────────────────


def test_per_purpose_encoder_freezes_backbone(
    encoder: PerPurposeLoRAEncoder, backbone: StandardEncoder,
) -> None:
    """Every backbone parameter has requires_grad=False."""
    backbone_param_ids = {id(p) for p in backbone.parameters()}
    for p in encoder.backbone.parameters():
        assert not p.requires_grad, \
            "backbone parameters must be frozen by PerPurposeLoRAEncoder"
    # Sanity: adapter params ARE trainable.
    adapter_trainable = [p for p in encoder.adapters.parameters() if p.requires_grad]
    assert len(adapter_trainable) > 0, "adapter parameters should be trainable"
    # Sanity: no aliasing between adapter and backbone params.
    for p in encoder.adapters.parameters():
        assert id(p) not in backbone_param_ids


def test_per_purpose_encoder_distinct_outputs(encoder: PerPurposeLoRAEncoder) -> None:
    """Different purposes give different outputs once adapters diverge."""
    # Force divergent B matrices for each purpose so adapter contributions
    # differ. Without this, B=0 means both purposes return the same backbone
    # output (which is the *correct* behaviour at init, see other test).
    torch.manual_seed(0)
    with torch.no_grad():
        for p in range(encoder.n_purposes):
            for adapter in encoder.adapters[p]:
                adapter.B.weight.copy_(
                    torch.randn_like(adapter.B.weight) * 0.5 * (p + 1)
                )

    x = torch.randn(8, encoder._linear_modules[0].in_features)
    out0 = encoder(x, 0)
    out1 = encoder(x, 1)
    out2 = encoder(x, 2)

    assert out0.shape == out1.shape == out2.shape
    assert not torch.allclose(out0, out1), \
        "purpose 0 and 1 outputs should differ once adapters are nonzero"
    assert not torch.allclose(out1, out2), \
        "purpose 1 and 2 outputs should differ once adapters are nonzero"


def test_per_purpose_encoder_zero_init_matches_backbone(
    encoder: PerPurposeLoRAEncoder, backbone: StandardEncoder,
) -> None:
    """At init (B=0), encoder(x, p) is bit-equal to backbone(x)."""
    encoder.eval()
    backbone.eval()
    torch.manual_seed(0)
    x = torch.randn(8, 20)

    expected = backbone(x)
    for p in range(encoder.n_purposes):
        out = encoder(x, p)
        assert torch.allclose(out, expected, atol=0, rtol=0), \
            f"encoder(x, {p}) must equal backbone(x) when B is zero"


def test_gradient_flows_to_adapters_only(encoder: PerPurposeLoRAEncoder) -> None:
    """Backward populates adapter grads but not backbone grads."""
    # Adapter must produce a nonzero gradient signal — but B starts at zero,
    # so the adapter contribution is zero AND its gradient w.r.t. loss is
    # also exactly zero (chain rule through B). Perturb B slightly to break
    # the zero-gradient symmetry while keeping LoRA semantics intact.
    torch.manual_seed(0)
    with torch.no_grad():
        for p in range(encoder.n_purposes):
            for adapter in encoder.adapters[p]:
                adapter.B.weight.copy_(torch.randn_like(adapter.B.weight) * 0.1)

    encoder.train()
    x = torch.randn(8, 20, requires_grad=False)
    out = encoder(x, 1)
    loss = out.pow(2).sum()
    loss.backward()

    for p in encoder.backbone.parameters():
        assert p.grad is None, \
            "backbone parameter received a gradient — it should be frozen"

    nonzero_adapter_grads = 0
    for adapter in encoder.adapters[1]:
        for p in adapter.parameters():
            assert p.grad is not None, "adapter param has no gradient"
            if p.grad.abs().sum().item() > 0:
                nonzero_adapter_grads += 1
    assert nonzero_adapter_grads > 0, "no adapter received a nonzero gradient"


def test_purpose_idx_out_of_range_raises(encoder: PerPurposeLoRAEncoder) -> None:
    """Out-of-range or non-int purpose_idx raises ValueError."""
    x = torch.randn(4, 20)
    with pytest.raises(ValueError):
        encoder(x, -1)
    with pytest.raises(ValueError):
        encoder(x, encoder.n_purposes)
    with pytest.raises(ValueError):
        encoder(x, 99)
    with pytest.raises(ValueError):
        encoder(x, "0")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        # Tensor with multiple distinct elements is unsupported.
        encoder(x, torch.tensor([0, 1, 2]))


def test_n_trainable_matches_lora_count(
    encoder: PerPurposeLoRAEncoder, backbone: StandardEncoder,
) -> None:
    """n_trainable counts adapter params only, and is < total backbone params."""
    n_trainable = encoder.n_trainable()
    n_via_iter = sum(p.numel() for p in encoder.trainable_parameters())
    assert n_trainable == n_via_iter

    backbone_total = sum(p.numel() for p in backbone.parameters())
    assert 0 < n_trainable < backbone_total, (
        f"expected 0 < n_trainable ({n_trainable}) < backbone ({backbone_total})"
    )

    # Cross-check against the closed-form expectation for our backbone:
    # for each Linear with weight (out, in), per-purpose adapter has
    # A: (rank, in), B: (out, rank), and an adapter-side bias (out,) that
    # absorbs the LEACE affine translation.
    expected_per_purpose = sum(
        m.in_features * encoder.rank + encoder.rank * m.out_features + m.out_features
        for m in encoder._linear_modules
    )
    expected_total = expected_per_purpose * encoder.n_purposes
    assert n_trainable == expected_total
