"""Smoke tests for the rebuttal-pilot erase-layer architecture.

Verifies:
1. ``StandardEncoder(use_erase_layer=True)`` registers a frozen erase Linear
   between ``network`` and ``repr_proj``, initialised to identity, and the
   forward pass still produces output of the right shape.
2. ``PerPurposeLoRAEncoder(lora_target="repr_proj_only")`` attaches LoRA to
   exactly one Linear (the last, i.e. ``repr_proj``) — and skips the
   ``_skip_lora=True`` erase layer even when it would otherwise be eligible.
3. Combining the two preserves the forward semantics: backward through LoRA
   parameters works; the erase layer's parameters stay frozen.

These tests are construction-only — no training is performed.
"""

from __future__ import annotations

import torch

from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder


def test_standard_encoder_erase_layer_off_by_default() -> None:
    """Default construction leaves the erase layer absent (backwards-compat)."""
    enc = StandardEncoder(input_dim=10, hidden_dims=[32, 32], repr_dim=16)
    assert enc.erase is None
    assert enc.use_erase_layer is False


def test_standard_encoder_erase_layer_frozen_identity() -> None:
    enc = StandardEncoder(
        input_dim=10, hidden_dims=[32, 32], repr_dim=16, use_erase_layer=True,
    )
    assert enc.erase is not None
    assert enc.erase.weight.shape == (32, 32)
    assert enc.erase.bias.shape == (32,)
    # Initialised to identity + zero bias.
    assert torch.allclose(enc.erase.weight.data, torch.eye(32))
    assert torch.allclose(enc.erase.bias.data, torch.zeros(32))
    # Parameters frozen.
    for p in enc.erase.parameters():
        assert p.requires_grad is False
    # Marker attribute so PerPurposeLoRAEncoder skips it.
    assert getattr(enc.erase, "_skip_lora", False) is True


def test_standard_encoder_forward_with_identity_erase_matches_off() -> None:
    """Identity-init erase is a no-op: forward output matches the no-erase encoder
    for the same input + matched manual weight copy."""
    torch.manual_seed(0)
    enc_off = StandardEncoder(input_dim=10, hidden_dims=[16], repr_dim=8)
    enc_on = StandardEncoder(
        input_dim=10, hidden_dims=[16], repr_dim=8, use_erase_layer=True,
    )
    # Copy weights so the only difference is the identity erase layer.
    enc_on.network.load_state_dict(enc_off.network.state_dict())
    enc_on.repr_proj.load_state_dict(enc_off.repr_proj.state_dict())
    enc_off.eval()
    enc_on.eval()

    x = torch.randn(4, 10)
    with torch.no_grad():
        y_off = enc_off(x)
        y_on = enc_on(x)
    assert torch.allclose(y_off, y_on, atol=1e-6)


def test_lora_repr_proj_only_attaches_one_adapter() -> None:
    enc = StandardEncoder(input_dim=10, hidden_dims=[32, 32], repr_dim=16)
    lora = PerPurposeLoRAEncoder(
        backbone=enc, n_purposes=3, rank=4, alpha=8.0,
        lora_target="repr_proj_only",
    )
    # Exactly one Linear is adapted, and it's the final repr_proj.
    assert len(lora._linear_modules) == 1
    assert lora._linear_modules[0] is enc.repr_proj
    # 3 purposes × 1 adapter each.
    assert sum(1 for _ in lora.adapters.parameters()) > 0
    # The default ("all_linear") attaches more adapters.
    lora_all = PerPurposeLoRAEncoder(
        backbone=StandardEncoder(input_dim=10, hidden_dims=[32, 32], repr_dim=16),
        n_purposes=3, rank=4, alpha=8.0,
    )
    assert len(lora_all._linear_modules) > len(lora._linear_modules)


def test_lora_skips_erase_layer_even_with_all_linear() -> None:
    """Erase layer must be excluded from LoRA targets even under the default
    ``lora_target='all_linear'`` (otherwise LoRA would bypass the structural
    compliance the erase layer is meant to enforce)."""
    enc = StandardEncoder(
        input_dim=10, hidden_dims=[32, 32], repr_dim=16, use_erase_layer=True,
    )
    lora = PerPurposeLoRAEncoder(
        backbone=enc, n_purposes=2, rank=4, alpha=8.0, lora_target="all_linear",
    )
    # Erase layer not in the adapted set.
    assert enc.erase not in lora._linear_modules


def test_lora_repr_proj_only_with_erase_forward_works() -> None:
    """Full pilot configuration: erase on + LoRA on repr_proj only. Forward
    should run without error and produce a tensor of the right shape."""
    enc = StandardEncoder(
        input_dim=8, hidden_dims=[16, 16], repr_dim=8, use_erase_layer=True,
    )
    lora = PerPurposeLoRAEncoder(
        backbone=enc, n_purposes=2, rank=2, alpha=4.0,
        lora_target="repr_proj_only",
    )
    x = torch.randn(5, 8)
    out_p0 = lora(x, 0)
    out_p1 = lora(x, 1)
    assert out_p0.shape == (5, 8)
    assert out_p1.shape == (5, 8)
    # The erase layer's params didn't acquire grads from the construction.
    for p in enc.erase.parameters():
        assert p.requires_grad is False


def test_lora_repr_proj_only_backward_only_touches_lora_params() -> None:
    enc = StandardEncoder(
        input_dim=8, hidden_dims=[16], repr_dim=4, use_erase_layer=True,
    )
    lora = PerPurposeLoRAEncoder(
        backbone=enc, n_purposes=2, rank=2, alpha=4.0,
        lora_target="repr_proj_only",
    )
    x = torch.randn(3, 8)
    loss = lora(x, 0).sum() + lora(x, 1).sum()
    loss.backward()
    # LoRA params have grads.
    has_lora_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in lora.adapters.parameters()
    )
    assert has_lora_grad
    # Erase layer params never accumulate grads (requires_grad=False).
    for p in enc.erase.parameters():
        assert p.grad is None
    # Backbone network linears: frozen too, no grads.
    for mod in enc.network.modules():
        if isinstance(mod, torch.nn.Linear):
            assert mod.weight.grad is None
