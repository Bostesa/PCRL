"""Unit-level checks for sequential FiLM composition.

These tests verify the principled property the rule was designed to give:
when one purpose drives a feature's gamma to zero, the composed gate is
zero regardless of the other purpose's gate. With additive embedding
composition this is not guaranteed.
"""

from __future__ import annotations

import torch

from pcrl.models.conditioning import FiLMConditioner
from pcrl.models.encoder import PurposeConditionedEncoder


def _make_two_purpose_film_encoder(
    input_dim: int = 8,
    hidden: int = 16,
    repr_dim: int = 4,
) -> PurposeConditionedEncoder:
    """Tiny encoder with deterministic init for crafted-weight tests."""
    torch.manual_seed(0)
    enc = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=[hidden, hidden],
        repr_dim=repr_dim,
        num_purposes=2,
        purpose_emb_dim=4,
        conditioning="film",
        dropout=0.0,
    )
    enc.eval()
    return enc


def test_sequential_zero_gamma_zeros_feature() -> None:
    """If purpose A drives gamma to zero on feature 0, sequential composition
    of [A, B] must zero feature 0 in the post-FiLM activation regardless of
    the gate B produces.
    """
    feature_dim, cond_dim = 8, 4
    film = FiLMConditioner(feature_dim, cond_dim)
    film.eval()
    with torch.no_grad():
        # Purpose A: gamma vector hard-coded to zero on feature 0, one elsewhere.
        # Beta hard-coded to zero so the FiLM acts purely as a multiplicative gate.
        film.gamma_proj.weight.data.zero_()
        film.gamma_proj.bias.data.fill_(1.0)
        film.gamma_proj.bias.data[0] = 0.0
        film.beta_proj.weight.data.zero_()
        film.beta_proj.bias.data.zero_()

    # With these weights, gamma(e_a) = gamma(e_b) for any e — both bias-only.
    # Apply [A, B] sequentially (same FiLM module, two embeddings):
    h = torch.randn(3, feature_dim)
    e_a = torch.randn(3, cond_dim)
    e_b = torch.randn(3, cond_dim)

    h1 = film(h, e_a)            # gamma * h + beta
    h_seq = film(h1, e_b)        # gamma * h1 + beta = gamma^2 * h

    # Effective gate at feature 0 should be 0 * 0 = 0; elsewhere 1 * 1 = 1.
    assert torch.allclose(h_seq[:, 0], torch.zeros(3), atol=1e-6), \
        f"feature 0 should be zeroed by sequential FiLM, got {h_seq[:, 0]}"
    assert torch.allclose(h_seq[:, 1:], h[:, 1:], atol=1e-5), \
        "non-zero gate features should pass through identity"

    # Sanity: h_seq must NOT be NaN / infinite.
    assert torch.isfinite(h_seq).all()


def test_sequential_vs_additive_diverge_in_general() -> None:
    """Sequential and additive composition produce different outputs once the
    FiLM modules carry a non-identity gamma. With identity init both paths
    collapse to the same identity transform, so we perturb the weights
    first."""
    enc = _make_two_purpose_film_encoder()
    # Knock the FiLM gamma generators away from identity so the two
    # composition rules can disagree.
    with torch.no_grad():
        for cond in enc.conditioning_layers:
            cond.gamma_proj.weight.data.normal_(0.0, 0.1)
            cond.beta_proj.weight.data.normal_(0.0, 0.1)
            cond.gamma_proj.bias.data.fill_(1.0)  # keep bias-only gamma near 1

    x = torch.randn(2, 8)
    e0 = enc.get_purpose_embedding(0)
    e1 = enc.get_purpose_embedding(1)

    add = enc.forward_with_embedding(x, (e0 + e1).expand(2, -1))
    seq = enc.forward_sequential_composition(x, [e0.expand(2, -1), e1.expand(2, -1)])

    assert add.shape == seq.shape
    assert torch.isfinite(add).all() and torch.isfinite(seq).all()
    assert not torch.allclose(add, seq, atol=1e-4), \
        "additive and sequential outputs are unexpectedly identical"


def test_sequential_single_purpose_matches_forward_with_embedding() -> None:
    """Sequential composition with a single purpose embedding must equal
    forward_with_embedding using the same embedding."""
    enc = _make_two_purpose_film_encoder()
    x = torch.randn(2, 8)
    e = enc.get_purpose_embedding(0).expand(2, -1)

    a = enc.forward_with_embedding(x, e)
    b = enc.forward_sequential_composition(x, [e])
    assert torch.allclose(a, b, atol=1e-6), \
        f"single-purpose sequential should equal forward_with_embedding, max diff = {(a - b).abs().max().item():.2e}"


if __name__ == "__main__":
    test_sequential_zero_gamma_zeros_feature()
    print("OK  test_sequential_zero_gamma_zeros_feature")
    test_sequential_vs_additive_diverge_in_general()
    print("OK  test_sequential_vs_additive_diverge_in_general")
    test_sequential_single_purpose_matches_forward_with_embedding()
    print("OK  test_sequential_single_purpose_matches_forward_with_embedding")
