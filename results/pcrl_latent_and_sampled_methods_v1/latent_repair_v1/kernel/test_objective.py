"""Data-free checks for the fixed D-K hard-code privacy objective."""

import math

import pytest
import torch

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.kernel.objective import (
    hard_code_mmd,
    multiscale_mmd_terms,
)


def _bits(values):
    """Pad tiny hand laws to the public 10-bit code width."""
    return torch.tensor([[value] + [0] * 9 for value in values], dtype=torch.float32)


def _off_diagonal_kernel():
    return sum(math.exp(-gamma) for gamma in (0.25, 0.5, 1.0, 2.0)) / 4.0


def test_biased_singleton_mmd_keeps_diagonals_and_pair_product():
    # SEX 0 has (0,0); SEX 1 has (1,1). Every diagonal kernel value is 1.
    sex = torch.tensor([0, 1], dtype=torch.long)
    terms = hard_code_mmd(_bits([0, 1]), _bits([0, 1]), sex)
    r = _off_diagonal_kernel()
    assert terms.local0.item() == pytest.approx(2 * (1 - r), abs=1e-12)
    assert terms.local1.item() == pytest.approx(2 * (1 - r), abs=1e-12)
    assert terms.pair.item() == pytest.approx(2 * (1 - r * r), abs=1e-12)
    assert terms.loss.item() == pytest.approx((terms.local0 + terms.local1 + terms.pair).item() / 3)


def test_xor_complementarity_has_zero_local_and_positive_pair_mmd():
    # SEX 0: 00/11; SEX 1: 01/10. Both local marginals are identical.
    sex = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    left = _bits([0, 1, 0, 1])
    right = _bits([0, 1, 1, 0])
    terms = hard_code_mmd(left, right, sex)
    r = _off_diagonal_kernel()
    assert terms.local0.item() == pytest.approx(0, abs=1e-12)
    assert terms.local1.item() == pytest.approx(0, abs=1e-12)
    assert terms.pair.item() == pytest.approx((1 - r) ** 2, abs=1e-12)
    assert tuple(value.item() for value in multiscale_mmd_terms(left, right, sex)) == pytest.approx(
        (0, 0, (1 - r) ** 2), abs=1e-12
    )


def test_hard_forward_st_bits_keep_backward_gradient_and_frozen_anchor():
    # The loss must see hard bits forward, while the ST path updates soft logits.
    sex = torch.tensor([0, 1], dtype=torch.long)
    soft = torch.tensor([[0.2] * 10, [0.8] * 10], dtype=torch.float32, requires_grad=True)
    hard = (soft >= 0.5).to(soft.dtype)
    straight_through = soft + (hard - soft).detach()
    anchor = _bits([0, 1]).detach()
    terms = hard_code_mmd(anchor, straight_through, sex)
    assert terms.local1.item() > 0
    assert terms.pair.item() > 0
    terms.loss.backward()
    assert soft.grad is not None
    assert torch.isfinite(soft.grad).all()
    assert torch.any(soft.grad.abs() > 0)
    assert terms.loss.dtype == torch.float64


def test_identical_empirical_groups_are_nonnegative_zero():
    sex = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    terms = hard_code_mmd(_bits([0, 1, 0, 1]), _bits([1, 0, 1, 0]), sex)
    assert all(value.item() == pytest.approx(0, abs=1e-12) for value in
               (terms.local0, terms.local1, terms.pair, terms.loss))


@pytest.mark.parametrize(
    "first,second,sex",
    [
        (_bits([0, 1]), _bits([0]), torch.tensor([0, 1])),
        (_bits([0, 1]), _bits([0, 1]), torch.tensor([0, 0])),
        (_bits([0, 1]), _bits([0, 1]), torch.tensor([0, 2])),
        (torch.zeros((2, 9)), _bits([0, 1]), torch.tensor([0, 1])),
        (torch.full((2, 10), 0.3), _bits([0, 1]), torch.tensor([0, 1])),
    ],
)
def test_invalid_or_nonhard_aligned_inputs_refuse(first, second, sex):
    with pytest.raises(ValueError):
        hard_code_mmd(first, second, sex)
