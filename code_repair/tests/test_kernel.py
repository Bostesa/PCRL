"""Data-free checks for the capacity-aware hard-code kernel."""

import pytest
import torch

from code_repair.kernel import hard_code_mmd


@pytest.mark.parametrize("width", [10, 16])
def test_xor_clue_has_zero_local_and_positive_aligned_pair_mmd(width):
    # S=U xor V, with each (U,V) combination represented once.
    u = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    v = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    sex = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    bits0 = torch.zeros((4, width), dtype=torch.float32)
    bits1 = torch.zeros((4, width), dtype=torch.float32)
    bits0[:, 0] = u
    bits1[:, 0] = v
    terms = hard_code_mmd(bits0, bits1, sex)
    assert torch.isclose(terms.local0, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)
    assert torch.isclose(terms.local1, torch.tensor(0.0, dtype=torch.float64), atol=1e-12)
    assert terms.pair.item() > 0.0


def test_kernel_rejects_mixed_width_and_nonbinary_forward():
    sex = torch.tensor([0, 1], dtype=torch.int64)
    code10 = torch.zeros((2, 10))
    code16 = torch.zeros((2, 16))
    with pytest.raises(ValueError):
        hard_code_mmd(code10, code16, sex)
    code10[0, 0] = 0.25
    with pytest.raises(ValueError):
        hard_code_mmd(code10, code10, sex)
