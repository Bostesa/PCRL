"""Fixed hard-code distribution-matching objective for Team D-K.

The numerical forward path reads the transmitted hard bits. Gradients pass
through the caller's straight-through tensor and are surrogate gradients,
not derivatives of the discontinuous public quantizer. This loss is a fitting
surrogate; it is not a privacy bound or a certificate of attack resistance.
"""

from dataclasses import dataclass

import torch


KERNEL_GAMMAS = (0.25, 0.5, 1.0, 2.0)
PUBLIC_CODE_WIDTH = 10


@dataclass(frozen=True)
class KernelMMDTerms:
    local0: torch.Tensor
    local1: torch.Tensor
    pair: torch.Tensor
    loss: torch.Tensor


def hard_code_mmd(bits0_st: torch.Tensor, bits1_st: torch.Tensor, sex: torch.Tensor) -> KernelMMDTerms:
    """Biased MMD² for both local views and the aligned product-kernel pair.

    All within-group ordered pairs, including diagonals, are counted. Inputs
    must be in the same person order; this function cannot reconstruct row
    identity from tensors. Each SEX group must occur in the batch. A frozen
    first code in sequential training can be supplied as ``bits0_st.detach()``.
    """
    _validate_inputs(bits0_st, bits1_st, sex)
    first = _multiscale_gram(bits0_st)
    second = _multiscale_gram(bits1_st)
    group0, group1 = sex == 0, sex == 1
    local0 = _biased_mmd2(first, group0, group1)
    local1 = _biased_mmd2(second, group0, group1)
    pair = _biased_mmd2(first * second, group0, group1)
    return KernelMMDTerms(local0, local1, pair, (local0 + local1 + pair) / 3.0)


def multiscale_mmd_terms(bits0_st: torch.Tensor, bits1_st: torch.Tensor, sex: torch.Tensor):
    """Tuple adapter for the D trainer; terms remain differentiable scalars."""
    terms = hard_code_mmd(bits0_st, bits1_st, sex)
    return terms.local0, terms.local1, terms.pair


def _validate_inputs(bits0_st: torch.Tensor, bits1_st: torch.Tensor, sex: torch.Tensor) -> None:
    if not all(isinstance(value, torch.Tensor) for value in (bits0_st, bits1_st, sex)):
        raise ValueError("hard codes and SEX must be torch tensors")
    if (bits0_st.ndim != 2 or bits1_st.ndim != 2 or
        bits0_st.shape != bits1_st.shape or bits0_st.shape[1] != PUBLIC_CODE_WIDTH or
        bits0_st.shape[0] < 2 or sex.ndim != 1 or sex.shape[0] != bits0_st.shape[0]):
        raise ValueError("aligned nonempty N-by-10 hard codes and N SEX labels required")
    if (bits0_st.device != bits1_st.device or bits0_st.device != sex.device or
        not bits0_st.is_floating_point() or not bits1_st.is_floating_point() or
        sex.dtype != torch.int64):
        raise ValueError("code tensors must float and SEX int64 on one device")
    for bits in (bits0_st, bits1_st):
        hard = bits.detach()
        if (not bool(torch.isfinite(hard).all()) or
            not bool(((hard - hard.round()).abs() <= 1e-6).all()) or
            not bool(((hard >= -1e-6) & (hard <= 1 + 1e-6)).all())):
            raise ValueError("code forward values must be finite binary bits")
    if (not bool(((sex == 0) | (sex == 1)).all()) or
        not bool((sex == 0).any()) or not bool((sex == 1).any())):
        raise ValueError("both binary SEX groups required")


def _multiscale_gram(bits_st: torch.Tensor) -> torch.Tensor:
    values = bits_st.to(dtype=torch.float64)
    squared_distance = (values[:, None, :] - values[None, :, :]).square().sum(dim=-1)
    return sum(torch.exp(-gamma * squared_distance) for gamma in KERNEL_GAMMAS) / len(KERNEL_GAMMAS)


def _biased_mmd2(kernel: torch.Tensor, group0: torch.Tensor, group1: torch.Tensor) -> torch.Tensor:
    raw = (kernel[group0][:, group0].mean() + kernel[group1][:, group1].mean()
           - 2.0 * kernel[group0][:, group1].mean())
    if float(raw.detach()) < -1e-10:
        raise ArithmeticError("PSD-kernel MMD squared was materially negative")
    return raw.clamp_min(0.0)
