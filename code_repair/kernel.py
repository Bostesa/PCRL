"""Capacity-aware version of the predecessor's fixed hard-code MMD.

The float64 forward uses actual hard bits. A straight-through caller supplies
surrogate gradients to the discontinuous mapper; MMD is no privacy guarantee.
"""

from dataclasses import dataclass

import torch


GAMMAS = (0.25, 0.5, 1.0, 2.0)
ALLOWED_WIDTHS = (10, 16)


@dataclass(frozen=True)
class MMDTerms:
    local0: torch.Tensor
    local1: torch.Tensor
    pair: torch.Tensor

    @property
    def joint(self) -> torch.Tensor:
        return (self.local0 + self.local1 + self.pair) / 3

    @property
    def local(self) -> torch.Tensor:
        return (self.local0 + self.local1) / 3


def _validate(bits0: torch.Tensor, bits1: torch.Tensor, sex: torch.Tensor) -> None:
    if not all(isinstance(x, torch.Tensor) for x in (bits0, bits1, sex)):
        raise ValueError("codes and SEX must be tensors")
    if (bits0.ndim != 2 or bits0.shape != bits1.shape
            or bits0.shape[1] not in ALLOWED_WIDTHS or bits0.shape[0] < 2
            or sex.shape != (bits0.shape[0],)
            or not bits0.is_floating_point() or not bits1.is_floating_point()
            or sex.dtype != torch.int64
            or bits0.device != bits1.device or bits0.device != sex.device):
        raise ValueError("aligned 10- or 16-bit hard codes and SEX required")
    for bits in (bits0, bits1):
        forward = bits.detach()
        if (not bool(torch.isfinite(forward).all())
                or not bool(((forward == 0) | (forward == 1)).all())):
            raise ValueError("code forward values must be binary")
    if (not bool(((sex == 0) | (sex == 1)).all())
            or not bool((sex == 0).any()) or not bool((sex == 1).any())):
        raise ValueError("both SEX groups required")


def _gram(bits: torch.Tensor) -> torch.Tensor:
    value = bits.to(torch.float64)
    squared_distance = (value[:, None, :] - value[None, :, :]).square().sum(-1)
    return sum(torch.exp(-gamma * squared_distance) for gamma in GAMMAS) / len(GAMMAS)


def _mmd2(kernel: torch.Tensor, zero: torch.Tensor, one: torch.Tensor) -> torch.Tensor:
    raw = (kernel[zero][:, zero].mean() + kernel[one][:, one].mean()
           - 2 * kernel[zero][:, one].mean())
    if float(raw.detach()) < -1e-10:
        raise ArithmeticError("materially negative MMD squared")
    return raw.clamp_min(0)


def hard_code_mmd(bits0: torch.Tensor, bits1: torch.Tensor,
                  sex: torch.Tensor) -> MMDTerms:
    """Return local and aligned product-kernel biased MMD squared terms."""
    _validate(bits0, bits1, sex)
    first, second = _gram(bits0), _gram(bits1)
    zero, one = sex == 0, sex == 1
    return MMDTerms(_mmd2(first, zero, one), _mmd2(second, zero, one),
                    _mmd2(first * second, zero, one))
