"""Ordinary adversarial comparator on the transmitted hard-code bits.

The critics are training devices, never part of the service wire. Fresh
complete-view attacks are required for the scientific endpoint.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def balanced_bce(logits: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
    """Equal-weight mean SEX cross entropy in each fitting batch."""
    if logits.ndim != 1 or sex.shape != logits.shape or not torch.isfinite(logits).all():
        raise ValueError("one finite critic logit and SEX label per row required")
    if not bool(torch.all((sex == 0) | (sex == 1))):
        raise ValueError("SEX must be binary")
    zero, one = sex == 0, sex == 1
    if not bool(zero.any()) or not bool(one.any()):
        raise ValueError("both SEX groups must occur in a critic minibatch")
    return (F.binary_cross_entropy_with_logits(logits[zero], sex[zero].float())
            + F.binary_cross_entropy_with_logits(logits[one], sex[one].float())) / 2


def _critic(width: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))


class HardCodeCritics(nn.Module):
    """Two local and one coalition hard-bit readers for the matched C arm."""

    def __init__(self, bits: int = 10, hidden: int = 64):
        super().__init__()
        if bits < 1 or hidden < 1:
            raise ValueError("positive bit and hidden widths required")
        self.bits = bits
        self.local_1 = _critic(bits, hidden)
        self.local_2 = _critic(bits, hidden)
        self.pair = _critic(2 * bits, hidden)

    def _views(self, bits1: torch.Tensor, bits2: torch.Tensor) -> dict[str, torch.Tensor]:
        if (bits1.ndim != 2 or bits1.shape != bits2.shape or bits1.shape[1] != self.bits
                or not torch.isfinite(bits1).all() or not torch.isfinite(bits2).all()
                or not bool(torch.all((bits1 == 0) | (bits1 == 1)))
                or not bool(torch.all((bits2 == 0) | (bits2 == 1)))):
            raise ValueError("critics require aligned hard binary public codes")
        return {"local_1": bits1, "local_2": bits2,
                "pair": torch.cat((bits1, bits2), dim=1)}

    def losses(self, bits1: torch.Tensor, bits2: torch.Tensor,
               sex: torch.Tensor) -> dict[str, torch.Tensor]:
        views = self._views(bits1, bits2)
        return {name: balanced_bce(getattr(self, name)(value).squeeze(1), sex)
                for name, value in views.items()}

    def critic_step(self, bits1: torch.Tensor, bits2: torch.Tensor,
                    sex: torch.Tensor, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        """One critic update; detached codes cannot train the mapper here."""
        losses = self.losses(bits1.detach(), bits2.detach(), sex)
        loss = torch.stack(tuple(losses.values())).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {name: float(value.detach()) for name, value in losses.items()}

    def mapper_loss(self, bits1: torch.Tensor, bits2: torch.Tensor,
                    sex: torch.Tensor, coefficient: float) -> torch.Tensor:
        """Gradient reversal: maximize fitted readers' balanced cross entropy."""
        if not 0 <= coefficient < float("inf"):
            raise ValueError("finite nonnegative adversarial coefficient required")
        parameters = tuple(self.parameters())
        previous = tuple(parameter.requires_grad for parameter in parameters)
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            losses = self.losses(bits1, bits2, sex)
            return -coefficient * torch.stack(tuple(losses.values())).mean()
        finally:
            for parameter, required in zip(parameters, previous):
                parameter.requires_grad_(required)
