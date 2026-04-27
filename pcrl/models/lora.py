"""Per-purpose LoRA adapters for PCRL v2.

Wraps a frozen backbone encoder with Low-Rank Adaptation matrices, one
``(A_p, B_p)`` pair per purpose per Linear layer. Replaces FiLM modulation
in v2: the same backbone is shared across all purposes (so generic task
features are preserved), and each purpose's adapter is the only set of
parameters that changes between purposes. Collapse on one purpose's
adapters cannot affect any other purpose's representations.

Reference: Hu et al. (ICLR 2022), "LoRA: Low-Rank Adaptation of Large
Language Models".

Adapter math (per Linear with weight W of shape (out, in)):

    h = W @ x + (B_p @ A_p) @ x * (alpha / r)

A_p has shape (r, in), B_p has shape (out, r). A is Kaiming-normal init,
B is zero-initialised so the adapter contribution is exactly zero before
any training (i.e. ``encoder(x, p) == backbone(x)`` at construction).

The wrapper finds Linear modules inside the backbone via ``.modules()``
and registers forward hooks during ``forward`` to inject the adapter
contribution. Hooks are removed in a ``finally`` block so an exception
in the backbone never leaves stale hooks attached.

Limitations / deviations from the spec:

* ``purpose_idx`` must be a single Python ``int`` or a 0-dim / single-
  element tensor. Mixed per-sample purpose indices are not supported,
  because hook-based dispatch can only inject one adapter per forward
  pass. This matches every existing PCRL training call site, which
  always passes a constant purpose index per batch.
* The backbone's ``forward`` is called with just ``x`` (no
  ``purpose_idx``). For backbones whose ``forward`` accepts
  ``purpose_idx`` only as an optional kwarg (e.g. ``StandardEncoder``)
  this is a drop-in. For backbones that *require* a purpose index
  (e.g. the v1 ``PurposeConditionedEncoder``) the user should pass a
  conditioning-free wrapper (typically ``StandardEncoder``) or a
  ``functools.partial`` binding.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """Single LoRA adapter for one Linear layer.

    Computes only the **adapter contribution** to the host Linear's
    output (``B(A(x)) * scaling``); the host Linear itself still
    contributes its own ``W @ x + b`` term independently. The wrapper's
    forward hook adds the two together at runtime.

    Args:
        in_features: Input dim of the host Linear.
        out_features: Output dim of the host Linear.
        rank: LoRA rank ``r``. Smaller = fewer adapter params.
        alpha: LoRA scaling. Defaults to ``rank`` (so ``alpha/r = 1``).
        dropout: Optional dropout on the adapter input ``x``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / float(rank)

        # A is (rank, in_features); B is (out_features, rank). Both
        # implemented as nn.Linear for parameter accounting; biases are
        # disabled because the host Linear already has its own bias.
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming-normal A (small variance), zero B. The zero B means the
        # adapter contribution starts at exactly zero, which is the
        # standard LoRA initialisation: the wrapped encoder is identical
        # to the frozen backbone before any training.
        nn.init.kaiming_normal_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(self.dropout(x))) * self.scaling


def _coerce_purpose_idx(purpose_idx: int | torch.Tensor, n_purposes: int) -> int:
    """Validate and reduce purpose_idx to a single int."""
    if isinstance(purpose_idx, torch.Tensor):
        if purpose_idx.numel() != 1:
            raise ValueError(
                "PerPurposeLoRAEncoder requires a scalar purpose_idx; "
                f"got tensor of shape {tuple(purpose_idx.shape)} with "
                f"{purpose_idx.numel()} elements"
            )
        purpose_idx = int(purpose_idx.item())
    if not isinstance(purpose_idx, int):
        raise ValueError(
            f"purpose_idx must be int or scalar tensor, got "
            f"{type(purpose_idx).__name__}"
        )
    if purpose_idx < 0 or purpose_idx >= n_purposes:
        raise ValueError(
            f"purpose_idx={purpose_idx} out of range [0, {n_purposes})"
        )
    return purpose_idx


class PerPurposeLoRAEncoder(nn.Module):
    """Frozen backbone encoder + per-purpose LoRA adapters.

    The backbone's parameters are frozen (``requires_grad=False``). For
    each Linear layer in the backbone and each purpose, a ``LoRAAdapter``
    is added to the layer's output during ``forward(x, purpose_idx)``.

    The backbone is called with the input tensor only — its ``forward``
    must therefore accept ``forward(x)``. ``StandardEncoder`` already
    satisfies this (its second arg defaults to ``None``).

    Args:
        backbone: An ``nn.Module`` whose ``forward`` takes a single
            tensor (or has additional args defaulted). Must contain at
            least one ``nn.Linear`` submodule.
        n_purposes: Number of distinct purposes (``|P|``).
        rank: Rank for every adapter.
        alpha: LoRA alpha for every adapter (defaults to ``rank``).
        dropout: Dropout applied on each adapter's input.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_purposes: int,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_purposes <= 0:
            raise ValueError(f"n_purposes must be positive, got {n_purposes}")

        self.backbone = backbone
        self.n_purposes = n_purposes
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else float(rank)

        # Freeze backbone parameters in-place. We don't strip them from
        # state_dict so checkpoints remain self-contained, but they will
        # not appear in `trainable_parameters()`.
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Reference list (NOT a submodule list) of Linear layers inside
        # the backbone, in traversal order. ``self.backbone`` is the
        # registered submodule that owns these Linears, so we don't need
        # to register them again — but we do need to keep the references
        # for the forward hook step.
        self._linear_modules: list[nn.Linear] = [
            m for m in self.backbone.modules() if isinstance(m, nn.Linear)
        ]
        if not self._linear_modules:
            raise ValueError(
                "backbone has no nn.Linear modules — LoRA has nothing to adapt"
            )

        # adapters[p][i] adapts self._linear_modules[i] for purpose p.
        self.adapters: nn.ModuleList = nn.ModuleList()
        for _ in range(n_purposes):
            per_purpose = nn.ModuleList(
                [
                    LoRAAdapter(
                        in_features=lin.in_features,
                        out_features=lin.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    for lin in self._linear_modules
                ]
            )
            self.adapters.append(per_purpose)

    @staticmethod
    def _make_hook(adapter: LoRAAdapter):
        def hook(_module, inputs, output):
            return output + adapter(inputs[0])
        return hook

    def forward(self, x: torch.Tensor, purpose_idx: int | torch.Tensor) -> torch.Tensor:
        """Encode ``x`` with the adapter for purpose ``purpose_idx``."""
        p = _coerce_purpose_idx(purpose_idx, self.n_purposes)

        handles = []
        try:
            for linear, adapter in zip(self._linear_modules, self.adapters[p]):
                handles.append(linear.register_forward_hook(self._make_hook(adapter)))
            return self.backbone(x)
        finally:
            for h in handles:
                h.remove()

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only LoRA adapter parameters (backbone is frozen)."""
        for p in self.adapters.parameters():
            yield p

    def n_trainable(self) -> int:
        """Total trainable parameter count (LoRA adapters only)."""
        return sum(p.numel() for p in self.adapters.parameters())
