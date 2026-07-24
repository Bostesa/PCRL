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
    output (``B(A(x)) * scaling + bias``); the host Linear itself still
    contributes its own ``W @ x + b`` term independently. The wrapper's
    forward hook adds the two together at runtime.

    The optional ``bias`` parameter (zero-initialised) lets a closed-form
    initialiser (e.g. LEACE) absorb the affine translation that a pure
    rank-r factor cannot represent. Standard LoRA training leaves it at
    zero, recovering the canonical no-bias-on-adapter behaviour.

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
        # Optional adapter-side bias (translation). LEACE's affine eraser
        # has a translation component (I-Q)μ that a rank-r BA factor
        # cannot represent; this parameter absorbs it. Zero by default.
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming-normal A (small variance), zero B. The zero B means the
        # adapter contribution starts at exactly zero, which is the
        # standard LoRA initialisation: the wrapped encoder is identical
        # to the frozen backbone before any training.
        nn.init.kaiming_normal_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(self.dropout(x))) * self.scaling + self.bias

    @torch.no_grad()
    def zero_out(self) -> None:
        """Zero the adapter contribution (B and bias; A may stay non-zero)."""
        self.B.weight.zero_()
        self.bias.zero_()


class LinearAdapter(nn.Module):
    """Full-rank per-purpose linear adapter for one Linear layer.

    Ablation counterpart to :class:`LoRAAdapter` (FAccT resubmission
    Ablation 1): the adapter contribution is an unfactored delta
    ``ΔW @ x + bias`` with ``ΔW`` of shape (out, in), zero-initialised so
    the wrapped encoder equals the frozen backbone at construction —
    identical starting point to zero-B LoRA. Equivalent to LoRA with
    rank = min(in, out) and no scaling, but parameterised directly.

    Args:
        in_features: Input dim of the host Linear.
        out_features: Output dim of the host Linear.
        dropout: Optional dropout on the adapter input ``x``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.delta = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.delta.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.delta(self.dropout(x)) + self.bias

    @torch.no_grad()
    def zero_out(self) -> None:
        """Zero the adapter contribution (delta and bias)."""
        self.delta.weight.zero_()
        self.bias.zero_()


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
        rank: Rank for every adapter (ignored when ``adapter_type="linear"``).
        alpha: LoRA alpha for every adapter (defaults to ``rank``; ignored
            when ``adapter_type="linear"``).
        dropout: Dropout applied on each adapter's input.
        adapter_type: ``"lora"`` (default, rank-r factorised delta) or
            ``"linear"`` (full-rank unfactored delta — Ablation 1 arm).
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_purposes: int,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
        lora_target: str = "all_linear",
        adapter_type: str = "lora",
    ) -> None:
        super().__init__()
        if n_purposes <= 0:
            raise ValueError(f"n_purposes must be positive, got {n_purposes}")
        if lora_target not in {"all_linear", "repr_proj_only"}:
            raise ValueError(
                f"lora_target must be 'all_linear' or 'repr_proj_only'; got {lora_target!r}"
            )
        if adapter_type not in {"lora", "linear"}:
            raise ValueError(
                f"adapter_type must be 'lora' or 'linear'; got {adapter_type!r}"
            )

        self.backbone = backbone
        self.n_purposes = n_purposes
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.lora_target = lora_target
        self.adapter_type = adapter_type

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
        #
        # `_skip_lora=True` modules (e.g. the frozen LEACE erase layer) are
        # excluded so adapters don't bypass the structural compliance.
        # `lora_target="repr_proj_only"` further restricts adapters to the
        # final Linear (mirrors §5.5 vision: only task_proj gets LoRA).
        all_linears = [
            m for m in self.backbone.modules() if isinstance(m, nn.Linear)
        ]
        all_linears = [m for m in all_linears if not getattr(m, "_skip_lora", False)]
        if lora_target == "repr_proj_only":
            if not all_linears:
                raise ValueError(
                    "backbone has no LoRA-eligible nn.Linear modules — LoRA has nothing to adapt"
                )
            self._linear_modules: list[nn.Linear] = [all_linears[-1]]
        else:
            self._linear_modules = all_linears
        if not self._linear_modules:
            raise ValueError(
                "backbone has no nn.Linear modules — LoRA has nothing to adapt"
            )

        # adapters[p][i] adapts self._linear_modules[i] for purpose p.
        self.adapters: nn.ModuleList = nn.ModuleList()
        for _ in range(n_purposes):
            if adapter_type == "linear":
                per_purpose = nn.ModuleList(
                    [
                        LinearAdapter(
                            in_features=lin.in_features,
                            out_features=lin.out_features,
                            dropout=dropout,
                        )
                        for lin in self._linear_modules
                    ]
                )
            else:
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
            h = self.backbone(x)
        finally:
            for handle in handles:
                handle.remove()

        # Frozen LEACE projection (opt-in via ``set_leace_projection``). When a
        # buffer ``leace_P_p{p}`` is registered for this purpose the LoRA-adapted
        # output is hard-projected back to the LEACE null space at every forward
        # pass: ``h_proj = h - P (h - μ)`` for the row-vector convention,
        # implemented as ``h - (h - μ) @ P.T`` because ``h`` has shape (B, d).
        # P is the concept-subspace projection (``I − eraser.P`` from
        # concept_erasure), so this exactly reproduces the LEACE eraser output
        # ``μ + (h − μ) @ eraser.P.T`` while keeping (P, μ) as non-trainable
        # buffers — Cui Wang Ning 2025 precedent (kernelised INLP buffer).
        # Idempotent w.r.t. the LoRA-side LEACE warm-start: re-applying LEACE to
        # an already-erased representation is a no-op (P² = P).
        buf_name_P = f"leace_P_p{p}"
        if hasattr(self, buf_name_P):
            P = getattr(self, buf_name_P)
            mu = getattr(self, f"leace_mu_p{p}")
            h = h - (h - mu) @ P.T
        return h

    @torch.no_grad()
    def set_leace_projection(
        self, purpose_idx: int, P: torch.Tensor, mu: torch.Tensor,
    ) -> None:
        """Register a non-trainable LEACE projection buffer for ``purpose_idx``.

        Once set, ``forward(x, purpose_idx)`` applies
        ``h_proj = h - (h - μ) @ P.T`` after the LoRA-adapted output ``h``,
        hard-projecting the representation onto the LEACE null space at every
        call. Drift along erased directions is removed; gradient updates to the
        LoRA adapters can only move ``h`` in the orthogonal complement of P.

        Args:
            purpose_idx: Which purpose to install the projection for.
            P: Concept-subspace projection matrix (``I − eraser.P`` from
                concept_erasure), shape ``(d, d)``.
            mu: LEACE shift vector (``eraser.bias``), shape ``(d,)``. Pass zeros
                if the eraser was fitted with ``affine=False``.

        Both tensors are stored as non-persistent... actually as persistent
        buffers so they survive ``state_dict()``/``load_state_dict()`` round
        trips and are checkpointed automatically. ``requires_grad=False`` is
        forced because buffers cannot accumulate gradients, but gradients still
        flow *through* the projection during backprop.
        """
        p = _coerce_purpose_idx(purpose_idx, self.n_purposes)
        if P.dim() != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(f"P must be a square (d, d) matrix; got {tuple(P.shape)}")
        if mu.dim() != 1 or mu.shape[0] != P.shape[0]:
            raise ValueError(
                f"mu must be (d,) matching P; got mu={tuple(mu.shape)} P={tuple(P.shape)}"
            )
        # Move buffer to the encoder's current device so the forward pass
        # (which runs on whatever device the encoder is on) doesn't hit a
        # CPU/GPU device mismatch when concept_erasure produces CPU tensors
        # but the encoder has been ``.to(cuda)``'d. Falls back to the input
        # tensor's device when the encoder has no parameters yet.
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = P.device
        self.register_buffer(
            f"leace_P_p{p}", P.detach().to(target_device).clone(), persistent=True,
        )
        self.register_buffer(
            f"leace_mu_p{p}", mu.detach().to(target_device).clone(), persistent=True,
        )

    def has_leace_projection(self, purpose_idx: int) -> bool:
        p = _coerce_purpose_idx(purpose_idx, self.n_purposes)
        return hasattr(self, f"leace_P_p{p}")

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only LoRA adapter parameters (backbone is frozen)."""
        for p in self.adapters.parameters():
            yield p

    def n_trainable(self) -> int:
        """Total trainable parameter count (LoRA adapters only)."""
        return sum(p.numel() for p in self.adapters.parameters())

    @torch.no_grad()
    def init_last_layer_from_affine(
        self, purpose_idx: int, Q: torch.Tensor, c: torch.Tensor,
    ) -> None:
        """Initialise this purpose's last-Linear LoRA to realise z' = Q z + c.

        For the host Linear (W, b) on the final repr_proj, the modified
        output we want is

            z'_p(h) = Q (W h + b) + c
                    = (Q W) h + (Q b + c)

        The LoRA contribution is ``s · BA · h + d`` (with s=alpha/rank and
        d the adapter-side bias). Matching:

            s · BA = (Q − I) W       (rank-r factor of a (out, in) matrix)
            d      = (Q − I) b + c   (closed form for the translation)

        ``s · BA`` is rank-r; if rank(Q − I) ≤ r — which holds when the
        concept Z spans ≤ r linearly-independent directions — the SVD
        truncation is exact. Otherwise it is the best rank-r approximation
        in Frobenius norm.

        Hidden-layer adapters are left at zero so the projection is
        applied only at the output step. Adapter A is set to a (rank,
        in_features) matrix whose rows form the right singular basis
        (orthonormal) so subsequent gradient updates can rotate inside
        that subspace without immediately undoing the LEACE structure.
        """
        last_idx = len(self._linear_modules) - 1
        host = self._linear_modules[last_idx]
        adapter = self.adapters[purpose_idx][last_idx]

        W = host.weight.detach().to(Q.device)  # (out, in)
        b = (
            host.bias.detach().to(Q.device)
            if host.bias is not None
            else torch.zeros(host.out_features, device=Q.device)
        )

        out_dim, in_dim = W.shape
        if Q.shape != (out_dim, out_dim):
            raise ValueError(
                f"Q must be ({out_dim}, {out_dim}); got {tuple(Q.shape)}"
            )
        if c.shape != (out_dim,):
            raise ValueError(f"c must be ({out_dim},); got {tuple(c.shape)}")

        target = (Q - torch.eye(out_dim, device=Q.device)) @ W  # (out, in)

        if isinstance(adapter, LinearAdapter):
            # Full-rank adapter realises z' = Q z + c EXACTLY: no SVD
            # truncation needed. ΔW = (Q − I) W, bias = (Q − I) b + c.
            adapter.delta.weight.copy_(target.to(adapter.delta.weight.dtype))
            adapter.bias.copy_(
                ((Q - torch.eye(out_dim, device=Q.device)) @ b + c).to(adapter.bias.dtype)
            )
            return

        # Rank-r SVD truncation: target ≈ U_r diag(σ_r) V_r^T
        U, S, Vh = torch.linalg.svd(target, full_matrices=False)
        r = adapter.rank
        U_r = U[:, :r]                       # (out, r)
        S_r = S[:r]                          # (r,)
        Vh_r = Vh[:r, :]                     # (r, in)

        # Distribute σ between A and B. Standard LoRA convention is to
        # absorb σ into B (so A has unit-norm rows from V^T). Equivalent
        # under gradient descent; A's rows form an orthonormal basis of
        # the input-side subspace where the projection acts.
        A_w = Vh_r                           # (r, in)
        B_w = U_r * S_r.unsqueeze(0)         # (out, r)

        # The adapter applies scaling s = alpha/r on top of (B_w A_w).
        # We want the effective contribution to be `target = (Q-I) W`,
        # so divide by s here.
        adapter.A.weight.copy_(A_w.to(adapter.A.weight.dtype))
        adapter.B.weight.copy_((B_w / adapter.scaling).to(adapter.B.weight.dtype))
        adapter.bias.copy_(((Q - torch.eye(out_dim, device=Q.device)) @ b + c).to(adapter.bias.dtype))
