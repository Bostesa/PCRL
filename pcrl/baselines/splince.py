"""SPLINCE baseline (Holstege, Ravfogel, Wouters; NeurIPS 2025; arXiv:2506.10703).

SPLINCE — Simultaneous Projection for LINear concept removal and Covariance
prEservation — is a closed-form linear post-hoc concept eraser. Given features
``x``, a sensitive attribute ``z``, and a target task label ``y``, SPLINCE solves

    arg min_{P}  E[||Px − x||²]
    subject to   P Σ_xz = 0     (kernel: linear guardedness for z)
                 P Σ_xy = Σ_xy  (range: exact preservation of cov(x, y))

Theorem 1 of the paper gives the closed form (for centered ``x``):

    P*_SPLINCE = W^+ V (Uᵀ V)^(-1) Uᵀ W

where W = Σ_xx^{+1/2} is the whitening matrix, the columns of U are an
orthonormal basis of (colsp(W Σ_xz))^⊥ (the "kernel" subspace, dimension
d − k_z), and the columns of V span colsp(W Σ_xy) ⊕ U^- with
U^- = U ∩ (colsp(W Σ_xz) ⊕ colsp(W Σ_xy))^⊥. The assumption U^⊥ ∩
colsp(W Σ_xy) = {0} requires that the concept and task covariance directions
are not perfectly aligned (see Theorem 1 statement in the paper).

This implementation works on torch tensors and is closed form (no iterative
optimisation). The applied transformation is

    r(x) = μ + (x − μ) @ P^T

so the mean is preserved (matches PCRL's LEACE convention).

For the head-to-head with PCRL we wrap a frozen ``PerPurposeLoRAEncoder`` plus
per-purpose SPLINCE projections in :class:`SplinceProjectedEncoder`, mirroring
the INLP/LEACE wrapper pattern in the rest of this package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ───────────────────────────────────────────────────────────────────────────
# Linear-algebra helpers (numpy for clarity; tensors are converted at edges).
# ───────────────────────────────────────────────────────────────────────────


def _orthonormal_basis(M: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis for ``colsp(M)`` via SVD. Returns shape ``(d, k)``
    where ``k = rank(M)``. Empty (d, 0) array if ``M`` is the zero matrix."""
    if M.size == 0 or np.allclose(M, 0):
        return np.zeros((M.shape[0], 0))
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    rank = int((S > tol * S.max()).sum())
    return U[:, :rank]


def _orthogonal_complement(B: np.ndarray, d: int) -> np.ndarray:
    """Orthonormal basis for the orthogonal complement of ``colsp(B)`` in
    ``R^d``. ``B`` is ``(d, k)`` with k <= d; returns ``(d, d − k)``."""
    if B.shape[1] == 0:
        return np.eye(d)
    full = np.eye(d) - B @ B.T  # projector onto the complement
    return _orthonormal_basis(full)


def _whitening_matrices(
    Sigma_xx: np.ndarray, eig_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-inverse square root of a PSD matrix and its inverse.

    Returns ``(W, W_pinv)`` where W = Σ^{+1/2} and W_pinv = Σ^{1/2}.
    Both are symmetric. Eigenvalues below ``eig_tol * max(eig)`` are dropped
    (handled by clamping to zero so the whitening is rank-stable).
    """
    Sigma_xx_sym = 0.5 * (Sigma_xx + Sigma_xx.T)
    eigvals, eigvecs = np.linalg.eigh(Sigma_xx_sym)
    eigvals = np.clip(eigvals, 0.0, None)
    cutoff = eig_tol * eigvals.max() if eigvals.max() > 0 else 0.0
    inv_sqrt = np.where(eigvals > cutoff, 1.0 / np.sqrt(eigvals + 1e-30), 0.0)
    sqrt = np.where(eigvals > cutoff, np.sqrt(eigvals), 0.0)
    W = eigvecs @ np.diag(inv_sqrt) @ eigvecs.T
    W_pinv = eigvecs @ np.diag(sqrt) @ eigvecs.T
    return W, W_pinv


def _onehot_centered(labels: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Centered one-hot encoding of integer labels. Constant columns (a class
    that never appears) are dropped — they would make ``Σ_xz`` rank-deficient
    in a way that confuses the orthonormal-basis step."""
    labels = np.asarray(labels).astype(np.int64)
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    eye = np.eye(n_classes, dtype=np.float64)
    Z = eye[labels]                      # (n, K)
    Z = Z - Z.mean(axis=0, keepdims=True)
    keep = np.linalg.norm(Z, axis=0) > 1e-12
    return Z[:, keep]


def _encode_concept(Z) -> np.ndarray:
    """Coerce a concept input to a centered design matrix.

    Accepts either a 1-D integer array (single multi-class attribute,
    one-hot encoded internally) or a 2-D float array (already encoded,
    just centered). For multi-attribute purposes the caller should pass a
    list/tuple of 1-D integer arrays which are one-hot encoded and column-
    concatenated.
    """
    if isinstance(Z, (list, tuple)):
        parts = [_onehot_centered(np.asarray(z)) for z in Z]
        return np.concatenate(parts, axis=1) if parts else np.zeros((0, 0))
    Z = np.asarray(Z)
    if Z.ndim == 1:
        return _onehot_centered(Z)
    return Z - Z.mean(axis=0, keepdims=True)


# ───────────────────────────────────────────────────────────────────────────
# SPLINCE eraser
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class SplinceFitInfo:
    """Diagnostics from a SPLINCE fit. Stored alongside (P, μ) for paper logs."""

    n: int
    d: int
    k_z: int                            # dim of concept subspace
    k_y: int                            # dim of task subspace
    dim_U_minus: int                    # dim of U ∩ (Z + Y)^⊥
    cond_UtV: float                     # 2-norm condition number of U^T V
    feasible: bool                      # assumption U^⊥ ∩ colsp(WΣ_xy) = {0} held
    fallback_to_leace: bool             # True iff concept/task spans coincided

    def to_dict(self) -> dict:
        return {
            "n": self.n, "d": self.d, "k_z": self.k_z, "k_y": self.k_y,
            "dim_U_minus": self.dim_U_minus, "cond_UtV": self.cond_UtV,
            "feasible": self.feasible, "fallback_to_leace": self.fallback_to_leace,
        }


def fit_splince(
    X: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
    *,
    cond_max: float = 1e6,
) -> tuple[np.ndarray, np.ndarray, SplinceFitInfo]:
    """Fit a SPLINCE eraser.

    Args:
        X: ``(n, d)`` features.
        Z: ``(n,)`` integer concept labels (one-hot encoded internally;
            constant classes dropped).
        Y: ``(n,)`` integer task labels.
        cond_max: If ``cond(U^T V) > cond_max`` we declare the SPLINCE feasibility
            assumption violated for this cell and fall back to LEACE
            (``P*_LEACE = W^+ U U^T W``). The fallback flag is recorded in
            :class:`SplinceFitInfo` so the caller can log it.

    Returns:
        ``(P, mu, info)`` with ``P`` of shape ``(d, d)``, ``mu`` of shape ``(d,)``,
        and ``info`` a :class:`SplinceFitInfo`.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    mu = X.mean(axis=0)
    Xc = X - mu

    Z_oh = _encode_concept(Z)           # (n, K_z) — list/tuple ok for multi-attr
    Y_oh = _encode_concept(Y)           # (n, K_y)

    denom = max(n - 1, 1)
    Sigma_xx = (Xc.T @ Xc) / denom
    Sigma_xz = (Xc.T @ Z_oh) / denom
    Sigma_xy = (Xc.T @ Y_oh) / denom

    W, W_pinv = _whitening_matrices(Sigma_xx)

    WSxz = W @ Sigma_xz                  # (d, K_z)
    WSxy = W @ Sigma_xy                  # (d, K_y)

    Z_basis = _orthonormal_basis(WSxz)   # (d, k_z)
    Y_basis = _orthonormal_basis(WSxy)   # (d, k_y)
    k_z = Z_basis.shape[1]
    k_y = Y_basis.shape[1]

    # U is the orthogonal complement of colsp(WΣ_xz) — kernel of the projection.
    U = _orthogonal_complement(Z_basis, d)              # (d, d - k_z)

    # U^- = U ∩ (Z ⊕ Y)^⊥. Equivalently the orthogonal complement of (Z ⊕ Y).
    if k_z + k_y == 0:
        ZY = np.zeros((d, 0))
    else:
        ZY = _orthonormal_basis(np.concatenate([Z_basis, Y_basis], axis=1))
    U_minus = _orthogonal_complement(ZY, d)             # (d, d - k_z - k_y)
    dim_U_minus = U_minus.shape[1]

    # V = colsp(WΣ_xy) ⊕ U^-. Re-orthogonalise the concatenation.
    if k_y + dim_U_minus == 0:
        V = np.zeros((d, 0))
    else:
        V = _orthonormal_basis(np.concatenate([Y_basis, U_minus], axis=1))

    # The dimensions of U and V should match — both are (d - k_z) by paper.
    if V.shape[1] != U.shape[1]:
        # Concept and task subspaces partially coincide; SPLINCE assumption
        # U^⊥ ∩ colsp(WΣ_xy) = {0} is violated. Fall back to LEACE.
        P_leace = W_pinv @ U @ U.T @ W
        info = SplinceFitInfo(
            n=n, d=d, k_z=k_z, k_y=k_y, dim_U_minus=dim_U_minus,
            cond_UtV=float("nan"), feasible=False, fallback_to_leace=True,
        )
        return P_leace, mu, info

    UtV = U.T @ V                                       # ((d - k_z), (d - k_z))
    cond_UtV = float(np.linalg.cond(UtV)) if UtV.size > 0 else 1.0

    if cond_UtV > cond_max:
        P_leace = W_pinv @ U @ U.T @ W
        info = SplinceFitInfo(
            n=n, d=d, k_z=k_z, k_y=k_y, dim_U_minus=dim_U_minus,
            cond_UtV=cond_UtV, feasible=False, fallback_to_leace=True,
        )
        return P_leace, mu, info

    UtV_inv = np.linalg.solve(UtV, np.eye(UtV.shape[0]))
    P = W_pinv @ V @ UtV_inv @ U.T @ W

    info = SplinceFitInfo(
        n=n, d=d, k_z=k_z, k_y=k_y, dim_U_minus=dim_U_minus,
        cond_UtV=cond_UtV, feasible=True, fallback_to_leace=False,
    )
    return P, mu, info


def apply_splince(
    X: np.ndarray | torch.Tensor,
    P: np.ndarray | torch.Tensor,
    mu: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Apply ``r(x) = μ + (x − μ) @ P^T``. Preserves input type/device."""
    if isinstance(X, torch.Tensor):
        device = X.device
        dtype = X.dtype
        Pt = P if isinstance(P, torch.Tensor) else torch.as_tensor(P, dtype=dtype, device=device)
        mut = mu if isinstance(mu, torch.Tensor) else torch.as_tensor(mu, dtype=dtype, device=device)
        return mut + (X - mut) @ Pt.T
    Pn = P.detach().cpu().numpy() if isinstance(P, torch.Tensor) else np.asarray(P)
    mun = mu.detach().cpu().numpy() if isinstance(mu, torch.Tensor) else np.asarray(mu)
    return mun + (np.asarray(X) - mun) @ Pn.T


# ───────────────────────────────────────────────────────────────────────────
# Encoder wrapper for the eval pipeline
# ───────────────────────────────────────────────────────────────────────────


class SplinceProjectedEncoder(nn.Module):
    """Wrap a base encoder with per-purpose SPLINCE projections.

    Mirrors the INLPEncoder pattern. The base encoder must produce a
    ``(B, d)`` representation given ``(features, purpose_idx)``. SPLINCE
    projections are stored as buffers ``splince_P_p{idx}`` and
    ``splince_mu_p{idx}``; for purposes without a registered projection we
    fall through to the base encoder's output unchanged.

    NOTE: the base encoder must NOT have its own LEACE projection enabled
    when used here, otherwise the comparison conflates SPLINCE with LEACE.
    The benchmark driver loads warmstart checkpoints WITHOUT calling
    ``set_leace_projection`` to keep the encoder pre-projection.
    """

    def __init__(self, base_encoder: nn.Module, n_purposes: int) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.n_purposes = n_purposes

    @torch.no_grad()
    def set_splince_projection(
        self, purpose_idx: int, P: np.ndarray | torch.Tensor,
        mu: np.ndarray | torch.Tensor,
    ) -> None:
        if purpose_idx < 0 or purpose_idx >= self.n_purposes:
            raise ValueError(f"purpose_idx={purpose_idx} out of range")
        if isinstance(P, np.ndarray):
            P = torch.from_numpy(P.astype(np.float32))
        if isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu.astype(np.float32))
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = P.device
        self.register_buffer(
            f"splince_P_p{purpose_idx}", P.to(target_device).float(), persistent=True,
        )
        self.register_buffer(
            f"splince_mu_p{purpose_idx}", mu.to(target_device).float(), persistent=True,
        )

    def forward(
        self, x: torch.Tensor, purpose_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        h = self.base_encoder(x, purpose_idx)
        p = int(purpose_idx) if not isinstance(purpose_idx, torch.Tensor) else int(purpose_idx.item())
        buf_P = f"splince_P_p{p}"
        if hasattr(self, buf_P):
            P = getattr(self, buf_P)
            mu = getattr(self, f"splince_mu_p{p}")
            h = mu + (h - mu) @ P.T
        return h
