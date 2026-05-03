"""Hilbert-Schmidt Independence Criterion (HSIC) — linear-kernel variants.

Replaces the in-batch OLS R² primal signal in BIOS Phase 1. Key property:
**unbiased** linear HSIC has expectation 0 under independence at any (n, d).
The d/N pathology that crippled OLS R² at d=768, batch=32 (saturating at 1.0
on pure noise) does not appear here.

Reference:
- Song, Smola, Gretton, Bedo, Borgwardt (2012). "Feature Selection via
  Dependence Maximization." JMLR 13:1393–1434. The unbiased HSIC₁ form
  (Equation 5) sets the kernel diagonals to zero, removing the bias term
  and yielding E[HSIC₁(X, Y)] = 0 under H₀ in finite samples.
- Pérez-Suay et al. 2017 (arXiv:1710.05578); Greenfeld & Shalit 2020
  (arXiv:1910.00270); Li et al. 2019 (arXiv:1911.04322) — HSIC fairness
  precedents the Round-2 plan cites.
- FFB benchmark (arXiv:2306.09468): HSIC > LAFTR/DANN on tabular fairness.

Normalized form (nHSIC) divides by sqrt(HSIC(X, X) * HSIC(Y, Y)) for a
Cauchy-Schwarz-bounded value in [0, 1] with semantics comparable to a
correlation coefficient. We use this as the differentiable primal signal so
``λ · nHSIC`` has bounded magnitude regardless of [CLS]-norm drift.
"""
from __future__ import annotations

import torch


def hsic_unbiased_linear(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC₁ (Song et al. 2012, eq. 5) with linear kernels.

    Args:
        X: ``(n, d_x)`` features.
        Y: ``(n, d_y)`` features (e.g. one-hot gender).

    Returns:
        Scalar tensor (differentiable). Returns 0 if ``n < 4`` (estimator
        is undefined; Song et al. requires ``n ≥ 4``).
    """
    n = X.shape[0]
    if n < 4:
        return torch.zeros((), device=X.device, dtype=X.dtype)
    K = X @ X.t()              # (n, n)
    L = Y @ Y.t()              # (n, n)
    # Song et al. zero-diagonal trick: removes the contribution of (i, i)
    # pairs, eliminating the bias term in the V-statistic.
    K = K - torch.diag(torch.diag(K))
    L = L - torch.diag(torch.diag(L))

    one_K_one = K.sum()
    one_L_one = L.sum()
    one_KL_one = (K @ L).sum()
    tr_KL = (K * L).sum()

    nf = float(n)
    hsic = (1.0 / (nf * (nf - 3.0))) * (
        tr_KL
        + one_K_one * one_L_one / ((nf - 1.0) * (nf - 2.0))
        - 2.0 * one_KL_one / (nf - 2.0)
    )
    return hsic


def nhsic_linear(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Normalized linear HSIC: ``HSIC(X, Y) / sqrt(HSIC(X, X) · HSIC(Y, Y))``.

    By Cauchy-Schwarz, the numerator is bounded by the denominator, so the
    result lies in ``[0, 1]`` (with possible numerical drift outside that
    range — clamped here for caller convenience).

    Args:
        X: ``(n, d_x)`` features.
        Y: ``(n, d_y)`` features (e.g. one-hot gender).

    Returns:
        Scalar tensor in ``[0, 1]``. Differentiable through both inputs.
    """
    h_xy = hsic_unbiased_linear(X, Y)
    h_xx = hsic_unbiased_linear(X, X)
    h_yy = hsic_unbiased_linear(Y, Y)
    denom = torch.sqrt(torch.clamp(h_xx * h_yy, min=1e-12))
    val = h_xy / denom
    # Numerical guard: values can drift slightly outside [0, 1] due to fp32
    # error on the unbiased estimator. Clamp without breaking gradient flow
    # for in-range values (clamp_min only kicks in below 0).
    return torch.clamp(val, min=0.0, max=1.0)


def gender_one_hot(g_int: torch.Tensor, n_classes: int = 2) -> torch.Tensor:
    """One-hot expansion for HSIC consumption. ``g_int`` is ``(n,)`` long."""
    return torch.eye(n_classes, device=g_int.device, dtype=torch.float32)[g_int]
