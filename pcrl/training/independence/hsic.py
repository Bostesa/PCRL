"""Hilbert-Schmidt Independence Criterion (HSIC) penalty.

Computes a normalized HSIC dependence measure between a continuous
representation X and a sensitive attribute Z (categorical or continuous).

The biased empirical HSIC numerator follows Gretton et al.:
    H(X,Z) = tr(K_x H K_z H) / (n - 1)^2
The returned value is the normalized variant
    nHSIC(X,Z) = H(X,Z) / sqrt(H(X,X) * H(Z,Z))
which lies in [0, 1] (Cauchy-Schwarz). Normalization is needed because the
absolute scale of the biased numerator is O(1/n) for typical kernels,
giving values too small to be useful as a stand-alone fairness loss when
n is large; nHSIC is dimension- and scale-invariant and remains an
independence diagnostic (zero iff independent under the chosen kernels).

References:
    Pérez-Suay et al. 2017, "Fair Kernel Learning."
    Greenfeld & Shalit 2020, "Robust Learning with the Hilbert-Schmidt
        Independence Criterion."
"""

from typing import Optional

import torch


def gaussian_kernel(X: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """RBF kernel matrix K[i,j] = exp(-||x_i - x_j||^2 / (2 * sigma^2)).

    If sigma is None, set via the median heuristic on the upper-triangular
    pairwise distances.

    Args:
        X: [n, d] tensor.
        sigma: bandwidth; if None, median heuristic is used.

    Returns:
        [n, n] kernel matrix on the same device/dtype as X.
    """
    if X.dim() == 1:
        X = X.unsqueeze(-1)
    n = X.shape[0]
    dists = torch.cdist(X, X, p=2)

    if sigma is None:
        if n < 2:
            sigma_val = torch.tensor(1.0, device=X.device, dtype=X.dtype)
        else:
            iu = torch.triu_indices(n, n, offset=1, device=X.device)
            upper = dists[iu[0], iu[1]]
            med = torch.median(upper)
            sigma_val = torch.where(med > 0, med, torch.ones_like(med))
    else:
        sigma_val = torch.tensor(float(sigma), device=X.device, dtype=X.dtype)

    return torch.exp(-(dists ** 2) / (2.0 * sigma_val ** 2))


def delta_kernel(Z: torch.Tensor) -> torch.Tensor:
    """Indicator kernel K[i,j] = 1 if Z[i] == Z[j] else 0.

    For categorical Z. Z may be [n] or [n, 1] with integer dtype.

    Args:
        Z: [n] or [n, 1] integer tensor.

    Returns:
        [n, n] float kernel matrix.
    """
    if Z.dim() > 1:
        Z = Z.view(-1)
    eq = Z.unsqueeze(0) == Z.unsqueeze(1)
    return eq.to(torch.get_default_dtype())


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Double-center K via H K H where H = I - (1/n) 1 1^T."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device, dtype=K.dtype) - 1.0 / n
    return H @ K @ H


def _is_categorical(Z: torch.Tensor) -> bool:
    """Z uses delta kernel iff its dtype is integer/bool.

    1D float Z is treated as continuous (reshaped to [n, 1] and run through
    the Gaussian kernel); using the delta kernel on continuous floats would
    collapse K_z to the identity matrix and make HSIC degenerate.
    """
    return Z.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
        torch.uint8,
        torch.bool,
    )


def _hsic_biased(Kx_centered: torch.Tensor, Kz_centered: torch.Tensor, n: int) -> torch.Tensor:
    """tr(K_x H K_z H) / (n-1)^2 from already-centered Gram matrices."""
    return torch.sum(Kx_centered * Kz_centered) / float((n - 1) ** 2)


def hsic(
    X: torch.Tensor,
    Z: torch.Tensor,
    sigma_x: Optional[float] = None,
    sigma_z: Optional[float] = None,
) -> torch.Tensor:
    """Normalized HSIC dependence measure.

    Internally computes biased HSIC numerator H(X,Z) = tr(K_x H K_z H)/(n-1)^2
    and returns H(X,Z) / sqrt(H(X,X) * H(Z,Z)).

    Args:
        X: [batch, dim_x] continuous features.
        Z: [batch] or [batch, 1] integer for categorical, or
           [batch, dim_z] / [batch] float for continuous.
        sigma_x: bandwidth for X kernel; None uses median heuristic.
        sigma_z: bandwidth for Z kernel (continuous); None uses median
            heuristic. Ignored when Z is categorical.

    Returns:
        Scalar tensor with grad. nHSIC value in [0, 1]; 0 iff X and Z are
        independent under the chosen kernels. Returns 0 when batch size < 3.
    """
    n = X.shape[0]
    if n < 3:
        return torch.zeros((), device=X.device, dtype=X.dtype, requires_grad=X.requires_grad)

    Kx = gaussian_kernel(X, sigma=sigma_x)

    if _is_categorical(Z):
        Kz = delta_kernel(Z)
        if Kz.dtype != Kx.dtype:
            Kz = Kz.to(Kx.dtype)
    else:
        Z_in = Z.to(Kx.dtype)
        if Z_in.dim() == 1:
            Z_in = Z_in.unsqueeze(-1)
        Kz = gaussian_kernel(Z_in, sigma=sigma_z)

    Kx_c = center_kernel(Kx)
    Kz_c = center_kernel(Kz)

    h_xz = _hsic_biased(Kx_c, Kz_c, n)
    h_xx = _hsic_biased(Kx_c, Kx_c, n)
    h_zz = _hsic_biased(Kz_c, Kz_c, n)

    eps = 1e-12
    return h_xz / torch.sqrt(h_xx * h_zz + eps)
