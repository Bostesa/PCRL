"""VICReg variance and covariance regularizer.

Architecturally prevents representation collapse by hinging per-dimension
standard deviation above a target gamma, and by penalizing off-diagonal
entries of the empirical covariance matrix to decorrelate dimensions.
The invariance term of the original VICReg is not included here — that
component is task-specific and supplied externally.

Ref:
    Bardes et al., ICLR 2022, "VICReg: Variance-Invariance-Covariance
        Regularization for Self-Supervised Learning."
"""

import torch
import torch.nn.functional as F


def variance_loss(X: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Hinge loss penalizing per-dimension std below gamma.

    Args:
        X: [batch, dim] representation tensor.
        gamma: target std (default 1.0).
        eps: numerical stability inside the sqrt.

    Returns:
        Scalar loss = mean over dims of max(0, gamma - sigma_d).
    """
    sigma = torch.sqrt(X.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - sigma))


def covariance_loss(X: torch.Tensor) -> torch.Tensor:
    """Off-diagonal covariance penalty.

    Args:
        X: [batch, dim] representation tensor.

    Returns:
        Scalar loss = sum of squared off-diagonal entries / dim.
    """
    n, d = X.shape
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov = (X_centered.T @ X_centered) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).sum() / d


def vicreg_loss(
    X: torch.Tensor,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Combined VICReg variance + covariance regularizer.

    The invariance term is task-specific (handled by L_task) and not included.
    Defaults follow Bardes et al. ICLR 2022.

    Args:
        X: [batch, dim] representation.
        lambda_var: weight on the variance hinge term.
        lambda_cov: weight on the off-diagonal covariance term.
        gamma: target per-dim std.
    """
    return lambda_var * variance_loss(X, gamma=gamma) + lambda_cov * covariance_loss(X)
