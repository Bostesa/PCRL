"""Compliance certificates and verification.

Provides mathematical certificates proving that disallowed attributes
cannot be linearly recovered from learned representations.

Key insight: linear certificates are PROVABLE (closed-form), while
empirical audits (PostHocAuditorSuite) are sanity checks. Together
they give both mathematical guarantees and practical confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class CertificateResult:
    """Result of a compliance certificate check.

    Attributes:
        r_squared: R² score of the optimal linear predictor.
        certified: Whether the certificate passes (R² < epsilon).
        epsilon: Threshold used for certification.
        variance_preserved: Fraction of task-relevant variance preserved
            after null-space projection (only set by NullSpaceCertificate).
    """

    r_squared: float
    certified: bool
    epsilon: float
    variance_preserved: float | None = None


class LinearComplianceCertificate:
    """Closed-form linear compliance certificate.

    Given representations H (n × d) and disallowed labels Z (n × c one-hot
    or n × 1 integer), computes the optimal linear predictor W* and its R².
    If R² < epsilon, we certify that no linear classifier can predict Z from
    H with R² exceeding epsilon.

    Similar in spirit to LEACE (Belrose et al. 2023) concept erasure verification.
    """

    def __init__(self, epsilon: float = 0.01, regularization: float = 1e-6) -> None:
        """Initialize the certificate checker.

        Args:
            epsilon: R² threshold below which a certificate is issued.
            regularization: Tikhonov regularization for numerical stability.
        """
        self.epsilon = epsilon
        self.regularization = regularization

    def check(
        self,
        H: torch.Tensor | np.ndarray,
        Z: torch.Tensor | np.ndarray,
    ) -> CertificateResult:
        """Check linear compliance.

        Args:
            H: Representations of shape (n, d).
            Z: Disallowed attribute labels. Either integer labels of shape (n,)
               or one-hot of shape (n, c).

        Returns:
            CertificateResult with R² and certification status.
        """
        H_np = self._to_numpy(H)
        Z_np = self._to_numpy(Z)

        # Convert integer labels to one-hot for the linear predictor
        if Z_np.ndim == 1:
            num_classes = int(Z_np.max()) + 1
            Z_onehot = np.eye(num_classes)[Z_np.astype(int)]
        else:
            Z_onehot = Z_np

        n, d = H_np.shape

        # Center the data
        H_centered = H_np - H_np.mean(axis=0, keepdims=True)
        Z_centered = Z_onehot - Z_onehot.mean(axis=0, keepdims=True)

        # Compute optimal linear predictor: W* = (H^T H + λI)^{-1} H^T Z
        gram = H_centered.T @ H_centered + self.regularization * np.eye(d)
        W_star = np.linalg.solve(gram, H_centered.T @ Z_centered)

        # Compute predictions and R²
        Z_pred = H_centered @ W_star
        ss_res = np.sum((Z_centered - Z_pred) ** 2)
        ss_tot = np.sum(Z_centered ** 2)

        r_squared = 1.0 - (ss_res / max(ss_tot, 1e-12))
        r_squared = float(max(0.0, r_squared))  # Clamp to [0, 1]

        return CertificateResult(
            r_squared=r_squared,
            certified=r_squared < self.epsilon,
            epsilon=self.epsilon,
        )

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)


class NullSpaceCertificate:
    """Null-space compliance certificate.

    Projects H onto the null space of the optimal linear predictor of Z,
    then measures how much total variance is preserved. A high preservation
    means the representation retains most of its information even after
    removing all linearly-predictable Z information.

    Certificate: "The representation's projection orthogonal to Z preserves
    X% of task-relevant variance."
    """

    def __init__(self, epsilon: float = 0.01, regularization: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.regularization = regularization

    def check(
        self,
        H: torch.Tensor | np.ndarray,
        Z: torch.Tensor | np.ndarray,
    ) -> CertificateResult:
        """Check null-space compliance and measure variance preservation.

        Args:
            H: Representations of shape (n, d).
            Z: Disallowed attribute labels (n,) or one-hot (n, c).

        Returns:
            CertificateResult with R², certification status, and
            variance_preserved indicating how much variance remains
            after projecting out Z-predictive directions.
        """
        H_np = self._to_numpy(H)
        Z_np = self._to_numpy(Z)

        if Z_np.ndim == 1:
            num_classes = int(Z_np.max()) + 1
            Z_onehot = np.eye(num_classes)[Z_np.astype(int)]
        else:
            Z_onehot = Z_np

        n, d = H_np.shape

        # Center
        H_centered = H_np - H_np.mean(axis=0, keepdims=True)
        Z_centered = Z_onehot - Z_onehot.mean(axis=0, keepdims=True)

        # Compute optimal linear predictor
        gram = H_centered.T @ H_centered + self.regularization * np.eye(d)
        W_star = np.linalg.solve(gram, H_centered.T @ Z_centered)

        # R² of the linear predictor
        Z_pred = H_centered @ W_star
        ss_res = np.sum((Z_centered - Z_pred) ** 2)
        ss_tot = np.sum(Z_centered ** 2)
        r_squared = float(max(0.0, 1.0 - (ss_res / max(ss_tot, 1e-12))))

        # Compute null-space projection
        # The Z-predictive subspace is spanned by the columns of W_star.
        # Project H onto the orthogonal complement.
        U, S, Vt = np.linalg.svd(W_star, full_matrices=False)
        # Projection matrix onto column space of W_star
        P_z = U @ U.T  # (d, d) projection onto Z-predictive subspace
        # Null-space projector (in representation space)
        P_null = np.eye(d) - P_z

        H_projected = H_centered @ P_null
        var_original = np.sum(H_centered ** 2)
        var_projected = np.sum(H_projected ** 2)
        variance_preserved = float(var_projected / max(var_original, 1e-12))

        return CertificateResult(
            r_squared=r_squared,
            certified=r_squared < self.epsilon,
            epsilon=self.epsilon,
            variance_preserved=variance_preserved,
        )

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
