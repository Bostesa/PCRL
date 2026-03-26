"""Post-hoc debiasing baselines: INLP and LEACE.

INLP (Iterative Null-space Projection) — Ravfogel et al. 2020:
    Train a standard encoder, then iteratively remove linear information
    about the disallowed attribute by projecting representations onto the
    null space of successive linear classifiers.

LEACE (LEAst-squares Concept Erasure) — Belrose et al. 2023:
    Train a standard encoder, then apply the closed-form orthogonal
    projection that erases ALL linear information about the disallowed
    attribute in one shot.

Both methods share a fundamental limitation: they produce a SINGLE
erased representation per input. They cannot handle different disallowed
attributes for different purposes without maintaining separate projection
matrices. And they cannot handle conflicting constraints where an
attribute is allowed for one purpose but disallowed for another.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class INLPProjector:
    """Iterative Null-space Projection (Ravfogel et al. 2020).

    Iteratively trains linear classifiers to predict a disallowed attribute
    from representations, then projects representations onto each
    classifier's null space. Repeats until the classifier can no longer
    beat chance, removing all linearly recoverable information.

    After fitting, call ``transform`` to apply the composed null-space
    projection to new representations.
    """

    def __init__(
        self,
        max_iters: int = 35,
        min_accuracy: float = 0.52,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            max_iters: Maximum number of projection iterations.
            min_accuracy: Stop when classifier accuracy drops below this.
            random_state: Seed for the linear classifiers.
        """
        self.max_iters = max_iters
        self.min_accuracy = min_accuracy
        self.random_state = random_state
        self.projection_matrix: np.ndarray | None = None
        self.num_iters_used: int = 0

    def fit(
        self,
        representations: np.ndarray,
        labels: np.ndarray,
    ) -> "INLPProjector":
        """Fit the INLP projection by iteratively removing linear directions.

        Args:
            representations: (n, d) array of encoder representations.
            labels: (n,) integer labels for the disallowed attribute.

        Returns:
            self
        """
        n, d = representations.shape
        P = np.eye(d, dtype=np.float64)
        H = representations.astype(np.float64).copy()

        for i in range(self.max_iters):
            clf = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=self.random_state + i,
            )
            clf.fit(H, labels)
            acc = clf.score(H, labels)

            if acc < self.min_accuracy:
                break

            # Get the weight matrix W (k x d) and compute null-space projection
            W = clf.coef_  # (num_classes, d)
            # Null-space projection: P_null = I - W^T (W W^T)^{-1} W
            WWT = W @ W.T
            try:
                WWT_inv = np.linalg.inv(WWT + 1e-10 * np.eye(WWT.shape[0]))
            except np.linalg.LinAlgError:
                break
            P_null = np.eye(d) - W.T @ WWT_inv @ W

            # Update cumulative projection and representations
            P = P_null @ P
            H = representations.astype(np.float64) @ P.T

        self.projection_matrix = P
        self.num_iters_used = i + 1 if acc >= self.min_accuracy else i
        return self

    def transform(self, representations: np.ndarray) -> np.ndarray:
        """Apply the learned null-space projection.

        Args:
            representations: (n, d) array.

        Returns:
            Projected representations (n, d).
        """
        if self.projection_matrix is None:
            raise RuntimeError("Must call fit() before transform().")
        return (representations.astype(np.float64) @ self.projection_matrix.T).astype(
            np.float32
        )


class LEACEEraser:
    """LEAst-squares Concept Erasure (Belrose et al. 2023).

    Computes the closed-form orthogonal projection that erases ALL linear
    information about a disallowed attribute from representations in a
    single step.  Equivalent to projecting onto the null space of the
    optimal least-squares predictor of the (centered, one-hot encoded)
    attribute from (centered) representations.

    After fitting, call ``transform`` to apply the erasure to new
    representations.
    """

    def __init__(self, regularization: float = 1e-6) -> None:
        """
        Args:
            regularization: Tikhonov regularization for numerical stability.
        """
        self.regularization = regularization
        self.projection_matrix: np.ndarray | None = None
        self.mean_H: np.ndarray | None = None

    def fit(
        self,
        representations: np.ndarray,
        labels: np.ndarray,
    ) -> "LEACEEraser":
        """Fit the LEACE erasure projection.

        Args:
            representations: (n, d) array of encoder representations.
            labels: (n,) integer labels for the disallowed attribute.

        Returns:
            self
        """
        H = representations.astype(np.float64)
        n, d = H.shape

        # One-hot encode labels
        num_classes = int(labels.max()) + 1
        Z = np.eye(num_classes, dtype=np.float64)[labels.astype(int)]

        # Center
        self.mean_H = H.mean(axis=0, keepdims=True)
        H_c = H - self.mean_H
        Z_c = Z - Z.mean(axis=0, keepdims=True)

        # Optimal linear predictor: W* = (H^T H + λI)^{-1} H^T Z
        gram = H_c.T @ H_c + self.regularization * np.eye(d)
        W_star = np.linalg.solve(gram, H_c.T @ Z_c)  # (d, k)

        # SVD of W_star to get the subspace spanned by its columns
        U, S, _ = np.linalg.svd(W_star, full_matrices=False)
        # Keep directions with non-negligible singular values
        rank = np.sum(S > 1e-10 * S[0])
        U_r = U[:, :rank]  # (d, rank)

        # Null-space projection: remove the column space of W_star
        self.projection_matrix = np.eye(d) - U_r @ U_r.T

        return self

    def transform(self, representations: np.ndarray) -> np.ndarray:
        """Apply the LEACE erasure projection.

        Args:
            representations: (n, d) array.

        Returns:
            Erased representations (n, d).
        """
        if self.projection_matrix is None or self.mean_H is None:
            raise RuntimeError("Must call fit() before transform().")
        H = representations.astype(np.float64)
        H_c = H - self.mean_H
        H_proj = H_c @ self.projection_matrix.T + self.mean_H
        return H_proj.astype(np.float32)
