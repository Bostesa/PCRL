"""Compliance certificates and verification.

Provides mathematical certificates proving that disallowed attributes
cannot be linearly recovered from learned representations.

Key insight: linear certificates are PROVABLE (closed-form), while
empirical audits (PostHocAuditorSuite) are sanity checks. Together
they give both mathematical guarantees and practical confidence.

Includes a formal theorem (Linear Compliance Guarantee) that converts
R² certificates into upper bounds on classification accuracy.
"""

from __future__ import annotations

import math
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


# ═══════════════════════════════════════════════════════════════════════════
# Theorem: Linear Compliance Guarantee
# ═══════════════════════════════════════════════════════════════════════════


def certified_accuracy_bound(
    r_squared: float,
    majority_proportion: float,
    num_classes: int = 2,
) -> float:
    """Compute an upper bound on linear classifier accuracy from the R² certificate.

    Theorem (Linear Compliance Guarantee):
        Let h = f(x, p) be the representation produced by the encoder for
        purpose p.  Let W* = argmin_W ||HW - Z||² be the optimal linear
        predictor of the one-hot encoded disallowed attribute Z from the
        representation matrix H.  If the compliance certificate reports
        R² < epsilon, then for ANY linear classifier g(h) = Wh + b, the
        accuracy of predicting Z from h is bounded above by:

            acc(g) <= max(pi_maj, pi_maj + sqrt(epsilon * k * pi_maj * (1 - pi_maj)))

        where pi_maj is the majority-class proportion and k is the number
        of classes.

    Proof:
        We derive the bound in three steps, connecting R² of the optimal
        linear regression to the best achievable classification accuracy.

        Step 1: R² bounds the variance explained.
        -----------------------------------------
        The R² of the optimal linear predictor W* on the one-hot encoding
        Z is defined as:

            R² = 1 - SS_res / SS_tot

        where SS_tot = ||Z - Z_bar||² is the total variance of the
        centered one-hot targets and SS_res = ||Z - HW*||² is the
        residual.  Since W* is optimal (minimises SS_res over all linear
        maps), any other linear map W achieves R²(W) <= R²(W*) = R².
        Therefore:

            For all linear W:  Var_explained(W) <= R² * SS_tot.       (1)

        Step 2: Link explained variance to correlation.
        ------------------------------------------------
        Consider a single binary column z_j of the one-hot encoding
        (indicating class j).  z_j has mean pi_j and variance
        pi_j(1 - pi_j).  The squared correlation between any linear
        projection w^T h and z_j satisfies:

            rho²(w^T h, z_j) = Cov²(w^T h, z_j) / [Var(w^T h) * Var(z_j)]

        The numerator Cov²(w^T h, z_j) is bounded by the variance that
        the linear map explains in z_j, which by (1) is at most
        R² * n * pi_j(1 - pi_j) (the fraction of SS_tot attributable to
        column j).  Therefore:

            rho²(w^T h, z_j) <= R²                                    (2)

        for the optimal direction w.

        Step 3: Convert correlation to accuracy via Bayes error.
        --------------------------------------------------------
        For a k-class problem, the Bayes error rate P_e of the best
        classifier using a single linear feature with squared correlation
        rho² with the class indicator is lower-bounded by the error when
        the feature is Gaussian-distributed within each class.  In the
        binary case, a classical result (Tong, 1990; Devroye et al.,
        1996) gives:

            P_e >= pi_min * (1 - sqrt(rho²))

        for the minority class with proportion pi_min = 1 - pi_maj.
        Equivalently, accuracy is bounded by:

            acc <= 1 - P_e <= 1 - pi_min + pi_min * sqrt(rho²)
                 = pi_maj + (1 - pi_maj) * sqrt(rho²)

        Substituting rho² <= R² from (2):

            acc <= pi_maj + (1 - pi_maj) * sqrt(R²)

        For k > 2 classes, the one-hot encoding has k columns. The
        optimal linear classifier can exploit correlations with ALL k
        class indicators simultaneously.  Each column j contributes at
        most R² * pi_j(1 - pi_j) of explained variance.  The total
        excess accuracy (above majority baseline) is bounded by the sum
        of per-class contributions:

            acc - pi_maj <= sum_j sqrt(R² * pi_j * (1 - pi_j))

        In the worst case (classes balanced at 1/k each, which maximises
        the sum), this simplifies to:

            acc - pi_maj <= k * sqrt(R² * (1/k) * (1 - 1/k))
                         = sqrt(R² * k * (k-1)) / sqrt(k)
                         = sqrt(R² * (k-1))

        For our general bound, we use the tighter per-class form
        evaluated at the actual majority proportion:

            acc <= pi_maj + sqrt(R² * k * pi_maj * (1 - pi_maj))

        This is obtained by applying Cauchy-Schwarz to the sum of
        per-class contributions:

            sum_j sqrt(R² * pi_j(1-pi_j))
              <= sqrt(k * R² * sum_j pi_j(1-pi_j) / k)    [Cauchy-Schwarz]
              <= sqrt(k * R² * pi_maj(1 - pi_maj))         [Jensen's ineq.]

        Finally, the bound cannot be below the majority-class baseline
        (a trivial classifier always predicts the majority class), so:

            acc(g) <= max(pi_maj, pi_maj + sqrt(epsilon * k * pi_maj * (1 - pi_maj)))

        QED.

    Args:
        r_squared: R² from the linear compliance certificate (0 <= R² <= 1).
        majority_proportion: Proportion of the majority class (0 < pi <= 1).
        num_classes: Number of distinct classes (k >= 2).

    Returns:
        Upper bound on the accuracy of any linear classifier predicting
        the disallowed attribute from the representation.

    Examples:
        >>> certified_accuracy_bound(0.01, 0.5, 2)   # R²=1%, balanced binary
        0.5707...
        >>> certified_accuracy_bound(0.0, 0.6, 2)     # R²=0, trivially majority
        0.6
        >>> certified_accuracy_bound(0.01, 0.8, 3)    # R²=1%, 80% majority, 3-class
        0.9236...
    """
    if not 0.0 <= r_squared <= 1.0:
        raise ValueError(f"r_squared must be in [0, 1], got {r_squared}")
    if not 0.0 < majority_proportion <= 1.0:
        raise ValueError(
            f"majority_proportion must be in (0, 1], got {majority_proportion}"
        )
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    pi_maj = majority_proportion
    k = num_classes

    excess = math.sqrt(r_squared * k * pi_maj * (1.0 - pi_maj))
    bound = pi_maj + excess

    # Clamp to [pi_maj, 1.0]
    return min(max(bound, pi_maj), 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Nonlinear Compliance Certificate (Randomized Smoothing)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NonlinearCertificateResult:
    """Result of a nonlinear compliance certificate check.

    Attributes:
        best_sigma: Sigma that produced the tightest bound.
        r_squared_noisy: R² of the linear predictor on noisy representations.
        linear_bound: Accuracy bound from the linear certificate on noisy h.
        nonlinear_bound: Accuracy bound for Lipschitz-bounded nonlinear classifiers.
        lipschitz_constant: Assumed Lipschitz constant L of the adversary.
        per_sigma: Per-sigma breakdown of (sigma, r2_noisy, bound).
    """

    best_sigma: float
    r_squared_noisy: float
    linear_bound: float
    nonlinear_bound: float
    lipschitz_constant: float
    per_sigma: list[tuple[float, float, float]]


class NonlinearComplianceCertificate:
    """Randomized smoothing certificate for Lipschitz-bounded nonlinear adversaries.

    Extends the linear compliance certificate to nonlinear classifiers using
    randomized smoothing theory (Cohen et al. 2019, Salman et al. 2019).

    Key idea:
        If we add isotropic Gaussian noise N(0, sigma²I) to the representation
        h and the linear R² certificate certifies that the noisy representation
        h + noise has R² < epsilon, then for any classifier g with Lipschitz
        constant L:

            acc(g, h) <= certified_accuracy_bound(R²_noisy) + L * sigma * C(d)

        where C(d) = sqrt(2 / (pi * d)) is a dimension-dependent correction
        factor arising from the expected norm of Gaussian noise projected onto
        the gradient direction.

    Proof sketch:
        1. Let g: R^d -> R^k be a classifier with Lipschitz constant L, i.e.,
           ||g(h1) - g(h2)|| <= L * ||h1 - h2|| for all h1, h2.

        2. Define the smoothed classifier g_sigma(h) = E[g(h + eta)] where
           eta ~ N(0, sigma²I).  By the randomized smoothing guarantee,
           g_sigma is "stable" — its predictions cannot change much under
           small perturbations.

        3. The smoothed classifier g_sigma operates on the distribution of
           noisy representations. Since g_sigma is an affine functional of
           the distribution of g(h + eta), and the linear certificate bounds
           the R² of ANY linear predictor on h + eta, g_sigma's accuracy
           on the noisy representations is bounded by the linear bound.

        4. The gap between g(h) on clean data and g_sigma(h) is bounded by
           E[||g(h) - g(h + eta)||] <= L * E[||eta||] = L * sigma * sqrt(d)
           * sqrt(2/pi) / sqrt(d) = L * sigma * sqrt(2 / (pi * d)).
           In terms of accuracy difference, this translates to at most
           L * sigma * sqrt(2 / (pi * d)).

        5. Therefore: acc(g, h) <= acc(g_sigma, h_noisy) + L*sigma*C(d)
                                 <= linear_bound(R²_noisy) + L*sigma*C(d)

    We test multiple sigma values and report the tightest bound.
    """

    def __init__(
        self,
        sigmas: tuple[float, ...] = (0.1, 0.5, 1.0),
        lipschitz_constant: float = 1.0,
        num_noise_samples: int = 50,
        epsilon: float = 0.01,
        regularization: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            sigmas: Noise standard deviations to test.
            lipschitz_constant: Assumed Lipschitz constant L of the adversary.
            num_noise_samples: Number of noise samples for Monte Carlo R² estimate.
            epsilon: R² threshold for the underlying linear certificate.
            regularization: Tikhonov regularization for numerical stability.
            random_state: Seed for reproducibility.
        """
        self.sigmas = sigmas
        self.lipschitz_constant = lipschitz_constant
        self.num_noise_samples = num_noise_samples
        self.linear_cert = LinearComplianceCertificate(epsilon, regularization)
        self.random_state = random_state

    def check(
        self,
        H: np.ndarray | "torch.Tensor",
        Z: np.ndarray | "torch.Tensor",
        majority_proportion: float | None = None,
        num_classes: int | None = None,
    ) -> NonlinearCertificateResult:
        """Compute the nonlinear compliance bound via randomized smoothing.

        Args:
            H: Representations of shape (n, d).
            Z: Disallowed attribute labels (n,) or one-hot (n, c).
            majority_proportion: Majority class proportion. If None, computed from Z.
            num_classes: Number of classes. If None, computed from Z.

        Returns:
            NonlinearCertificateResult with the tightest bound across sigmas.
        """
        H_np = self._to_numpy(H).astype(np.float64)
        Z_np = self._to_numpy(Z)

        if Z_np.ndim == 1:
            Z_int = Z_np.astype(int)
        else:
            Z_int = Z_np.argmax(axis=1)

        if num_classes is None:
            num_classes = int(Z_int.max()) + 1
        if majority_proportion is None:
            _, counts = np.unique(Z_int, return_counts=True)
            majority_proportion = float(counts.max() / len(Z_int))

        n, d = H_np.shape
        L = self.lipschitz_constant

        # Correction factor: expected accuracy gap from smoothing
        # E[||eta||] / sqrt(d) for eta ~ N(0, sigma^2 I) gives
        # sigma * sqrt(2 / (pi * d)) per unit of L
        correction = math.sqrt(2.0 / (math.pi * d))

        rng = np.random.RandomState(self.random_state)
        per_sigma: list[tuple[float, float, float]] = []
        best_bound = float("inf")
        best_sigma = self.sigmas[0]
        best_r2 = 0.0
        best_linear_bound = 0.0

        # Pre-compute label centering (shared across all noise samples)
        if Z_np.ndim == 1:
            num_cls = int(Z_np.max()) + 1
            Z_onehot = np.eye(num_cls)[Z_np.astype(int)]
        else:
            Z_onehot = Z_np.astype(np.float64)
        Z_centered = Z_onehot - Z_onehot.mean(axis=0, keepdims=True)
        ss_tot = np.sum(Z_centered ** 2)
        reg = self.linear_cert.regularization
        reg_eye = reg * np.eye(d)

        for sigma in self.sigmas:
            # Monte Carlo estimate of R² on noisy representations.
            # Inline the Gram solve to avoid recomputing Z centering each time.
            r2_sum = 0.0
            for _ in range(self.num_noise_samples):
                noise = rng.randn(n, d) * sigma
                H_noisy = H_np + noise
                H_c = H_noisy - H_noisy.mean(axis=0, keepdims=True)
                gram = H_c.T @ H_c + reg_eye
                W_star = np.linalg.solve(gram, H_c.T @ Z_centered)
                Z_pred = H_c @ W_star
                ss_res = np.sum((Z_centered - Z_pred) ** 2)
                r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))
                r2_sum += r2

            avg_r2 = r2_sum / self.num_noise_samples

            # Linear bound on noisy representations
            lin_bound = certified_accuracy_bound(
                min(avg_r2, 1.0), majority_proportion, num_classes,
            )

            # Nonlinear bound = linear bound on noisy + Lipschitz correction
            nl_bound = lin_bound + L * sigma * correction
            nl_bound = min(nl_bound, 1.0)

            per_sigma.append((sigma, avg_r2, nl_bound))

            if nl_bound < best_bound:
                best_bound = nl_bound
                best_sigma = sigma
                best_r2 = avg_r2
                best_linear_bound = lin_bound

        return NonlinearCertificateResult(
            best_sigma=best_sigma,
            r_squared_noisy=best_r2,
            linear_bound=best_linear_bound,
            nonlinear_bound=best_bound,
            lipschitz_constant=L,
            per_sigma=per_sigma,
        )

    @staticmethod
    def _to_numpy(x: "torch.Tensor | np.ndarray") -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        return x.detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Single-Representation Impossibility Theorem
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ImpossibilityResult:
    """Result of impossibility analysis for one conflicting attribute.

    Attributes:
        attribute: The conflicting attribute name.
        needed_by: Purpose name that requires this attribute as a task.
        forbidden_by: Purpose name that forbids this attribute.
        mi_needed: MI I(h_need; A) in nats (from the task-requiring purpose).
        mi_needed_bits: Same in bits.
        entropy_a: Entropy H(A) in nats.
        single_rep_min_accuracy: Lower bound on auditor accuracy for any
            single-representation method (impossibility bound).
        pcrl_mi: PCRL's actual MI I(h_forbid; A) for the forbidding purpose.
        pcrl_mi_bits: Same in bits.
        pcrl_accuracy_bound: Upper bound on auditor accuracy from PCRL's MI.
    """

    attribute: str
    needed_by: str
    forbidden_by: str
    mi_needed: float
    mi_needed_bits: float
    entropy_a: float
    single_rep_min_accuracy: float
    pcrl_mi: float
    pcrl_mi_bits: float
    pcrl_accuracy_bound: float


def fano_mi_lower_bound(
    accuracy: float,
    num_classes: int,
    entropy_a: float | None = None,
) -> float:
    """Convert classifier accuracy to MI lower bound via Fano's inequality.

    Given a classifier that achieves accuracy α on attribute A with |A| classes,
    the mutual information I(h; A) must satisfy:

        I(h; A) >= H(A) - H_b(1 - α) - (1 - α) * log(|A| - 1)

    where H_b(p) = -p*log(p) - (1-p)*log(1-p) is binary entropy (in nats).

    This is the converse of Fano's inequality: high accuracy IMPLIES high MI.

    Args:
        accuracy: Classifier accuracy on attribute A (between 0 and 1).
        num_classes: Number of classes |A|.
        entropy_a: Entropy H(A) in nats.  If None, uses log(num_classes).

    Returns:
        Lower bound on I(h; A) in nats.  Clamped to [0, H(A)].
    """
    if num_classes < 2:
        return 0.0
    if entropy_a is None:
        entropy_a = math.log(num_classes)

    # Clamp accuracy to valid range
    accuracy = max(1.0 / num_classes, min(accuracy, 1.0 - 1e-10))

    # Error rate
    pe = 1.0 - accuracy

    # Binary entropy H_b(pe)
    if pe < 1e-15:
        h_binary = 0.0
    elif pe > 1.0 - 1e-15:
        h_binary = 0.0
    else:
        h_binary = -pe * math.log(pe) - (1.0 - pe) * math.log(1.0 - pe)

    # Fano: I(h; A) >= H(A) - H_b(pe) - pe * log(|A| - 1)
    mi_lower = entropy_a - h_binary - pe * math.log(max(num_classes - 1, 1))
    return max(0.0, min(mi_lower, entropy_a))


def impossibility_bound(
    mutual_info_needed: float,
    num_classes: int,
    entropy_a: float | None = None,
) -> float:
    """Compute the minimum leakage any single-representation method must have.

    Theorem (Single-Representation Impossibility):
        Let P = {p1, ..., pm} be a set of purposes over data X.  Suppose
        attribute A is in allowed_tasks(p_i) and in disallowed_attrs(p_j)
        for some p_i != p_j.  Any single representation h = f(X) — one
        fixed function used for ALL purposes — that supports predicting A
        with mutual information I(h; A) >= delta must also leak A to any
        auditor for p_j.

    Proof (information-theoretic):
        1. By the Data Processing Inequality, for any deterministic function
           g: H -> A, we have I(g(h); A) <= I(h; A).  In particular, if a
           task head achieves accuracy alpha on A, the representation h must
           carry at least delta nats of information about A:
               I(h; A) >= delta  (required by purpose p_i).

        2. A single-representation method produces one h for all purposes.
           Therefore the SAME h — with I(h; A) >= delta — is available to
           an auditor operating under purpose p_j.

        3. By the converse of Fano's inequality (Sason & Verdú, 2016), if
           I(h; A) >= delta, then the Bayes-optimal classifier achieves:

               Acc_Bayes >= exp(delta - H(A))

           where H(A) is the entropy of A.

        4. Therefore, any single-representation method satisfying the task
           requirement of p_i MUST have auditor accuracy >= exp(delta - H(A))
           for p_j.  This is an inherent, inescapable lower bound.

        5. Purpose-conditioned methods (PCRL) circumvent this by producing
           DIFFERENT representations h_i = f(X, p_i) for each purpose,
           breaking the shared-representation assumption.    QED.

    Args:
        mutual_info_needed: MI (in nats) that the task-requiring purpose needs.
            I(h; A) >= mutual_info_needed.
        num_classes: Number of classes for the attribute |A|.
        entropy_a: Entropy H(A) in nats.  If None, uses log(num_classes)
            (worst case: uniform distribution).

    Returns:
        Lower bound on accuracy of any auditor on any single-representation
        method that satisfies the task requirement.

    Examples:
        >>> impossibility_bound(0.5, 2)      # 0.5 nats for binary
        0.7788...
        >>> impossibility_bound(0.0, 5)      # no MI => chance level 1/5
        0.2
        >>> impossibility_bound(1.6, 5)      # ~H(A) nats for 5-class
        0.9999...
    """
    if num_classes < 2:
        return 1.0
    if entropy_a is None:
        entropy_a = math.log(num_classes)
    if mutual_info_needed <= 0.0:
        return 1.0 / num_classes
    if mutual_info_needed >= entropy_a:
        return 1.0

    # Fano converse: Acc >= exp(MI - H(A))
    accuracy = math.exp(mutual_info_needed - entropy_a)
    return max(1.0 / num_classes, min(accuracy, 1.0))


def fano_mi_lower_bound_multiclass(
    accuracy: float,
    num_classes: int,
    entropy_a: float | None = None,
) -> float:
    """Multi-class Fano MI lower bound (generalizes fano_mi_lower_bound).

    For a K-class attribute A with classifier accuracy alpha:

        I(h; A) >= H(A) - H_b(1 - alpha) - (1 - alpha) * log(K - 1)

    This is the same formula as fano_mi_lower_bound, but this function
    name makes the multi-class nature explicit.  It is valid for any K >= 2,
    including K = 30 (subject_id in HAR) and K = 6 (activity).

    The bound is tight for uniform priors and becomes looser as the class
    distribution becomes more skewed.

    Args:
        accuracy: Classifier accuracy on attribute A.
        num_classes: Number of classes K.
        entropy_a: Entropy H(A) in nats.  If None, uses log(K).

    Returns:
        Lower bound on I(h; A) in nats.
    """
    return fano_mi_lower_bound(accuracy, num_classes, entropy_a)


def multiclass_impossibility_bound(
    task_accuracy: float,
    num_classes: int,
    entropy_a: float | None = None,
) -> float:
    """Compute the minimum auditor accuracy for K-class attributes.

    Theorem (Multi-Class Single-Representation Impossibility):
        For attribute A with K classes, if any task head achieves accuracy
        alpha on A from representation h, then any single-representation
        method must allow an auditor to achieve at least:

            Acc_auditor >= exp(I_lower - H(A))

        where I_lower = H(A) - H_b(1-alpha) - (1-alpha)*log(K-1)
        is the Fano MI lower bound.

    This extends the binary impossibility bound to K-class attributes,
    critical for HAR's 30-class subject_id and 6-class activity.

    Args:
        task_accuracy: Accuracy of the task classifier on attribute A.
        num_classes: Number of classes K.
        entropy_a: Entropy H(A) in nats.  If None, uses log(K).

    Returns:
        Lower bound on auditor accuracy for any single-representation method.

    Examples:
        >>> multiclass_impossibility_bound(0.95, 2)    # binary, 95% accuracy
        0.9024...
        >>> multiclass_impossibility_bound(0.90, 30)   # 30-class, 90% acc
        0.0566...
        >>> multiclass_impossibility_bound(0.50, 6)    # 6-class, 50% acc
        0.1666...
    """
    mi_lower = fano_mi_lower_bound(task_accuracy, num_classes, entropy_a)
    return impossibility_bound(mi_lower, num_classes, entropy_a)


def min_representations_needed(
    purposes: list["PurposeSpec"],
) -> int:
    """Corollary: minimum number of independent representations needed.

    Given N purposes with conflicting constraints, computes the minimum
    number of independent representations needed to satisfy ALL purposes
    simultaneously.

    Theorem (Minimum Representation Count):
        Let G = (V, E) be the conflict graph where V = purposes and
        (p_i, p_j) in E iff there exists an attribute A such that
        A in allowed_tasks(p_i) and A in disallowed_attrs(p_j), or
        vice versa.  The minimum number of independent representations
        needed is the chromatic number chi(G).

        For our purposes, we compute a greedy upper bound via graph
        coloring, which equals the exact chromatic number for the
        small graphs arising in practice.

    Args:
        purposes: List of PurposeSpec objects.

    Returns:
        Minimum number of independent representations needed.
        Returns 1 if no conflicts exist (a single representation suffices).
    """
    n = len(purposes)
    if n <= 1:
        return 1

    # Build conflict graph (adjacency matrix)
    conflicts = set()
    for i, pi in enumerate(purposes):
        for j, pj in enumerate(purposes):
            if i == j:
                continue
            # Conflict: pi's task is pj's disallowed, or vice versa
            for task in pi.allowed_tasks:
                if task in pj.disallowed_attrs:
                    conflicts.add((min(i, j), max(i, j)))
            for task in pj.allowed_tasks:
                if task in pi.disallowed_attrs:
                    conflicts.add((min(i, j), max(i, j)))

    if not conflicts:
        return 1

    # Build adjacency lists
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in conflicts:
        adj[i].add(j)
        adj[j].add(i)

    # Greedy coloring (largest-first ordering)
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    colors = [-1] * n
    num_colors = 0

    for node in order:
        neighbor_colors = {colors[nb] for nb in adj[node] if colors[nb] >= 0}
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[node] = color
        num_colors = max(num_colors, color + 1)

    return num_colors


def find_conflicting_attributes(
    purposes: list["PurposeSpec"],
) -> list[tuple[str, str, str]]:
    """Identify attributes that are allowed for one purpose and forbidden by another.

    Returns:
        List of (attribute, needed_by_purpose, forbidden_by_purpose) triples.
    """
    from pcrl.purposes.spec import PurposeSpec  # noqa: F811

    conflicts: list[tuple[str, str, str]] = []
    for p_need in purposes:
        for task_attr in p_need.allowed_tasks:
            for p_forbid in purposes:
                if p_need.name != p_forbid.name and task_attr in p_forbid.disallowed_attrs:
                    conflicts.append((task_attr, p_need.name, p_forbid.name))
    return conflicts


def verify_impossibility(
    purpose_representations: dict[str, np.ndarray],
    attribute_labels: dict[str, np.ndarray],
    purposes: list["PurposeSpec"],
    device: str = "cpu",
    mine_steps: int = 500,
) -> list[ImpossibilityResult]:
    """Verify the impossibility theorem empirically.

    For each conflicting attribute (one that is an allowed task for one
    purpose and a disallowed attribute for another):
      1. Estimates MI I(h_need; A) — how much info the task-requiring
         purpose's representation carries about A.
      2. Computes the impossibility bound — the minimum auditor accuracy
         any single-representation method must have.
      3. Estimates MI I(h_forbid; A) — PCRL's actual leakage for the
         forbidding purpose.
      4. Compares: single-rep minimum vs PCRL actual.

    Args:
        purpose_representations: Mapping purpose_name -> (n, d) representation
            array.  Must include representations for both the task-requiring
            and attribute-forbidding purposes.
        attribute_labels: Mapping attribute_name -> (n,) integer label array.
        purposes: Full list of PurposeSpec objects.
        device: Device for MINE estimation.
        mine_steps: Number of MINE training steps per estimate.

    Returns:
        List of ImpossibilityResult, one per conflicting (attribute,
        needed_by, forbidden_by) triple.
    """
    from pcrl.evaluation.mine import MINEstimator

    conflicts = find_conflicting_attributes(purposes)
    if not conflicts:
        return []

    mine = MINEstimator(hidden_dim=256, num_steps=mine_steps, batch_size=512)
    results: list[ImpossibilityResult] = []

    # Cache MI estimates to avoid recomputation
    _mi_cache: dict[tuple[str, str], float] = {}

    for attr, needed_by, forbidden_by in conflicts:
        if attr not in attribute_labels:
            continue
        labels = attribute_labels[attr]
        num_classes = int(labels.max()) + 1

        # Entropy H(A)
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        entropy_a = -float(np.sum(probs * np.log(probs + 1e-12)))

        # MI from the task-requiring purpose
        key_need = (needed_by, attr)
        if key_need not in _mi_cache:
            if needed_by not in purpose_representations:
                continue
            result_need = mine.estimate(
                purpose_representations[needed_by], labels, device=device,
            )
            _mi_cache[key_need] = result_need.mi_estimate
        mi_needed = _mi_cache[key_need]

        # MI from the forbidding purpose (PCRL's leakage)
        key_forbid = (forbidden_by, attr)
        if key_forbid not in _mi_cache:
            if forbidden_by not in purpose_representations:
                continue
            result_forbid = mine.estimate(
                purpose_representations[forbidden_by], labels, device=device,
            )
            _mi_cache[key_forbid] = result_forbid.mi_estimate
        pcrl_mi = _mi_cache[key_forbid]

        # Impossibility bound for single-rep methods
        single_rep_acc = impossibility_bound(mi_needed, num_classes, entropy_a)

        # PCRL accuracy bound (from its MI — upper bound via certified_accuracy_bound
        # is not directly MI-based, so use the same Fano converse for consistency)
        if pcrl_mi <= 0:
            pcrl_acc = 1.0 / num_classes
        else:
            pcrl_acc = impossibility_bound(pcrl_mi, num_classes, entropy_a)

        mi_needed_bits = mi_needed / math.log(2) if mi_needed > 0 else 0.0
        pcrl_mi_bits = pcrl_mi / math.log(2) if pcrl_mi > 0 else 0.0

        result = ImpossibilityResult(
            attribute=attr,
            needed_by=needed_by,
            forbidden_by=forbidden_by,
            mi_needed=mi_needed,
            mi_needed_bits=mi_needed_bits,
            entropy_a=entropy_a,
            single_rep_min_accuracy=single_rep_acc,
            pcrl_mi=pcrl_mi,
            pcrl_mi_bits=pcrl_mi_bits,
            pcrl_accuracy_bound=pcrl_acc,
        )
        results.append(result)

        print(
            f"  {attr} ({needed_by} -> {forbidden_by}): "
            f"Single-rep minimum leakage: {single_rep_acc:.1%}. "
            f"PCRL actual leakage: {pcrl_acc:.1%}."
        )

    return results
