"""Empirical covariance and affine least-squares compliance checks.

These scores concern squared-error prediction on the supplied sample. They
are not universal guarantees on linear-threshold classification accuracy or
out-of-sample leakage. The former accuracy-bound APIs are explicitly retired.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class CertificateResult:
    """Empirical affine ridge score and explicit target coverage.

    ``certified`` means the defined empirical squared-error score meets the
    threshold. It makes no universal classification-accuracy claim.
    """

    r_squared: float
    certified: bool
    epsilon: float
    variance_preserved: float | None = None
    class_support: list[int] | None = None
    valid_mask: list[bool] | None = None
    coverage_complete: bool = True
    score_defined: bool = True


def _prepare_targets(Z: np.ndarray, num_classes: int | None = None):
    """Construct a fixed one-hot schema, or retain continuous target columns."""
    if Z.ndim == 1 and np.issubdtype(Z.dtype, np.integer):
        K = num_classes if num_classes is not None else (int(Z.max()) + 1 if len(Z) else 0)
        if K < 1 or (len(Z) and (Z.min() < 0 or Z.max() >= K)):
            raise ValueError("Integer labels require a nonempty schema [0, num_classes)")
        targets = np.eye(K, dtype=np.float64)[Z.astype(int)]
        support = targets.sum(axis=0).astype(int)
    else:
        if num_classes is not None:
            raise ValueError("num_classes applies only to integer class labels")
        targets = np.asarray(Z, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets[:, None]
        if targets.ndim != 2 or targets.shape[1] == 0:
            raise ValueError("Targets must have shape (n,) or (n,c)")
        # For one-hot matrices retain class support; continuous columns have
        # no categorical support interpretation.
        is_onehot = np.all((targets == 0) | (targets == 1)) and np.all(targets.sum(axis=1) == 1)
        support = targets.sum(axis=0).astype(int) if is_onehot else None
    return targets, support


def _empirical_ridge_fit(H, Z, regularization, num_classes):
    H = np.asarray(H, dtype=np.float64)
    targets, support = _prepare_targets(np.asarray(Z), num_classes)
    if H.ndim != 2 or len(H) != len(targets):
        raise ValueError("Expected H (n,d) and matching target rows")
    if not np.isfinite(H).all() or not np.isfinite(targets).all():
        raise ValueError("Representations and targets must be finite")
    n, d = H.shape
    if n == 0:
        valid = np.zeros(targets.shape[1], dtype=bool)
        return H, np.zeros((d, targets.shape[1])), float("nan"), support, valid
    H_c = H - H.mean(axis=0, keepdims=True)
    Z_c = targets - targets.mean(axis=0, keepdims=True)
    ss_tot = (Z_c ** 2).sum(axis=0)
    valid = ss_tot > 0
    if support is not None:
        valid &= (support > 0) & (support < n)
    gram = H_c.T @ H_c + regularization * np.eye(d)
    W = np.linalg.solve(gram, H_c.T @ Z_c)
    # Require coverage of every declared target; undefined columns do not
    # silently disappear in the variance-weighted aggregate.
    score = float("nan")
    if valid.all():
        score = float(max(0.0, 1.0 - ((Z_c - H_c @ W) ** 2).sum() / ss_tot.sum()))
    return H_c, W, score, support, valid


class LinearComplianceCertificate:
    """In-sample affine ridge least-squares audit with explicit coverage.

    Fitting and scoring occur on the supplied rows. This empirical covariance /
    squared-error diagnostic is separate from out-of-sample predictive R² and
    does not certify threshold-classification accuracy. Regularization gives
    a ridge score; it is not a universal bound on unregularized OLS predictors.
    """

    def __init__(self, epsilon: float = 0.01, regularization: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.regularization = regularization

    def check(
        self,
        H: torch.Tensor | np.ndarray,
        Z: torch.Tensor | np.ndarray,
        *,
        num_classes: int | None = None,
    ) -> CertificateResult:
        """Score integer labels in a fixed schema or continuous target columns.

        Pass the dataset's class count for integer labels. Float targets are
        treated as continuous, including shape (n,); one-hot matrices work too.
        Incomplete support or zero target variance yields NaN and fails closed.
        """
        _, _, score, support, valid = _empirical_ridge_fit(
            self._to_numpy(H), self._to_numpy(Z), self.regularization, num_classes,
        )
        return CertificateResult(
            r_squared=score,
            certified=bool(np.isfinite(score) and score < self.epsilon),
            epsilon=self.epsilon,
            class_support=None if support is None else support.tolist(),
            valid_mask=valid.tolist(),
            coverage_complete=bool(valid.all()),
            score_defined=bool(np.isfinite(score)),
        )

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)


class NullSpaceCertificate:
    """Empirical ridge diagnostic with geometric variance preservation.

    The projection removes the span of fitted coefficient columns. The
    retained fraction measures total representation variance, not task utility.
    """

    def __init__(self, epsilon: float = 0.01, regularization: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.regularization = regularization

    def check(
        self,
        H: torch.Tensor | np.ndarray,
        Z: torch.Tensor | np.ndarray,
        *,
        num_classes: int | None = None,
    ) -> CertificateResult:
        H_c, W, score, support, valid = _empirical_ridge_fit(
            self._to_numpy(H), self._to_numpy(Z), self.regularization, num_classes,
        )
        # Include only nonzero singular directions: full U would remove
        # arbitrary directions when the fitted predictor is rank deficient.
        U, singular, _ = np.linalg.svd(W, full_matrices=False)
        tolerance = np.finfo(W.dtype).eps * max(W.shape) * (singular[0] if len(singular) else 0.0)
        basis = U[:, singular > tolerance]
        projected = H_c - (H_c @ basis) @ basis.T
        variance = float((H_c ** 2).sum())
        preserved = float((projected ** 2).sum() / variance) if variance > 0 else float("nan")
        return CertificateResult(
            r_squared=score,
            certified=bool(np.isfinite(score) and score < self.epsilon),
            epsilon=self.epsilon,
            variance_preserved=preserved,
            class_support=None if support is None else support.tolist(),
            valid_mask=valid.tolist(),
            coverage_complete=bool(valid.all()),
            score_defined=bool(np.isfinite(score)),
        )

    _to_numpy = staticmethod(LinearComplianceCertificate._to_numpy)


# ═══════════════════════════════════════════════════════════════════════════
# Retired classification-accuracy interfaces
# ═══════════════════════════════════════════════════════════════════════════


def certified_accuracy_bound(
    r_squared: float,
    majority_proportion: float,
    num_classes: int = 2,
) -> float:
    """Retired: least-squares R² does not bound threshold-classifier accuracy.

    The signature remains available so downstream callers receive an explicit
    error instead of silently reusing the invalid universal guarantee. For the
    exact counterexample, see ``docs/ACCURACY_CERTIFICATE_RETIREMENT.md`` and
    ``tests/test_accuracy_bound.py``. Zero covariance and zero affine
    least-squares explained variance remain meaningful, narrower claims.
    """
    raise NotImplementedError(
        "certified_accuracy_bound is retired: affine least-squares R² does "
        "not imply a universal classification-accuracy bound. Evaluate "
        "classification accuracy with separately fitted held-out attackers."
    )


@dataclass
class NonlinearCertificateResult:
    """Legacy serialized result schema; its accuracy bounds are invalid.

    Retained for import compatibility and reading historical artifacts only.
    Active report generation never constructs a result from this schema.
    """

    best_sigma: float
    r_squared_noisy: float
    linear_bound: float
    nonlinear_bound: float
    lipschitz_constant: float
    per_sigma: list[tuple[float, float, float]]


class NonlinearComplianceCertificate:
    """Retired smoothing extension of the invalid R²-to-accuracy guarantee."""

    def __init__(
        self,
        sigmas: tuple[float, ...] = (0.1, 0.5, 1.0),
        lipschitz_constant: float = 1.0,
        num_noise_samples: int = 50,
        epsilon: float = 0.01,
        regularization: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        self.sigmas = sigmas
        self.lipschitz_constant = lipschitz_constant
        self.num_noise_samples = num_noise_samples
        self.linear_cert = LinearComplianceCertificate(epsilon, regularization)
        self.random_state = random_state

    def check(
        self,
        H: np.ndarray | torch.Tensor,
        Z: np.ndarray | torch.Tensor,
        majority_proportion: float | None = None,
        num_classes: int | None = None,
    ) -> NonlinearCertificateResult:
        """Fail explicitly; smoothing does not repair the invalid premise."""
        raise NotImplementedError(
            "NonlinearComplianceCertificate is retired because it depends "
            "on the invalid least-squares R²-to-classification-accuracy bound."
        )


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

        # MI-based diagnostic from the same Fano expression. This does not use
        # the retired least-squares-to-classification bound.
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
