"""Full compliance audit protocol.

Combines empirical affine least-squares checks with separately fitted
classification audits for each (purpose, disallowed_attr) pair. Least-squares
R² is not a universal classification-accuracy guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

import torch.nn as nn

from pcrl.models.auditor import PostHocAuditorSuite
from pcrl.purposes.spec import PurposeRegistry
from pcrl.purposes.verification import (
    CertificateResult,
    LinearComplianceCertificate,
    NullSpaceCertificate,
)


# ═══════════════════════════════════════════════════════════════════════════
# Framework D: Dominant-Axis Auditing
# ═══════════════════════════════════════════════════════════════════════════
#
# Standard one-hot R² systematically underestimates leakage on imbalanced
# multi-class sensitive attributes. The Convex-Combination Identity:
#
#     R²_onehot = Σ_k w_k · R²_OvR_k,    w_k = π_k(1-π_k) / Σ_j π_j(1-π_j)
#
# weights each one-vs-rest R² by class variance. A high-leakage minority
# class (small π_k) is downweighted by its small π_k(1-π_k). The dominant-
# axis metric R²_DA = max_k R²_OvR_k surfaces this hidden leakage.
#
# Implementation: closed-form Tikhonov-regularized linear regression of
# H → z_k for each binary indicator z_k = 1[y == k], following the same
# centering and regularization conventions as LinearComplianceCertificate
# so that the identity above holds tightly.

def compute_dominant_axis_r2(
    features: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    regularization: float = 1e-6,
    num_classes: int | None = None,
) -> dict:
    """In-sample maximum OvR ridge R² with fixed-schema coverage.

    Fit and score use the supplied rows, so this is an empirical least-squares
    diagnostic, not an out-of-sample attacker score. Pass the dataset's fixed
    class count: missing/constant classes have NaN scores, and ``r2_da`` is
    undefined unless all declared classes have positive and negative support.
    ``observed_r2_da`` summarizes valid classes without implying a privacy pass.
    """
    H = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
    y = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Dominant-axis labels must be a one-dimensional integer array")
    H = H.astype(np.float64, copy=False)
    n, d = H.shape
    K = num_classes if num_classes is not None else (int(y.max()) + 1 if len(y) else 0)
    if len(y) != n or K < 1 or (n and (y.min() < 0 or y.max() >= K)):
        raise ValueError("Labels must match feature rows and fixed schema [0, num_classes)")
    if not np.isfinite(H).all():
        raise ValueError("Features must be finite")
    support = np.bincount(y.astype(np.int64), minlength=K)
    priors = support / max(n, 1)
    variance = priors * (1.0 - priors)
    valid = (support > 0) & (support < n) & (variance > 0)
    scores = np.full(K, np.nan)
    if n and valid.any():
        H_c = H - H.mean(axis=0, keepdims=True)
        targets = np.eye(K)[y]
        Z_c = targets - targets.mean(axis=0, keepdims=True)
        gram = H_c.T @ H_c + regularization * np.eye(d)
        weights = np.linalg.solve(gram, H_c.T @ Z_c)
        residual = ((Z_c - H_c @ weights) ** 2).sum(axis=0)
        total = (Z_c ** 2).sum(axis=0)
        scores[valid] = np.maximum(0.0, 1.0 - residual[valid] / total[valid])
    argmax = int(np.nanargmax(scores)) if valid.any() else -1
    observed = float(scores[argmax]) if argmax >= 0 else float("nan")
    return {
        "r2_da": observed if valid.all() else float("nan"),
        "observed_r2_da": observed,
        "argmax_class": argmax,
        "per_class_r2": scores.tolist(),
        "priors": priors.tolist(),
        "class_support": support.tolist(),
        "target_variance": variance.tolist(),
        "valid_mask": valid.tolist(),
        "coverage_complete": bool(valid.all()),
    }


def compute_mlp_ovr_delta(
    train_features: np.ndarray | torch.Tensor,
    train_labels: np.ndarray | torch.Tensor,
    test_features: np.ndarray | torch.Tensor,
    test_labels: np.ndarray | torch.Tensor,
    *,
    hidden: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    dropout: float = 0.3,
    batch_size: int = 256,
    device: str = "cpu",
    random_state: int = 0,
    num_classes: int | None = None,
) -> dict:
    """Nonlinear MLP one-vs-rest probe for dominant-axis auditing.

    For each class k, train a 2-layer MLP (hidden=256, ReLU, dropout=0.3,
    Adam lr=1e-3, BCE) on (features, z_k = 1[label == k]) and evaluate test
    accuracy. Report ``delta_k = test_acc_k − binary_majority_baseline_k``
    where the majority prediction is selected on training labels and its
    accuracy is evaluated on test labels. ``mlp_da_delta`` is undefined unless
    every fixed-schema class has positive and negative support in both splits.
    Unsupported columns have NaN scores; no classifier is trained for them.

    Args:
        train_features: (n_tr, d) features for fitting.
        train_labels: (n_tr,) integer multi-class labels.
        test_features: (n_te, d) features for evaluation.
        test_labels: (n_te,) integer multi-class labels.
        hidden: Hidden width.
        epochs: Training epochs per class.
        lr: Adam learning rate.
        dropout: Dropout between hidden and output.
        batch_size: Mini-batch size.
        device: Torch device.
        random_state: Seed for the MLP init and data shuffling.
        num_classes: Fixed class schema; fallback infers from training only.

    Returns:
        Dict with keys ``mlp_da_delta``, ``argmax_class``,
        ``per_class_delta``, ``per_class_acc``, ``per_class_baseline``,
        train/test support counts and valid masks, and ``coverage_complete``.
        ``observed_mlp_da_delta`` summarizes only supported columns; it cannot
        establish privacy for the entire schema when coverage is incomplete.
    """
    H_tr = train_features.detach().cpu().numpy() if isinstance(train_features, torch.Tensor) else np.asarray(train_features)
    y_tr = train_labels.detach().cpu().numpy() if isinstance(train_labels, torch.Tensor) else np.asarray(train_labels)
    H_te = test_features.detach().cpu().numpy() if isinstance(test_features, torch.Tensor) else np.asarray(test_features)
    y_te = test_labels.detach().cpu().numpy() if isinstance(test_labels, torch.Tensor) else np.asarray(test_labels)

    H_tr = H_tr.astype(np.float32, copy=False)
    H_te = H_te.astype(np.float32, copy=False)
    if any(y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) for y in (y_tr, y_te)):
        raise ValueError("MLP OvR labels must be one-dimensional integer arrays")
    y_tr = y_tr.astype(np.int64, copy=False)
    y_te = y_te.astype(np.int64, copy=False)
    K = num_classes if num_classes is not None else (int(y_tr.max()) + 1 if len(y_tr) else 0)
    if K < 1 or any(len(y) and (y.min() < 0 or y.max() >= K) for y in (y_tr, y_te)):
        raise ValueError("Labels must lie in the fixed schema [0, num_classes)")
    if (H_tr.ndim != 2 or H_te.ndim != 2 or H_tr.shape[1] != H_te.shape[1]
            or len(H_tr) != len(y_tr) or len(H_te) != len(y_te)):
        raise ValueError("Features must have matching dimensions and label rows")
    if not np.isfinite(H_tr).all() or not np.isfinite(H_te).all():
        raise ValueError("Features must be finite")
    train_support = np.bincount(y_tr, minlength=K)
    test_support = np.bincount(y_te, minlength=K)
    train_valid = (train_support > 0) & (train_support < len(y_tr))
    test_valid = (test_support > 0) & (test_support < len(y_te))
    valid = train_valid & test_valid
    majority_prediction = (2 * train_support > len(y_tr)).astype(int)
    d = H_tr.shape[1]
    dev = torch.device(device)

    H_tr_t = torch.from_numpy(H_tr).to(dev)
    H_te_t = torch.from_numpy(H_te).to(dev)

    per_class_acc: list[float] = []
    per_class_baseline: list[float] = []
    per_class_delta: list[float] = []

    for k in range(K):
        z_tr = (y_tr == k).astype(np.float32)
        z_te = (y_te == k).astype(np.int64)

        # The decision is fit only on training labels (ties predict zero).
        # Test labels evaluate that fixed predictor; they never choose it.
        if not valid[k]:
            per_class_acc.append(float("nan"))
            per_class_baseline.append(float("nan"))
            per_class_delta.append(float("nan"))
            continue
        baseline = float((z_te == majority_prediction[k]).mean())

        torch.manual_seed(random_state + k)

        model = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        ).to(dev)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        z_tr_t = torch.from_numpy(z_tr).to(dev)
        n_tr = len(z_tr)
        rng = np.random.RandomState(random_state + k)

        model.train()
        for _ in range(epochs):
            perm = rng.permutation(n_tr)
            for start in range(0, n_tr, batch_size):
                idx = perm[start:start + batch_size]
                xb = H_tr_t[idx]
                yb = z_tr_t[idx]
                opt.zero_grad()
                logits = model(xb).squeeze(-1)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(H_te_t).squeeze(-1)).cpu().numpy()
        preds = (probs >= 0.5).astype(np.int64)
        acc = float((preds == z_te).mean())

        per_class_acc.append(acc)
        per_class_baseline.append(baseline)
        per_class_delta.append(float(acc - baseline))

    arr = np.asarray(per_class_delta)
    argmax = int(np.nanargmax(arr)) if valid.any() else -1
    observed = float(arr[argmax]) if argmax >= 0 else float("nan")
    return {
        "mlp_da_delta": observed if valid.all() else float("nan"),
        "observed_mlp_da_delta": observed,
        "argmax_class": argmax,
        "per_class_delta": per_class_delta,
        "per_class_acc": per_class_acc,
        "per_class_baseline": per_class_baseline,
        "majority_prediction": majority_prediction.tolist(),
        "train_class_support": train_support.tolist(),
        "test_class_support": test_support.tolist(),
        "train_valid_mask": train_valid.tolist(),
        "test_valid_mask": test_valid.tolist(),
        "valid_mask": valid.tolist(),
        "coverage_complete": bool(valid.all()),
    }


@dataclass
class ComplianceReport:
    """Compliance audit report for one (purpose, attribute) pair.

    Attributes:
        purpose_name: Name of the purpose.
        attr_name: Name of the disallowed attribute.
        linear_r2: R² of the optimal linear predictor.
        linear_certified: Whether the linear certificate passes.
        null_space_r2: R² from the null-space certificate.
        variance_preserved: Fraction of variance preserved after null-space projection.
        empirical_best_acc: Best accuracy among all post-hoc auditors.
        empirical_chance_acc: Chance-level accuracy for this attribute.
        empirical_results: Per-classifier accuracy breakdown.
        certified: Overall certification (linear_certified AND empirical below threshold).
        r2_da: Dominant-axis R² = max_k R²_OvR_k. For binary attrs equals
            ``linear_r2`` trivially.
        r2_da_argmax: Class index k* attaining the dominant axis (-1 if not computed).
        r2_da_per_class: Per-class OvR R² list (empty if not computed).
        mlp_da_delta: Max over k of (MLP test accuracy on z_k vs rest minus
            binary majority baseline). None if MLP probe disabled.
        mlp_da_argmax: Class index attaining the max delta (-1 if not computed).
        mlp_da_per_class: Per-class delta list (empty if not computed).
    """

    purpose_name: str
    attr_name: str
    linear_r2: float
    linear_certified: bool
    null_space_r2: float = 0.0
    variance_preserved: float = 1.0
    empirical_best_acc: float = 0.0
    empirical_chance_acc: float = 0.0
    empirical_results: dict[str, dict[str, float]] = field(default_factory=dict)
    certified: bool = False
    majority_proportion: float = 0.5
    num_classes: int = 2
    nonlinear_bound: float | None = None
    nonlinear_best_sigma: float | None = None
    # Legacy bound fields stay None; no classification guarantee is issued.
    accuracy_guarantee_status: str = "retired_invalid"
    class_support: list[int] = field(default_factory=list)
    valid_mask: list[bool] = field(default_factory=list)
    coverage_complete: bool = False
    r2_da: float = 0.0
    r2_da_argmax: int = -1
    r2_da_per_class: list[float] = field(default_factory=list)
    mlp_da_delta: float | None = None
    mlp_da_argmax: int = -1
    mlp_da_per_class: list[float] = field(default_factory=list)
    mlp_da_coverage: dict = field(default_factory=dict)


def _extract_representations_and_labels(
    encoder: nn.Module,
    loader: DataLoader,
    purpose_idx: int,
    attr_names: list[str],
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract representations and sensitive attribute labels in a single pass.

    IMPORTANT: Representations and labels MUST be extracted in the same
    iteration of the DataLoader.  If the loader has shuffle=True, separate
    iterations produce different sample orderings, causing row-level
    misalignment between representations and labels.  This was the root
    cause of bug a50810e (April 2026).

    Args:
        encoder: Trained encoder.
        loader: Data loader yielding batches with "features" and
            "sensitive_attrs".
        purpose_idx: Integer purpose index for the encoder.
        attr_names: Names of sensitive attributes to extract.
        device: Device to run the encoder on.

    Returns:
        Tuple of (representations, labels_dict) where representations is
        shape (n, repr_dim) and labels_dict maps attr_name -> (n,) array.
    """
    device = torch.device(device)
    encoder.eval()

    all_reprs: list[np.ndarray] = []
    all_labels: dict[str, list[np.ndarray]] = {a: [] for a in attr_names}

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)
            all_reprs.append(h.cpu().numpy())
            for attr in attr_names:
                all_labels[attr].append(batch["sensitive_attrs"][attr].numpy())

    reprs = np.concatenate(all_reprs, axis=0)
    labels = {a: np.concatenate(v) for a, v in all_labels.items()}
    return reprs, labels


class LinearAudit:
    """Run LinearComplianceCertificate for every (purpose, disallowed_attr) pair."""

    def __init__(self, epsilon: float = 0.01, regularization: float = 1e-6) -> None:
        self.linear_cert = LinearComplianceCertificate(epsilon, regularization)
        self.null_cert = NullSpaceCertificate(epsilon, regularization)

    def audit(
        self,
        representations: np.ndarray,
        labels: np.ndarray,
        *,
        num_classes: int | None = None,
    ) -> tuple[CertificateResult, CertificateResult]:
        """Run both linear and null-space certificates.

        Args:
            representations: (n, d) array.
            labels: (n,) integer labels.

        Returns:
            Tuple of (linear_result, null_space_result).
        """
        linear_result = self.linear_cert.check(
            representations, labels, num_classes=num_classes,
        )
        null_result = self.null_cert.check(
            representations, labels, num_classes=num_classes,
        )
        return linear_result, null_result


class EmpiricalAudit:
    """Run PostHocAuditorSuite as a sanity check.

    On large datasets (> max_fit_samples rows) the RBF-SVM in the suite is
    O(n²) in time and memory and effectively never finishes. We subsample
    the training set to `max_fit_samples` rows (stratified by label when
    possible) before fitting; evaluation stays on the full test set.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_fit_samples: int = 20000,
    ) -> None:
        self.random_state = random_state
        self.max_fit_samples = max_fit_samples

    def audit(
        self,
        train_reprs: np.ndarray,
        train_labels: np.ndarray,
        test_reprs: np.ndarray,
        test_labels: np.ndarray,
    ) -> tuple[float, dict[str, dict[str, float]]]:
        """Run empirical audit.

        Args:
            train_reprs: Training representations for fitting auditors.
            train_labels: Training labels.
            test_reprs: Test representations for evaluation.
            test_labels: Test labels.

        Returns:
            Tuple of (best_accuracy, per_classifier_results).
        """
        fit_reprs, fit_labels = train_reprs, train_labels
        n = len(train_labels)
        if n > self.max_fit_samples:
            rng = np.random.RandomState(self.random_state)
            try:
                from sklearn.model_selection import train_test_split
                fit_reprs, _, fit_labels, _ = train_test_split(
                    train_reprs, train_labels,
                    train_size=self.max_fit_samples,
                    stratify=train_labels,
                    random_state=self.random_state,
                )
            except ValueError:
                # stratify can fail when a class has <2 rows → fall back to uniform
                idx = rng.choice(n, size=self.max_fit_samples, replace=False)
                fit_reprs = train_reprs[idx]
                fit_labels = train_labels[idx]
        suite = PostHocAuditorSuite(random_state=self.random_state)
        suite.fit(fit_reprs, fit_labels)
        results = suite.evaluate(test_reprs, test_labels)
        best_acc = max(m["accuracy"] for m in results.values())
        return best_acc, results


def generate_report(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    purpose_registry: PurposeRegistry,
    device: torch.device | str = "cpu",
    linear_epsilon: float = 0.01,
    empirical_threshold: float = 0.05,
    random_state: int = 42,
    compute_dominant_axis: bool = True,
    compute_mlp_da: bool = False,
    mlp_da_kwargs: dict | None = None,
) -> list[ComplianceReport]:
    """Generate a full compliance report for all (purpose, disallowed_attr) pairs.

    Args:
        encoder: Trained purpose-conditioned encoder.
        train_loader: Training data loader (for fitting empirical auditors).
        test_loader: Test data loader (for evaluation).
        purpose_registry: Registry of all purposes.
        device: Device for encoder inference.
        linear_epsilon: R² threshold for linear certificates.
        empirical_threshold: Maximum allowed (best_acc - chance_acc) for empirical audit.
        random_state: Seed for empirical auditors.
        compute_dominant_axis: If True, also compute the linear R²_DA
            metric (Framework D) on the test split. Cheap, default on.
        compute_mlp_da: If True, also train an MLP one-vs-rest probe
            per class and report the max accuracy delta over the binary
            majority baseline. Expensive (50 epochs × K classes per pair);
            opt-in.
        mlp_da_kwargs: Optional kwargs forwarded to ``compute_mlp_ovr_delta``.

    Returns:
        List of ComplianceReport, one per (purpose, disallowed_attr) pair.
    """
    mlp_da_kwargs = dict(mlp_da_kwargs or {})
    reports: list[ComplianceReport] = []

    linear_audit = LinearAudit(epsilon=linear_epsilon)
    empirical_audit = EmpiricalAudit(random_state=random_state)

    # Collect all unique attr names needed across purposes, then extract
    # representations AND labels in a single pass per (purpose_idx, loader).
    # Single-pass extraction is critical: if the loader has shuffle=True,
    # separate iterations produce different sample orderings, breaking the
    # row-level alignment between representations and labels.
    train_cache: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    test_cache: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}

    # Pre-compute which attrs each purpose needs
    purpose_attrs: dict[int, list[str]] = {}
    for idx, purpose in enumerate(purpose_registry.purposes):
        if purpose.disallowed_attrs:
            purpose_attrs[idx] = list(purpose.disallowed_attrs)

    # Collect ALL attr names that any purpose needs (for each purpose's
    # single-pass extraction — we extract all attrs the purpose cares about
    # in one loop over the loader).
    for purpose_idx in purpose_attrs:
        if purpose_idx not in train_cache:
            attrs = purpose_attrs[purpose_idx]
            train_cache[purpose_idx] = _extract_representations_and_labels(
                encoder, train_loader, purpose_idx, attrs, device
            )
            test_cache[purpose_idx] = _extract_representations_and_labels(
                encoder, test_loader, purpose_idx, attrs, device
            )

    for purpose_idx, purpose in enumerate(purpose_registry.purposes):
        if not purpose.disallowed_attrs:
            continue

        for attr_name in purpose.disallowed_attrs:
            train_reprs = train_cache[purpose_idx][0]
            test_reprs = test_cache[purpose_idx][0]
            train_labels = train_cache[purpose_idx][1][attr_name]
            test_labels = test_cache[purpose_idx][1][attr_name]

            # Fixed schema comes from the purpose, never from the test subset.
            num_classes = purpose.disallowed_attr_dims.get(
                attr_name, int(train_labels.max()) + 1,
            )
            # These refitted test-sample scores are descriptive covariance /
            # least-squares statistics, not out-of-sample attacker scores.
            linear_result, null_result = linear_audit.audit(
                test_reprs, test_labels, num_classes=num_classes,
            )

            # Empirical audit
            all_labels = np.concatenate([train_labels, test_labels])
            unique_classes, class_counts = np.unique(all_labels, return_counts=True)
            chance_acc = 1.0 / max(num_classes, 1)
            majority_proportion = float(class_counts.max() / len(all_labels))

            best_acc, emp_results = empirical_audit.audit(
                train_reprs, train_labels, test_reprs, test_labels
            )

            # Overall certification
            empirical_ok = (best_acc - chance_acc) < empirical_threshold
            certified = linear_result.certified and empirical_ok

            # Dominant-axis auditing (Framework D)
            r2_da_val = 0.0
            r2_da_argmax = -1
            r2_da_per_class: list[float] = []
            if compute_dominant_axis:
                da = compute_dominant_axis_r2(
                    test_reprs, test_labels, num_classes=num_classes,
                )
                r2_da_val = da["r2_da"]
                r2_da_argmax = da["argmax_class"]
                r2_da_per_class = da["per_class_r2"]

            mlp_da_val: float | None = None
            mlp_da_argmax = -1
            mlp_da_per_class: list[float] = []
            mlp_coverage: dict = {}
            if compute_mlp_da:
                probe_kwargs = dict(mlp_da_kwargs, num_classes=num_classes)
                mlp = compute_mlp_ovr_delta(
                    train_reprs, train_labels, test_reprs, test_labels,
                    **probe_kwargs,
                )
                mlp_da_val = mlp["mlp_da_delta"]
                mlp_da_argmax = mlp["argmax_class"]
                mlp_da_per_class = mlp["per_class_delta"]
                mlp_coverage = {key: mlp[key] for key in (
                    "train_class_support", "test_class_support", "train_valid_mask",
                    "test_valid_mask", "valid_mask", "coverage_complete",
                )}
                certified = certified and mlp["coverage_complete"]

            reports.append(
                ComplianceReport(
                    purpose_name=purpose.name,
                    attr_name=attr_name,
                    linear_r2=linear_result.r_squared,
                    linear_certified=linear_result.certified,
                    null_space_r2=null_result.r_squared,
                    variance_preserved=null_result.variance_preserved or 1.0,
                    empirical_best_acc=best_acc,
                    empirical_chance_acc=chance_acc,
                    empirical_results=emp_results,
                    certified=certified,
                    majority_proportion=majority_proportion,
                    num_classes=num_classes,
                    class_support=linear_result.class_support or [],
                    valid_mask=linear_result.valid_mask or [],
                    coverage_complete=linear_result.coverage_complete,
                    r2_da=r2_da_val,
                    r2_da_argmax=r2_da_argmax,
                    r2_da_per_class=r2_da_per_class,
                    mlp_da_delta=mlp_da_val,
                    mlp_da_argmax=mlp_da_argmax,
                    mlp_da_per_class=mlp_da_per_class,
                    mlp_da_coverage=mlp_coverage,
                )
            )

    return reports


def print_compliance_table(reports: list[ComplianceReport]) -> None:
    """Print a clean summary table of compliance reports.

    Shows descriptive least-squares R² alongside held-out empirical attacker
    accuracy. PASS refers only to the configured empirical audit criteria.

    Args:
        reports: List of ComplianceReport objects.
    """
    header = (
        f"{'Purpose':<25} {'Attribute':<14} {'Lin R²':>8} {'Lin':>5} "
        f"{'Var%':>6} {'EmpAcc':>8} {'Coverage':>9} "
        f"{'Chance':>8} {'Status':>8}"
    )
    print("=" * len(header))
    print("COMPLIANCE AUDIT REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in reports:
        status = "PASS" if r.certified else "FAIL"
        lin_status = "PASS" if r.linear_certified else "FAIL"
        coverage = f"{sum(r.valid_mask)}/{r.num_classes}"

        print(
            f"{r.purpose_name:<25} {r.attr_name:<14} "
            f"{r.linear_r2:>8.4f} {lin_status:>5} "
            f"{r.variance_preserved:>5.1%} "
            f"{r.empirical_best_acc:>7.1%} {coverage:>9} "
            f"{r.empirical_chance_acc:>7.1%} "
            f"{status:>8}"
        )

    print("-" * len(header))
    num_pass = sum(1 for r in reports if r.certified)
    print(f"Overall: {num_pass}/{len(reports)} pairs pass empirical audit criteria")
    print("Classification accuracy guarantee: unavailable (R² bound retired as invalid).")
    print("=" * len(header))
