"""Full compliance audit protocol.

Combines linear certificates (provable, closed-form) with empirical
audits (PostHocAuditorSuite) for each (purpose, disallowed_attr) pair.
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
    NonlinearCertificateResult,
    NonlinearComplianceCertificate,
    NullSpaceCertificate,
    certified_accuracy_bound,
)


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
    ) -> tuple[CertificateResult, CertificateResult]:
        """Run both linear and null-space certificates.

        Args:
            representations: (n, d) array.
            labels: (n,) integer labels.

        Returns:
            Tuple of (linear_result, null_space_result).
        """
        linear_result = self.linear_cert.check(representations, labels)
        null_result = self.null_cert.check(representations, labels)
        return linear_result, null_result


class EmpiricalAudit:
    """Run PostHocAuditorSuite as a sanity check."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

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
        suite = PostHocAuditorSuite(random_state=self.random_state)
        suite.fit(train_reprs, train_labels)
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

    Returns:
        List of ComplianceReport, one per (purpose, disallowed_attr) pair.
    """
    reports: list[ComplianceReport] = []

    linear_audit = LinearAudit(epsilon=linear_epsilon)
    empirical_audit = EmpiricalAudit(random_state=random_state)
    nonlinear_cert = NonlinearComplianceCertificate(
        sigmas=(0.1, 0.5, 1.0),
        lipschitz_constant=1.0,
        num_noise_samples=50,
        epsilon=linear_epsilon,
        random_state=random_state,
    )

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

            # Linear + null-space certificates (on test data)
            linear_result, null_result = linear_audit.audit(test_reprs, test_labels)

            # Empirical audit
            all_labels = np.concatenate([train_labels, test_labels])
            unique_classes, class_counts = np.unique(all_labels, return_counts=True)
            num_classes = len(unique_classes)
            chance_acc = 1.0 / max(num_classes, 1)
            majority_proportion = float(class_counts.max() / len(all_labels))

            best_acc, emp_results = empirical_audit.audit(
                train_reprs, train_labels, test_reprs, test_labels
            )

            # Nonlinear certificate
            nl_result = nonlinear_cert.check(
                test_reprs, test_labels,
                majority_proportion=majority_proportion,
                num_classes=num_classes,
            )

            # Overall certification
            empirical_ok = (best_acc - chance_acc) < empirical_threshold
            certified = linear_result.certified and empirical_ok

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
                    nonlinear_bound=nl_result.nonlinear_bound,
                    nonlinear_best_sigma=nl_result.best_sigma,
                )
            )

    return reports


def print_compliance_table(reports: list[ComplianceReport]) -> None:
    """Print a clean summary table of compliance reports.

    For each certified pair, shows the theoretical accuracy bound from the
    Linear Compliance Guarantee theorem alongside the empirical auditor
    accuracy. The bound should be close to or above the empirical accuracy,
    confirming the certificate is meaningful.

    Args:
        reports: List of ComplianceReport objects.
    """
    header = (
        f"{'Purpose':<25} {'Attribute':<14} {'Lin R²':>8} {'Lin':>5} "
        f"{'Var%':>6} {'EmpAcc':>8} {'Bound':>8} {'NL Bound':>9} "
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
        bound = certified_accuracy_bound(
            r.linear_r2, r.majority_proportion, r.num_classes,
        )
        nl_str = f"{r.nonlinear_bound:>8.1%}" if r.nonlinear_bound is not None else "    N/A "

        print(
            f"{r.purpose_name:<25} {r.attr_name:<14} "
            f"{r.linear_r2:>8.4f} {lin_status:>5} "
            f"{r.variance_preserved:>5.1%} "
            f"{r.empirical_best_acc:>7.1%} {bound:>7.1%} {nl_str} "
            f"{r.empirical_chance_acc:>7.1%} "
            f"{status:>8}"
        )

    print("-" * len(header))
    num_pass = sum(1 for r in reports if r.certified)
    print(f"Overall: {num_pass}/{len(reports)} pairs certified")
    print("=" * len(header))
