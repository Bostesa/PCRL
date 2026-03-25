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
    NullSpaceCertificate,
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


def _extract_representations_and_labels(
    encoder: nn.Module,
    loader: DataLoader,
    purpose_idx: int,
    attr_name: str,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract representations and attribute labels from a data loader.

    Args:
        encoder: Trained encoder.
        loader: Data loader yielding batches with "features" and "sensitive_attrs".
        purpose_idx: Integer purpose index for the encoder.
        attr_name: Name of the sensitive attribute to extract.
        device: Device to run the encoder on.

    Returns:
        Tuple of (representations, labels) as numpy arrays.
    """
    device = torch.device(device)
    encoder.eval()

    all_reprs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)
            all_reprs.append(h.cpu().numpy())

            labels = batch["sensitive_attrs"][attr_name]
            all_labels.append(labels.numpy())

    return np.concatenate(all_reprs, axis=0), np.concatenate(all_labels, axis=0)


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

    for purpose_idx, purpose in enumerate(purpose_registry.purposes):
        for attr_name in purpose.disallowed_attrs:
            # Extract representations
            train_reprs, train_labels = _extract_representations_and_labels(
                encoder, train_loader, purpose_idx, attr_name, device
            )
            test_reprs, test_labels = _extract_representations_and_labels(
                encoder, test_loader, purpose_idx, attr_name, device
            )

            # Linear + null-space certificates (on test data)
            linear_result, null_result = linear_audit.audit(test_reprs, test_labels)

            # Empirical audit
            num_classes = len(np.unique(np.concatenate([train_labels, test_labels])))
            chance_acc = 1.0 / max(num_classes, 1)

            best_acc, emp_results = empirical_audit.audit(
                train_reprs, train_labels, test_reprs, test_labels
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
                )
            )

    return reports


def print_compliance_table(reports: list[ComplianceReport]) -> None:
    """Print a clean summary table of compliance reports.

    Args:
        reports: List of ComplianceReport objects.
    """
    header = (
        f"{'Purpose':<25} {'Attribute':<18} {'Lin R²':>8} {'Lin Cert':>9} "
        f"{'Var Pres':>9} {'Best Acc':>9} {'Chance':>8} {'Certified':>10}"
    )
    print("=" * len(header))
    print("COMPLIANCE AUDIT REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in reports:
        status = "PASS" if r.certified else "FAIL"
        lin_status = "PASS" if r.linear_certified else "FAIL"
        print(
            f"{r.purpose_name:<25} {r.attr_name:<18} "
            f"{r.linear_r2:>8.4f} {lin_status:>9} "
            f"{r.variance_preserved:>8.1%} "
            f"{r.empirical_best_acc:>8.1%} {r.empirical_chance_acc:>7.1%} "
            f"{status:>10}"
        )

    print("-" * len(header))
    num_pass = sum(1 for r in reports if r.certified)
    print(f"Overall: {num_pass}/{len(reports)} pairs certified")
    print("=" * len(header))
