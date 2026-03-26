"""Tests for the Linear Compliance Guarantee theorem and certified_accuracy_bound."""

import math

import numpy as np
import pytest
import torch

from pcrl.purposes.verification import (
    CertificateResult,
    LinearComplianceCertificate,
    NullSpaceCertificate,
    certified_accuracy_bound,
)
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
)


# ── Unit tests for certified_accuracy_bound ──────────────────────────────


class TestCertifiedAccuracyBound:
    """Test the bound formula itself."""

    def test_zero_r_squared_gives_majority(self):
        """If R²=0, no info leaks — bound equals majority baseline."""
        assert certified_accuracy_bound(0.0, 0.6, 2) == 0.6
        assert certified_accuracy_bound(0.0, 0.8, 3) == 0.8

    def test_perfect_r_squared_gives_one(self):
        """If R²=1, full info leaks — bound should be 1.0 (or close)."""
        bound = certified_accuracy_bound(1.0, 0.5, 2)
        assert bound == 1.0

    def test_bound_at_least_majority(self):
        """Bound should never be below majority baseline."""
        for r2 in [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]:
            for pi in [0.5, 0.6, 0.7, 0.9]:
                for k in [2, 3, 5]:
                    bound = certified_accuracy_bound(r2, pi, k)
                    assert bound >= pi - 1e-10, (
                        f"Bound {bound} < majority {pi} for "
                        f"R²={r2}, pi={pi}, k={k}"
                    )

    def test_bound_at_most_one(self):
        """Bound should never exceed 1.0."""
        for r2 in [0.5, 0.9, 1.0]:
            for pi in [0.3, 0.5, 0.8]:
                for k in [2, 5, 10]:
                    bound = certified_accuracy_bound(r2, pi, k)
                    assert bound <= 1.0 + 1e-10

    def test_monotonic_in_r_squared(self):
        """Higher R² should give a higher (looser) bound."""
        r2_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
        bounds = [certified_accuracy_bound(r2, 0.6, 2) for r2 in r2_values]
        for i in range(len(bounds) - 1):
            assert bounds[i] <= bounds[i + 1] + 1e-10

    def test_balanced_binary_formula(self):
        """For balanced binary (pi=0.5, k=2), verify the closed form."""
        r2 = 0.04
        # pi_maj + sqrt(r2 * k * pi_maj * (1 - pi_maj))
        # = 0.5 + sqrt(0.04 * 2 * 0.5 * 0.5)
        # = 0.5 + sqrt(0.02) = 0.5 + 0.14142...
        expected = 0.5 + math.sqrt(0.02)
        actual = certified_accuracy_bound(r2, 0.5, 2)
        assert abs(actual - expected) < 1e-10

    def test_input_validation(self):
        with pytest.raises(ValueError, match="r_squared"):
            certified_accuracy_bound(-0.1, 0.5, 2)
        with pytest.raises(ValueError, match="r_squared"):
            certified_accuracy_bound(1.5, 0.5, 2)
        with pytest.raises(ValueError, match="majority_proportion"):
            certified_accuracy_bound(0.01, 0.0, 2)
        with pytest.raises(ValueError, match="num_classes"):
            certified_accuracy_bound(0.01, 0.5, 1)

    def test_small_r_squared_tight_bound(self):
        """With R²=0.01 and balanced binary, bound should be close to 0.5."""
        bound = certified_accuracy_bound(0.01, 0.5, 2)
        # Should be 0.5 + sqrt(0.01 * 2 * 0.25) = 0.5 + sqrt(0.005) ≈ 0.5707
        assert 0.55 < bound < 0.60


# ── Integration test: bound holds on synthetic data ──────────────────────


class TestBoundHoldsOnSyntheticData:
    """Verify the theoretical bound is valid on actual trained representations.

    Creates synthetic representations with controlled R², fits classifiers,
    and checks that their accuracy never exceeds the certified bound.
    """

    @pytest.fixture
    def balanced_binary_data(self):
        """Create representations with known R² for balanced binary labels."""
        rng = np.random.RandomState(42)
        n = 2000
        d = 16

        # Binary labels, balanced
        labels = np.array([0] * (n // 2) + [1] * (n // 2))
        rng.shuffle(labels)

        # Representations: mostly noise, one dimension correlated with label
        H = rng.randn(n, d).astype(np.float32)
        # Inject controlled correlation in first dimension
        signal_strength = 0.3  # Controls R²
        H[:, 0] += signal_strength * (2 * labels - 1)

        return H, labels

    @pytest.fixture
    def imbalanced_binary_data(self):
        """Create representations with imbalanced binary labels (70/30)."""
        rng = np.random.RandomState(123)
        n = 2000
        d = 16

        # Imbalanced: 70% class 0, 30% class 1
        n_maj = int(0.7 * n)
        labels = np.array([0] * n_maj + [1] * (n - n_maj))
        rng.shuffle(labels)

        H = rng.randn(n, d).astype(np.float32)
        signal_strength = 0.4
        H[:, 0] += signal_strength * (2 * labels - 1)

        return H, labels

    @pytest.fixture
    def multiclass_data(self):
        """Create representations with 4-class labels."""
        rng = np.random.RandomState(99)
        n = 2000
        d = 16
        k = 4

        labels = rng.randint(0, k, size=n)

        H = rng.randn(n, d).astype(np.float32)
        # Inject signal for each class
        for c in range(k):
            mask = labels == c
            H[mask, c % d] += 0.3

        return H, labels

    def _run_bound_check(self, H, labels):
        """Core check: compute R², derive bound, verify empirical acc <= bound."""
        n = len(labels)
        n_train = int(0.7 * n)
        train_H, test_H = H[:n_train], H[n_train:]
        train_labels, test_labels = labels[:n_train], labels[n_train:]

        # Get R² from linear certificate
        cert = LinearComplianceCertificate(epsilon=1.0)  # large eps so it always "passes"
        result = cert.check(test_H, test_labels)
        r_squared = result.r_squared

        # Get majority proportion
        _, counts = np.unique(labels, return_counts=True)
        majority_proportion = counts.max() / len(labels)
        num_classes = len(counts)

        # Compute theoretical bound
        bound = certified_accuracy_bound(r_squared, majority_proportion, num_classes)

        # Get empirical accuracy from post-hoc auditors
        audit = EmpiricalAudit(random_state=42)
        best_acc, results = audit.audit(train_H, train_labels, test_H, test_labels)

        # The bound must hold: empirical accuracy <= theoretical bound
        # Allow small tolerance for finite-sample effects
        tolerance = 0.05
        assert best_acc <= bound + tolerance, (
            f"Bound violated! empirical_acc={best_acc:.4f} > "
            f"bound={bound:.4f} + tol={tolerance} "
            f"(R²={r_squared:.4f}, pi_maj={majority_proportion:.2f}, k={num_classes})"
        )

        return r_squared, bound, best_acc

    def test_balanced_binary(self, balanced_binary_data):
        H, labels = balanced_binary_data
        r2, bound, emp_acc = self._run_bound_check(H, labels)
        # With low signal, bound should be reasonably tight
        assert bound < 0.85, f"Bound too loose: {bound}"

    def test_imbalanced_binary(self, imbalanced_binary_data):
        H, labels = imbalanced_binary_data
        r2, bound, emp_acc = self._run_bound_check(H, labels)

    def test_multiclass(self, multiclass_data):
        H, labels = multiclass_data
        r2, bound, emp_acc = self._run_bound_check(H, labels)

    def test_no_signal_bound_equals_majority(self):
        """With pure noise representations, R² ≈ 0 and bound ≈ majority."""
        rng = np.random.RandomState(7)
        # Use large n relative to d so OLS doesn't overfit
        n = 5000
        d = 4

        labels = np.array([0] * (n // 2) + [1] * (n // 2))
        H = rng.randn(n, d).astype(np.float32)  # Pure noise, no signal

        cert = LinearComplianceCertificate(epsilon=1.0)
        result = cert.check(H, labels)

        bound = certified_accuracy_bound(result.r_squared, 0.5, 2)
        # R² should be near zero → bound should be near 0.5
        assert result.r_squared < 0.05, f"R² too high for noise: {result.r_squared}"
        assert bound < 0.65, f"Bound should be near majority for noise: {bound}"

    def test_strong_signal_bound_near_one(self):
        """With strong signal, R² ≈ 1 and bound ≈ 1.0."""
        rng = np.random.RandomState(42)
        n = 1000
        d = 8

        labels = np.array([0] * (n // 2) + [1] * (n // 2))
        H = rng.randn(n, d).astype(np.float32)
        H[:, 0] += 5.0 * (2 * labels - 1)  # Very strong signal

        cert = LinearComplianceCertificate(epsilon=1.0)
        result = cert.check(H, labels)

        bound = certified_accuracy_bound(result.r_squared, 0.5, 2)
        assert result.r_squared > 0.5, f"R² too low for strong signal: {result.r_squared}"
        assert bound > 0.85


# ── Integration test: bound holds on PCRL synthetic dataset ──────────────


class TestBoundOnPCRLSynthetic:
    """End-to-end test using the actual PCRL synthetic dataset and encoder."""

    def test_bound_holds_after_training(self):
        """Train PCRL on synthetic data and verify bound >= empirical accuracy."""
        from pcrl.data.synthetic import SyntheticPCRLDataset, get_synthetic_purposes
        from pcrl.data.base import collate_pcrl_batch
        from pcrl.models.encoder import PurposeConditionedEncoder
        from pcrl.models.auditor import MultiAttributeAuditor
        from pcrl.models.task_head import TaskHead
        from pcrl.purposes.spec import PurposeRegistry
        from pcrl.training.trainer import PCRLTrainer, TrainerConfig
        from pcrl.evaluation.certificates import generate_report
        from torch.utils.data import DataLoader

        purposes = get_synthetic_purposes()
        registry = PurposeRegistry()
        for p in purposes:
            registry.register(p)

        train_ds = SyntheticPCRLDataset(n_samples=1000, seed=42)
        test_ds = SyntheticPCRLDataset(n_samples=500, seed=99)

        train_loader = DataLoader(
            train_ds, batch_size=64, shuffle=True,
            collate_fn=collate_pcrl_batch,
        )
        test_loader = DataLoader(
            test_ds, batch_size=64, shuffle=False,
            collate_fn=collate_pcrl_batch,
        )

        torch.manual_seed(42)
        encoder = PurposeConditionedEncoder(
            input_dim=20, hidden_dims=[64, 64], repr_dim=32,
            num_purposes=len(purposes), purpose_emb_dim=16,
        )

        task_heads = {}
        auditors = {}
        for p in purposes:
            task_name = p.allowed_tasks[0]
            task_heads[p.name] = TaskHead(repr_dim=32, output_dim=2)
            auditors[p.name] = MultiAttributeAuditor(
                repr_dim=32,
                attr_output_dims=p.disallowed_attr_dims,
                hidden_dim=64, num_layers=2,
            )

        config = TrainerConfig(
            batch_size=64, lr_encoder=1e-3, lr_auditor=1e-3,
            lambda_adv=1.0, lambda_verify=0.5,
            auditor_steps=3, epochs=10,
            early_stopping_patience=10,
            confusion_type="entropy",
        )

        trainer = PCRLTrainer(
            encoder=encoder, task_heads=task_heads, auditors=auditors,
            config=config, purpose_registry=registry, device="cpu",
        )
        trainer.train(train_loader, val_loader=test_loader)

        reports = generate_report(
            encoder=encoder,
            train_loader=train_loader,
            test_loader=test_loader,
            purpose_registry=registry,
            device="cpu",
        )

        tolerance = 0.05
        for r in reports:
            bound = certified_accuracy_bound(
                r.linear_r2, r.majority_proportion, r.num_classes,
            )
            assert r.empirical_best_acc <= bound + tolerance, (
                f"Bound violated for {r.purpose_name}/{r.attr_name}: "
                f"emp={r.empirical_best_acc:.4f} > bound={bound:.4f}+{tolerance} "
                f"(R²={r.linear_r2:.4f}, pi={r.majority_proportion:.2f})"
            )
