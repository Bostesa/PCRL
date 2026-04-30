"""Tests for Dominant-Axis Auditing (Framework D).

Validates:
1. compute_dominant_axis_r2 returns the correct argmax class and per-class R²
   on synthetic data where one class is engineered to be linearly separable.
2. The Convex-Combination Identity:
       R²_onehot = Σ_k w_k · R²_OvR_k,    w_k = π_k(1-π_k) / Σ_j π_j(1-π_j)
   holds within tolerance on synthetic multi-class data.
"""
from __future__ import annotations

import numpy as np
import pytest

from pcrl.evaluation.certificates import (
    compute_dominant_axis_r2,
    compute_mlp_ovr_delta,
)
from pcrl.purposes.verification import LinearComplianceCertificate


def _make_synthetic_multiclass(
    n: int = 4000,
    d: int = 16,
    priors: tuple[float, ...] = (0.85, 0.10, 0.03, 0.015, 0.005),
    target_ovr_r2_class1: float = 0.30,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct features H and labels y so that class 1 is linearly recoverable
    with approximately ``target_ovr_r2_class1`` and other classes have small R².

    Strategy: sample features from N(0, I); inject a signal into one direction
    that scales with z_1 (1 if class==1 else 0). Calibrate signal magnitude to
    hit the target OvR R² for class 1.
    """
    rng = np.random.RandomState(seed)
    priors = np.asarray(priors, dtype=np.float64)
    priors = priors / priors.sum()
    K = len(priors)

    # Sample labels from prior
    y = rng.choice(K, size=n, p=priors)

    # Base features iid N(0,1)
    H = rng.randn(n, d).astype(np.float64)

    # Inject a 1-class-vs-rest signal: project (z_1 - π_1) onto direction e_0.
    # OvR R² for class 1 in pop limit equals: Var(signal) / [Var(signal) + sigma_noise^2]
    # where signal = a * (z_1 - π_1), noise has unit variance in that direction.
    # Set Var(signal) = a² * π_1*(1 - π_1). Solve a so that OvR R²_1 = target.
    pi1 = priors[1]
    var_signal_per_a2 = pi1 * (1.0 - pi1)
    target = target_ovr_r2_class1
    # R² = a²·v / (a²·v + 1) ⇒ a² = R²/(v·(1-R²))
    a_sq = target / max(var_signal_per_a2 * (1.0 - target), 1e-12)
    a = np.sqrt(a_sq)

    z1 = (y == 1).astype(np.float64)
    H[:, 0] = H[:, 0] + a * (z1 - pi1)

    return H, y


class TestDominantAxisR2:
    """Tests for compute_dominant_axis_r2."""

    def test_argmax_picks_injected_class(self):
        """OvR R² for the engineered class should dominate."""
        H, y = _make_synthetic_multiclass(seed=0)
        result = compute_dominant_axis_r2(H, y)
        # argmax must be class 1 (the one we injected).
        assert result["argmax_class"] == 1, (
            f"expected argmax=1, got {result['argmax_class']}; "
            f"per-class R²={result['per_class_r2']}"
        )

    def test_max_r2_close_to_target(self):
        """R²_DA should match the engineered target ~0.30 within ±0.05."""
        H, y = _make_synthetic_multiclass(seed=0)
        result = compute_dominant_axis_r2(H, y)
        assert abs(result["r2_da"] - 0.30) < 0.05, (
            f"R²_DA = {result['r2_da']:.4f}, expected ~0.30"
        )

    def test_minority_classes_small(self):
        """Minority non-injected classes (k>=2) should each have R²_OvR < 0.05.

        Note: injecting on class 1 contaminates the majority class 0 via
        z_0 = 1 - Σ_{k>=1} z_k (mutual exclusivity), so we don't claim R²_0
        is small. Only the *minority* non-injected classes are.
        """
        H, y = _make_synthetic_multiclass(seed=0)
        result = compute_dominant_axis_r2(H, y)
        per_class = result["per_class_r2"]
        for k in range(2, len(per_class)):
            assert per_class[k] < 0.05, (
                f"minority class {k}: R²_OvR={per_class[k]:.4f} should be < 0.05; "
                f"all classes: {per_class}"
            )

    def test_per_class_array_length_matches_K(self):
        """per_class_r2 must have one entry per class (5 classes here)."""
        H, y = _make_synthetic_multiclass(seed=0)
        result = compute_dominant_axis_r2(H, y)
        assert len(result["per_class_r2"]) == 5

    def test_convex_combination_identity(self):
        """R²_onehot ≈ Σ_k w_k · R²_OvR_k where w_k = π_k(1-π_k)/Σπ_j(1-π_j)."""
        H, y = _make_synthetic_multiclass(seed=0)

        result = compute_dominant_axis_r2(H, y)
        per_class_r2 = np.asarray(result["per_class_r2"])

        # Compute observed one-hot R² with the same regularization the OvR
        # solves use (so the identity is meaningful).
        cert = LinearComplianceCertificate(epsilon=0.01, regularization=1e-6)
        observed_r2_onehot = cert.check(H, y).r_squared

        # Compute weights from empirical priors.
        K = len(per_class_r2)
        _, counts = np.unique(y, return_counts=True)
        priors = counts / counts.sum()
        weights = priors * (1.0 - priors)
        weights = weights / weights.sum()

        predicted_r2_onehot = float(np.dot(weights, per_class_r2))

        assert abs(predicted_r2_onehot - observed_r2_onehot) < 0.01, (
            f"Convex-combination identity failed: "
            f"predicted={predicted_r2_onehot:.4f}, "
            f"observed={observed_r2_onehot:.4f}, "
            f"per_class={per_class_r2.tolist()}, weights={weights.tolist()}"
        )


class TestMLPOvRDelta:
    """Tests for compute_mlp_ovr_delta."""

    def test_argmax_picks_injected_class_and_delta_positive(self):
        """On synthetic data with a strong class-1 signal and balanced
        priors, MLP delta is largest for class 1 and clearly positive.

        Balanced priors are used here because the OvR delta metric is
        inherently biased toward majority classes when injection on a
        minority class indirectly carries information about the majority
        (via z_0 = 1 - Σ_{k>=1} z_k mutual exclusivity). With balanced
        priors, the contamination is symmetric across classes, leaving the
        injected class with the largest delta.
        """
        rng = np.random.RandomState(0)
        K = 5
        priors = np.full(K, 1.0 / K)
        n_tr, n_te, d = 4000, 2000, 8

        def _make(n, seed):
            r = np.random.RandomState(seed)
            y = r.choice(K, size=n, p=priors)
            H = r.randn(n, d).astype(np.float64) * 0.5
            # Strong, clean class-1 separation: shift in a fixed direction
            # only for class-1 rows. No other class indicator carries any
            # extra info about this dimension beyond "is class 1?".
            H[y == 1, 0] += 4.0
            return H, y

        H_tr, y_tr = _make(n_tr, 0)
        H_te, y_te = _make(n_te, 1)

        # Fast config (small hidden, few epochs) to keep the test quick
        # while still demonstrating the metric works.
        result = compute_mlp_ovr_delta(
            H_tr, y_tr, H_te, y_te,
            hidden=64, epochs=15, lr=1e-3, dropout=0.0,
            batch_size=512, random_state=0,
        )

        per_class_delta = result["per_class_delta"]
        assert len(per_class_delta) == 5
        assert result["argmax_class"] == 1, (
            f"expected argmax=1 (the engineered class), got "
            f"{result['argmax_class']}; per_class={per_class_delta}"
        )
        # With strong injection on a balanced 1/5-prior class, baseline=0.8
        # and the MLP should achieve high test accuracy → delta well above 0.02.
        assert result["mlp_da_delta"] > 0.05, (
            f"max delta {result['mlp_da_delta']:.4f} too small; "
            f"per_class={per_class_delta}"
        )

    def test_returns_correct_keys(self):
        """Result dict has the expected schema."""
        rng = np.random.RandomState(0)
        H_tr = rng.randn(200, 8)
        y_tr = rng.randint(0, 3, size=200)
        H_te = rng.randn(100, 8)
        y_te = rng.randint(0, 3, size=100)
        result = compute_mlp_ovr_delta(
            H_tr, y_tr, H_te, y_te,
            hidden=16, epochs=2, batch_size=64, random_state=0,
        )
        assert "mlp_da_delta" in result
        assert "argmax_class" in result
        assert "per_class_delta" in result
        assert "per_class_acc" in result
        assert "per_class_baseline" in result
        assert len(result["per_class_delta"]) == 3
