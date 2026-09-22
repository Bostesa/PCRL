"""Tests for the claims-foundation audit.

Two kinds of test:
* exact fixtures (analysis/pcrl_claims_foundation_v1/exact_fixtures.py), each
  asserting a counterexample or identity on an explicit rational table;
* implementation parity: the repository's own aggregate one-hot certificate and
  dominant-axis function, loaded from source, must satisfy the convex-combination
  identity to numerical precision when given matched rows, and must refuse
  incomplete class support rather than silently reweighting.
"""

from __future__ import annotations

import ast
import sys
import types
from fractions import Fraction as F
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.pcrl_claims_foundation_v1 import exact_fixtures as fx  # noqa: E402


def _load_functions(path: Path, names: set[str]) -> dict:
    """Execute only the named top-level functions from a module's source."""
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = names - {n.name for n in body}
    assert not missing, missing
    try:
        import torch  # noqa: F401
    except ImportError:  # minimal stand-in: the functions only use isinstance
        torch = types.SimpleNamespace(Tensor=type("Tensor", (), {}))
    ns = {"np": np, "torch": torch, "__name__": "loaded"}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


def _impl():
    ver = _load_functions(ROOT / "pcrl/purposes/verification.py", {"_prepare_targets", "_empirical_ridge_fit"})
    cert = _load_functions(ROOT / "pcrl/evaluation/certificates.py", {"compute_dominant_axis_r2"})
    return ver, cert


# --------------------------------------------------------------------- exact

def test_every_exact_fixture_passes():
    out = fx.run_all()
    assert set(out) == {f.__name__ for f in fx.FIXTURES}


def test_retired_guarantee_counterexample_is_exact():
    r = fx.f1_retired_accuracy_guarantee()
    assert r["affine_ls_r2"] == "0" and r["threshold_h_gt_0_accuracy"] == "9/10"


def test_rare_class_bound_is_vacuous_when_class_weight_small():
    r = fx.f2b_rare_class_masking()
    assert r["rare_class_r2"] == "1" and r["aggregate_float"] < 0.1 and r["bound_is_vacuous"]


def test_max_ovr_is_not_the_worst_sensitive_direction():
    r = fx.f3_max_ovr_vs_all_directions()
    assert F(r["max_ovr_r2_DA"]) < F(r["r2_best_class_contrast"])


def test_approximate_composition_bound_is_tight_in_the_example():
    r = fx.f4_composition()
    assert r["individual_r2"] == ["1/101", "1/101"] and r["joint_r2"] == "1"
    assert r["lambda_min_bound"] == "1"


def test_xor_coarsening_hides_full_conditional_leakage():
    r = fx.f5_xor_coarsening()
    assert r["I(S;Z)"] == "0" and r["I(S;Z|H)"] == "1*log2" and r["I(S;Z|coarse(H)=const)"] == "0"


def test_restricted_family_increment_is_not_a_cmi_floor():
    r = fx.f9_restricted_family_increment_without_information()
    assert r["I(S;Z|H)"] == "0" and r["measured_increment_nats"] > 0.6


# ------------------------------------------------------ implementation parity

def test_repository_aggregate_equals_prevalence_weighted_per_class_scores():
    ver, cert = _impl()
    rng = np.random.default_rng(0)
    n, d, K = 3000, 6, 5
    priors = np.array([0.62, 0.2, 0.1, 0.06, 0.02])
    y = rng.choice(K, size=n, p=priors)
    H = rng.normal(size=(n, d)) + 0.4 * np.eye(K, d)[y] * np.array([1, 0.5, 2, 3, 4])[y, None]
    _, _, agg, _, _ = ver["_empirical_ridge_fit"](H, y, 1e-6, K)
    da = cert["compute_dominant_axis_r2"](H, y, num_classes=K)
    p = np.asarray(da["priors"])
    w = p * (1 - p) / (p * (1 - p)).sum()
    assert abs(agg - float(w @ np.asarray(da["per_class_r2"]))) < 1e-12
    assert da["r2_da"] == max(da["per_class_r2"])


def test_repository_identity_holds_for_float32_input_after_cast():
    """Current code casts to float64 in both paths (the historical aggregate did not)."""
    ver, cert = _impl()
    rng = np.random.default_rng(1)
    n, K = 2000, 4
    y = rng.choice(K, size=n, p=[0.5, 0.3, 0.15, 0.05])
    base = rng.normal(size=(n, 1))
    H = np.hstack([base, base + 1e-4 * rng.normal(size=(n, 1)), np.eye(K)[y][:, :2] * 0.1]).astype(np.float32)
    _, _, agg, _, _ = ver["_empirical_ridge_fit"](H, y, 1e-6, K)
    da = cert["compute_dominant_axis_r2"](H, y, num_classes=K)
    p = np.asarray(da["priors"])
    w = p * (1 - p) / (p * (1 - p)).sum()
    assert abs(agg - float(w @ np.asarray(da["per_class_r2"]))) < 1e-9


def test_repository_refuses_missing_declared_class():
    ver, cert = _impl()
    rng = np.random.default_rng(2)
    H = rng.normal(size=(100, 3))
    y = rng.choice(3, size=100)  # class 3 of 4 is absent
    _, _, agg, _, valid = ver["_empirical_ridge_fit"](H, y, 1e-6, 4)
    da = cert["compute_dominant_axis_r2"](H, y, num_classes=4)
    assert np.isnan(agg) and not valid[3]
    assert np.isnan(da["r2_da"]) and not da["coverage_complete"]
