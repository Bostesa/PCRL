"""Tests for the guarantee review. Exact fixtures assert counterexamples and identities on explicit
tables. Numerical fixtures are coverage checks, not proofs. The prospective check re-derives
the registered decisions from committed aggregates. It skips if the pinned commit is unavailable."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.pcrl_guarantee_review_v1 import composition_fixtures as cf  # noqa: E402
from analysis.pcrl_guarantee_review_v1 import fullview_fixtures as fv  # noqa: E402
from analysis.pcrl_guarantee_review_v1 import kernel_guarantees as kg  # noqa: E402


def test_chain_rule_both_signs_exact():
    r = fv.g1_chain_rule()
    assert r["xor"]["I(S;Z|H)"] == "log2" and r["xor"]["I(S;Z|B)"] == "0"
    assert r["overstatement"]["I(S;Z|H)"] == "0" and r["overstatement"]["I(S;Z|B)"] == "log2"


def test_radius_bound_holds_and_is_tight():
    r = fv.g2_row_radius_bound(trials=150)
    assert r["violations"] == 0 and r["tightness_bsc_quarter"]["I(S;Z|W)"] == r["tightness_bsc_quarter"]["radius"]


def test_state_dependent_kernel_bounds_only_views_containing_state():
    r = fv.g2b_public_state_dependence()
    assert r["I(S;Z)"] == "log2"


def test_utility_ceiling_equals_bayes_gain_exact():
    assert "3/4 log3 - log2" in fv.g3_utility_ceiling()["I(Y;Z|H)"]


def test_deterministic_kernel_guarantee_is_vacuous():
    D = [[1.0 if z == t else 0.0 for z in range(17)] for t in range(17)]
    out = kg.analyse(D)
    assert out["eta_TV"] == 1.0 and not out["informative_vs_log2"] and not out["informative_vs_log9"]
    assert abs(out["radius_upper_certificate"] - math.log(17)) < 1e-9


def test_kernel_script_certificate_on_bsc():
    out = kg.analyse([[0.75, 0.25], [0.25, 0.75]])
    assert abs(out["radius_upper_certificate"] - (0.75 * math.log(3) - math.log(2))) < 1e-9


def test_envelope_sampling_underestimates_and_bin_radius_does_not_cover_h():
    r = fv.g5_envelope()
    assert r["max_over_2000_interior_samples_nats"] < r["exact_vertex_worst_case_nats"]
    assert r["bin_level_vs_h_level"]["I(S;T|H)"] == "log2"


def test_unrestricted_envelope_impossible_and_homogeneity_makes_binned_conservative():
    assert fv.g6_unrestricted_impossibility()["nats"] > 0
    g8 = fv.g8_conditional_homogeneity()
    assert g8["assumption_holds"]["binned_is_upper_bound"]
    assert not g8["assumption_violated"]["binned_is_upper_bound"]


def test_composition_lemma_and_ridge_form():
    assert cf.check(trials=300)["violations_pooled"] == 0
    assert cf.ridge_form_check(trials=2000)["violations"] == 0


def test_prospective_decisions_reproduce():
    pin = "5e154e5c4fdaeb23d327a0ebefe838525f1a19cb"
    if subprocess.run(["git", "cat-file", "-e", pin], cwd=ROOT, capture_output=True).returncode:
        pytest.skip("pinned evidence commit not available")
    out = json.loads(subprocess.check_output(
        [sys.executable, str(ROOT / "analysis/pcrl_guarantee_review_v1/verify_prospective.py")], cwd=ROOT))
    for c in ("Q", "D17"):
        p = out["primary"][c]
        assert p["clauses_passed"] == 8 and not p["conjunction_passed"] and p["task_passed"] == 0
        assert p["task_ub_negative_both"]
    assert out["sensitive_all_pass_at_one_sided_bonferroni_20"] == {"Q": True, "D17": True}
    assert out["Q_vs_D17"]["D17_task_better_resolved"] == ["unweighted"]
    assert not out["Q_vs_D17"]["full_domination_by_D17"]
