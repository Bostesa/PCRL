"""Test 3: an XOR signal with zero linear cross-covariance passes every linear check yet is recovered by a
nonlinear attacker; the outcome is C3 (outside the linear guarantee's scope), never C4."""
import numpy as np

from stored_model_eval.attackers import GBTAttacker, LinearAttacker
from stored_model_eval.fixtures import make_synthetic
from stored_model_eval.inference import cluster_bootstrap, decide_all, macro_auc_stat, resolve_units
from stored_model_eval.metrics import cca_rho2, cross_covariance_norm, ols_r2, r2_onehot_ridge
from stored_model_eval.pipeline import classify_outcome


def test_xor_is_c3_not_c4(auth, cfg, record_property):
    fx = make_synthetic("xor", n_units=2000, seed=13)
    H, S, r = fx["H"], fx["S"], fx["roles"]
    fi, vi, ei = (np.flatnonzero(r == k) for k in ("attacker_fit", "attacker_val", "evaluation"))
    # linear world: zero cross-covariance, R2 ~ 0 in-sample (fitting check) and held-out
    # population cross-covariance is exactly 0; the sample one is at the noise floor (|z| < 4 per column)
    z = np.array([np.corrcoef(H[:, j], S)[0, 1] for j in range(H.shape[1])]) * np.sqrt(len(S))
    assert np.abs(z).max() < 4, z
    assert cross_covariance_norm(H, S) < 4 * np.sqrt(H.shape[1] * 2 * 0.25 / len(S))
    fit_r2 = r2_onehot_ridge(H[fi], S[fi])
    held_r2 = r2_onehot_ridge(H, S, fit_idx=fi, score_idx=ei, clamp=False)
    assert fit_r2 <= cfg["r2"]["tau"] and held_r2 <= cfg["r2"]["tau"]
    assert ols_r2(H, S, fit_idx=fi, score_idx=ei) < 0.01
    assert cca_rho2(H, S)["value"] < 0.01  # the linear contrast diagnostic is also blind, correctly
    units = resolve_units(fx["units"][ei])["unit_index"]
    out = {}
    for name, att in (("linear", LinearAttacker()), ("gbt", GBTAttacker({"max_iter": 100}))):
        att.fit(H[fi], S[fi], H[vi], S[vi], auth=auth, synthetic=True)
        res = cluster_bootstrap(macro_auc_stat(S[ei], att.predict_proba(H[ei]), 20), units, n_boot=300, seed=3)
        record_property(f"{name}_auc", [res["point"], res["interval"]])
        out[name] = decide_all(res, cfg["bars"])
    record_property("fit_r2", fit_r2); record_property("heldout_r2", held_r2)
    record_property("max_abs_corr_z", float(np.abs(z).max())); record_property("decisions", out)
    assert out["gbt"]["0.60"] == "ESTABLISHED_ABOVE"
    assert out["linear"]["0.60"] != "ESTABLISHED_ABOVE"
    for population in (False, True):
        scope = {"attacker_classes": ["linear"], "surfaces": ["rep"], "metric": "macro_ovr_auc",
                 "population": population}
        c = classify_outcome(fitting_check_passed=fit_r2 <= 0.05, heldout_check_passed=held_r2 <= 0.05,
                             attack_decision=out["gbt"]["0.60"], attacker_family="gbt", surface="rep",
                             scope=scope, metric="macro_ovr_auc")
        assert c["category"] == "C3", c
    # the same recovery by an in-scope attacker under a population guarantee would be C4
    c4 = classify_outcome(fitting_check_passed=True, heldout_check_passed=True, attack_decision="ESTABLISHED_ABOVE",
                          attacker_family="linear", surface="rep", metric="macro_ovr_auc",
                          scope={"attacker_classes": ["linear"], "surfaces": ["rep"], "metric": "macro_ovr_auc",
                                 "population": True})
    assert c4["category"] == "C4"
