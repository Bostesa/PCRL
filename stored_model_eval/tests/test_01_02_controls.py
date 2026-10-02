"""Tests 1-2: positive control is recovered; null control is UNRESOLVED (never 'pass')."""
import numpy as np

from stored_model_eval.attackers import GBTAttacker, LinearAttacker
from stored_model_eval.fixtures import make_synthetic
from stored_model_eval.inference import DECISIONS, cluster_bootstrap, decide_all, macro_auc_stat, resolve_units
from stored_model_eval.metrics import cca_rho2, ols_r2


def _fit_score(fx, att, auth, cfg):
    r = fx["roles"]
    fi, vi, ei = (np.flatnonzero(r == k) for k in ("attacker_fit", "attacker_val", "evaluation"))
    att.fit(fx["H"][fi], fx["S"][fi], fx["H"][vi], fx["S"][vi], auth=auth, synthetic=True)
    P = att.predict_proba(fx["H"][ei])
    units = resolve_units(fx["units"][ei], fx["record_keys"][ei])
    res = cluster_bootstrap(macro_auc_stat(fx["S"][ei], P, 20), units["unit_index"], n_boot=cfg["bootstrap"]["n_boot"],
                            seed=1)
    return res, decide_all(res, cfg["bars"])


def test_positive_control_recovered(auth, cfg, record_property):
    fx = make_synthetic("direct", n_units=1500, seed=11)
    res, dec = _fit_score(fx, LinearAttacker(), auth, cfg)
    record_property("auc_point", res["point"]); record_property("auc_interval", res["interval"])
    record_property("decisions", dec)
    assert res["point"] > 0.9
    assert set(dec.values()) == {"ESTABLISHED_ABOVE"}
    assert ols_r2(fx["H"], fx["S"]) > 0.5
    assert cca_rho2(fx["H"], fx["S"])["value"] > 0.5


def test_null_control_unresolved_not_pass(auth, cfg, record_property):
    fx = make_synthetic("null", n_units=1000, seed=12)
    for att in (LinearAttacker(), GBTAttacker({"max_iter": 50})):
        res, dec = _fit_score(fx, att, auth, cfg)
        lo, hi = res["interval"]
        record_property(f"{att.name}_point", res["point"]); record_property(f"{att.name}_interval", res["interval"])
        record_property(f"{att.name}_decisions", dec)
        assert lo < 0.5 < hi, "null interval must straddle chance"
        assert dec["0.52"] == "UNRESOLVED"
        assert "ESTABLISHED_ABOVE" not in dec.values()
        assert set(dec.values()) <= set(DECISIONS)
        assert not any("pass" in v.lower() for v in dec.values())
    assert "pass" not in " ".join(DECISIONS).lower()
