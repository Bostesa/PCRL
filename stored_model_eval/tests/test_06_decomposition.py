"""Test 6: the surface / attacker / metric decomposition sums exactly, is reported per axis, exposes order
dependence, and refuses to subtract quantities on different scales."""
import numpy as np
import pytest

from stored_model_eval.access import decompose, decomposition_rows
from stored_model_eval.attackers import LinearAttacker
from stored_model_eval.fixtures import make_synthetic
from stored_model_eval.metrics import macro_ovr_auc, worst_class_auc
from stored_model_eval.surfaces import build_surface


def test_decomposition_sums_and_is_separate():
    S_, A_, M_ = ("rep", "rep+outputs"), ("linear", "gbt"), ("macro", "worst_pair")
    base = {"rep": 0.50, "rep+outputs": 0.62}
    att = {"linear": 0.0, "gbt": 0.05}
    met = {"macro": 0.0, "worst_pair": 0.08}
    vals = {}
    for s in S_:
        for a in A_:
            for m in M_:
                inter = 0.03 if (s == "rep+outputs" and a == "gbt") else 0.0  # surface x attacker interaction
                vals[(s, a, m)] = base[s] + att[a] + met[m] + inter
    d = decompose(vals, ("rep", "linear", "macro"), ("rep+outputs", "gbt", "worst_pair"),
                  scales={"macro": "auc", "worst_pair": "auc"})
    assert abs(d["total"] - (0.12 + 0.05 + 0.08 + 0.03)) < 1e-12
    assert abs(d["sum_check"]["sequential"]) < 1e-12 and abs(d["sum_check"]["shapley"]) < 1e-12
    seq = d["sequential_surface_attacker_metric"]
    assert seq["surface"] == pytest.approx(0.12) and seq["attacker"] == pytest.approx(0.08)
    assert seq["metric"] == pytest.approx(0.08)
    assert d["shapley"]["surface"] == pytest.approx(0.135) and d["shapley"]["attacker"] == pytest.approx(0.065)
    assert d["order_dependence_range"]["surface"] == pytest.approx((0.12, 0.15))
    rows = decomposition_rows(d)
    assert [r["axis"] for r in rows] == ["surface", "attacker", "metric", "total"]
    with pytest.raises(ValueError):
        decompose({**vals, ("rep", "linear", "r2"): 0.01}, ("rep", "linear", "r2"),
                  ("rep+outputs", "gbt", "worst_pair"), scales={"r2": "r2", "worst_pair": "auc"})


def test_decomposition_on_fitted_surfaces(auth, record_property):
    fx = make_synthetic("output_leak", n_units=1200, seed=16)
    r = fx["roles"]
    fi, vi, ei = (np.flatnonzero(r == k) for k in ("attacker_fit", "attacker_val", "evaluation"))
    vals = {}
    for s in ("rep", "rep+outputs"):
        X, _ = build_surface(s, fx["H"], fx["outputs"])
        att = LinearAttacker().fit(X[fi], fx["S"][fi], X[vi], fx["S"][vi], auth=auth, synthetic=True)
        P = att.predict_proba(X[ei])
        vals[(s, "linear", "macro")] = macro_ovr_auc(fx["S"][ei], P, 20)
        vals[(s, "linear", "worst_class")] = worst_class_auc(fx["S"][ei], P, 20)["value"]
    d = decompose(vals, ("rep", "linear", "macro"), ("rep+outputs", "linear", "worst_class"))
    record_property("values", {"|".join(k): v for k, v in vals.items()})
    record_property("sequential", d["sequential_surface_attacker_metric"])
    assert d["sequential_surface_attacker_metric"]["surface"] > 0.2  # the outputs carry the attribute
    assert d["sequential_surface_attacker_metric"]["attacker"] == 0.0
    assert abs(d["sum_check"]["sequential"]) < 1e-12
