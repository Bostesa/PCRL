"""Test 4: a minority-class contrast that pooled one-hot R2 and per-class averaging hide is revealed by
rho1^2; max pair AUC and the contrast R2 are reported as distinct quantities."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from stored_model_eval.fixtures import make_synthetic
from stored_model_eval.metrics import contrast_report, per_class_r2, r2_onehot_ridge


def test_contrast_hidden_by_averaging_revealed_by_rho2(record_property):
    fx = make_synthetic("contrast", n_units=6000, d=6, K=30, seed=14)
    fx["H"][:, 0] += 0.15 * np.random.default_rng(0).normal(size=len(fx["S"]))  # extra noise on the contrast axis
    H, S = fx["H"], fx["S"]
    P = LinearDiscriminantAnalysis().fit(H, S).predict_proba(H)  # synthetic-only scoring model
    rep = contrast_report(H, S, P, min_support=20)
    for k, v in rep.items():
        record_property(k, v)
    # the check passes and averaging hides it ...
    assert rep["pooled_onehot_r2"] <= 0.05
    assert rep["per_class_r2_mean"] <= 0.05
    assert rep["macro_ovr_auc"] - 0.5 < (rep["max_pair_auc"] - 0.5) / 3  # macro averaging dilutes the pair signal
    # ... but the canonical diagnostic reveals it
    assert rep["rho1_sq"] > 0.5
    assert rep["rho1_sq"] > rep["per_class_r2_max"] > rep["pooled_onehot_r2"]
    assert abs(rep["canonical_contrast_r2"] - rep["rho1_sq"]) < 1e-3
    # distinct quantities, distinct scales: pair AUC (ranking) is not the contrast R2 (variance explained)
    assert rep["max_pair"] == (0, 1)
    assert rep["max_pair_auc"] > 0.99
    assert abs(rep["max_pair_auc"] - rep["canonical_contrast_r2"]) > 0.1
    assert {"max_pair_auc", "canonical_contrast_r2", "rho1_sq"} <= set(rep)
    # convex-combination identity that defines the PCRL pooled convention
    pc = per_class_r2(H, S)
    pri = np.array(pc["priors"])
    w = pri * (1 - pri) / (pri * (1 - pri)).sum()
    assert abs(float(np.dot(w, pc["per_class"])) - r2_onehot_ridge(H, S)) < 1e-9
