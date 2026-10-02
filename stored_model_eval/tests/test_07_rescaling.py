"""Test 7: rescaling H flips the per_dim_std health label and moves the fixed-penalty ridge R2 slightly,
while OLS R2, rho1^2 and the held-out AUC are invariant."""
import numpy as np

from stored_model_eval.fixtures import make_synthetic
from stored_model_eval.metrics import (auc_binary, cca_rho2, cca_rho2_heldout, health, ols_heldout_scores, ols_r2,
                                       r2_onehot_relridge, r2_onehot_ridge)


def test_rescaling_health_vs_invariants(record_property):
    fx = make_synthetic("direct", n_units=1000, seed=17, signal=0.15)
    H, S = fx["H"], fx["S"]
    fi, ei = np.arange(600), np.arange(600, 1000)
    c = 0.01
    h1, h2 = health(H), health(c * H)
    assert h1["label"] == "healthy" and h2["label"] == "collapsed"  # scale-dependent label flips
    assert abs(h2["per_dim_std_mean"] - c * h1["per_dim_std_mean"]) < 1e-12
    assert abs(h2["effective_rank"] - h1["effective_rank"]) < 1e-9
    r_a, r_b = r2_onehot_ridge(H, S), r2_onehot_ridge(c * H, S)
    assert 1e-12 < abs(r_a - r_b) < 1e-3, (r_a, r_b)  # fixed absolute penalty: small, real change
    assert abs(ols_r2(H, S) - ols_r2(c * H, S)) < 1e-10
    assert abs(ols_r2(H, S, fi, ei) - ols_r2(c * H, S, fi, ei)) < 1e-10
    assert abs(cca_rho2(H, S)["value"] - cca_rho2(c * H, S)["value"]) < 1e-10
    assert abs(cca_rho2_heldout(H, S, fi, ei) - cca_rho2_heldout(c * H, S, fi, ei)) < 1e-10
    for rho in (0.0, 1e-4, 1e-2):  # methodology recipe R02: relative floor + relative ridge -> invariant
        assert abs(r2_onehot_relridge(H, S, fi, ei, rho) - r2_onehot_relridge(c * H, S, fi, ei, rho)) < 1e-10
    # conventions differ only in SS_tot centring (R02: fit-row means; ols_r2 held-out: score-row means)
    assert 0 < r2_onehot_relridge(H, S, fi, ei, 0.0) - ols_r2(H, S, fi, ei) < 0.02  # fit-mean SS_tot >= score-mean SS_tot
    a1 = auc_binary(S[ei] == 1, ols_heldout_scores(H, S, fi, ei)[:, 1])
    a2 = auc_binary(S[ei] == 1, ols_heldout_scores(c * H, S, fi, ei)[:, 1])
    record_property("health_labels", [h1["label"], h2["label"]])
    record_property("per_dim_std", [h1["per_dim_std_mean"], h2["per_dim_std_mean"]])
    record_property("ridge_r2", [r_a, r_b]); record_property("ridge_r2_abs_change", abs(r_a - r_b))
    record_property("ols_r2_abs_change", abs(ols_r2(H, S) - ols_r2(c * H, S)))
    record_property("heldout_auc", [a1, a2])
    assert abs(a1 - a2) < 1e-12 and a1 > 0.55
