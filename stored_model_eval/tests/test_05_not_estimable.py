"""Test 5: absent, singleton and below-support classes are NOT_ESTIMABLE with counts and coverage; the
sentinel can never be read as pass/fail/0/1."""
import numpy as np
import pytest

from stored_model_eval.inference import cluster_bootstrap, decide, decide_all, macro_auc_stat
from stored_model_eval.metrics import (NotEstimable, cca_rho2, is_estimable, macro_ovr_auc, pairwise_auc,
                                       per_class_ovr_auc, worst_class_auc)
from stored_model_eval.pipeline import classify_outcome


def _data():
    rng = np.random.default_rng(5)
    y = np.r_[np.zeros(200), np.ones(200), np.full(5, 2), [3]].astype(int)  # class 4 absent
    P = rng.dirichlet(np.ones(5), size=len(y))
    return y, P


def test_unsupported_classes_not_estimable():
    y, P = _data()
    pc = per_class_ovr_auc(y, P, min_support=20)
    assert is_estimable(pc[0]) and is_estimable(pc[1])
    for k, n, kind in ((2, 5, "below_min_support"), (3, 1, "singleton"), (4, 0, "absent")):
        assert isinstance(pc[k], NotEstimable)
        assert pc[k].counts["n_pos"] == n and kind in pc[k].reason
    m = macro_ovr_auc(y, P, min_support=20)
    assert isinstance(m, NotEstimable) and m.coverage == (2, 5)
    assert "macro_over_supported_classes" in m.partial
    wp = pairwise_auc(y, P, min_support=20)
    assert wp["coverage"] == (1, 10) and wp["argmax"] == (0, 1)
    assert sum(isinstance(v, NotEstimable) for v in wp["per_pair"].values()) == 9
    wc = worst_class_auc(y, P, min_support=20)
    assert wc["coverage"] == (2, 5) and set(wc["unsupported"]) == {2, 3, 4}
    none = worst_class_auc(y, P, min_support=500)
    assert isinstance(none["value"], NotEstimable)
    j = m.to_json()
    assert j["status"] == "NOT_ESTIMABLE" and j["coverage"]["fraction"] == 0.4


def test_sentinel_is_never_pass_fail_or_number():
    ne = NotEstimable("x", counts={"n": 1})
    with pytest.raises(TypeError):
        bool(ne)
    with pytest.raises(TypeError):
        ne < 0.55  # noqa: B015
    with pytest.raises(TypeError):
        float(ne)
    assert decide(ne, 0.55) == "NOT_ESTIMABLE"
    y, P = _data()
    res = cluster_bootstrap(macro_auc_stat(y, P, 20), np.arange(len(y)), n_boot=50)
    assert isinstance(res["point"], NotEstimable)
    assert set(decide_all(res, [0.52, 0.55, 0.60]).values()) == {"NOT_ESTIMABLE"}
    assert isinstance(cca_rho2(np.ones((10, 3)), np.zeros(10, int)), NotEstimable)
    c = classify_outcome(fitting_check_passed=None, heldout_check_passed=None, attack_decision="NOT_ESTIMABLE",
                         attacker_family="gbt", surface="rep", scope={})
    assert c["category"] == "C5"
