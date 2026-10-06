"""Synthetic tests of the qpc utility / audit / baselines / assessment code (role D; no real data, no real labels).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q qpc/tests/test_audit.py
"""
from __future__ import annotations

import numpy as np
import pytest

from dpc import utility as DU
from qpc import utility as UT


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def DT():
    return UT.synthetic_task_D(seed=0)


@pytest.fixture(scope="module")
def TU(DT):
    return UT.synthetic_teacher(DT, seed=0)


def _class_mean_release(D, t):
    """Class-preserving 1-state-per-class code of a teacher: q = smoothed DEFENSE_FIT mean of p within the class."""
    fit = D["idx"]["OSF_DEFENSE_FIT"]
    z = {"row_id": t["row_id"]}
    for i, K in ((1, 2), (2, 6)):
        p, d = t[f"p{i}"], t[f"d{i}"]
        Q = np.zeros((K, K))
        for c in range(K):
            m = fit[d[fit] == c]
            mu = p[m].mean(0) if len(m) else np.eye(K)[c]
            eps = 1e-12
            Q[c] = (mu + eps + eps * np.eye(K)[c]) / (1 + (K + 1) * eps)
        z[f"tok{i}"], z[f"q{i}"], z[f"hard{i}"], z[f"alpha{i}"] = d, Q[d], Q[d].argmax(1), np.asarray(K)
    return z


def _m(acc, ll, br, const=0.5, n=100):
    return {"acc": acc, "logloss": ll, "brier": br, "const_acc": const, "n": n}


def _pair(a, b):
    return {"income": a, "occupation": b}


# ------------------------------------------------------------------ metric conventions
def test_task_metrics_conventions_exact():
    y = np.array([0, 1, 1, 0])
    P = np.array([[1.0, 0.0], [0.2, 0.8], [0.6, 0.4], [0.5, 0.5]])
    m = UT.task_metrics(P, y, P.argmax(1), 2)
    ll = -np.log(np.array([1.0, 0.8, 0.4, 0.5]))
    assert m["logloss"] == pytest.approx(ll.mean(), abs=0, rel=1e-15)
    br = np.array([0.0, 0.08, 0.72, 0.5])
    assert m["brier"] == pytest.approx(br.mean(), rel=1e-15)
    assert m["acc"] == 0.75 and m["recalls"] == {"0": 1.0, "1": 0.5}
    assert m["balanced_acc"] is None                      # < 30 rows per class: unsupported, never dropped
    assert m["unsupported_classes"] == [0, 1]
    # the clip: a zero true-class probability costs -log(1e-12), not infinity
    z = UT.task_metrics(np.array([[0.0, 1.0]]), np.array([0]), np.array([1]), 2)
    assert z["logloss"] == pytest.approx(-np.log(1e-12))
    # ECE: 10 bins, released decision
    e = UT.task_metrics(np.array([[0.9, 0.1], [0.9, 0.1]]), np.array([0, 1]), np.array([0, 0]), 2)["ece"]
    assert e == pytest.approx(0.4)
    with pytest.raises(ValueError, match="REFUSED"):
        UT.task_metrics(P, y, P.argmax(1), 3)


def test_metrics_match_dpc_on_every_shared_key():
    rng = np.random.default_rng(3)
    n, K = 500, 6
    y = rng.integers(0, K, n)
    P = rng.dirichlet(np.ones(K), n)
    prior = np.bincount(y, minlength=K) / n
    a, b = UT.metrics(P, P.argmax(1), y, K, 2, prior), DU.metrics(P, P.argmax(1), y, K, 2, prior)
    for k in b:
        assert a[k] == b[k], k


def test_sealed_labels_refused_and_constant_reads_defense_fit_only(DT, TU):
    with pytest.raises(PermissionError):
        UT.release_utility(TU, DT, "OSF_DEVELOPMENT_ASSESSMENT", procedure="selection")
    with pytest.raises(PermissionError):
        UT.release_utility(TU, DT, "OSF_DEVELOPMENT_ASSESSMENT", procedure="assessment")   # sealed D
    D2 = {**DT, "y": {k: v.copy() for k, v in DT["y"].items()}}
    c0 = UT.constant_class(DT, "occupation")
    other = np.setdiff1d(np.arange(len(DT["row_id"])), DT["idx"]["OSF_DEFENSE_FIT"])
    other = other[D2["y"]["occupation_group"][other] >= 0]
    D2["y"]["occupation_group"][other] = 5                 # flood every non-fitting row with class 5
    assert UT.constant_class(D2, "occupation") == c0 == 0
    # ties -> lower class
    D3 = {**DT, "y": {k: v.copy() for k, v in DT["y"].items()}}
    f = DT["idx"]["OSF_DEFENSE_FIT"]
    D3["y"]["income"][f] = np.arange(len(f)) % 2
    assert UT.constant_class(D3, "income") == 0


# ------------------------------------------------------------------ release utility and the gate on synthetic units
def test_release_inner_utility_and_identity_receipts(DT, TU):
    probs = {1: TU["p1"], 2: TU["p2"]}
    hard = {1: TU["d1"], 2: TU["d2"]}
    U_u = UT.release_inner_utility(probs, hard, DT)
    assert set(U_u) == {"income", "occupation"} and U_u["income"]["n"] == 2235
    for k in ("acc", "const_acc", "gain", "logloss", "brier", "balanced_acc", "recalls", "ece"):
        assert k in U_u["occupation"]
    z = _class_mean_release(DT, TU)
    code_u = UT.release_inner_utility({1: z["q1"], 2: z["q2"]}, {1: z["hard1"], 2: z["hard2"]}, DT,
                                      u=(probs, hard))
    dp = UT.decision_preservation(z, TU, DT)
    assert dp["ok"] and dp["tasks"]["1"]["mismatches_by_role"]["OSF_DEVELOPMENT_ASSESSMENT"] == 0
    g = UT.gate_record(code_u, U_u, {1: dp["tasks"]["0"]["ok"], 2: dp["tasks"]["1"]["ok"]})
    for t in ("income", "occupation"):
        r = g[t]["receipts"]
        assert r["acc_identity_exact"] and r["acc_minus_u"] == 0.0 and r["gain_minus_u_gain"] == 0.0
        assert g[t]["acc_ok"] and g[t]["retention_ok"]       # identities under class preservation, still reported
        assert code_u[t]["acc_identity_exact"]
    # a 1-state-per-class code loses confidence: its excess is positive and visible
    assert g["occupation"]["ll_excess"] > 0 and g["occupation"]["norm_excess"] > 0
    # U against itself: zero excess, eligible with headroom
    gu = UT.gate_record(U_u, U_u, {1: True, 2: True})
    assert gu["eligible"] and gu["headroom"] and gu["norm_excess"] == 0.0
    with pytest.raises(PermissionError):
        UT.release_inner_utility(probs, hard, DT, rows="AUDIT_FIT")


def test_gate_boundaries_are_inclusive_and_every_gate_binds():
    U = _pair(_m(0.84, 0.34, 0.22, 0.76), _m(0.475, 1.27, 0.65, 0.28))
    ok = {1: True, 2: True}
    edge = _pair(_m(0.84 - 0.01, 0.34 + 0.01, 0.22 + 0.005, 0.76), _m(0.475, 1.27, 0.65, 0.28))
    g = UT.gate_record(edge, U, ok)
    assert g["income"]["acc_ok"] and g["income"]["ll_ok"] and g["income"]["brier_ok"]
    # each allowance exceeded by 1e-9 fails exactly that gate
    for key, bump, flag in (("logloss", 1e-9, "ll_ok"), ("brier", 1e-9, "brier_ok")):
        c = _pair(dict(edge["income"]), edge["occupation"])
        c["income"][key] += bump
        r = UT.gate_record(c, U, ok)
        assert not r["income"][flag] and not r["eligible"]
    # gain < 0.03 fails G_gain even with retention ok when U's own gain is small
    Us = _pair(_m(0.80, 0.4, 0.3, 0.775), U["occupation"])
    c = _pair(_m(0.80, 0.4, 0.3, 0.775), U["occupation"])
    r = UT.gate_record(c, Us, ok)
    assert r["income"]["retention_ok"] and not r["income"]["gain_ok"] and not r["eligible"]
    # retention boundary: gain = 0.8 gain(U) passes, below fails
    Ur = _pair(_m(0.86, 0.34, 0.22, 0.76), U["occupation"])
    c = _pair(_m(0.76 + 0.08, 0.34, 0.22, 0.76), U["occupation"])
    assert UT.gate_record(c, Ur, ok)["income"]["margins"]["G_ret"] == pytest.approx(0.0, abs=1e-15)
    c = _pair(_m(0.835, 0.34, 0.22, 0.76), U["occupation"])
    assert not UT.gate_record(c, Ur, ok)["income"]["retention_ok"]
    # decision preservation: all numeric gates pass but a single changed decision fails the release
    r = UT.gate_record(U, U, {1: True, 2: False})
    assert r["income"]["eligible"] and not r["occupation"]["eligible"] and not r["eligible"]
    assert r["occupation"]["ll_ok"] and not r["occupation"]["preserved"]
    with pytest.raises(ValueError, match="REFUSED"):
        UT.gate_record(U, U, {1: True})
    with pytest.raises(ValueError, match="different rows"):
        UT.gate_record(_pair(_m(0.84, 0.34, 0.22, 0.70), U["occupation"]), U, ok)


def test_headroom_flags_are_tighter_preferences_not_the_gate():
    U = _pair(_m(0.84, 0.34, 0.22, 0.76), _m(0.475, 1.27, 0.65, 0.28))
    ok = {1: True, 2: True}
    mid = _pair(_m(0.84, 0.34, 0.22, 0.76), _m(0.475, 1.27 + 0.009, 0.65 + 0.003, 0.28))
    r = UT.gate_record(mid, U, ok)
    assert r["eligible"] and not r["headroom"]
    assert not r["occupation"]["headroom_ll"] and r["occupation"]["headroom_brier"]
    assert r["occupation"]["norm_excess"] == pytest.approx(0.9)
    tight = _pair(_m(0.84, 0.34, 0.22, 0.76), _m(0.475, 1.27 + 0.0075, 0.65 + 0.0035, 0.28))
    r = UT.gate_record(tight, U, ok)
    assert r["eligible"] and r["headroom"]                 # inclusive at the headroom edge
    br = _pair(_m(0.84, 0.34, 0.22, 0.76), _m(0.475, 1.27, 0.65 + 0.004, 0.28))
    r = UT.gate_record(br, U, ok)
    assert r["eligible"] and not r["occupation"]["headroom_brier"]
    assert r["norm_excess"] == pytest.approx(0.8)


def test_all_seed_gate_never_averages_and_normalised_excess_is_the_worst_cell():
    U = {"0": _m(0.84, 0.34, 0.22, 0.76), "1": _m(0.475, 1.27, 0.65, 0.28)}
    good = {"0": _m(0.84, 0.341, 0.2205, 0.76), "1": _m(0.475, 1.272, 0.651, 0.28)}
    bad = {"0": _m(0.84, 0.34, 0.22, 0.76), "1": _m(0.475, 1.2801, 0.65, 0.28)}      # occupation ll +0.0101
    dec = {0: True, 1: True, 2: True}
    agg = UT.gate_all_seeds({0: good, 1: good, 2: bad}, {0: U, 1: U, 2: U}, dec)
    # the mean occupation excess over seeds is ~0.0047 (< 0.01) but one seed fails -> ineligible
    assert not agg["eligible"] and agg["per_seed"]["0"]["pass"] and not agg["per_seed"]["2"]["pass"]
    assert agg["normalised_excess"] == pytest.approx(1.01)
    assert agg["worst_seed_logloss"]["1"] == pytest.approx(1.2801)
    assert UT.gate_all_seeds({0: good, 1: good}, {0: U, 1: U, 2: U}, dec)["missing_seeds"] == ["2"]
    assert not UT.gate_all_seeds({0: good, 1: good}, {0: U, 1: U, 2: U}, dec)["eligible"]
    ok = UT.gate_all_seeds({0: good, 1: good, 2: good}, {0: U, 1: U, 2: U}, dec)
    assert ok["eligible"] and ok["headroom_all"]
    assert ok["normalised_excess"] == pytest.approx(max(0.001 / 0.01, 0.0005 / 0.005, 0.002 / 0.01, 0.001 / 0.005))
    no_dec = UT.gate_all_seeds({0: good, 1: good, 2: good}, {0: U, 1: U, 2: U}, {0: True, 1: True, 2: False})
    assert not no_dec["eligible"] and not no_dec["decision_preserved_all"]


def test_decision_preservation_detects_one_flip_on_any_role(DT, TU):
    z = _class_mean_release(DT, TU)
    a = DT["idx"]["OSF_DEVELOPMENT_ASSESSMENT"][5]
    z2 = {k: np.array(v) for k, v in z.items()}
    z2["hard2"][a] = (z2["hard2"][a] + 1) % 6
    dp = UT.decision_preservation(z2, TU, DT)
    assert not dp["ok"] and dp["tasks"]["0"]["ok"]
    assert dp["tasks"]["1"]["mismatches_by_role"]["OSF_DEVELOPMENT_ASSESSMENT"] == 1
    assert dp["tasks"]["1"]["mismatches_by_role"]["INNER_SELECTION"] == 0
    bad = dict(z, row_id=z["row_id"][::-1])
    with pytest.raises(ValueError, match="aligned"):
        UT.decision_preservation(bad, TU, DT)


# ====================================================================== audit (qpc.audit) and baselines
import json  # noqa: E402

from dpc import audit as DA  # noqa: E402
from jcv import audit as JA  # noqa: E402
from jcv.finalize import save_unit  # noqa: E402
from qpc import audit as A  # noqa: E402
from qpc import baselines as BL  # noqa: E402
from smf import audit as SA  # noqa: E402

SMALL = dict(n_fit=3000, n_head=10, n_audit=1500, n_sel=800, n_assess=1200)


@pytest.fixture(autouse=True)
def tiny_slate(monkeypatch):
    """A small slate for synthetic tests (the pinned FINAL slate is timed separately; dpc.audit is not edited)."""
    monkeypatch.setitem(DA.SLATES, "tiny", lambda: [("LR_C1", lambda s: JA._lr(1.0, s)),
                                                    ("MLP_16", lambda s: JA._mlp((16,), s)),
                                                    ("DA_LR", SA.da_lr)])


@pytest.fixture(scope="module")
def DS():
    return A.synthetic_D(seed=1, **SMALL)


@pytest.fixture(scope="module")
def TS(DS):
    return A.synthetic_teacher(DS, seed=1)


def _renumber_q(z, seed):
    rng = np.random.default_rng(seed)
    zp = dict(z)
    for i in (1, 2):
        perm = rng.permutation(int(z[f"alpha{i}"]))
        zp[f"tok{i}"] = perm[np.asarray(z[f"tok{i}"])]
    return zp


def _sexbit_release(D, t, i=2):
    """Collision split of recipient i by a noisy copy of the SEX label (decoded vectors unchanged)."""
    z = A.synthetic_release(D, t, 2, 4)
    S = np.asarray(D["sex"])
    noisy = SA.noisy_sex(np.where(S < 0, 0, S), 5)
    return DA.split_tokens(z, i, noisy, collide=True)


def test_release_key_contract_refuses_extra_and_missing(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4)
    A.check_release_keys(z)
    with pytest.raises(ValueError, match="extra \\['p1'\\]"):
        A.policy_views({**z, "p1": TS["p1"]}, DS)              # a continuous probability may never ride along
    with pytest.raises(ValueError, match="extra \\['fine2'\\]"):
        A.policy_views({**z, "fine2": z["tok2"]}, DS)
    with pytest.raises(ValueError, match="missing \\['q2'\\]"):
        A.policy_views({k: v for k, v in z.items() if k != "q2"}, DS)
    V = A.policy_views(z, DS)
    a2 = int(z["alpha2"])
    assert V["X"]["v2"].shape[1] == a2 + 6 + 6                   # one-hot over the FULL alphabet, q, one-hot decision
    assert set(np.unique(V["X"]["v2"][:, :a2])) == {0.0, 1.0}   # categorical, not an ordinal id column


def test_renumbering_leaves_every_prediction_unchanged(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.3)
    r0, a0 = A.inner_family(A.policy_views(z, DS), DS, slate="tiny")
    r1, a1 = A.inner_family(A.policy_views(_renumber_q(z, 3), DS), DS, slate="tiny")
    assert np.array_equal(r0["inner_predictions"]["P"], r1["inner_predictions"]["P"])
    assert r0["auc"] == r1["auc"] and r0["selected"] == r1["selected"]
    for k in a0:
        assert np.array_equal(a0[k], a1[k])


def test_two_ids_sharing_one_decoded_vector_stay_distinct(DS, TS):
    zc = _sexbit_release(DS, TS, i=2)
    assert DA.token_decoder_check(zc["tok2"], zc["q2"], zc["hard2"])["ok"]
    r, _ = A.inner_family(A.policy_views(zc, DS), DS, slate="tiny")
    assert r["auc"]["v2"] > 0.8                                    # token identity carries the bit
    qv = {"family": "q_only", "tokens": None, "X": {"v1": zc["q1"], "v2": zc["q2"],
                                                    "pair": np.hstack([zc["q1"], zc["q2"]])}}
    rq, _ = A.inner_family(qv, DS, slate="tiny")
    assert rq["auc"]["v2"] < 0.6                                   # decoded probabilities alone cannot see it


def test_unseen_local_token_uses_fit_prior_and_unseen_pair_rule_is_inner_selected():
    # exact fixture: AUDIT_FIT rows 0..7, INNER rows 8..15
    D = {"idx": {"AUDIT_FIT": np.arange(8), "INNER_SELECTION": np.arange(8, 16)}, "row_id": np.arange(16),
         "sex": np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1])}
    t1 = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 7, 1, 0, 0])   # token 7 unseen in AUDIT_FIT
    t2 = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 5, 5])   # tuples (0,5) etc unseen
    fit, sel = D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"]
    prior = D["sex"][fit].mean()
    out, info = DA.cc_local(t1, D["sex"], fit, sel)
    assert out["CC_alpha1.0"][12 - 8] == pytest.approx(prior)          # unseen local -> AUDIT_FIT SEX prior
    n0 = (t1[fit] == 0).sum()
    n01 = ((t1[fit] == 0) & (D["sex"][fit] == 1)).sum()
    assert out["CC_alpha1.0"][0] == pytest.approx((n01 + prior) / (n0 + 1))
    Pp, ip = DA.cc_pair(t1, t2, D["sex"], fit, sel, np.arange(8))
    seen = ip["seen"]
    keys = DA._pair_keys(t1, t2)
    assert np.array_equal(seen, np.isin(keys[sel], keys[fit]))
    for a, rule in ip["fallback_rule"].items():
        assert rule in DA.FALLBACK_RULES
        ces = ip["fallback_ce_on_unseen_inner_rows"][a]
        assert rule == min(DA.FALLBACK_RULES, key=lambda k: (ces[k], DA.FALLBACK_RULES.index(k)))
    # coverage receipt counts the same fallbacks
    Dc = {**D, "idx": {**D["idx"], "OSF_DEFENSE_FIT": np.arange(0), "OSF_DEVELOPMENT_ASSESSMENT": np.arange(0)}}
    cov = A.coverage_receipt(t1, t2, Dc, 8, 6)
    assert cov["v1"]["fallback_use"]["INNER_SELECTION"]["fallback_rows"] == 1
    assert cov["v1"]["fallback_use"]["AUDIT_FIT"]["fallback_rows"] == 0
    assert cov["pair"]["fallback_use"]["INNER_SELECTION"]["fallback_rows"] == int((~seen).sum())
    assert cov["v1"]["audit_fit_count_histogram"] == {"3": 1, "5": 1}
    assert cov["pair"]["declared_tuple_space"] == 48


def test_auc_orientation_fixed_never_flipped(DS):
    S = np.asarray(DS["sex"])
    fit, sel = DS["idx"]["AUDIT_FIT"], DS["idx"]["INNER_SELECTION"]
    tok = np.zeros(len(S), dtype=np.int64)
    tok[fit] = S[fit]                     # fit: token = SEX
    tok[sel] = 1 - S[sel]                 # inner: token = 1 - SEX (reversed relation)
    q = np.where(tok[:, None] == 1, [0.3, 0.7], [0.7, 0.3])
    z = {"row_id": DS["row_id"], "tok1": tok, "q1": q, "hard1": q.argmax(1), "alpha1": np.asarray(2),
         "tok2": np.zeros(len(S), dtype=np.int64), "q2": np.tile(np.eye(6)[0] * 0.4 + 0.1, (len(S), 1)),
         "hard2": np.zeros(len(S), dtype=np.int64), "alpha2": np.asarray(1)}
    r, _ = A.inner_family(A.policy_views(z, DS), DS, slate="tiny")
    assert r["auc"]["v1"] < 0.05 and r["auc_seed0"]["v1"] < 0.05      # far below chance, reported as is


def test_auc_and_ce_selection_are_separate():
    y = np.array([0, 1] * 50)
    rank_perfect_uncal = np.where(y == 1, 0.501, 0.499)          # AUC 1, CE ~ log 2
    calibrated = np.where(y == 1, 0.9, 0.1)
    calibrated[:10] = 0.95                                       # a few ranking errors on y = 0 rows
    preds = {"v1:A": rank_perfect_uncal, "v1:B": calibrated, "v2:A": calibrated, "v2:B": calibrated,
             "pair:A": calibrated, "pair:B": calibrated}
    sel, _ = DA._select(preds, [""], y, np.arange(len(y)))
    assert sel["v1"]["auc"]["attacker"] == "A" and sel["v1"]["ce"]["attacker"] == "B"


def test_selected_attackers_refit_at_seeds_and_mean_is_reported(DS, TS):
    t = dict(TS)
    S = np.asarray(DS["sex"])
    rng = np.random.default_rng(0)
    t["c1"] = t["c1"] + 0.3 * np.where(S < 0, 0, S)[:, None] * np.array([1, -1]) + rng.normal(0, 0.01, t["c1"].shape)
    V = BL.source_view_sets(t, DS, families=("interface",))["interface"]
    r, arr = A.inner_family(V, DS, slate="tiny")
    for w in DA.PRIMARY_VIEWS:
        assert r["auc"][w] == pytest.approx(np.mean(r["auc_per_seed"][w]))
        assert r["auc_per_seed"][w][0] == pytest.approx(r["auc_seed0"][w])
        assert arr[f"auc_{w}"].shape == (3, len(DS["idx"]["INNER_SELECTION"]))
        assert r["ce"][w] == pytest.approx(np.mean(r["ce_per_seed"][w]))
    assert r["attacker_seeds"] == [0, 1, 2]


def test_mi_diagnostic_exact_and_fixed_permutations(DS):
    fit = DS["idx"]["OSF_DEFENSE_FIT"]
    S = np.asarray(DS["sex"])
    p = S[fit].mean()
    H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    t_full = np.where(S < 0, 0, S)
    t_const = np.zeros(len(S), dtype=np.int64)
    m = A.mi_diagnostic(t_full, t_const, DS, perms=10)
    assert m["views"]["v1"]["fitted_mi"] == pytest.approx(H, rel=1e-12)
    assert m["views"]["v2"]["fitted_mi"] == pytest.approx(0.0, abs=1e-15)
    assert m["views"]["pair"]["fitted_mi"] == pytest.approx(H, rel=1e-12)
    assert m["views"]["v1"]["null_max"] < 0.01 and m["views"]["v1"]["z"] > 10
    m2 = A.mi_diagnostic(t_full, t_const, DS, perms=10)
    assert m2["perm_list_sha256"] == m["perm_list_sha256"] and m2["views"] == m["views"]
    # sparse alphabet: plug-in MI of an independent many-state token is mostly permutation-null bias
    rng = np.random.default_rng(4)
    t_many = rng.integers(0, 2000, len(S))
    mm = A.mi_diagnostic(t_many, t_const, DS, perms=10)["views"]["v1"]
    assert mm["fitted_mi"] > 0.02 and abs(mm["excess_over_null_mean"]) < 0.25 * mm["fitted_mi"]


def test_reference_views_cells_and_value_tokens(DS, TS):
    zr = A.synthetic_reference_cells(DS, TS, ncell=(5, 7))
    sets, out, _ = BL.reference_view_sets("F", 0, DS, arrays=zr, with_outputs=True)
    assert set(sets) == {"interface", "scores", "probs", "decisions", "cells"}
    assert sets["cells"]["tokens"]["v2"] is not None and sets["interface"]["tokens"] is not None
    assert np.array_equal(out["hard2"], zr["d2"])
    se = BL.reference_view_sets("E", 0, DS, arrays=zr)
    assert "complete" in se and se["interface"]["tokens"] is None      # E's views are continuous


# ------------------------------------------------------------------ end-to-end unit contract + composed closure
def _save(units, name, files, record):
    save_unit(units / name, {f: (lambda p, a=a: np.savez_compressed(p, **a)) for f, a in files.items()}, record)


def _save_inner(units, rec):
    files = rec.pop("_files")
    _save(units, f"inner__{rec['unit_of']}", files, rec)


@pytest.fixture()
def store(tmp_path, DS, TS):
    units = tmp_path / "units"
    units.mkdir()
    _save(units, "tea__s0__U", {"teacher.npz": TS}, {"t": "U"})
    _save(units, "tea__s0__RAW-J_b0.3", {"teacher.npz": A.synthetic_teacher(DS, seed=9)}, {"t": "RAW-J"})
    leaky = A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.6)          # synthetic leak (not a function of p)
    plain = A.synthetic_release(DS, TS, 2, 4)
    cls = A.synthetic_release(DS, TS, 1, 1)
    _save(units, "pol__s0__U_DIRECT-TASK_i2o4", {"release.npz": plain}, {})
    _save(units, "pol__s0__U_JOINT_i2o4_l0.1", {"release.npz": leaky}, {})
    _save(units, "pol__s0__U_CLASS_i1o1", {"release.npz": cls}, {})
    _save(units, "ref__s0__F", {"reference.npz": A.synthetic_reference_cells(DS, TS, ncell=(5, 7))}, {})
    return units


POLS = ["U|DIRECT-TASK|i2o4", "U|JOINT|i2o4|l0.1", "U|CLASS|i1o1"]


def test_inner_unit_contract_and_composed_source_bank(store, DS):
    recs = {}
    for c in POLS:
        r = A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny")
        recs[c] = json.loads(json.dumps({k: v for k, v in r.items() if k != "_files"}))
        _save_inner(store, r)
    p = recs["U|JOINT|i2o4|l0.1"]
    for key in ("recovery", "utility", "preserved", "token_states", "coverage_receipt", "mi_diagnostic"):
        assert key in p
    assert set(p["recovery"]["auc"]) == {"v1", "v2", "pair"} and "ce" in p["recovery"]
    assert set(p["utility"]) == {"income", "occupation"} and p["preserved"] == {"1": True, "2": True}
    assert p["token_states"] == 4 + 21                           # 2 x 2 income + 5 x 4 occupation + 1 reserved
    assert p["utility_gate"]["income"]["preserved"]
    assert p["recovery"]["auc"]["v2"] > recs["U|DIRECT-TASK|i2o4"]["recovery"]["auc"]["v2"]
    # the source bank composes with every code BEFORE selection; the leaky code's reader wins
    src = A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)
    c = src["composed"]
    assert c["winner"]["v2"] == "U|JOINT|i2o4|l0.1"
    assert src["recovery"]["auc"]["v2"] == pytest.approx(p["recovery"]["auc"]["v2"])
    assert src["recovery"]["own"]["auc"]["v2"] < src["recovery"]["auc"]["v2"]
    assert "U|JOINT|i2o4|l0.1" in c["freeze"] and c["closure"]["ok"]
    assert src["token_states"] is None and src["preserved"] == {"1": True, "2": True}
    dec = src["families"]["decisions"]["composed"]
    assert dec["policies"] == ["U|CLASS|i1o1"]                   # decisions compose with the class-only code only
    assert src["utility"]["income"]["acc"] == pytest.approx(UT.release_inner_utility(
        {1: BL.load_teacher("U", 0, DS, store)[0]["p1"], 2: BL.load_teacher("U", 0, DS, store)[0]["p2"]},
        {1: BL.load_teacher("U", 0, DS, store)[0]["d1"], 2: BL.load_teacher("U", 0, DS, store)[0]["d2"]},
        DS)["income"]["acc"])
    _save_inner(store, src)
    assert A.composed_freeze_list(0, "U", store) == c["freeze"]
    # RAW-J has no codes: its composed bank is its own bank
    rj = A.inner_unit("source", 0, "SRC|RAW-J_b0.3", DS, units_dir=store, slate="tiny", policy_cids=[])
    assert rj["composed"]["freeze"] == [] and rj["recovery"]["auc"] == rj["recovery"]["own"]["auc"]
    ref = A.inner_unit("reference", 0, "REF|F", DS, units_dir=store, slate="tiny")
    assert ref["preserved"] == {"1": True, "2": True} and "cells" in ref["families"]
    assert ref["coverage_receipt"]["v2"]["occupied"]["AUDIT_FIT"] == 7


def test_composed_bank_refuses_without_closure(store, DS):
    for c in POLS[:2]:
        r = A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny")
        _save_inner(store, r)
    with pytest.raises(SystemExit, match="not closed"):           # the CLASS code's inner unit is missing
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)
    with pytest.raises(SystemExit, match="unexpected_release_units"):   # a fitted code left out of the list
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS[:2])


def test_policy_preservation_failure_is_recorded(store, DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4)
    z["hard2"] = z["hard2"].copy()
    j = int(np.flatnonzero(z["hard2"] == 0)[0])
    z["q2"] = z["q2"].copy()
    z["q2"][j] = np.eye(6)[1] * 0.88 + 0.02                         # decodes consistently, but changes one decision
    z["hard2"][j] = 1
    z["tok2"] = z["tok2"].copy()
    z["tok2"][j] = int(z["alpha2"])
    z["alpha2"] = np.asarray(int(z["alpha2"]) + 1)
    _save(store, "pol__s0__U_FINE-TASK_i2o4", {"release.npz": z}, {})
    r = A.inner_unit("policy", 0, "U|FINE-TASK|i2o4", DS, units_dir=store, slate="tiny")
    assert r["preserved"] == {"1": True, "2": False} and not r["utility_gate"]["eligible"]


# ------------------------------------------------------------------ controls with explicit pass rules
def test_controls_detect_every_plant_and_pass_the_null(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4)
    r = A.controls_for_release("code", A.policy_views(z, DS), DS, z_policy=z, slate="tiny")
    ch = r["checks"]
    assert ch["NULL"]["ok"], ch["NULL"]
    for p in ("CONF_r1", "CONF_r2", "COLL_r1", "COLL_r2", "XOR"):
        assert ch[p]["ok"], (p, ch[p])
    assert ch["CONF_r2"]["decisions_only_misses_it"]               # a decision-only audit cannot see the plant
    assert ch["COLL_r2"]["decoded_probability_only_misses_it"]      # nor can a decoded-probability-only audit
    assert ch["XOR"]["locals_null_ok"]
    assert r["all_ok"]
    V = BL.source_view_sets(TS, DS, families=("interface",))["interface"]
    rs = A.controls_for_release("src", V, DS, rotate=True, slate="tiny")
    assert rs["checks"]["ROT_r1"]["ok"] and rs["checks"]["ROT_r2"]["ok"], rs["checks"]
    tab = rs["checks"]["ROT_r2"]["per_attacker_auc_B"]["v2"]
    assert tab["own:v2:DA_LR"] > 0.75                              # the scale-aware reader sees the 1e-6 clue
    assert rs["checks"]["ROT_r2"]["serialisation_exact"]


def test_control_failure_is_reported_not_dropped(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4)
    rng = np.random.default_rng(0)
    S = np.asarray(DS["sex"])
    halves = SA.null_split(DS)
    Sp, _ = SA.frozen_permutation(S, DS, halves)
    z2 = DA.split_tokens(z, 1, SA.noisy_sex(Sp, A.CONTROL_SEED), collide=True)   # a release carrying S* itself
    del rng
    r = A.controls_for_release("leaky", A.policy_views(z2, DS), DS, slate="tiny")
    assert not r["checks"]["NULL"]["ok"] and "NULL" in r["failures"] and not r["all_ok"]


def test_stage_controls_shards_merge_and_refuse_unsealed(store, DS, tmp_path):
    plan = {"policies": [(0, "U|DIRECT-TASK|i2o4")], "sources": [], "references": [], "rotate_references": [],
            "null_policy": (0, "U|DIRECT-TASK|i2o4"), "null_reps": 2}
    out = tmp_path / "AUDIT_PRELOCK_CHECKS.json"
    p0 = A.stage_controls(DS, "0/2", units_dir=store, out_path=out, slate="tiny", plan=plan)
    assert p0["status"] == "PARTIAL" and not out.exists()
    v = A.stage_controls(DS, "1/2", units_dir=store, out_path=out, slate="tiny", plan=plan)
    assert out.exists() and v["all_ok"] and v["null_calibration_present"]
    pub = json.loads(out.read_text())
    assert pub["verdict"]["positive_controls"]["U|DIRECT-TASK|i2o4|s0"]["XOR"]
    with pytest.raises(SystemExit, match="sealed"):
        A.stage_controls({**DS, "sealed": False}, None, units_dir=store, out_path=out, slate="tiny", plan=plan)


def test_control_plan_is_structural():
    p = A.control_plan({"stage_b": {"rates": [[8, 16], [4, 32]], "lams": [0.01, 0.1, 1.0]}})
    assert [c for _, c in p["policies"]] == ["U|DIRECT-TASK|i8o64", "U|JOINT|i4o32|l1", "U|CLASS|i1o1"]
    assert A.control_plan({})["policies"] == [(0, "U|DIRECT-TASK|i8o64")]


# ====================================================================== the locked assessment (qpc.assess)
import subprocess  # noqa: E402

from qpc import assess as AS  # noqa: E402


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)


@pytest.fixture()
def repo(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    r = tmp_path / "wt"
    r.mkdir()
    _git(r, "init", "-q", "-b", AS.STUDY_BRANCH)
    _git(r, "config", "user.email", "t@example.org")
    _git(r, "config", "user.name", "t")
    _git(r, "remote", "add", "origin", str(origin))
    (r / "README").write_text("x\n")
    _git(r, "add", "README")
    _git(r, "commit", "-q", "-m", "init")
    _git(r, "push", "-q", "origin", AS.STUDY_BRANCH)
    return r


def _write_lock(repo, body=None):
    p = repo / AS.LOCK_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body or {"seeds": {}, "locked_code_files": {}}))
    return p


def _push(repo, msg="lock"):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    _git(repo, "push", "-q", "origin", AS.STUDY_BRANCH)


def test_gate_refuses_uncommitted_unpushed_modified_and_misnamed_locks(repo):
    p = _write_lock(repo)
    assert AS.lock_is_pushed(p, repo, fetch=False)["reason"] == "lock file is not committed"
    _git(repo, "add", AS.LOCK_REL)
    _git(repo, "commit", "-q", "-m", "lock")
    r = AS.lock_is_pushed(p, repo, fetch=False)
    assert not r["ok"] and "not on origin" in r["reason"]
    _git(repo, "push", "-q", "origin", AS.STUDY_BRANCH)
    assert AS.lock_is_pushed(p, repo)["ok"]
    p.write_text(p.read_text() + " ")
    assert not AS.lock_is_pushed(p, repo)["ok"]
    other = repo / "results" / "EVALUATION_LOCK.json"
    other.write_text("{}")
    assert not AS.lock_is_pushed(other, repo)["ok"]
    wrong = repo / "LOCK.json"
    wrong.write_text("{}")
    assert "must be named" in AS.lock_is_pushed(wrong, repo)["reason"]


def test_open_requires_chain_hashes_loaded_modules_and_outer_requires_open(repo):
    AS.close_assessment()
    with pytest.raises(SystemExit, match="sealed"):
        AS.outer_unit({"sealed": False}, 0, "x", {"kind": "policy", "cid": "U|CLASS|i1o1"})
    with pytest.raises(SystemExit, match="sealed"):
        AS.load_unsealed()
    f = repo / "code.py"
    f.write_text("print(1)\n")
    p = _write_lock(repo, {"seeds": {}, "locked_code_files": {"code.py": AS._sha(f)}})
    _push(repo)
    with pytest.raises(SystemExit, match="does not list"):
        AS.open_assessment(p, repo, chain=("code.py", "other.py"))
    with pytest.raises(SystemExit, match="loaded worktree modules are not locked"):
        AS.open_assessment(p, repo, chain=("code.py",))          # qpc modules are loaded but not in this lock
    lock = json.loads(p.read_text())
    assert AS.verify_code(lock, repo, chain=("code.py",), check_modules=False)["code.py"] == "matches lock"
    f.write_text("print(2)\n")
    with pytest.raises(SystemExit, match="differs from the locked code hash"):
        AS.verify_code(lock, repo, chain=("code.py",), check_modules=False)
    AS.close_assessment()


def test_qpc_data_refuses_unsealing_outside_assess():
    from qpc import data as QD
    with pytest.raises((PermissionError, SystemExit)):
        QD.load(unseal=True)


def test_jobs_are_only_the_locked_list_and_shards_partition_it():
    lock = {"seeds": {str(k): {"score": {f"L{j}": {"kind": "policy", "cid": "U|CLASS|i1o1"} for j in range(5)}}
                      for k in (0, 1, 2)}}
    jobs = AS.jobs_from_lock(lock)
    assert len(jobs) == 15
    sh = [AS.jobs_from_lock(lock, shard=f"{i}/2") for i in (0, 1)]
    assert sorted(map(str, sh[0] + sh[1])) == sorted(map(str, jobs)) and not set(map(str, sh[0])) & set(map(str, sh[1]))
    assert AS.jobs_from_lock(lock, seeds=(1,), labels=["L2"]) == [(1, "L2", lock["seeds"]["1"]["score"]["L2"])]
    assert AS._policy_cid_of("pol__s0__U_JOINT_i8o64_l0.1") == "U|JOINT|i8o64|l0.1"
    assert AS._policy_cid_of("U|CLASS|i1o1") == "U|CLASS|i1o1"


def _unsealed(D, seed=3):
    rng = np.random.default_rng(seed)
    Du = {**D, "y": {k: v.copy() for k, v in D["y"].items()}, "sealed": False}
    a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    Du["y"]["income"][a] = rng.integers(0, 2, len(a))
    Du["y"]["occupation_group"][a] = rng.integers(0, 6, len(a))
    Du["race"] = rng.integers(0, 3, len(D["row_id"]))
    return Du


def test_outer_unit_end_to_end_composed_freeze_and_restore_hook(repo, store, DS, monkeypatch):
    for c in POLS:
        _save_inner(store, A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny"))
    src = A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)
    _save_inner(store, src)
    freeze = src["composed"]["freeze"]
    assert freeze
    score = {"SRC|U": {"kind": "source", "cid": "SRC|U", "unit": "tea__s0__U"},
             "J*": {"kind": "policy", "cid": "U|JOINT|i2o4|l0.1"},
             "REF|F": {"kind": "reference", "cid": "REF|F"}}
    lock = {"seeds": {"0": {"u_label": "SRC|U", "score": score,
                            "composed_policies": {"U": ["U|DIRECT-TASK|i2o4", "U|CLASS|i1o1"]}}},
            "locked_code_files": {}}
    p = _write_lock(repo, lock)
    _push(repo)
    Du = _unsealed(DS)
    monkeypatch.setitem(DA.SLATES, "final", DA.SLATES["tiny"])           # fast synthetic run (pinned slate untouched)
    L = AS.open_assessment(p, repo, check_code=False)
    try:
        with pytest.raises(SystemExit, match="not frozen in the lock"):     # the composed winner must be locked
            AS.outer_unit(Du, 0, "SRC|U", score["SRC|U"], units_dir=store)
        L["seeds"]["0"]["composed_policies"] = ["pol__s1__U_JOINT_i2o4_l0.1"]
        with pytest.raises(SystemExit, match="not units of seed 0"):
            AS.outer_unit(Du, 0, "SRC|U", score["SRC|U"], units_dir=store)
        # qpc.eval_lock schema: one list of every fitted code's unit (each source keeps its own teacher's codes)
        L["seeds"]["0"]["composed_policies"] = [A.unit_of(0, c) for c in POLS]
        na = len(Du["idx"]["OSF_DEVELOPMENT_ASSESSMENT"])
        for k, label, spec in AS.jobs_from_lock(L, (0,)):
            r = AS.outer_unit(Du, k, label, spec, units_dir=store)
            rr, z = AS.load_outer(k, label, store)
            for key in ("assess_row_id", "assess_unit", "sex", "y_income", "y_occ", "hard1", "hard2", "ll1", "br2",
                        "u_hard1", "u_ll2", "const1"):
                assert z[key].shape == (na,), key
            assert z["prob1"].shape == (na, 2) and z["prob2"].shape == (na, 6) and z["const_class"].shape == (2,)
            for w in DA.PRIMARY_VIEWS:
                assert z[f"P_auc_{w}"].shape == (3, na, 2) and z[f"P_ce_{w}"].shape == (3, na, 2)
                assert np.allclose(z[f"P_auc_{w}"].sum(2), 1.0)
            assert np.array_equal(z["y_income"], Du["y"]["income"][Du["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]])
            assert set(r["utility"]) == {"0", "1"} and r["utility"]["1"]["u_acc"] is not None
            unit = {"policy": "pol__s0__U_JOINT_i2o4_l0.1", "source": "tea__s0__U", "reference": "ref__s0__F"}[
                spec["kind"]]
            if spec["kind"] == "source":
                assert r["composed"]["class_only_units"] == ["pol__s0__U_CLASS_i1o1"]
                assert "P_auc_decisions_pair" in z
                np.testing.assert_array_equal(z["prob2"], z["u_prob2"])
                assert r["families"]["interface"]["scored"]["v2"]["auc"]["candidate"].startswith("composed[")
            if spec["kind"] == "policy":
                assert r["decision_preservation"]["ok"]
            # restore hook: refit from the stored units on a SEALED D reproduces the saved predictions bitwise
            for w in ("pair", "v2"):
                P = AS.refit_selected_attacker(store, DS, unit, r["unit"], view=w)
                assert np.array_equal(P, z[f"P_auc_{w}"]), (label, w)
            assert np.array_equal(AS.refit_selected_attacker(store, DS, unit, r["unit"], attacker_seed=1),
                                  z["P_auc_pair"][1])
    finally:
        AS.close_assessment()


def test_lazy_code_views_are_bit_identical_in_the_final_audit(DS, TS):
    z = A.synthetic_release(DS, TS, 2, 4, sex_tilt=0.3)
    V, L = A.policy_views(z, DS), A.lazy_policy_views(z, DS)
    for w in DA.PRIMARY_VIEWS:
        assert np.array_equal(V["X"][w], L["X"][w])
    Du = _unsealed(DS)
    a = Du["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    Vi = BL.source_view_sets(TS, DS, families=("interface",))["interface"]
    r1, P1 = DA.final_audit(Vi, Du, a, composed=[("c", V)], slate="tiny")
    r2, P2 = DA.final_audit(Vi, Du, a, composed=[("c", L)], slate="tiny")
    assert r1["selected"] == r2["selected"]
    for k in P1:
        assert np.array_equal(P1[k], P2[k])
    with pytest.raises(ValueError, match="extra"):
        A.lazy_policy_views({**z, "p1": TS["p1"]}, DS)


def test_composed_bank_refuses_records_from_other_rows_or_slates(store, DS):
    for c in POLS:
        r = A.inner_unit("policy", 0, c, DS, units_dir=store, slate="tiny")
        if c == "U|CLASS|i1o1":
            r["recovery"]["sel_row_id_sha256"] = "0" * 64              # scored on different INNER rows
        _save_inner(store, r)
    with pytest.raises(SystemExit, match="differs from the source on sel_row_id_sha256"):
        A.inner_unit("source", 0, "SRC|U", DS, units_dir=store, slate="tiny", policy_cids=POLS)
