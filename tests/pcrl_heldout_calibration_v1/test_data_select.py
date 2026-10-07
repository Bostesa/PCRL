"""Synthetic tests of the hcal role split, label allowlist, selection rules and primary family (role A). No real data."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from hcal import data as HD
from hcal import family as FAM
from hcal import ids as I
from hcal import select as SEL

ROLES = ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION", "OSF_DEVELOPMENT_ASSESSMENT")


def synth_D(seed=0, sizes=(3000, 200, 2600, 400, 900), dup=0.05):
    """Rows grouped into exact-record groups (some duplicated rows); every group lies in one role."""
    rng = np.random.default_rng(seed)
    role, unit, rid = [], [], []
    g = 0
    r = 0
    for name, n in zip(ROLES, sizes):
        made = 0
        while made < n:
            k = 2 if rng.random() < dup and made + 2 <= n else 1
            for _ in range(k):
                role.append(name)
                unit.append(g)
                rid.append(r)
                r += 1
            g += 1
            made += k
    perm = rng.permutation(len(rid))
    role, unit, rid = np.asarray(role)[perm], np.asarray(unit)[perm], np.asarray(rid)[perm] * 7 + 3
    D = {"role": role, "unit": unit.astype(np.int64), "row_id": rid.astype(np.int64), "sealed": True,
         "idx": {nm: np.flatnonzero(role == nm) for nm in ROLES}}
    n = len(rid)
    D["sex"] = rng.integers(0, 2, n)
    D["y_income"] = rng.integers(0, 2, n)
    D["y_occupation_group"] = rng.integers(0, 6, n)
    a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    for kk in ("sex", "y_income", "y_occupation_group"):
        D[kk][a] = -1
    return D


def test_rank_is_sha256_of_salted_decimal_id():
    order, keys = HD.rank_groups([5, 12, 7], "salt|")
    exp = sorted((hashlib.sha256(f"salt|{u}".encode()).hexdigest(), u) for u in (5, 12, 7))
    assert order == [u for _, u in exp] and keys == [h for h, _ in exp]


def test_split_counts_purity_representatives_and_determinism():
    D = synth_D()
    HD.N_CAL, n0 = 500, HD.N_CAL
    try:
        roles, meta = HD.build_roles(D)
        roles2, _ = HD.build_roles(D)
    finally:
        HD.N_CAL = n0
    for k in roles:
        assert np.array_equal(roles[k], roles2[k])               # deterministic
    cal = roles["CALIBRATION_HELDOUT"]
    assert cal.size == 500 and np.unique(D["unit"][cal]).size == 500
    # representative = smallest row id of its group, every member of a selected group is in the group rows
    for u in np.unique(D["unit"][cal]):
        members = np.flatnonzero(D["unit"] == u)
        rep = members[np.argmin(D["row_id"][members])]
        assert rep in cal and set(members) <= set(roles["CALIBRATION_HELDOUT_GROUP_ROWS"].tolist())
    a = D["idx"]["AUDIT_FIT"]
    assert np.array_equal(np.sort(np.concatenate([roles["CALIBRATION_HELDOUT_GROUP_ROWS"], roles["ATTACK_FIT_NEW"]])),
                          np.sort(a))
    assert not np.intersect1d(D["unit"][roles["CALIBRATION_HELDOUT_GROUP_ROWS"]],
                              D["unit"][roles["ATTACK_FIT_NEW"]]).size
    assert set(roles["CALIBRATION_TRAIN_MATCHED"].tolist()) <= set(D["idx"]["OSF_DEFENSE_FIT"].tolist())


def test_split_ignores_labels():
    D = synth_D()
    D2 = {**D, "sex": D["sex"][::-1].copy(), "y_income": 1 - np.abs(D["y_income"])}
    r1, _ = HD.build_roles(D)
    r2, _ = HD.build_roles(D2)
    assert all(np.array_equal(r1[k], r2[k]) for k in r1)


def test_receipt_disjointness_ok():
    D = synth_D()
    HD.N_CAL, n0 = 500, HD.N_CAL
    try:
        roles, meta = HD.build_roles(D)
        rec = HD.role_receipt(D, roles, meta)
    finally:
        HD.N_CAL = n0
    assert rec["ok"] and all(v["shared_rows"] == 0 for v in rec["disjointness"].values())


def test_allowlist_refusals():
    D = synth_D()
    HD.N_CAL, n0 = 500, HD.N_CAL
    try:
        roles, meta = HD.build_roles(D)
    finally:
        HD.N_CAL = n0
    D["hcal"] = {"roles": roles}
    rows, y = HD.task_labels(D, "income", "calibration", "CALIBRATION_HELDOUT")
    assert np.array_equal(rows, roles["CALIBRATION_HELDOUT"]) and (y >= 0).all()
    with pytest.raises(PermissionError):
        HD.labels_for(D, "calibration", "HEAD_VALIDATION")
    with pytest.raises(PermissionError):
        HD.labels_for(D, "attack", "HEAD_VALIDATION")
    with pytest.raises(PermissionError):
        HD.labels_for(D, "calibration", "INNER_SELECTION")
    with pytest.raises(PermissionError):
        HD.labels_for(D, "attack", "OSF_DEVELOPMENT_ASSESSMENT")
    with pytest.raises(PermissionError):
        HD.labels_for(D, "assessment", "OSF_DEVELOPMENT_ASSESSMENT")      # sealed
    with pytest.raises(PermissionError):
        HD.labels_for(D, "unknown", "AUDIT_FIT")
    r, s = HD.sex_labels(D, "attack", "ATTACK_FIT_NEW")
    assert np.array_equal(r, roles["ATTACK_FIT_NEW"])


def test_unseal_gate_refuses_other_callers():
    with pytest.raises(PermissionError):
        HD.unseal_gate("lra.assess")
    with pytest.raises(PermissionError):
        HD.unseal_gate("hcal.select")


# ------------------------------------------------------------------ ids
def test_bank_counts():
    assert len(I.partitions()) == 57 and len(I.legacy_partitions()) == 27 and len(I.lra_partitions()) == 30
    assert len(I.original_ids()) == 84 and len(set(I.original_ids())) == 84
    rids = I.code_release_ids()
    assert len(rids) == 27 * 5 + 30 * 5 + 4 and len(set(rids)) == len(rids)
    for r in rids:
        p, d = I.parse_release(r)
        assert I.release_id(p, d) == r
    assert sum(I.privacy_trained(p) for p in I.partitions()) == 53
    assert not any(I.nominee_capable(I.release_id(p, "MEAN")) for p in I.lra_partitions())
    assert not any(I.nominee_capable(r) for r in rids if r.endswith("|T-TOKEN32"))
    assert not any(I.task_only_candidate(r) for r in rids if r.endswith("|T-TOKEN32"))
    assert all(I.task_only_candidate(r) for r in I.u_release_ids())


# ------------------------------------------------------------------ selection
def _m(ll, br, acc=0.84, const=0.75):
    return {"acc": acc, "logloss": ll, "brier": br, "const_acc": const, "gain": acc - const, "n": 100,
            "balanced_acc": 0.8, "recalls": {}, "class_counts": {}, "supported_classes": [], "unsupported_classes": [],
            "ece": 0.0, "reliability": []}


def _u(ll1, ll2, br1=0.2, br2=0.6, pres=True, acc2=0.47, const2=0.3):
    return {"inner": {"income": _m(ll1, br1), "occupation": _m(ll2, br2, acc2, const2)},
            "preserved": {"1": pres, "2": pres}, "finite": True}


def test_ucal_star_rule_and_ties():
    util = {k: {I.U_ID: _u(0.3, 1.3), f"{I.U_ID}|H-GLOBAL-TEMP": _u(0.3, 1.29), f"{I.U_ID}|H-CLASS-TEMP": _u(0.3, 1.28)}
            for k in I.SEEDS}
    assert SEL.ucal_star(util)["family"] == "H-CLASS-TEMP"
    util = {k: {I.U_ID: _u(0.3, 1.3), f"{I.U_ID}|H-GLOBAL-TEMP": _u(0.3, 1.3), f"{I.U_ID}|H-CLASS-TEMP": _u(0.3, 1.3)}
            for k in I.SEEDS}
    assert SEL.ucal_star(util)["family"] == "identity"           # exact tie -> identity
    util[0][f"{I.U_ID}|H-CLASS-TEMP"] = _u(0.3, 1.0, pres=False)  # decision preservation failure -> invalid
    assert SEL.ucal_star(util)["family"] == "identity"


def _fake(n_good_priv=True, task_ok=True, benefit=0.03, guard=0.0):
    util, com = {}, {}
    for k in I.SEEDS:
        util[k] = {}
        for rid in I.code_release_ids() + I.u_release_ids():
            util[k][rid] = _u(0.305, 1.305)
        util[k][I.U_ID] = _u(0.30, 1.30)
        if not task_ok:
            for rid in I.code_release_ids():
                if I.task_only_candidate(rid):
                    util[k][rid] = _u(0.31, 1.40)
        com[k] = {"SRC|U": {"auc": {"v1": 0.7, "v2": 0.8, "pair": 0.86}}}
        for p in I.partitions():
            base = 0.85 if I.task_only(p) else 0.84
            com[k][p] = {"auc": {"v1": 0.70, "v2": 0.80, "pair": base}}
        com[k]["U|JOINT|i8o64|l0.1"] = {"auc": {"v1": 0.70 + guard, "v2": 0.80, "pair": 0.85 - benefit}}
        if not n_good_priv:
            for rid in I.code_release_ids():
                if I.nominee_capable(rid):
                    util[k][rid] = _u(0.31, 1.40)
    return util, com


def test_selection_valid_nominee():
    util, com = _fake()
    sel = SEL.select(util, com, I.partitions())
    assert sel["T*"]["status"] == "NOMINEE" and sel["P*"]["status"] == "NOMINEE"
    assert I.parse_release(sel["P*"]["release"])[0] == "U|JOINT|i8o64|l0.1"
    assert sel["P*"]["ranking"][0]["pair_benefit"] == pytest.approx(0.03)


def test_selection_benefit_and_guard_gates():
    util, com = _fake(benefit=0.019)
    sel = SEL.select(util, com, I.partitions())
    assert sel["P*"]["status"] == "NO_ELIGIBLE_COMPETITIVE_NOMINEE"
    util, com = _fake(guard=0.0051)
    sel = SEL.select(util, com, I.partitions())
    assert sel["P*"]["status"] == "NO_ELIGIBLE_COMPETITIVE_NOMINEE"
    assert any(r["reason"] == "LOCAL_GUARD_FAILURE" for r in sel["P*"]["rejected"])
    util, com = _fake(guard=0.005)                                # inclusive guard
    sel = SEL.select(util, com, I.partitions())
    assert sel["P*"]["status"] == "NOMINEE"


def test_selection_missing_comparator_and_missing_nominee():
    util, com = _fake(task_ok=False)
    for k in I.SEEDS:                                              # U itself fails the Ucal gate? keep U0 identity
        for rid in I.u_release_ids()[1:]:
            util[k][rid] = _u(0.30, 1.20)                           # Ucal* much better -> U identity ineligible
    sel = SEL.select(util, com, I.partitions())
    assert sel["T*"]["status"] == "NO_ELIGIBLE" or I.parse_release(sel["T*"]["release"])[0] is None
    util, com = _fake(n_good_priv=False)
    sel = SEL.select(util, com, I.partitions())
    assert sel["P*"]["status"] == "NO_ELIGIBLE_COMPETITIVE_NOMINEE" and sel["P*"]["release"] is None


def test_audit_plan_rule():
    util, com = _fake(n_good_priv=False)
    table = SEL.utility_table(util, "identity")
    plan = SEL.audit_plan(table)
    assert set(I.TASK_ONLY_PARTITIONS) <= set(plan["audited"]) and "U|JOINT|i8o64|l0.1" in plan["audited"]
    assert plan["partitions"]["U|JOINT|i8o64|l0.06"]["reason"] == "PREDECLARED_UTILITY_INELIGIBLE"


# ------------------------------------------------------------------ family
def test_family_size_z_and_truth_table():
    assert len(FAM.PRIMARY) == 23
    from statistics import NormalDist
    assert FAM.Z_PRIMARY == NormalDist().inv_cdf(1 - 0.05 / 46)
    labels = {r["case"]: r["label"] for r in FAM.truth_table()}
    assert labels["all fifteen pass"] == "CALIBRATED_PRIVATE_RELEASE_DEVELOPMENT_CRITERION_ESTABLISHED"
    assert labels["eleven pass, P12 precision"] == "ORIGINAL_REQUIREMENTS_MET_CALIBRATED_REFERENCE_NOT_ESTABLISHED"
    assert labels["no eligible nominee"] == "NO_ELIGIBLE_COMPETITIVE_NOMINEE"
    assert labels["P07 precision failure"] == "NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED"
    assert labels["technical defect"] == "INCOMPLETE_OR_INVALID"
    assert labels["no eligible comparator, no nominee"] == "INCOMPLETE_OR_INVALID"


def test_clause_and_diag_outcomes():
    assert FAM.clause_outcome("upper<", 0.01, 0.005, 0.001, 0.009) == "PASS"
    assert FAM.clause_outcome("upper<", 0.01, 0.008, 0.004, 0.012) == "NOT_ESTABLISHED_PRECISION"
    assert FAM.clause_outcome("upper<", 0.01, 0.02, 0.011, 0.03) == "MEASURED_VIOLATION"
    assert FAM.clause_outcome("lower>", 0.02, 0.03, 0.021, 0.04) == "PASS"
    assert FAM.clause_outcome("lower>", 0.02, float("nan"), 0, 1) == "INVALID"
    assert FAM.diag_outcome(0.01, 0.002, 0.02) == "SUPPORTS_POSITIVE"
    assert FAM.diag_outcome(-0.01, -0.02, -0.002) == "SUPPORTS_NEGATIVE"
    assert FAM.diag_outcome(0.01, -0.002, 0.02) == "UNRESOLVED"
    assert FAM.diag_summary(["SUPPORTS_POSITIVE", "UNRESOLVED", "SUPPORTS_POSITIVE", "SUPPORTS_POSITIVE"]) == "MIXED"


def test_u_class_temp_identity_rows_pass_through_unnormalised_within_check_probs():
    """MATH_REVIEW finding 2: rows whose class keeps alpha = 1 are P exactly even if their sums are off by < 1e-9."""
    from hcal import calib as C
    rng = np.random.default_rng(3)
    P = rng.dirichlet(np.ones(6), 400)
    P = P * (1 + 5e-11)                                         # row sums off by 5e-11 (inside check_probs' 1e-9)
    d = P.argmax(1)
    tab = {"target": "U", "K": 6, "alpha": None, "alphas": [1.0] * 6}
    q = C.apply_u(P, tab, d)
    assert np.array_equal(q, P)
    tab["alphas"] = [0.8, 1.0, 1.0, 1.0, 1.0, 1.0]
    q = C.apply_u(P, tab, d)
    assert np.array_equal(q[d != 0], P[d != 0]) and np.allclose(q[d == 0].sum(1), 1, atol=1e-12)


# ------------------------------------------------------------------ role E findings E-S3 / E-S4 / E-S5 / E-B1 / E-S6
def test_every_variant_resolves_to_its_partitions_single_common_and_attack_record():
    """E-S3: selection, inference and the evaluation lock map every decoder variant of a partition to ONE common record
    and ONE attack unit (and the three U variants to SRC|U)."""
    from hcal import eval_lock as EL
    from hcal import infer as INF
    com = {k: {p: {"auc": {"v1": 0.6 + i * 1e-3, "v2": 0.7, "pair": 0.8}} for i, p in enumerate(I.partitions())}
           for k in I.SEEDS}
    for k in I.SEEDS:
        com[k]["SRC|U"] = {"auc": {"v1": 0.9, "v2": 0.9, "pair": 0.95}}
    for p in I.partitions():
        rids = [I.release_id(p, d) for d in I.decoders_of(p)]
        recs = [SEL.recovery(com, r) for r in rids]
        assert all(r == recs[0] for r in recs)
        assert {INF.partition_key(r) for r in rids} == {p}
        assert {INF.att_name(0, INF.partition_key(r)) for r in rids} == {INF.att_name(0, p)}
        assert EL.attack_keys(rids) == [p]
    assert EL.attack_keys(I.u_release_ids()) == ["SRC|U"]
    assert all(SEL.recovery(com, r) == SEL.recovery(com, I.U_ID) for r in I.u_release_ids())


def test_no_eligible_task_only_comparator_path():
    """E-S4: every task-only code AND every U variant ineligible -> T* NO_ELIGIBLE and P* NOT_SELECTED_NO_COMPARATOR."""
    util, com = _fake(task_ok=False)
    for k in I.SEEDS:
        util[k][I.U_ID] = _u(0.30, 1.30)                           # U0
        util[k][f"{I.U_ID}|H-GLOBAL-TEMP"] = _u(0.32, 1.20)        # Ucal* (summed 1.52) but income > U0 + 0.01
        util[k][f"{I.U_ID}|H-CLASS-TEMP"] = _u(0.30, 1.30)         # fails the Ucal gate on occupation
    sel = SEL.select(util, com, I.partitions())
    assert sel["ucal_star"]["family"] == "H-GLOBAL-TEMP"
    assert sel["T*"]["status"] == "NO_ELIGIBLE" and sel["T*"]["release"] is None
    assert sel["P*"]["status"] == "NOT_SELECTED_NO_COMPARATOR"


def test_ordering_keys_and_inclusive_boundaries():
    """E-S5: benefit exactly 0.02 qualifies (inclusive); utility tie-break orders equal-AUC variants; Ucal margin form."""
    util, com = _fake(benefit=0.02)
    for k in I.SEEDS:
        com[k]["U|JOINT|i8o64|l0.1"]["auc"]["pair"] = com[k]["U|FINE-TASK|i8o64"]["auc"]["pair"] - 0.02
    sel = SEL.select(util, com, I.partitions())
    tref = SEL.recovery(com, sel["T*"]["release"])["pair"]
    rows = [r for r in sel["P*"]["ranking"]] + [r for r in sel["P*"]["rejected"] if r.get("pair_benefit") is not None]
    jt = [r for r in rows if r["partition"] == "U|JOINT|i8o64|l0.1"]
    assert jt and all(abs(r["pair_benefit"] - SEL.mean3([tref[k] - com[k]["U|JOINT|i8o64|l0.1"]["auc"]["pair"]
                                                          for k in I.SEEDS])) < 1e-15 for r in jt)
    # equal AUC across the variants of the winning partition: the lower worst-seed normalised excess wins
    util, com = _fake()
    best = "U|JOINT|i8o64|l0.1|H-GLOBAL-TEMP"
    for k in I.SEEDS:
        util[k][best] = _u(0.300, 1.300)
    sel = SEL.select(util, com, I.partitions())
    assert sel["P*"]["release"] == best
    # Ucal gate in the qpc margin form: exactly at the allowance passes
    u, ucal = _u(0.31, 1.31), _u(0.30, 1.30)
    g = SEL.gate_seed(u, ucal, ucal)
    m = g["ucal"]["income"]["ll_margin"]
    assert g["ucal"]["income"]["ll_ok"] == (m >= 0)


def test_controls_verdict_feeds_selection_and_fails_safe():
    """E-B1: the controls verdict is read from verdict.all_ok; missing or false -> every role TECHNICAL_FAILURE."""
    util, com = _fake()
    for ctl, ok in (({"verdict": {"all_ok": True}}, True), ({"verdict": {"all_ok": False}}, False), ({}, False),
                    ({"all_ok": True}, False)):
        sel = SEL.apply_controls(SEL.select(util, com, I.partitions()), ctl)
        assert sel["controls_all_ok"] is ok
        if ok:
            assert sel["P*"]["status"] == "NOMINEE"
        else:
            assert sel["P*"]["status"] == sel["T*"]["status"] == "TECHNICAL_FAILURE"
            assert sel["P*"]["status_if_controls_had_passed"] == "NOMINEE"


def test_token32_k6_against_independent_slsqp():
    """E-S6: occupation-sized (K = 6) token solves are not beaten by an independent constrained minimiser of the stated
    objective on the class-dominant simplex."""
    from scipy.optimize import minimize
    from hcal import calib as C
    rng = np.random.default_rng(11)
    K, eps = 6, 1e-12
    for trial in range(6):
        d = int(rng.integers(0, K))
        mu = rng.dirichlet(np.ones(K) * 2.0)
        mu[d] = mu.max() + 0.05
        mu = mu / mu.sum()
        n = int(rng.integers(1, 30))
        y = np.bincount(rng.integers(0, K, n), minlength=K).astype(float)
        _, Q, _, _ = C.token32_solve(y[None], mu[None], np.array([n]), np.array([d]))

        def f(u):
            u = np.clip(u, 0, None)
            q = (u + eps + eps * (np.arange(K) == d)) / (1 + (K + 1) * eps)
            ll = -np.sum(y * np.log(q))
            br = 0.5 * np.sum([np.sum((q - np.eye(K)[c]) ** 2) * y[c] for c in range(K)])
            kl = np.sum(mu * (np.log(mu) - np.log(q)))
            return ll + br + 32.0 * kl
        cons = [{"type": "eq", "fun": lambda u: np.sum(u) - 1.0}] + \
               [{"type": "ineq", "fun": (lambda u, j=j: u[d] - u[j])} for j in range(K) if j != d]
        best = np.inf
        for start in (mu, np.full(K, 1.0 / K), np.eye(K)[d] * 0.5 + 0.5 / K):
            r = minimize(f, start, method="SLSQP", bounds=[(0, 1)] * K, constraints=cons,
                         options={"ftol": 1e-14, "maxiter": 500})
            if r.success:
                best = min(best, r.fun)
        u_star = (Q[0] * (1 + (K + 1) * eps)) - eps - eps * (np.arange(K) == d)
        assert f(u_star) <= best + 1e-8, (trial, f(u_star), best)
