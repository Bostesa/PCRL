"""Synthetic tests of the ccm access guard, label allowlist, frozen-temperature reference transform and the toy-law
construction check (role F). No real data is loaded; no label of any real role is read."""
from __future__ import annotations

import functools
import json
import math
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from ccm import data as F
from ccm import ids as I

FIT, HEAD, HELD, ATT, INNER, AUDIT, ASSESS, TM = (F.FIT, F.HEAD, F.HELD, F.ATT, F.INNER, F.AUDIT, F.ASSESS,
                                                 F.TRAIN_MATCHED)
SRC_ROLES = (FIT, HEAD, AUDIT, INNER, ASSESS)


def synth_D(seed=0, sizes=(300, 40, 120, 50, 160), n_cols=5, n_held=30):
    """A sealed pinned-loader-shaped D: exact-record groups (some duplicated), five source roles, hcal subroles."""
    rng = np.random.default_rng(seed)
    role, unit = [], []
    g = 0
    for name, n in zip(SRC_ROLES, sizes):
        made = 0
        while made < n:
            k = 2 if rng.random() < 0.08 and made + 2 <= n else 1
            role += [name] * k
            unit += [g] * k
            g += 1
            made += k
    perm = rng.permutation(len(role))
    role, unit = np.asarray(role)[perm], np.asarray(unit, dtype=np.int64)[perm]
    rid = (np.arange(len(role)) * 7 + 3).astype(np.int64)[rng.permutation(len(role))]
    D = {"role": role, "unit": unit, "row_id": rid, "sealed": True, "X": rng.normal(size=(len(role), n_cols)),
         "feature_names": np.asarray([f"f{j}" for j in range(n_cols)]),
         "idx": {r: np.flatnonzero(role == r) for r in SRC_ROLES}}
    n = len(role)
    D["sex"] = rng.integers(0, 2, n)
    D["race"] = rng.integers(0, 2, n)
    D["y_income"] = rng.integers(0, 2, n)
    D["y_occupation_group"] = rng.integers(0, 6, n)
    a = D["idx"][ASSESS]
    for k in ("sex", "race", "y_income", "y_occupation_group"):
        D[k][a] = -1
    D["y"] = {"income": D["y_income"], "occupation_group": D["y_occupation_group"]}
    # hcal subroles: first n_held AUDIT groups (in a label-free order) -> one representative each; the rest -> ATT
    au = D["idx"][AUDIT]
    groups = np.unique(unit[au])
    held_g = groups[:n_held]
    reps = np.asarray([au[unit[au] == u][np.argmin(rid[au[unit[au] == u]])] for u in held_g], dtype=np.int64)
    att = au[~np.isin(unit[au], held_g)]
    fit = D["idx"][FIT]
    tm = np.asarray([fit[unit[fit] == u][0] for u in np.unique(unit[fit])[:20]], dtype=np.int64)
    D["hcal"] = {"roles": {HELD: np.sort(reps), ATT: np.sort(att), TM: np.sort(tm),
                           "CALIBRATION_HELDOUT_GROUP_ROWS": np.sort(au[np.isin(unit[au], held_g)])}}
    return D


def expected_of(D):
    pos = F._role_positions(D)
    return {r: (int(pos[r].size), int(np.unique(D["unit"][pos[r]]).size)) for r in F.ROLES}


def restrict_synth(D, **kw):
    return F.restrict(D, expected=expected_of(D), n_columns=D["X"].shape[1],
                      forbidden_lengths=(len(D["row_id"]), len(D["idx"][ASSESS])), **kw)


# ------------------------------------------------------------------ allowlist
ALLOWED = {("defense_fit", FIT): ("sex",), ("attack", ATT): ("sex",), ("attack", INNER): ("sex",),
           ("utility", INNER): ("income", "occupation")}


@pytest.mark.parametrize("proc", ["geometry", "defense_fit", "attack", "utility", "calibration", "selection",
                                  "fitting", "assessment", "attack_diagnostic", "inner_audit", "head"])
@pytest.mark.parametrize("role", [FIT, HELD, ATT, INNER, HEAD, AUDIT, ASSESS, TM])
def test_allowlist_table(proc, role):
    if (proc, role) in ALLOWED:
        assert F.allowed_kinds(proc, role) == ALLOWED[(proc, role)]
    else:
        with pytest.raises(F.AccessRefused):
            F.allowed_kinds(proc, role)


def test_geometry_reads_no_labels_and_head_validation_is_always_refused():
    assert F.ALLOW["geometry"] == {}
    for proc in F.ALLOW:
        with pytest.raises(F.AccessRefused):
            F.allowed_kinds(proc, HEAD)
        with pytest.raises(F.AccessRefused):
            F.allowed_kinds(proc, ASSESS)
        assert HELD not in F.ALLOW[proc] and AUDIT not in F.ALLOW[proc]


def test_refusals_happen_before_any_data_access(monkeypatch):
    def boom():
        raise AssertionError("data touched")
    monkeypatch.setattr(F, "_source", boom)
    for proc, role in [("geometry", FIT), ("attack", HEAD), ("utility", HEAD), ("defense_fit", ASSESS),
                       ("calibration", HELD), ("attack", FIT), ("utility", ATT)]:
        with pytest.raises(F.AccessRefused):
            F.labels(proc, role)
    monkeypatch.setattr(F, "pilot_lock_pushed", lambda *a, **k: (False, "PILOT_LOCK.json does not exist"))
    for proc, role in ALLOWED:
        with pytest.raises(F.AccessRefused, match="pilot-only"):
            F.labels(proc, role)


def test_pilot_lock_check_refuses_missing_and_uncommitted(tmp_path):
    ok, why = F.pilot_lock_pushed(lock=tmp_path / "PILOT_LOCK.json")
    assert not ok and "does not exist" in why
    p = tmp_path / "PILOT_LOCK.json"
    p.write_text(json.dumps({"name": "PILOT_LOCK", "synthetic": True}))
    ok, why = F.pilot_lock_pushed(lock=p)
    assert not ok and "not committed" in why


def test_labels_allowed_path_on_synthetic_source(monkeypatch):
    D = synth_D()
    monkeypatch.setattr(F, "_source", lambda: D)
    monkeypatch.setattr(F, "pilot_lock_pushed", lambda *a, **k: (True, "synthetic"))
    monkeypatch.setattr(F, "restrict", functools.partial(F.restrict, expected=expected_of(D),
                                                         n_columns=D["X"].shape[1],
                                                         forbidden_lengths=(len(D["row_id"]),
                                                                            len(D["idx"][ASSESS]))))
    W = F.restrict(D)
    pos_of = {int(r): j for j, r in enumerate(D["row_id"])}
    for (proc, role), kinds in ALLOWED.items():
        out = F.labels(proc, role, data=W)
        assert np.array_equal(out["positions"], W["idx"][role])
        assert np.array_equal(out["row_id"], W["row_id"][W["idx"][role]])
        src = np.asarray([pos_of[int(r)] for r in out["row_id"]])
        for kind in kinds:
            key = {"sex": "sex", "income": "y_income", "occupation": "y_occupation_group"}[kind]
            assert np.array_equal(out[kind], D[key][src])
            assert np.all(out[kind] >= 0)
        assert set(out) == {"procedure", "role", "positions", "row_id", *kinds}
    other = synth_D(seed=5)
    with pytest.raises(F.AccessRefused, match="not the restriction"):
        F.labels("attack", ATT, data=restrict_synth(other))


# ------------------------------------------------------------------ guard
def test_restrict_drops_every_assessment_row_and_keeps_no_label():
    D = synth_D()
    W = restrict_synth(D)
    a_rid = D["row_id"][D["idx"][ASSESS]]
    assert np.intersect1d(W["row_id"], a_rid).size == 0
    n = W["row_id"].size
    assert n == sum(expected_of(D)[r][0] for r in F.ROLES)
    for p, arr in F._walk_arrays(W):
        assert arr.shape[0] not in (len(D["row_id"]), len(a_rid)), p
    for k in ("sex", "race", "y", "y_income", "y_occupation_group", "labels"):
        assert k not in W
    assert W["X"].shape == (n, D["X"].shape[1]) and W["unit"].shape == (n,) and W["role"].shape == (n,)
    for r in F.ROLES:
        assert np.all(W["role"][W["idx"][r]] == r)
        assert set(W["row_id"][W["idx"][r]].tolist()) == set(D["row_id"][F._role_positions(D)[r]].tolist())
    assert np.isin(W["idx"][TM], W["idx"][FIT]).all()
    held_group_rows = D["hcal"]["roles"]["CALIBRATION_HELDOUT_GROUP_ROWS"]
    non_rep = np.setdiff1d(held_group_rows, D["hcal"]["roles"][HELD])
    assert np.intersect1d(W["row_id"], D["row_id"][non_rep]).size == 0
    assert W["guard"]["dropped"] == {"assessment_rows": int(a_rid.size), "other_rows_in_no_ccm_role": int(non_rep.size)}
    pos_of = {int(r): j for j, r in enumerate(D["row_id"])}
    assert np.all(np.diff([pos_of[int(r)] for r in W["row_id"]]) > 0)          # source row order is kept


def test_guard_check_refuses_assessment_ids_forbidden_lengths_and_label_keys():
    D = synth_D()
    W = restrict_synth(D)
    a_rid = D["row_id"][D["idx"][ASSESS]]
    fl = (len(D["row_id"]), len(a_rid))
    bad = dict(W, row_id=W["row_id"].copy())
    bad["row_id"][0] = a_rid[0]
    with pytest.raises(F.AccessRefused, match="assessment row id"):
        F.guard_check(bad, a_rid, fl)
    with pytest.raises(F.AccessRefused, match="length"):
        F.guard_check(dict(W, extra=np.zeros(len(a_rid))), a_rid, fl)
    with pytest.raises(F.AccessRefused, match="length"):
        F.guard_check(dict(W, idx=dict(W["idx"], leak=np.zeros(len(D["row_id"])))), a_rid, fl)
    with pytest.raises(F.AccessRefused, match="label keys"):
        F.guard_check(dict(W, sex=np.zeros(W["row_id"].size)), a_rid, fl)


def test_restrict_refuses_unsealed_overlapping_or_miscounted_sources():
    D = synth_D()
    with pytest.raises(F.AccessRefused, match="sealed"):
        restrict_synth(dict(D, sealed=False))
    exp = expected_of(D)
    exp[FIT] = (exp[FIT][0] + 1, exp[FIT][1])
    with pytest.raises(F.AccessRefused, match="pinned"):
        F.restrict(D, expected=exp, n_columns=D["X"].shape[1])
    D2 = synth_D()
    D2["hcal"]["roles"][ATT] = np.sort(np.concatenate([D2["hcal"]["roles"][ATT], D2["idx"][ASSESS][:1]]))
    with pytest.raises(F.AccessRefused):
        restrict_synth(D2)
    with pytest.raises(F.AccessRefused, match="columns"):
        F.restrict(D, expected=expected_of(D), n_columns=D["X"].shape[1] + 1)


def test_reference_refuses_non_reference_roles_before_loading(monkeypatch):
    monkeypatch.setattr(F, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded")))
    for role in (HEAD, ASSESS, AUDIT, TM, "anything"):
        with pytest.raises(F.AccessRefused):
            F.reference(0, role)


def test_private_cache_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PCRL_CCM_PRIVATE_CACHE", str(tmp_path))
    assert F.ref_dir() == tmp_path / "ref"


def test_teacher_parity_reads_only_permitted_rows(monkeypatch, tmp_path):
    rng = np.random.default_rng(3)
    n_all = 50
    rid = (np.arange(n_all) * 11 + 1)[rng.permutation(n_all)].astype(np.int64)
    p1 = rng.dirichlet([1, 1], n_all)
    p2 = rng.dirichlet(np.ones(6), n_all)
    path = tmp_path / "teacher.npz"
    np.savez(path, row_id=rid, p1=p1, p2=p2, d1=p1.argmax(1), d2=p2.argmax(1))
    monkeypatch.setattr(F, "teacher_npz", lambda k: path)
    pin = {"teacher_npz_sha256": F.sha_file(path)}
    keep = np.sort(rng.choice(n_all, 20, replace=False))
    U0 = {1: p1[keep], 2: p2[keep]}
    d = {1: p1[keep].argmax(1), 2: p2[keep].argmax(1)}
    out = F.teacher_parity(0, rid[keep], U0, d, pin)
    assert out["ok"] and out["rows_compared"] == 20
    U0b = {1: U0[1].copy(), 2: U0[2]}
    U0b[1][3, 0] = np.nextafter(U0b[1][3, 0], 1.0)
    assert not F.teacher_parity(0, rid[keep], U0b, d, pin)["ok"]
    with pytest.raises(F.AccessRefused, match="admitted hash"):
        F.teacher_parity(0, rid[keep], U0, d, {"teacher_npz_sha256": "0" * 64})


def test_no_assess_module_is_imported():
    code = ("import sys; import ccm.data, hcal.data, hcal.calib, dpc.deploy; "
            "print([m for m in sys.modules if m.split('.')[-1] == 'assess'])")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(I.WT),
                       env={"PYTHONPATH": str(I.WT), "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


# ------------------------------------------------------------------ frozen temperature (Ucal)
@pytest.mark.parametrize("K", [2, 6])
@pytest.mark.parametrize("alpha", [0.25, 0.8347144059483327, 1.0, 1.7, 4.0])
def test_ucal_preserves_argmax_on_every_row(K, alpha):
    rng = np.random.default_rng(K)
    P = rng.dirichlet(np.full(K, 0.6), 400)
    P[0] = np.full(K, 1.0 / K) + np.r_[1e-9, -1e-9 / (K - 1) * np.ones(K - 1)]      # near-tie row
    P[1] = np.r_[1.0 - 1e-13 * (K - 1), np.full(K - 1, 1e-13)]                    # floor row
    P[2] = np.r_[np.full(K - 1, 1e-14), 1.0 - 1e-14 * (K - 1)]
    P = P / P.sum(1, keepdims=True)
    d = P.argmax(1)
    tab = {"target": "U", "kind": "H-GLOBAL-TEMP", "K": K, "alpha": alpha, "alphas": None}
    q = F.apply_frozen(P, tab, d)
    assert np.array_equal(q.argmax(1), d)
    assert np.allclose(q.sum(1), 1.0, atol=1e-12, rtol=0)
    if alpha == 1.0:
        assert np.array_equal(q, P)
    sub = np.arange(0, 400, 3)
    assert np.array_equal(F.apply_frozen(P[sub], tab, d[sub]), q[sub])     # row-independent: subsets are exact


def test_ucal_refuses_a_decision_that_is_not_the_teacher_argmax():
    P = np.array([[0.7, 0.3], [0.2, 0.8]])
    tab = {"target": "U", "kind": "H-GLOBAL-TEMP", "K": 2, "alpha": 0.9, "alphas": None}
    with pytest.raises(ValueError):
        F.apply_frozen(P, tab, np.array([1, 1]))


# ------------------------------------------------------------------ toy laws
def test_exp_bracket_is_rigorous_and_tight():
    for lo, hi, x in ((*F.EXP_D, 0.005), (*F.EXP_NEG_D, -0.005)):
        assert lo < hi and hi - lo < Fraction(1, 10 ** 40)
        assert float(lo) <= math.exp(x) <= float(hi)


def test_bin_status_classes():
    fr = lambda *v: tuple(Fraction(x) for x in v)
    st, q = F.bin_status([fr("9/10", "1/10"), fr("902/1000", "98/1000")])
    assert st == "CERTIFIED" and F._certify(q, [fr("9/10", "1/10"), fr("902/1000", "98/1000")], 0)
    assert F.bin_status([fr("9/10", "1/10"), fr("91/100", "9/100")])[0] == "INFEASIBLE_NLL"
    assert F.bin_status([fr("9/10", "1/10"), fr("1/10", "9/10")])[0] == "INFEASIBLE_CLASS"
    # NLL-necessary holds (sum max = 1.0045 <= e^d) but the Brier guard has no room: never claimed infeasible
    assert F.bin_status([fr("6/10", "4/10"), fr("6045/10000", "3955/10000")])[0] == "UNDECIDED"


def test_ambiguous_law_is_refused():
    doc = json.loads(F.TOY_LAWS.read_text())
    law = json.loads(json.dumps(doc["laws"]["NULL"]))
    for x, p in zip(law["inputs"][:2], (("6/10", "4/10"), ("6045/10000", "3955/10000"))):
        x["p1"] = list(p)
        x["p1_float"] = [float(Fraction(v)) for v in p]
    with pytest.raises(ValueError, match="UNDECIDED"):
        F.toy_law_check(law)


def test_pinned_toy_laws_reproduce_and_hold():
    res = F.check_toy_laws()
    assert res["construction_check_reproduced"]
    assert all(res["intended_properties_hold"].values())
    doc = json.loads(F.TOY_LAWS.read_text())
    assert set(doc["laws"]) == set(F.LAW_NAMES)
    for law in doc["laws"].values():
        assert len(law["inputs"]) <= 8
        assert law["recipients"]["1"]["K"] == 2 and law["recipients"]["2"]["K"] == 3
    man = I.PKG / "DATA_ACCESS_MANIFEST.json"
    if man.exists():
        assert json.loads(man.read_text())["toy_laws"]["sha256"] == res["file_sha256"]


def test_amendment_kept_the_original_laws_and_joint_headroom_holds():
    import hashlib
    doc = json.loads(F.TOY_LAWS.read_text())
    canon = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (a1,) = doc["amendments"]
    assert a1["previous_file_sha256"] == "bdb8b0fe48ab2c19201c6545b9453f5c1ac3d35e0798c815d2fb1463c6b03c12"
    for name, h in a1["unchanged_laws_canonical_sha256"].items():
        assert canon(doc["laws"][name]) == h
    for name, h in a1["unchanged_construction_checks_canonical_sha256"].items():
        assert canon(doc["construction_check"][name]) == h
    co = F.toy_law_check(doc["laws"]["JOINT_HEADROOM"])["coalition"]
    for arm in ("sequential_1_to_2_mi_range", "sequential_2_to_1_mi_range", "local_pair_mi_range"):
        assert co["joint_min_mi"] < co[arm][0] - 0.05
