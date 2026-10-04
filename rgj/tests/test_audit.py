"""Tests for the independent audit, the outer writer gate and the reference baselines (synthetic data only).

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m pytest rgj/tests/test_audit.py -q
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from rgj import audit as AU


# ------------------------------------------------------------------ synthetic fixtures
def synth_D(n=3000, seed=0, extra_roles=("SCORE",)):
    """Rows split into AUDIT_FIT / INNER_SELECTION (+ extra scored roles), SEX ~ Bern(1/2)."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    perm = rng.permutation(n)
    roles = ("AUDIT_FIT", "INNER_SELECTION") + tuple(extra_roles)
    cuts = np.array_split(perm, len(roles))
    return {"sex": s, "idx": {r: np.sort(c) for r, c in zip(roles, cuts)}}, rng


def xor_views(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
    s = a ^ b
    v1 = np.column_stack([2 * a - 1 + 0.05 * rng.normal(size=n), rng.normal(size=(n, 3))])
    v2 = np.column_stack([2 * b - 1 + 0.05 * rng.normal(size=n), rng.normal(size=(n, 3))])
    D, _ = synth_D(n, seed + 1)
    D["sex"] = s
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2])}, D


def leaky_local_views(n=3000, seed=0):
    """v1 reads SEX through a noisy coordinate, v2 is pure noise: the coalition bank must keep recipient 1's attack."""
    D, rng = synth_D(n, seed)
    s = D["sex"]
    v1 = np.column_stack([s + rng.normal(scale=0.8, size=n), rng.normal(size=(n, 3))])
    v2 = rng.normal(size=(n, 4))
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2])}, D


# ------------------------------------------------------------------ coalition bank
def test_coalition_bank_contains_both_ignore_candidates_and_selects_inner_auc_argmax():
    V, D = leaky_local_views()
    r = AU.inner_audit(V, D)
    bank = r["selection"]["pair"]["bank"]
    n_slate = len(AU.inner_slate(False))
    kinds = [b["candidate"] for b in bank]
    assert kinds.count("coalition") == n_slate
    assert kinds.count("ignore_recipient_2") == n_slate and kinds.count("ignore_recipient_1") == n_slate
    assert {b["view"] for b in bank if b["candidate"] == "ignore_recipient_2"} == {"v1"}
    assert {b["view"] for b in bank if b["candidate"] == "ignore_recipient_1"} == {"v2"}
    aucs = np.array([b["inner_auc"] for b in bank])
    j = r["selection"]["pair"]["auc"]["bank_index"]
    assert j == int(np.argmax(aucs)) and r["auc"]["pair"] == aucs.max()
    assert r["auc"]["pair"] >= max(r["auc"]["v1"], r["auc"]["v2"])
    # ignore tables equal the local slates' tables (same fitted models)
    v1_aucs = [x["inner_auc"] for x in r["tables"]["v1"]]
    assert [b["inner_auc"] for b in bank if b["candidate"] == "ignore_recipient_2"] == v1_aucs
    # the CE selection is the bank's CE argmin, chosen separately
    ces = np.array([b["inner_ce"] for b in bank])
    assert r["selection"]["pair"]["ce"]["bank_index"] == int(np.argmin(ces))


class _Fixed:
    """Deterministic 'attacker' with a fixed score function of the view's first column (no fitting)."""

    def __init__(self, f):
        self.f = f

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.clip(self.f(X[:, 0]), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - p, p])


def _fixed_slate(finite=False):
    sharp = lambda x: 1 / (1 + np.exp(-40 * (x - 0.5)))              # best ranking, badly over-confident
    soft = lambda x: 0.5 + 0.05 * np.sign(x - 0.5)                    # coarse ranking, well calibrated
    return [("SHARP", lambda s: _Fixed(sharp)), ("SOFT", lambda s: _Fixed(soft))]


def test_dual_selection_picks_auc_and_ce_attackers_separately_and_ignore_candidate_can_win():
    D, rng = synth_D(3000, 17)
    s = D["sex"]
    x = s + rng.normal(scale=0.8, size=len(s))
    V = {"v1": x[:, None], "v2": rng.normal(size=(len(s), 1)), "pair": None}
    V["pair"] = np.column_stack([x + rng.normal(scale=1.5, size=len(s)), x])   # pair's first column is noisier
    rec, _ = AU.select_bank(V, s, D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"], _fixed_slate)
    t = {r["attacker"]: r for r in rec["tables"]["v1"]}
    assert t["SHARP"]["inner_auc"] > t["SOFT"]["inner_auc"] and t["SHARP"]["inner_ce"] > t["SOFT"]["inner_ce"]
    assert rec["selection"]["v1"]["auc"]["attacker"] == "SHARP"
    assert rec["selection"]["v1"]["ce"]["attacker"] == "SOFT"
    pa = rec["selection"]["pair"]["auc"]
    assert (pa["candidate"], pa["view"], pa["attacker"]) == ("ignore_recipient_2", "v1", "SHARP")
    assert pa["inner_auc"] == t["SHARP"]["inner_auc"]


def test_ignore_candidate_wins_when_the_other_view_only_adds_noise():
    """Local signal + a noise recipient: if a v1-only attacker is the bank's best, it is selected (no clamp/omission)."""
    V, D = leaky_local_views(seed=3)
    rng = np.random.default_rng(5)
    V["v2"] = rng.normal(size=(len(D["sex"]), 60))       # high-dimensional noise hurts joint attackers
    V["pair"] = np.hstack([V["v1"], V["v2"]])
    r = AU.inner_audit(V, D)
    best_local = max(x["inner_auc"] for x in r["tables"]["v1"])
    assert r["auc"]["pair"] >= best_local
    if r["selection"]["pair"]["auc"]["candidate"] != "coalition":
        assert r["selection"]["pair"]["auc"]["candidate"] == "ignore_recipient_2"


def test_xor_pair_detected_by_coalition_bank_while_locals_are_uninformative():
    V, D = xor_views()
    r = AU.inner_audit(V, D)
    assert r["auc"]["v1"] < 0.57 and r["auc"]["v2"] < 0.57, r["auc"]
    assert r["auc"]["pair"] > 0.9, r["auc"]
    assert r["selection"]["pair"]["auc"]["candidate"] == "coalition"
    assert r["coalition_minus_best_local"] > 0.3


def test_auc_orientation_is_fixed():
    y = np.array([0, 0, 1, 1])
    P = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])   # perfectly inverted scores
    assert AU.auc_fixed(y, P) == 0.0


# ------------------------------------------------------------------ final slate
def test_final_audit_dual_selection_refits_and_role_guard():
    V, D = leaky_local_views(n=2400, seed=7)
    f, v, sc = D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"], D["idx"]["SCORE"]
    rec, P = AU.final_audit(V, D["sex"], f, v, sc, slate_fn=AU.inner_slate)
    for w in ("v1", "v2", "pair"):
        for crit in ("auc", "ce"):
            assert P[f"{crit}_{w}"].shape == (3, len(sc), 2)
            assert np.allclose(P[f"{crit}_{w}"].sum(2), 1)
        sel = rec["selection"][w]
        rows = rec["tables"][w] if w != "pair" else sel["bank"]
        assert sel["auc"]["inner_auc"] == max(x["inner_auc"] for x in rows)
        assert sel["ce"]["inner_ce"] == min(x["inner_ce"] for x in rows)
        assert rec["scored"][w]["auc"]["seeds"] == [0, 1, 2]
    assert {"coalition", "ignore_recipient_2", "ignore_recipient_1"} <= {b["candidate"] for b in rec["selection"]["pair"]["bank"]}
    with pytest.raises(AssertionError):
        AU.final_audit(V, D["sex"], f, v, v, slate_fn=AU.inner_slate)      # scoring on selection rows is refused


def test_final_slate_detects_tiny_amplitude_clue():
    D, rng = synth_D(2400, 11)
    s = D["sex"]
    X = np.column_stack([rng.normal(size=(len(s), 6)), 1e-6 * (s - 0.5 + rng.normal(scale=0.3, size=len(s)))])
    rec, _ = AU.final_audit({"v": X}, s, D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"], D["idx"]["SCORE"])
    assert rec["selection"]["v"]["auc"]["inner_auc"] > 0.75
    assert rec["scored"]["v"]["auc"]["auc_mean"] > 0.75


def test_multiclass_secondary_audit_with_remapped_classes():
    D, rng = synth_D(3000, 13)
    y = rng.integers(0, 3, len(D["sex"]))
    X = np.column_stack([np.eye(3)[y] + rng.normal(scale=0.7, size=(len(y), 3)), rng.normal(size=(len(y), 2))])
    V = {"v1": X[:, :3], "v2": X[:, 3:], "pair": X}
    rec, P = AU.final_audit(V, y, D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"], D["idx"]["SCORE"],
                            slate_fn=AU.secondary_slate, K=3, classes=[0, 1, 2])
    assert P["auc_pair"].shape == (3, len(D["idx"]["SCORE"]), 3)
    assert rec["scored"]["pair"]["auc"]["auc_mean"] > 0.7
    assert rec["scored"]["v2"]["auc"]["auc_mean"] < 0.58


# ------------------------------------------------------------------ controls
def test_null_near_chance_and_planted_leaks_detected():
    V, D = xor_views(n=2400, seed=21)
    D["sex"] = np.random.default_rng(99).integers(0, 2, len(D["sex"]))   # views carry no SEX information
    c = AU.controls(V, D, slate="inner")
    for w, d in c["views"].items():
        assert d["null_auc"] <= AU.NULL_MAX, (w, d)
        assert d["PLANT_ONEHOT_auc"] > AU.PLANT_MIN and d["PLANT_TINY_auc"] > AU.PLANT_MIN, (w, d)
        assert d["null_ok"] and d["plant_ok"]
    assert c["all_ok"]
    assert c["views"]["pair"]["PLANT_ONEHOT_selected"].startswith("coalition:pair:")   # plant sits in the pair copy only


def test_null_permutation_is_frozen_and_within_role():
    D, _ = synth_D(1200, 2)
    S = D["sex"]
    a, ha = AU.frozen_permutation(S, D)
    b, hb = AU.frozen_permutation(S, D)
    assert ha == hb and np.array_equal(a, b)
    for r in ("AUDIT_FIT", "INNER_SELECTION"):
        ix = D["idx"][r]
        assert np.bincount(a[ix], minlength=2).tolist() == np.bincount(S[ix], minlength=2).tolist()
    other = D["idx"]["SCORE"]
    assert np.array_equal(a[other], S[other])


# ------------------------------------------------------------------ output closure (centred logits from r_i only)
def _head_D(n=3000, seed=4):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    tr, va = np.sort(perm[: n // 2]), np.sort(perm[n // 2: 3 * n // 4])
    R1, R2 = rng.normal(size=(n, 16)), rng.normal(size=(n, 16))
    y1 = (R1[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    y2 = np.argmax(R2[:, :6] + 0.5 * rng.normal(size=(n, 6)), 1).astype(np.int64)
    D = {"idx": {"DEFENSE_FIT": tr, "HEAD_VALIDATION": va}, "y": {"income": y1, "occupation_group": y2}}
    return D, R1, R2


def test_output_closure_centred_logits_depend_only_on_own_released_features():
    from rgj import finalize as FN
    D, R1, R2 = _head_D()
    out, heads, _ = FN.heads_and_outputs([R1, R2], D)
    # (a) recomputing each recipient's outputs from its own released features alone reproduces them
    for i, R in ((0, R1), (1, R2)):
        cen, P, hard = FN.outputs(heads[i], R)
        assert np.array_equal(cen, out[f"c{i + 1}"]) and np.array_equal(hard, out[f"hard{i + 1}"])
        assert np.allclose(cen.sum(1), 0)
    # (b) changing the other recipient's features leaves recipient 1's view untouched
    out2, _, _ = FN.heads_and_outputs([R1, R2 + 5.0 * np.random.default_rng(1).normal(size=R2.shape)], D)
    assert np.array_equal(out2["c1"], out["c1"]) and np.array_equal(out2["r1"], out["r1"])
    # (c) the centred logits are exactly affine in r_i (no raw-output or untreated-head bypass)
    for i, R in ((0, R1), (1, R2)):
        A = np.hstack([R, np.ones((len(R), 1))])
        coef, *_ = np.linalg.lstsq(A, out[f"c{i + 1}"], rcond=None)
        assert np.max(np.abs(A @ coef - out[f"c{i + 1}"])) < 1e-8
    V = FN.views_from_release(out)
    assert np.array_equal(V["v1"], np.hstack([out["r1"], out["c1"]]))
    assert np.array_equal(V["pair"], np.hstack([V["v1"], V["v2"]]))


# ------------------------------------------------------------------ outer writer gate (EVALUATION_LOCK.json)
BRANCH = "research/pcrl-refreshed-guarded-joint-v1"


def _git(cwd, *args):
    r = subprocess.run(["git", "-c", "user.name=rgj-test", "-c", "user.email=", *args], cwd=cwd, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _lock_repo(tmp_path, push=True, content='{"seeds": {}}'):
    origin, wt = tmp_path / "origin.git", tmp_path / "wt"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(tmp_path, "clone", "-q", str(origin), str(wt))
    _git(wt, "checkout", "-q", "-b", BRANCH)
    lock = wt / "EVALUATION_LOCK.json"
    lock.write_text(content)
    _git(wt, "add", lock.name)
    _git(wt, "commit", "-q", "-m", "lock")
    if push:
        _git(wt, "push", "-q", "-u", "origin", BRANCH)
    return wt, lock


def test_outer_writer_refuses_without_an_opened_pushed_lock(tmp_path):
    from rgj import assess as AS
    AS.close_assessment()
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.outer_unit({}, 0, "J-G", {"kind": "neural", "unit": "x"}, units_dir=tmp_path)
    wt, lock = _lock_repo(tmp_path, push=False)
    assert AS.lock_is_pushed(lock, wt, BRANCH)["reason"].startswith("lock commit")       # committed, not pushed
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.open_assessment(lock, wt, BRANCH)
    _git(wt, "push", "-q", "-u", "origin", BRANCH)
    assert AS.lock_is_pushed(lock, wt, BRANCH)["ok"]
    lock.write_text('{"seeds": {"0": {}}}')                                             # edited after the push
    assert not AS.lock_is_pushed(lock, wt, BRANCH)["ok"]
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.open_assessment(lock, wt, BRANCH)
    other = wt / "OTHER_LOCK.json"
    other.write_text("{}")
    assert not AS.lock_is_pushed(other, wt, BRANCH)["ok"]                               # wrong file
    uncommitted = tmp_path / "loose" / "EVALUATION_LOCK.json"
    uncommitted.parent.mkdir()
    uncommitted.write_text("{}")
    assert not AS.lock_is_pushed(uncommitted, wt, BRANCH)["ok"]                         # outside the repository


def test_open_assessment_is_revoked_when_the_lock_changes(tmp_path):
    from rgj import assess as AS
    wt, lock = _lock_repo(tmp_path)
    AS.open_assessment(lock, wt, BRANCH)
    lock.write_text('{"changed": true}')
    with pytest.raises(SystemExit, match="REFUSED"):
        AS._require_open()
    AS.close_assessment()


# ------------------------------------------------------------------ synthetic study-shaped data
def study_D(n=2600, seed=0):
    rng = np.random.default_rng(seed)
    roles = np.array(["DEFENSE_FIT"] * 10 + ["HEAD_VALIDATION"] * 2 + ["AUDIT_FIT"] * 4 + ["INNER_SELECTION"] * 3 +
                     ["DEVELOPMENT_ASSESSMENT"] * 3)
    role = roles[rng.integers(0, len(roles), n)]
    sex = rng.integers(0, 2, n)
    race = rng.choice(5, n, p=[0.01, 0.15, 0.25, 0.01, 0.58])
    X = rng.normal(size=(n, 83)).astype(np.float32)
    X[:, 0] += 1.5 * sex
    y_inc = (X[:, 1] + 0.5 * X[:, 0] + rng.normal(size=n) > 0.5).astype(np.int64)
    y_occ = np.argmax(X[:, 2:8] + rng.normal(size=(n, 6)), 1).astype(np.int64)
    D = {"row_id": np.arange(n, dtype=np.int64) * 7 + 3, "unit": np.arange(n, dtype=np.int64), "role": role, "X": X,
         "sex": sex, "race": race, "y": {"income": y_inc, "occupation_group": y_occ}}
    D["idx"] = {r: np.flatnonzero(role == r) for r in np.unique(roles)}
    return D


def _neural_unit(D, units, name, seed=1):
    import joblib

    from rgj import finalize as FN
    rng = np.random.default_rng(seed)
    X = D["X"].astype(np.float64)
    R1 = X[:, :16] @ rng.normal(size=(16, 16)) * 0.3
    R2 = X[:, np.r_[0, 2:17]] @ rng.normal(size=(16, 16)) * 0.3
    out, heads, meta = FN.heads_and_outputs([R1, R2], D)
    files = {"release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out)}
    for i, h in heads.items():
        files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
    FN.save_unit(Path(units) / name, files, {"unit": name, "heads": meta})


def test_outer_unit_end_to_end_writes_documented_preds(tmp_path):
    from rgj import assess as AS
    from rgj import baselines as BL
    D = study_D()
    units = tmp_path / "units"
    _neural_unit(D, units, "ck__test")
    for i in (0, 1):   # finite FARE-style releases from fixed cells (head refit on the new roles)
        cells = (D["X"][:, i + 2] > 0).astype(np.int64) + 2 * (D["X"][:, 0] > 0.7)
        BL.write_fare_unit(f"fare__s0__p{i}__c9", 0, i, cells, 4, {"id": 9}, D, {"origin": "synthetic test"}, units)
    wt, lock = _lock_repo(tmp_path)
    AS.open_assessment(lock, wt, BRANCH)
    try:
        r = AS.outer_unit(D, 0, "C*", {"kind": "neural", "unit": "ck__test"}, units_dir=units)
        rf = AS.outer_unit(D, 0, "F", {"kind": "fare", "units": ["fare__s0__p0__c9", "fare__s0__p1__c9"]},
                           units_dir=units)
    finally:
        AS.close_assessment()
    assert (units / "outer__s0__Cstar" / "COMPLETE.json").exists() and r["label"] == "C*"
    a = D["idx"]["DEVELOPMENT_ASSESSMENT"]
    z = np.load(units / "outer__s0__Cstar" / "preds.npz")
    assert np.array_equal(z["assess_row_id"], D["row_id"][a]) and np.array_equal(z["sex"], D["sex"][a])
    for w in ("v1", "v2", "pair", "p1", "p2", "ppair", "h1", "h2", "hpair"):
        for crit in ("auc", "ce"):
            assert z[f"P_{crit}_{w}"].shape == (3, len(a), 2)
    assert list(z["race_codes"]) == r["race"]["supported_codes"] == [1, 2, 4]
    assert z["Prace_auc_pair"].shape == (3, len(z["race_pos"]), 3)
    assert np.array_equal(D["race"][a][z["race_pos"]], z["race_codes"][z["race_y"]])
    # recorded assessment AUC is recomputable from the saved per-row probabilities (fixed orientation)
    sc = r["primary"]["scored"]["pair"]["auc"]
    assert np.isclose(np.mean([AU.auc_fixed(z["sex"], P) for P in z["P_auc_pair"]]), sc["auc_mean"])
    assert r["primary"]["scored"]["v1"]["auc"]["auc_mean"] > 0.6      # v1 carries SEX through X[:, 0]
    assert {"coalition", "ignore_recipient_2", "ignore_recipient_1"} <= \
        {b["candidate"] for b in r["primary"]["selection"]["pair"]["bank"]}
    lin = r["linear_diagnostics"]["r1+r2"]
    assert lin["ols_fit_rows_r2"] > 0.1 and abs(lin["ols_fit_rows_null_r2"]) < 0.05
    assert set(r["preds_keys"]) == set(z.files)
    assert rf["native"]["fare_certificate"][0]["status"].startswith("not recomputed")
    assert any(row["attacker"].startswith("CC_") for row in rf["primary"]["tables"]["v1"])   # finite -> CC


# ------------------------------------------------------------------ baselines
def test_gate_definitions_match_the_lead_selection():
    from rgj import baselines as BL
    from rgj.select import gate_margins
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = {i: {"acc": rng.uniform(.3, .9), "const_acc": rng.uniform(.2, .8)} for i in (0, 1)}
        ur = {i: {"acc": rng.uniform(.3, .9), "const_acc": u[i]["const_acc"]} for i in (0, 1)}
        gm = gate_margins(u, ur)
        for i in (0, 1):
            ok, worst, g = BL.gate_one(u[i], ur[i])
            assert g == pytest.approx(gm[i]) and ok == all(x >= 0 for x in gm[i].values())


def test_fare_selection_rule_matches_predecessor():
    from rgj import baselines as BL
    rows = [{"config": 1, "gate_ok": True, "R_local": 0.70, "margin": 0.02},
            {"config": 2, "gate_ok": True, "R_local": 0.60, "margin": 0.01},
            {"config": 3, "gate_ok": True, "R_local": 0.60, "margin": 0.00},
            {"config": 4, "gate_ok": False, "R_local": 0.50, "margin": -0.01}]
    s = BL.pick_fare_config(rows)
    assert s["status"] == "NOMINEE" and s["config"] == 2            # lowest admissible AUC; tie -> lower id
    none = [dict(r, gate_ok=False) for r in rows]
    s = BL.pick_fare_config(none)
    assert s["status"] == "NO_FEASIBLE_NOMINEE" and s["config"] == 1   # closest by worst-gate margin, descriptive


def test_leace_unit_uses_official_leace_on_U_features_and_refuses_inconsistent_U(tmp_path):
    import torch

    from jcv.train import Model
    from rgj import baselines as BL
    from rgj import finalize as FN
    import joblib
    D = study_D(n=1800, seed=3)
    units = tmp_path / "units"
    model = Model(83, [2, 6], 0)
    out, heads, meta = FN.finalize_model(model, D)
    st = model.state_dict()
    files = {"model.pt": lambda p: torch.save(st, p),
             "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out)}
    for i, h in heads.items():
        files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
    FN.save_unit(units / "tl__s0__e30", files, {"unit": "tl__s0__e30"})
    rec = BL.leace_unit(0, "tl__s0__e30", D, units)
    assert rec["native_check_status"] == {"0": "WITHIN_TOLERANCE", "1": "WITHIN_TOLERANCE"}
    assert "official LEACE" in rec["leace"]["0"]["estimator"]
    assert rec["leace"]["0"]["n_fit"] == len(D["idx"]["DEFENSE_FIT"]) and rec["fit_role"] == "DEFENSE_FIT"
    assert BL.leace_unit(0, "tl__s0__e30", D, units) == rec                    # complete unit is skipped
    z = np.load(units / "lc__s0__E" / "release.npz")
    tr = D["idx"]["DEFENSE_FIT"]
    for i in (1, 2):   # erased features carry no linear SEX signal on the fitting rows
        R = z[f"r{i}"][tr] - z[f"r{i}"][tr].mean(0)
        s = D["sex"][tr] - D["sex"][tr].mean()
        assert np.max(np.abs(R.T @ s)) / len(tr) < 1e-8
    assert set(z.files) == {"row_id", "r1", "c1", "p1", "hard1", "r2", "c2", "p2", "hard2"}
    # a U whose released features are not its encoder outputs is refused
    out2 = dict(out, r1=out["r1"] + 1e-3)
    FN.save_unit(units / "tl__s1__e30", {"model.pt": lambda p: torch.save(st, p),
                                         "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out2)},
                 {"unit": "tl__s1__e30"})
    with pytest.raises(SystemExit, match="REFUSED"):
        BL.leace_unit(1, "tl__s1__e30", D, units)
