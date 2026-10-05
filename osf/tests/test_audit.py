"""Audit / baseline / assessment-gate fixtures for the online-strength frontier study (synthetic data only).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest osf/tests/test_audit.py -q
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from osf import assess as AS
from osf import audit as OA
from osf import baselines as BL
from rgj import finalize as FN
from smf import audit as AU

ROLES = ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION", "OSF_DEVELOPMENT_ASSESSMENT")


# ------------------------------------------------------------------ synthetic osf-shaped data
def osf_D(n=2600, seed=0, sealed=False, races=(0.01, 0.15, 0.25, 0.01, 0.58)):
    rng = np.random.default_rng(seed)
    bag = np.array(["OSF_DEFENSE_FIT"] * 10 + ["HEAD_VALIDATION"] * 2 + ["AUDIT_FIT"] * 4 + ["INNER_SELECTION"] * 3 +
                   ["OSF_DEVELOPMENT_ASSESSMENT"] * 3)
    role = bag[rng.integers(0, len(bag), n)]
    sex = rng.integers(0, 2, n)
    race = rng.choice(len(races), n, p=list(races))
    X = rng.normal(size=(n, 83)).astype(np.float32)
    X[:, 0] += 1.5 * sex
    y_inc = (X[:, 1] + 0.5 * X[:, 0] + rng.normal(size=n) > 0.5).astype(np.int64)
    y_occ = np.argmax(X[:, 2:8] + rng.normal(size=(n, 6)), 1).astype(np.int64)
    D = {"row_id": np.arange(n, dtype=np.int64) * 7 + 3, "unit": np.arange(n, dtype=np.int64), "role": role, "X": X,
         "sex": sex, "race": race, "y_income": y_inc, "y_occupation_group": y_occ, "sealed": False}
    D["idx"] = {r: np.flatnonzero(role == r) for r in ROLES}
    D["idx"]["DEFENSE_FIT"] = D["idx"]["OSF_DEFENSE_FIT"]
    D["idx"]["DEVELOPMENT_ASSESSMENT"] = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    if sealed:
        a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
        for k in ("sex", "race", "y_income", "y_occupation_group"):
            D[k] = D[k].copy()
            D[k][a] = -1
        D["sealed"] = True
    D["y"] = {"income": D["y_income"], "occupation_group": D["y_occupation_group"]}
    return D


def neural_unit(D, units, name, seed=1, leak=0.3):
    import joblib
    rng = np.random.default_rng(seed)
    X = D["X"].astype(np.float64)
    R1 = X[:, :16] @ rng.normal(size=(16, 16)) * leak
    R2 = X[:, np.r_[0, 2:17]] @ rng.normal(size=(16, 16)) * leak
    out, heads, meta = FN.heads_and_outputs([R1, R2], D)
    files = {"release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out)}
    for i, h in heads.items():
        files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
    FN.save_unit(Path(units) / name, files, {"unit": name, "heads": meta})
    return out


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in ("fit_s", "wall_s")}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


# ------------------------------------------------------------------ role guard and sealed labels
def test_inner_audit_refuses_sealed_labels_and_assessment_rows_in_attacker_roles():
    D = osf_D(sealed=True)
    X = D["X"].astype(np.float64)
    V = {"v1": X[:, :6], "v2": X[:, 6:12], "pair": X[:, :12]}
    a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    bad = dict(D, idx=dict(D["idx"], AUDIT_FIT=np.sort(np.r_[D["idx"]["AUDIT_FIT"], a[:5]])))
    with pytest.raises(ValueError, match="REFUSED"):
        OA.inner_audit(V, bad)
    norole = {k: v for k, v in bad.items() if k != "role"}             # without role strings: overlap still refused
    with pytest.raises(ValueError, match="REFUSED"):
        OA.inner_audit(V, norole)
    S = D["sex"].copy()
    S[D["idx"]["INNER_SELECTION"][:3]] = -1
    with pytest.raises(ValueError, match="sealed"):
        OA.inner_audit(V, dict(D, sex=S))
    with pytest.raises(ValueError, match="sealed"):
        OA.final_audit(V, D["sex"], D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"], a, slate_fn=AU.inner_slate)


def test_inner_audit_and_utility_never_read_assessment_labels():
    Ds, Dr = osf_D(sealed=True), osf_D()
    a = Dr["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    for k in ("sex", "y_income", "y_occupation_group"):
        Dr[k] = Dr[k].copy()
        Dr[k][a] = 1 - np.minimum(Dr[k][a], 1)
    Dr["y"] = {"income": Dr["y_income"], "occupation_group": Dr["y_occupation_group"]}
    X = Ds["X"].astype(np.float64)
    V = {"v1": X[:, :6], "v2": X[:, 6:12], "pair": X[:, :12]}
    assert _strip(OA.inner_audit(V, Ds)) == _strip(OA.inner_audit(V, Dr))
    out = {"hard1": (X[:, 1] > 0).astype(int), "hard2": np.argmax(X[:, 2:8], 1)}
    assert OA.inner_utility(out, Ds) == OA.inner_utility(out, Dr)


def test_inner_utility_constant_is_the_defense_fit_majority_class():
    D = osf_D()
    tr, v = D["idx"]["OSF_DEFENSE_FIT"], D["idx"]["INNER_SELECTION"]
    out = {"hard1": np.zeros(len(D["sex"]), int), "hard2": np.zeros(len(D["sex"]), int)}
    u = OA.inner_utility(out, D)
    for i, t in enumerate(OA.TASKS):
        c = int(np.argmax(np.bincount(D["y"][t][tr])))
        assert u[i]["const_class"] == c and u[i]["const_acc"] == pytest.approx(float((D["y"][t][v] == c).mean()))


# ------------------------------------------------------------------ coalition bank, dual selection, no clamp
class _Fixed:
    def __init__(self, f):
        self.f = f

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.clip(self.f(X[:, 0]), 1e-9, 1 - 1e-9)
        return np.column_stack([1 - p, p])


def _fixed_slate(finite=False):
    sharp = lambda x: 1 / (1 + np.exp(-40 * (x - 0.5)))              # best ranking, over-confident
    soft = lambda x: 0.5 + 0.05 * np.sign(x - 0.5)                    # coarse ranking, calibrated
    return [("SHARP", lambda s: _Fixed(sharp)), ("SOFT", lambda s: _Fixed(soft))]


def test_auc_and_ce_selected_attackers_are_separate_and_ignore_recipient_candidate_can_win():
    D = osf_D(3000, 17)
    rng = np.random.default_rng(3)
    s = D["sex"]
    x = s + rng.normal(scale=0.8, size=len(s))
    V = {"v1": x[:, None], "v2": rng.normal(size=(len(s), 1)),
         "pair": np.column_stack([x + rng.normal(scale=1.5, size=len(s)), x])}
    rec, _ = OA.select_bank(V, s, D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"], _fixed_slate)
    assert rec["selection"]["v1"]["auc"]["attacker"] == "SHARP" and rec["selection"]["v1"]["ce"]["attacker"] == "SOFT"
    pa = rec["selection"]["pair"]["auc"]
    assert (pa["candidate"], pa["view"]) == ("ignore_recipient_2", "v1")
    kinds = [b["candidate"] for b in rec["selection"]["pair"]["bank"]]
    assert kinds.count("coalition") == kinds.count("ignore_recipient_1") == kinds.count("ignore_recipient_2") == 2


def test_scored_coalition_auc_is_never_clamped_to_a_local_result():
    """The pair attacker wins on INNER_SELECTION but is useless on the scored rows; its scored AUC stays below the
    scored local AUC (no max with the locals) and equals the AUC of its own saved probabilities."""
    D = osf_D(3000, 5)
    rng = np.random.default_rng(9)
    s = D["sex"].astype(float)
    sc = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    v1 = (s + rng.normal(scale=0.8, size=len(s)))[:, None]
    good = s + rng.normal(scale=0.2, size=len(s))
    good[sc] = rng.normal(size=len(sc)) + 0.5                       # no signal on the scored rows
    V = {"v1": v1, "v2": rng.normal(size=(len(s), 1)), "pair": np.column_stack([good, v1[:, 0]])}
    rec, P = OA.final_audit(V, D["sex"], D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"], sc,
                            slate_fn=_fixed_slate)
    assert rec["selection"]["pair"]["auc"]["candidate"] == "coalition"
    pair, loc = rec["scored"]["pair"]["auc"]["auc_mean"], rec["scored"]["v1"]["auc"]["auc_mean"]
    assert pair < 0.6 < loc, (pair, loc)
    assert pair == pytest.approx(np.mean([AU.auc_fixed(D["sex"][sc], p) for p in P["auc_pair"]]))
    assert AU.auc_fixed(np.array([0, 0, 1, 1]), np.array([[.1, .9], [.2, .8], [.8, .2], [.9, .1]])) == 0.0  # fixed


def test_xor_pair_is_found_by_the_coalition_while_locals_are_blind():
    D = osf_D(3000, 21)
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 2, 3000), rng.integers(0, 2, 3000)
    D["sex"] = a ^ b
    v1 = np.column_stack([2 * a - 1 + 0.05 * rng.normal(size=3000), rng.normal(size=(3000, 2))])
    v2 = np.column_stack([2 * b - 1 + 0.05 * rng.normal(size=3000), rng.normal(size=(3000, 2))])
    r = OA.inner_audit({"v1": v1, "v2": v2, "pair": np.hstack([v1, v2])}, D)
    assert r["auc"]["v1"] < 0.58 and r["auc"]["v2"] < 0.58 and r["auc"]["pair"] > 0.9
    assert r["selection"]["pair"]["auc"]["candidate"] == "coalition"


# ------------------------------------------------------------------ views and inner units
def test_release_views_record_features_outputs_and_complete_views_separately(tmp_path):
    D = osf_D(1200, 2)
    out = neural_unit(D, tmp_path, "rel__s0__U")
    V = OA.release_views(out)
    assert np.array_equal(V["v1"], np.hstack([out["r1"], out["c1"]])) and np.array_equal(V["r2"], out["r2"])
    assert np.array_equal(V["pair"], np.hstack([V["v1"], V["v2"]])) and V["rpair"].shape[1] == 32
    assert V["h2"].shape[1] == 6 and np.array_equal(V["h2"].argmax(1), out["hard2"]) and set(np.unique(V["h1"])) <= {0, 1}
    assert {f: OA.FAMILIES[f][0] for f in OA.FAMILIES} == {"complete": "pair", "features": "rpair", "logits": "cpair",
                                                          "probs": "ppair", "hard": "hpair"}


def test_inner_unit_writes_selection_record_and_refuses_a_stale_audit(tmp_path):
    D = osf_D(1500, 4, sealed=True)
    neural_unit(D, tmp_path, "rel__s0__X")
    r = OA.inner_unit("rel__s0__X", D, tmp_path, extended=True)
    assert set(r["recovery"]["auc"]) == {"v1", "v2", "pair"}
    assert set(r["utility"]) == {"0", "1"} and {"acc", "const_acc", "const_class"} <= set(r["utility"]["0"])
    assert set(r["recovery_features_only"]["auc"]) == {"r1", "r2", "rpair"}
    assert set(r["recovery_outputs_only"]["hard"]["auc"]) == {"h1", "h2", "hpair"}
    s = OA.inner_summary("rel__s0__X", tmp_path)
    assert s["auc"]["pair"] == r["recovery"]["auc"]["pair"] and s["utility"][0]["acc"] == r["utility"]["0"]["acc"]
    neural_unit(D, tmp_path, "rel__s0__X", seed=2)                      # the release changed after its audit
    with pytest.raises(RuntimeError, match="another version"):
        OA.inner_unit("rel__s0__X", D, tmp_path)


# ------------------------------------------------------------------ defense-aware reader on real-shaped releases
def test_rotated_1e6_clue_survives_serialisation_and_is_detected(tmp_path):
    D, z = AU.synthetic_D(n_fit=3000, n_sel=1500, seed=5)
    rp = OA.rotated_plant(z, D, slate="inner", recipients=(1, 2), workdir=tmp_path)
    assert rp["all_ok"], rp
    for i in ("1", "2"):
        x = rp["recipients"][i]
        assert x["serialisation"]["reread_arrays_identical"] and x[f"v{i}"]["selected_on_A"].endswith(":DA_LR")
        assert max(v for k, v in x[f"v{i}"]["per_attacker_auc_B"].items() if not k.startswith("DA")) < 0.6
        assert x["plant"]["max_coordinate_share_of_clue_direction"] < 0.9


def _projected_view(n, rng, round_after):
    s = rng.integers(0, 2, n)
    u = rng.normal(size=8)
    u /= np.linalg.norm(u)
    R = (rng.normal(size=(n, 8)) * np.exp(rng.uniform(-2, 1, 8)) + 3.0 * (s - 0.5)[:, None] * u + 40.0)
    if round_after:                                                    # float32 rounding AFTER the exact removal
        R = (R - np.outer(R @ u, u)).astype(np.float32).astype(np.float64)
    else:                                                              # release path: float32 encoder, float64 map
        R = R.astype(np.float32).astype(np.float64)
        R = R - np.outer(R @ u, u)
    L = R @ rng.normal(size=(8, 6)) + rng.normal(size=6)              # float64 affine head on the released r
    return np.hstack([R, L - L.mean(1, keepdims=True)]), s


def test_exact_affine_logit_and_projection_nulls_are_dropped_not_amplified():
    X, s = _projected_view(4000, np.random.default_rng(31), round_after=False)
    c = OA.Canon().fit(X[:3000])
    rc = c.receipt()
    assert c.rank_ == 7 and rc["tol_over_s1"] > 100 * rc["max_dropped_over_s1"], rc
    assert rc["min_kept_over_s1"] > 1e6 * rc["tol_over_s1"]
    P = OA.proba(AU.da_lr(0).fit(X[:3000], s[:3000]), X[3000:], 2)
    assert OA.auc_fixed(s[3000:], P) < 0.5 + OA.NULL_Z * OA.null_sd(s[3000:])
    # float32 rounding after the removal leaves a kept direction at rounding scale: whitened, but label-free
    X2, s2 = _projected_view(4000, np.random.default_rng(31), round_after=True)
    c2 = OA.Canon().fit(X2[:3000])
    assert c2.rank_ == 8
    P2 = OA.proba(AU.da_lr(0).fit(X2[:3000], s2[:3000]), X2[3000:], 2)
    assert OA.auc_fixed(s2[3000:], P2) < 0.5 + OA.NULL_Z * OA.null_sd(s2[3000:])
    pr = OA.precision_receipt({"v": X2}, {"idx": {"AUDIT_FIT": np.arange(3000)}}, expected_rank={"v": 7})
    assert pr["v"]["rank_kept"] == 8 and not pr["v"]["rank_matches_expected"]


def test_controls_null_on_held_out_half_and_planted_leaks_detected():
    D, z = AU.synthetic_D(n_fit=3000, n_sel=1500, seed=11)
    z["r1"][:, 0] += 1.5 * D["sex"]                                   # leaks real SEX: plants use S*, still valid
    c = OA.controls(FN.views_from_release(z), D, slate="inner")
    assert c["all_ok"] and not c["failures"]
    assert c["statistic"].startswith("AUC on half B")


# ------------------------------------------------------------------ assessment gate
def _git(cwd, *args):
    r = subprocess.run(["git", "-c", "user.name=osf-test", "-c", "user.email=", *args], cwd=cwd, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _lock_repo(tmp_path, content, push=True, code=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    origin, wt = tmp_path / "origin.git", tmp_path / "wt"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(tmp_path, "clone", "-q", str(origin), str(wt))
    _git(wt, "checkout", "-q", "-b", AS.STUDY_BRANCH)
    for rel, text in (code or {}).items():
        (wt / rel).parent.mkdir(parents=True, exist_ok=True)
        (wt / rel).write_text(text)
        _git(wt, "add", rel)
    lock = wt / AS.LOCK_REL
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(content)
    _git(wt, "add", AS.LOCK_REL)
    _git(wt, "commit", "-q", "-m", "lock")
    if push:
        _git(wt, "push", "-q", "-u", "origin", AS.STUDY_BRANCH)
    return wt, lock


def _code_lock(extra=None):
    import hashlib
    code = {f: f"# {f}\n" for f in AS.CHAIN}
    files = {f: hashlib.sha256(t.encode()).hexdigest() for f, t in code.items()}
    return code, {"seeds": {}, "locked_code_files": files, **(extra or {})}


def test_assessment_refuses_unpushed_edited_or_misplaced_locks(tmp_path):
    AS.close_assessment()
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.load_unsealed()
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.outer_unit({"sealed": False}, 0, "U", {"kind": "release", "unit": "x"}, units_dir=tmp_path)
    code, L = _code_lock()
    wt, lock = _lock_repo(tmp_path, json.dumps(L), push=False, code=code)
    assert AS.lock_is_pushed(lock, wt)["reason"].startswith("lock commit")             # committed, not pushed
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.open_assessment(lock, wt)
    _git(wt, "push", "-q", "-u", "origin", AS.STUDY_BRANCH)
    assert AS.lock_is_pushed(lock, wt)["ok"]
    lock.write_text(json.dumps({**L, "edited": True}))                               # edited after the push
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.open_assessment(lock, wt)
    _git(wt, "checkout", "-q", "--", AS.LOCK_REL)
    assert not AS.lock_is_pushed(lock, wt, "research/pcrl-strength-matched-feedback-v1")["ok"]   # other branch
    stray = wt / "EVALUATION_LOCK.json"
    stray.write_text("{}")
    _git(wt, "add", "EVALUATION_LOCK.json")
    _git(wt, "commit", "-q", "-m", "stray")
    _git(wt, "push", "-q")
    assert "registered" in AS.lock_is_pushed(stray, wt)["reason"]
    AS.open_assessment(lock, wt)                                                      # the real lock opens
    AS.close_assessment()


def test_assessment_refuses_mismatched_or_missing_code_hashes(tmp_path):
    code, L = _code_lock()
    L["locked_code_files"]["osf/audit.py"] = "0" * 64
    wt, lock = _lock_repo(tmp_path / "a", json.dumps(L), code=code)
    with pytest.raises(SystemExit, match="locked code hash"):
        AS.open_assessment(lock, wt)
    code, L = _code_lock()
    del L["locked_code_files"]["osf/assess.py"]
    wt, lock = _lock_repo(tmp_path / "b", json.dumps(L), code=code)
    with pytest.raises(SystemExit, match="does not list"):
        AS.open_assessment(lock, wt)
    code, L = _code_lock()
    code["osf/report.py"] = "# report\n"
    L["locked_code_files"]["osf/report.py"] = "0" * 64                   # outside the chain: recorded, not refused
    wt, lock = _lock_repo(tmp_path / "c", json.dumps(L), code=code)
    AS.open_assessment(lock, wt)
    assert AS._OPENED["code"]["osf/report.py"].startswith("CHANGED") and AS._OPENED["code"]["osf/audit.py"] == "matches lock"
    try:
        (wt / "osf/audit.py").write_text("# changed after the lock\n")
        AS.close_assessment()
        with pytest.raises(SystemExit, match="locked code hash"):
            AS.open_assessment(lock, wt)
    finally:
        AS.close_assessment()


def test_open_assessment_is_revoked_when_the_lock_changes_and_sealed_D_is_refused(tmp_path):
    code, L = _code_lock()
    wt, lock = _lock_repo(tmp_path, json.dumps(L), code=code)
    AS.open_assessment(lock, wt)
    try:
        with pytest.raises(SystemExit, match="REFUSED: D is sealed"):
            AS.outer_unit({"sealed": True}, 0, "U", {"kind": "release", "unit": "x"}, units_dir=tmp_path)
        lock.write_text('{"changed": true}')
        with pytest.raises(SystemExit, match="REFUSED"):
            AS._require_open()
    finally:
        AS.close_assessment()


def test_outer_unit_end_to_end_preds_utility_gain_retention_and_race_support(tmp_path):
    D = osf_D(2400, 8)
    units = tmp_path / "units"
    outU = neural_unit(D, units, "rel__s0__U", seed=1)
    neural_unit(D, units, "rel__s0__RAW-J_b0.3", seed=3, leak=0.2)
    for i in (0, 1):
        cells = (D["X"][:, i + 2] > 0).astype(np.int64) + 2 * (D["X"][:, 0] > 0.7)
        BL.write_fare_unit(f"fare__s0__p{i}__c9", 0, i, cells, 4, {"id": 9}, D, {"origin": "synthetic"}, units)
    lf = {n: json.loads((units / n / "COMPLETE.json").read_text())["files"] for n in ("rel__s0__U", "rel__s0__RAW-J_b0.3")}
    code, L = _code_lock({"seeds": {"0": {"unit_file_sha256": lf, "score": {
        "U": {"kind": "release", "unit": "rel__s0__U"}, "RAW-J|b0.3": {"kind": "release", "unit": "rel__s0__RAW-J_b0.3"},
        "F": {"kind": "fare", "units": ["fare__s0__p0__c9", "fare__s0__p1__c9"]}}}}})
    wt, lock = _lock_repo(tmp_path, json.dumps(L), code=code)
    AS.open_assessment(lock, wt)
    try:
        r = AS.outer_unit(D, 0, "RAW-J|b0.3", L["seeds"]["0"]["score"]["RAW-J|b0.3"], units_dir=units)
        rf = AS.outer_unit(D, 0, "F", L["seeds"]["0"]["score"]["F"], units_dir=units)
    finally:
        AS.close_assessment()
    assert (units / "outer__s0__RAW-J_b0.3" / "COMPLETE.json").exists() and r["label"] == "RAW-J|b0.3"
    a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    z = np.load(units / "outer__s0__RAW-J_b0.3" / "preds.npz")
    assert np.array_equal(z["assess_row_id"], D["row_id"][a]) and np.array_equal(z["assess_unit"], D["unit"][a])
    for w in ("v1", "v2", "pair", "r1", "r2", "rpair", "c1", "c2", "cpair", "p1", "p2", "ppair", "h1", "h2", "hpair"):
        for crit in ("auc", "ce"):
            assert z[f"P_{crit}_{w}"].shape == (3, len(a), 2)
    assert np.array_equal(z["u_hard1"], outU["hard1"][a]) and z["const1"].shape == (len(a),)
    u = r["utility_deployed"]["0"]
    yy = D["y_income"][a]
    acc, accU, cacc = (z["hard1"] == yy).mean(), (z["u_hard1"] == yy).mean(), (z["const1"] == yy).mean()
    assert u["gain_retention"] == pytest.approx(acc - 0.8 * accU - 0.2 * cacc)
    assert u["gain_over_const"] == pytest.approx(acc - cacc)
    for key in ("balanced_accuracy_supported", "minority_recall", "brier", "log_loss"):
        assert u[key] is not None
    race = r["race"]
    assert race["supported_codes"] == [1, 2, 4] and set(race["unsupported_codes"]) == {"0", "3"}
    assert all(v["status"] != "SUPPORTED" for v in race["unsupported_codes"].values())
    assert np.array_equal(D["race"][a][z["race_pos"]], z["race_codes"][z["race_y"]])
    pv = race["per_view"]["pair"]
    assert set(pv["per_class_ovr_auc_mean"]) == {"1", "2", "4"} and pv["contrast_max_minus_min_ovr_auc"] >= 0
    sc = r["primary"]["scored"]["pair"]["auc"]
    assert np.isclose(np.mean([AU.auc_fixed(z["sex"], P) for P in z["P_auc_pair"]]), sc["auc_mean"])
    assert {"coalition", "ignore_recipient_2", "ignore_recipient_1"} <= \
        {b["candidate"] for b in r["primary"]["selection"]["pair"]["bank"]}
    assert set(r["preds_keys"]) == set(z.files)
    assert rf["finite"] and any(x["attacker"].startswith("CC_") for x in rf["primary"]["tables"]["v1"])


def test_tampered_release_and_unknown_spec_are_refused(tmp_path):
    D = osf_D(800)
    neural_unit(D, tmp_path, "rel__s0__x")
    with pytest.raises(SystemExit, match="differ from the hashes"):
        AS.release_for({"kind": "release", "unit": "rel__s0__x"}, D, tmp_path, {"rel__s0__x": {"release.npz": "0" * 64}})
    with pytest.raises(SystemExit, match="unknown release spec"):
        AS.release_for({"kind": "other", "unit": "rel__s0__x"}, D, tmp_path)
    assert AS.safe("NORM-L|r3|a2") == "NORM-L_r3_a2" and AS.safe("N*") == "Nstar"


# ------------------------------------------------------------------ baselines
def test_gates_and_global_fare_rule():
    u = {0: {"acc": 0.85, "const_acc": 0.76}, 1: {"acc": 0.475, "const_acc": 0.28}}
    ur = {0: {"acc": 0.855, "const_acc": 0.76}, 1: {"acc": 0.48, "const_acc": 0.28}}
    ok, worst, g = BL.gates(u, ur)
    assert g[0]["G1"] == pytest.approx(0.005) and g[1]["G2"] == pytest.approx(0.195 - 0.8 * 0.2) and ok
    assert worst == pytest.approx(min(x for gi in g.values() for x in gi.values()))
    u[1]["acc"] = 0.469                                                  # G1 on occupation fails by 0.001
    ok, worst, g = BL.gates(u, ur)
    assert not ok and worst == pytest.approx(-0.001)
    rows = {1: {"gate_ok_all": True, "mean_R_local": 0.7, "min_margin": 0.01},
            2: {"gate_ok_all": True, "mean_R_local": 0.6, "min_margin": 0.0},
            3: {"gate_ok_all": False, "mean_R_local": 0.5, "min_margin": -0.01}}
    assert BL.pick_global(rows) == {"status": "SELECTED", "config": 2}
    s = BL.pick_global({c: dict(r, gate_ok_all=False) for c, r in rows.items()})
    assert s["status"] == "NO_FEASIBLE_CONFIGURATION" and s["config"] == 1
    assert [c["id"] for c in BL.FARE_GRID] == [1, 2, 3, 4, 5, 6]


def test_partition_alias_detection():
    a = np.array([0, 0, 1, 2, 2, 1])
    assert BL.partition_equal(a, np.array([5, 5, 3, 0, 0, 3]))            # relabelled: alias
    assert not BL.partition_equal(a, np.array([0, 1, 1, 2, 2, 1]))         # a split cell: not an alias


def _smf_U(D, root, k=0):
    import joblib
    import torch

    from jcv.train import Model
    model = Model(83, [2, 6], k)
    out, heads, _ = FN.finalize_model(model, D)
    st = model.state_dict()
    files = {"model.pt": lambda p: torch.save(st, p),
             "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out)}
    for i, h in heads.items():
        files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
    FN.save_unit(Path(root) / f"tl__s{k}__e40", files, {"unit": f"tl__s{k}__e40"})


def test_leace_admission_verifies_receipts_and_refits_when_they_fail(tmp_path):
    from smf import baselines as SB
    D = osf_D(1800, 3, sealed=True)
    smf_root = tmp_path / "smf"
    _smf_U(D, smf_root)
    Ds = dict(D, idx=dict(D["idx"], NEW_DEFENSE_FIT=D["idx"]["DEFENSE_FIT"]))
    SB.leace_unit(0, "tl__s0__e40", Ds, smf_root)                         # predecessor maps (smf code, smf format)
    r = BL.leace_unit(0, D, tmp_path / "a", smf_units=smf_root)
    assert r["decision"] == "ADMITTED_SMF_MAPS" and not r["admission"]["failed"]
    assert all(v == 0.0 for v in r["replay_vs_smf_release_max_abs_diff"].values())
    assert r["native_check_status"] == {"0": "WITHIN_TOLERANCE", "1": "WITHIN_TOLERANCE"}
    z = np.load(tmp_path / "a" / "lc__s0__E" / "release.npz")
    assert np.array_equal(z["row_id"], D["row_id"]) and z["r1"].shape[0] == len(D["row_id"])   # every row released
    # a different OSF_DEFENSE_FIT preprocessing changes U's fitting features: H_fit receipt fails -> refit
    D2 = dict(D, X=D["X"].copy())
    D2["X"][D["idx"]["OSF_DEFENSE_FIT"][:10], 0] += 0.5
    r2 = BL.leace_unit(0, D2, tmp_path / "b", smf_units=smf_root)
    assert r2["decision"] == "REFIT_ON_OSF_DEFENSE_FIT" and any("H_fit" in f for f in r2["admission"]["failed"])
    assert r2["native_check_status"] == {"0": "WITHIN_TOLERANCE", "1": "WITHIN_TOLERANCE"}


def _fare_available():
    from oar import fare_official as FO
    return FO.fare_python().exists()


@pytest.mark.skipif(not _fare_available(), reason="FARE environment not installed")
def test_fare_tree_admission_encodes_every_row_and_refits_on_receipt_failure(tmp_path):
    from smf import baselines as SB
    D = osf_D(3000, 5, sealed=True)
    Ds = dict(D, idx=dict(D["idx"], NEW_DEFENSE_FIT=D["idx"]["DEFENSE_FIT"]))
    smf_units, smf_cache = tmp_path / "smf_units", tmp_path / "smf_cache"
    for k in (0, 1):
        SB.fare_fit(k, 0, 3, Ds, False, smf_units, smf_cache, synthetic=True)
    for k in (0, 1):
        r = BL.fare_unit(k, 0, 3, D, units_dir=tmp_path / "u", smf_cache=smf_cache, smf_units=smf_units,
                         synthetic=True)
        p = r["provenance"]
        assert p["decision"] == "ADMITTED_SMF_TREE" and not p["admission"]["failed"]
        assert p["encode_cross_check"]["official_vs_portable_mismatches"] == 0
        assert p["encode_cross_check"]["vs_smf_cells_mismatches"] == 0
    a1 = BL._alias_of("fare__s1__p0__c3", D, tmp_path / "u")
    z0 = np.load(tmp_path / "u" / "fare__s0__p0__c3" / "release.npz")
    z1 = np.load(tmp_path / "u" / "fare__s1__p0__c3" / "release.npz")
    assert (a1 == "fare__s0__p0__c3") == BL.partition_equal(z0["cells"], z1["cells"])
    # a changed fit label breaks the label receipt -> a NEW official fit on OSF_DEFENSE_FIT (synthetic auth)
    D2 = dict(D, sex=D["sex"].copy())
    D2["sex"][D["idx"]["OSF_DEFENSE_FIT"][:20]] ^= 1
    r2 = BL.fare_unit(0, 0, 3, D2, units_dir=tmp_path / "v", smf_cache=smf_cache, smf_units=smf_units,
                      fare_cache=tmp_path / "osf_cache", synthetic=True)
    assert r2["provenance"]["decision"] == "REFIT_ON_OSF_DEFENSE_FIT"
    assert "fit_labels_sha256" in r2["provenance"]["admission_failed"]["failed"]


def test_reference_units_and_inner_audits_are_deterministic_and_resumable(tmp_path):
    """Re-running the reference path from scratch reproduces every array/file byte for byte (records differ only in
    timing fields); a complete unit is skipped on resume, and an interrupted write (.tmp) is replaced."""
    from smf import baselines as SB
    D = osf_D(1800, 3, sealed=True)
    smf_root = tmp_path / "smf"
    _smf_U(D, smf_root)
    SB.leace_unit(0, "tl__s0__e40", dict(D, idx=dict(D["idx"], NEW_DEFENSE_FIT=D["idx"]["DEFENSE_FIT"])), smf_root)
    for run in ("a", "b"):
        r = BL.leace_unit(0, D, tmp_path / run, smf_units=smf_root)
        OA.inner_unit(r["unit"], D, tmp_path / run)
    for u in ("lc__s0__E", "inner__lc__s0__E"):
        c = BL.compare_unit_dirs(tmp_path / "a" / u, tmp_path / "b" / u)
        assert c["identical"], c
    rec = json.loads((tmp_path / "b" / "lc__s0__E" / "record.json").read_text())
    rec["native_check_status"]["0"] = "OUTSIDE_TOLERANCE"
    (tmp_path / "b" / "lc__s0__E" / "record.json").write_text(json.dumps(rec))
    assert not BL.compare_unit_dirs(tmp_path / "a" / "lc__s0__E", tmp_path / "b" / "lc__s0__E")["identical"]
    before = (tmp_path / "a" / "lc__s0__E" / "COMPLETE.json").read_bytes()
    (tmp_path / "a" / "lc__s1__E.tmp").mkdir()                           # debris of an interrupted write
    assert BL.leace_unit(0, D, tmp_path / "a", smf_units=smf_root)["unit"] == "lc__s0__E"
    assert (tmp_path / "a" / "lc__s0__E" / "COMPLETE.json").read_bytes() == before   # resumed: skipped, unchanged
