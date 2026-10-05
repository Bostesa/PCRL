"""Tests for the independent audit, the outer writer gate and the official baselines (synthetic data only).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest smf/tests/test_audit.py -q
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from jcv import audit as JA
from rgj import finalize as FN
from smf import audit as AU


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
    v1 = np.column_stack([2 * a - 1 + 0.05 * rng.normal(size=n), rng.normal(size=(n, 3))])
    v2 = np.column_stack([2 * b - 1 + 0.05 * rng.normal(size=n), rng.normal(size=(n, 3))])
    D, _ = synth_D(n, seed + 1)
    D["sex"] = a ^ b
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2])}, D


def leaky_local_views(n=3000, seed=0):
    """v1 reads SEX through a noisy coordinate, v2 is pure noise."""
    D, rng = synth_D(n, seed)
    s = D["sex"]
    v1 = np.column_stack([s + rng.normal(scale=0.8, size=n), rng.normal(size=(n, 3))])
    v2 = rng.normal(size=(n, 4))
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2])}, D


def _non_da_max(table):
    return max(v for k, v in table.items() if not k.startswith("DA"))


# ------------------------------------------------------------------ coalition bank and dual selection
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
    assert [b["inner_auc"] for b in bank if b["candidate"] == "ignore_recipient_2"] == \
        [x["inner_auc"] for x in r["tables"]["v1"]]
    ces = np.array([b["inner_ce"] for b in bank])
    assert r["selection"]["pair"]["ce"]["bank_index"] == int(np.argmin(ces))
    assert "DA_LR" in r["slate_members"]


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
    V = {"v1": x[:, None], "v2": rng.normal(size=(len(s), 1)),
         "pair": np.column_stack([x + rng.normal(scale=1.5, size=len(s)), x])}   # pair's first column is noisier
    rec, _ = AU.select_bank(V, s, D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"], _fixed_slate)
    t = {r["attacker"]: r for r in rec["tables"]["v1"]}
    assert t["SHARP"]["inner_auc"] > t["SOFT"]["inner_auc"] and t["SHARP"]["inner_ce"] > t["SOFT"]["inner_ce"]
    assert rec["selection"]["v1"]["auc"]["attacker"] == "SHARP"
    assert rec["selection"]["v1"]["ce"]["attacker"] == "SOFT"
    pa = rec["selection"]["pair"]["auc"]
    assert (pa["candidate"], pa["view"], pa["attacker"]) == ("ignore_recipient_2", "v1", "SHARP")
    assert pa["inner_auc"] == t["SHARP"]["inner_auc"]


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
    with pytest.raises(AssertionError):
        AU.final_audit(V, D["sex"], f, v, v, slate_fn=AU.inner_slate)      # scoring on selection rows is refused


def test_final_slate_members_and_multiclass_secondary_audit():
    names = [nm for nm, _ in AU.final_slate(False)]
    assert names.count("DA_LR") == 1 and names.count("DA_MLP") == 1 and "DA_canonical_MLP" not in names
    assert sum(nm.startswith("LR_") for nm in names) == 5 and sum(nm.startswith("MLP_") for nm in names) == 4
    assert sum(nm.startswith("HGB_") for nm in names) == 4
    assert [nm for nm, _ in AU.final_slate(True)][-3:] == ["CC_alpha0.1", "CC_alpha1.0", "CC_alpha10.0"]
    D, rng = synth_D(3000, 13)
    y = rng.integers(0, 3, len(D["sex"]))
    X = np.column_stack([np.eye(3)[y] + rng.normal(scale=0.7, size=(len(y), 3)), rng.normal(size=(len(y), 2))])
    V = {"v1": X[:, :3], "v2": X[:, 3:], "pair": X}
    rec, P = AU.final_audit(V, y, D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"], D["idx"]["SCORE"],
                            slate_fn=AU.secondary_slate, K=3, classes=[0, 1, 2])
    assert P["auc_pair"].shape == (3, len(D["idx"]["SCORE"]), 3)
    assert rec["scored"]["pair"]["auc"]["auc_mean"] > 0.7
    assert rec["scored"]["v2"]["auc"]["auc_mean"] < 0.58


# ------------------------------------------------------------------ the fixed defense-aware reader
def test_rotated_1e6_clue_survives_serialisation_and_transform_and_is_detected(tmp_path):
    D, z = AU.synthetic_D(n_fit=3000, n_sel=1500, seed=5)
    rp = AU.rotated_plant(z, D, slate="inner", recipients=(1, 2), workdir=tmp_path)
    assert rp["all_ok"], rp
    for i in ("1", "2"):
        x = rp["recipients"][i]
        w = f"v{i}"
        assert x["serialisation"]["reread_arrays_identical"] and x["serialisation"]["reread_hash_complete"]
        assert x["ok"] and x[w]["heldout_auc_B"] > AU.PLANT_MIN and x["pair"]["heldout_auc_B"] > AU.PLANT_MIN
        assert x[w]["selected_on_A"].endswith(":DA_LR")
        assert _non_da_max(x[w]["per_attacker_auc_B"]) < 0.6          # rotated: scale-blind members see nothing
        rc = x["canon_receipt"][w]
        assert rc["min_kept_over_s1"] < 1e-5 < 1 and rc["tol_over_s1"] < 1e-10      # the clue direction is kept
        assert x["plant"]["max_coordinate_share_of_clue_direction"] < 0.9          # genuinely rotated
    assert not list(tmp_path.iterdir())                                            # temporary release removed


def test_predecessor_reader_drops_the_rotated_clue_and_the_fixed_reader_keeps_it():
    D, z = AU.synthetic_D(n_fit=3000, n_sel=1500, seed=6)
    S = D["sex"]
    zp, _ = AU.planted_release(z, S, 1)
    X = FN.views_from_release(zp)["v1"]                                 # 17 genuine directions + 2 exact nulls
    f, v = D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"]
    old, new = JA.Canon(0).fit(X[f]), AU.Canon().fit(X[f])
    assert int(old.keep.sum()) == 16 and new.rank_ == 17
    a_old = AU.auc_fixed(S[v], AU.proba(JA._da(0).fit(X[f], S[f]), X[v], 2))
    a_new = AU.auc_fixed(S[v], AU.proba(AU.da_lr(0).fit(X[f], S[f]), X[v], 2))
    assert a_old < 0.6 < AU.PLANT_MIN < a_new, (a_old, a_new)


def test_rotated_clue_three_orders_below_the_requirement_is_still_kept():
    D, z = AU.synthetic_D(n_fit=3000, n_sel=1500, seed=8)
    S = D["sex"]
    zp, _ = AU.planted_release(z, S, 1, amp=1e-9)
    X = FN.views_from_release(zp)["v1"]
    f, v = D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"]
    assert AU.Canon().fit(X[f]).rank_ == 17
    assert AU.auc_fixed(S[v], AU.proba(AU.da_lr(0).fit(X[f], S[f]), X[v], 2)) > 0.8


def _nulled_view(offset, n=4500):
    rng = np.random.default_rng(31)
    D, _ = synth_D(n, 32, extra_roles=())
    s = D["sex"]
    u = rng.normal(size=8)
    u /= np.linalg.norm(u)
    R = rng.normal(size=(n, 8)) * np.exp(rng.uniform(-2, 1, 8)) + 3.0 * (s - 0.5)[:, None] * u + offset
    R = R - np.outer(R @ u, u)                                   # exact removal: residue along u is rounding only
    L = R @ rng.normal(size=(8, 6)) + rng.normal(size=6)
    return np.hstack([R, L - L.mean(1, keepdims=True)]), D, u    # rank 7 = 8 - 1 (projection); logits add none


def test_exact_null_directions_are_dropped_and_not_amplified():
    """SEX lives only in a direction the 'defense' removed exactly (float64 projection, as LEACE collapses); centred
    affine logits add more exact nulls. The reader keeps the true rank, with a wide margin even for large uncentred
    offsets (rounding scales with the uncentred magnitude), and finds no SEX signal."""
    for offset in (0.0, 40.0, 1000.0):
        X, D, u = _nulled_view(offset)
        f = D["idx"]["AUDIT_FIT"]
        c = AU.Canon().fit(X[f])
        rc = c.receipt()
        assert c.rank_ == 7, (offset, rc)
        assert rc["tol_over_s1"] > 100 * rc["max_dropped_over_s1"], (offset, rc)
        assert rc["min_kept_over_s1"] > 1e6 * rc["tol_over_s1"], (offset, rc)
        assert np.abs(np.r_[u, np.zeros(6)] @ c.W_).max() < 1e-8                  # no kept weight on u
    X, D, u = _nulled_view(40.0)
    s, f, v = D["sex"], D["idx"]["AUDIT_FIT"], D["idx"]["INNER_SELECTION"]
    thr = 0.5 + AU.NULL_Z * AU.null_sd(s[v])
    for fac in (AU.da_lr, AU.da_mlp):
        auc = AU.auc_fixed(s[v], AU.proba(fac(0).fit(X[f], s[f]), X[v], 2))
        assert auc <= thr, (fac.__name__, auc)
    # the same view under a zero tolerance would whiten the rounding residue to unit variance (what is avoided)
    c0 = AU.Canon(tol_mult=0.0).fit(X[f])
    assert c0.rank_ > 7
    assert c0.s_[-1] / c0.s_[0] < 1e-14 and 0.5 < c0.transform(X[f])[:, -1].std(ddof=1) < 2.0   # >1e14 x gain


def test_rank_tolerance_matches_the_documented_formula():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 6)) + 5.0
    tol, s, scale = AU.rank_tolerance(X)
    assert np.isclose(scale, max(s[0], np.linalg.norm(X, 2)))
    assert np.isclose(tol, max(X.shape) * np.finfo(np.float64).eps * scale)
    assert AU.Canon().fit(X).tol_ == pytest.approx(tol)


# ------------------------------------------------------------------ controls: split null and planted leaks
def test_null_split_halves_are_group_disjoint_and_cover_inner_selection():
    D, _ = synth_D(2000, 9, extra_roles=())
    D["unit"] = np.arange(2000) // 2                                    # exact-record groups of two rows
    a, b = AU.null_split(D)
    sel = D["idx"]["INNER_SELECTION"]
    assert not np.intersect1d(a, b).size and np.array_equal(np.sort(np.r_[a, b]), np.arange(len(sel)))
    assert not np.intersect1d(D["unit"][sel[a]], D["unit"][sel[b]]).size
    assert 0.4 < len(a) / len(sel) < 0.6
    a2, b2 = AU.null_split(D)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)


def test_null_permutation_is_frozen_and_within_each_block():
    D, _ = synth_D(1200, 2)
    S = D["sex"]
    h = AU.null_split(D)
    a, ha = AU.frozen_permutation(S, D, h)
    b, hb = AU.frozen_permutation(S, D, h)
    assert ha == hb and np.array_equal(a, b)
    sel = D["idx"]["INNER_SELECTION"]
    for ix in (D["idx"]["AUDIT_FIT"], sel[h[0]], sel[h[1]]):
        assert np.bincount(a[ix], minlength=2).tolist() == np.bincount(S[ix], minlength=2).tolist()
    other = D["idx"]["SCORE"]
    assert np.array_equal(a[other], S[other])


def test_null_calibration_on_synthetic_null_releases():
    res = AU.calibrate_null(reps=6, slate="inner", n_fit=1500, n_sel=900, seed0=500)
    sm = res["summary"]
    assert sm["tests"] == 18 and len(res["rows"]) == 18
    assert sm["exceedances"] == 0
    assert abs(sm["heldout_z_mean"]) < 1.0
    assert sm["bank_max_full_mean"] > sm["heldout_mean"]          # the bank maximum is selection-optimistic
    assert all(r["threshold"] == pytest.approx(0.5 + AU.NULL_Z * r["sd0"]) for r in res["rows"])


def test_controls_on_a_leaky_release_null_passes_and_plants_are_detected():
    D, z = AU.synthetic_D(n_fit=3000, n_sel=1500, seed=11)
    z["r1"][:, 0] += 1.5 * D["sex"]                                     # the release leaks real SEX
    V = FN.views_from_release(z)
    c = AU.controls(V, D, slate="inner")
    for w, d in c["views"].items():
        assert d["null_ok"] and d["plant_ok"], (w, d)
        assert d["PLANT_ONEHOT_heldout_auc_B"] > AU.PLANT_MIN and d["PLANT_TINY_heldout_auc_B"] > AU.PLANT_MIN
    assert c["all_ok"] and not c["failures"]
    assert c["views"]["pair"]["PLANT_ONEHOT_selected_on_A"].startswith("coalition:pair:")
    assert c["thresholds"]["null_max_heldout_auc"] == pytest.approx(0.5 + AU.NULL_Z * c["thresholds"]["null_sd_B"])
    assert c["split"]["rows_A"] + c["split"]["rows_B"] == len(D["idx"]["INNER_SELECTION"])


def test_null_control_fails_and_is_preserved_when_the_view_encodes_the_shuffled_label():
    D, z = AU.synthetic_D(n_fit=3000, n_sel=1500, seed=13)
    Sp, _ = AU.frozen_permutation(D["sex"], D, AU.null_split(D))
    rng = np.random.default_rng(14)
    V = FN.views_from_release(z)
    V["v1"] = np.hstack([V["v1"], (Sp + rng.normal(scale=0.7, size=len(Sp)))[:, None]])   # a broken release/slate
    V["pair"] = np.hstack([V["v1"], V["v2"]])
    c = AU.controls(V, D, slate="inner")
    assert not c["views"]["v1"]["null_ok"] and "v1:null_ok" in c["failures"] and not c["all_ok"]
    assert c["views"]["v2"]["null_ok"]


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
    D, R1, R2 = _head_D()
    out, heads, _ = FN.heads_and_outputs([R1, R2], D)
    for i, R in ((0, R1), (1, R2)):
        cen, P, hard = FN.outputs(heads[i], R)
        assert np.array_equal(cen, out[f"c{i + 1}"]) and np.array_equal(hard, out[f"hard{i + 1}"])
        assert np.allclose(cen.sum(1), 0)
    out2, _, _ = FN.heads_and_outputs([R1, R2 + 5.0 * np.random.default_rng(1).normal(size=R2.shape)], D)
    assert np.array_equal(out2["c1"], out["c1"]) and np.array_equal(out2["r1"], out["r1"])
    for i, R in ((0, R1), (1, R2)):
        A = np.hstack([R, np.ones((len(R), 1))])
        coef, *_ = np.linalg.lstsq(A, out[f"c{i + 1}"], rcond=None)
        assert np.max(np.abs(A @ coef - out[f"c{i + 1}"])) < 1e-8
    V = FN.views_from_release(out)
    assert np.array_equal(V["v1"], np.hstack([out["r1"], out["c1"]]))
    assert np.array_equal(V["pair"], np.hstack([V["v1"], V["v2"]]))
    # the reader sees exactly the logits' affine redundancy: rank 16 of 18 (binary) and 16 of 22 (six classes)
    assert AU.Canon().fit(V["v1"][D["idx"]["DEFENSE_FIT"]]).rank_ == 16
    assert AU.Canon().fit(V["v2"][D["idx"]["DEFENSE_FIT"]]).rank_ == 16


# ------------------------------------------------------------------ outer writer gate (EVALUATION_LOCK.json)
def _git(cwd, *args):
    r = subprocess.run(["git", "-c", "user.name=smf-test", "-c", "user.email=", *args], cwd=cwd, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _lock_repo(tmp_path, push=True, content='{"seeds": {}}', rel=None):
    from smf import assess as AS
    rel = rel or AS.LOCK_REL
    origin, wt = tmp_path / "origin.git", tmp_path / "wt"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(tmp_path, "clone", "-q", str(origin), str(wt))
    _git(wt, "checkout", "-q", "-b", AS.STUDY_BRANCH)
    lock = wt / rel
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(content)
    _git(wt, "add", rel)
    _git(wt, "commit", "-q", "-m", "lock")
    if push:
        _git(wt, "push", "-q", "-u", "origin", AS.STUDY_BRANCH)
    return wt, lock


def test_outer_writer_refuses_without_an_opened_pushed_lock(tmp_path):
    from smf import assess as AS
    B = AS.STUDY_BRANCH
    AS.close_assessment()
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.outer_unit({"sealed": False}, 0, "J-F", {"kind": "neural", "unit": "x"}, units_dir=tmp_path)
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.load_unsealed()
    wt, lock = _lock_repo(tmp_path, push=False)
    assert AS.lock_is_pushed(lock, wt, B)["reason"].startswith("lock commit")             # committed, not pushed
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.open_assessment(lock, wt, B)
    _git(wt, "push", "-q", "-u", "origin", B)
    assert AS.lock_is_pushed(lock, wt, B)["ok"]
    lock.write_text('{"seeds": {"0": {}}}')                                               # edited after the push
    assert not AS.lock_is_pushed(lock, wt, B)["ok"]
    with pytest.raises(SystemExit, match="REFUSED"):
        AS.open_assessment(lock, wt, B)
    _git(wt, "checkout", "-q", "--", AS.LOCK_REL)
    assert AS.lock_is_pushed(lock, wt, B)["ok"]
    assert not AS.lock_is_pushed(lock, wt, "some-other-branch")["ok"]                     # wrong branch
    other = lock.parent / "OTHER_LOCK.json"
    other.write_text("{}")
    assert not AS.lock_is_pushed(other, wt, B)["ok"]                                      # wrong file name
    stray = wt / "EVALUATION_LOCK.json"                                                    # right name, wrong place
    stray.write_text("{}")
    _git(wt, "add", "EVALUATION_LOCK.json")
    _git(wt, "commit", "-q", "-m", "stray")
    _git(wt, "push", "-q")
    assert "registered" in AS.lock_is_pushed(stray, wt, B)["reason"]
    loose = tmp_path / "loose" / "EVALUATION_LOCK.json"
    loose.parent.mkdir()
    loose.write_text("{}")
    assert not AS.lock_is_pushed(loose, wt, B)["ok"]                                      # outside the repository


def test_open_assessment_is_revoked_when_the_lock_changes_and_sealed_D_is_refused(tmp_path):
    from smf import assess as AS
    wt, lock = _lock_repo(tmp_path)
    AS.open_assessment(lock, wt, AS.STUDY_BRANCH)
    try:
        with pytest.raises(SystemExit, match="REFUSED: D is sealed"):
            AS.outer_unit({"sealed": True}, 0, "J-F", {"kind": "neural", "unit": "x"}, units_dir=tmp_path)
        lock.write_text('{"changed": true}')
        with pytest.raises(SystemExit, match="REFUSED"):
            AS._require_open()
    finally:
        AS.close_assessment()


def test_locked_code_hash_mismatch_refuses(tmp_path):
    from smf import assess as AS
    wt, lock = _lock_repo(tmp_path, content=json.dumps({"seeds": {}, "locked_code_files": {"smf/audit.py": "0" * 64}}))
    with pytest.raises(SystemExit, match="locked code hash"):
        AS.open_assessment(lock, wt, AS.STUDY_BRANCH)
    AS.close_assessment()


# ------------------------------------------------------------------ synthetic study-shaped data
def study_D(n=2600, seed=0, sealed=False):
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
         "sex": sex, "race": race, "y_income": y_inc, "y_occupation_group": y_occ, "sealed": False}
    D["idx"] = {r: np.flatnonzero(role == r) for r in np.unique(roles)}
    D["idx"]["NEW_DEFENSE_FIT"] = D["idx"]["DEFENSE_FIT"]
    D["idx"]["NEW_DEVELOPMENT_ASSESSMENT"] = D["idx"]["DEVELOPMENT_ASSESSMENT"]
    if sealed:
        a = D["idx"]["DEVELOPMENT_ASSESSMENT"]
        for k in ("sex", "race", "y_income", "y_occupation_group"):
            D[k] = D[k].copy()
            D[k][a] = -1
        D["sealed"] = True
    D["y"] = {"income": D["y_income"], "occupation_group": D["y_occupation_group"]}
    return D


def _neural_unit(D, units, name, seed=1):
    import joblib
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
    from smf import assess as AS
    from smf import baselines as BL
    D = study_D()
    units = tmp_path / "units"
    _neural_unit(D, units, "ck__test")
    for i in (0, 1):
        cells = (D["X"][:, i + 2] > 0).astype(np.int64) + 2 * (D["X"][:, 0] > 0.7)
        BL.write_fare_unit(f"fare__s0__p{i}__c9", 0, i, cells, 4, {"id": 9}, D, {"origin": "synthetic test"}, units)
    files = json.loads((units / "ck__test" / "COMPLETE.json").read_text())["files"]
    wt, lock = _lock_repo(tmp_path, content=json.dumps({"seeds": {"0": {"unit_file_sha256": {"ck__test": files}}}}))
    AS.open_assessment(lock, wt, AS.STUDY_BRANCH)
    try:
        r = AS.outer_unit(D, 0, "C*", {"kind": "neural", "unit": "ck__test"}, units_dir=units)
        rf = AS.outer_unit(D, 0, "F", {"kind": "fare", "units": ["fare__s0__p0__c9", "fare__s0__p1__c9"]},
                           units_dir=units)
    finally:
        AS.close_assessment()
    assert (units / "outer__s0__Cstar" / "COMPLETE.json").exists() and r["label"] == "C*"
    assert r["release_provenance"]["ck__test"]["matches_lock_unit_hashes"] is True
    a = D["idx"]["DEVELOPMENT_ASSESSMENT"]
    z = np.load(units / "outer__s0__Cstar" / "preds.npz")
    assert np.array_equal(z["assess_row_id"], D["row_id"][a]) and np.array_equal(z["sex"], D["sex"][a])
    for w in ("v1", "v2", "pair", "p1", "p2", "ppair", "h1", "h2", "hpair"):
        for crit in ("auc", "ce"):
            assert z[f"P_{crit}_{w}"].shape == (3, len(a), 2)
    assert list(z["race_codes"]) == r["race"]["supported_codes"] == [1, 2, 4]
    assert z["Prace_auc_pair"].shape == (3, len(z["race_pos"]), 3)
    assert np.array_equal(D["race"][a][z["race_pos"]], z["race_codes"][z["race_y"]])
    sc = r["primary"]["scored"]["pair"]["auc"]
    assert np.isclose(np.mean([AU.auc_fixed(z["sex"], P) for P in z["P_auc_pair"]]), sc["auc_mean"])
    assert r["primary"]["scored"]["v1"]["auc"]["auc_mean"] > 0.6
    assert {"coalition", "ignore_recipient_2", "ignore_recipient_1"} <= \
        {b["candidate"] for b in r["primary"]["selection"]["pair"]["bank"]}
    assert set(r["preds_keys"]) == set(z.files)
    assert rf["native"]["fare_certificate"][0]["status"] == "NOT_AVAILABLE_UNDER_NEW_ROLES"
    assert any(row["attacker"].startswith("CC_") for row in rf["primary"]["tables"]["v1"])


def test_tampered_release_unit_is_refused_against_lock_hashes(tmp_path):
    from smf import assess as AS
    D = study_D(n=800)
    units = tmp_path / "units"
    _neural_unit(D, units, "ck__x")
    with pytest.raises(SystemExit, match="differ from the hashes"):
        AS.release_for({"kind": "neural", "unit": "ck__x"}, D, units, {"ck__x": {"release.npz": "0" * 64}})


# ------------------------------------------------------------------ sealed labels
def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in ("fit_s", "wall_s")}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def test_inner_audit_and_controls_never_read_assessment_labels():
    Ds, Dr = study_D(sealed=True), study_D()
    Dr["sex"] = Dr["sex"].copy()
    Dr["sex"][Dr["idx"]["DEVELOPMENT_ASSESSMENT"]] = 1 - Dr["sex"][Dr["idx"]["DEVELOPMENT_ASSESSMENT"]]   # other labels
    X = Ds["X"].astype(np.float64)
    V = {"v1": X[:, :6], "v2": X[:, 6:12], "pair": X[:, :12]}
    assert _strip(AU.inner_audit(V, Ds)) == _strip(AU.inner_audit(V, Dr))
    assert _strip(AU.controls(V, Ds, slate="inner")) == _strip(AU.controls(V, Dr, slate="inner"))


def test_fitting_and_scoring_entry_points_refuse_sealed_labels():
    D = study_D(sealed=True)
    X = D["X"].astype(np.float64)
    V = {"v1": X[:, :6], "v2": X[:, 6:12], "pair": X[:, :12]}
    idx = D["idx"]
    with pytest.raises(ValueError, match="sealed"):
        AU.final_audit(V, D["sex"], idx["AUDIT_FIT"], idx["INNER_SELECTION"], idx["DEVELOPMENT_ASSESSMENT"],
                       slate_fn=AU.inner_slate)
    with pytest.raises(ValueError, match="sealed"):           # a caller pointing a fit role at assessment rows
        AU.select_bank(V, D["sex"], idx["DEVELOPMENT_ASSESSMENT"], idx["INNER_SELECTION"], AU.inner_slate)
    Dbad = dict(D, idx=dict(idx, AUDIT_FIT=idx["DEVELOPMENT_ASSESSMENT"]))
    with pytest.raises(ValueError, match="sealed"):
        AU.inner_audit(V, Dbad)


# ------------------------------------------------------------------ baselines
def test_gate_definitions_match_the_predecessor_selection():
    from rgj.select import gate_margins
    from smf import baselines as BL
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = {i: {"acc": rng.uniform(.3, .9), "const_acc": rng.uniform(.2, .8)} for i in (0, 1)}
        ur = {i: {"acc": rng.uniform(.3, .9), "const_acc": u[i]["const_acc"]} for i in (0, 1)}
        gm = gate_margins(u, ur)
        assert all(BL.gate_margins(u, ur)[i] == pytest.approx(gm[i]) for i in (0, 1))
        for i in (0, 1):
            ok, worst, g = BL.gate_one(u[i], ur[i])
            assert g == pytest.approx(gm[i]) and ok == all(x >= 0 for x in gm[i].values())
    try:
        from smf import select as SS
    except ImportError:
        return
    if hasattr(SS, "gate_margins"):
        u = {i: {"acc": 0.8, "const_acc": 0.6} for i in (0, 1)}
        ur = {i: {"acc": 0.82, "const_acc": 0.6} for i in (0, 1)}
        assert all(SS.gate_margins(u, ur)[i] == pytest.approx(BL.gate_margins(u, ur)[i]) for i in (0, 1))


def test_fare_selection_rule_matches_predecessor():
    from smf import baselines as BL
    rows = [{"config": 1, "gate_ok": True, "R_local": 0.70, "margin": 0.02},
            {"config": 2, "gate_ok": True, "R_local": 0.60, "margin": 0.01},
            {"config": 3, "gate_ok": True, "R_local": 0.60, "margin": 0.00},
            {"config": 4, "gate_ok": False, "R_local": 0.50, "margin": -0.01}]
    s = BL.pick_fare_config(rows)
    assert s["status"] == "NOMINEE" and s["config"] == 2
    s = BL.pick_fare_config([dict(r, gate_ok=False) for r in rows])
    assert s["status"] == "NO_FEASIBLE_NOMINEE" and s["config"] == 1
    assert [c["id"] for c in BL.FARE_GRID] == [1, 2, 3, 4, 5, 6]
    assert BL.CERT_STATUS["status"] == "NOT_AVAILABLE_UNDER_NEW_ROLES"


def _u_unit(D, units, name, seed=0):
    import joblib
    import torch

    from jcv.train import Model
    model = Model(83, [2, 6], seed)
    out, heads, _ = FN.finalize_model(model, D)
    st = model.state_dict()
    files = {"model.pt": lambda p: torch.save(st, p),
             "release.npz": lambda p: np.savez_compressed(p, row_id=D["row_id"], **out)}
    for i, h in heads.items():
        files[f"head_{i}.joblib"] = (lambda p, h=h: joblib.dump(h, p))
    FN.save_unit(Path(units) / name, files, {"unit": name})
    return model, out


def test_leace_unit_official_on_U_features_fit_rows_only_and_blind_to_assessment_labels(tmp_path):
    import torch

    from smf import baselines as BL
    Ds, Dr = study_D(n=1800, seed=3, sealed=True), study_D(n=1800, seed=3)
    recs, zs = [], []
    for D, sub in ((Ds, "a"), (Dr, "b")):
        units = tmp_path / sub
        _u_unit(D, units, "tl__s0__U")
        recs.append(BL.leace_unit(0, "tl__s0__U", D, units))
        zs.append(np.load(units / "lc__s0__E" / "release.npz"))
    rec = recs[0]
    assert rec["native_check_status"] == {"0": "WITHIN_TOLERANCE", "1": "WITHIN_TOLERANCE"}
    assert "official LEACE" in rec["leace"]["0"]["estimator"]
    assert rec["leace"]["0"]["n_fit"] == len(Ds["idx"]["DEFENSE_FIT"]) and rec["n_fit"] == len(Ds["idx"]["DEFENSE_FIT"])
    assert all(np.array_equal(zs[0][k], zs[1][k]) for k in zs[0].files)          # assessment labels never read
    assert set(zs[0].files) == {"row_id", "r1", "c1", "p1", "hard1", "r2", "c2", "p2", "hard2"}
    tr = Ds["idx"]["DEFENSE_FIT"]
    for i in (1, 2):
        R = zs[0][f"r{i}"][tr] - zs[0][f"r{i}"][tr].mean(0)
        s = Ds["sex"][tr] - Ds["sex"][tr].mean()
        assert np.max(np.abs(R.T @ s)) / len(tr) < 1e-8
    # the LEACE-collapsed direction is an exact null that the reader drops
    v1 = FN.views_from_release(zs[0])["v1"][Ds["idx"]["AUDIT_FIT"]]
    assert AU.Canon().fit(v1).rank_ == 15
    assert BL.leace_unit(0, "tl__s0__U", Ds, tmp_path / "a") == rec                  # complete unit is skipped
    # a U whose released features are not its encoder outputs is refused
    units = tmp_path / "a"
    st = torch.load(units / "tl__s0__U" / "model.pt")
    z = dict(np.load(units / "tl__s0__U" / "release.npz"))
    z["r1"] = z["r1"] + 1e-3
    FN.save_unit(units / "tl__s1__U", {"model.pt": lambda p: torch.save(st, p),
                                       "release.npz": lambda p: np.savez_compressed(p, **z)}, {"unit": "tl__s1__U"})
    with pytest.raises(SystemExit, match="REFUSED"):
        BL.leace_unit(1, "tl__s1__U", Ds, units)


def _fare_available():
    from oar import fare_official as FO
    return FO.fare_python().exists()


@pytest.mark.skipif(not _fare_available(), reason="FARE environment not installed")
def test_fare_units_new_official_fits_select_fare_and_sealed_label_blindness(tmp_path):
    from smf import baselines as BL
    grid = [BL.CFG[3], BL.CFG[6]]
    Ds, Dr = study_D(n=4000, seed=5, sealed=True), study_D(n=4000, seed=5)
    out = []
    for D, sub in ((Ds, "a"), (Dr, "b")):
        units, cache = tmp_path / sub / "units", tmp_path / sub / "fare_cache"
        fu = BL.fare_units(0, D, units, fare_cache=cache, synthetic=True, grid=grid)
        assert set(fu) == {f"fare__s0__p{i}__c{c['id']}" for i in (0, 1) for c in grid}
        uref = {0: {"acc": 0.70, "const_acc": 0.5}, 1: {"acc": 0.30, "const_acc": 0.2}}
        sel = BL.select_fare(0, D, units, uref, fare_cache=cache, synthetic=True, grid=grid)
        out.append((units, sel))
    (units, sel), (units_b, sel_b) = out
    rec = json.loads((units / "fare__s0__p0__c3" / "record.json").read_text())
    prov = rec["provenance"]
    assert prov["n_fit"] == len(Ds["idx"]["DEFENSE_FIT"]) and prov["fit_role"].startswith("NEW_DEFENSE_FIT")
    assert prov["origin"].startswith("NEW official FARE fit") and prov["seed"] == 0 and not prov["cache_hit"]
    assert rec["certificate"]["status"] == "NOT_AVAILABLE_UNDER_NEW_ROLES"
    for arm in ("F", "F0"):
        a = sel[arm]
        assert set(a["auc"]) == {"v1", "v2", "pair"} and a["auc"]["pair"] >= max(a["auc"]["v1"], a["auc"]["v2"])
        for key in ("status", "units", "configs", "utility", "gates_ok", "worst_gate_margin"):
            assert key in a
        assert not any("guard" in key for key in a)                                   # no local guard
    assert sel["F0"]["units"] == [f"fare__s0__p{i}__Z{sel['F']['configs'][i]}" for i in (0, 1)]
    assert all((units / u / "COMPLETE.json").exists() for u in sel["F0"]["units"])
    zrec = json.loads((units / sel["F0"]["units"][0] / "record.json").read_text())
    assert float(zrec["config"]["gamma"]) == 0.0 and zrec["arm"] == "F0"
    for i in (0, 1):
        assert {r["config"] for r in sel["per_purpose"][i]["table"]} == {3, 6}
    ok_purposes = all(sel["per_purpose"][i]["status"] == "NOMINEE" for i in (0, 1))
    assert (sel["F"]["status"] == "NOMINEE") == (ok_purposes and sel["F"]["gates_ok"])
    assert (sel["F0"]["status"] == "NOMINEE") == sel["F0"]["gates_ok"]
    # identical releases and selections whether or not the assessment labels are sealed
    for u in sel["F"]["units"] + sel["F0"]["units"]:
        za, zb = np.load(units / u / "release.npz"), np.load(units_b / u / "release.npz")
        assert all(np.array_equal(za[k], zb[k]) for k in za.files)
    assert _strip(sel["F"]["auc"]) == _strip(sel_b["F"]["auc"]) and sel["F"]["configs"] == sel_b["F"]["configs"]


def test_controls_only_mode_runs_split_null_plants_and_rotated_clue_on_neural_and_fare_units(tmp_path):
    from smf import assess as AS
    from smf import baselines as BL
    D = study_D(n=3000, seed=21, sealed=True)                  # controls-only mode never needs assessment labels
    units = tmp_path / "units"
    _neural_unit(D, units, "ck__ctl")
    for i in (0, 1):
        cells = (D["X"][:, i + 2] > 0).astype(np.int64) + 2 * (D["X"][:, 0] > 0.7)
        BL.write_fare_unit(f"fare__s0__p{i}__c9", 0, i, cells, 4, {"id": 9}, D, {"origin": "synthetic test"}, units)
    res = AS.controls_for(D, {"J-F": {"kind": "neural", "unit": "ck__ctl"},
                              "F": {"kind": "fare", "units": ["fare__s0__p0__c9", "fare__s0__p1__c9"]}},
                          units, slate="inner", recipients=(1,))
    for lab in ("J-F", "F"):
        r = res[lab]
        assert r["all_ok"], (lab, r["failures"], r["rotated_plant"]["recipients"]["1"]["ok"])
        assert r["rotated_plant"]["recipients"]["1"]["serialisation"]["reread_arrays_identical"]
        assert set(r["views"]) == {"v1", "v2", "pair"}
    assert res["all_ok"]
