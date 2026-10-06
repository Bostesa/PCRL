"""Synthetic tests of the dpc audit / utility / assessment code (no real data, no labels of any real role).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m pytest -q dpc/tests/test_audit.py
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from dpc import assess as AS
from dpc import audit as A
from dpc import utility as UT

SMALL = {"n_fit": 1500, "n_sel": 800}


def _D(seed=0, n_score=0, **kw):
    return A.synthetic_D(seed=seed, n_score=n_score, **{**SMALL, **kw})


def _policy(D, seed=0, states=(8, 24), leak=0.0):
    return A.synthetic_policy(D, states, seed=seed, leak=leak)


def _renumber(z, seed):
    """Invertible renumbering of both recipients' token IDs (decoder table moves with the IDs)."""
    rng = np.random.default_rng(seed)
    zp = dict(z)
    for i in (1, 2):
        a = int(z[f"alpha{i}"])
        perm = rng.permutation(a)
        zp[f"tok{i}"] = perm[np.asarray(z[f"tok{i}"])]
    return zp


def _split_bit(z, i, bit, collide=True):
    return A.split_tokens(z, i, bit, collide=collide)


# ------------------------------------------------------------------ token identity
def test_renumbering_leaves_views_and_every_prediction_unchanged():
    D = _D(1)
    z = _policy(D, 1, leak=0.8)
    z2 = _renumber(z, 7)
    assert not np.array_equal(z["tok2"], z2["tok2"])
    V1, V2 = A.policy_views(z, D), A.policy_views(z2, D)
    for w in A.PRIMARY_VIEWS:
        assert np.array_equal(V1["X"][w], V2["X"][w]), "design matrix must not depend on token IDs"
    r1, r2 = A.inner_audit(V1, D, slate="inner"), A.inner_audit(V2, D, slate="inner")
    assert r1["inner_predictions"]["keys"] == r2["inner_predictions"]["keys"]
    assert np.array_equal(r1["inner_predictions"]["P"], r2["inner_predictions"]["P"])
    assert r1["auc"] == r2["auc"] and r1["selected"] == r2["selected"]
    assert A.view_fingerprint(V1) == A.view_fingerprint(V2)


def test_cell_readers_are_renumbering_invariant_directly():
    D = _D(2)
    rng = np.random.default_rng(0)
    t = rng.integers(0, 30, len(D["sex"]))
    f, s = A.roles(D)
    p, _ = A.cc_local(t, D["sex"], f, s)
    perm = rng.permutation(30)
    p2, _ = A.cc_local(perm[t], D["sex"], f, s)
    for k in p:
        assert np.array_equal(p[k], p2[k])


def test_decoder_collision_ids_stay_distinct_and_are_detected():
    D = _D(3)
    z = _policy(D, 3)
    bit = A._noisy(D["sex"], 5)                       # carries SEX (80% exact)
    zc = _split_bit(z, 1, bit, collide=True)
    t = zc["tok1"]
    # identical decoded probabilities and decisions for the two copies of a token
    assert np.array_equal(zc["q1"], z["q1"]) and np.array_equal(zc["hard1"], z["hard1"])
    assert len(np.unique(t)) > len(np.unique(z["tok1"]))
    V = A.policy_views(zc, D)
    r = A.inner_audit(V, D, slate="inner")
    assert r["auc"]["v1"] > 0.8, "the token identity must be audited, not only the decoded probabilities"
    qv = {"family": "q_only", "tokens": None, "X": {"v1": zc["q1"], "v2": zc["q2"],
                                                    "pair": np.hstack([zc["q1"], zc["q2"]])}}
    rq = A.inner_audit(qv, D, slate="inner")
    assert rq["auc"]["v1"] < 0.6, "a decoded-probability-only reader cannot see the collision plant"


def test_token_carrying_per_person_state_is_refused():
    D = _D(4)
    z = _policy(D, 4)
    z["q1"] = z["q1"].copy()
    z["q1"][0] = z["q1"][0] * 0.999 + 0.001 * np.eye(2)[z["hard1"][0]]   # token 0's decoded output differs by row
    z["q1"][0] /= z["q1"][0].sum()
    with pytest.raises(ValueError, match="REFUSED"):
        A.policy_views(z, D)


def test_decoder_check_rejects_argmax_mismatch():
    tok = np.array([0, 1, 1])
    q = np.array([[0.6, 0.4], [0.3, 0.7], [0.3, 0.7]])
    assert A.token_decoder_check(tok, q, np.array([0, 1, 1]))["ok"]
    assert not A.token_decoder_check(tok, q, np.array([1, 1, 1]))["ok"]


# ------------------------------------------------------------------ coalition and null fixtures
def test_coalition_positive_fixture_detected_by_the_pair_reader_only():
    D = _D(5)
    z = _policy(D, 5)
    rng = np.random.default_rng(11)
    b1 = rng.integers(0, 2, len(D["sex"]))
    b2 = b1 ^ D["sex"]                                 # each clue alone independent of S; together they reveal it
    zp = _split_bit(_split_bit(z, 1, b1), 2, b2)
    r = A.inner_audit(A.policy_views(zp, D), D, slate="inner")
    sd = A.null_sd(D["sex"][D["idx"][A.SEL_ROLE]])
    assert r["auc"]["v1"] < 0.5 + 4 * sd + 0.02 and r["auc"]["v2"] < 0.5 + 4 * sd + 0.02
    assert r["auc"]["pair"] > 0.95
    assert r["selected"]["pair"].startswith("coalition:pair:")
    assert r["coalition_minus_best_local"] > 0.3


def test_null_fixture_heldout_within_threshold():
    D = _D(6)
    z = _policy(D, 6)                                  # SEX independent of the release
    V = A.policy_views(z, D)
    halves = A.AU.null_split(D)
    Sp, _ = A.AU.frozen_permutation(D["sex"], D, halves)
    thr, _ = A.AU.null_threshold(Sp, D, halves)
    r = A._split_audit(V, Sp, D, halves, "inner")
    for w in A.PRIMARY_VIEWS:
        assert r[w]["heldout_auc_B"] <= thr


# ------------------------------------------------------------------ fallbacks and coverage
def test_unseen_tokens_and_pairs_fall_back_as_declared():
    D = _D(7)
    n = len(D["sex"])
    f, s = A.roles(D)
    rng = np.random.default_rng(3)
    t1 = rng.integers(0, 4, n)
    t2 = rng.integers(0, 4, n)
    t1[s[:50]] = 9                                     # tokens never seen in AUDIT_FIT
    t2[s[50:120]] = 3
    t1[s[50:120]] = 3
    keep = ~((t1[f] == 3) & (t2[f] == 3))              # the (3, 3) pair is absent from AUDIT_FIT
    t1[f[~keep]] = 2
    p, c = A.cc_local(t1, D["sex"], f, s)
    prior = float(np.mean(D["sex"][f] == 1))
    assert np.all(p["CC_alpha1.0"][:50] == prior) and not c["seen"][:50].any()
    sel_pos = np.arange(len(s))
    pp, cp = A.cc_pair(t1, t2, D["sex"], f, s, sel_pos)
    unseen = ~cp["seen"]
    assert unseen[50:120].all() and unseen[:50].all()
    for name, rule in cp["fallback_rule"].items():
        assert rule in A.FALLBACK_RULES
        a = float(name.split("alpha")[1])
        loc1, _ = A.cc_local(t1, D["sex"], f, s, alphas=(a,))
        loc2, _ = A.cc_local(t2, D["sex"], f, s, alphas=(a,))
        ref = {"local_1": loc1[f"CC_alpha{a}"], "local_2": loc2[f"CC_alpha{a}"], "prior": np.full(len(s), prior)}
        assert np.allclose(pp[name][unseen], ref[rule][unseen])
    # fixed rule reproduces exactly on other rows (frozen before scoring)
    pp2, _ = A.cc_pair(t1, t2, D["sex"], f, s, sel_pos, rules=cp["fallback_rule"])
    for name in pp:
        assert np.array_equal(pp[name], pp2[name])


def test_coverage_counts_published():
    D = _D(8)
    z = _policy(D, 8, states=(16, 48))
    r = A.inner_audit(A.policy_views(z, D), D, slate="inner")
    cov = r["coverage"]
    assert set(cov) == {"v1", "v2", "pair"}
    for w in cov:
        assert cov[w]["inner_rows"] == len(D["idx"][A.SEL_ROLE])
        assert cov[w]["inner_rows_seen_in_fit"] + cov[w]["inner_rows_fallback"] == cov[w]["inner_rows"]
    assert set(cov["pair"]["fallback_rule"]) == {f"CCpair_alpha{a}" for a in A.CC_ALPHAS}


# ------------------------------------------------------------------ orientation
def test_auc_orientation_is_fixed_never_flipped():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    assert A.auc1(y, p) == 0.0
    D = _D(9)
    f, s = A.roles(D)
    n = len(D["sex"])
    t = np.zeros(n, dtype=np.int64)
    t[f] = D["sex"][f]                                 # fit rows: token = S
    t[s] = 1 - D["sex"][s]                             # selection rows: relation reversed
    V = A.decision_views(t, t, D)
    r = A.inner_audit(V, D, slate="inner")
    assert r["auc"]["v1"] < 0.05, "an attacker whose relation reverses must score below 0.5 (no flip)"


# ------------------------------------------------------------------ composed readers
def _src_and_policy(D, seed, teacher="U", k=0, m=4):
    """Source probability view in which S is only exposed by a fine partition that a policy code makes explicit."""
    rng = np.random.default_rng(seed)
    n = len(D["sex"])
    p1 = np.clip(rng.random(n), 0.01, 0.99)
    # S shifts p1 into alternating thin bands (hard for LR on p1, easy for a code that indexes the band)
    band = (np.floor(p1 * 40).astype(int) % 2)
    flip = rng.random(n) < 0.15
    D["sex"][:] = np.where(flip, 1 - band, band)
    P1 = np.stack([1 - p1, p1], 1)
    p2 = rng.dirichlet(np.ones(6), n)
    t = {"p1": P1, "p2": p2, "d1": P1.argmax(1), "d2": p2.argmax(1), "c1": np.log(P1) - np.log(P1).mean(1, keepdims=True),
         "c2": np.log(p2) - np.log(p2).mean(1, keepdims=True), "r1": rng.normal(size=(n, 4)),
         "r2": rng.normal(size=(n, 4)), "row_id": D["row_id"]}
    tok1 = np.floor(p1 * 40).astype(np.int64)
    Q1 = np.zeros((40, 2))
    for c in range(40):
        mid = (c + 0.5) / 40
        Q1[c] = [1 - mid, mid]
    hard1 = Q1[tok1].argmax(1)
    t["d1"] = hard1
    tok2 = t["d2"].astype(np.int64)
    Q2 = np.full((6, 6), 0.1) + 0.4 * np.eye(6)
    z = {"row_id": D["row_id"], "tok1": tok1, "q1": Q1[tok1], "hard1": hard1, "alpha1": np.asarray(40),
         "tok2": tok2, "q2": Q2[tok2], "hard2": tok2, "alpha2": np.asarray(6)}
    t["p1"] = np.where(np.isclose(p1, 0.5)[:, None], [[0.49, 0.51]], P1)
    sv = A.source_views(t, D, "probs", meta={"kind": "source", "teacher": teacher, "seed": k, "unit": "src"})
    pv = A.policy_views(z, D, meta={"kind": "policy", "teacher": teacher, "seed": k, "unit": f"pol_{m}", "m": m})
    return sv, pv


def test_composed_inner_reuses_policy_predictions_and_equals_refit_bank():
    D = _D(10)
    sv, pv = _src_and_policy(D, 10)
    rs = A.attach_labels(A.inner_audit(sv, D, slate="inner"), D)
    rp = A.inner_audit(pv, D, slate="inner")
    comp = A.composed_inner(rs, [rp])
    for w in A.PRIMARY_VIEWS:
        assert comp["auc"][w] == pytest.approx(max(rs["auc"][w], rp["auc"][w]), abs=1e-15)
    assert comp["auc"]["v1"] > rs["auc"]["v1"] + 0.01 and comp["winner_is_composed"]["v1"]
    assert comp["selected"]["v1"].startswith("composed[pol_4]:own:v1:")
    refit = A.inner_audit(sv, D, slate="inner", composed=[("pol_4", pv)])
    for w in A.PRIMARY_VIEWS:
        assert refit["auc"][w] == pytest.approx(comp["auc"][w], abs=1e-12)
        assert refit["selected"][w] == comp["selected"][w]


def test_composed_inner_refusals():
    D = _D(11)
    sv, pv = _src_and_policy(D, 11)
    rs = A.inner_audit(sv, D, slate="inner")
    rp = A.inner_audit(pv, D, slate="inner")
    bad = dict(rp, meta={**rp["meta"], "teacher": "RAW-J_b0.3"})
    with pytest.raises(ValueError, match="same teacher"):
        A.composed_inner(rs, [bad])
    bad = dict(rp, meta={**rp["meta"], "seed": 1})
    with pytest.raises(ValueError, match="REFUSED"):
        A.composed_inner(rs, [bad])
    dec = dict(rs, family="decisions")
    with pytest.raises(ValueError, match="decisions-only"):
        A.composed_inner(dec, [rp])
    ok = dict(rp, meta={**rp["meta"], "m": 1})
    A.composed_inner(dec, [ok])                         # a class-only code is a function of the decision
    other = dict(rp, sel_row_id_sha256="0" * 64)
    with pytest.raises(ValueError, match="different INNER_SELECTION rows"):
        A.composed_inner(rs, [other])


# ------------------------------------------------------------------ final audit
def test_final_audit_shapes_roles_and_no_clamp():
    D = _D(12, n_score=600)
    z = _policy(D, 12, leak=0.8)
    V = A.policy_views(z, D)
    a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    rec, P = A.final_audit(V, D, a, slate="inner")
    for crit in ("auc", "ce"):
        for w in A.PRIMARY_VIEWS:
            assert P[f"{crit}_{w}"].shape == (3, len(a), 2)
            assert np.allclose(P[f"{crit}_{w}"].sum(2), 1.0)
    assert set(rec["coverage"]["own"]["pair"]) >= {"scored_rows", "scored_rows_fallback", "fallback_counts_by_alpha"}
    with pytest.raises(ValueError, match="overlap"):
        A.final_audit(V, D, np.concatenate([a, D["idx"][A.SEL_ROLE][:3]]), slate="inner")
    D2 = dict(D, sex=D["sex"].copy())
    D2["sex"][a] = -1
    with pytest.raises(ValueError, match="sealed"):
        A.final_audit(V, D2, a, slate="inner")


def test_final_audit_composed_candidates_join_source_bank():
    D = _D(13, n_score=500)
    sv, pv = _src_and_policy(D, 13)
    a = D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    rec, P = A.final_audit(sv, D, a, slate="inner", composed=[("pol_4", pv)])
    assert rec["scored"]["v1"]["auc"]["candidate"].startswith("composed[pol_4]")
    assert rec["scored"]["v1"]["auc"]["auc_mean"] > 0.7
    assert P["auc_v1"].shape == (3, len(a), 2)


# ------------------------------------------------------------------ plants (synthetic run of the control code)
def test_plant_controls_detect_confidence_collision_and_coalition_leaks():
    D = _D(14)
    z = _policy(D, 14, states=(8, 24))
    V = A.policy_views(z, D)
    res = A.controls_for_release("synthetic", V, D, z_policy=z, slate="inner")
    ch = res["checks"]
    assert all(ch["null"][w]["ok"] for w in A.PRIMARY_VIEWS)
    for p in ("CONF_r1", "CONF_r2", "COLL_r1", "COLL_r2", "XOR"):
        assert ch[p]["ok"], p
        assert ch[p]["serialisation_exact"]
    assert ch["CONF_r1"]["decisions_only_misses_it"] and ch["CONF_r2"]["decisions_only_misses_it"]
    assert ch["COLL_r1"]["decoded_probability_only_audit"]["v1"]["heldout_auc_B"] < 0.6
    assert ch["XOR"]["locals_null_ok"]
    assert res["all_ok"]


def test_confidence_plant_keeps_decisions_and_class():
    D = _D(15)
    z = _policy(D, 15)
    bit = np.random.default_rng(0).integers(0, 2, len(D["sex"]))
    zc = A.split_tokens(z, 2, bit, collide=False)
    assert np.array_equal(zc["hard2"], z["hard2"])
    assert np.array_equal(zc["q2"].argmax(1), z["hard2"])
    assert not np.array_equal(zc["q2"], z["q2"])
    assert A.token_decoder_check(zc["tok2"], zc["q2"], zc["hard2"])["ok"]


# ------------------------------------------------------------------ utility and gates
def test_utility_metric_conventions():
    y = np.array([0, 1, 1, 0])
    P = np.array([[1.0, 0.0], [0.2, 0.8], [0.6, 0.4], [0.5, 0.5]])
    pr = UT.per_row(P, y)
    assert pr["ll"][0] == pytest.approx(0.0)
    P0 = np.array([[0.0, 1.0]])
    assert UT.per_row(P0, np.array([0]))["ll"][0] == pytest.approx(-np.log(1e-12))
    assert pr["br"][1] == pytest.approx(0.2 ** 2 + 0.2 ** 2)
    e, _ = UT.ece(np.array([[0.9, 0.1], [0.9, 0.1]]), np.array([0, 0]), np.array([0, 1]))
    assert e == pytest.approx(0.4)
    m = UT.metrics(P, P.argmax(1), y, 2, const=0)
    assert m["acc"] == pytest.approx(0.75) and m["const_acc"] == pytest.approx(0.5)
    assert m["gain"] == pytest.approx(0.25) and m["logloss"] == pytest.approx(np.mean(pr["ll"]))
    assert m["balanced_acc"] is None and m["recalls"] == {"0": 1.0, "1": 0.5}   # < 30 rows: unsupported


def _util(acc, ll, br, const=0.5):
    return {"acc": acc, "logloss": ll, "brier": br, "const_acc": const}


def test_gate_margins_and_shortfalls():
    u = {"0": _util(0.84, 0.34, 0.22, 0.76), "1": _util(0.475, 1.27, 0.65, 0.28)}
    ok = {"0": _util(0.84, 0.345, 0.224, 0.76), "1": _util(0.475, 1.279, 0.654, 0.28)}
    g = UT.gate(ok, u)
    assert g["pass"]
    bad = {"0": _util(0.84, 0.355, 0.22, 0.76), "1": _util(0.475, 1.27, 0.66, 0.28)}
    g = UT.gate(bad, u)
    assert not g["pass"]
    assert g["tasks"]["0"]["shortfalls"]["G_ll"] == pytest.approx(0.005)
    assert g["tasks"]["1"]["shortfalls"]["G_brier"] == pytest.approx(0.005)
    low = {"0": _util(0.78, 0.34, 0.22, 0.76), "1": _util(0.475, 1.27, 0.65, 0.28)}
    g = UT.gate(low, u)
    assert g["tasks"]["0"]["margins"]["G_gain"] < 0 and g["tasks"]["0"]["margins"]["G_ret"] < 0
    allseeds = UT.gate_all_seeds({0: ok, 1: ok, 2: bad}, {0: u, 1: u, 2: u})
    assert not allseeds["eligible"]
    assert UT.gate_all_seeds({0: ok, 1: ok}, {0: u, 1: u, 2: u})["missing_seeds"] == ["2"]


def test_class_preserved_helper():
    assert UT.class_preserved(np.array([0, 1, 2]), np.array([0, 1, 2]))["ok"]
    assert UT.class_preserved(np.array([0, 1, 2]), np.array([0, 1, 1]))["mismatches"] == 1


# ------------------------------------------------------------------ assessment gate refusals
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


def test_open_assessment_requires_chain_hashes_and_outer_unit_requires_open(repo):
    AS.close_assessment()
    with pytest.raises(SystemExit, match="sealed"):
        AS.outer_unit({"sealed": False}, 0, "x", {"kind": "policy", "cid": "U|CLASS|m1"})
    f = repo / "code.py"
    f.write_text("print(1)\n")
    good = {"seeds": {}, "locked_code_files": {"code.py": AS._sha(f)}}
    p = _write_lock(repo, good)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "lock")
    _git(repo, "push", "-q", "origin", AS.STUDY_BRANCH)
    with pytest.raises(SystemExit, match="does not list"):
        AS.open_assessment(p, repo, chain=("code.py", "other.py"))
    AS.open_assessment(p, repo, chain=("code.py",))
    with pytest.raises(SystemExit, match="D is sealed"):
        AS.outer_unit({"sealed": True}, 0, "x", {"kind": "policy", "cid": "U|CLASS|m1"})
    f.write_text("print(2)\n")
    with pytest.raises(SystemExit, match="differs from the locked code hash"):
        AS.open_assessment(p, repo, chain=("code.py",))
    AS.close_assessment()


def test_unsealing_is_refused_without_an_open_lock():
    AS.close_assessment()
    with pytest.raises(SystemExit, match="sealed"):
        AS.load_unsealed()


def test_policy_unit_name_parsing():
    assert AS._policy_cid_of("pol__s0__RAW-J_b0.3_JOINT_m8_l1") == "RAW-J_b0.3|JOINT|m8|l1"
    assert AS._policy_cid_of("pol__s2__U_CLASS_m1") == "U|CLASS|m1"
    assert A.unit_of(1, "U|FINE-TASK|m4") == "pol__s1__U_FINE-TASK_m4"
    assert A.unit_of(0, "SRC|U") == "tea__s0__U" and A.unit_of(0, "REF|F0") == "ref__s0__F0"
    assert A.parse_cid("RAW-J_b0.3|JOINT|m8|l0.1")["lam"] == 0.1


# ------------------------------------------------------------------ inner_unit contract (synthetic units on disk)
def test_inner_unit_contract_on_synthetic_units(tmp_path):
    from rgj import finalize as FN
    D = _D(16)
    D["idx"]["OSF_DEFENSE_FIT"] = D["idx"][A.FIT_ROLE]          # synthetic: constant from the fit rows
    rng = np.random.default_rng(1)
    n = len(D["sex"])
    D["y"] = {"income": rng.integers(0, 2, n), "occupation_group": rng.integers(0, 6, n)}
    z = _policy(D, 16)
    P1 = np.stack([1 - z["q1"][:, 1] * 0.9, z["q1"][:, 1] * 0.9], 1)
    P1 = np.where(P1.argmax(1)[:, None] == z["hard1"][:, None], P1, z["q1"])
    t = {"row_id": D["row_id"], "p1": P1, "p2": z["q2"], "d1": z["hard1"], "d2": z["hard2"],
         "c1": np.log(P1) - np.log(P1).mean(1, keepdims=True), "c2": np.log(z["q2"]) - np.log(z["q2"]).mean(1, keepdims=True),
         "r1": rng.normal(size=(n, 3)), "r2": rng.normal(size=(n, 3))}
    FN.save_unit(tmp_path / "tea__s0__U", {"teacher.npz": lambda p: np.savez_compressed(p, **t)}, {})
    FN.save_unit(tmp_path / "pol__s0__U_FINE-TASK_m4", {"release.npz": lambda p: np.savez_compressed(p, **z)}, {})
    for kind, cid in (("policy", "U|FINE-TASK|m4"), ("source", "SRC|U")):
        r = A.inner_unit(kind, 0, cid, D, units_dir=tmp_path, slate="inner")
        files = r.pop("_files")
        json.dumps(r)
        assert set(r["recovery"]["auc"]) == {"v1", "v2", "pair"}
        assert set(r["utility"]) == {"0", "1"}
        for key in ("acc", "logloss", "brier", "const_acc", "balanced_acc", "recalls", "ece"):
            assert key in r["utility"]["0"]
        if kind == "policy":
            assert r["class_preservation_ok"] and isinstance(r["token_states"], int)
        else:
            assert r["primary_family"] == "interface"
            assert set(r["families"]) == {"complete", "scores", "probs", "decisions"}
        FN.save_unit(tmp_path / f"inner__{A.unit_of(0, cid)}", files, r)
    loaded = A.load_inner("pol__s0__U_FINE-TASK_m4", tmp_path)
    assert "code" in loaded and loaded["code"]["inner_predictions"]["P"].shape[1] == len(D["idx"][A.SEL_ROLE])
    comp = A.composed_from_units(0, "U", ["U|FINE-TASK|m4"], D, family="probs", units_dir=tmp_path)
    assert comp["policies"] == ["pol__s0__U_FINE-TASK_m4"]
    comp_dec = A.composed_from_units(0, "U", ["U|FINE-TASK|m4"], D, family="decisions", units_dir=tmp_path)
    assert comp_dec["policies"] == []                  # only class-only policies compose into decisions


# ------------------------------------------------------------------ end-to-end outer unit (synthetic; real lock gate)
def _synthetic_units(tmp_path, D, rng):
    from rgj import finalize as FN
    n = len(D["sex"])
    z = _policy(D, 21, leak=0.6)
    z1 = _policy(D, 22)
    cls1 = A.synthetic_policy(D, (2, 6), seed=23)          # class-only shaped (one token per class)
    t = {"row_id": D["row_id"], "p1": z["q1"], "p2": z["q2"], "d1": z["hard1"], "d2": z["hard2"],
         "c1": np.log(z["q1"]) - np.log(z["q1"]).mean(1, keepdims=True),
         "c2": np.log(z["q2"]) - np.log(z["q2"]).mean(1, keepdims=True),
         "r1": rng.normal(size=(n, 3)), "r2": rng.normal(size=(n, 3))}
    FN.save_unit(tmp_path / "tea__s0__U", {"teacher.npz": lambda p: np.savez_compressed(p, **t)}, {})
    for name, zz in (("pol__s0__U_JOINT_m8_l1", z), ("pol__s0__U_FINE-TASK_m4", z1), ("pol__s0__U_CLASS_m1", cls1)):
        FN.save_unit(tmp_path / name, {"release.npz": lambda p, zz=zz: np.savez_compressed(p, **zz)}, {})
    return t


def test_outer_unit_end_to_end_synthetic(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "SLATES", {**A.SLATES, "final": A.SLATES["inner"]})   # fast synthetic run
    rng = np.random.default_rng(2)
    D = _D(20, n_score=700)
    n = len(D["sex"])
    D["sealed"] = False
    D["idx"]["OSF_DEFENSE_FIT"] = D["idx"][A.FIT_ROLE]
    D["race"] = rng.integers(0, 3, n)
    D["y"] = {"income": rng.integers(0, 2, n), "occupation_group": rng.integers(0, 6, n)}
    units = tmp_path / "units"
    units.mkdir()
    _synthetic_units(units, D, rng)
    lock = {"seeds": {"0": {"u_label": "SRC|U",
                            "score": {"SRC|U": {"kind": "source", "unit": "tea__s0__U", "cid": "SRC|U"},
                                      "J*": {"kind": "policy", "unit": "pol__s0__U_JOINT_m8_l1",
                                             "cid": "U|JOINT|m8|l1"}},
                            "composed_policies": {"U": ["pol__s0__U_JOINT_m8_l1", "pol__s0__U_CLASS_m1"]}}},
            "locked_code_files": {}}
    p = _write_lock(repo, lock)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "lock")
    _git(repo, "push", "-q", "origin", AS.STUDY_BRANCH)
    L = AS.open_assessment(p, repo, chain=())
    try:
        for k, label, spec in AS.jobs_from_lock(L, (0,)):
            r = AS.outer_unit(D, k, label, spec, units_dir=units)
            z = np.load(units / r["unit"] / "preds.npz")
            na = len(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"])
            for key in ("assess_row_id", "assess_unit", "sex", "y_income", "y_occ", "hard1", "hard2", "ll1", "br2",
                        "u_hard1", "u_ll2"):
                assert z[key].shape == (na,), key
            assert z["prob1"].shape == (na, 2) and z["prob2"].shape == (na, 6) and z["const_class"].shape == (2,)
            for w in A.PRIMARY_VIEWS:
                assert z[f"P_auc_{w}"].shape == (3, na, 2) and z[f"P_ce_{w}"].shape == (3, na, 2)
            if spec["kind"] == "source":
                assert r["primary_family"] == "interface"
                for fam in ("complete", "scores", "probs", "decisions"):
                    assert f"P_auc_{fam}_pair" in z.files
                assert r["composed"]["units"] == ["pol__s0__U_JOINT_m8_l1", "pol__s0__U_CLASS_m1"]
                assert r["families"]["decisions"]["composed_units"] == ["pol__s0__U_CLASS_m1"]
                assert r["families"]["interface"]["composed_units"] == r["composed"]["units"]
                np.testing.assert_array_equal(z["prob1"], z["u_prob1"])
            assert r["utility"]["0"]["gain_retention"] is not None
    finally:
        AS.close_assessment()


def test_reference_view_sets_cells_and_value_tokens():
    from dpc import baselines as BL
    D = _D(30)
    rng = np.random.default_rng(4)
    n = len(D["sex"])
    arr = {"row_id": D["row_id"]}
    for i, K, nc in ((1, 2, 12), (2, 6, 15)):
        cells = rng.integers(0, nc, n)
        tab = rng.dirichlet(np.ones(K), nc)
        P = tab[cells]
        C = np.log(P) - np.log(P).mean(1, keepdims=True)
        arr.update({f"cells{i}": cells, f"r{i}": np.eye(nc)[cells], f"p{i}": P, f"c{i}": C, f"d{i}": P.argmax(1)})
    sets = BL.reference_view_sets("F", 0, D, arrays=arr)
    assert set(sets) == {"interface", "scores", "probs", "decisions", "cells"}
    assert all(sets[f]["tokens"] for f in sets)
    assert sets["cells"]["X"]["v1"].shape[1] == 12 + 2 + 2
    e = BL.reference_view_sets("E", 0, D, arrays={**arr, "r1": rng.normal(size=(n, 4)), "r2": rng.normal(size=(n, 4))})
    assert set(e) == {"interface", "scores", "probs", "decisions", "complete"} and not e["interface"]["tokens"]
