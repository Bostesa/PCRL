"""[lra port of lcr/tests/test_mapper.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Role C: lra.mapper on SYNTHETIC data only (cbp.fit.synthetic_teacher + lra.mapper.synthetic_labels; small shapes).

Covers: incremental vs from-scratch terms, budgets (incl. after paired moves), the sequential temporary partner not
required feasible, refusal of an infeasible paired update, determinism, memo misses on changed statistics, class
preservation, caps, equal ceilings, no SEX in C-TASK, batch-vs-scalar decoder parity and input refusals.
Run: OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m lra.sema --label C:tests -- env ... <python> -m pytest -q lra/tests/test_mapper.py
"""
import copy
import json

import numpy as np
import pytest

from cbp import fit as CF
from lra import decoder as DEC
from lra import mapper as MP
from lra import run as RN
from qpc import compress as QC
from qpc import partition as PT
from qpc import release as RL

META = {"teacher_model_sha256": "a" * 64, "feature_names_sha256": "b" * 64, "fixture": True}
CAPS = [3, 5]
BUDGET = {"ll": 0.05, "brier": 0.03}     # fixture-mode allowance so the constrained arms have feasible starts


@pytest.fixture(scope="module")
def world():
    T, tr, S = CF.synthetic_teacher(N=3000, n_fit=1600, seed=1, underflow=10)
    Y = MP.synthetic_labels(T, tr, 1)
    P1, d1, P2, d2 = T["p1"][tr], T["d1"][tr], T["p2"][tr], T["d2"][tr]
    f1, f2, _ = PT.fit_fine_pair(P1, d1, P2, d2, caps={1: 6, 2: 9})
    fine = {"fine1": f1.to_dict(), "fine2": f2.to_dict()}
    src = {}
    pp, _ = QC.fit_policy_pair("FINE-TASK", f1, f2, P1, d1, P2, d2, S, CAPS[0], CAPS[1], None,
                               baseline_diagnostic=False)
    src[RN.d0_id("FINE-TASK")] = pp.to_dict()
    for lam in (0.01, 0.1):
        for fam in ("LOCAL", "SEQ-12", "SEQ-21"):
            pp, _ = QC.fit_policy_pair(fam, f1, f2, P1, d1, P2, d2, S, CAPS[0], CAPS[1], lam,
                                       baseline_diagnostic=False)
            src[RN.d0_id(fam, lam)] = pp.to_dict()
    refs0 = {"caps": CAPS, "budget": BUDGET}
    rc, fc = MP.fit_unit("C-TASK", fine, T, tr, Y, S, starts={RN.d0_id("FINE-TASK"): src[RN.d0_id("FINE-TASK")]},
                         refs=refs0, meta=META)
    refs = {**refs0, **MP.refs_from_ctask(rc)}
    ct = {RN.ctask_id(): fc["policy.json"]}
    out = {}
    for arm, extra in (("K-LOCAL", {}), ("K-SEQ-12", {RN.d0_id("SEQ-12", 0.01): src[RN.d0_id("SEQ-12", 0.01)]}),
                       ("K-SEQ-21", {RN.d0_id("SEQ-21", 0.1): src[RN.d0_id("SEQ-21", 0.1)]})):
        r, f = MP.fit_unit(arm, fine, T, tr, Y, S, starts={**ct, **extra}, refs=refs, meta=META)
        out[arm] = (r, f)
    wit = {**ct, RN.constrained_id("LOCAL"): out["K-LOCAL"][1]["policy.json"],
           RN.constrained_id("SEQ-12"): out["K-SEQ-12"][1]["policy.json"],
           RN.constrained_id("SEQ-21"): out["K-SEQ-21"][1]["policy.json"], **src}
    for arm in ("K-JOINT-SINGLE", "K-JOINT-PAIR"):
        out[arm] = MP.fit_unit(arm, fine, T, tr, Y, S, witnesses=wit, refs=refs, meta=META)
    return {"T": T, "tr": tr, "S": S, "Y": Y, "fine": fine, "src": src, "refs0": refs0, "refs": refs,
            "ctask": (rc, fc), "ct": ct, "out": out, "wit": wit}


def _problem(w, sex=True):
    T, tr = w["T"], w["tr"]
    f1, f2 = PT.load_fine(w["fine"])
    P = {1: T["p1"][tr], 2: T["p2"][tr]}
    dd = {1: T["d1"][tr], 2: T["d2"][tr]}
    return MP.Problem(f1, f2, P, dd, w["Y"], w["S"] if sex else None, CAPS, BUDGET, DEC)


def _labels(pz):
    pp = RL.PolicyPair.from_dict(pz)
    return {1: QC.labels_from_policy(pp.p1), 2: QC.labels_from_policy(pp.p2)}


def _strip(rec):
    """Record without timing fields (for determinism)."""
    z = json.loads(json.dumps(RN._finite(rec)))

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items() if not (k.startswith("cpu") or k.startswith("wall"))}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    return walk(z)


# ------------------------------------------------------------------ incremental vs from scratch
def test_incremental_terms_match_from_scratch_after_moves(world):
    pb = _problem(world)
    st = MP.State(pb, _labels(world["ctask"][1]["policy.json"]))
    w = (1.0, 0.5, 0.3, 0.7)            # all four terms active
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        r = int(rng.integers(1, 3))
        f = int(rng.integers(0, pb.fine[r].F))
        if pb.ncell[r][f] == 0:
            continue
        cur = st.terms()
        res = MP._cell(st, r, f, w, cur, (), need_feas_all=True)
        if res is None:
            continue
        j = int(rng.integers(0, res["B"].size))
        b = int(res["B"][j])
        st.apply(r, f, b)
        new = st.terms()
        fresh = MP.State(pb, {1: st.canonical_labels(1), 2: st.canonical_labels(2)}).terms()
        for k in ("L1", "L2", "B1", "B2", "I1", "I2", "I12"):
            assert new[k] == fresh[k], k                     # exact state value is a function of the partition
        assert abs((new[f"L{r}"] - cur[f"L{r}"]) - res["dL"][j]) <= 1e-12
        assert abs((new[f"B{r}"] - cur[f"B{r}"]) - res["dB"][j]) <= 1e-12
        assert abs((new[f"I{r}"] - cur[f"I{r}"]) - res["dI"][j]) <= 1e-12
        assert abs((new["I12"] - cur["I12"]) - res["dI12"][j]) <= 1e-12
        checked += 1
    assert checked >= 20
    # undo restores everything bitwise
    t0 = st.terms()
    lab0 = {r: st.labels(r) for r in (1, 2)}
    T12 = st.T12.copy()
    f = int(np.flatnonzero(pb.ncell[2] > 0)[0])
    res = MP._cell(st, 2, f, w, t0, (), need_feas_all=True)
    if res is not None:
        u = st.apply(2, f, int(res["B"][0]))
        st.undo(u)
        assert st.terms() == t0 and np.array_equal(st.T12, T12)
        assert all(np.array_equal(st.labels(r), lab0[r]) for r in (1, 2))


def test_deployed_recomputation_parity_every_unit(world):
    units = [world["ctask"]] + list(world["out"].values())
    for rec, files in units:
        dep = rec["deployed"]
        assert dep["parity_ok"]
        assert max(dep["parity_abs_diff"].values()) <= 1e-10
        assert all(v["bitwise_equal"] for v in dep["q_vs_state"].values())
        # recompute the deployed terms independently from the files
        rel = files["release.npz"]
        tr = world["tr"]
        for i in (1, 2):
            from dpc.utility import per_row
            pr = per_row(rel[f"q{i}"][tr], world["Y"][i])
            assert abs(pr["ll"].mean() - rec["final_state_terms"][f"L{i}"]) <= 1e-10


# ------------------------------------------------------------------ budgets / caps / class preservation
def test_budgets_and_local_caps_enforced_including_paired(world):
    for arm in ("K-LOCAL", "K-SEQ-12", "K-SEQ-21", "K-JOINT-SINGLE", "K-JOINT-PAIR"):
        rec, files = world["out"][arm]
        assert rec["status"] == "FEASIBLE", arm
        dep = rec["deployed"]
        assert dep["feasible"], arm
        for i in ("1", "2"):
            c = dep["constraints"][i]
            assert c["L"] <= c["L_limit"] and c["B"] <= c["B_limit"]
            assert c["I_slack_table_exact"] >= 0
            assert rec["final_state_terms"][f"I{i[0]}"] <= world["refs"]["I_ctask"][i]
        # every refined candidate the arm calls eligible satisfies all constraints
        for s in rec["starts"]:
            fin = s.get("final")
            if fin and s.get("eligible"):
                assert fin["feasible_all"]
    pr = world["out"]["K-JOINT-PAIR"][0]
    steps = [p for s in pr["starts"] for p in s.get("pair_steps", [])]
    assert steps, "the paired step must run after each single sweep"
    assert pr["work"]["pair_evals"] > 0


def test_paired_step_refuses_infeasible_update(world, monkeypatch):
    pb = _problem(world)
    pb.capI = {1: world["refs"]["I_ctask"]["1"], 2: world["refs"]["I_ctask"]["2"]}
    st = MP.State(pb, _labels(world["ctask"][1]["policy.json"]))
    w = MP.weights("K-JOINT-PAIR", None)
    cur = st.terms()
    lab0 = {r: st.labels(r) for r in (1, 2)}
    T12 = st.T12.copy()
    real = MP.feasible
    base = {r: st.canonical_labels(r) for r in (1, 2)}

    def jointly_infeasible(pb_, t, recips):        # any state where BOTH recipients moved is declared infeasible
        moved = [not np.array_equal(st.canonical_labels(r), base[r]) for r in (1, 2)]
        if all(moved):
            return False
        return real(pb_, t, recips)
    monkeypatch.setattr(MP, "feasible", jointly_infeasible)
    ok, t, v, info = MP._pair_step(st, w, (1, 2), cur, MP.objective(cur, w), budget_left=10_000)
    assert not ok and info["accepted"] is None
    assert st.ct["pair_evals"] > 0 and st.ct["pair_rejected_infeasible"] == st.ct["pair_evals"]
    assert st.terms() == cur and np.array_equal(st.T12, T12)
    assert all(np.array_equal(st.labels(r), lab0[r]) for r in (1, 2))


def test_pair_bank_keeps_non_improving_proposals(world):
    pb = _problem(world)
    pb.capI = {1: world["refs"]["I_ctask"]["1"], 2: world["refs"]["I_ctask"]["2"]}
    st = MP.State(pb, _labels(world["ctask"][1]["policy.json"]))
    w = MP.weights("K-JOINT-PAIR", None)
    cur = st.terms()
    ok, t, v, info = MP._pair_step(st, w, (1, 2), cur, MP.objective(cur, w), budget_left=10_000)
    for r in ("1", "2"):
        props = info["proposals"][r]
        assert 0 < len(props) <= MP.PAIR_KEEP_PHI + MP.PAIR_KEEP_TASK
        keys = [(p["fine_cell"], p["target_canon"]) for p in props]
        assert len(set(keys)) == len(keys)
    allp = info["proposals"]["1"] + info["proposals"]["2"]
    assert any(p["dPhi_one_sided"] >= 0 for p in allp) or any(p["dTask"] >= 0 for p in allp)


def test_class_preservation_and_caps(world):
    T = world["T"]
    for rec, files in [world["ctask"]] + list(world["out"].values()):
        rel = files["release.npz"]
        pair = RL.PolicyPair.from_dict(files["policy.json"])
        for i, pol in ((1, pair.p1), (2, pair.p2)):
            assert np.array_equal(rel[f"hard{i}"], T[f"d{i}"])
            assert np.array_equal(rel[f"q{i}"].argmax(1), T[f"d{i}"])
            assert all(x <= CAPS[i - 1] for x in pol.tokens_per_class())
            assert all(x <= CAPS[i - 1] for x in rec["token_counts"][str(i)])
        cid, d1, d2, sha = DEC.load_decoder_pair(files["decoder.json"], pair)
        assert sha == rec["decoder_sha256"] and cid == rec["config"]


def test_start_over_caps_refused(world):
    w = world
    with pytest.raises(ValueError, match="cap"):
        MP.fit_unit("C-TASK", w["fine"], w["T"], w["tr"], w["Y"], w["S"],
                    starts={RN.d0_id("FINE-TASK"): w["src"][RN.d0_id("FINE-TASK")]},
                    refs={"caps": [1, 1], "budget": BUDGET}, meta=META)


# ------------------------------------------------------------------ sequential temporary partner
def test_sequential_temporary_partner_not_required_feasible(world):
    rec, _ = world["out"]["K-SEQ-12"]
    any_partner_violation = False
    for s in rec["starts"]:
        s1 = s["stage1"]
        assert s1["partner"] == "CLASS-ONLY" and s1["partner_constraints_enforced"] is False
        if s1.get("status") == "REFINED":
            pc = s1["end"]["partner_class_only_constraints"]
            any_partner_violation |= (pc["L_slack"] < 0 or pc["B_slack"] < 0)
    # the class-only occupation partner is far outside its budgets, yet stage 1 refined and the arm is feasible
    assert any_partner_violation
    assert rec["status"] == "FEASIBLE" and rec["deployed"]["feasible"]


# ------------------------------------------------------------------ determinism, cache, scalar path, no SEX
def test_determinism(world):
    w = world
    a = MP.fit_unit("K-SEQ-21", w["fine"], w["T"], w["tr"], w["Y"], w["S"],
                    starts={**w["ct"], RN.d0_id("SEQ-21", 0.1): w["src"][RN.d0_id("SEQ-21", 0.1)]},
                    refs=w["refs"], meta=META)
    r0, f0 = w["out"]["K-SEQ-21"]
    assert _strip(a[0]) == _strip(r0)
    assert json.dumps(a[1]["policy.json"]) == json.dumps(f0["policy.json"])
    assert a[1]["decoder.json"]["decoder_sha256"] == f0["decoder.json"]["decoder_sha256"]
    for k in f0["release.npz"]:
        assert np.array_equal(a[1]["release.npz"][k], f0["release.npz"][k])


def test_memo_misses_on_changed_statistics(world):
    pb = _problem(world)
    st = MP.State(pb, _labels(world["ctask"][1]["policy.json"]))
    r, c = 2, 1
    c0, c1 = pb.crange[r][c]
    slots = st.alive_slots(r, c)
    st.ensure(r, c, np.arange(c0, c1), slots)
    h0, m0 = st.ct["memo_hits"], st.ct["memo_misses"]
    out = st.ensure(r, c, np.arange(c0, c1), slots, verify=True)
    assert st.ct["memo_misses"] == m0 and st.ct["memo_hits"] - h0 == (c1 - c0) * slots.size
    # (1) a move changes two slots' statistics: exactly their entries miss and are re-solved
    f = c0
    res = MP._cell(st, r, f, (1.0, 0.5, 0.0, 0.0), st.terms(), ())
    b = int(res["B"][0])
    a_ = int(st.lab[r][f])
    st.apply(r, f, b)
    m0 = st.ct["memo_misses"]
    alive = st.alive_slots(r, c)
    n, Y, S, Q, ll, br = st.ensure(r, c, np.arange(c0, c1), alive, verify=True)
    changed = int(np.isin(alive, [a_, b]).sum())
    assert st.ct["memo_misses"] - m0 == (c1 - c0) * changed
    for i in range(c1 - c0):
        for j in range(alive.size):
            if n[i, j] > 0 and alive[j] in (a_, b):
                _, Qd, _, _ = DEC.solve_batch(Y[i, j][None], S[i, j][None], np.array([n[i, j]]), np.array([c]))
                assert np.array_equal(Q[i, j], Qd[0])
    # (2) a stored entry whose statistics differ (1 ulp teacher sum, or one label count) is never served silently
    m = st.memo[r][c]
    for key, fn in (("S", lambda x: np.nextafter(x, 2.0)), ("Y", lambda x: x + 1.0)):
        saved = m[key][0, 0, 0]
        m[key][0, 0, 0] = fn(saved)
        with pytest.raises(AssertionError, match="stale cache key"):
            st.ensure(r, c, np.arange(c0, c1), alive, verify=True)
        m[key][0, 0, 0] = saved
    st.ensure(r, c, np.arange(c0, c1), alive, verify=True)
    assert out[0].shape == (c1 - c0, slots.size)


def test_memo_version_lookup_equals_verified_lookup(world):
    pb = _problem(world)
    st = MP.State(pb, _labels(world["ctask"][1]["policy.json"]))
    w = (1.0, 0.5, 0.3, 0.7)
    rng = np.random.default_rng(5)
    for _ in range(25):
        r = int(rng.integers(1, 3))
        f = int(rng.integers(0, pb.fine[r].F))
        res = MP._cell(st, r, f, w, st.terms(), (), need_feas_all=True)
        if res is None:
            continue
        u = st.apply(r, f, int(res["B"][int(rng.integers(0, res["B"].size))]))
        if rng.random() < 0.3:
            st.undo(u)
        else:
            st.refresh(r, [u["a"], u["b"]])
        for rr in (1, 2):
            for c in range(pb.K[rr]):
                c0, c1 = pb.crange[rr][c]
                sl = st.alive_slots(rr, c)
                if sl.size:
                    st.ensure(rr, c, np.arange(c0, c1), sl, verify=True)     # raises on any stale entry
    assert st.ct["memo_verified"] > 0


def test_scalar_decoder_path_gives_identical_moves(world, monkeypatch):
    w = world
    r0, f0 = w["out"]["K-LOCAL"]
    real = DEC.solve_batch

    def scalar(Y, S, n, d, kappa=32.0, eps=1e-12):       # = solve_token row by row (solve_token is M = 1)
        rows = [real(Y[i:i + 1], S[i:i + 1], n[i:i + 1], d[i:i + 1]) for i in range(len(n))]
        U = np.vstack([x[0] for x in rows])
        Q = np.vstack([x[1] for x in rows])
        certs = {k: np.concatenate([np.atleast_1d(x[3][k]) for x in rows]) for k in rows[0][3]}
        return U, Q, np.concatenate([x[2] for x in rows]), certs
    monkeypatch.setattr(DEC, "solve_batch", scalar)
    r1, f1 = MP.fit_unit("K-LOCAL", w["fine"], w["T"], w["tr"], w["Y"], w["S"], starts=w["ct"], refs=w["refs"],
                         meta=META)
    monkeypatch.setattr(DEC, "solve_batch", real)
    mv0 = [s.get("moves") for st in r0["starts"] for s in st["stages"]]
    mv1 = [s.get("moves") for st in r1["starts"] for s in st["stages"]]
    assert mv0 == mv1
    assert json.dumps(f1["policy.json"]) == json.dumps(f0["policy.json"])


def test_ctask_uses_no_sex(world):
    w = world
    S2 = w["S"][np.random.default_rng(3).permutation(w["S"].size)]
    r1, f1 = MP.fit_unit("C-TASK", w["fine"], w["T"], w["tr"], w["Y"], S2,
                         starts={RN.d0_id("FINE-TASK"): w["src"][RN.d0_id("FINE-TASK")]}, refs=w["refs0"],
                         meta=META)
    r0, f0 = w["ctask"]
    assert r0["sex_used_in_search"] is False
    assert json.dumps(f1["policy.json"]) == json.dumps(f0["policy.json"])
    rep = r0["ctask_report"]
    assert rep["d1_initialisation"]["start"] == RN.d0_id("FINE-TASK")
    assert rep["refined"]["T"] <= rep["d1_initialisation"]["terms"]["T"]
    assert set(rep["d0_same_maps"]) == {"start", "refined"}


# ------------------------------------------------------------------ equal ceilings
def test_equal_ceilings_and_ceiling_stop(world, monkeypatch):
    for arm in ("K-SEQ-12", "K-SEQ-21", "K-JOINT-SINGLE", "K-JOINT-PAIR"):
        assert world["out"][arm][0]["work"]["eval_ceiling"] == MP.EVAL_CEILING
    rules = MP.rules()
    assert rules["equal_work"]["EVAL_CEILING_per_unit"] == MP.EVAL_CEILING
    w = world
    monkeypatch.setattr(MP, "EVAL_CEILING", 200)
    for arm in ("K-JOINT-SINGLE", "K-JOINT-PAIR"):
        rec, _ = MP.fit_unit(arm, w["fine"], w["T"], w["tr"], w["Y"], w["S"], witnesses=w["wit"], refs=w["refs"],
                             meta=META)
        n_starts = len(rec["starts_given"])
        share = 200 // n_starts
        maxB = max(max(CAPS) - 1, 1)
        for s in rec["starts"]:
            if s.get("status") == "REFINED":
                assert s["evals"] <= share + maxB + MP.PAIR_KEEP_PHI * 0 + 64 * 0 + 10 * maxB * 2
                assert s["stop"] in ("eval_ceiling", "no_change_sweep", "sweep_cap")
        assert rec["work"]["eval_ceiling"] == 200
        assert any(s.get("stop") == "eval_ceiling" for s in rec["starts"])
    rec, _ = MP.fit_unit("K-SEQ-12", w["fine"], w["T"], w["tr"], w["Y"], w["S"],
                         starts={**w["ct"], RN.d0_id("SEQ-12", 0.01): w["src"][RN.d0_id("SEQ-12", 0.01)]},
                         refs=w["refs"], meta=META)
    stage_share = 200 // (2 * len(rec["starts_given"]))
    for s in rec["starts"]:
        for k in ("stage1", "stage2"):
            if s.get(k, {}).get("status") == "REFINED":
                assert s[k]["evals"] <= stage_share + max(CAPS)


# ------------------------------------------------------------------ joint witnesses and refusals
def test_joint_infeasible_witness_never_eligible(world):
    rec, _ = world["out"]["K-JOINT-SINGLE"]
    names = [s["name"] for s in rec["starts"]]
    assert names == [k for k in MP.registered_starts("K-JOINT-SINGLE") if k in world["wit"]]
    for s in rec["starts"]:
        if not s["unchanged"]["feasible_all"]:
            assert s["status"] == "EXCLUDED_INFEASIBLE_WITNESS" and not s["eligible"]
            assert rec["winner"]["start"] != s["name"] or rec["status"] != "FEASIBLE"
    win = [s for s in rec["starts"] if s["name"] == rec["winner"]["start"]][0]
    assert win["status"] == "REFINED"
    assert rec["final_state_terms"]["Phi"] <= min(s["unchanged"]["terms"]["Phi"] for s in rec["starts"]
                                                  if s["unchanged"]["feasible_all"])


def test_refusals(world):
    w = world
    st = {RN.d0_id("FINE-TASK"): w["src"][RN.d0_id("FINE-TASK")]}
    with pytest.raises(ValueError, match="I_ctask"):
        MP.fit_unit("K-LOCAL", w["fine"], w["T"], w["tr"], w["Y"], w["S"], starts=w["ct"], refs=w["refs0"],
                    meta=META)
    bad = copy.deepcopy(w["refs"])
    bad["I_ctask"]["1"] = float(np.nextafter(bad["I_ctask"]["1"], 1.0))
    with pytest.raises(ValueError, match="differ"):
        MP.fit_unit("K-LOCAL", w["fine"], w["T"], w["tr"], w["Y"], w["S"], starts=w["ct"], refs=bad, meta=META)
    with pytest.raises(ValueError, match="unregistered"):
        MP.fit_unit("C-TASK", w["fine"], w["T"], w["tr"], w["Y"], w["S"], starts={**st, "U|JOINT|i8o64|l0.5": 0},
                    refs=w["refs0"], meta=META)
    with pytest.raises(ValueError, match="lam"):
        MP.fit_unit("K-LOCAL", w["fine"], w["T"], w["tr"], w["Y"], w["S"], lam=0.01, starts=w["ct"],
                    refs=w["refs"], meta=META)
    with pytest.raises(ValueError, match="grid"):
        MP.fit_unit("W-LOCAL", w["fine"], w["T"], w["tr"], w["Y"], w["S"], lam=0.5, starts=w["ct"],
                    meta={k: v for k, v in META.items() if k != "fixture"})
    with pytest.raises(ValueError, match="profile"):
        MP.fit_unit("C-TASK", w["fine"], w["T"], w["tr"], w["Y"], w["S"], starts=st, refs=w["refs0"],
                    meta={k: v for k, v in META.items() if k != "fixture"})
    with pytest.raises(ValueError, match="missing"):
        MP.fit_unit("W-JOINT", w["fine"], w["T"], w["tr"], w["Y"], w["S"], lam=0.01, witnesses=w["ct"],
                    refs=w["refs0"], meta=META)
    with pytest.raises(ValueError, match="sha256"):
        MP.fit_unit("C-TASK", w["fine"], w["T"], w["tr"], w["Y"], w["S"], starts=st, refs=w["refs0"],
                    meta={"fixture": True})


def test_rules_are_finite_json_and_registered_starts():
    z = MP.rules()
    json.dumps(z, allow_nan=False)
    assert z["starts"]["C-TASK"] == ["U|FINE-TASK|i8o64"]
    assert len(z["starts"]["K-SEQ-12"]) == 7 and len(z["starts"]["K-JOINT-PAIR"]) == 29
    assert len(z["starts"]["W-JOINT"]) == 29 and z["starts"]["K-LOCAL"] == ["U|C-TASK|i8o64|D1"]
    assert z["tolerances"]["TOL"] == 1e-12 and z["neighbourhood"]["sweeps"] == 5
    assert z["budgets"]["budget_values"] == {"ll": 0.005, "brier": 0.003}
