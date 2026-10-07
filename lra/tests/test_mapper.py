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
SEED, N_ROWS, N_FIT, FINE_CAPS = 4, 4000, 2400, {1: 8, 2: 12}   # a synthetic shape whose K-JOINT-PAIR accepts pairs
CAPS = [4, 6]
BUDGET = {"ll": 0.05, "brier": 0.03}     # fixture-mode allowance so the constrained arms have feasible starts
LAM = 0.01


@pytest.fixture(scope="module")
def world():
    T, tr, S = CF.synthetic_teacher(N=N_ROWS, n_fit=N_FIT, seed=SEED, underflow=10)
    Y = MP.synthetic_labels(T, tr, SEED)
    P1, d1, P2, d2 = T["p1"][tr], T["d1"][tr], T["p2"][tr], T["d2"][tr]
    f1, f2, _ = PT.fit_fine_pair(P1, d1, P2, d2, caps=FINE_CAPS)
    fine = {"fine1": f1.to_dict(), "fine2": f2.to_dict()}
    src = {}
    pp, _ = QC.fit_policy_pair("FINE-TASK", f1, f2, P1, d1, P2, d2, S, CAPS[0], CAPS[1], None,
                               baseline_diagnostic=False)
    src[RN.d0_id("FINE-TASK")] = pp.to_dict()
    pp, _ = QC.fit_policy_pair("CLASS", f1, f2, P1, d1, P2, d2, S, 1, 1, None, baseline_diagnostic=False)
    src[RN.d0_id("CLASS")] = pp.to_dict()                      # a joint witness (feasible-only)
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
    # the weighted (D bank) controls at one lambda, with their registered starts / witnesses
    wout = {"W-LOCAL": MP.fit_unit("W-LOCAL", fine, T, tr, Y, S, lam=LAM, starts=ct, refs=refs0, meta=META)}
    for a in ("SEQ-12", "SEQ-21"):
        wout[f"W-{a}"] = MP.fit_unit(f"W-{a}", fine, T, tr, Y, S, lam=LAM,
                                     starts={**ct, RN.d0_id(a, LAM): src[RN.d0_id(a, LAM)]}, refs=refs0, meta=META)
    wwit = {**ct, **{RN.weighted_id(f, LAM): wout[f"W-{f}"][1]["policy.json"] for f in ("LOCAL", "SEQ-12", "SEQ-21")},
            **src}
    wout["W-JOINT"] = MP.fit_unit("W-JOINT", fine, T, tr, Y, S, lam=LAM, witnesses=wwit, refs=refs0, meta=META)
    units = {"C-TASK": (rc, fc, refs0), **{a: (*v, refs) for a, v in out.items()},
             **{a: (*v, refs0) for a, v in wout.items()}}
    return {"T": T, "tr": tr, "S": S, "Y": Y, "fine": fine, "src": src, "refs0": refs0, "refs": refs,
            "ctask": (rc, fc), "ct": ct, "out": out, "wit": wit, "wout": wout, "units": units}


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
            return {k: walk(v) for k, v in o.items() if not (k.startswith("cpu") or k.startswith("wall")
                                                             or k.endswith("cpu_s"))}
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
    assert a[1]["trace.json"] == f0["trace.json"] and a[0]["trace_sha256"] == r0["trace_sha256"]
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
    assert len(z["starts"]["K-SEQ-12"]) == 7 and len(z["starts"]["K-JOINT-PAIR"]) == 30
    assert len(z["starts"]["W-JOINT"]) == 30 and z["starts"]["K-LOCAL"] == ["U|C-TASK|i8o64|D1"]
    for a in ("K-JOINT-SINGLE", "K-JOINT-PAIR", "W-JOINT"):
        assert z["starts"][a][4:6] == ["U|FINE-TASK|i8o64", "U|CLASS|i1o1"]
        assert z["starts"][a][6:] == [RN.d0_id(f, l) for l in RN.LAMS for f in RN.PRIVACY]
    assert z["starts"]["K-JOINT-PAIR"] == MP.registered_starts("K-JOINT-PAIR") == z["starts"]["K-JOINT-SINGLE"]
    assert z["tolerances"]["TOL"] == 1e-12 and z["neighbourhood"]["sweeps"] == 5
    assert z["budgets"]["budget_values"] == {"ll": 0.005, "brier": 0.003}


# ------------------------------------------------------------------ persisted trace and replay (checks 6, 7, 8)
ALL_ARMS = ("C-TASK", "W-LOCAL", "W-SEQ-12", "W-SEQ-21", "W-JOINT", "K-LOCAL", "K-SEQ-12", "K-SEQ-21",
            "K-JOINT-SINGLE", "K-JOINT-PAIR")


def _replay(w, rec, files, refs, trace=None, **kw):
    return MP.replay_unit(rec, w["fine"], w["T"], w["tr"], w["Y"], w["S"], refs,
                          trace=files["trace.json"] if trace is None else trace, files=files, **kw)


def _saved(trace):
    """The trace exactly as lra.run.save writes and a reader loads it (finite JSON)."""
    return json.loads(json.dumps(RN._finite(trace), allow_nan=False))


def _codes(rep):
    return {v["code"] for v in rep["violations"]}


def _retrace(rec, trace):
    """A record whose trace hash matches a tampered trace, so the replay must catch the defect by content."""
    r = copy.deepcopy(rec)
    r["trace_sha256"] = MP.json_sha(trace)
    return r


def test_trace_persisted_for_every_arm_and_round_trips(world):
    assert set(world["units"]) == set(ALL_ARMS)
    for arm, (rec, files, refs) in world["units"].items():
        tr_ = files["trace.json"]
        assert tr_["schema"] == MP.TRACE_SCHEMA and tr_["arm"] == arm and rec["trace_schema"] == MP.TRACE_SCHEMA
        loaded = _saved(tr_)
        assert loaded == tr_ and MP.json_sha(loaded) == rec["trace_sha256"] == MP.json_sha(tr_)
        body = json.dumps(RN._finite(rec))
        assert "stats_sha256" not in body and "start_labels" not in body         # the trace is not in record.json
        assert [z["name"] for z in tr_["starts"]] == rec["starts_given"]
        F = {r: len(world["fine"][f"fine{r}"]["n"]) for r in (1, 2)}
        for z in tr_["starts"]:
            assert all(len(z["labels"][str(r)]) == F[r] for r in (1, 2))
            for g in z["stages"]:
                assert {"stage", "enforced", "start_labels", "start_terms", "start_stats_sha256", "status", "moves",
                        "termination"} <= set(g)
                if g["status"] == "REFINED":
                    rc = g["termination"]
                    assert rc["stop"] in ("no_change_sweep", "sweep_cap", "eval_ceiling")
                    assert rc["accepted"] == len(g["moves"]) and rc["evals"] >= 0 and rc["ceiling_share"] > 0
                    for m in g["moves"]:
                        for p in m["parts"]:
                            assert len(p["stats_sha256"]) == 2 and len(p["q_sha256"]) == 2 and p["stats_sha256"][1]
        assert tr_["winner"]["start"] == rec["winner"]["start"] and tr_["winner"]["kind"] == rec["winner"]["kind"]
        pp = RL.PolicyPair.from_dict(files["policy.json"])
        assert tr_["winner"]["labels"]["1"] == QC.labels_from_policy(pp.p1).tolist()
        assert tr_["winner"]["labels"]["2"] == QC.labels_from_policy(pp.p2).tolist()
    # sequential: the stage-1 partner (CLASS-ONLY) and the frozen map are persisted
    tz = world["units"]["K-SEQ-12"][1]["trace.json"]
    f2 = PT.load_fine(world["fine"])[1]
    for z in tz["starts"]:
        g1 = z["stages"][0]
        assert g1["stage"] == "seq1" and g1["enforced"] == [1] and g1["partner"]["view"] == "CLASS-ONLY"
        assert g1["partner"]["constraints_enforced"] is False
        assert g1["partner"]["labels"] == QC.class_labels(f2).tolist() == g1["start_labels"]["2"]
        if g1["status"] == "REFINED":
            g2 = z["stages"][1]
            assert g2["frozen"]["recipient"] == 1 and g2["start_labels"]["1"] == g2["frozen"]["labels"]
            assert g2["start_labels"]["2"] == z["labels"]["2"]


def test_replay_passes_on_every_arm_including_accepted_pair(world):
    pairs = 0
    for arm, (rec, files, refs) in world["units"].items():
        rep = _replay(world, rec, files, refs, trace=_saved(files["trace.json"]))
        assert rep["ok"], (arm, rep["violations"][:3])
        assert all(v is not False for v in rep["checks"].values()), (arm, rep["checks"])
        assert rep["max_abs_diff"] <= MP.REPLAY_TERM_ATOL and rep["max_delta_abs_diff"] <= MP.REPLAY_DELTA_ATOL
        assert rep["n_states"] == len(rep["per_state"]) >= 1
        moves = sum(len(g["moves"]) for z in files["trace.json"]["starts"] for g in z["stages"])
        assert rep["n_moves"]["single"] + rep["n_moves"]["pair"] == moves
        if arm.startswith("K-"):                 # every ACCEPTED state meets its enforced budgets and caps
            acc = [s for s in rep["per_state"] if s["kind"] in ("single", "pair")]
            assert acc and all(s["feasible_enforced"] for s in acc)
        pairs += rep["n_moves"]["pair"] if arm == "K-JOINT-PAIR" else 0
        if arm in ("K-SEQ-12", "K-SEQ-21"):
            assert rep["checks"]["check7"] and rep["partner_infeasible_states"] > 0   # partner violated, not required
    assert pairs >= 1, "the fixture must exercise an accepted atomic paired move"
    # the paired states are accepted states that satisfy both budgets and both caps (check 6)
    rec, files, refs = world["units"]["K-JOINT-PAIR"]
    rep = _replay(world, rec, files, refs)
    ps = [s for s in rep["per_state"] if s["kind"] == "pair"]
    assert ps and all(s["feasible_both"] and s["enforced"] == [1, 2] for s in ps)
    # replay also accepts the start maps and cross-checks their labels
    rep = _replay(world, rec, files, refs, starts=world["wit"])
    assert rep["ok"]


def _first_move(trace, typ="single"):
    for z in trace["starts"]:
        for g in z["stages"]:
            for k, m in enumerate(g["moves"]):
                if m["type"] == typ:
                    return z, g, k, m
    raise AssertionError("no move")


def test_replay_catches_corrupted_move(world):
    rec, files, refs = world["units"]["K-JOINT-SINGLE"]
    fine2 = PT.load_fine(world["fine"])
    # (a) a move redirected to another same-class token
    tz = copy.deepcopy(files["trace.json"])
    z, g, k, m = _first_move(tz)
    p = m["parts"][0]
    lab = np.asarray(g["start_labels"][str(p["r"])])
    for mv in g["moves"][:k]:
        for q in mv["parts"]:
            if q["r"] == p["r"]:
                lab[q["f"]] = q["to_canon"]
                lab = MP.canon_labels(lab)
    fr = fine2[p["r"] - 1]
    alt = [c for c in np.unique(lab) if fr.cell_class[c] == fr.cell_class[p["f"]] and c not in (p["to_canon"],
                                                                                               p["from_canon"])]
    assert alt, "the fixture must offer an alternative same-class target"
    p["to_canon"] = int(alt[0])
    rep = _replay(world, _retrace(rec, tz), files, refs, trace=tz)
    assert not rep["ok"] and _codes(rep) & {"TERMS_MISMATCH", "STATS_HASH_MISMATCH", "DELTA_MISMATCH"}
    assert not rep["checks"]["check8"]
    # (b) a move to a token of another predicted class / a non-token label
    tz = copy.deepcopy(files["trace.json"])
    z, g, k, m = _first_move(tz)
    m["parts"][0]["to_canon"] = int(m["parts"][0]["f"])
    rep = _replay(world, _retrace(rec, tz), files, refs, trace=tz)
    assert "MOVE_INVALID" in _codes(rep) and not rep["ok"]
    # (c) a corrupted vectorised delta, (d) corrupted incremental terms, (e) a dropped move
    for edit, code in ((lambda m: m["parts"][0].__setitem__("dL", m["parts"][0]["dL"] + 1e-9), "DELTA_MISMATCH"),
                       (lambda m: m["terms_after"].__setitem__("I12", m["terms_after"]["I12"] + 1e-9),
                        "TERMS_MISMATCH")):
        tz = copy.deepcopy(files["trace.json"])
        edit(_first_move(tz)[3])
        rep = _replay(world, _retrace(rec, tz), files, refs, trace=tz)
        assert code in _codes(rep) and not rep["checks"]["check8"]
    tz = copy.deepcopy(files["trace.json"])
    z, g, k, m = _first_move(tz)
    del g["moves"][k]
    rep = _replay(world, _retrace(rec, tz), files, refs, trace=tz)
    assert not rep["ok"]
    # an untouched record hash also exposes any edit
    rep = _replay(world, rec, files, refs, trace=tz)
    assert "TRACE_HASH_MISMATCH" in _codes(rep)


def test_replay_catches_stale_cache_key(world, monkeypatch):
    w = world
    rec, files, refs = w["units"]["K-JOINT-SINGLE"]
    # (a) trace level: a move part carrying the PRE-move statistics hash of its target token (a stale key)
    tz = copy.deepcopy(files["trace.json"])
    z, g, k, m = _first_move(tz)
    p = m["parts"][0]
    r = p["r"]
    lab = np.asarray(g["start_labels"][str(r)])
    for mv in g["moves"][:k]:
        for q in mv["parts"]:
            if q["r"] == r:
                lab[q["f"]] = q["to_canon"]
                lab = MP.canon_labels(lab)
    fr = PT.load_fine(w["fine"])[r - 1]
    Pr, dr = w["T"][f"p{r}"][w["tr"]], w["T"][f"d{r}"][w["tr"]]
    tok = RL.canonical_tokens(fr, lab)
    n, Sx, Yx = DEC.token_stats(fr, tok, DEC.cell_label_counts(fr, Pr, dr, w["Y"][r]))
    t_to = int(tok[p["to_canon"]])
    stale_key = MP.stats_sha(n[t_to], Yx[t_to], Sx[t_to])          # the target token BEFORE the cell joined it
    assert stale_key != p["stats_sha256"][1]
    p["stats_sha256"][1] = stale_key
    rep = _replay(w, _retrace(rec, tz), files, refs, trace=tz)
    assert "STATS_HASH_MISMATCH" in _codes(rep) and not rep["checks"]["check8"]
    # (b) a real defect the search itself does not notice: a cached solve served for the right statistics key but
    # holding the target token's PRE-move solve (stale entry); the replay's fresh solve exposes it. On a constrained
    # arm the mapper's own final deployed-feasibility guard may already refuse such a fit, so the weighted SEQ arm
    # (no hard budget, hence no such guard) is used: only the replay can see the defect there.
    real = MP.State.ensure

    def stale(self, r, c, cells, slots, verify=False):
        out = real(self, r, c, cells, slots, verify=verify)
        if verify and len(slots) == 2:                       # the apply() lookup of (f -> a, f -> b)
            n, Y, S, Q, ll, br = out
            b = int(slots[1])
            Q, ll, br = Q.copy(), ll.copy(), br.copy()
            Q[0, 1], ll[0, 1], br[0, 1] = self.tQ[r][b], self.tll[r][b], self.tbr[r][b]
            out = (n, Y, S, Q, ll, br)
        return out
    monkeypatch.setattr(MP.State, "ensure", stale)
    starts = {**w["ct"], RN.d0_id("SEQ-12", LAM): w["src"][RN.d0_id("SEQ-12", LAM)]}
    r2, f2 = MP.fit_unit("W-SEQ-12", w["fine"], w["T"], w["tr"], w["Y"], w["S"], lam=LAM, starts=starts,
                         refs=w["refs0"], meta=META)
    monkeypatch.setattr(MP.State, "ensure", real)
    assert r2["deployed"]["parity_ok"] and r2["trace_counts"]["moves"] > 0   # the fresh final release looks fine
    rep = _replay(w, r2, f2, w["refs0"])
    assert "CACHED_SOLVE_MISMATCH" in _codes(rep) and not rep["checks"]["cached_solves"] and not rep["ok"]
    assert not rep["checks"]["check8"]
    # the honest unit of the same arm replays cleanly
    rec0, files0, refs0 = w["units"]["W-SEQ-12"]
    assert _replay(w, rec0, files0, refs0)["ok"]


def test_replay_catches_infeasible_paired_update(world, monkeypatch):
    w = world
    # a defective paired step: the own-budget pool screen and the joint re-check are both skipped
    real_pair, real_feas, real_cell = MP._pair_step, MP.feasible, MP._cell

    def loose_cell(*a, **k):
        res = real_cell(*a, **k)
        if res is not None:
            res["own_feasible"] = np.ones(res["hood"].size, dtype=bool)
        return res

    def bad_pair(*a, **k):
        MP.feasible, MP._cell = (lambda pb, t, recips: True), loose_cell
        try:
            return real_pair(*a, **k)
        finally:
            MP.feasible, MP._cell = real_feas, real_cell
    monkeypatch.setattr(MP, "_pair_step", bad_pair)
    rec, files = MP.fit_unit("K-JOINT-PAIR", w["fine"], w["T"], w["tr"], w["Y"], w["S"], witnesses=w["wit"],
                             refs=w["refs"], meta=META)
    monkeypatch.undo()
    assert rec["work"]["pair_accepted"] >= 1
    rep = _replay(w, rec, files, w["refs"])
    assert "PAIR_INFEASIBLE" in _codes(rep) and not rep["checks"]["check6"] and not rep["ok"]
    bad = [v for v in rep["violations"] if v["code"] == "PAIR_INFEASIBLE"]
    tz = files["trace.json"]
    for v in bad:
        g = [g for z in tz["starts"] if z["name"] == v["start"] for g in z["stages"]][0]
        assert g["moves"][v["step"]]["type"] == "pair"
    # the honest unit passes the same check
    rec0, files0, refs0 = w["units"]["K-JOINT-PAIR"]
    assert _replay(w, rec0, files0, refs0)["checks"]["check6"]


def test_replay_catches_partner_wrongly_required_feasible(world, monkeypatch):
    w = world
    rec, files, refs = w["units"]["K-SEQ-12"]
    # (a) trace level: stage 1 recorded as enforcing the partner's constraints
    tz = copy.deepcopy(files["trace.json"])
    tz["starts"][0]["stages"][0]["enforced"] = [1, 2]
    rep = _replay(w, _retrace(rec, tz), files, refs, trace=tz)
    assert "PARTNER_REQUIRED_FEASIBLE" in _codes(rep) and not rep["checks"]["check7"]
    # (b) a real defect: stage 1 requires the temporary CLASS-ONLY partner to be feasible
    real = MP.feasible

    def strict(pb, t, recips):
        return real(pb, t, (1, 2) if tuple(recips) == (1,) else recips)
    monkeypatch.setattr(MP, "feasible", strict)
    starts = {**w["ct"], RN.d0_id("SEQ-12", 0.01): w["src"][RN.d0_id("SEQ-12", 0.01)]}
    r2, f2 = MP.fit_unit("K-SEQ-12", w["fine"], w["T"], w["tr"], w["Y"], w["S"], starts=starts, refs=w["refs"],
                         meta=META)
    monkeypatch.setattr(MP, "feasible", real)
    assert all(s["stage1"]["status"] == "INFEASIBLE_START" for s in r2["starts"])    # the baseline made impossible
    rep = _replay(w, r2, f2, w["refs"])
    assert "PARTNER_REQUIRED_FEASIBLE" in _codes(rep) and not rep["checks"]["check7"] and not rep["ok"]
    # (c) the partner is not the CLASS-ONLY view
    tz = copy.deepcopy(files["trace.json"])
    g1 = tz["starts"][0]["stages"][0]
    g1["partner"]["labels"] = tz["starts"][0]["labels"]["2"]
    rep = _replay(w, _retrace(rec, tz), files, refs, trace=tz)
    assert "PARTNER_NOT_CLASS_ONLY" in _codes(rep)


def test_replay_catches_ineligible_witness_and_final_mismatch(world):
    rec, files, refs = world["units"]["K-JOINT-SINGLE"]
    tz = files["trace.json"]
    exc = [k for k, z in enumerate(tz["starts"]) if z["joint_witness"]["status"] == "EXCLUDED_INFEASIBLE_WITNESS"]
    assert exc, "the fixture must have an infeasible (excluded) witness"
    cls = [z for z in tz["starts"] if z["name"] == RN.d0_id("CLASS")]
    assert cls and (cls[0]["joint_witness"]["feasible_all"] or
                    cls[0]["joint_witness"]["status"] == "EXCLUDED_INFEASIBLE_WITNESS")
    t2 = copy.deepcopy(tz)
    t2["starts"][exc[0]]["eligible"] = True
    rep = _replay(world, _retrace(rec, t2), files, refs, trace=t2)
    assert "INFEASIBLE_WITNESS_ELIGIBLE" in _codes(rep) and not rep["checks"]["witnesses"]
    r2 = copy.deepcopy(rec)
    r2["final_state_terms"]["L1"] += 1e-9
    rep = _replay(world, r2, files, refs)
    assert "FINAL_STATE_MISMATCH" in _codes(rep)
    f2 = dict(files)
    rel = {k: np.array(v, copy=True) for k, v in files["release.npz"].items()}
    i = int(world["tr"][0])
    rel["q1"][i] = np.nextafter(rel["q1"][i], 2.0)
    f2["release.npz"] = rel
    rep = _replay(world, rec, f2, refs)
    assert "RELEASE_MISMATCH" in _codes(rep)


def test_replay_requires_registered_witness_order(world):
    rec, files, refs = world["units"]["K-JOINT-PAIR"]
    tz = copy.deepcopy(files["trace.json"])
    r2 = _retrace(rec, tz)
    k = [z["name"] for z in tz["starts"]].index(RN.d0_id("CLASS"))
    tz["starts"][k], tz["starts"][k - 1] = tz["starts"][k - 1], tz["starts"][k]
    r2["starts_given"] = [z["name"] for z in tz["starts"]]
    r2["trace_sha256"] = MP.json_sha(tz)
    rep = _replay(world, r2, files, refs, trace=tz)
    assert "START_MISMATCH" in _codes(rep)


def test_pair_step_stops_at_its_ceiling_share(world):
    """F R-4: the pool screening and the pair evaluations both count toward and stop at the remaining share."""
    pb = _problem(world)
    pb.capI = {1: world["refs"]["I_ctask"]["1"], 2: world["refs"]["I_ctask"]["2"]}
    w = MP.weights("K-JOINT-PAIR", None)
    st = MP.State(pb, _labels(world["ctask"][1]["policy.json"]))
    cur = st.terms()
    e0 = st.ct["evals"]
    ok, t, v, info = MP._pair_step(st, w, (1, 2), cur, MP.objective(cur, w), budget_left=10 ** 9)
    full = st.ct["evals"] - e0
    screen = st.ct["pair_pool_screened"]
    assert not info["hit_ceiling"] and full == screen + st.ct["pair_evals"]
    for budget in (7, screen + 5):
        st = MP.State(pb, _labels(world["ctask"][1]["policy.json"]))
        e0 = st.ct["evals"]
        ok, t, v, info = MP._pair_step(st, w, (1, 2), cur, MP.objective(cur, w), budget_left=budget)
        assert info["hit_ceiling"] and st.ct["evals"] - e0 <= budget + max(CAPS) - 1
        if budget == 7:
            assert st.ct["pair_evals"] == 0 and info["proposals"] is None
        else:
            assert st.ct["pair_evals"] <= 5 + max(CAPS) - 1
