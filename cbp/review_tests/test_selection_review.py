"""Role C (statistics and selection reviewer) tests for the confidence-budgeted privacy study (cbp).

Owned by role C; lives in cbp/review_tests/, which no lock globs. Synthetic fixtures only: no real data, no fits, no
assessment labels. Run under the shared semaphore:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.sema \
        --label C:selection-review -- ~/PCRL/.venv/bin/python -m pytest -q cbp/review_tests/test_selection_review.py

Structure:
  1. ORACLE: an independent transcription of the execution prompt (sections 9 and 11), written from the prompt text and
     NOT from cbp/qpc code. Eligibility, headroom, the three shortfalls, comparators, guards, ordering, nominees,
     fallbacks, the clause classifier and the claim/overall labels.
  2. Oracle self-tests on hand-computed fixtures, and "defect discrimination" checks: each deliberate defect of prompt
     section 14 that falls in the selection/inference scope changes the oracle's answer on a named fixture, so a
     conformance test on that fixture would catch the defect.
  3. Conformance tests of the lead's cbp.select / cbp.family / cbp.infer and the registered JSON against the oracle
     (skipped until the lead's module or file exists).
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "results" / "pcrl_confidence_budgeted_privacy_v1"

# =====================================================================================================================
# 1. ORACLE (prompt text transcription)
# =====================================================================================================================
SEEDS = (0, 1, 2)
TASKS = ("income", "occupation")
LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
PRIV_FAMS = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")

# section 9 ordinary eligibility (source rule) and the NEW headroom rule
ALLOW_ACC, ALLOW_LL, ALLOW_BRIER, RETAIN, MIN_GAIN = 0.01, 0.01, 0.005, 0.8, 0.03
HEAD_LL, HEAD_BRIER = 0.006, 0.0035
GUARD = 0.005

# section 11 family
FAMILY_SIZE = 37
ALPHA = 0.05
B_REPS = 1999
BOOT_SEED = 20261008


def cid(fam, lam=None):
    if fam == "CLASS":
        return "U|CLASS|i1o1"
    return f"U|{fam}|i8o64" + (f"|l{lam:g}" if fam in PRIV_FAMS else "")


DIRECT, FINE, CLASS = cid("DIRECT-TASK"), cid("FINE-TASK"), cid("CLASS")
SRC_U, RAWJ = "SRC|U", "SRC|RAW-J_b0.3"
REF_E, REF_F, REF_F0 = "REF|E", "REF|F", "REF|F0"
PRIV = [cid(f, lam) for f in PRIV_FAMS for lam in LAMS]                       # 24 privacy configurations
JOINTS = [cid("JOINT", lam) for lam in LAMS]
BANK = PRIV + [DIRECT, FINE, CLASS, SRC_U, RAWJ, REF_E, REF_F, REF_F0]        # 32 configurations per seed
# closed lists (prompt section 9)
T_LIST = [DIRECT, FINE, SRC_U, CLASS, REF_F0]                                 # RAW-J EXCLUDED (correction disclosed)
C_RATE_LIST = [DIRECT, FINE] + [cid(f, lam) for f in ("LOCAL", "SEQ-12", "SEQ-21") for lam in LAMS]
C_GLOBAL_LIST = C_RATE_LIST + [CLASS, SRC_U, RAWJ, REF_E, REF_F, REF_F0]
CONTINUOUS = {SRC_U, RAWJ, REF_E, REF_F, REF_F0}


def fam_of(c):
    if c.startswith(("SRC|", "REF|")):
        return c.split("|")[0]
    return c.split("|")[1]


def z_family():
    """z for the fixed 37-slot nominal family, computed by bisection on the normal upper tail (math.erfc), not by
    copying the number from the prompt."""
    a = ALPHA / (2 * FAMILY_SIZE)
    lo, hi = 0.0, 10.0
    for _ in range(200):
        m = (lo + hi) / 2
        if 0.5 * math.erfc(m / math.sqrt(2)) > a:
            lo = m
        else:
            hi = m
    return lo


def _finite(*xs):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in xs)


def seed_status(rec, U):
    """One seed, one configuration. rec/U: {"u": {task: {acc, logloss, brier, const_acc}}, "preserved": {1, 2},
    "auc": {v1, v2, pair}}. Returns None if non-estimable or decision failure (separate INVALID status), else the
    per-task excesses and pass flags."""
    if not all(rec["preserved"].get(i) is True for i in (1, 2)):
        return {"invalid": "DECISION_FAILURE"}
    vals = [rec["auc"][w] for w in ("v1", "v2", "pair")]
    for t in TASKS:
        vals += [rec["u"][t][x] for x in ("acc", "logloss", "brier", "const_acc")]
        vals += [U["u"][t][x] for x in ("acc", "logloss", "brier", "const_acc")]
    if not _finite(*vals):
        return {"invalid": "NON_ESTIMABLE"}
    out = {"invalid": None, "tasks": {}}
    for t in TASKS:
        c, u = rec["u"][t], U["u"][t]
        a, aU = c["acc"], u["acc"]
        g, gU = c["acc"] - c["const_acc"], u["acc"] - u["const_acc"]
        dll, db = c["logloss"] - u["logloss"], c["brier"] - u["brier"]
        ordinary = (a >= aU - ALLOW_ACC and c["logloss"] <= u["logloss"] + ALLOW_LL
                    and c["brier"] <= u["brier"] + ALLOW_BRIER and g >= RETAIN * gU and g >= MIN_GAIN)
        o_sf = max(0.0, (aU - ALLOW_ACC - a) / ALLOW_ACC, (dll - ALLOW_LL) / ALLOW_LL, (db - ALLOW_BRIER) / ALLOW_BRIER,
                   (RETAIN * gU - g) / max(RETAIN * gU, MIN_GAIN), (MIN_GAIN - g) / MIN_GAIN)
        h_sf = max(0.0, (dll - HEAD_LL) / HEAD_LL, (db - HEAD_BRIER) / HEAD_BRIER)
        out["tasks"][t] = {"ordinary": ordinary, "headroom": dll <= HEAD_LL and db <= HEAD_BRIER,
                           "dll": dll, "db": db, "ordinary_sf": o_sf, "headroom_sf": h_sf}
    return out


def config_row(c, recs):
    """recs: {(k, cid): seed record}. Every-seed, every-task aggregation; never a seed mean for eligibility."""
    U = {k: recs.get((k, SRC_U)) for k in SEEDS}
    per, invalid = {}, []
    for k in SEEDS:
        r = recs.get((k, c))
        if r is None or U[k] is None:
            invalid.append((k, "MISSING"))
            continue
        s = seed_status(r, U[k])
        if s["invalid"]:
            invalid.append((k, s["invalid"]))
            continue
        per[k] = s
    row = {"config": c, "invalid": invalid}
    if invalid:
        return row
    rs = [recs[(k, c)] for k in SEEDS]
    st = [r.get("states") for r in rs]
    row.update({
        "ordinary": all(per[k]["tasks"][t]["ordinary"] for k in SEEDS for t in TASKS),
        "headroom": all(per[k]["tasks"][t]["headroom"] for k in SEEDS for t in TASKS),
        "ordinary_sf": max(per[k]["tasks"][t]["ordinary_sf"] for k in SEEDS for t in TASKS),
        "headroom_sf": max(per[k]["tasks"][t]["headroom_sf"] for k in SEEDS for t in TASKS),
        "mean_pair": sum(r["auc"]["pair"] for r in rs) / 3,
        "mean_sum_ll": sum(r["u"]["income"]["logloss"] + r["u"]["occupation"]["logloss"] for r in rs) / 3,
        "mean_states": math.inf if any(s is None for s in st) else sum(st) / 3,
        "auc": {k: dict(recs[(k, c)]["auc"]) for k in SEEDS},
    })
    return row


def order_key(row, tol_decimals=12):
    return (round(row["mean_pair"], tol_decimals), round(row["mean_sum_ll"], tol_decimals), row["mean_states"],
            row["config"])


def guard_ok(row, g):
    return all(row["auc"][k][w] <= g["auc"][k][w] + GUARD for k in SEEDS for w in ("v1", "v2"))


def guard_sf(row, guards):
    xs = [(row["auc"][k][w] - g["auc"][k][w] - GUARD) / GUARD for g in guards for k in SEEDS for w in ("v1", "v2")]
    return max([0.0] + xs)


def strongest(rows, cands, pred):
    if any(rows[c]["invalid"] for c in cands):
        return {"status": "INVALID", "config": None, "invalid": [c for c in cands if rows[c]["invalid"]]}
    el = [rows[c] for c in cands if pred(rows[c])]
    if not el:
        return {"status": "NO_ELIGIBLE", "config": None}
    return {"status": "SELECTED", "config": min(el, key=order_key)["config"]}


def fallback(rows, cands, guards):
    """Fixed deterministic fallback (prompt section 9). guards: list of guard rows, or None when a required
    comparator is missing (no guard-based rank; dependency INVALID)."""
    valid = [rows[c] for c in cands if not rows[c]["invalid"]]
    if guards is None:
        return {"status": "INVALID", "reason": "required guard comparator missing; no guard-based rank",
                "config": None}
    if not valid:
        return {"status": "INVALID", "config": None}
    best = min(valid, key=lambda r: (round(r["ordinary_sf"], 12), round(r["headroom_sf"], 12),
                                     round(guard_sf(r, guards), 12)) + order_key(r))
    return {"status": "DESCRIPTIVE_ONLY", "config": best["config"], "ordinary_sf": best["ordinary_sf"],
            "headroom_sf": best["headroom_sf"], "guard_sf": guard_sf(best, guards)}


def select(recs, *, headroom=True, seed_mean=False, t_list=T_LIST):
    """Oracle selection. headroom=False / seed_mean=True / t_list=... produce the DEFECTIVE variants used only to show
    that the fixtures discriminate (prompt section 14)."""
    rows = {c: config_row(c, recs) for c in BANK}
    if seed_mean:                              # DEFECT: eligibility on the seed-mean metrics
        for c, r in rows.items():
            if r["invalid"]:
                continue
            m = {k: recs[(k, c)] for k in SEEDS}
            mU = {k: recs[(k, SRC_U)] for k in SEEDS}
            avg = lambda d, t, x: sum(d[k]["u"][t][x] for k in SEEDS) / 3   # noqa: E731
            ok, hd = True, True
            for t in TASKS:
                hd &= (avg(m, t, "logloss") - avg(mU, t, "logloss") <= HEAD_LL
                       and avg(m, t, "brier") - avg(mU, t, "brier") <= HEAD_BRIER)
                a, aU = avg(m, t, "acc"), avg(mU, t, "acc")
                g, gU = a - avg(m, t, "const_acc"), aU - avg(mU, t, "const_acc")
                ok &= (a >= aU - ALLOW_ACC and avg(m, t, "logloss") <= avg(mU, t, "logloss") + ALLOW_LL
                       and avg(m, t, "brier") <= avg(mU, t, "brier") + ALLOW_BRIER and g >= RETAIN * gU
                       and g >= MIN_GAIN)
            r["ordinary"], r["headroom"] = ok, hd
    ordn = lambda r: r["ordinary"]                                              # noqa: E731
    head = (lambda r: r["ordinary"] and r["headroom"]) if headroom else ordn   # noqa: E731
    out = {"rows": rows}
    out["T*"] = strongest(rows, t_list, ordn)
    out["C_rate"] = strongest(rows, C_RATE_LIST, ordn)
    out["C_global"] = strongest(rows, C_GLOBAL_LIST, ordn)
    T = rows[out["T*"]["config"]] if out["T*"]["config"] else None
    CR = rows[out["C_rate"]["config"]] if out["C_rate"]["config"] else None
    CG = rows[out["C_global"]["config"]] if out["C_global"]["config"] else None
    if T is None:
        out["P*"] = {"status": "INVALID", "config": None, "reason": "T* missing"}
    else:
        out["P*"] = strongest(rows, PRIV, lambda r: head(r) and guard_ok(r, T))
        if out["P*"]["status"] == "NO_ELIGIBLE":
            out["P*"]["fallback"] = fallback(rows, PRIV, [T])
    if CR is None or CG is None:
        out["J*"] = {"status": "INVALID", "config": None, "reason": "C_rate or C_global missing"}
    else:
        out["J*"] = strongest(rows, JOINTS, lambda r: head(r) and guard_ok(r, CR) and guard_ok(r, CG))
        if out["J*"]["status"] == "NO_ELIGIBLE":
            out["J*"]["fallback"] = fallback(rows, JOINTS, [CR, CG])
    # prespecified diagnostics (variant registered by the lead is checked in section 3)
    if T is not None:
        out["standard_privacy_winner_guarded"] = strongest(rows, PRIV, lambda r: r["ordinary"] and guard_ok(r, T))
        out["family_headroom_winner"] = {
            f: strongest(rows, [cid(f, lam) for lam in LAMS], lambda r: head(r) and guard_ok(r, T)) for f in PRIV_FAMS}
    out["standard_privacy_winner_unguarded"] = strongest(rows, PRIV, ordn)
    return out


# ---------------------------------------------------------------- clauses and labels
def clause_class(side, target, point, lo, hi):
    """PASS; NOT_ESTABLISHED_PRECISION (point on the passing side, bound fails: assessment precision failure);
    NOT_ESTABLISHED_POINT (point on the failing side, interval still covers the target); VIOLATION_ESTABLISHED (the whole
    interval lies on the failing side: a measured violation supported by a bound); INVALID (nonfinite)."""
    if not _finite(point, lo, hi):
        return "INVALID"
    if side == "lower>":
        if lo > target:
            return "PASS"
        if hi <= target:
            return "VIOLATION_ESTABLISHED"
        return "NOT_ESTABLISHED_PRECISION" if point > target else "NOT_ESTABLISHED_POINT"
    if hi < target:
        return "PASS"
    if lo >= target:
        return "VIOLATION_ESTABLISHED"
    return "NOT_ESTABLISHED_PRECISION" if point < target else "NOT_ESTABLISHED_POINT"


def claim_label(nominee, comparator, clauses, *, controls_ok=True, work_complete=True):
    """nominee: SELECTED / NO_ELIGIBLE / INVALID; comparator: SELECTED / NO_ELIGIBLE / INVALID; clauses: 11 decisions
    (PASS / NOT_ESTABLISHED / INVALID). Prompt section 11: missing/invalid comparator, nonfinite primary quantity, failed
    required control, incomplete required work -> INCOMPLETE_OR_INVALID; no eligible nominee ->
    NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE; a missing comparator is never a completed head-to-head negative; an ineligible
    fallback never passes."""
    if not controls_ok or not work_complete or nominee == "INVALID" or comparator == "INVALID":
        return "INCOMPLETE_OR_INVALID"
    if nominee == "NO_ELIGIBLE":              # precedence over a NO_ELIGIBLE comparator (registered; see SEL-N2)
        return "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
    if comparator == "NO_ELIGIBLE":
        return "INCOMPLETE_OR_INVALID"
    if any(c == "INVALID" for c in clauses):
        return "INCOMPLETE_OR_INVALID"
    return "PASS" if clauses and all(c == "PASS" for c in clauses) else "NOT_ESTABLISHED"


def overall(claims, q, family=None):
    """claims: {"A","B","C"} -> claim_label; q: PASS / NOT_ESTABLISHED / INCOMPLETE_OR_INVALID."""
    labels = []
    if claims["C"] == "PASS":
        labels.append(f"PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET ({family})")
    if claims["A"] == "PASS" and claims["B"] == "PASS":
        labels.append("JOINT_DEVELOPMENT_CRITERION_MET")
    incomplete = sorted(c for c, s in claims.items() if s == "INCOMPLETE_OR_INVALID")
    if q == "INCOMPLETE_OR_INVALID":
        incomplete.append("Q")
    if labels:
        return labels, incomplete
    if incomplete:
        return ["INCOMPLETE_OR_INVALID"], incomplete
    if q == "PASS":
        return ["CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"], []
    return ["EXPERIMENTAL_NO_ADVANTAGE"], []


# =====================================================================================================================
# synthetic bank
# =====================================================================================================================
U_ANCHOR = {"income": {"acc": 0.845, "logloss": 0.330, "brier": 0.225, "const_acc": 0.760},
            "occupation": {"acc": 0.420, "logloss": 1.420, "brier": 0.700, "const_acc": 0.270}}


def _u(dll=(0.0, 0.0), db=(0.0, 0.0), dacc=(0.0, 0.0)):
    out = {}
    for j, t in enumerate(TASKS):
        out[t] = dict(U_ANCHOR[t])
        out[t]["logloss"] += dll[j]
        out[t]["brier"] += db[j]
        out[t]["acc"] += dacc[j]
    return out


def make_bank(seed=0, *, p_bad=0.25, joint_bonus=0.0):
    """Every configuration on every seed; privacy codes spread across the confidence range around the 0.006 headroom
    and 0.01 ordinary limits; decisions preserved, finite."""
    rng = np.random.default_rng(seed)
    recs = {}
    for c in BANK:
        f = fam_of(c)
        lam = float(c.rsplit("|l", 1)[1]) if f in PRIV_FAMS else 0.0
        base = {"SRC": 0.858, "REF": 0.80, "CLASS": 0.739, "DIRECT-TASK": 0.849, "FINE-TASK": 0.846}.get(f, 0.85)
        jb = joint_bonus if f == "JOINT" else 0.0
        if f in PRIV_FAMS:
            base = 0.846 - 0.35 * lam - (0.004 if f == "JOINT" else 0.0) - jb
        for k in SEEDS:
            if c == SRC_U:
                u = _u()
            else:
                ll = float(rng.uniform(0.0, 0.004)) + (0.06 * lam if f in PRIV_FAMS else 0.0)
                if rng.random() < p_bad and c not in (DIRECT, FINE):
                    ll += float(rng.uniform(0.0, 0.006))
                ll += {CLASS: 0.16, REF_E: 0.16, REF_F: 0.06, REF_F0: 0.02, RAWJ: 0.013}.get(c, 0.0)  # source-like
                u = _u(dll=(float(rng.uniform(0, 0.003)), ll), db=(float(rng.uniform(0, 0.002)),
                                                                     float(rng.uniform(0, 0.004))))
            pair = base + float(rng.normal(0, 0.004))
            recs[(k, c)] = {"auc": {"v1": 0.70 + float(rng.normal(0, 0.004)) - (0.1 * lam if f in PRIV_FAMS else 0) - jb,
                                    "v2": pair - 0.04 + float(rng.normal(0, 0.004)), "pair": pair},
                            "u": u, "preserved": {1: True, 2: True},
                            "states": None if c in CONTINUOUS else (2 if c == CLASS else int(rng.integers(60, 144)))}
    return recs


def clone(recs):
    return {k: json.loads(json.dumps(v)) | {"preserved": dict(v["preserved"])} for k, v in recs.items()}


# =====================================================================================================================
# 2. oracle self-tests and defect discrimination
# =====================================================================================================================
def test_z_is_verified_not_copied():
    z = NormalDist().inv_cdf(1 - ALPHA / (2 * FAMILY_SIZE))
    assert abs(z - z_family()) < 1e-12
    assert repr(z) == "3.2048452050105634"
    # the plausible wrong critical values are far away (defect: wrong family critical value)
    for wrong in (NormalDist().inv_cdf(1 - 0.05 / 37), NormalDist().inv_cdf(1 - 0.05 / 148),
                  NormalDist().inv_cdf(1 - 0.05 / 66), NormalDist().inv_cdf(0.975)):
        assert abs(wrong - z) > 0.01


def test_shortfall_formulas_hand_computed():
    U = {"u": _u(), "preserved": {1: True, 2: True}, "auc": {"v1": .7, "v2": .8, "pair": .85}}
    # occupation dLL 0.012 (ordinary sf 0.2, headroom sf 1.0); income Brier excess 0.0040 (headroom sf 1/7)
    r = {"u": _u(dll=(0.0, 0.012), db=(0.004, 0.0)), "preserved": {1: True, 2: True}, "auc": U["auc"]}
    s = seed_status(r, U)
    assert s["tasks"]["occupation"]["ordinary_sf"] == pytest.approx(0.2)
    assert s["tasks"]["occupation"]["headroom_sf"] == pytest.approx(1.0)
    assert s["tasks"]["income"]["headroom_sf"] == pytest.approx((0.004 - 0.0035) / 0.0035)
    assert s["tasks"]["income"]["ordinary_sf"] == 0.0 and s["tasks"]["income"]["ordinary"]
    # accuracy shortfall: a = aU - 0.015 -> 0.5; gain shortfall uses max(0.8 gU, 0.03)
    r2 = {"u": _u(dacc=(-0.015, 0.0)), "preserved": {1: True, 2: True}, "auc": U["auc"]}
    gU = U_ANCHOR["income"]["acc"] - U_ANCHOR["income"]["const_acc"]
    g = gU - 0.015
    exp = max(0.5, (0.8 * gU - g) / max(0.8 * gU, 0.03))
    assert seed_status(r2, U)["tasks"]["income"]["ordinary_sf"] == pytest.approx(exp)
    # decision failure is a separate INVALID status, never a large finite shortfall
    r3 = {"u": _u(), "preserved": {1: True, 2: False}, "auc": U["auc"]}
    assert seed_status(r3, U)["invalid"] == "DECISION_FAILURE"
    r4 = {"u": _u(dll=(float("nan"), 0.0)), "preserved": {1: True, 2: True}, "auc": U["auc"]}
    assert seed_status(r4, U)["invalid"] == "NON_ESTIMABLE"


def test_ordinary_shortfall_is_zero_iff_ordinarily_eligible():
    rng = np.random.default_rng(7)
    U = {"u": _u(), "preserved": {1: True, 2: True}, "auc": {"v1": .7, "v2": .8, "pair": .85}}
    for _ in range(4000):
        r = {"u": _u(dll=tuple(rng.uniform(-0.002, 0.014, 2)), db=tuple(rng.uniform(-0.001, 0.007, 2)),
                     dacc=tuple(rng.uniform(-0.014, 0.002, 2))), "preserved": {1: True, 2: True}, "auc": U["auc"]}
        s = seed_status(r, U)
        for t in TASKS:
            assert (s["tasks"][t]["ordinary_sf"] == 0.0) == s["tasks"][t]["ordinary"]
            assert (s["tasks"][t]["headroom_sf"] == 0.0) == s["tasks"][t]["headroom"]


def test_clause_classifier_distinguishes_precision_point_and_established_violation():
    # the source's occupation result: point 0.008072, upper 0.012122 vs 0.01 -> precision failure, not a violation
    assert clause_class("upper<", 0.01, 0.008072, 0.004022, 0.012122) == "NOT_ESTABLISHED_PRECISION"
    assert clause_class("upper<", 0.01, 0.011, 0.007, 0.015) == "NOT_ESTABLISHED_POINT"
    assert clause_class("upper<", 0.01, 0.014, 0.0105, 0.0175) == "VIOLATION_ESTABLISHED"
    assert clause_class("upper<", 0.01, 0.004, 0.001, 0.007) == "PASS"
    assert clause_class("lower>", 0.02, 0.0336, 0.0286, 0.0386) == "PASS"
    assert clause_class("lower>", 0.02, 0.025, 0.015, 0.035) == "NOT_ESTABLISHED_PRECISION"
    assert clause_class("lower>", 0.02, 0.003, -0.004, 0.010) == "VIOLATION_ESTABLISHED"
    assert clause_class("lower>", -0.01, 0.0, 0.0, 0.0) == "PASS"            # zero-variance accuracy identity
    assert clause_class("upper<", 0.01, float("nan"), 0.0, 0.0) == "INVALID"


def test_label_oracle_mixed_and_absent_cases():
    P, N = ["PASS"] * 11, ["PASS"] * 10 + ["NOT_ESTABLISHED"]
    # absent comparator never passes, even with all-pass numbers
    assert claim_label("SELECTED", "INVALID", P) == "INCOMPLETE_OR_INVALID"
    assert claim_label("SELECTED", "NO_ELIGIBLE", P) == "INCOMPLETE_OR_INVALID"
    # absent nominee is a completed negative, never a pass through fallback numbers
    assert claim_label("NO_ELIGIBLE", "SELECTED", P) == "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
    assert claim_label("SELECTED", "SELECTED", N) == "NOT_ESTABLISHED"
    assert claim_label("SELECTED", "SELECTED", P[:10] + ["INVALID"]) == "INCOMPLETE_OR_INVALID"
    assert claim_label("SELECTED", "SELECTED", P, controls_ok=False) == "INCOMPLETE_OR_INVALID"
    # mixed valid/incomplete: C passes, A incomplete -> favourable label AND the gap displayed
    lab, inc = overall({"A": "INCOMPLETE_OR_INVALID", "B": "NOT_ESTABLISHED", "C": "PASS"}, "PASS", "SEQ-12")
    assert lab == ["PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (SEQ-12)"] and inc == ["A"]
    assert overall({"A": "NOT_ESTABLISHED", "B": "NOT_ESTABLISHED", "C": "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"},
                   "PASS")[0] == ["CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"]
    assert overall({"A": "NOT_ESTABLISHED", "B": "NOT_ESTABLISHED", "C": "NOT_ESTABLISHED"},
                   "NOT_ESTABLISHED")[0] == ["EXPERIMENTAL_NO_ADVANTAGE"]
    assert overall({"A": "PASS", "B": "INCOMPLETE_OR_INVALID", "C": "NOT_ESTABLISHED"},
                   "PASS") == (["INCOMPLETE_OR_INVALID"], ["B"])


def test_bank_dimensions_match_prompt():
    assert len(PRIV) == 24 and len(JOINTS) == 6 and len(BANK) == 32
    assert len(C_RATE_LIST) == 20 and len(C_GLOBAL_LIST) == 26 and RAWJ not in T_LIST and RAWJ in C_GLOBAL_LIST
    assert not set(JOINTS) & set(C_GLOBAL_LIST)
    assert len(PRIV) * len(SEEDS) == 72


# ---------------------------------------------------------------- named discriminating fixtures (section 14)
def fixture_seed_average():
    """A privacy code best on pair AUC whose occupation LL excess is 0.0115 on seed 2 and 0.002 on seeds 0-1: passes
    on the seed mean (0.0052, even headroom), fails every-seed eligibility."""
    recs = make_bank(11, p_bad=0.0)
    c = cid("SEQ-21", 0.06)
    for k in SEEDS:
        recs[(k, c)]["u"] = _u(dll=(0.0, 0.0115 if k == 2 else 0.002))
        recs[(k, c)]["auc"]["pair"] = 0.60
        recs[(k, c)]["auc"]["v1"] = recs[(k, c)]["auc"]["v2"] = 0.55
    return recs, c


def fixture_headroom_at_ordinary_limit():
    """A JOINT code (best pair AUC) with occupation LL excess 0.008 on every seed: ordinarily eligible, fails the
    0.006 headroom. Headroom at the ordinary limit (0.01) would nominate it."""
    recs = make_bank(12, p_bad=0.0)
    c = cid("JOINT", 0.08)
    for k in SEEDS:
        recs[(k, c)]["u"] = _u(dll=(0.001, 0.008), db=(0.0, 0.002))
        recs[(k, c)]["auc"]["pair"] = 0.60
        recs[(k, c)]["auc"]["v1"] = recs[(k, c)]["auc"]["v2"] = 0.55
    return recs, c


def fixture_comparator_fails_headroom_only():
    """The strongest nonjoint control (SEQ-12 0.1, the known strong source control) is ordinarily eligible but fails
    the new headroom (occupation dLL 0.008). It must stay C_rate/C_global; dropping it would hand JOINT an easy guard."""
    recs = make_bank(13, p_bad=0.0)
    c = cid("SEQ-12", 0.1)
    for k in SEEDS:
        recs[(k, c)]["u"] = _u(dll=(0.0, 0.008))
        recs[(k, c)]["auc"]["pair"] = 0.70
        recs[(k, c)]["auc"]["v1"] = 0.50
        recs[(k, c)]["auc"]["v2"] = 0.50
    return recs, c


def fixture_rawj_in_t_list():
    """RAW-J is ordinarily eligible and has the lowest pair AUC of the untrained list + RAW-J; with RAW-J (legacy list)
    T* changes."""
    recs = make_bank(14, p_bad=0.0)
    for k in SEEDS:
        recs[(k, RAWJ)]["u"] = _u(dll=(0.001, 0.003))
        recs[(k, RAWJ)]["auc"]["pair"] = 0.70
    return recs


def test_fixtures_discriminate_the_deliberate_defects():
    recs, c = fixture_seed_average()
    good, bad = select(recs), select(recs, seed_mean=True)
    assert good["rows"][c]["ordinary"] is False and bad["rows"][c]["ordinary"] is True
    assert good["P*"]["config"] != c and bad["P*"]["config"] == c

    recs, c = fixture_headroom_at_ordinary_limit()
    good = select(recs)
    assert good["rows"][c]["ordinary"] and not good["rows"][c]["headroom"]
    assert good["P*"]["config"] != c
    assert good["standard_privacy_winner_guarded"]["config"] == c          # diagnostic sees it
    assert good["J*"]["config"] != c

    recs, c = fixture_comparator_fails_headroom_only()
    good = select(recs)
    assert good["rows"][c]["ordinary"] and not good["rows"][c]["headroom"]
    assert good["C_rate"]["config"] == c and good["C_global"]["config"] == c

    recs = fixture_rawj_in_t_list()
    assert select(recs)["T*"]["config"] != RAWJ
    assert select(recs, t_list=T_LIST + [RAWJ])["T*"]["config"] == RAWJ


def test_guard_is_per_seed_per_recipient_inclusive():
    recs = make_bank(21, p_bad=0.0)
    out0 = select(recs)
    T = out0["rows"][out0["T*"]["config"]]
    c = cid("LOCAL", 0.04)
    for k in SEEDS:
        recs[(k, c)]["u"] = _u()
        recs[(k, c)]["auc"]["pair"] = 0.55
        for w in ("v1", "v2"):
            recs[(k, c)]["auc"][w] = T["auc"][k][w] + 0.004
    assert select(recs)["P*"]["config"] == c
    recs[(1, c)]["auc"]["v2"] = T["auc"][1]["v2"] + 0.0051          # one recipient on one seed
    out = select(recs)
    assert out["P*"]["config"] != c
    # the seed-mean guard would still pass (+0.0044 on average): per-seed is binding
    assert np.mean([recs[(k, c)]["auc"]["v2"] - T["auc"][k]["v2"] for k in SEEDS]) <= GUARD


def test_missing_comparator_gives_no_guard_rank():
    recs = make_bank(22)
    recs[(1, DIRECT)]["auc"]["v1"] = float("nan")        # one T* candidate non-estimable on one seed
    out = select(recs)
    assert out["rows"][DIRECT]["invalid"] == [(1, "NON_ESTIMABLE")]
    assert out["T*"]["status"] == "INVALID"
    assert out["P*"]["status"] == "INVALID" and "fallback" not in out["P*"]
    assert fallback(out["rows"], PRIV, None)["status"] == "INVALID"


def test_random_banks_basic_invariants():
    for s in range(40):
        recs = make_bank(100 + s)
        out = select(recs)
        rows = out["rows"]
        for role, lst in (("T*", T_LIST), ("C_rate", C_RATE_LIST), ("C_global", C_GLOBAL_LIST)):
            assert out[role]["config"] in lst
        if out["P*"]["config"]:
            r = rows[out["P*"]["config"]]
            assert r["ordinary"] and r["headroom"] and guard_ok(r, rows[out["T*"]["config"]])
            sw = out["standard_privacy_winner_guarded"]["config"]
            # headroom can only give up privacy (same guard, nested eligibility)
            assert order_key(rows[sw]) <= order_key(r)
        if out["J*"]["config"]:
            assert fam_of(out["J*"]["config"]) == "JOINT"


# =====================================================================================================================
# 3. conformance of the lead's implementation (skipped until it exists)
# =====================================================================================================================
def to_inner_records(recs):
    """Neutral synthetic bank -> inner__<unit> records in the qpc/cbp inner-record format read by the selection."""
    out = {}
    for (k, c), r in recs.items():
        u = {t: {**r["u"][t], "gain": r["u"][t]["acc"] - r["u"][t]["const_acc"]} for t in TASKS}
        out[(k, c)] = {"recovery": {"auc": dict(r["auc"])}, "utility": u,
                       "preserved": {str(i): bool(v) for i, v in r["preserved"].items()},
                       "token_states": r["states"], "config": c, "seed": k}
    return out


def _lead(name):
    return pytest.importorskip(f"cbp.{name}")


@pytest.fixture
def lead_select(monkeypatch, tmp_path):
    """Run cbp.select.select_all on synthetic inner records (cbp.run record access monkeypatched; outputs to tmp)."""
    RUN = _lead("run")
    SEL = _lead("select")

    def go(recs, extra=None, fingerprints=None):
        inner = to_inner_records(recs)
        if extra:
            for key, v in extra.items():
                inner[key].update(v)
        store = {f"inner__{RUN.unit_for(k, c)}": r for (k, c), r in inner.items()}
        for (k, c), fp in (fingerprints or {}).items():          # policy unit records (deployed-map fingerprints)
            store[RUN.unit_for(k, c)] = {"pair_fingerprint": fp}
        monkeypatch.setattr(RUN, "rec", lambda n: json.loads(json.dumps(store[n])))
        monkeypatch.setattr(RUN, "done", lambda n: n in store)
        monkeypatch.setattr(RUN, "RUN", tmp_path)
        monkeypatch.setattr(RUN, "PKG", tmp_path)
        monkeypatch.setattr(RUN, "event", lambda *a, **k: None)
        for mod in (SEL,):
            for attr in ("RUN", "PKG"):
                if hasattr(mod, attr) and isinstance(getattr(mod, attr), Path):
                    monkeypatch.setattr(mod, attr, tmp_path)
        return SEL.select_all()
    go.tmp = tmp_path
    return go


def _rules():
    p = PKG / "HEADROOM_SELECTION_RULES.json"
    if not p.exists():
        pytest.skip("HEADROOM_SELECTION_RULES.json not written yet")
    return json.loads(p.read_text())


def test_rules_json_registers_prompt_section_9_constants():
    R = _rules()
    txt = json.dumps(R)
    for s in ("0.006", "0.0035", "0.005", "U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "SRC|U", "U|CLASS|i1o1",
              "REF|F0", "EXCLUDED", "(dLL - 0.006)/0.006", "(dB - 0.0035)/0.0035",
              "(0.8*g_U - g)/max(0.8*g_U, 0.03)", "(0.03 - g)/0.03"):
        assert s in txt, s
    assert "RAW-J" not in R["roles"]["T*"]["candidates"]
    assert "NOT headroom" in R["roles"]["C_rate"]["eligibility"] and "NOT headroom" in R["roles"]["C_global"]["eligibility"]


def test_registered_tie_tolerance_text_matches_the_rounding_rule():
    """SEL-R1 fixture. The rules say keys 1-2 are 'compared after rounding to 12 decimal places (values within 5e-13 are
    tied)'. Rounding is a quantisation: two values 2e-14 apart can fall in different bins (not tied), and values
    8e-13 apart can share a bin. The parenthetical is a different rule from the executable one, and the independent
    verifier implements from the registered text."""
    tt = _rules()["ordering"]["tie_tolerance"]
    x, y = 0.81660000000049, 0.81660000000051          # |x - y| = 2e-14
    if "within 5e-13 are tied" in tt:
        assert round(x, 12) == round(y, 12), "registered text says tied; round(x, 12) says not tied"


# ---------------------------------------------------------------- cbp.family
_ROLE_MAP = {"SELECTED": "ELIGIBLE", "NO_ELIGIBLE": "NO_ELIGIBLE", "INVALID": "TECHNICAL_FAILURE"}
_OUT_MAP = {"PASS": "PASS", "NOT_ESTABLISHED_PRECISION": "NOT_ESTABLISHED_PRECISION",
            "NOT_ESTABLISHED_POINT": "NOT_ESTABLISHED_POINT", "VIOLATION_ESTABLISHED": "MEASURED_VIOLATION",
            "INVALID": "INVALID"}


def _claim_ids(F, claim):
    return [e["id"] for e in F.PRIMARY if e["claim"] == claim]


def test_family_slots_constants_and_independent_z():
    F = _lead("family")
    assert F.B == B_REPS == 1999 and F.BOOT_SEED == BOOT_SEED == 20261008
    assert abs(F.Z_PRIMARY - z_family()) < 1e-12 and F.PRIMARY_SIZE == 37
    assert [e["id"] for e in F.PRIMARY] == [f"P{i:02d}" for i in range(1, 38)]
    assert F.CLAIMS == {"A": ("J*", "C_rate"), "B": ("J*", "C_global"), "C": ("P*", "T*")}
    want = []
    for claim, (nom, ref) in F.CLAIMS.items():
        want += [("coalition", None, 0.02, "lower>", nom, ref), ("local", None, 0.01, "upper<", nom, ref),
                 ("local", None, 0.01, "upper<", nom, ref)]
        for kind, target, side in (("acc", -0.01, "lower>"), ("logloss", 0.01, "upper<"), ("brier", 0.005, "upper<"),
                                   ("retain", 0.0, "lower>")):
            want += [(kind, 0, target, side, nom, None), (kind, 1, target, side, nom, None)]
    want += [("logloss", 0, 0.01, "upper<", "Q", None), ("logloss", 1, 0.01, "upper<", "Q", None),
             ("brier", 0, 0.005, "upper<", "Q", None), ("brier", 1, 0.005, "upper<", "Q", None)]
    got = [(e["kind"], e.get("task"), e["target"], e["side"], e["nominee"], e.get("ref")) for e in F.PRIMARY]
    assert got == want
    assert [e["view"] for e in F.PRIMARY if e["kind"] == "local"] == ["v1", "v2"] * 3
    assert sum(e["claim"] == c for e in F.PRIMARY for c in "ABCQ") == 37
    assert {e["claim"]: 0 for e in F.PRIMARY} and [len(_claim_ids(F, c)) for c in "ABCQ"] == [11, 11, 11, 4]


def test_clause_outcome_matches_oracle_off_boundary():
    F = _lead("family")
    rng = np.random.default_rng(3)
    for _ in range(20000):
        side = ("lower>", "upper<")[int(rng.integers(2))]
        target = float(rng.choice([0.02, 0.01, -0.01, 0.0, 0.005]))
        pt = target + float(rng.normal(0, 0.01))
        se = abs(float(rng.normal(0, 0.004)))
        lo, hi = pt - 3.2 * se, pt + 3.2 * se
        if min(abs(lo - target), abs(hi - target), abs(pt - target)) < 1e-12:
            continue
        assert F.clause_outcome(side, target, pt, lo, hi) == _OUT_MAP[clause_class(side, target, pt, lo, hi)]
    # zero-variance structural identity (accuracy difference of a decision-preserving code): PASS, not INVALID
    assert F.clause_outcome("lower>", -0.01, 0.0, 0.0, 0.0) == "PASS"
    for bad in (float("nan"), float("inf"), None):
        assert F.clause_outcome("upper<", 0.01, bad, 0.0, 0.0) == "INVALID"
        assert F.clause_outcome("upper<", 0.01, 0.0, 0.0, bad) == "INVALID"
    # the source occupation result is a precision failure, never a measured violation
    assert F.clause_outcome("upper<", 0.01, 0.008072, 0.004022, 0.012122) == "NOT_ESTABLISHED_PRECISION"


def test_claim_status_matches_oracle_for_every_role_and_single_clause_failure():
    F = _lead("family")
    ids = _claim_ids(F, "C")
    patterns = [{i: "PASS" for i in ids}]
    for i in ids:                                            # every single failed clause, every failure kind
        for o in ("NOT_ESTABLISHED_PRECISION", "NOT_ESTABLISHED_POINT", "MEASURED_VIOLATION", "INVALID"):
            patterns.append({**{j: "PASS" for j in ids}, i: o})
    states = ("SELECTED", "NO_ELIGIBLE", "INVALID")
    for nom, comp, ctl in itertools.product(states, states, (True, False)):
        for pat in patterns:
            mine = claim_label(nom, comp, ["INVALID" if o == "INVALID" else "PASS" if o == "PASS" else
                                           "NOT_ESTABLISHED" for o in pat.values()], controls_ok=ctl)
            got = F.claim_status(_ROLE_MAP[nom], _ROLE_MAP[comp], dict(pat), required_control_ok=ctl)
            assert got[0] == mine, (nom, comp, ctl, pat, got)
            if got[0] == "NOT_ESTABLISHED":
                assert got[2] == sorted(i for i, o in pat.items() if o != "PASS")
                o = next(o for o in pat.values() if o != "PASS")
                assert got[1] == {"MEASURED_VIOLATION": "MEASURED_VIOLATION_SUPPORTED_BY_BOUND",
                                  "NOT_ESTABLISHED_PRECISION": "ASSESSMENT_PRECISION_FAILURE"}.get(
                    o, "CLAUSE_NOT_ESTABLISHED"), (o, got)


def test_q_status_distinguishes_measured_violation_and_refuses_empty():
    """SEL-R3 fixture. The prompt's classifier must distinguish 'assessment precision failure' from 'measured violation
    supported by a bound'. claim_status does; q_status maps a Q clause whose lower bound already exceeds the 0.01
    allowance to the generic CLAUSE_NOT_ESTABLISHED, and maps an EMPTY outcome set (Q resolved, nothing scored) to
    NOT_ESTABLISHED / ASSESSMENT_PRECISION_FAILURE instead of INCOMPLETE_OR_INVALID."""
    F = _lead("family")
    q = _claim_ids(F, "Q")
    viol = {**{i: "PASS" for i in q}, q[1]: "MEASURED_VIOLATION"}
    st = F.q_status(True, viol)
    assert st[0] == "NOT_ESTABLISHED" and st[1] == "MEASURED_VIOLATION_SUPPORTED_BY_BOUND", st
    assert F.q_status(True, {})[0] == "INCOMPLETE_OR_INVALID", F.q_status(True, {})
    assert F.q_status(True, {**{i: "PASS" for i in q}, q[0]: "NOT_ESTABLISHED_PRECISION"})[1] == \
        "ASSESSMENT_PRECISION_FAILURE"
    assert F.q_status(True, {i: "PASS" for i in q})[0] == "PASS"
    assert F.q_status(False, {i: "PASS" for i in q})[0] == "INCOMPLETE_OR_INVALID"


def test_claim_status_refuses_a_shrunken_conjunction():
    """SEL-C5 (adopted): a conjunction must be over the claim's registered slots (11 or 4)."""
    F = _lead("family")
    assert F.claim_status("ELIGIBLE", "ELIGIBLE", {"P23": "PASS"}, claim="C")[0] == "INCOMPLETE_OR_INVALID"
    assert F.claim_status("ELIGIBLE", "ELIGIBLE", {i: "PASS" for i in F.claim_ids("C")}, claim="C")[0] == "PASS"
    assert F.q_status(True, {"P34": "PASS"})[0] == "INCOMPLETE_OR_INVALID"
    # alias slots are kept: claim B always needs its 11 ids, including the P15-P22 structural aliases of A
    assert F.claim_status("ELIGIBLE", "ELIGIBLE", {i: "PASS" for i in F.claim_ids("B")[:3]}, claim="B")[1] == \
        "MISSING_SLOTS"


@pytest.mark.xfail(strict=False, reason="SEL-N3 NOTE: claim= is optional; a caller omitting it can still shrink")
def test_claim_status_requires_claim_argument():
    F = _lead("family")
    assert F.claim_status("ELIGIBLE", "ELIGIBLE", {"P23": "PASS"})[0] != "PASS"


def test_overall_label_matches_oracle_exhaustively():
    F = _lead("family")
    S = ("PASS", "NOT_ESTABLISHED", "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", "INCOMPLETE_OR_INVALID")
    for a, b, c, q in itertools.product(S, S, S, ("PASS", "NOT_ESTABLISHED", "INCOMPLETE_OR_INVALID")):
        claims = {"A": a, "B": b, "C": c}
        lab, shown = F.overall_label(claims, q, technical_valid=True, winning_family="SEQ-12")
        mine, inc = overall(claims, q, "SEQ-12")
        if c == "PASS" and a == b == "PASS":
            assert lab == "PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (SEQ-12) + JOINT_DEVELOPMENT_CRITERION_MET"
        else:
            assert lab == mine[0], (claims, q, lab, mine)
        assert shown == {**claims, "Q": q}                  # every claim always displayed (mixed claims shown)
        lab2, _ = F.overall_label(claims, q, technical_valid=False, winning_family="SEQ-12")
        assert lab2 == "INCOMPLETE_OR_INVALID"


# ---------------------------------------------------------------- cbp.select
_ST = {"NOMINEE": "SELECTED", "NO_ELIGIBLE_NOMINEE": "NO_ELIGIBLE", "NO_ELIGIBLE_COMPARATOR": "NO_ELIGIBLE",
       "INVALID_NOMINEE": "INVALID", "INVALID_COMPARATOR": "INVALID"}


def _cfg(s):
    return s.get("config") if s["status"] == "NOMINEE" else None


def _strict_json(p):
    def bad(x):
        raise AssertionError(f"nonfinite JSON token {x} in {p.name}")
    return json.loads(p.read_text(), parse_constant=bad)


def test_select_matches_oracle_on_random_banks(lead_select):
    seen = {"P*": 0, "J*": 0, "P*_fb": 0, "J*_fb": 0}
    for s in range(40):
        recs = make_bank(500 + s, p_bad=(0.1, 0.3, 0.6, 0.9)[s % 4], joint_bonus=(0.0, 0.02, 0.04)[(s // 4) % 3])
        out = lead_select(recs)
        ref = select(recs)
        st = out["statuses"]
        for role in ("T*", "C_rate", "C_global", "P*", "J*"):
            assert _ST[st[role]["status"]] == ref[role]["status"], (s, role, st[role], ref[role])
            assert _cfg(st[role]) == ref[role]["config"], (s, role)
        for role in ("P*", "J*"):
            if ref[role]["status"] == "NO_ELIGIBLE":
                assert st[role]["descriptive_config"] == ref[role]["fallback"]["config"], (s, role)
                seen[role + "_fb"] += 1
            seen[role] += ref[role]["status"] == "SELECTED"
        d = out["diagnostics"]
        assert _cfg(d["ordinary_privacy_winner_no_headroom"]) == ref["standard_privacy_winner_guarded"]["config"]
        for f in PRIV_FAMS:
            assert _cfg(d["family_headroom_winners"][f]) == ref["family_headroom_winner"][f]["config"], (s, f)
        assert st["Q"]["status"] == "NOMINEE" and st["Q"]["config"] == DIRECT
        # three shortfalls recorded separately and equal to the oracle's
        for c in PRIV:
            r = out["rows"][c]
            assert r["ordinary_shortfall"] == pytest.approx(ref["rows"][c]["ordinary_sf"], abs=1e-12)
            assert r["headroom_shortfall"] == pytest.approx(ref["rows"][c]["headroom_sf"], abs=1e-12)
    assert min(seen.values()) >= 3, seen


def test_select_every_seed_not_seed_mean(lead_select):
    recs, c = fixture_seed_average()
    out = lead_select(recs)
    assert out["rows"][c]["ordinary"] is False
    assert _cfg(out["statuses"]["P*"]) != c and select(recs, seed_mean=True)["P*"]["config"] == c


def test_select_headroom_is_0006_not_ordinary_limit(lead_select):
    recs, c = fixture_headroom_at_ordinary_limit()
    out = lead_select(recs)
    assert out["rows"][c]["ordinary"] is True and out["rows"][c]["headroom"] is False
    assert _cfg(out["statuses"]["P*"]) != c and _cfg(out["statuses"]["J*"]) != c
    assert _cfg(out["diagnostics"]["ordinary_privacy_winner_no_headroom"]) == c
    # Brier side: 0.0036 Brier excess on one task of one seed fails headroom, 0.0034 passes (0.0035 itself is not
    # representable as an excess: 0.2285 - 0.225 = 0.003500000000000003, so it fails in oracle and code alike)
    recs2 = make_bank(31, p_bad=0.0)
    c2 = cid("LOCAL", 0.025)
    for k in SEEDS:
        recs2[(k, c2)]["u"] = _u(db=(0.0034 if k else 0.0, 0.0))
        recs2[(k, c2)]["auc"].update(pair=0.55, v1=0.5, v2=0.5)
    assert _cfg(lead_select(recs2)["statuses"]["P*"]) == c2
    recs2[(1, c2)]["u"] = _u(db=(0.0036, 0.0))
    assert _cfg(lead_select(recs2)["statuses"]["P*"]) != c2


def test_select_comparators_keep_headroom_failures(lead_select):
    recs, c = fixture_comparator_fails_headroom_only()
    out = lead_select(recs)
    assert _cfg(out["statuses"]["C_rate"]) == c and _cfg(out["statuses"]["C_global"]) == c


def test_select_t_star_closed_list_excludes_rawj(lead_select):
    recs = fixture_rawj_in_t_list()
    out = lead_select(recs)
    assert _cfg(out["statuses"]["T*"]) in T_LIST and _cfg(out["statuses"]["T*"]) != RAWJ
    assert _cfg(out["statuses"]["C_global"]) == RAWJ           # RAW-J kept as a strong C_global control


def test_select_guard_is_inclusive_at_the_registered_bound(lead_select):
    """SEL-R4 fixture. The rules register 'individual inner AUC <= T* + 0.005' (inclusive; qpc's review tested that a
    code exactly at +0.005 is nominated). cbp.select decides the guard by guard_shortfall == 0.0 with
    (a - g - 0.005)/0.005: for a = g + 0.005 (float), (a - g) - 0.005 is a few ulp above zero, so EVERY exact-boundary
    case is rejected (100,000 / 100,000 random g in [0.55, 0.85])."""
    recs = make_bank(41, p_bad=0.0)
    out0 = lead_select(recs)
    T = out0["rows"][out0["statuses"]["T*"]["config"]]
    c = cid("SEQ-12", 0.04)
    for k in SEEDS:
        recs[(k, c)]["u"] = _u()
        recs[(k, c)]["auc"]["pair"] = 0.55
        for w in ("v1", "v2"):
            recs[(k, c)]["auc"][w] = T["seeds"][str(k) if str(k) in T["seeds"] else k]["auc"][w] + GUARD
    assert select(recs)["P*"]["config"] == c                     # the registered inclusive rule nominates it
    assert _cfg(lead_select(recs)["statuses"]["P*"]) == c


def test_select_decision_failure_is_invalid_not_a_shortfall(lead_select):
    recs = make_bank(42)
    c = cid("JOINT", 0.06)
    recs[(2, c)]["preserved"][1] = False
    out = lead_select(recs)
    assert out["statuses"]["P*"]["status"] == "INVALID_NOMINEE"
    assert out["statuses"]["J*"]["status"] == "INVALID_NOMINEE"
    assert out["rows"][c]["ok"] is False
    assert [f["code"] for f in out["rows"][c]["technical_failure"]] == ["DECISION_PRESERVATION_FAILURE"]
    assert "DECISION_PRESERVATION_FAILURE" in out["statuses"]["P*"]["reason"]        # SEL-C8 exact root cause
    assert out["statuses"]["P*"].get("descriptive_config") is None    # never ranked as a fallback


def test_select_nonfinite_candidate_is_invalid(lead_select):
    recs = make_bank(43)
    recs[(0, REF_E)]["auc"]["pair"] = float("nan")
    out = lead_select(recs)
    assert out["statuses"]["C_global"]["status"] == "INVALID_COMPARATOR"
    assert out["statuses"]["J*"]["status"] == "INVALID_NOMINEE"       # guard comparator missing: never dropped
    assert out["statuses"]["T*"]["status"] == "NOMINEE"               # T* list does not contain REF|E
    _strict_json(lead_select.tmp / "SELECTION.json")


def test_missing_guard_status_agrees_with_registered_truth_table(lead_select):
    """SEL-R5 fixture. C_global is INVALID (one REF|E seed non-estimable) and EVERY JOINT fails headroom (none is
    eligible, so none is 'an eligible nominee blocked only by a missing guard comparator'). LABEL_TRUTH_TABLE.json
    gives J* NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE (HEADROOM_SELECTION_FAILURE) -> claim A (comparator C_rate valid) a
    completed negative; cbp.select returns INVALID_NOMINEE / MISSING_GUARD_COMPARATOR -> claim A INCOMPLETE_OR_INVALID.
    Text and executable must agree (either amend the text to 'any missing guard comparator makes the nominee role
    INVALID_NOMINEE (MISSING_GUARD_COMPARATOR) whatever the candidates' eligibility', or change the code)."""
    tt = json.loads((PKG / "LABEL_TRUTH_TABLE.json").read_text())
    recs = make_bank(44, p_bad=0.0)
    recs[(0, REF_E)]["auc"]["pair"] = float("nan")
    for lam in LAMS:
        for k in SEEDS:
            recs[(k, cid("JOINT", lam))]["u"] = _u(dll=(0.0, 0.008))
    out = lead_select(recs)
    rows = out["rows"]
    assert not any(rows[j]["headroom"] for j in JOINTS) and out["statuses"]["C_rate"]["status"] == "NOMINEE"
    rule = next(r for r in tt["claim_status_rules_in_precedence_order"] if r.startswith("nominee technically failed"))
    text_says_invalid = "eligible nominee blocked only by a missing guard" not in rule or "whatever" in rule
    code_says_invalid = out["statuses"]["J*"]["status"] == "INVALID_NOMINEE"
    assert text_says_invalid == code_says_invalid, (rule, out["statuses"]["J*"]["status"],
                                                    out["statuses"]["J*"].get("reason"))


def test_select_ignores_assessment_and_kl_fields(lead_select):
    """Deliberate defects: an assessment-rescored weight selected as nominee; KL used instead of true-label loss.
    Planted assessment/KL fields that would reverse the choice must not move any role."""
    recs = make_bank(45)
    base = lead_select(recs)["resolved"]
    extra = {}
    for (k, c) in recs:
        good = c == cid("LOCAL", 0.1)
        extra[(k, c)] = {"assessment": {"auc": {"pair": 0.5 if good else 0.9}}, "outer": {"pair": 0.5},
                         "kl": {"income": 0.0, "occupation": 0.0}, "D": [0.0, 0.0]}
    again = lead_select(recs, extra=extra)["resolved"]
    assert again == base
    src = (ROOT / "cbp" / "select.py").read_text()
    for token in ("OSF_DEVELOPMENT_ASSESSMENT", "outer__", "assess_", "\"kl\"", "'kl'"):
        assert token not in src, token


def test_select_json_finite_and_continuous_states_null(lead_select):
    out = lead_select(make_bank(46))
    sel = _strict_json(lead_select.tmp / "SELECTION.json")
    assert sel["rows"][SRC_U]["mean_states"] is None and sel["rows"][DIRECT]["mean_states"] is not None
    _strict_json(lead_select.tmp / "selection.json")
    assert (lead_select.tmp / "HEADROOM_VS_STANDARD_SELECTION.csv").exists()
    assert out["statuses"]["T*"]["status"] == "NOMINEE"


def test_unchanged_joint_alias_names_the_simpler_family(lead_select):
    """SEL-R2 (adopted). JOINT 0.04 is bitwise the same deployed maps as SEQ-12 0.04 on every seed, so their inner
    records tie on keys 1-3 and the config ID picks 'U|JOINT|'. The label must name the simpler family."""
    recs = make_bank(51, p_bad=0.0)
    j, s12 = cid("JOINT", 0.04), cid("SEQ-12", 0.04)
    fps = {}
    for k in SEEDS:
        recs[(k, s12)] = json.loads(json.dumps(recs[(k, s12)]))
        recs[(k, s12)]["preserved"] = {1: True, 2: True}
        recs[(k, s12)]["u"] = _u(dll=(0.001, 0.003))
        recs[(k, s12)]["auc"].update(pair=0.60, v1=0.55, v2=0.55)
        recs[(k, j)] = {**json.loads(json.dumps(recs[(k, s12)])), "preserved": {1: True, 2: True}}
        for c in PRIV:
            fps[(k, c)] = f"fp-{k}-{c}"
        fps[(k, j)] = fps[(k, s12)] = f"fp-{k}-shared"
    fps[(2, cid("LOCAL", 0.04))] = "fp-2-shared"                  # a partial (one-seed) alias
    out = lead_select(recs, fingerprints=fps)
    P = out["statuses"]["P*"]
    assert P["config"] == j                                         # ordering unchanged (ID tie-break)
    assert P["winning_family"] == "SEQ-12" and P["config_family"] == "JOINT"
    assert P["aliases"]["full"] == sorted([j, s12]) and P["aliases"]["decided_by_config_id_tiebreak"] is True
    assert list(P["aliases"]["partial"]) == [cid("LOCAL", 0.04)]
    # without fingerprints (no policy unit readable) no alias is claimed and the raw family is named
    out2 = lead_select(recs)
    assert out2["statuses"]["P*"]["winning_family"] == "JOINT" and out2["statuses"]["P*"]["aliases"]["full"] == []


def test_missing_guard_fallback_rank_is_not_a_zero_field(lead_select):
    """SEL-R6 fixture. C_global INVALID and no JOINT headroom-eligible: J* is NO_ELIGIBLE (registered; a completed
    negative for claim A). Its DESCRIPTIVE fallback, however, is ranked with guard_shortfall 'or 0.0' for the missing
    guard and the gap is labelled 'missing_guards_not_needed'. Prompt sec. 9: 'do not use a zero field to hide missing
    comparator coverage. If a required comparator is missing, compute no guard-based rank and report that dependency as
    invalid'; HEADROOM_SELECTION_RULES.json fallback_ordering.invalid says the same."""
    recs = make_bank(44, p_bad=0.0)
    recs[(0, REF_E)]["auc"]["pair"] = float("nan")
    for lam in LAMS:
        for k in SEEDS:
            recs[(k, cid("JOINT", lam))]["u"] = _u(dll=(0.0, 0.008))
    J = lead_select(recs)["statuses"]["J*"]
    assert J["status"] == "NO_ELIGIBLE_NOMINEE" and J["reason"] == "HEADROOM_SELECTION_FAILURE"
    assert all(e["guard_shortfall"] is None for e in J["evaluated"])      # recorded as missing, not zero (good)
    flags = json.dumps({k: v for k, v in J.items() if k != "evaluated"})
    assert "INVALID" in flags and "C_global" in flags, J                  # the guard dependency is reported invalid


# ---------------------------------------------------------------- cbp.infer (synthetic saved predictions only)
JSTAR, LW = cid("JOINT", 0.06), cid("LOCAL", 0.04)
S21 = cid("SEQ-21", 0.1)
INFER_LABELS = [SRC_U, DIRECT, FINE, S21, JSTAR, LW, CLASS]
AUC_STRENGTH = {SRC_U: 1.6, DIRECT: 1.5, FINE: 1.5, S21: 1.2, JSTAR: 0.9, LW: 0.6, CLASS: 0.8}


def make_preds(n=320, n_dup=24, seed=0, nan_at=None):
    """Synthetic assessment preds.npz arrays per (seed, label) in the qpc/cbp format. Rows of one exact-record group
    share the group id (n_dup duplicated groups). Codes keep U's decisions (exact preservation), so accuracy
    differences to U are structural zeros."""
    rng = np.random.default_rng(1000 + seed)
    units = np.arange(n)
    units[-n_dup:] = units[:n_dup]                                   # duplicate records share a group
    perm = rng.permutation(n)
    units = units[perm] * 7 + 3                                      # non-contiguous, unsorted group ids
    sex = (rng.random(n) < 0.33).astype(np.int64)
    y1 = (rng.random(n) < 0.25).astype(np.int64)
    y2 = rng.integers(0, 6, n)
    out = {}
    for k in SEEDS:
        rk = np.random.default_rng(10 * seed + k)
        lu1 = rk.normal(0, 1, (n, 2)) + 1.5 * np.eye(2)[y1]
        lu2 = rk.normal(0, 1, (n, 6)) + 1.2 * np.eye(6)[y2]
        pu1 = np.exp(lu1) / np.exp(lu1).sum(1, keepdims=True)
        pu2 = np.exp(lu2) / np.exp(lu2).sum(1, keepdims=True)
        h1, h2 = pu1.argmax(1), pu2.argmax(1)
        for lab in INFER_LABELS:
            if lab == SRC_U:
                p1, p2 = pu1, pu2
            else:                                                     # coarser code: shrink toward the decision
                a = 0.15 if lab != CLASS else 0.6
                p1 = (1 - a) * pu1 + a * np.eye(2)[h1]
                p2 = (1 - a) * pu2 + a * np.eye(6)[h2]
            P = {}
            for v, sh in (("v1", -0.4), ("v2", -0.2), ("pair", 0.0)):
                sc = np.stack([1 / (1 + np.exp(-(AUC_STRENGTH[lab] + sh) * (sex - 0.5) * 2
                                                 - rk.normal(0, 1, n))) for _ in range(3)])
                P[f"P_auc_{v}"] = np.stack([1 - sc, sc], axis=-1)
            if nan_at and nan_at == (k, lab):
                P["P_auc_pair"][1, 0, 1] = np.nan
            out[(k, lab)] = {"assess_row_id": np.arange(n) + 50000, "assess_unit": units, "sex": sex,
                             "y_income": y1, "y_occ": y2, "const_class": np.array([0, 2]),
                             "hard1": h1, "hard2": h2, "prob1": p1, "prob2": p2, **P}
    return out


def make_el(statuses):
    resolved = {x: (s.get("config") or s.get("descriptive_config")) for x, s in statuses.items()}
    return {"statuses": statuses, "resolved": resolved, "technical_validity": {"ok": True},
            "seeds": {str(k): {"score": {lab: {"cid": lab} for lab in INFER_LABELS}} for k in SEEDS}}


GOOD_STATUSES = {"P*": {"status": "NOMINEE", "config": JSTAR, "winning_family": "JOINT"},
                 "J*": {"status": "NOMINEE", "config": JSTAR},
                 "T*": {"status": "NOMINEE", "config": DIRECT},
                 "C_rate": {"status": "NOMINEE", "config": S21},
                 "C_global": {"status": "NOMINEE", "config": S21},
                 "Q": {"status": "NOMINEE", "config": DIRECT}}


@pytest.fixture
def lead_infer(monkeypatch, tmp_path):
    RUN = _lead("run")
    INF = _lead("infer")
    captured = {}
    real = INF.UnitBootstrap

    class Spy(real):
        def __init__(self, units, B, seed, chunk):
            captured.update(units=np.asarray(units).copy(), B=B, seed=seed, chunk=chunk)
            super().__init__(units, B, seed, chunk)
    monkeypatch.setattr(INF, "UnitBootstrap", Spy)
    monkeypatch.setattr(RUN, "RUN", tmp_path)
    monkeypatch.setattr(RUN, "PKG", tmp_path)

    def go(statuses=None, preds=None, failures=None):
        preds = preds or make_preds()
        units = tmp_path / "units"
        for (k, lab), arr in preds.items():
            d = units / f"outer__s{k}__{INF.safe(lab)}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "preds.npz", **arr)
        el = tmp_path / "EVALUATION_LOCK.json"
        el.write_text(json.dumps(make_el(json.loads(json.dumps(statuses or GOOD_STATUSES)))))
        argv = ["--evaluation-lock", str(el)]
        if failures is not None:
            fp = tmp_path / "failures.json"
            fp.write_text(json.dumps(failures))
            argv += ["--failures", str(fp)]
        out = INF.main(argv, check_prior=False, units=units)
        return out, preds
    go.captured = captured
    go.tmp = tmp_path
    return go


def _slot(out, i):
    return next(e for e in out["primary"] if e["id"] == i)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score                       # third-party check of the AUC point
    return float(roc_auc_score(y, s))


def test_infer_bootstrap_groups_seed_B_and_z(lead_infer):
    out, preds = lead_infer()
    cap = lead_infer.captured
    units = preds[(0, SRC_U)]["assess_unit"]
    assert np.array_equal(cap["units"], units), "resampling unit must be the exact-record group, not the row"
    assert len(np.unique(cap["units"])) < len(cap["units"])         # duplicates really share a group here
    assert cap["B"] == 1999 and cap["seed"] == 20261008
    assert abs(out["z"] - z_family()) < 1e-12 and out["B"] == 1999 and out["seed"] == 20261008
    assert out["n_groups"] == len(np.unique(units)) and out["n_assessment"] == len(units)


def test_infer_points_equal_seed_means_with_true_label_loss(lead_infer):
    out, preds = lead_infer()
    sex, y = preds[(0, SRC_U)]["sex"], {0: preds[(0, SRC_U)]["y_income"], 1: preds[(0, SRC_U)]["y_occ"]}
    rec = lambda k, lab, v: np.mean([_auc(sex, preds[(k, lab)][f"P_auc_{v}"][s][:, 1]) for s in range(3)])  # noqa
    p01 = np.mean([rec(k, S21, "pair") - rec(k, JSTAR, "pair") for k in SEEDS])
    assert _slot(out, "P01")["point"] == pytest.approx(p01, abs=1e-12)
    p24 = np.mean([rec(k, JSTAR, "v1") - rec(k, DIRECT, "v1") for k in SEEDS])
    assert _slot(out, "P24")["point"] == pytest.approx(p24, abs=1e-12)

    def ll(k, lab, j):
        p = preds[(k, lab)][f"prob{j + 1}"]
        return float(np.mean(-np.log(np.clip(p[np.arange(len(y[j])), y[j]], 1e-12, 1))))

    def kl(k, lab, j):                                               # DEFECT: teacher KL instead of true-label loss
        p, q = preds[(k, SRC_U)][f"prob{j + 1}"], preds[(k, lab)][f"prob{j + 1}"]
        return float(np.mean((p * (np.log(p) - np.log(q))).sum(1)))
    for sid, j in (("P06", 0), ("P07", 1), ("P34", 0), ("P35", 1)):
        nom = JSTAR if sid in ("P06", "P07") else DIRECT
        want = np.mean([ll(k, nom, j) - ll(k, SRC_U, j) for k in SEEDS])
        assert _slot(out, sid)["point"] == pytest.approx(want, abs=1e-12)
        assert abs(np.mean([kl(k, nom, j) for k in SEEDS]) - want) > 1e-4      # fixture discriminates the defect
    br = lambda k, lab, j: float(np.mean(((preds[(k, lab)][f"prob{j + 1}"] - np.eye(preds[(k, lab)][f"prob{j + 1}"].shape[1])[y[j]]) ** 2).sum(1)))  # noqa
    assert _slot(out, "P09")["point"] == pytest.approx(np.mean([br(k, JSTAR, 1) - br(k, SRC_U, 1) for k in SEEDS]),
                                                       abs=1e-12)


def test_infer_group_bootstrap_se_reproduced_independently(lead_infer):
    """SE of P06 (income log-loss excess of J*) from an independent replay: B = 1999 sequential multinomial draws over
    the sorted unique exact-record groups with numpy default_rng(20261008), one draw stream for every arm and seed.
    The row-level bootstrap (deliberate defect: wrong group) gives a different SE."""
    out, preds = lead_infer()
    units = preds[(0, SRC_U)]["assess_unit"]
    y = preds[(0, SRC_U)]["y_income"]
    _, inv = np.unique(units, return_inverse=True)
    G = inv.max() + 1

    def per_row(k, lab):
        p = preds[(k, lab)]["prob1"]
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    d = [per_row(k, JSTAR) for k in SEEDS]
    u = [per_row(k, SRC_U) for k in SEEDS]

    def boot(groups, ncell):
        rng = np.random.default_rng(20261008)
        reps = []
        for _ in range(1999):
            w = rng.multinomial(ncell, np.full(ncell, 1.0 / ncell))[groups].astype(float)
            reps.append(np.mean([(d[k] @ w) / w.sum() - (u[k] @ w) / w.sum() for k in SEEDS]))
        return float(np.std(reps, ddof=1))
    se = _slot(out, "P06")["se"]
    assert se == pytest.approx(boot(inv, G), rel=1e-9)
    assert abs(boot(np.arange(len(units)), len(units)) - se) > 1e-9 * se


def test_infer_zero_variance_identities_pass_and_are_kept(lead_infer):
    out, _ = lead_infer()
    for i in ("P04", "P05", "P15", "P16", "P26", "P27"):
        s = _slot(out, i)
        assert s["point"] == 0.0 and s["se"] == 0.0 and s["outcome"] == "PASS" and s["decision"] == "PASS", s
    assert len(out["primary"]) == 37
    _strict_json(lead_infer.tmp / "inference.json")


def test_infer_nonfinite_probability_makes_loss_slots_invalid(lead_infer):
    preds = make_preds()
    preds[(2, JSTAR)]["prob2"] = preds[(2, JSTAR)]["prob2"].copy()
    preds[(2, JSTAR)]["prob2"][5] = np.nan
    out, _ = lead_infer(preds=preds)
    for i in ("P07", "P09", "P18", "P20", "P29", "P31"):
        assert _slot(out, i)["outcome"] == "INVALID", _slot(out, i)
    assert all(out["claim_status"][c]["status"] == "INCOMPLETE_OR_INVALID" for c in "ABC")


def test_infer_nonfinite_primary_is_invalid_never_dropped(lead_infer):
    """SEL-R7 fixture. One NaN attacker score (seed 1, attacker seed 1, row 0) of J*'s pair reader. The AUC helper
    (stored_model_eval.pilot_infer._auc_prep) ranks scores with np.unique, which sorts NaN LAST: the NaN row silently
    becomes the most-SEX=1 row, the AUC stays finite and P01/P12/P23 get a numeric outcome (PASS here). Prompt sec. 6/11:
    nonfinite primary quantities are INVALID, never sortable successes. Neither cbp.assess nor cbp.infer checks the
    saved attacker scores for finiteness."""
    out, _ = lead_infer(preds=make_preds(nan_at=(1, JSTAR)))
    for i in ("P01", "P12", "P23"):
        assert _slot(out, i)["outcome"] == "INVALID", _slot(out, i)
    for c in "ABC":
        assert out["claim_status"][c]["status"] == "INCOMPLETE_OR_INVALID"
        assert out["claim_status"][c]["root_cause"] == "NONFINITE_PRIMARY_QUANTITY"
    assert "PRIVACY_COMPRESSION" not in out["label"] and len(out["primary"]) == 37
    _strict_json(lead_infer.tmp / "inference.json")


def test_infer_absent_comparator_never_passes(lead_infer):
    st = json.loads(json.dumps(GOOD_STATUSES))
    st["T*"] = {"status": "INVALID_COMPARATOR", "config": None, "reason": "FIT_OR_ADMISSION_FAILURE"}
    out, _ = lead_infer(statuses=st)
    for i in ("P23", "P24", "P25"):
        assert _slot(out, i)["outcome"] == "INVALID"
    assert out["claim_status"]["C"]["status"] == "INCOMPLETE_OR_INVALID"
    assert out["claim_status"]["C"]["root_cause"] == "INVALID_OR_MISSING_COMPARATOR"
    st["T*"] = {"status": "NO_ELIGIBLE_COMPARATOR", "config": None, "descriptive_config": FINE}
    out, _ = lead_infer(statuses=st)
    assert out["claim_status"]["C"]["status"] == "INCOMPLETE_OR_INVALID"
    assert out["claim_status"]["C"]["root_cause"] == "NO_ELIGIBLE_COMPARATOR"
    # the comparator slots are DESCRIPTIVE; P26-P33 measure P* against U only and keep their numeric decision
    assert all(_slot(out, f"P{i}")["decision"] == "DESCRIPTIVE_ONLY" for i in (23, 24, 25))


def test_infer_fallback_nominee_never_passes(lead_infer):
    st = json.loads(json.dumps(GOOD_STATUSES))
    st["P*"] = {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": LW,
                "reason": "HEADROOM_SELECTION_FAILURE"}
    out, _ = lead_infer(statuses=st)
    c = out["claim_status"]["C"]
    assert c["status"] == "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE" and c["root_cause"] == "HEADROOM_SELECTION_FAILURE"
    assert all(_slot(out, f"P{i}")["decision"] == "DESCRIPTIVE_ONLY" for i in range(23, 34))
    assert "PRIVACY_COMPRESSION" not in out["label"]


def test_infer_roles_come_only_from_the_lock(lead_infer):
    """Deliberate defect: an assessment-rescored weight selected as nominee. LOCAL 0.04 has the lowest assessment
    pair AUC of every scored arm; the locked P* (JOINT 0.06) must stay the scored nominee."""
    out, _ = lead_infer()
    assert out["resolved"]["P*"] == JSTAR and out["claim_status"]["C"]["nominee"] == JSTAR
    lv = out["levels"]
    lw = lv[f"Rmean#{LW}#primary#pair"]["point"]
    assert all(lw <= lv[f"Rmean#{lab}#primary#pair"]["point"] for lab in INFER_LABELS)


def test_infer_clauses_passing_counts_only_scored_decisions(lead_infer):
    """SEL-C10 fixture (RECOMMENDED): with a DESCRIPTIVE fallback nominee, clauses_passing must not count fallback
    numbers as passing clauses (qpc named the numeric count 'clauses_passing_numeric')."""
    st = json.loads(json.dumps(GOOD_STATUSES))
    st["P*"] = {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": LW,
                "reason": "HEADROOM_SELECTION_FAILURE"}
    out, _ = lead_infer(statuses=st)
    n_desc_pass = sum(_slot(out, f"P{i}")["outcome"] == "PASS" for i in range(23, 34))
    assert n_desc_pass > 0                                           # the fixture has passing fallback numbers
    assert out["clauses_passing_scored"]["C"] == 0
    assert out["clauses_passing_numeric_including_descriptive"]["C"] == n_desc_pass
    assert "clauses_passing" not in out                              # no ambiguous count left


# ---------------------------------------------------------------- registered JSON / protocol text vs executable
def test_primary_family_json_equals_executable():
    F = _lead("family")
    p = PKG / "PRIMARY_FAMILY.json"
    if not p.exists():
        pytest.skip("PRIMARY_FAMILY.json not written yet")
    J = _strict_json(p)
    assert J["size"] == 37 and J["B"] == 1999 and J["bootstrap_seed"] == 20261008
    assert abs(J["z"] - z_family()) < 1e-12 and "0.05/74" in J["z_definition"]
    assert [{k: v for k, v in e.items()} for e in J["slots"]] == json.loads(json.dumps(F.PRIMARY))
    assert {c: (v["nominee"], v.get("comparator")) for c, v in J["claims"].items() if c != "Q"} == \
        {c: tuple(v) for c, v in F.CLAIMS.items()}
    assert "exact-record group" in J["bootstrap"] and "identical draws" in J["bootstrap"]
    assert "equal-weight mean over model seeds" in J["aggregation"]


def test_protocol_selection_text_carries_the_registered_numbers():
    p = PKG / "PROTOCOL.md"
    if not p.exists():
        pytest.skip("PROTOCOL.md not written yet")
    t = p.read_text()
    for s in ("0.006", "0.0035", "T\\* + 0.005", "C_rate + 0.005", "C_global +", "20261008", "B = 1999",
              "0.05/74", "3.2048452050105634", "RAW-J is excluded from T\\*", "never dropped for failing headroom",
              "exact-record groups", "round(·, 12)", "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE",
              "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION", "EXPERIMENTAL_NO_ADVANTAGE"):
        assert s in t, s


def test_registered_truth_table_text_agrees_with_executable_reasons():
    F = _lead("family")
    tt = _strict_json(PKG / "LABEL_TRUTH_TABLE.json")
    for r in F.SELECTION_REASONS:
        if r:
            assert r in tt["selection_reason_classes"], r
    assert set(F.CLAUSE_OUTCOMES) == set(tt["clause_outcomes"])


@pytest.mark.xfail(strict=False, reason="SEL-C11 RECOMMENDED: HEADROOM_VS_STANDARD_SELECTION.csv lacks rows/columns")
def test_headroom_vs_standard_csv_carries_every_prespecified_diagnostic(lead_select):
    """Prompt sec. 9 diagnostics and the named deliverable HEADROOM_VS_STANDARD_SELECTION.csv: the unguarded reading,
    the source lambda 0.1 JOINT / SEQ-12 / SEQ-21 controls, whether headroom changes the winner and the pair AUC given up
    (mean and per seed), and a DESCRIPTIVE / fallback-rank flag on fallback rows."""
    import csv as _csv
    lead_select(make_bank(61))
    rows = list(_csv.DictReader(open(lead_select.tmp / "HEADROOM_VS_STANDARD_SELECTION.csv")))
    text = json.dumps(rows)
    for c in ("U|JOINT|i8o64|l0.1", "U|SEQ-12|i8o64|l0.1", "U|SEQ-21|i8o64|l0.1"):
        assert c in text, c
    assert "unguarded" in text
    assert any("given_up" in k or "give_up" in k for k in rows[0]) or "give_up" in text
    assert any("descriptive" in k.lower() or "fallback_rank" in k for k in rows[0])
