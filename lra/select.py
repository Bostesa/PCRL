"""[lra port of lcr/select.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md and
REVIEW_FINDINGS_DISPOSITION.json (findings F01-F14 of source_reviews/SELECTION_STACK_REVIEW.json)]
Registered inner-only nomination (SELECTION_RULES.json; PROTOCOL.md section 10; frozen in SCIENCE_LOCK).

Adapted from cbp/select.py at 7f3ec67 WITHOUT cbp's failed 0.006/0.0035 every-seed headroom buffer (prompt section 10:
the explicit 0.005/0.003 budgets act during FITTING; nomination uses the ORIGINAL inner eligibility). No assessment value
is read: release identities are hashed on the permitted OSF_DEFENSE_FIT + AUDIT_FIT + INNER_SELECTION rows only (F07/F11).
Inputs: aud__<unit> records (lra.run.inner_name; lra.audit; attackers fitted on AUDIT_FIT, selected/scored on
INNER_SELECTION; utility on INNER_SELECTION via qpc.utility) for every candidate of lra.run.scored_ids().

Per seed (anchor = SRC|U utility of that seed), with gr = qpc.utility.gate_record(code, U, preserved):
  ordinary eligible   gr["eligible"] on EVERY task (inclusive): acc >= U-0.01, LL <= U+0.01, Brier <= U+0.005,
                      gain >= 0.8 U gain, gain >= 0.03, decision preservation -> required on EVERY seed
  ordinary shortfall  max(0, (a_U-0.01-a)/0.01, (dLL-0.01)/0.01, (dB-0.005)/0.005, (0.8g_U-g)/max(0.8g_U,0.03),
                      (0.03-g)/0.03) over tasks and seeds
  guard               local inner AUC <= guard comparator local AUC + 0.005 (inclusive) for each recipient and seed
  guard shortfall     exactly 0.0 iff the guard holds, else max over guards/seeds/recipients of (a - g - 0.005)/0.005
Roles (prompt section 10):
  T*      strongest ordinarily eligible privacy-UNTRAINED release from the closed list T_STAR_CLOSED (RAW-J excluded)
  P*      strongest eligible release of the registered privacy-trained CODE pool (D0 privacy, D1 fixed-map privacy,
          weighted, constrained; private_code), guard vs T*
  C*      strongest eligible incumbent/baseline private release (D0 / D1 fixed-map privacy maps, weighted controls;
          NO constrained arm); comparator: ordinary eligibility only
  N*      strongest eligible NEW constrained release (any of the 5 arms), guards vs BOTH T* and C*
  C_pair* strongest eligible release other than K-JOINT-PAIR (every scored candidate except it and except infeasible
          constrained fits); comparator
  J*      strongest eligible K-JOINT-PAIR release, guards vs BOTH T* and C_pair*
  Q       fixed D0 U|DIRECT-TASK|i8o64 (ordinary eligibility prerequisite)
  Constrained arms are eligible only when their fit record is FEASIBLE on every seed (fit_feasible). An infeasible
  record gives CONSTRAINED_FIT_INFEASIBLE ("infeasible under this registered decoder"), a selection outcome. A
  missing, not hash-complete, unreadable or schema-incomplete fit record is TECHNICAL
  (FIT_RECORD_TECHNICAL_FAILURE), never CONSTRAINED_FIT_INFEASIBLE (F01).
Ordering: mean INNER pair AUC, mean summed true-label log loss (both float64 seed means (s0+s1+s2)/3 in seed order, then
round(x, 12); equal rounded values tie), mean actual token states (continuous = +inf for ordering only; null in JSON),
configuration ID. A code release must carry a finite positive integer token_states and a continuous one null; anything
else is NON_ESTIMABLE_INNER_METRIC (technical, F06).
Fallbacks (DESCRIPTIVE_ONLY, never a pass; fallback_class distinguishes the four registered kinds):
  UTILITY            ORDINARY_UTILITY_FAILURE / CONSTRAINED_FIT_INFEASIBLE: (fit-feasible first, ordinary shortfall,
                     guard shortfall, ordering keys)
  LOCAL_GUARD        LOCAL_GUARD_FAILURE: the same ranking
  MISSING_COMPARATOR MISSING_GUARD_COMPARATOR: a required guard role has no valid nominee. If some candidate is
                     eligible the role is INVALID with the strongest eligible candidate (ordering keys) as its fallback
                     (F04); otherwise NO_ELIGIBLE ranked on (fit-feasible, ordinary shortfall, ordering keys) with
                     fallback_rank_status INVALID_MISSING_GUARD_COMPARATOR
  TECHNICAL          any FIT_OR_ADMISSION_FAILURE / FIT_RECORD_TECHNICAL_FAILURE / DECISION_PRESERVATION_FAILURE /
                     NON_ESTIMABLE_INNER_METRIC in the pool: INVALID, no fallback (the evaluation lock refuses
                     unresolved technical failures, F13)
Aliases (F02/F03/F07/F09/F11): for EVERY role the exact alias set = configurations whose deployed release (tokens,
decoded probabilities, decisions and declared alphabets on the permitted rows) is byte-identical on all three seeds; the
representative is one real member of the role's OWN candidate pool chosen by (construction rank, family rank, config
ID); identical_to_untrained lists privacy-untrained exact aliases (disclosed beside the role and the label; no
privacy-training credit without a changed release). Same tokens with a different decoder are NOT aliases. Canonical
token renamings are recorded separately as canonical_equivalent (informational, never an alias).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math

import numpy as np

from lra import family as FAM
from lra import run as R
from qpc import utility as UT

BUFFER = 0.005
SEEDS = R.SEEDS
VIEWS = ("v1", "v2", "pair")
TASKS = ("income", "occupation")
ROLES = ("P*", "N*", "J*", "T*", "C*", "C_pair*", "Q")
PRIVACY_ROLES = ("P*", "N*", "J*")
T_STAR_CLOSED = ("U|C-TASK|i8o64|D1", "U|DIRECT-TASK|i8o64", "U|DIRECT-TASK|i8o64|D1", "U|FINE-TASK|i8o64",
                 "U|FINE-TASK|i8o64|D1", "SRC|U", "U|CLASS|i1o1", "U|CLASS|i1o1|D1", "REF|F0")   # CLASS|D1: F R-1
Q_CONFIG = "U|DIRECT-TASK|i8o64"
JOINT_PAIR = "U|K-JOINT-PAIR|i8o64|D1"
PRIVACY_TRAINED_REFERENCES = ("SRC|RAW-J_b0.3", "REF|F", "REF|E")     # RAW-J (prompt sec. 10), FARE, LEACE (F05)
PERMITTED_ROLES = ("DEFENSE_FIT", "AUDIT_FIT", "INNER_SELECTION")       # release-identity rows (prompt sec. 10)
RELEASE_IDENTITY_KEYS = ("tok1", "q1", "hard1", "tok2", "q2", "hard2")
TECHNICAL_REASONS = ("FIT_OR_ADMISSION_FAILURE", "FIT_RECORD_TECHNICAL_FAILURE", "DECISION_PRESERVATION_FAILURE",
                     "NON_ESTIMABLE_INNER_METRIC")
FALLBACK_CLASS = {"ORDINARY_UTILITY_FAILURE": "UTILITY", "CONSTRAINED_FIT_INFEASIBLE": "UTILITY",
                  "LOCAL_GUARD_FAILURE": "LOCAL_GUARD", "MISSING_GUARD_COMPARATOR": "MISSING_COMPARATOR",
                  **{r: "TECHNICAL" for r in TECHNICAL_REASONS}}
# representative rank (F09): construction first (existing < calibrated < task-only < weighted < constrained), then
# family (LOCAL < SEQ-12 = SEQ-21 < JOINT = JOINT-SINGLE < JOINT-PAIR; task families after), then config ID
CONSTRUCTION_RANK = FAM.CONSTRUCTION_SIMPLICITY
CONSTRUCTION_NAME = FAM.CONSTRUCTION_NAME
assert set(TECHNICAL_REASONS) == set(FAM.TECHNICAL_SELECTION_REASONS)


def arm(cid):
    p = R.parse_id(cid)
    return p["kind"] if p["kind"] != "policy" else p["arm"]


def private_code(cid):
    """P* pool membership: the registered privacy-trained CODE arms (24 D0, 24 D1 fixed-map, 24 weighted, 5 constrained;
    references and diagnostics are never candidates)."""
    p = R.parse_id(cid)
    return p["kind"] == "policy" and bool(p["privacy_trained"]) and not p.get("diagnostic_only")


def privacy_trained(cid):
    """Truthful label of every scored id (F05): privacy-trained codes AND the privacy-trained references."""
    return private_code(cid) or cid in PRIVACY_TRAINED_REFERENCES


def training(cid):
    """Display only (never a role input): privacy-trained / -untrained code or reference."""
    if R.parse_id(cid)["kind"] == "policy":
        return "privacy-trained code" if private_code(cid) else "privacy-untrained code"
    return "privacy-trained reference" if cid in PRIVACY_TRAINED_REFERENCES else "privacy-untrained reference"


def private_all(ids):
    return [c for c in ids if private_code(c)]


def incumbent_private(ids):
    return [c for c in ids if private_code(c) and arm(c) in ("d0", "d1_fixed", "weighted")]


def constrained(ids):
    return [c for c in ids if arm(c) == "constrained"]


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def ordinary_shortfall(u, uU):
    s = 0.0
    for t in TASKS:
        a, aU = u[t]["acc"], uU[t]["acc"]
        c = u[t]["const_acc"]
        g, gU = a - c, aU - c
        dll, dbr = u[t]["logloss"] - uU[t]["logloss"], u[t]["brier"] - uU[t]["brier"]
        s = max(s, (aU - 0.01 - a) / 0.01, (dll - 0.01) / 0.01, (dbr - 0.005) / 0.005,
                (0.8 * gU - g) / max(0.8 * gU, 0.03), (0.03 - g) / 0.03)
    return max(0.0, s)


# ------------------------------------------------------------------ release identity (permitted rows only)
def permitted_rows(D):
    """Positions (D order) and row ids of OSF_DEFENSE_FIT + AUDIT_FIT + INNER_SELECTION, in that order. Never an
    assessment or HEAD_VALIDATION row (prompt sec. 10: release hashes and aliases on permitted fitting/inner rows)."""
    if D is None:
        raise ValueError("release identities need the (sealed) data handle for the permitted rows")
    pos = np.concatenate([np.asarray(D["idx"][r], dtype=np.int64) for r in PERMITTED_ROLES])
    forbidden = np.concatenate([np.asarray(D["idx"][r], dtype=np.int64)
                                for r in ("OSF_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION") if r in D["idx"]])
    if np.intersect1d(pos, forbidden).size or np.unique(pos).size != pos.size:
        raise ValueError("permitted release-identity rows overlap the assessment / head-validation rows or repeat")
    return {"pos": pos, "row_id": np.asarray(D["row_id"])[pos],
            "rows_sha256": hashlib.sha256(np.ascontiguousarray(np.asarray(D["row_id"])[pos]).tobytes()).hexdigest()}


def _canonical(tok):
    """Token ids renamed by first occurrence on the permitted rows (renaming-invariant identity)."""
    _, first, inv = np.unique(tok, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    return order[inv].astype(np.int64)


def release_identity(k, cid, rows):
    """(byte sha256, canonical sha256) of the DEPLOYED code release on the permitted rows; (None, None) for continuous
    releases. Byte identity covers tok_i, q_i, hard_i (permitted rows) and the declared alphabets alpha_i; the canonical
    hash renames tokens by first occurrence (informational equivalence only, never an alias)."""
    if R.parse_id(cid)["kind"] != "policy":
        return None, None
    z = np.load(R.U(R.unit_for(k, cid)) / "release.npz", allow_pickle=False)
    pos = rows["pos"]
    if not np.array_equal(np.asarray(z["row_id"])[pos], rows["row_id"]):
        raise ValueError(f"{cid} s{k}: release rows are not aligned with the permitted rows")
    hb, hc = hashlib.sha256(), hashlib.sha256()
    for h in (hb, hc):
        h.update(rows["rows_sha256"].encode())
    for x in RELEASE_IDENTITY_KEYS:
        a = np.ascontiguousarray(np.asarray(z[x])[pos])
        hb.update(x.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        c = _canonical(a) if x.startswith("tok") else a
        c = np.ascontiguousarray(c)
        hc.update(x.encode() + str(c.dtype).encode() + str(c.shape).encode() + c.tobytes())
    for x in ("alpha1", "alpha2"):
        a = np.ascontiguousarray(np.asarray(z[x]).astype(np.int64).ravel())
        hb.update(x.encode() + a.tobytes())
        hc.update(x.encode() + a.tobytes())
    return hb.hexdigest(), hc.hexdigest()


# ------------------------------------------------------------------ constrained fit records (F01)
def fit_feasible(cid):
    """(value, detail). Constrained arms: True iff the fit record is FEASIBLE with deployed.feasible True on EVERY seed;
    False if some seed is a complete, consistent INFEASIBLE record (a selection outcome: CONSTRAINED_FIT_INFEASIBLE);
    None (TECHNICAL -> FIT_RECORD_TECHNICAL_FAILURE) if a fit unit is missing, not hash-complete or unreadable, or its
    record lacks status in {FEASIBLE, INFEASIBLE} / a boolean deployed.feasible, names another configuration or seed, or
    has a status that disagrees with deployed.feasible. Other arms have no fitting-budget requirement: True."""
    if arm(cid) != "constrained":
        return True, None
    out = True
    for k in SEEDS:
        u = R.unit_for(k, cid)
        try:
            if not R.done(u):
                return None, f"{u}: fit unit missing or not hash-complete"
            f = R.rec(u)
            st, dep = f["status"], f["deployed"]["feasible"]
        except Exception as e:                                               # noqa: BLE001
            return None, f"{u}: fit record unreadable or schema-incomplete ({type(e).__name__}: {e})"
        if st not in ("FEASIBLE", "INFEASIBLE") or not isinstance(dep, bool):
            return None, f"{u}: status {st!r} / deployed.feasible {dep!r} outside the registered schema"
        if f.get("config", cid) != cid or int(f.get("seed", k)) != k:
            return None, f"{u}: record names {f.get('config')!r} seed {f.get('seed')!r}"
        if (st == "FEASIBLE") != dep:                  # the registered mapper never writes a disagreeing record
            return None, f"{u}: status {st} disagrees with deployed.feasible {dep} (inconsistent record)"
        out = out and st == "FEASIBLE" and dep is True
    return out, None


def _token_states_ok(cid, ts):
    if R.parse_id(cid)["kind"] == "policy":
        return _fin(ts) and float(ts) > 0 and float(ts) == int(ts)
    return ts is None


def candidate_rows(ids, rows_perm=None):
    anchors = {}
    for k in SEEDS:
        try:
            anchors[k] = R.rec(R.inner_name(k, "SRC|U"))["utility"]
        except Exception:                                                    # noqa: BLE001
            anchors[k] = None
    rows = {}
    for cid in ids:
        seeds, fail = {}, []
        for k in SEEDS:
            n = R.inner_name(k, cid)
            ru = R.unit_for(k, cid)
            if not R.done(ru):
                fail.append({"unit": ru, "code": "FIT_OR_ADMISSION_FAILURE", "detail": "release unit missing or not "
                                                                                        "hash-complete"})
                continue
            if not R.done(n):
                fail.append({"unit": n, "code": "FIT_OR_ADMISSION_FAILURE", "detail": "missing or not hash-complete"})
                continue
            try:
                r = R.rec(n)
                a = {w: float(r["recovery"]["auc"][w]) for w in VIEWS}
                u = r["utility"]
                pres = {int(i): bool(v) for i, v in r["preserved"].items()}
                assert set(pres) == {1, 2}
                assert all(_fin(a[w]) for w in VIEWS)
                assert all(_fin(u[t][x]) for t in TASKS for x in ("acc", "logloss", "brier", "const_acc"))
                assert anchors[k] is not None, "SRC|U anchor missing"
                gr = UT.gate_record(u, anchors[k], pres)
                assert all(_fin(gr[t][x]) for t in TASKS for x in ("ll_excess", "brier_excess"))
                ts = r.get("token_states")
                assert _token_states_ok(cid, ts), (f"token_states {ts!r}: must be a finite positive integer for a code "
                                                   "release and null for a continuous one")
                rh, rc = release_identity(k, cid, rows_perm)
            except Exception as e:                                           # noqa: BLE001
                fail.append({"unit": n, "code": "NON_ESTIMABLE_INNER_METRIC", "detail": f"{type(e).__name__}: {e}"})
                continue
            if not all(pres.values()):
                fail.append({"unit": n, "code": "DECISION_PRESERVATION_FAILURE",
                             "detail": "decision preservation failed (invalid, not a shortfall)"})
                continue
            seeds[k] = {"unit": ru, "auc": a,
                        "utility": {t: {x: u[t][x] for x in ("acc", "logloss", "brier", "const_acc")} for t in TASKS},
                        "ll_excess": {t: gr[t]["ll_excess"] for t in TASKS},
                        "brier_excess": {t: gr[t]["brier_excess"] for t in TASKS},
                        "ordinary": bool(gr["eligible"]), "ordinary_shortfall": ordinary_shortfall(u, anchors[k]),
                        "token_states": None if ts is None else int(ts),
                        "composed_winner": (r.get("composed") or {}).get("winner"), "release_hash": rh,
                        "canonical_hash": rc}
        fit_ok, fit_detail = (None, None)
        if not fail and len(seeds) == len(SEEDS):
            fit_ok, fit_detail = fit_feasible(cid)
            if fit_ok is None:                                               # F01: technical, never infeasible
                fail.append({"unit": ",".join(R.unit_for(k, cid) for k in SEEDS), "code": "FIT_RECORD_TECHNICAL_FAILURE",
                             "detail": "constrained fit record technical failure (never CONSTRAINED_FIT_INFEASIBLE): "
                                       + str(fit_detail)})
        ok = not fail and len(seeds) == len(SEEDS)
        p = R.parse_id(cid)
        row = {"fit_feasible": fit_ok if ok else None, "config": cid, "arm": arm(cid),
               "family": (p.get("base_family") or arm(cid)), "kind": p["kind"],
               "privacy_trained": privacy_trained(cid), "private_code": private_code(cid), "training": training(cid),
               "technical_failure": fail, "ok": ok, "seeds": seeds}
        if ok:
            mean = lambda f: (f(seeds[0]) + f(seeds[1]) + f(seeds[2])) / 3      # noqa: E731  seed order (rules)
            st = [s["token_states"] for s in seeds.values()]
            row.update({"mean_pair": mean(lambda s: s["auc"]["pair"]), "mean_v1": mean(lambda s: s["auc"]["v1"]),
                        "mean_v2": mean(lambda s: s["auc"]["v2"]),
                        "mean_sum_logloss": mean(lambda s: s["utility"]["income"]["logloss"] +
                                                 s["utility"]["occupation"]["logloss"]),
                        "mean_states": None if p["kind"] != "policy" else sum(st) / len(st),
                        "ordinary_inner": all(s["ordinary"] for s in seeds.values()),
                        "ordinary": all(s["ordinary"] for s in seeds.values()) and fit_ok is True,
                        "ordinary_shortfall": max(s["ordinary_shortfall"] for s in seeds.values())})
        rows[cid] = row
    return rows


def key(r):
    st = r["mean_states"] if r["mean_states"] is not None else math.inf      # continuous: +inf for ordering only
    return (round(r["mean_pair"], 12), round(r["mean_sum_logloss"], 12), st, r["config"])


def guard_ok(r, guards):
    return all(r["seeds"][k]["auc"][w] <= g["seeds"][k]["auc"][w] + BUFFER
               for g in guards.values() for k in SEEDS for w in ("v1", "v2"))


def guard_shortfall(r, guards):
    if guard_ok(r, guards):
        return 0.0
    xs = [(r["seeds"][k]["auc"][w] - g["seeds"][k]["auc"][w] - BUFFER) / BUFFER
          for g in guards.values() for k in SEEDS for w in ("v1", "v2")]
    return max(0.0, max(xs))


def _tech_reason(rows):
    codes = sorted({f["code"] for r in rows for f in r["technical_failure"]})
    return "+".join(codes) if codes else "FIT_OR_ADMISSION_FAILURE"


def pick(rows, guards=None, nominee=True):
    """rows: candidate rows of one role; guards: {name: row or None (guard role has no valid nominee)}."""
    none = "NO_ELIGIBLE_NOMINEE" if nominee else "NO_ELIGIBLE_COMPARATOR"
    bad = "INVALID_NOMINEE" if nominee else "INVALID_COMPARATOR"
    if not rows:
        return {"status": bad, "config": None, "reason": "FIT_OR_ADMISSION_FAILURE", "detail": "empty candidate set",
                "fallback_class": "TECHNICAL", "descriptive_config": None}
    tf = [r for r in rows if not r["ok"]]
    if tf:
        return {"status": bad, "config": None, "reason": _tech_reason(tf), "failed": [r["config"] for r in tf],
                "failures": [f for r in tf for f in r["technical_failure"]], "fallback_class": "TECHNICAL",
                "descriptive_config": None,
                "note": "technical failure: no fallback; the evaluation lock refuses until it is resolved"}
    guards = guards or {}
    missing = sorted(g for g, row in guards.items() if row is None)
    live = {g: row for g, row in guards.items() if row is not None}
    ev = []
    for r in rows:
        ok = (not missing) and guard_ok(r, live)
        ev.append({"config": r["config"], "arm": r["arm"], "family": r["family"], "ordinary": r["ordinary"],
                   "fit_feasible": r.get("fit_feasible"), "ordinary_inner": r.get("ordinary_inner"),
                   "ordinary_shortfall": r["ordinary_shortfall"],
                   "guard_shortfall": None if missing else guard_shortfall(r, live), "eligible": r["ordinary"],
                   "guard_ok": ok, "nominable": r["ordinary"] and ok})
    byc = {r["config"]: r for r in rows}
    if missing and any(e["eligible"] for e in ev):                            # F04: invalid, fallback still fixed
        fb = min((byc[e["config"]] for e in ev if e["eligible"]), key=key)
        return {"status": bad, "config": None, "reason": "MISSING_GUARD_COMPARATOR", "missing_guards": missing,
                "blocked": [e["config"] for e in ev if e["eligible"]], "evaluated": ev, "descriptive_only": True,
                "descriptive_config": fb["config"], "fallback_rank_status": "INVALID_MISSING_GUARD_COMPARATOR",
                "fallback_rank_keys": ["eligible (guard not evaluable)", "ordering"],
                "fallback_class": "MISSING_COMPARATOR"}
    el = [byc[e["config"]] for e in ev if e["nominable"]]
    if el:
        return {"status": "NOMINEE", "config": min(el, key=key)["config"], "evaluated": ev}
    feas = [byc[e["config"]].get("fit_feasible") is not False for e in ev]
    reason = ("CONSTRAINED_FIT_INFEASIBLE" if not any(feas) else
              "ORDINARY_UTILITY_FAILURE" if not any(e["ordinary"] for e in ev) else "LOCAL_GUARD_FAILURE")
    out = {"status": none, "config": None, "descriptive_only": True, "reason": reason, "evaluated": ev}
    if missing:
        fb = min(ev, key=lambda e: (byc[e["config"]].get("fit_feasible") is False, round(e["ordinary_shortfall"], 12)) +
                 key(byc[e["config"]]))
        out.update({"missing_guards": missing, "fallback_rank_status": "INVALID_MISSING_GUARD_COMPARATOR",
                    "fallback_rank_keys": ["fit_feasible", "ordinary_shortfall", "ordering"],
                    "fallback_class": "MISSING_COMPARATOR"})
    else:
        fb = min(ev, key=lambda e: (byc[e["config"]].get("fit_feasible") is False, round(e["ordinary_shortfall"], 12),
                                    round(e["guard_shortfall"], 12)) + key(byc[e["config"]]))
        out.update({"fallback_rank_status": "VALID", "fallback_class": FALLBACK_CLASS[reason],
                    "fallback_rank_keys": ["fit_feasible", "ordinary_shortfall", "guard_shortfall", "ordering"]})
    out["descriptive_config"] = fb["config"]
    return out


# ------------------------------------------------------------------ aliases and the representative (F02/F03/F07/F09/F11)
def _rep_key(rows, c):
    r = rows[c]
    return (CONSTRUCTION_RANK.get(r["arm"], 9), FAM.FAMILY_SIMPLICITY.get(r["family"], 9), c)


def alias_set(rows, cid, pool=None):
    """Exact deployed-release aliases of cid (byte-identical on the permitted rows of all three seeds), partial
    aliases (some seeds), canonical-renaming equivalents (informational), ONE real representative drawn from the
    role's OWN candidate pool (P*: the privacy-trained codes, so untrained aliases are never named) and the
    privacy-untrained exact aliases. Continuous releases alias by configuration ID only."""
    empty = {"full": [], "partial": {}, "canonical_equivalent": [], "representative": None,
             "representative_family": None, "representative_construction": None, "identical_to_untrained": [],
             "decided_by_config_id_tiebreak": False}
    if not cid or cid not in rows or not rows[cid]["ok"]:
        return empty
    if rows[cid]["seeds"][0]["release_hash"] is None:                       # continuous: identity = configuration
        return {**empty, "full": [cid], "representative": cid, "representative_family": rows[cid]["family"],
                "representative_construction": "continuous", "identity": "configuration ID (continuous release)"}
    me = rows[cid]["seeds"]
    full, part, canon = [], {}, []
    for c, r in rows.items():
        if not r["ok"] or r["seeds"][0]["release_hash"] is None:
            continue
        same = [k for k in SEEDS if me[k]["release_hash"] == r["seeds"][k]["release_hash"]]
        if len(same) == len(SEEDS):
            full.append(c)
        else:
            if same:
                part[c] = same
            if all(me[k]["canonical_hash"] == r["seeds"][k]["canonical_hash"] for k in SEEDS):
                canon.append(c)
    own = [c for c in full if pool is None or c in pool]                    # the role's own pool (contains cid)
    named = sorted(own or full, key=lambda c: _rep_key(rows, c))             # untrained aliases disclosed, not named
    rep = named[0]
    tie = len([c for c in named if _rep_key(rows, c)[:2] == _rep_key(rows, rep)[:2]]) > 1
    return {"full": sorted(full), "partial": part, "canonical_equivalent": sorted(canon), "representative": rep,
            "representative_family": rows[rep]["family"],
            "representative_construction": CONSTRUCTION_NAME.get(rows[rep]["arm"], rows[rep]["arm"]),
            "identical_to_untrained": sorted(c for c in full if not rows[c]["privacy_trained"]),
            "decided_by_config_id_tiebreak": tie,
            "identity": "sha256 of tok_i, q_i, hard_i on OSF_DEFENSE_FIT + AUDIT_FIT + INNER_SELECTION rows and alpha_i, "
                        "all three seeds"}


def winning_name(st):
    """'<family>; <construction>' of ONE real exact alias of P* (the representative), plus the identical_to_untrained
    disclosure (no privacy-training credit without a changed release)."""
    a = st.get("aliases") or {}
    if st.get("status") != "NOMINEE" or not a.get("representative"):
        return None
    name = f"{a['representative_family']}; {a['representative_construction']}"
    if a.get("identical_to_untrained"):
        name += "; identical to privacy-untrained " + ", ".join(a["identical_to_untrained"])
    return name


def role_aliases(resolved, statuses):
    """Role-level aliases by actual release identity (F07/F11): two roles alias iff they resolve to the same
    configuration or to configurations that are exact deployed-release aliases on all three seeds."""
    full = {x: set(((statuses.get(x) or {}).get("aliases") or {}).get("full") or ()) for x in ROLES}
    canon = {x: set(((statuses.get(x) or {}).get("aliases") or {}).get("canonical_equivalent") or ()) for x in ROLES}
    out, info = {}, {}
    for i, a in enumerate(ROLES):
        for b in ROLES[i + 1:]:
            ca, cb = resolved.get(a), resolved.get(b)
            if not (ca and cb):
                continue
            if ca == cb:
                out[f"{a}=={b}"] = ca
            elif cb in full[a] or ca in full[b]:
                out[f"{a}=={b}"] = f"{ca} == {cb} (exact release alias)"
            elif cb in canon[a] or ca in canon[b]:
                info[f"{a}~{b}"] = f"{ca} ~ {cb} (canonical token renaming; informational, not an alias)"
    return out, info


def inner_validation(D):
    if D is None:
        return {"ok": None, "status": "NOT_RUN (no data handle; synthetic call)"}
    from lra import audit as AU
    v = AU.validate_all(D)
    (R.RUN / "inner_validation.json").write_text(json.dumps(R._finite(v), indent=1, allow_nan=False) + "\n")
    return {"ok": bool(v["ok"]), "checked": v["checked"], "n_defects": len(v["defects"]),
            "n_missing": len(v["missing"]), "defects_head": v["defects"][:5], "missing_head": v["missing"][:5]}


def _rs(s):
    return FAM.role_state(s)


def paired_d0(cid):
    """The D0 mean-teacher decoded version of the EXACT same map (prompt sec. 13 stage 5): the admitted D0 map of a D1
    fixed-map control, the d0same diagnostic of a new D1 fit (C-TASK, weighted, constrained), None for a D0 code."""
    if not cid or R.parse_id(cid)["kind"] != "policy":
        return None
    a = arm(cid)
    if a == "d1_fixed":
        return cid[:-len("|D1")]
    if a in ("ctask", "weighted", "constrained"):
        return cid + R.D0SAME
    return None


def same_map_fit_inner(rows, pr):
    """Prompt sec. 11 (descriptive): D1 - D0 true-label log loss / Brier on the SAME map, per task, on OSF_DEFENSE_FIT
    (dec__ unit record 'fitting', mean over seeds) and on INNER_SELECTION (inner rows, mean over seeds). Only for pairs
    whose D0 member is in the bank; the d0same members are reported by their own unit records (build_d0same)."""
    d1, d0 = pr.get("d1"), pr.get("d0")
    out = {"fit": None, "inner": None}
    if not (d1 and d0) or d0.endswith(R.D0SAME):
        out["note"] = "same-map D0 diagnostic: fit/inner contrasts in the d0s__ unit records (stage d0same)"
        return out
    r1, r0 = rows.get(d1), rows.get(d0)
    if r1 and r0 and r1["ok"] and r0["ok"]:
        out["inner"] = {t: {m: sum(r1["seeds"][k]["utility"][t][m] - r0["seeds"][k]["utility"][t][m] for k in SEEDS) /
                            len(SEEDS) for m in ("logloss", "brier")} for t in TASKS}
    try:
        fs = [R.rec(R.unit_for(k, d1))["fitting"] for k in SEEDS]
        out["fit"] = {t: {m: sum(f[str(i)][f"{x}_D1"] - f[str(i)][f"{x}_D0"] for f in fs) / len(SEEDS)
                          for m, x in (("logloss", "L"), ("brier", "B"))} for i, t in ((1, "income"), (2, "occupation"))}
    except Exception as e:                                                   # noqa: BLE001  (descriptive only)
        out["fit_unavailable"] = f"{type(e).__name__}: {e}"
    return out


def _resolved(s):
    return (s or {}).get("config") or (s or {}).get("descriptive_config")


def select_all(D=None, shard_spec=None):
    ids = R.scored_ids()
    rows = candidate_rows(ids, permitted_rows(D) if D is not None else None)
    get = lambda cs: [rows[c] for c in cs]                                      # noqa: E731
    st = {}
    q = rows[Q_CONFIG]
    st["Q"] = ({"status": "INVALID_NOMINEE", "config": None, "reason": _tech_reason([q]), "fallback_class": "TECHNICAL",
                "descriptive_config": None} if not q["ok"] else
               {"status": "NOMINEE", "config": Q_CONFIG} if q["ordinary"] else
               {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": Q_CONFIG, "descriptive_only": True,
                "reason": "ORDINARY_UTILITY_FAILURE", "fallback_class": "UTILITY", "fallback_rank_status": "VALID"})
    st["T*"] = pick(get(T_STAR_CLOSED), nominee=False)
    gv = lambda x: rows[st[x]["config"]] if st[x]["status"] == "NOMINEE" else None   # noqa: E731
    st["P*"] = pick(get(private_all(ids)), guards={"T*": gv("T*")})
    st["C*"] = pick(get(incumbent_private(ids)), nominee=False)
    st["N*"] = pick(get(constrained(ids)), guards={"T*": gv("T*"), "C*": gv("C*")})
    cpair_pool = [c for c in ids if c != JOINT_PAIR and rows[c].get("fit_feasible") is not False]
    st["C_pair*"] = pick(get(cpair_pool), nominee=False)
    st["J*"] = pick(get([JOINT_PAIR]), guards={"T*": gv("T*"), "C_pair*": gv("C_pair*")})
    pools = {"P*": private_all(ids), "N*": constrained(ids), "J*": [JOINT_PAIR], "T*": list(T_STAR_CLOSED),
             "C*": incumbent_private(ids), "C_pair*": cpair_pool, "Q": [Q_CONFIG]}     # = the pick pools
    for x in ROLES:                                                           # F11: every role, comparators and Q
        st[x]["aliases"] = alias_set(rows, _resolved(st[x]), pool=set(pools[x]))
        st[x]["identical_to_untrained"] = st[x]["aliases"]["identical_to_untrained"] if x in PRIVACY_ROLES else []
    st["P*"]["winning"] = winning_name(st["P*"])
    if _resolved(st["P*"]):
        st["P*"]["config_arm"] = rows[_resolved(st["P*"])]["arm"]
    resolved = {x: _resolved(s) for x, s in st.items()}
    ral, rinfo = role_aliases(resolved, st)
    # prespecified diagnostics (scored-list members; never primary roles)
    d1p = [c for c in private_all(ids) if arm(c) == "d1_fixed"]
    diag = {"best_d1_fixed_privacy": pick(get(d1p), guards={"T*": gv("T*")}),
            "best_weighted_privacy": pick(get([c for c in ids if arm(c) == "weighted"]), guards={"T*": gv("T*")}),
            "per_constrained_arm": {c: {x: rows[c].get(x) for x in ("ok", "ordinary", "fit_feasible", "mean_pair",
                                                                     "mean_v1", "mean_v2", "ordinary_shortfall")}
                                    for c in constrained(ids)},
            "ctask": {x: rows[R.ctask_id()].get(x) for x in ("ok", "ordinary", "mean_pair", "mean_v1", "mean_v2")},
            "strongest_ordinary_private_unguarded": pick(get(private_all(ids)), guards={})}
    b = diag["best_d1_fixed_privacy"]
    b["paired_d0"] = paired_d0(_resolved(b))
    pc = _resolved(st["P*"])
    pairs = [{"name": "best_d1_fixed_privacy", "d1": _resolved(b), "d0": b["paired_d0"], "registered_sentence": True},
             {"name": "P*", "d1": pc, "d0": paired_d0(pc), "registered_sentence": False,
              "role_status": st["P*"]["status"]},
             {"name": "C-TASK", "d1": R.ctask_id(), "d0": paired_d0(R.ctask_id()), "registered_sentence": False},
             {"name": "CLASS", "d1": R.d1_id("CLASS"), "d0": R.d0_id("CLASS"), "registered_sentence": False}]
    diag["same_map_decoder_pairs"] = [p for p in pairs if p["d1"] and p["d0"]]
    for p in diag["same_map_decoder_pairs"]:
        p["fit_inner_d1_minus_d0"] = same_map_fit_inner(rows, p)
    diag["same_map_decoder_pairs_absent"] = [{**p, "why": "P* is a D0 release (no D1 decoder on that map)"
                                              if p["d1"] else "role unresolved (technical)"}
                                             for p in pairs if not (p["d1"] and p["d0"])]
    diag["d0same_targets"] = sorted({p["d1"] for p in diag["same_map_decoder_pairs"]
                                     if p["d0"].endswith(R.D0SAME)})
    claims = {c: {"nominee": _rs(st[n]), "comparator": _rs(st[m])} for c, (n, m) in
              {"A": ("P*", "T*"), "B": ("N*", "C*"), "C": ("J*", "C_pair*")}.items()}
    out = {"schema": "lra-selection-v2", "rule": "SELECTION_RULES.json; PROTOCOL.md section 10",
           "candidates": ids, "inner_validation": inner_validation(D), "statuses": st, "diagnostics": diag,
           "claim_role_states": claims, "resolved": resolved, "role_aliases": ral,
           "role_canonical_equivalence": rinfo,
           "release_identity_rows": None if D is None else {"roles": list(PERMITTED_ROLES),
                                                            "rows_sha256": permitted_rows(D)["rows_sha256"]},
           "technical_failures": {c: r["technical_failure"] for c, r in rows.items() if r["technical_failure"]},
           "rows": rows}
    out = R._finite(out)
    (R.RUN / "selection.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    pub = {k: v for k, v in out.items() if k != "rows"}
    pub["rows"] = {c: {x: r.get(x) for x in ("arm", "family", "privacy_trained", "private_code", "training", "ok",
                                            "fit_feasible", "ordinary_inner", "ordinary", "mean_pair",
                                            "mean_v1", "mean_v2", "mean_sum_logloss", "mean_states",
                                            "ordinary_shortfall")} for c, r in out["rows"].items()}
    (R.PKG / "SELECTION.json").write_text(json.dumps(pub, indent=1, allow_nan=False) + "\n")
    write_tables(out)
    R.event("selection written", statuses={x: [s["status"], _resolved(s)] for x, s in st.items()})
    return out


def write_tables(out):
    rows = out["rows"]
    f6 = lambda x: "" if x is None else f"{x:.6f}"                              # noqa: E731
    with open(R.PKG / "INNER_SELECTION_TABLE.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["config", "arm", "family", "lam", "training", "seed", "auc_v1", "auc_v2", "auc_pair",
                    "acc_income", "acc_occupation", "ll_income", "ll_occupation", "brier_income", "brier_occupation",
                    "ll_excess_income", "ll_excess_occupation", "brier_excess_income", "brier_excess_occupation",
                    "ordinary_seed", "ordinary_shortfall_seed", "token_states"])
        for c, r in rows.items():
            p = R.parse_id(c)
            lam = p.get("lam") if p["kind"] == "policy" else None
            for k, s in sorted(r["seeds"].items(), key=lambda x: int(x[0])):
                w.writerow([c, r["arm"], r["family"], "" if lam is None else f"{lam:g}", r["training"], k] +
                           [f6(s["auc"][v]) for v in VIEWS] + [f6(s["utility"][t]["acc"]) for t in TASKS] +
                           [f6(s["utility"][t]["logloss"]) for t in TASKS] +
                           [f6(s["utility"][t]["brier"]) for t in TASKS] + [f6(s["ll_excess"][t]) for t in TASKS] +
                           [f6(s["brier_excess"][t]) for t in TASKS] +
                           [s["ordinary"], f6(s["ordinary_shortfall"]),
                            "" if s["token_states"] is None else int(s["token_states"])])


# ------------------------------------------------------------------ same-map D0 diagnostic releases (stage d0same)
def build_d0same(k, cid, D, fit=None):
    """(release arrays, record) of the D0 mean-teacher decoded version of the EXACT map of the D1 release cid: the
    map's policy.json (token_proto = the pinned D0 vector) re-encoded on the same teacher outputs (qpc.release; no label
    read for the release). Refused unless tokens, decisions and alphabets are bitwise those of the D1 release."""
    from qpc import release as RL
    p = R.parse_id(cid)
    assert p["kind"] == "policy" and p["arm"] in ("ctask", "weighted", "constrained"), cid
    u1 = R.unit_for(k, cid)
    if not R.done(u1):
        raise SystemExit(f"REFUSED: {u1} is missing or not hash-complete")
    pair = RL.load_policy(R.U(u1) / "policy.json")
    T = R.teacher(k)
    rel = RL.release_arrays(pair, T["row_id"], T["p1"], T["d1"], T["p2"], T["d2"])
    z1 = R.npz(u1, "release.npz")
    for x in ("row_id", "tok1", "hard1", "alpha1", "tok2", "hard2", "alpha2"):
        if not np.array_equal(np.asarray(rel[x]), np.asarray(z1[x])):
            raise AssertionError(f"{cid} s{k}: the D0 re-encode differs from the D1 release on {x} (map not identical)")
    rec = {"schema": "lra-d0same-v1", "config": cid + R.D0SAME, "of": cid, "seed": k, "teacher": p["teacher"],
           "teacher_unit": f"tea__s{k}__{p['teacher']}", "d1_unit": u1,
           "policy_pair_fingerprint": pair.fingerprint(), "tokens_bitwise_equal_d1": True,
           "decoder": "D0 mean-teacher (policy token_proto, qpc.release.release_arrays)",
           "label_use": "none for the release; fitting/inner losses below read OSF_DEFENSE_FIT / INNER_SELECTION labels "
                        "through the allowlist (procedures fitting / selection) for the diagnostic only",
           "note": "diagnostic only: never a candidate, never composed, never nominated; same tokens as its D1 release"}
    if fit is not None:
        tr, Y, _S = fit
        out = {}
        for i, key_ in ((1, "income"), (2, "occupation")):
            for nm, zz in (("D0", rel), ("D1", z1)):
                pr = np.asarray(zz[f"q{i}"], dtype=np.float64)[tr]
                y = Y[i]
                ll = float(np.mean(-np.log(np.clip(pr[np.arange(len(y)), y], 1e-12, 1.0))))
                br = float(np.mean(((pr - np.eye(pr.shape[1])[y]) ** 2).sum(1)))
                out[f"{key_}_{nm}"] = {"logloss": ll, "brier": br}
        rec["fitting_losses"] = out
        rec["fitting_d1_minus_d0"] = {t: {m: out[f"{t}_D1"][m] - out[f"{t}_D0"][m] for m in ("logloss", "brier")}
                                      for t in TASKS}
    if D is not None:
        ui = {}
        for nm, zz in (("D0", rel), ("D1", z1)):
            ui[nm] = UT.release_inner_utility({1: zz["q1"], 2: zz["q2"]}, {1: zz["hard1"], 2: zz["hard2"]}, D)
        rec["inner_utility"] = {nm: {t: {m: v[t][m] for m in ("acc", "logloss", "brier")} for t in TASKS}
                                for nm, v in ui.items()}
        rec["inner_d1_minus_d0"] = {t: {m: ui["D1"][t][m] - ui["D0"][t][m] for m in ("logloss", "brier")}
                                    for t in TASKS}
    return rel, rec


def stage_d0same(D, shard_spec=None):
    """SCIENCE_LOCK stage after select: the d0same diagnostic units named by selection.json
    diagnostics.d0same_targets (P* when it is a new D1 fit, and C-TASK), all three seeds."""
    S = json.loads((R.RUN / "selection.json").read_text())
    targets = S["diagnostics"]["d0same_targets"]
    fit = R.fit_data(D) if D is not None else None
    jobs = [(k, c) for c in targets for k in SEEDS]
    for k, cid in R.shard(jobs, shard_spec):
        n = R.unit_for(k, cid + R.D0SAME)
        if R.done(n):
            continue
        rel, rec = build_d0same(k, cid, D, fit)
        R.save(n, {"release.npz": rel, "policy.json": (R.U(R.unit_for(k, cid)) / "policy.json").read_text()}, rec)
