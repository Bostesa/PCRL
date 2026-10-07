"""Registered inner-only nomination (SELECTION_RULES.json; PROTOCOL.md section 10; frozen in SCIENCE_LOCK).

Adapted from cbp/select.py at 7f3ec67 WITHOUT cbp's failed 0.006/0.0035 every-seed headroom buffer (prompt section 10:
the explicit 0.005/0.003 budgets act during FITTING; nomination uses the ORIGINAL inner eligibility). No assessment value
is read. Inputs: aud__<unit> records (lcr.run.inner_name; lcr.audit; attackers fitted on AUDIT_FIT, selected/scored on INNER_SELECTION;
utility on INNER_SELECTION via qpc.utility) for every candidate of lcr.run.scored_ids().

Per seed (anchor = SRC|U utility of that seed), with gr = qpc.utility.gate_record(code, U, preserved):
  ordinary eligible   gr["eligible"] on EVERY task (inclusive): acc >= U-0.01, LL <= U+0.01, Brier <= U+0.005,
                      gain >= 0.8 U gain, gain >= 0.03, decision preservation -> required on EVERY seed
  ordinary shortfall  max(0, (a_U-0.01-a)/0.01, (dLL-0.01)/0.01, (dB-0.005)/0.005, (0.8g_U-g)/max(0.8g_U,0.03),
                      (0.03-g)/0.03) over tasks and seeds
  guard               local inner AUC <= guard comparator local AUC + 0.005 (inclusive) for each recipient and seed
  guard shortfall     exactly 0.0 iff the guard holds, else max over guards/seeds/recipients of (a - g - 0.005)/0.005
Roles (prompt section 10):
  T*      strongest ordinarily eligible privacy-UNTRAINED release from the closed list T_STAR_CLOSED (RAW-J excluded)
  P*      strongest eligible privacy-trained release from ALL private arms (D0 privacy, D1 fixed-map privacy, weighted,
          constrained), guard vs T*
  C*      strongest eligible incumbent/baseline private release (D0 / D1 fixed-map privacy maps, weighted controls;
          NO constrained arm); comparator: ordinary eligibility only
  N*      strongest eligible NEW constrained release (any of the 5 arms), guards vs BOTH T* and C*
  C_pair* strongest eligible release other than K-JOINT-PAIR (every scored candidate except it); comparator
  J*      strongest eligible K-JOINT-PAIR release, guards vs BOTH T* and C_pair*
  Q       fixed D0 U|DIRECT-TASK|i8o64 (ordinary eligibility prerequisite)
Ordering: mean INNER pair AUC, mean summed true-label log loss (both float64 seed means (s0+s1+s2)/3 in seed order, then
round(x, 12); equal rounded values tie), mean actual token states (continuous = +inf for ordering only; null in JSON),
configuration ID. Fallback when no candidate is nominable: (ordinary shortfall, guard shortfall) then the ordering keys;
DESCRIPTIVE_ONLY. A missing / nonfinite field or a decision-preservation failure is a TECHNICAL failure of the role
(INVALID_*), never a shortfall. A required guard comparator without a valid nominee: INVALID (MISSING_GUARD_COMPARATOR)
when some candidate is eligible; otherwise NO_ELIGIBLE with fallback_rank_status INVALID_MISSING_GUARD_COMPARATOR.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math

import numpy as np

from lcr import run as R
from qpc import utility as UT

BUFFER = 0.005
SEEDS = R.SEEDS
VIEWS = ("v1", "v2", "pair")
TASKS = ("income", "occupation")
T_STAR_CLOSED = ("U|C-TASK|i8o64|D1", "U|DIRECT-TASK|i8o64", "U|DIRECT-TASK|i8o64|D1", "U|FINE-TASK|i8o64",
                 "U|FINE-TASK|i8o64|D1", "SRC|U", "U|CLASS|i1o1", "REF|F0")
Q_CONFIG = "U|DIRECT-TASK|i8o64"
JOINT_PAIR = "U|K-JOINT-PAIR|i8o64|D1"


def arm(cid):
    p = R.parse_id(cid)
    return p["kind"] if p["kind"] != "policy" else p["arm"]


def privacy_trained(cid):
    p = R.parse_id(cid)
    return p["kind"] == "policy" and p["privacy_trained"]


def private_all(ids):
    return [c for c in ids if privacy_trained(c)]


def incumbent_private(ids):
    return [c for c in ids if privacy_trained(c) and arm(c) in ("d0", "d1_fixed", "weighted")]


def constrained(ids):
    return [c for c in ids if arm(c) == "constrained"]


def _fin(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


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


def release_hash(k, cid):
    """sha256 of the DEPLOYED release (tok1, q1, tok2, q2 bytes) of a code unit; None for continuous releases."""
    if R.parse_id(cid)["kind"] != "policy":
        return None
    z = np.load(R.U(R.unit_for(k, cid)) / "release.npz")
    h = hashlib.sha256()
    for x in ("tok1", "q1", "tok2", "q2"):
        a = np.ascontiguousarray(z[x])
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def candidate_rows(ids):
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
            if not R.done(n):
                fail.append({"unit": n, "code": "FIT_OR_ADMISSION_FAILURE", "detail": "missing or not hash-complete"})
                continue
            r = R.rec(n)
            try:
                a = {w: float(r["recovery"]["auc"][w]) for w in VIEWS}
                u = r["utility"]
                pres = {int(i): bool(v) for i, v in r["preserved"].items()}
                assert set(pres) == {1, 2}
                assert all(_fin(a[w]) for w in VIEWS)
                assert all(_fin(u[t][x]) for t in TASKS for x in ("acc", "logloss", "brier", "const_acc"))
                assert anchors[k] is not None
                gr = UT.gate_record(u, anchors[k], pres)
                assert all(_fin(gr[t][x]) for t in TASKS for x in ("ll_excess", "brier_excess"))
                rh = release_hash(k, cid)
            except Exception as e:                                           # noqa: BLE001
                fail.append({"unit": n, "code": "NON_ESTIMABLE_INNER_METRIC", "detail": f"{type(e).__name__}: {e}"})
                continue
            if not all(pres.values()):
                fail.append({"unit": n, "code": "DECISION_PRESERVATION_FAILURE",
                             "detail": "decision preservation failed (invalid, not a shortfall)"})
                continue
            ts = r.get("token_states")
            seeds[k] = {"unit": R.unit_for(k, cid), "auc": a,
                        "utility": {t: {x: u[t][x] for x in ("acc", "logloss", "brier", "const_acc")} for t in TASKS},
                        "ll_excess": {t: gr[t]["ll_excess"] for t in TASKS},
                        "brier_excess": {t: gr[t]["brier_excess"] for t in TASKS},
                        "ordinary": bool(gr["eligible"]), "ordinary_shortfall": ordinary_shortfall(u, anchors[k]),
                        "token_states": float(ts) if ts is not None else None,
                        "composed_winner": (r.get("composed") or {}).get("winner"), "release_hash": rh}
        ok = not fail and len(seeds) == len(SEEDS)
        row = {"config": cid, "arm": arm(cid), "family": (R.parse_id(cid).get("base_family") or arm(cid)),
               "privacy_trained": privacy_trained(cid), "technical_failure": fail, "ok": ok, "seeds": seeds}
        if ok:
            mean = lambda f: (f(seeds[0]) + f(seeds[1]) + f(seeds[2])) / 3      # noqa: E731  seed order (rules)
            st = [s["token_states"] for s in seeds.values()]
            row.update({"mean_pair": mean(lambda s: s["auc"]["pair"]), "mean_v1": mean(lambda s: s["auc"]["v1"]),
                        "mean_v2": mean(lambda s: s["auc"]["v2"]),
                        "mean_sum_logloss": mean(lambda s: s["utility"]["income"]["logloss"] +
                                                 s["utility"]["occupation"]["logloss"]),
                        "mean_states": None if any(x is None for x in st) else sum(st) / len(st),
                        "ordinary": all(s["ordinary"] for s in seeds.values()),
                        "ordinary_shortfall": max(s["ordinary_shortfall"] for s in seeds.values())})
        rows[cid] = row
    return rows


def key(r):
    st = r["mean_states"] if r["mean_states"] is not None else math.inf
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
        return {"status": bad, "config": None, "reason": "FIT_OR_ADMISSION_FAILURE", "detail": "empty candidate set"}
    tf = [r for r in rows if not r["ok"]]
    if tf:
        return {"status": bad, "config": None, "reason": _tech_reason(tf), "failed": [r["config"] for r in tf],
                "failures": [f for r in tf for f in r["technical_failure"]]}
    guards = guards or {}
    missing = sorted(g for g, row in guards.items() if row is None)
    live = {g: row for g, row in guards.items() if row is not None}
    ev = []
    for r in rows:
        ok = (not missing) and guard_ok(r, live)
        ev.append({"config": r["config"], "arm": r["arm"], "family": r["family"], "ordinary": r["ordinary"],
                   "ordinary_shortfall": r["ordinary_shortfall"],
                   "guard_shortfall": None if missing else guard_shortfall(r, live), "eligible": r["ordinary"],
                   "guard_ok": ok, "nominable": r["ordinary"] and ok})
    byc = {r["config"]: r for r in rows}
    if missing and any(e["eligible"] for e in ev):
        return {"status": bad, "config": None, "reason": "MISSING_GUARD_COMPARATOR", "missing_guards": missing,
                "blocked": [e["config"] for e in ev if e["eligible"]], "evaluated": ev}
    el = [byc[e["config"]] for e in ev if e["nominable"]]
    if el:
        return {"status": "NOMINEE", "config": min(el, key=key)["config"], "evaluated": ev}
    reason = "ORDINARY_UTILITY_FAILURE" if not any(e["ordinary"] for e in ev) else "LOCAL_GUARD_FAILURE"
    out = {"status": none, "config": None, "descriptive_only": True, "reason": reason, "evaluated": ev}
    if missing:
        fb = min(ev, key=lambda e: (round(e["ordinary_shortfall"], 12),) + key(byc[e["config"]]))
        out.update({"missing_guards": missing, "fallback_rank_status": "INVALID_MISSING_GUARD_COMPARATOR",
                    "fallback_rank_keys": ["ordinary_shortfall", "ordering"]})
    else:
        fb = min(ev, key=lambda e: (round(e["ordinary_shortfall"], 12), round(e["guard_shortfall"], 12)) +
                 key(byc[e["config"]]))
        out["fallback_rank_status"] = "VALID"
    out["descriptive_config"] = fb["config"]
    return out


def alias_set(rows, cid):
    """Configurations whose DEPLOYED release (tokens AND released probabilities) equals cid's on every seed (full) or
    some seeds (partial). A D1 fixed-map release shares tokens with its D0 map but not probabilities: not an alias."""
    if not cid or cid not in rows or not rows[cid]["ok"] or rows[cid]["seeds"][0]["release_hash"] is None:
        return {"full": [], "partial": {}, "simplest": None}
    me = rows[cid]["seeds"]
    full, part = [], {}
    for c, r in rows.items():
        if not r["ok"] or r["seeds"][0]["release_hash"] is None:
            continue
        same = [k for k in SEEDS if me[k]["release_hash"] == r["seeds"][k]["release_hash"]]
        if len(same) == len(SEEDS):
            full.append(c)
        elif same:
            part[c] = same
    from lcr import family as FAM
    fams = [rows[c]["family"] for c in full]
    cons = sorted({rows[c]["arm"] for c in full}, key=lambda a: FAM.CONSTRUCTION_SIMPLICITY.get(a, 9))
    return {"full": sorted(full), "partial": part, "simplest_family": FAM.simplest_family(fams) if fams else None,
            "simplest_construction": FAM.CONSTRUCTION_NAME.get(cons[0]) if cons else None,
            "decided_by_config_id_tiebreak": len(full) > 1}


def winning_name(rows, st):
    """'<family>; <construction>' of P* over its exact alias set (simplest story)."""
    a = st.get("aliases") or {}
    if st.get("status") != "NOMINEE" or not a.get("full"):
        return None
    return f"{a['simplest_family']}; {a['simplest_construction']}"


def inner_validation(D):
    if D is None:
        return {"ok": None, "status": "NOT_RUN (no data handle; synthetic call)"}
    from lcr import audit as AU
    v = AU.validate_all(D)
    (R.RUN / "inner_validation.json").write_text(json.dumps(R._finite(v), indent=1, allow_nan=False) + "\n")
    return {"ok": bool(v["ok"]), "checked": v["checked"], "n_defects": len(v["defects"]),
            "n_missing": len(v["missing"]), "defects_head": v["defects"][:5], "missing_head": v["missing"][:5]}


def _rs(s):
    from lcr import family as FAM
    return FAM.role_state(s)


def select_all(D=None, shard_spec=None):
    ids = R.scored_ids()
    rows = candidate_rows(ids)
    get = lambda cs: [rows[c] for c in cs]                                      # noqa: E731
    st = {}
    q = rows[Q_CONFIG]
    st["Q"] = ({"status": "INVALID_NOMINEE", "config": None, "reason": _tech_reason([q])} if not q["ok"] else
               {"status": "NOMINEE", "config": Q_CONFIG} if q["ordinary"] else
               {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": Q_CONFIG,
                "reason": "ORDINARY_UTILITY_FAILURE"})
    st["T*"] = pick(get(T_STAR_CLOSED), nominee=False)
    gv = lambda x: rows[st[x]["config"]] if st[x]["status"] == "NOMINEE" else None   # noqa: E731
    st["P*"] = pick(get(private_all(ids)), guards={"T*": gv("T*")})
    st["C*"] = pick(get(incumbent_private(ids)), nominee=False)
    st["N*"] = pick(get(constrained(ids)), guards={"T*": gv("T*"), "C*": gv("C*")})
    st["C_pair*"] = pick(get([c for c in ids if c != JOINT_PAIR]), nominee=False)
    st["J*"] = pick(get([JOINT_PAIR]), guards={"T*": gv("T*"), "C_pair*": gv("C_pair*")})
    for x in ("P*", "N*", "J*"):
        st[x]["aliases"] = alias_set(rows, st[x].get("config") or st[x].get("descriptive_config"))
    st["P*"]["winning"] = winning_name(rows, st["P*"])
    if st["P*"]["status"] == "NOMINEE":
        st["P*"]["config_arm"] = rows[st["P*"]["config"]]["arm"]
    # prespecified diagnostics (scored-list members; never primary roles)
    d1p = [c for c in private_all(ids) if arm(c) == "d1_fixed"]
    diag = {"best_d1_fixed_privacy": pick(get(d1p), guards={"T*": gv("T*")}),
            "best_weighted_privacy": pick(get([c for c in ids if arm(c) == "weighted"]), guards={"T*": gv("T*")}),
            "per_constrained_arm": {c: {x: rows[c].get(x) for x in ("ok", "ordinary", "mean_pair", "mean_v1", "mean_v2",
                                                                     "ordinary_shortfall")} for c in constrained(ids)},
            "ctask": {x: rows[R.ctask_id()].get(x) for x in ("ok", "ordinary", "mean_pair", "mean_v1", "mean_v2")},
            "strongest_ordinary_private_unguarded": pick(get(private_all(ids)), guards={})}
    b = diag["best_d1_fixed_privacy"]
    bc = b.get("config") or b.get("descriptive_config")
    diag["best_d1_fixed_privacy"]["paired_d0"] = bc[:-3] if bc and bc.endswith("|D1") else None
    claims = {c: {"nominee": _rs(st[n]), "comparator": _rs(st[m])} for c, (n, m) in
              {"A": ("P*", "T*"), "B": ("N*", "C*"), "C": ("J*", "C_pair*")}.items()}
    out = {"schema": "lcr-selection-v1", "rule": "SELECTION_RULES.json; PROTOCOL.md section 10",
           "candidates": ids, "inner_validation": inner_validation(D), "statuses": st, "diagnostics": diag,
           "claim_role_states": claims,
           "resolved": {x: (s.get("config") or s.get("descriptive_config")) for x, s in st.items()},
           "technical_failures": {c: r["technical_failure"] for c, r in rows.items() if r["technical_failure"]},
           "rows": rows}
    out = R._finite(out)
    (R.RUN / "selection.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    pub = {k: v for k, v in out.items() if k != "rows"}
    pub["rows"] = {c: {x: r.get(x) for x in ("arm", "family", "privacy_trained", "ok", "ordinary", "mean_pair",
                                            "mean_v1", "mean_v2", "mean_sum_logloss", "mean_states",
                                            "ordinary_shortfall")} for c, r in out["rows"].items()}
    (R.PKG / "SELECTION.json").write_text(json.dumps(pub, indent=1, allow_nan=False) + "\n")
    write_tables(out)
    R.event("selection written", statuses={x: [s["status"], s.get("config") or s.get("descriptive_config")]
                                           for x, s in st.items()})
    return out


def write_tables(out):
    rows = out["rows"]
    f6 = lambda x: "" if x is None else f"{x:.6f}"                              # noqa: E731
    with open(R.PKG / "INNER_SELECTION_TABLE.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["config", "arm", "family", "lam", "privacy_trained", "seed", "auc_v1", "auc_v2", "auc_pair",
                    "acc_income", "acc_occupation", "ll_income", "ll_occupation", "brier_income", "brier_occupation",
                    "ll_excess_income", "ll_excess_occupation", "brier_excess_income", "brier_excess_occupation",
                    "ordinary_seed", "ordinary_shortfall_seed", "token_states"])
        for c, r in rows.items():
            p = R.parse_id(c)
            lam = p.get("lam") if p["kind"] == "policy" else None
            for k, s in sorted(r["seeds"].items(), key=lambda x: int(x[0])):
                w.writerow([c, r["arm"], r["family"], "" if lam is None else f"{lam:g}", r["privacy_trained"], k] +
                           [f6(s["auc"][v]) for v in VIEWS] + [f6(s["utility"][t]["acc"]) for t in TASKS] +
                           [f6(s["utility"][t]["logloss"]) for t in TASKS] +
                           [f6(s["utility"][t]["brier"]) for t in TASKS] + [f6(s["ll_excess"][t]) for t in TASKS] +
                           [f6(s["brier_excess"][t]) for t in TASKS] +
                           [s["ordinary"], f6(s["ordinary_shortfall"]),
                            "" if s["token_states"] is None else int(s["token_states"])])
