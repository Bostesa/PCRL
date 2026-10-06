"""Registered inner-only selection (HEADROOM_SELECTION_RULES.json; PROTOCOL.md section 9; frozen before any fit).

Adapted from qpc/select.py at d0c8a45. No assessment value is read. Inputs: inner__<unit> records (cbp.audit; attackers
fitted on AUDIT_FIT and selected/scored on INNER_SELECTION; utility on INNER_SELECTION via qpc.utility) for every
candidate of cbp.run.scored_ids() (the 11 composition-only maps are never candidates).

Per seed (anchor = SRC|U utility of that seed), with g = gate = qpc.utility.gate_record(code, U, preserved):
  ordinary eligible   g["eligible"] (every task; inclusive)                     -> required on EVERY seed
  headroom            ordinary AND ll_excess <= 0.006 AND brier_excess <= 0.0035 for EVERY task (recomputed here from
                      the excesses; qpc's own 0.0075 'headroom' flag is NOT used)
  ordinary shortfall  max(0, (a_U - 0.01 - a)/0.01, (dLL - 0.01)/0.01, (dB - 0.005)/0.005,
                         (0.8 g_U - g)/max(0.8 g_U, 0.03), (0.03 - g)/0.03) over tasks and seeds
  headroom shortfall  max(0, (dLL - 0.006)/0.006, (dB - 0.0035)/0.0035) over tasks and seeds
  guard shortfall     max(0, (local AUC - comparator local AUC - 0.005)/0.005) over guards, seeds, recipients
Roles: T* (closed list, ordinary), C_rate (fixed rate nonjoint, ordinary), C_global (all nonjoint, ordinary),
P* (privacy, headroom, guard T*), J* (JOINT, headroom, guards C_rate and C_global), Q (fixed DIRECT-TASK i8o64).
Ordering: mean pair AUC, mean summed log loss (both rounded to 12 decimals), mean actual states (continuous = +inf for
ordering only; null in JSON), configuration ID. Fallback: (ordinary, headroom, guard shortfalls, ordering key).
A missing/nonfinite field or a decision-preservation failure is a technical failure (INVALID), never a shortfall.
"""
from __future__ import annotations

import csv
import json
import math

from cbp import run as R
from qpc import utility as UT

BUFFER = 0.005
HEAD_LL, HEAD_BR = 0.006, 0.0035
SEEDS = R.SEEDS
VIEWS = ("v1", "v2", "pair")
TASKS = ("income", "occupation")
PRIVACY = R.PRIVACY
NONJOINT_CODES = ("DIRECT-TASK", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
T_STAR_CLOSED = ("U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "SRC|U", "U|CLASS|i1o1", "REF|F0")
Q_CONFIG = "U|DIRECT-TASK|i8o64"
SOURCE_L01 = ("U|JOINT|i8o64|l0.1", "U|SEQ-12|i8o64|l0.1", "U|SEQ-21|i8o64|l0.1")


def family(cid):
    if cid.startswith("SRC|"):
        return "SRC"
    if cid.startswith("REF|"):
        return "REF"
    return cid.split("|")[1]


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


def headroom_shortfall(gr):
    return max(0.0, max(max((gr[t]["ll_excess"] - HEAD_LL) / HEAD_LL, (gr[t]["brier_excess"] - HEAD_BR) / HEAD_BR)
                        for t in TASKS))


def candidate_rows(ids):
    anchors = {}
    for k in SEEDS:
        try:
            anchors[k] = R.rec(f"inner__{R.unit_for(k, 'SRC|U')}")["utility"]
        except Exception:                                                    # noqa: BLE001
            anchors[k] = None
    rows = {}
    for cid in ids:
        seeds, fail = {}, []
        for k in SEEDS:
            n = f"inner__{R.unit_for(k, cid)}"
            if not R.done(n):
                fail.append(f"{n}: missing or not hash-complete")
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
            except Exception as e:                                           # noqa: BLE001
                fail.append(f"{n}: {type(e).__name__}: {e}")
                continue
            if not all(pres.values()):
                fail.append(f"{n}: decision preservation failed (invalid, not a shortfall)")
                continue
            ts = r.get("token_states")
            fp = None
            if R.parse_id(cid)["kind"] == "policy":
                try:
                    fp = R.rec(R.unit_for(k, cid)).get("pair_fingerprint")
                except Exception:                                            # noqa: BLE001
                    fp = None
            ordinary = bool(gr["eligible"])
            head = ordinary and all(gr[t]["ll_excess"] <= HEAD_LL and gr[t]["brier_excess"] <= HEAD_BR for t in TASKS)
            seeds[k] = {"unit": R.unit_for(k, cid), "auc": a,
                        "utility": {t: {x: u[t][x] for x in ("acc", "logloss", "brier", "const_acc")} for t in TASKS},
                        "ll_excess": {t: gr[t]["ll_excess"] for t in TASKS},
                        "brier_excess": {t: gr[t]["brier_excess"] for t in TASKS},
                        "ordinary": ordinary, "headroom": head,
                        "ordinary_shortfall": ordinary_shortfall(u, anchors[k]),
                        "headroom_shortfall": headroom_shortfall(gr),
                        "token_states": float(ts) if ts is not None else None,
                        "composed_winner": (r.get("composed") or {}).get("winner"), "pair_fingerprint": fp}
        ok = not fail and len(seeds) == len(SEEDS)
        row = {"config": cid, "family": family(cid), "technical_failure": fail, "ok": ok, "seeds": seeds}
        if ok:
            mean = lambda f: (f(seeds[0]) + f(seeds[1]) + f(seeds[2])) / 3      # noqa: E731  seed order (rules)
            st = [s["token_states"] for s in seeds.values()]
            row.update({"mean_pair": mean(lambda s: s["auc"]["pair"]), "mean_v1": mean(lambda s: s["auc"]["v1"]),
                        "mean_v2": mean(lambda s: s["auc"]["v2"]),
                        "mean_sum_logloss": mean(lambda s: s["utility"]["income"]["logloss"] +
                                                 s["utility"]["occupation"]["logloss"]),
                        "mean_states": None if any(x is None for x in st) else sum(st) / len(st),
                        "ordinary": all(s["ordinary"] for s in seeds.values()),
                        "headroom": all(s["headroom"] for s in seeds.values()),
                        "ordinary_shortfall": max(s["ordinary_shortfall"] for s in seeds.values()),
                        "headroom_shortfall": max(s["headroom_shortfall"] for s in seeds.values())})
        rows[cid] = row
    return rows


def key(r):
    st = r["mean_states"] if r["mean_states"] is not None else math.inf
    return (round(r["mean_pair"], 12), round(r["mean_sum_logloss"], 12), st, r["config"])


def guard_shortfall(r, guards):
    xs = [(r["seeds"][k]["auc"][w] - g["seeds"][k]["auc"][w] - BUFFER) / BUFFER
          for g in guards.values() for k in SEEDS for w in ("v1", "v2")]
    return max(0.0, max(xs)) if xs else 0.0


def pick(rows, guards=None, need_headroom=False, nominee=True):
    """rows: candidate rows of one role; guards: {name: row or None (guard role has no valid nominee)}."""
    none = "NO_ELIGIBLE_NOMINEE" if nominee else "NO_ELIGIBLE_COMPARATOR"
    bad = "INVALID_NOMINEE" if nominee else "INVALID_COMPARATOR"
    if not rows:
        return {"status": bad, "config": None, "reason": "FIT_OR_ADMISSION_FAILURE", "detail": "empty candidate set"}
    tf = [r["config"] for r in rows if not r["ok"]]
    if tf:
        return {"status": bad, "config": None, "reason": "FIT_OR_ADMISSION_FAILURE", "failed": tf}
    guards = guards or {}
    missing = sorted(g for g, row in guards.items() if row is None)
    live = {g: row for g, row in guards.items() if row is not None}
    ev = []
    for r in rows:
        elig = r["ordinary"] and (r["headroom"] or not need_headroom)
        gs = None if missing else guard_shortfall(r, live)
        ev.append({"config": r["config"], "family": r["family"], "ordinary": r["ordinary"], "headroom": r["headroom"],
                   "ordinary_shortfall": r["ordinary_shortfall"],
                   "headroom_shortfall": r["headroom_shortfall"] if need_headroom else 0.0,
                   "guard_shortfall": gs, "eligible": elig, "guard_ok": gs == 0.0, "nominable": elig and gs == 0.0})
    byc = {r["config"]: r for r in rows}
    if missing and any(e["eligible"] for e in ev):
        return {"status": bad, "config": None, "reason": "MISSING_GUARD_COMPARATOR", "missing_guards": missing,
                "blocked": [e["config"] for e in ev if e["eligible"]], "evaluated": ev}
    el = [byc[e["config"]] for e in ev if e["nominable"]]
    if el:
        return {"status": "NOMINEE", "config": min(el, key=key)["config"], "evaluated": ev}
    reason = ("ORDINARY_UTILITY_FAILURE" if not any(e["ordinary"] for e in ev) else
              "HEADROOM_SELECTION_FAILURE" if need_headroom and not any(e["eligible"] for e in ev) else
              "LOCAL_GUARD_FAILURE")
    fb = min(ev, key=lambda e: (round(e["ordinary_shortfall"], 12), round(e["headroom_shortfall"], 12),
                                round(e["guard_shortfall"] or 0.0, 12)) + key(byc[e["config"]]))
    out = {"status": none, "config": None, "descriptive_config": fb["config"], "descriptive_only": True,
           "reason": reason, "evaluated": ev}
    if missing:
        out["missing_guards_not_needed"] = missing
    return out


def alias_set(rows, cid):
    """Privacy configurations whose deployed pair maps equal cid's on ALL seeds (full) or on some seeds (partial)."""
    if not cid or cid not in rows or rows[cid]["family"] not in PRIVACY or not rows[cid]["ok"]:
        return {"full": [], "partial": {}, "simplest_family": None}
    me = rows[cid]["seeds"]
    full, part = [], {}
    for c, r in rows.items():
        if r["family"] not in PRIVACY or not r["ok"]:
            continue
        same = [k for k in SEEDS if me[k]["pair_fingerprint"] and me[k]["pair_fingerprint"] ==
                r["seeds"][k]["pair_fingerprint"]]
        if len(same) == len(SEEDS):
            full.append(c)
        elif same:
            part[c] = same
    from cbp import family as FAM
    return {"full": sorted(full), "partial": part,
            "simplest_family": FAM.simplest_family([rows[c]["family"] for c in full]) if full else None,
            "decided_by_config_id_tiebreak": len({rows[c]["family"] for c in full}) > 1}


def _give_up(rows, a, b):
    """mean and per-seed inner pair AUC(a) - pair AUC(b); a status string when either is absent."""
    if not a or not b:
        return {"status": "ABSENT", "mean": None, "per_seed": None}
    return {"status": "OK", "mean": rows[a]["mean_pair"] - rows[b]["mean_pair"],
            "per_seed": {k: rows[a]["seeds"][k]["auc"]["pair"] - rows[b]["seeds"][k]["auc"]["pair"] for k in SEEDS}}


def select_all(D=None, shard_spec=None):
    ids = R.scored_ids()
    rows = candidate_rows(ids)
    allr = list(rows.values())
    priv = [r for r in allr if r["family"] in PRIVACY]
    st = {}
    q = rows[Q_CONFIG]
    st["Q"] = ({"status": "INVALID_NOMINEE", "config": None, "reason": "FIT_OR_ADMISSION_FAILURE"} if not q["ok"] else
               {"status": "NOMINEE", "config": Q_CONFIG} if q["ordinary"] else
               {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": Q_CONFIG,
                "reason": "ORDINARY_UTILITY_FAILURE"})
    st["T*"] = pick([rows[c] for c in T_STAR_CLOSED], nominee=False)
    st["C_rate"] = pick([r for r in allr if r["family"] in NONJOINT_CODES], nominee=False)
    st["C_global"] = pick([r for r in allr if r["family"] in NONJOINT_CODES + ("CLASS", "SRC", "REF")], nominee=False)
    g = lambda x: rows[st[x]["config"]] if st[x]["status"] == "NOMINEE" else None   # noqa: E731
    st["P*"] = pick(priv, guards={"T*": g("T*")}, need_headroom=True)
    st["J*"] = pick([r for r in priv if r["family"] == "JOINT"], guards={"C_rate": g("C_rate"),
                                                                         "C_global": g("C_global")}, need_headroom=True)
    for x in ("P*", "J*"):
        c = st[x].get("config") or st[x].get("descriptive_config")
        st[x]["aliases"] = alias_set(rows, c)
    if st["P*"]["status"] == "NOMINEE":
        st["P*"]["config_family"] = family(st["P*"]["config"])
        st["P*"]["winning_family"] = st["P*"]["aliases"]["simplest_family"] or family(st["P*"]["config"])
    # prespecified diagnostics (never roles in the primary family)
    fam_w = {f: pick([r for r in priv if r["family"] == f], guards={"T*": g("T*")}, need_headroom=True)
             for f in PRIVACY}
    for f, w in fam_w.items():
        w["aliases"] = alias_set(rows, w.get("config") or w.get("descriptive_config"))
    diag = {"ordinary_privacy_winner_no_headroom": pick(priv, guards={"T*": g("T*")}, need_headroom=False),
            "strongest_ordinary_privacy_unguarded": pick(priv, guards={}, need_headroom=False),
            "family_headroom_winners": fam_w,
            "joint_family_winner_is_not_J*": {"joint_family_T*_guarded": fam_w["JOINT"].get("config") or
                                              fam_w["JOINT"].get("descriptive_config"),
                                              "J*": st["J*"].get("config") or st["J*"].get("descriptive_config")},
            "source_lambda_0.1_controls": {c: {x: rows[c].get(x) for x in ("ok", "ordinary", "headroom", "mean_pair",
                                                                         "mean_v1", "mean_v2", "ordinary_shortfall",
                                                                         "headroom_shortfall")} for c in SOURCE_L01}}
    ow, pw = diag["ordinary_privacy_winner_no_headroom"], st["P*"]
    oc, pc = ow.get("config"), pw.get("config")
    diag["headroom_changes_winner"] = {
        "ordinary_winner": oc, "ordinary_winner_status": ow["status"], "headroom_winner": pc,
        "headroom_winner_status": pw["status"], "changed": (oc != pc) if oc and pc else None,
        "pair_auc_given_up_by_headroom": _give_up(rows, pc, oc),
        "descriptive_fallbacks": {"ordinary": ow.get("descriptive_config"), "headroom": pw.get("descriptive_config"),
                                  "give_up_DESCRIPTIVE": _give_up(rows, pc or pw.get("descriptive_config"),
                                                                  oc or ow.get("descriptive_config"))}}
    claims = {c: {"nominee": _rs(st[n]), "comparator": _rs(st[m])} for c, (n, m) in
              {"A": ("J*", "C_rate"), "B": ("J*", "C_global"), "C": ("P*", "T*")}.items()}
    out = {"schema": "cbp-selection-v1", "rule": "HEADROOM_SELECTION_RULES.json; PROTOCOL.md section 9",
           "candidates": ids, "statuses": st, "diagnostics": diag, "claim_role_states": claims,
           "resolved": {x: (s.get("config") or s.get("descriptive_config")) for x, s in st.items()},
           "technical_failures": {c: r["technical_failure"] for c, r in rows.items() if r["technical_failure"]},
           "rows": rows}
    out = R._finite(out)
    (R.RUN / "selection.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    pub = {k: v for k, v in out.items() if k != "rows"}
    pub["rows"] = {c: {x: r.get(x) for x in ("family", "ok", "ordinary", "headroom", "mean_pair", "mean_v1", "mean_v2",
                                            "mean_sum_logloss", "mean_states", "ordinary_shortfall",
                                            "headroom_shortfall")} for c, r in out["rows"].items()}
    (R.PKG / "SELECTION.json").write_text(json.dumps(pub, indent=1, allow_nan=False) + "\n")
    write_tables(out)
    R.event("selection written", statuses={x: [s["status"], s.get("config") or s.get("descriptive_config")]
                                           for x, s in st.items()})
    return out


def _rs(s):
    from cbp import family as FAM
    return FAM.role_state(s)


def write_tables(out):
    rows = out["rows"]
    with open(R.PKG / "INNER_SELECTION_TABLE.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["config", "family", "lam", "seed", "auc_v1", "auc_v2", "auc_pair", "ll_income", "ll_occupation",
                    "brier_income", "brier_occupation", "ll_excess_income", "ll_excess_occupation",
                    "brier_excess_income", "brier_excess_occupation", "ordinary_seed", "headroom_seed",
                    "ordinary_shortfall_seed", "headroom_shortfall_seed", "token_states", "composed_winner_pair"])
        for c, r in rows.items():
            lam = R.parse_id(c).get("lam") if R.parse_id(c)["kind"] == "policy" else None
            for k, s in sorted(r["seeds"].items(), key=lambda x: int(x[0])):
                f6 = lambda x: "" if x is None else f"{x:.6f}"                  # noqa: E731
                w.writerow([c, r["family"], "" if lam is None else f"{lam:g}", k] +
                           [f6(s["auc"][v]) for v in VIEWS] + [f6(s["utility"][t]["logloss"]) for t in TASKS] +
                           [f6(s["utility"][t]["brier"]) for t in TASKS] + [f6(s["ll_excess"][t]) for t in TASKS] +
                           [f6(s["brier_excess"][t]) for t in TASKS] +
                           [s["ordinary"], s["headroom"], f6(s["ordinary_shortfall"]), f6(s["headroom_shortfall"]),
                            "" if s["token_states"] is None else int(s["token_states"]),
                            (s["composed_winner"] or {}).get("pair", "") if s["composed_winner"] else ""])
    d = out["diagnostics"]
    with open(R.PKG / "HEADROOM_VS_STANDARD_SELECTION.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["selection", "rule", "status", "config", "reason", "mean_pair", "mean_v1", "mean_v2",
                    "worst_ll_excess", "worst_brier_excess", "ordinary", "headroom"])

        def put(name, rule, s):
            c = s.get("config") or s.get("descriptive_config")
            r = rows.get(c) if c else None
            wl = max(x["ll_excess"][t] for x in r["seeds"].values() for t in TASKS) if r and r["ok"] else None
            wb = max(x["brier_excess"][t] for x in r["seeds"].values() for t in TASKS) if r and r["ok"] else None
            f6 = lambda x: "" if x is None else f"{x:.6f}"                      # noqa: E731
            w.writerow([name, rule, s["status"], c or "", s.get("reason") or ""] +
                       ([f6(r.get("mean_pair")), f6(r.get("mean_v1")), f6(r.get("mean_v2")), f6(wl), f6(wb),
                         r.get("ordinary"), r.get("headroom")] if r else [""] * 7))
        put("P* (headroom)", "headroom + T* guard", out["statuses"]["P*"])
        put("ordinary privacy winner", "ordinary + T* guard (no headroom)", d["ordinary_privacy_winner_no_headroom"])
        for fam, s in d["family_headroom_winners"].items():
            put(f"{fam} headroom winner", "headroom + T* guard", s)
        put("J* (headroom)", "headroom + C_rate and C_global guards", out["statuses"]["J*"])
