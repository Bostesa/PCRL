"""Registered inner-only selection (PROTOCOL.md sections 7-8). Inputs: inner__<unit> records written by dpc.run's inner
stage (attackers fitted on AUDIT_FIT, selected and scored on INNER_SELECTION; utility on INNER_SELECTION). No
assessment value is read anywhere.

Candidates (one GLOBAL configuration across seeds 0-2):
  policies   "<teacher>|<FAMILY>|m<m>[|l<lam>]" for FINE-TASK, DIRECT-TASK, LOCAL, SEQ-12, SEQ-21, JOINT, CLASS (m1)
  sources    "SRC|<teacher>": the teacher's continuous score interface [centred scores, probabilities]; its inner
             recovery is COMPOSED: per view the max inner AUC over its own FINAL-slate candidates and the code readers of
             every fitted policy of the same teacher and seed (the maps are public), fixed before any selection
  references "REF|E", "REF|F", "REF|F0": official reference score releases
Inner task eligibility (each seed, both tasks; anchor = SRC|U utility on that seed):
  acc >= U - 0.01;  logloss <= U + 0.01;  Brier <= U + 0.005;  acc - const >= 0.8 (U - const);  acc - const >= 0.03;
  policies must also pass pointwise class preservation against their own teacher.
Tie rule (every pick): lower mean inner pair AUC, then lower mean task log loss, then fewer token states, then id.
  C_match(t, m)  nonjoint policies (FINE-TASK, DIRECT-TASK, LOCAL, SEQ-12, SEQ-21) of teacher t and rate m, all lambda
  C_global       nonjoint policies (all teachers/rates) + CLASS + sources + references
  J*             JOINT, task-eligible, locals <= C_match(t, m) + 0.005 and <= C_global + 0.005 on every seed and recipient
  T*             privacy-untrained: FINE-TASK, DIRECT-TASK, CLASS, sources (both teachers, all rates) and the official
                 no-fairness FARE compression REF|F0 (review R1)
  P*             LOCAL, SEQ-12, SEQ-21, JOINT; task-eligible; locals <= T* + 0.005 on every seed and recipient
No eligible candidate -> NO_FEASIBLE_NOMINEE (comparators: NO_FEASIBLE_CONTROL) with a deterministic minimum-shortfall
fallback (summed positive gate shortfalls in native units + guard excess in AUC units, then the tie rule) that is
DESCRIPTIVE_ONLY. A missing guard comparator makes the dependent nominee INVALID_MISSING_COMPARATOR.
Deployable compact release: lowest-tie-rule task-eligible finite-code policy (any family incl. CLASS).
"""
from __future__ import annotations

import csv
import json

from dpc import run as R

BUFFER = 0.005
SEEDS = R.SEEDS
NONJOINT = ("FINE-TASK", "DIRECT-TASK", "LOCAL", "SEQ-12", "SEQ-21")
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
UNTRAINED = ("FINE-TASK", "DIRECT-TASK", "CLASS")


def family(cid):
    if cid.startswith("SRC|"):
        return "SRC"
    if cid.startswith("REF|"):
        return "REF"
    return cid.split("|")[1]


def teacher_of(cid):
    return cid.split("|")[1] if cid.startswith("SRC|") else (None if cid.startswith("REF|") else cid.split("|")[0])


def rate_of(cid):
    if family(cid) in ("SRC", "REF"):
        return None
    return int(cid.split("|")[2][1:])


def inner_rec(k, cid):
    return R.rec(f"inner__{R.unit_for(k, cid)}")


def utility(rec):
    return {int(i): v for i, v in rec["utility"].items()}


def gate_margins(u, uref):
    out = {}
    for i in (0, 1):
        a, ll, br, c = u[i]["acc"], u[i]["logloss"], u[i]["brier"], u[i]["const_acc"]
        ua, ull, ubr = uref[i]["acc"], uref[i]["logloss"], uref[i]["brier"]
        out[i] = {"acc": a - (ua - 0.01), "logloss": (ull + 0.01) - ll, "brier": (ubr + 0.005) - br,
                  "retain": (a - c) - 0.8 * (ua - c), "gain": (a - c) - 0.03}
    return out


def shortfall(gm):
    return sum(max(0.0, -x) for i in gm for x in gm[i].values())


def composed_auc(k, cid, own, policy_ids):
    """Source score attacker: max over its own candidates and every same-teacher, same-seed policy's code readers."""
    best = dict(own)
    winner = {w: "source" for w in own}
    for p in policy_ids:
        a = inner_rec(k, p)["recovery"]["auc"]
        for w in ("v1", "v2", "pair"):
            if a[w] > best[w]:
                best[w], winner[w] = float(a[w]), p
    return best, winner


def candidate_rows(ids, policy_ids_by_teacher):
    U = {k: utility(inner_rec(k, "SRC|U")) for k in SEEDS}
    rows = {}
    for cid in ids:
        seeds = {}
        for k in SEEDS:
            r = inner_rec(k, cid)
            a = {w: float(r["recovery"]["auc"][w]) for w in ("v1", "v2", "pair")}
            winners = None
            if cid.startswith("SRC|"):
                a, winners = composed_auc(k, cid, a, policy_ids_by_teacher[teacher_of(cid)])
            u = utility(r)
            gm = gate_margins(u, U[k])
            cp = r.get("class_preservation_ok", True)
            seeds[k] = {"unit": R.unit_for(k, cid), "auc": a, "composed_winner": winners, "utility": u,
                        "gate_margins": gm, "gate_shortfall": shortfall(gm) + (0.0 if cp else 1.0),
                        "class_preservation_ok": cp, "token_states": r.get("token_states"),
                        "task_eligible": shortfall(gm) == 0.0 and cp}
        mean = lambda f: sum(f(s) for s in seeds.values()) / len(SEEDS)     # noqa: E731
        rows[cid] = {"config": cid, "family": family(cid), "teacher": teacher_of(cid), "m": rate_of(cid),
                     "seeds": seeds, "mean_pair": mean(lambda s: s["auc"]["pair"]),
                     "mean_v1": mean(lambda s: s["auc"]["v1"]), "mean_v2": mean(lambda s: s["auc"]["v2"]),
                     "mean_logloss": mean(lambda s: (s["utility"][0]["logloss"] + s["utility"][1]["logloss"]) / 2),
                     "token_states": sum((s["token_states"] or 0) for s in seeds.values()) if family(cid) not in
                     ("SRC", "REF") else float("inf"),
                     "task_eligible": all(s["task_eligible"] for s in seeds.values()),
                     "gate_shortfall": sum(s["gate_shortfall"] for s in seeds.values())}
    return rows


def tie(r):
    return (round(r["mean_pair"], 12), round(r["mean_logloss"], 12), r["token_states"], r["config"])


def guard_excess(r, guards):
    return {g: {k: {w: r["seeds"][k]["auc"][w] - (gr["seeds"][k]["auc"][w] + BUFFER) for w in ("v1", "v2")}
                for k in SEEDS} for g, gr in guards.items()}


def pick(rows, guards=None, ok="NOMINEE", none="NO_FEASIBLE_NOMINEE"):
    guards = guards or {}
    ev = []
    for r0 in rows:
        r = dict(r0)
        r["guard_excess"] = guard_excess(r, guards)
        r["guard_ok"] = all(x <= 0 for g in r["guard_excess"].values() for k in g.values() for x in k.values())
        r["eligible"] = r["task_eligible"] and r["guard_ok"]
        r["nomination_shortfall"] = r["gate_shortfall"] + sum(max(0.0, x) for g in r["guard_excess"].values()
                                                              for k in g.values() for x in k.values())
        ev.append(r)
    el = [r for r in ev if r["eligible"]]
    if el:
        b = min(el, key=tie)
        return {"status": ok, "config": b["config"], "row": b, "evaluated": ev}
    if not ev:
        return {"status": "ABSENT", "config": None, "row": None, "evaluated": []}
    b = min(ev, key=lambda r: (round(r["nomination_shortfall"], 12),) + tie(r))
    return {"status": none, "config": None, "descriptive_config": b["config"], "row": b, "evaluated": ev}


def select_all(D, shard_spec=None):
    ids = R.locked_bank_ids()
    pol = [c for c in ids if not c.startswith(("SRC|", "REF|"))]
    by_teacher = {t: [c for c in pol if c.startswith(t + "|")] for t in R.TEACHERS}
    rows = candidate_rows(ids, by_teacher)
    out = {"rule": "PROTOCOL.md sections 7-8 (frozen in SELECTION_AND_AUDIT_LOCK)", "candidates": ids, "rows": rows,
           "U_valid": all(rows["SRC|U"]["seeds"][k]["utility"][i]["acc"] - rows["SRC|U"]["seeds"][k]["utility"][i]["const_acc"]
                          >= 0.03 for k in SEEDS for i in (0, 1))}
    nonjoint = [rows[c] for c in pol if family(c) in NONJOINT]
    C_match = {}
    for t in R.TEACHERS:
        for m in R.RATES:
            C_match[f"{t}|m{m}"] = pick([r for r in nonjoint if r["teacher"] == t and r["m"] == m],
                                        none="NO_FEASIBLE_CONTROL")
    glob = nonjoint + [rows[c] for c in ids if family(c) in ("CLASS", "SRC", "REF")]
    C_global = pick(glob, none="NO_FEASIBLE_CONTROL")
    T_star = pick([rows[c] for c in ids if family(c) in UNTRAINED + ("SRC",) or c == "REF|F0"],   # review R1
                  none="NO_FEASIBLE_CONTROL")
    # J*: per-config guards against its own C_match and C_global
    jrows = []
    for c in pol:
        if family(c) != "JOINT":
            continue
        r = rows[c]
        cm = C_match[f"{r['teacher']}|m{r['m']}"]
        g = {}
        missing = []
        for name, P in (("C_match", cm), ("C_global", C_global)):
            if P["status"] == "NOMINEE":
                g[name] = rows[P["config"]]
            else:
                missing.append(name)
        jrows.append((r, g, missing))
    evaluated, el = [], []
    for r, g, missing in jrows:
        e = pick([r], guards=g)["evaluated"][0]
        e["missing_guards"] = missing
        e["eligible"] = e["eligible"] and not missing
        evaluated.append(e)
        if e["eligible"]:
            el.append(e)
    if el:
        b = min(el, key=tie)
        J = {"status": "NOMINEE", "config": b["config"], "row": b, "evaluated": evaluated}
    elif evaluated:
        b = min(evaluated, key=lambda r: (round(r["nomination_shortfall"], 12),) + tie(r))
        blocked = any(e["task_eligible"] and e["guard_ok"] and e["missing_guards"] for e in evaluated)
        J = {"status": "INVALID_MISSING_COMPARATOR" if blocked else "NO_FEASIBLE_NOMINEE", "config": None,
             "descriptive_config": b["config"], "row": b, "evaluated": evaluated}
    else:
        J = {"status": "ABSENT", "config": None, "row": None, "evaluated": []}
    jm = J.get("config") or J.get("descriptive_config")
    J["C_match_key"] = f"{rows[jm]['teacher']}|m{rows[jm]['m']}" if jm else None
    if T_star["status"] == "NOMINEE":
        P_star = pick([rows[c] for c in pol if family(c) in PRIVACY], guards={"T*": rows[T_star["config"]]})
    else:
        P_star = pick([rows[c] for c in pol if family(c) in PRIVACY])
        if P_star["status"] == "NOMINEE":
            P_star.update(status="INVALID_MISSING_COMPARATOR", descriptive_config=P_star["config"], config=None)
    cm_for_j = C_match.get(J["C_match_key"]) if J["C_match_key"] else None
    out["statuses"] = {"J*": J, "P*": P_star, "T*": T_star, "C_global": C_global,
                       "C_match": cm_for_j or {"status": "ABSENT", "config": None}}
    out["C_match_all"] = {key: {kk: v.get(kk) for kk in ("status", "config", "descriptive_config")}
                          for key, v in C_match.items()}
    compact = [r for r in (rows[c] for c in pol) if r["task_eligible"]]
    best = min(compact, key=tie) if compact else None
    out["deployable_compact"] = ({"config": best["config"], "family": best["family"], "mean_inner_pair_auc": best["mean_pair"]}
                                 if best else {"config": None, "note": "no finite-code policy met the score contract; "
                                               "the truthful baseline is U's continuous output"})
    out["scored_labels"] = scored_labels(out, rows, pol)
    (R.RUN / "selection.json").write_text(json.dumps(out, indent=1, default=float))
    public(out)
    R.event("selection written", statuses={x: (s["status"], s.get("config") or s.get("descriptive_config"))
                                           for x, s in out["statuses"].items()})
    return out


def scored_labels(out, rows, pol):
    """The locked descriptive assessment set (fixed before the assessment; PROTOCOL section 10)."""
    lab = set(["SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"])
    for t in R.TEACHERS:
        lab.add(f"{t}|CLASS|m1")
        for m in R.RATES:
            for f in ("FINE-TASK", "DIRECT-TASK"):
                lab.add(f"{t}|{f}|m{m}")
    for x, s in out["statuses"].items():
        c = s.get("config") or s.get("descriptive_config")
        if c:
            lab.add(c)
    for f in PRIVACY:                                   # best eligible and nearest ineligible per privacy family
        fr = [rows[c] for c in pol if family(c) == f]
        el = [r for r in fr if r["task_eligible"]]
        inel = [r for r in fr if not r["task_eligible"]]
        if el:
            lab.add(min(el, key=tie)["config"])
        if inel:
            lab.add(min(inel, key=lambda r: (round(r["gate_shortfall"], 12),) + tie(r))["config"])
    J = out["statuses"]["J*"]
    j = J.get("config") or J.get("descriptive_config")
    if j:                                               # matched joint vs local vs sequential vs task-only
        t, _, m, l = j.split("|")
        for f in ("LOCAL", "SEQ-12", "SEQ-21", "JOINT"):
            lab.add(f"{t}|{f}|{m}|{l}")
        for f in ("FINE-TASK", "DIRECT-TASK"):
            lab.add(f"{t}|{f}|{m}")
    if out["deployable_compact"].get("config"):
        lab.add(out["deployable_compact"]["config"])
    return sorted(lab)


def public(out):
    st = {}
    for x, s in out["statuses"].items():
        r = s.get("row") or {}
        st[x] = {"status": s["status"], "config": s.get("config"), "descriptive_config": s.get("descriptive_config"),
                 "mean_inner_pair_auc": r.get("mean_pair"), "mean_inner_v1_auc": r.get("mean_v1"),
                 "mean_inner_v2_auc": r.get("mean_v2"), "mean_inner_logloss": r.get("mean_logloss"),
                 "nomination_shortfall": r.get("nomination_shortfall"), "missing_guards": r.get("missing_guards")}
    pub = {"schema": "dpc-selection-v1", "U_valid": out["U_valid"], "statuses": st, "C_match_all": out["C_match_all"],
           "deployable_compact": out["deployable_compact"], "scored_labels": out["scored_labels"],
           "n_candidates": len(out["candidates"])}
    (R.PKG / "SELECTION.json").write_text(json.dumps(pub, indent=1, default=float) + "\n")
    roles = {}
    for x, s in out["statuses"].items():
        c = s.get("config") or s.get("descriptive_config")
        if c:
            roles.setdefault(c, []).append(f"{x}:{s['status']}")
    rows = []
    for cid, r in out["rows"].items():
        for k, s in r["seeds"].items():
            rows.append({"config": cid, "family": r["family"], "seed": k, "auc_v1": s["auc"]["v1"], "auc_v2": s["auc"]["v2"],
                         "auc_pair": s["auc"]["pair"], "composed_pair_winner": (s["composed_winner"] or {}).get("pair", ""),
                         "acc_income": s["utility"][0]["acc"], "acc_occ": s["utility"][1]["acc"],
                         "logloss_income": s["utility"][0]["logloss"], "logloss_occ": s["utility"][1]["logloss"],
                         "brier_income": s["utility"][0]["brier"], "brier_occ": s["utility"][1]["brier"],
                         "seed_task_eligible": s["task_eligible"], "gate_shortfall": s["gate_shortfall"],
                         "token_states": s["token_states"], "roles": ";".join(roles.get(cid, []))})
    with open(R.PKG / "INNER_SELECTION_TABLE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for q in rows:
            w.writerow({kk: (f"{v:.6f}" if isinstance(v, float) else v) for kk, v in q.items()})
