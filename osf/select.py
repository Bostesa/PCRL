"""Registered inner-only selection (PROTOCOL.md section 8). Inputs: inner__<unit> records (AUDIT_FIT -> INNER_SELECTION)
and the audit/baseline owner's reference candidates (E, F, F0). No assessment value is consulted anywhere.

Task gates (point estimates on INNER_SELECTION, deployed heads, both tasks, EVERY seed; const = OSF_DEFENSE_FIT majority):
  G1 Acc >= Acc(U) - 0.01;  G2 Acc - const >= 0.8 (Acc(U) - const);  G3 Acc - const >= 0.03.
  U must itself have nontrivial utility (G3 on every seed and task), else every status is INVALID.
A configuration is one global setting across seeds; it is task-feasible iff it passes every gate on every seed.
Recovery = inner AUC of the AUC-selected inner attacker (fixed orientation); coalition = pair view.
  L*  lowest mean coalition AUC among task-feasible RAW-L and NORM-L configurations; ties lower measured compute (critic
      + encoder optimizer updates over the three seeds), then lexicographic config id.
  C*  lowest mean coalition AUC among task-feasible controls: every RAW-J, RAW-L and NORM-L configuration, U, E, F, F0
      (no local guard); ties lexicographic id.
  N*  among NORM-J configurations that are task-feasible and, on every seed and for both recipients, have
      AUC_i <= AUC_i(L*) + 0.005 and AUC_i <= AUC_i(C*) + 0.005: lowest mean coalition AUC; ties lower rho, |a - 1|
      smaller, then id.
  R*  among RAW-J configurations that are task-feasible and have AUC_i <= AUC_i(L*) + 0.005 on every seed for both
      recipients: lowest mean coalition AUC; ties lower beta, then id.
  No eligible configuration -> NO_FEASIBLE_NOMINEE; the descriptive fallback minimises the summed positive nomination
  shortfalls over seeds (gate shortfalls in accuracy units + guard excess in AUC units), then mean coalition AUC, then
  the tie rules. A missing guard comparator is recorded (affected claims are invalid).
  Deployable best development model: lowest mean coalition AUC among the valid N*, R*, C* (ties id); no control demoted.
"""
from __future__ import annotations

import csv
import json

from osf import run as R
from osf import train as T

BUFFER = 0.005
SEEDS = R.SEEDS


def inner(name):
    return R.rec(f"inner__{name}")


def util(name):
    return {int(i): v for i, v in inner(name)["utility"].items()}


def auc(name):
    a = inner(name)["recovery"]["auc"]
    return {"v1": float(a["v1"]), "v2": float(a["v2"]), "pair": float(a["pair"])}


def gate_margins(u, uref):
    out = {}
    for i in (0, 1):
        a, ar, c = u[i]["acc"], uref[i]["acc"], u[i]["const_acc"]
        out[i] = {"G1": a - (ar - 0.01), "G2": (a - c) - 0.8 * (ar - c), "G3": (a - c) - 0.03}
    return out


def shortfall(gm):
    return sum(max(0.0, -x) for i in gm for x in gm[i].values())


def compute_of(k, cid):
    if cid in ("U",):
        return 0
    try:
        d = R.rec(R.run_name(k, cid))["diag"]
        return int(d["critic_online_updates"] + d["encoder_updates"])
    except FileNotFoundError:
        return None


def family_of(cid):
    if cid in ("U", "E", "F", "F0"):
        return cid
    return cid.split("|")[0]


def per_seed(k, cid, uref, ref_cands):
    if cid in ("E", "F", "F0"):
        rc = ref_cands[k][cid]
        a = {w: float(rc["inner"][w]) for w in ("v1", "v2", "pair")}
        u = {int(i): v for i, v in rc["utility"].items()}
        unit = rc.get("unit") or rc.get("units")
    else:
        unit = R.rel_name(k, cid)
        a, u = auc(unit), util(unit)
    gm = gate_margins(u, uref)
    return {"seed": k, "unit": unit, "auc": a, "utility": u, "gate_margins": gm, "gate_shortfall": shortfall(gm),
            "task_feasible": shortfall(gm) == 0.0, "compute": compute_of(k, cid) if cid not in ("E", "F", "F0") else None}


def config_row(cid, uref, ref_cands):
    seeds = {k: per_seed(k, cid, uref[k], ref_cands) for k in SEEDS}
    comp = [s["compute"] for s in seeds.values()]
    return {"config": cid, "family": family_of(cid), "seeds": seeds,
            "mean_pair": sum(s["auc"]["pair"] for s in seeds.values()) / len(SEEDS),
            "mean_v1": sum(s["auc"]["v1"] for s in seeds.values()) / len(SEEDS),
            "mean_v2": sum(s["auc"]["v2"] for s in seeds.values()) / len(SEEDS),
            "task_feasible": all(s["task_feasible"] for s in seeds.values()),
            "gate_shortfall": sum(s["gate_shortfall"] for s in seeds.values()),
            "compute": sum(comp) if all(c is not None for c in comp) else None}


def guard_excess(row, guards):
    """Per guard, seed and recipient: AUC_i(config) - (AUC_i(guard) + BUFFER) (> 0 violates)."""
    return {g: {k: {w: row["seeds"][k]["auc"][w] - (gr["seeds"][k]["auc"][w] + BUFFER) for w in ("v1", "v2")}
                for k in SEEDS} for g, gr in guards.items()}


def nomination_shortfall(row, ge):
    return row["gate_shortfall"] + sum(max(0.0, x) for g in ge.values() for k in g.values() for x in k.values())


def pick(rows, guards=None, status_ok="NOMINEE", tie=None):
    """rows are evaluated on private copies (guard fields never leak between roles)."""
    guards = guards or {}
    rows = [dict(r) for r in rows]
    for r in rows:
        r["guard_excess"] = guard_excess(r, guards)
        r["guard_ok"] = all(x <= 0 for g in r["guard_excess"].values() for k in g.values() for x in k.values())
        r["eligible"] = r["task_feasible"] and r["guard_ok"]
        r["nomination_shortfall"] = nomination_shortfall(r, r["guard_excess"])
    el = [r for r in rows if r["eligible"]]
    if el:
        b = min(el, key=lambda r: (round(r["mean_pair"], 12),) + tie(r))
        return {"status": status_ok, "config": b["config"], "row": b, "evaluated": rows}
    if not rows:
        return {"status": "ABSENT", "config": None, "row": None, "evaluated": []}
    b = min(rows, key=lambda r: (round(r["nomination_shortfall"], 12), round(r["mean_pair"], 12)) + tie(r))
    return {"status": "NO_FEASIBLE_NOMINEE", "config": None, "descriptive_config": b["config"], "row": b,
            "evaluated": rows}


def lock_bank():
    return R.locked_bank()[1]


def select_all(D, shard_spec=None):
    from osf import baselines as BL
    ids = lock_bank()
    uref = {k: util(R.rel_name(k, "U")) for k in SEEDS}
    u_valid = all(uref[k][i]["acc"] - uref[k][i]["const_acc"] >= 0.03 for k in SEEDS for i in (0, 1))
    ref_cands = {k: BL.reference_candidates(k) for k in SEEDS}
    rows = {cid: config_row(cid, uref, ref_cands) for cid in ids + ["E", "F", "F0"]}
    for c in ("F", "F0"):                       # the reference path's own feasibility status is binding
        for k in SEEDS:
            st = ref_cands[k][c].get("status")
            if st not in (None, "NOMINEE"):
                rows[c]["seeds"][k]["task_feasible"] = False
                rows[c]["seeds"][k]["reference_status"] = st
        rows[c]["task_feasible"] = all(s["task_feasible"] for s in rows[c]["seeds"].values())
    out = {"rule": "PROTOCOL.md section 8 (frozen in SELECTION_AND_AUDIT_LOCK)", "U_valid": u_valid, "bank": ids,
           "rows": rows, "statuses": {}}
    if not u_valid:
        out["statuses"] = {x: {"status": "INVALID", "config": None, "reason": "U lacks nontrivial utility"}
                           for x in ("L*", "C*", "N*", "R*")}
    else:
        local = [rows[c] for c in ids if family_of(c) in ("RAW-L", "NORM-L")]
        L = pick(local, tie=lambda r: ((r["compute"] if r["compute"] is not None else 1e18), r["config"]),
                 status_ok="NOMINEE")
        ctrl = [rows[c] for c in ids if family_of(c) in ("RAW-J", "RAW-L", "NORM-L", "U")] + [rows[c] for c in ("E", "F", "F0")]
        C = pick(ctrl, tie=lambda r: (r["config"],))
        guards = {}
        if L["status"] == "NOMINEE":
            guards["L*"] = rows[L["config"]]
        gN = dict(guards)
        if C["status"] == "NOMINEE":
            gN["C*"] = rows[C["config"]]
        normj = [rows[c] for c in ids if family_of(c) == "NORM-J"]
        N = pick(normj, guards=gN,
                 tie=lambda r: (T.parse_id(r["config"])["rho"], abs(T.parse_id(r["config"])["a"] - 1.0), r["config"]))
        rawj = [rows[c] for c in ids if family_of(c) == "RAW-J"]
        Rs = pick(rawj, guards=guards, tie=lambda r: (T.parse_id(r["config"])["beta"], r["config"]))
        N["guards_used"], Rs["guards_used"] = sorted(gN), sorted(guards)
        N["missing_guards"] = [g for g in ("L*", "C*") if g not in gN]
        Rs["missing_guards"] = [g for g in ("L*",) if g not in guards]
        for P in (N, Rs):                    # review S1: a missing guard comparator invalidates the nomination
            if P["missing_guards"] and P["status"] == "NOMINEE":
                P["status"], P["descriptive_config"], P["config"] = "INVALID_MISSING_COMPARATOR", P["config"], None
        out["statuses"] = {"L*": L, "C*": C, "N*": N, "R*": Rs}
        out["nomination"] = {r["config"]: {kk: r[kk] for kk in ("guard_ok", "guard_excess", "eligible",
                                                                "nomination_shortfall")}
                             for P in (N, Rs) for r in P["evaluated"]}
    valid = [(rows[s["config"]]["mean_pair"], s["config"], x) for x, s in out["statuses"].items()
             if x in ("N*", "R*", "C*") and s["status"] == "NOMINEE"]
    best = min(valid) if valid else None
    out["deployable_best"] = ({"config": best[1], "as": best[2], "family": family_of(best[1]),
                               "kind": ("normalized joint training" if family_of(best[1]) == "NORM-J" else
                                        "ordinary raw joint training" if family_of(best[1]) == "RAW-J" else
                                        "official reference" if best[1] in ("E", "F", "F0") else
                                        "task-only" if best[1] == "U" else "local protection (control)")}
                              if best else {"config": "U", "as": "truthful baseline", "family": "U",
                                            "kind": "no protected model met the requirements"})
    (R.RUN / "selection.json").write_text(json.dumps(out, indent=1, default=float))
    public(out)
    R.event("selection written", statuses={x: (s["status"], s.get("config") or s.get("descriptive_config"))
                                           for x, s in out["statuses"].items()})
    return out


def public(out):
    st = {}
    for x, s in out["statuses"].items():
        r = s.get("row") or {}
        st[x] = {"status": s["status"], "config": s.get("config"), "descriptive_config": s.get("descriptive_config"),
                 "mean_inner_pair_auc": r.get("mean_pair"), "mean_inner_v1_auc": r.get("mean_v1"),
                 "mean_inner_v2_auc": r.get("mean_v2"), "guards_used": s.get("guards_used"),
                 "missing_guards": s.get("missing_guards"), "nomination_shortfall": r.get("nomination_shortfall"),
                 "per_seed_units": {str(k): v["unit"] for k, v in r.get("seeds", {}).items()}}
    pub = {"schema": "osf-selection-v1", "U_valid": out["U_valid"], "statuses": st,
           "deployable_best": out["deployable_best"], "bank": out["bank"]}
    (R.PKG / "SELECTION.json").write_text(json.dumps(pub, indent=1, default=float) + "\n")
    chosen = {s.get("config") or s.get("descriptive_config"): x for x, s in out["statuses"].items()}
    fr, sel = [], []
    for cid, r in out["rows"].items():
        for k, s in r["seeds"].items():
            row = {"config": cid, "family": r["family"], "seed": k, "auc_v1": s["auc"]["v1"], "auc_v2": s["auc"]["v2"],
                   "auc_pair": s["auc"]["pair"], "acc_income": s["utility"][0]["acc"], "acc_occ": s["utility"][1]["acc"],
                   "const_income": s["utility"][0]["const_acc"], "const_occ": s["utility"][1]["const_acc"],
                   "seed_task_feasible": s["task_feasible"], "gate_shortfall": s["gate_shortfall"]}
            fr.append(row)
        sel.append({"config": cid, "family": r["family"], "mean_auc_pair": r["mean_pair"], "mean_auc_v1": r["mean_v1"],
                    "mean_auc_v2": r["mean_v2"], "task_feasible_all_seeds": r["task_feasible"],
                    "gate_shortfall_sum": r["gate_shortfall"],
                    "nomination_guard_ok": out.get("nomination", {}).get(cid, {}).get("guard_ok", ""),
                    "nomination_shortfall": out.get("nomination", {}).get(cid, {}).get("nomination_shortfall", ""),
                    "compute": r["compute"],
                    "role": chosen.get(cid, "")})
    for fn, rows in (("INNER_FRONTIERS.csv", fr), ("SELECTION_TABLE.csv", sel)):
        with open(R.PKG / fn, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            for q in rows:
                w.writerow({kk: (f"{v:.6f}" if isinstance(v, float) else v) for kk, v in q.items()})
