"""Registered inner-only selection rules (PROTOCOL.md sections 4 and 6). Inner roles: AUDIT_FIT -> INNER_SELECTION.

Task gates (both tasks, deployed heads, INNER_SELECTION point estimates; const = NEW_DEFENSE_FIT majority class):
  G1 Acc >= Acc(ref) - 0.01;  G2 (Acc - const) >= 0.8 (Acc(ref) - const);  G3 Acc - const >= 0.03
  ref = U epoch 20 in Phase A; U epoch 40 in Phase B.  Gate shortfall = sum over tasks and gates of max(0, -margin).
Recovery = inner AUC of the AUC-selected inner attacker (smf.audit.inner_audit; fixed orientation).

PHASE A (final-epoch normalized points only):
  1. per schedule and seed: among task-feasible JOINT points, lowest coalition AUC; ties lower rho.
  2. among schedules with a point on every seed: minimise mean selected coalition AUC; ties lower actual compute
     (critic optimizer updates of the selected units), then ONLINE, ONLINE_MATCHED, REFRESHED.
  3. fallback (no schedule has points on every seed): minimise the sum over seeds of the smallest joint-point gate
     shortfall (ties lower rho per point); schedule ties by compute then the order above; status FALLBACK.
  Local reference per seed (selected schedule): among task-feasible LOCAL points, lowest worse-local AUC, then mean
  local AUC, then lower rho; U epoch 20 is a disclosed fallback candidate (TASK_ONLY_ALIAS); U lacking nontrivial
  utility (G3 against itself fails) -> NO_VALID_REFERENCE.
PHASE B (final-epoch points of J-F, L-F, J-N, L-N at rho grid; raw RAW-J/RAW-L at epoch 40, beta grid):
  controls L-F, J-N, L-N, RAW-J, RAW-L: task-feasible (vs U e40) points; lowest coalition AUC; ties lower rho/beta.
  single-configuration controls U (e40), E (official LEACE on U e40), F (official FARE, per-purpose rule), F0.
  C* = lowest coalition AUC among task-feasible controls {L-F, J-N, L-N, RAW-J, RAW-L, E, F, F0, U} (no local
  guard); ties by that order. J-F nominee: task-feasible AND AUC_i(J-F) <= AUC_i(L-F) + 0.005 (if L-F is a nominee)
  AND AUC_i(J-F) <= AUC_i(C*) + 0.005 (if C* exists), i = 1, 2; lowest coalition AUC, then lower rho.
  No eligible J-F -> NO_FEASIBLE_NOMINEE; descriptive fallback minimises the summed positive shortfalls of all
  nomination gates (task gates in accuracy units + local buffers in AUC units), then coalition AUC, then lower rho.
No assessment value is consulted anywhere.
"""
from __future__ import annotations

import csv
import json

from smf import run as R

ORDER_SCHED = ["ONLINE", "ONLINE_MATCHED", "REFRESHED"]
CONTROL_ORDER = ["L-F", "J-N", "L-N", "RAW-J", "RAW-L", "E", "F", "F0", "U"]
BUFFER = 0.005


def inner(name):
    return R.rec(f"inner__{name}")


def util(name):
    return {int(i): v for i, v in inner(name)["utility"].items()}


def auc(name):
    a = inner(name)["recovery"]["auc"]
    return {"v1": a["v1"], "v2": a["v2"], "pair": a["pair"]}


def gate_margins(u, uref):
    out = {}
    for i in (0, 1):
        a, ar, c = u[i]["acc"], uref[i]["acc"], u[i]["const_acc"]
        out[i] = {"G1": a - (ar - 0.01), "G2": (a - c) - 0.8 * (ar - c), "G3": (a - c) - 0.03}
    return out


def shortfall(gm):
    return sum(max(0.0, -x) for i in gm for x in gm[i].values())


def candidate(name, uref, rho, extra=None):
    u, a = util(name), auc(name)
    gm = gate_margins(u, uref)
    c = {"unit": name, "rho": rho, "auc": a, "utility": u, "gate_margins": gm, "gate_shortfall": shortfall(gm),
         "task_feasible": shortfall(gm) == 0.0, "worse_local": max(a["v1"], a["v2"]),
         "mean_local": (a["v1"] + a["v2"]) / 2}
    c.update(extra or {})
    return c


def compute_of(run_name):
    d = R.rec(run_name)["diag"]
    return d["critic_online_updates"] + d.get("critic_matched_extra_updates", 0) + d.get("critic_refit_updates", 0)


# ------------------------------------------------------------------ Phase A
def select_A(D):
    out = {"rule": "PROTOCOL.md section 4 (frozen in PHASE_A_PROTOCOL_LOCK)", "per_schedule": {}, "seeds": {}}
    uref = {k: util(f"tl__s{k}__e20") for k in R.SEEDS}
    best = {}
    for s in R.SCHEDS:
        best[s] = {}
        for k in R.SEEDS:
            C = [candidate(R.a_name(k, "NJ", s, r), uref[k], r, {"compute": compute_of(R.a_run(k, "NJ", s, r))})
                 for r in R.RHOS]
            feas = [c for c in C if c["task_feasible"]]
            pick = min(feas, key=lambda c: (round(c["auc"]["pair"], 12), c["rho"])) if feas else None
            closest = min(C, key=lambda c: (c["gate_shortfall"], c["rho"]))
            best[s][k] = {"selected": pick, "closest": closest, "table": C}
        out["per_schedule"][s] = best[s]
    complete = [s for s in R.SCHEDS if all(best[s][k]["selected"] for k in R.SEEDS)]
    if complete:
        def key(s):
            sel = [best[s][k]["selected"] for k in R.SEEDS]
            return (round(sum(c["auc"]["pair"] for c in sel) / 3, 12), sum(c["compute"] for c in sel), ORDER_SCHED.index(s))
        chosen, status = min(complete, key=key), "SELECTED"
    else:
        def key(s):
            cl = [best[s][k]["closest"] for k in R.SEEDS]
            return (round(sum(c["gate_shortfall"] for c in cl), 12), sum(c["compute"] for c in cl), ORDER_SCHED.index(s))
        chosen, status = min(R.SCHEDS, key=key), "FALLBACK"
    out["schedule"] = {"selected": chosen, "status": status, "complete_schedules": complete,
                       "mean_selected_pair_auc": {s: (sum(best[s][k]["selected"]["auc"]["pair"] for k in R.SEEDS) / 3
                                                      if s in complete else None) for s in R.SCHEDS}}
    for k in R.SEEDS:
        ub = f"tl__s{k}__e20"
        u_c = candidate(ub, uref[k], 0.0)
        L = [candidate(R.a_name(k, "NL", chosen, r), uref[k], r) for r in R.RHOS]
        pool = [c for c in L if c["task_feasible"]] + ([dict(u_c, alias=True)] if u_c["task_feasible"] else [])
        if not u_c["task_feasible"]:
            ref = {"status": "NO_VALID_REFERENCE", "note": "U epoch 20 lacks nontrivial task utility"}
        else:
            b = min(pool, key=lambda c: (round(c["worse_local"], 12), round(c["mean_local"], 12), c["rho"]))
            ref = {"status": "TASK_ONLY_ALIAS" if b.get("alias") else "NOMINEE", **b}
        out["seeds"][str(k)] = {"reference": ref, "local_table": L, "U_e20": u_c}
    (R.RUN / "selection_A.json").write_text(json.dumps(out, indent=1, default=float))
    R.PKG.mkdir(parents=True, exist_ok=True)
    pub = {"schema": "smf-phase-a-freeze-v1", "schedule": out["schedule"],
           "seeds": {s: {"reference": {kk: v["reference"].get(kk) for kk in ("status", "unit", "rho", "auc", "worse_local",
                                                                            "mean_local", "gate_shortfall")}}
                     for s, v in out["seeds"].items()},
           "per_schedule_selected": {s: {str(k): (best[s][k]["selected"] or {}).get("unit") for k in R.SEEDS}
                                     for s in R.SCHEDS}}
    (R.PKG / "PHASE_A_SELECTION.json").write_text(json.dumps(pub, indent=1, default=float) + "\n")
    write_frontiers(best, out)
    R.event("selection A written", schedule=chosen, status=status)
    return out


def write_frontiers(best, out):
    rows = []
    for s, per in best.items():
        for k, v in per.items():
            for c in v["table"]:
                rows.append({"schedule": s, "seed": k, "rho": c["rho"], "unit": c["unit"], "auc_v1": c["auc"]["v1"],
                             "auc_v2": c["auc"]["v2"], "auc_pair": c["auc"]["pair"], "acc_income": c["utility"][0]["acc"],
                             "acc_occ": c["utility"][1]["acc"], "task_feasible": c["task_feasible"],
                             "gate_shortfall": c["gate_shortfall"], "compute_critic_updates": c["compute"],
                             "selected_for_schedule": v["selected"] is not None and v["selected"]["unit"] == c["unit"]})
    for k, v in out["seeds"].items():
        for c in v["local_table"]:
            rows.append({"schedule": out["schedule"]["selected"] + " (local)", "seed": int(k), "rho": c["rho"],
                         "unit": c["unit"], "auc_v1": c["auc"]["v1"], "auc_v2": c["auc"]["v2"], "auc_pair": c["auc"]["pair"],
                         "acc_income": c["utility"][0]["acc"], "acc_occ": c["utility"][1]["acc"],
                         "task_feasible": c["task_feasible"], "gate_shortfall": c["gate_shortfall"],
                         "compute_critic_updates": "", "selected_for_schedule": v["reference"].get("unit") == c["unit"]})
    with open(R.PKG / "PHASE_A_FRONTIERS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({kk: (f"{v:.6f}" if isinstance(v, float) else v) for kk, v in r.items()})


# ------------------------------------------------------------------ Phase B
def run_baselines(D, shard_spec=None):
    from smf import baselines as BL
    for k in R.shard(list(R.SEEDS), shard_spec):
        BL.leace_unit(k, f"tl__s{k}__e40", D, R.UNITS)
        BL.fare_units(k, D, R.UNITS)


def pick_control(C):
    feas = [c for c in C if c["task_feasible"]]
    if feas:
        return {"status": "NOMINEE", **min(feas, key=lambda c: (round(c["auc"]["pair"], 12), c["rho"]))}
    return {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE",
            **min(C, key=lambda c: (c["gate_shortfall"], c["rho"]))}


def nomination_shortfall(c, guards):
    s = c["gate_shortfall"]
    for g in guards.values():
        for w in ("v1", "v2"):
            s += max(0.0, c["auc"][w] - (g[w] + BUFFER))
    return s


def select_B(D):
    from smf import baselines as BL
    SA = json.loads((R.RUN / "selection_A.json").read_text())
    out = {}
    for k in R.SEEDS:
        s = str(k)
        res = {"seed": k, "arms": {}}
        ref = SA["seeds"][s]["reference"]
        res["valid_reference"] = ref["status"] != "NO_VALID_REFERENCE"
        res["reference"] = {kk: ref.get(kk) for kk in ("status", "unit", "rho")}
        un = f"tl__s{k}__e40"
        uref = util(un)
        res["U"] = {"unit": un, "utility": uref}
        if not res["valid_reference"]:
            out[s] = res
            continue
        for arm in ("L-F", "J-N", "L-N"):
            C = [candidate(R.b_name(k, arm, r), uref, r) for r in R.RHOS]
            res["arms"][arm] = {**pick_control(C), "table": C}
        for arm in ("RAW-J", "RAW-L"):
            C = [candidate(f"raw__s{k}__{arm}__b{R.g(b)}__e40", uref, b) for b in R.RAW_BETAS]
            res["arms"][arm] = {**pick_control(C), "table": C}
        for arm, name in (("U", un), ("E", f"lc__s{k}__E")):
            c = candidate(name, uref, 0.0)
            res["arms"][arm] = {"status": "NOMINEE" if c["task_feasible"] else "INFEASIBLE_CONTROL", **c}
        fare = BL.select_fare(k, D, R.UNITS, uref)
        for arm in ("F", "F0"):
            f = fare[arm]
            f.setdefault("task_feasible", f.get("gates_ok", False))
            res["arms"][arm] = f
        res["fare_selection"] = fare.get("per_purpose")
        cands = sorted((round(res["arms"][a]["auc"]["pair"], 12), j, a) for j, a in enumerate(CONTROL_ORDER)
                       if res["arms"][a].get("status") == "NOMINEE" and res["arms"][a].get("task_feasible", False))
        cs = cands[0][2] if cands else None
        res["comparator"] = {"arm": cs, "unit": (res["arms"][cs].get("unit") or res["arms"][cs].get("units")) if cs else None,
                             "status": res["arms"][cs]["status"] if cs else "ABSENT",
                             "is_task_only_release": cs == "U", "candidates": [c[2] for c in cands],
                             "rule": "lowest inner coalition AUC among task-feasible controls (no local guard); ties "
                                     + ",".join(CONTROL_ORDER)}
        guards = {}
        if res["arms"]["L-F"]["status"] == "NOMINEE":
            guards["L-F"] = res["arms"]["L-F"]["auc"]
        if cs is not None:
            guards["C*"] = res["arms"][cs]["auc"]
        C = []
        for r in R.RHOS:
            c = candidate(R.b_name(k, "J-F", r), uref, r)
            c["guard_excess"] = {g: {w: c["auc"][w] - (v[w] + BUFFER) for w in ("v1", "v2")} for g, v in guards.items()}
            c["guard_ok"] = all(x <= 0 for gg in c["guard_excess"].values() for x in gg.values())
            c["eligible"] = c["task_feasible"] and c["guard_ok"]
            c["nomination_shortfall"] = nomination_shortfall(c, guards)
            C.append(c)
        el = [c for c in C if c["eligible"]]
        if el:
            res["arms"]["J-F"] = {"status": "NOMINEE", **min(el, key=lambda c: (round(c["auc"]["pair"], 12), c["rho"])),
                                  "table": C}
        else:
            res["arms"]["J-F"] = {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE", "table": C,
                                  **min(C, key=lambda c: (round(c["nomination_shortfall"], 12), round(c["auc"]["pair"], 12),
                                                          c["rho"]))}
        res["arms"]["J-F"]["nomination_guards"] = guards
        out[s] = res
    (R.RUN / "selection_B.json").write_text(json.dumps(out, indent=1, default=float))
    status = {s: {"valid_reference": r.get("valid_reference"), "reference": r.get("reference"), "U": r["U"]["unit"],
                  "comparator": r.get("comparator"),
                  "arms": {a: {kk: v.get(kk) for kk in ("status", "unit", "units", "rho", "auc", "task_feasible",
                                                        "gate_shortfall", "guard_excess", "nomination_shortfall",
                                                        "nomination_guards")}
                           for a, v in r.get("arms", {}).items()}} for s, r in out.items()}
    (R.PKG / "SEED_STATUS.json").write_text(json.dumps(status, indent=1, default=float) + "\n")
    write_selection_table(out)
    R.event("selection B written")
    return out


def write_selection_table(out):
    rows = []
    for s, res in out.items():
        for arm, a in res.get("arms", {}).items():
            for c in a.get("table", [a]):
                if "auc" not in c:
                    continue
                chosen = a.get("unit") == c.get("unit")
                rows.append({"seed": s, "arm": arm, "unit": c.get("unit") or str(c.get("units")), "rho_or_beta": c.get("rho"),
                             "auc_v1": c["auc"]["v1"], "auc_v2": c["auc"]["v2"], "auc_pair": c["auc"]["pair"],
                             "acc_income": c["utility"][0]["acc"] if "utility" in c else "",
                             "acc_occ": c["utility"][1]["acc"] if "utility" in c else "",
                             "task_feasible": c.get("task_feasible"), "gate_shortfall": c.get("gate_shortfall"),
                             "guard_ok": c.get("guard_ok", ""), "arm_status": a.get("status"),
                             "outcome": ("selected" if chosen and a.get("status") == "NOMINEE" else
                                         "descriptive (infeasible)" if chosen else
                                         "task gates failed" if not c.get("task_feasible") else
                                         "local buffer failed" if c.get("guard_ok") is False else "feasible, not best")})
    cols = list(rows[0]) if rows else []
    with open(R.PKG / "SELECTION_TABLE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({kk: (f"{v:.6f}" if isinstance(v, float) else v) for kk, v in r.items()})
