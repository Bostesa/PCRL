"""Registered inner-only selection (PROTOCOL.md section 6). Inner roles: AUDIT_FIT -> INNER_SELECTION.

Utility gates (both tasks, deployed heads, INNER_SELECTION point estimates; const = DEFENSE_FIT majority class):
  G1 Acc >= Acc(ref) - 0.01;  G2 (Acc - const) >= 0.8 (Acc(ref) - const);  G3 Acc - const >= 0.03
  ref = U-B (task line, epoch 20) in stage B; ref = U (task line, epoch 20 + e_LR) in stage C.
Recovery = inner AUC of the AUC-selected inner attacker (rgj.audit.inner_audit; fixed orientation).

Stage B (per seed): L-R among gate-passing candidates {beta x checkpoint} plus the U-B alias (beta 0): minimise
  worse-local AUC, then mean local AUC, then smaller beta, then earlier checkpoint. The alias wins -> TASK_ONLY_ALIAS.
  U-B itself fails G3 -> NO_VALID_REFERENCE. L-O by the same rule (its own control comparison). Frozen before stage C.
Stage C (per seed), in this order:
  1. controls L-G, J-R, J-O: gate-passing (vs U) candidates with local AUC_i <= L-R's AUC_i (i = 1, 2; zero buffer);
     minimise coalition AUC, then smaller beta, then earlier checkpoint.
  2. single-configuration controls U, E (official LEACE on U), F (official FARE, per-purpose published rule), F0
     (zero-fairness twin): feasible iff gates (vs U) and the same local guard versus L-R. L-R and L-O (frozen stage-B
     selections) are re-checked against U's gates and the local guard.
  3. C* = lowest inner coalition AUC among feasible controls {L-G, L-R, L-O, J-R, J-O, U, E, F, F0}; ties by that order.
     A feasible task-only control (U, or an L-R / L-O TASK_ONLY_ALIAS) is eligible and flagged (review R2).
  4. J-G nominee: gate-passing candidates with AUC_i(J-G) <= min(L-R_i, L-G_i if nominee, C*_i if present) for i = 1, 2;
     minimise coalition AUC, then smaller beta, then earlier checkpoint.
No feasible candidate -> NO_FEASIBLE_NOMINEE; the closest candidate (smallest worst violation over gates and guards;
ties -> smaller beta, earlier checkpoint) is scored descriptively and can never pass a claim. J-G assessment values
are never consulted for any selection (none exist before EVALUATION_LOCK.json).
"""
from __future__ import annotations

import json

from rgj import run as R
from rgj import train as T

CKPTS = T.HP["checkpoints"]
CONTROL_ORDER = ["L-G", "L-R", "L-O", "J-R", "J-O", "U", "E", "F", "F0"]


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


def gates_ok(gm):
    return all(x >= 0 for i in gm for x in gm[i].values())


def worst_gate(gm):
    return min(x for i in gm for x in gm[i].values())


def candidate(name, uref, beta, epoch, guard=None):
    u, a = util(name), auc(name)
    gm = gate_margins(u, uref)
    c = {"unit": name, "beta": beta, "epoch": epoch, "auc": a, "utility": u, "gate_margins": gm,
         "gates_ok": gates_ok(gm), "worst_gate_margin": worst_gate(gm),
         "worse_local": max(a["v1"], a["v2"]), "mean_local": (a["v1"] + a["v2"]) / 2}
    if guard is not None:
        exc = {w: a[w] - guard[w] for w in ("v1", "v2")}
        c["guard_ref"], c["guard_excess"] = guard, exc
        c["guard_ok"] = all(x <= 0 for x in exc.values())
        c["feasible"] = c["gates_ok"] and c["guard_ok"]
        c["worst_violation"] = max([0.0, -c["worst_gate_margin"]] + [max(0.0, x) for x in exc.values()])
    else:
        c["feasible"] = c["gates_ok"]
        c["worst_violation"] = max(0.0, -c["worst_gate_margin"])
    return c


def closest(cands):
    return min(cands, key=lambda c: (c["worst_violation"], c["beta"], c["epoch"]))


# ------------------------------------------------------------------ stage B
def select_B(D):
    out = {}
    for k in R.SEEDS:
        ub = f"tl__s{k}__e20"
        uref = util(ub)
        ub_c = candidate(ub, uref, 0.0, 20)
        res = {"U-B": {"unit": ub, "gates_ok_vs_self": ub_c["gates_ok"], "utility": uref, "auc": ub_c["auc"]}}
        for arm in R.STAGE_B_ARMS:
            C = [candidate(R.ck_name("B", k, arm, b, e), uref, b, e) for b in R.BETAS for e in CKPTS]
            alias = dict(ub_c, alias=True)
            pool = [c for c in C + [alias] if c["feasible"]]
            if not ub_c["gates_ok"]:
                res[arm] = {"status": "NO_VALID_REFERENCE", "table": C, "note": "U-B fails its own G3 (no nontrivial utility)"}
                continue
            best = min(pool, key=lambda c: (round(c["worse_local"], 12), round(c["mean_local"], 12), c["beta"], c["epoch"]))
            status = "TASK_ONLY_ALIAS" if best.get("alias") else "NOMINEE"
            res[arm] = {"status": status, **{kk: best[kk] for kk in ("unit", "beta", "epoch", "auc", "utility",
                                                                      "worse_local", "mean_local", "worst_gate_margin")},
                        "table": C,
                        "rule": "min worse-local AUC, mean local, smaller beta, earlier checkpoint; U-B alias included"}
        out[str(k)] = res
    (R.RUN / "selection_B.json").write_text(json.dumps(out, indent=1, default=float))
    R.PKG.mkdir(parents=True, exist_ok=True)
    pub = {"schema": "rgj-stage-b-freeze-v1", "rule": "PROTOCOL.md section 4 step 1 (inner roles only)",
           "seeds": {s: {arm: {kk: v.get(kk) for kk in ("status", "unit", "beta", "epoch", "auc", "worse_local",
                                                         "mean_local", "worst_gate_margin")}
                         for arm, v in r.items() if arm in R.STAGE_B_ARMS} | {"U-B": r["U-B"]} for s, r in out.items()}}
    (R.PKG / "STAGE_B_SELECTION.json").write_text(json.dumps(pub, indent=1, default=float) + "\n")
    write_table(out, "B")
    R.event("selection B written")
    return out


def write_table(out, stage):
    """SELECTION_TABLE.csv rows for every candidate considered (inner values and the reason it was not chosen)."""
    import csv
    path = R.PKG / "SELECTION_TABLE.csv"
    rows = []
    if path.exists():
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r["stage"] != stage]
    for s, res in out.items():
        arms = res.get("arms", res)
        for arm, a in arms.items():
            if not isinstance(a, dict) or "table" not in a:
                continue
            chosen = a.get("unit")
            for c in a["table"]:
                why = "selected" if c["unit"] == chosen and a.get("status") in ("NOMINEE",) else (
                    "closest (descriptive; infeasible)" if c["unit"] == chosen else
                    ("gates failed" if not c["gates_ok"] else ("local guard failed" if not c.get("guard_ok", True)
                                                                else "feasible, not best by rule")))
                rows.append({"stage": stage, "seed": s, "arm": arm, "unit": c["unit"], "beta": c["beta"],
                             "epoch": c["epoch"], "auc_v1": f"{c['auc']['v1']:.6f}", "auc_v2": f"{c['auc']['v2']:.6f}",
                             "auc_pair": f"{c['auc']['pair']:.6f}", "acc_income": f"{c['utility'][0]['acc']:.6f}",
                             "acc_occ": f"{c['utility'][1]['acc']:.6f}", "gates_ok": c["gates_ok"],
                             "worst_gate_margin": f"{c['worst_gate_margin']:.6f}", "guard_ok": c.get("guard_ok", ""),
                             "feasible": c["feasible"], "arm_status": a.get("status"), "outcome": why})
    cols = ["stage", "seed", "arm", "unit", "beta", "epoch", "auc_v1", "auc_v2", "auc_pair", "acc_income", "acc_occ",
            "gates_ok", "worst_gate_margin", "guard_ok", "feasible", "arm_status", "outcome"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ stage C
def u_unit(k):
    return f"tl__s{k}__e{20 + R.lr_init(k)[3]}"


def run_baselines(D, shard_spec=None):
    from rgj import baselines as BL
    for k in R.shard(list(R.SEEDS), shard_spec):
        BL.leace_unit(k, u_unit(k), D, R.UNITS)
        BL.fare_units(k, D, R.UNITS)


def select_C(D):
    from rgj import baselines as BL
    SB = R.selB()
    out = {}
    for k in R.SEEDS:
        s = str(k)
        res = {"seed": k, "arms": {}}
        if SB[s]["L-R"]["status"] == "NO_VALID_REFERENCE":
            res["valid_reference"] = False
            out[s] = res
            continue
        res["valid_reference"] = True
        uname = u_unit(k)
        uref = util(uname)
        LR = SB[s]["L-R"]
        lr_auc = LR["auc"]
        guard = {"v1": lr_auc["v1"], "v2": lr_auc["v2"]}
        res["U"] = {"unit": uname, "utility": uref}
        res["L-R_reference"] = {"unit": LR["unit"], "status": LR["status"], "auc": lr_auc}
        # 1. trained controls
        for arm in ("L-G", "J-R", "J-O"):
            C = [candidate(R.ck_name("C", k, arm, b, e), uref, b, e, guard) for b in R.BETAS for e in CKPTS]
            feas = [c for c in C if c["feasible"]]
            if feas:
                b = min(feas, key=lambda c: (round(c["auc"]["pair"], 12), c["beta"], c["epoch"]))
                res["arms"][arm] = {"status": "NOMINEE", **b, "table": C}
            else:
                res["arms"][arm] = {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE", **closest(C), "table": C}
        # 2. single-configuration and frozen controls
        for arm, name, beta, ep in (("L-R", LR["unit"], LR.get("beta", 0.0), LR.get("epoch", 20)),
                                    ("L-O", SB[s]["L-O"].get("unit"), SB[s]["L-O"].get("beta", 0.0), SB[s]["L-O"].get("epoch", 20)),
                                    ("U", uname, 0.0, 0), ("E", f"lc__s{k}__E", 0.0, 0)):
            if name is None:
                res["arms"][arm] = {"status": "ABSENT"}
                continue
            c = candidate(name, uref, beta, ep, guard)
            st = "NOMINEE" if c["feasible"] else "INFEASIBLE_CONTROL"
            if arm == "L-R":
                st = "NOMINEE" if c["feasible"] and LR["status"] == "NOMINEE" else (
                    "TASK_ONLY_ALIAS" if LR["status"] == "TASK_ONLY_ALIAS" and c["feasible"] else "INFEASIBLE_CONTROL")
            if arm == "L-O" and SB[s]["L-O"]["status"] != "NOMINEE":
                st = SB[s]["L-O"]["status"] if not c["feasible"] else ("TASK_ONLY_ALIAS" if SB[s]["L-O"]["status"] == "TASK_ONLY_ALIAS" else st)
            res["arms"][arm] = {"status": st, "stage_B_status": (SB[s][arm]["status"] if arm in SB[s] else None), **c}
        fare = BL.select_fare(k, D, R.UNITS, uref, guard)
        for arm in ("F", "F0"):
            res["arms"][arm] = fare[arm]
        res["fare_selection"] = fare.get("per_purpose")
        # 3. C*
        cands = []
        for j, arm in enumerate(CONTROL_ORDER):
            a = res["arms"].get(arm, {})
            if a.get("status") in ("NOMINEE", "TASK_ONLY_ALIAS") and a.get("feasible", False):   # review R2
                cands.append((round(a["auc"]["pair"], 12), j, arm))
        cs = min(cands)[2] if cands else None
        res["comparator"] = {"arm": cs, "unit": (res["arms"][cs].get("unit") or res["arms"][cs].get("units")) if cs else None,
                             "status": res["arms"][cs]["status"] if cs else "ABSENT",
                             "is_task_only_release": cs == "U" or (cs in ("L-R", "L-O") and res["arms"][cs]["status"] == "TASK_ONLY_ALIAS"),
                             "candidates": [c[2] for c in sorted(cands)],
                             "rule": "lowest inner coalition AUC among feasible controls; ties " + ",".join(CONTROL_ORDER)}
        # 4. J-G nomination
        lg = res["arms"]["L-G"]
        jg_guard = dict(guard)
        for w in ("v1", "v2"):
            if lg["status"] == "NOMINEE":
                jg_guard[w] = min(jg_guard[w], lg["auc"][w])
            if cs is not None:
                jg_guard[w] = min(jg_guard[w], res["arms"][cs]["auc"][w])
        C = [candidate(R.ck_name("C", k, "J-G", b, e), uref, b, e, jg_guard) for b in R.BETAS for e in CKPTS]
        feas = [c for c in C if c["feasible"]]
        if feas:
            b = min(feas, key=lambda c: (round(c["auc"]["pair"], 12), c["beta"], c["epoch"]))
            res["arms"]["J-G"] = {"status": "NOMINEE", **b, "table": C}
        else:
            res["arms"]["J-G"] = {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE", **closest(C), "table": C}
        res["arms"]["J-G"]["nomination_guard"] = jg_guard
        out[s] = res
    (R.RUN / "selection_C.json").write_text(json.dumps(out, indent=1, default=float))
    status = {s: {"valid_reference": r.get("valid_reference"), "U": r.get("U", {}).get("unit"),
                  "L-R_reference": r.get("L-R_reference"), "comparator": r.get("comparator"),
                  "arms": {arm: {kk: a.get(kk) for kk in ("status", "unit", "units", "beta", "epoch", "auc", "gates_ok",
                                                          "guard_ok", "feasible", "worst_gate_margin", "guard_excess",
                                                          "nomination_guard", "pairing_to_F")}
                           for arm, a in r.get("arms", {}).items()}} for s, r in out.items()}
    (R.PKG / "SEED_STATUS.json").write_text(json.dumps(status, indent=1, default=float) + "\n")
    write_table(out, "C")
    R.event("selection C written")
    return out
