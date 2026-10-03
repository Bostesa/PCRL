"""Registered inner selection (running ladder C-E). Inner roles only: attacker_fit -> attacker_val.

Common utility gates (per seed, both tasks, deployed heads, attacker_val point estimates):
  G1  Acc_j(arm) >= Acc_j(U) - 0.01
  G2  Acc_j(arm) - const_j >= 0.8 * (Acc_j(U) - const_j)        (const_j: defense_train majority class)
  G3  Acc_j(arm) - const_j >= 0.03
Order:
  1. L: feasible betas; minimise max(R_v1, R_v2) (worse local SEX recovery); ties -> mean local -> smaller beta. Frozen.
  2. J, JP, S12, S21: feasible betas with R_vi <= R_vi(L) + 0.01 (i = 1, 2); minimise R_pair; ties -> smaller beta.
  3. U, E: single configuration (no privacy tuning).
  4. F (official FARE, declared published rule within the gates): per purpose, among grid configurations passing that
     purpose's gates, lowest inner local SEX AUC; ties -> lower grid id. F0: zero-fairness twin at F's selected config.
  5. Comparator C* (claim 2): among nominees of {E, L, JP, S12, S21, F, F0} passing G1-G3 for both tasks and the local
     allowance versus L, smallest inner coalition recovery; ties -> order E, L, JP, S12, S21, F, F0. If L wins, the
     claim-2 slots alias claim 1. None -> claim 2 NOT_ESTABLISHED (no substitution).
No feasible configuration -> NO_FEASIBLE_NOMINEE; the predeclared closest-utility configuration (smallest worst gate
shortfall; ties -> smaller beta / lower id) is scored after the lock, labelled INFEASIBLE, and cannot pass a claim.
"""
from __future__ import annotations

import json
import time

import numpy as np

from jcv import run as R

ORDER = ["E", "L", "JP", "S12", "S21", "F", "F0"]


def gates(util, utilU):
    sh = []
    ok = True
    for i in (0, 1):
        a, aU, c = util[i]["acc"], utilU[i]["acc"], util[i]["const_acc"]
        g = {"G1": a - (aU - 0.01), "G2": (a - c) - 0.8 * (aU - c), "G3": (a - c) - 0.03}
        sh.append(min(g.values()))
        ok &= all(x >= 0 for x in g.values())
    return ok, min(sh)


def gate_one(u, uU):
    a, aU, c = u["acc"], uU["acc"], u["const_acc"]
    g = {"G1": a - (aU - 0.01), "G2": (a - c) - 0.8 * (aU - c), "G3": (a - c) - 0.03}
    return all(x >= 0 for x in g.values()), min(g.values())


def inner(name):
    return json.loads((R.U(f"inner__{name}") / "record.json").read_text())


def select_seed(D, k):
    betas = R.T.HP["betas"]
    out = {"seed": k, "arms": {}}
    uU = {int(i): v for i, v in inner(f"nn__s{k}__U")["utility"].items()}

    def cand(arm, b):
        nm = f"nn__s{k}__{arm}__b{b:g}"
        r = inner(nm)
        ok, sh = gates({int(i): v for i, v in r["utility"].items()}, uU)
        return {"unit": nm, "beta": b, "gates_ok": ok, "worst_gate_margin": sh, **{f"R_{w}": r["recovery"][w] for w in ("v1", "v2", "pair")}}

    # 1. L
    Lc = [cand("L", b) for b in betas]
    feas = [c for c in Lc if c["gates_ok"]]
    if feas:
        Lsel = min(feas, key=lambda c: (max(c["R_v1"], c["R_v2"]), (c["R_v1"] + c["R_v2"]) / 2, c["beta"]))
        out["arms"]["L"] = {"status": "NOMINEE", **Lsel, "table": Lc}
    else:
        clo = max(Lc, key=lambda c: (c["worst_gate_margin"], -c["beta"]))
        out["arms"]["L"] = {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE", **clo, "table": Lc}
    Lref = out["arms"]["L"]
    # 2. coalition-aware arms
    for arm in ("J", "JP", "S12", "S21"):
        C = [cand(arm, b) for b in betas]
        if Lref["status"] == "NOMINEE":
            feas = [c for c in C if c["gates_ok"] and c["R_v1"] <= Lref["R_v1"] + 0.01 and c["R_v2"] <= Lref["R_v2"] + 0.01]
        else:
            feas = []
        if feas:
            sel = min(feas, key=lambda c: (c["R_pair"], c["beta"]))
            out["arms"][arm] = {"status": "NOMINEE", **sel, "table": C}
        else:
            clo = max(C, key=lambda c: (c["worst_gate_margin"], -c["beta"]))
            out["arms"][arm] = {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE", **clo, "table": C,
                                "note": "no configuration met the gates and the local allowance versus frozen L"
                                        if Lref["status"] == "NOMINEE" else "L has no nominee: local allowance undefined"}
    # 3. U, E
    for arm in ("U", "E"):
        r = inner(f"nn__s{k}__{arm}")
        ok, sh = gates({int(i): v for i, v in r["utility"].items()}, uU)
        out["arms"][arm] = {"status": "NOMINEE" if ok or arm == "U" else "GATES_FAILED", "unit": f"nn__s{k}__{arm}",
                            "gates_ok": ok, "worst_gate_margin": sh, **{f"R_{w}": r["recovery"][w] for w in ("v1", "v2", "pair")}}
    # 4. F per purpose, then F0 at F's budget
    fsel = {}
    for i in (0, 1):
        rows = []
        for cfg in R.FARE_GRID:
            r = inner(f"fare__s{k}__p{i}__c{cfg['id']}")
            ok, sh = gate_one(r["utility"], uU[i])
            rows.append({"config": cfg["id"], "unit": f"fare__s{k}__p{i}__c{cfg['id']}", "gate_ok": ok, "margin": sh,
                         "R_local": r["recovery_local"], "acc": r["utility"]["acc"]})
        adm = [x for x in rows if x["gate_ok"]]
        if adm:
            b = min(adm, key=lambda x: (round(x["R_local"], 12), x["config"]))
            fsel[i] = {"status": "NOMINEE", **b, "table": rows}
        else:
            b = max(rows, key=lambda x: (x["margin"], -x["config"]))
            fsel[i] = {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE", **b, "table": rows}
    return out, fsel


def finish_fare(D, k, out, fsel):
    """Fit F0 at F's selected budget, assemble F/F0 pair views, inner-audit them, add C*."""
    from stored_model_eval.guards import FitAuthorization
    auth = FitAuthorization(execute_scientific_fits=True)
    cfg = {c["id"]: c for c in R.FARE_GRID}
    for i in (0, 1):
        R._fare_release(f"fare__s{k}__p{i}__Z{fsel[i]['config']}", k, i, cfg[fsel[i]["config"]], D, auth, zero=True)
    uU = {int(i): v for i, v in inner(f"nn__s{k}__U")["utility"].items()}
    for arm, tag in (("F", "c"), ("F0", "Z")):
        n1, n2 = f"fare__s{k}__p0__{tag}{fsel[0]['config']}", f"fare__s{k}__p1__{tag}{fsel[1]['config']}"
        iname = f"inner__pair__s{k}__{arm}"
        if not R.done(iname):
            V = R.fare_pair_views(n1, n2)
            rec = {"unit": iname, "of": [n1, n2], "recovery": R.inner_recovery(V, D, finite=True),
                   "utility": R.inner_utility(V["out"], D)}
            R.FN.save_unit(R.U(iname), {}, rec)
        r = json.loads((R.U(iname) / "record.json").read_text())
        ok, sh = gates({int(i): v for i, v in r["utility"].items()}, uU)
        feasible = ok and all(fsel[i]["status"] == "NOMINEE" for i in (0, 1))
        status = "NOMINEE" if feasible else ("GATES_FAILED" if arm == "F0" else "NO_FEASIBLE_NOMINEE")
        out["arms"][arm] = {"status": status, "units": [n1, n2], "gates_ok": ok, "worst_gate_margin": sh,
                            "configs": [fsel[0]["config"], fsel[1]["config"]],
                            **{f"R_{w}": r["recovery"][w] for w in ("v1", "v2", "pair")}}
    out["fare_selection"] = fsel
    # comparator C*
    L = out["arms"]["L"]
    cands = []
    if L["status"] == "NOMINEE":
        for arm in ORDER:
            a = out["arms"][arm]
            if a["status"] == "NOMINEE" and a.get("gates_ok", True) and a["R_v1"] <= L["R_v1"] + 0.01 and \
                    a["R_v2"] <= L["R_v2"] + 0.01:
                cands.append((a["R_pair"], ORDER.index(arm), arm))
    out["comparator"] = {"arm": min(cands)[2] if cands else None, "candidates": [c[2] for c in cands],
                         "rule": "smallest inner coalition recovery among feasible controls; ties E,L,JP,S12,S21,F,F0"}
    return out


def run_selection(D, seeds):
    res = {}
    for k in seeds:
        out, fsel = select_seed(D, k)
        res[k] = finish_fare(D, k, out, fsel)
        R.event("selection seed done", seed=k)
    (R.RUN / "selection.json").write_text(json.dumps(res, indent=1, default=float))
    return res
