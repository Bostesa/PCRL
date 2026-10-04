"""Registered inner selection for the focused no-erasure penalty study (inner roles only: attacker_fit -> attacker_val).

Gates (per seed, both tasks, deployed heads, attacker_val; unchanged from the predecessor):
  G1 Acc_j >= Acc_j(U) - 0.01;  G2 Acc_j - const_j >= 0.8 (Acc_j(U) - const_j);  G3 Acc_j - const_j >= 0.03.
Order:
  1. LN: candidates beta in {0 (= U alias), 0.1, 1, 10}; feasible ones; minimise max(R_v1, R_v2); ties -> mean local,
     then smaller beta. Frozen before PN. If beta = 0 is chosen the release is TASK_ONLY_ALIAS.
  2. PN: candidates beta in {0, 0.1, 1, 10}; feasible AND R_vi <= R_vi(LN) + 0.01 (i = 1, 2); minimise R_pair; ties ->
     smaller beta. beta = 0 chosen -> TASK_ONLY_ALIAS (cannot support a joint-training claim).
  3. JP (existing, erased): candidates beta in {0.1, 1, 10}, same rule as PN (no beta = 0 alias exists for JP).
  4. E: single configuration, gates only. U: the reference; recorded as a feasibility witness.
  5. F: per purpose, the declared published rule (lowest inner local AUC among gate-passing grid configs; ties lower id).
     F's status = NOMINEE only if both purposes have nominees and the pair passes the gates.
  6. F0: frozen units fare__s{k}__p{i}__Z1 (zero-fairness twin of grid config 1, the predecessor's frozen budget). Its OWN
     gate status and its pairing to F are separate fields; it is never labelled infeasible because F lacks a nominee.
  7. C*: among {LN, U, E, JP, F, F0} candidates that pass their own gates and the local allowance versus frozen LN,
     smallest inner R_pair; ties -> order LN, U, E, JP, F, F0. None -> claim B NOT_ESTABLISHED (no substitution).
Infeasible arms keep a descriptive closest-utility configuration (largest worst-gate margin; ties smaller beta),
labelled INFEASIBLE.
"""
from __future__ import annotations

import json

from jcv import select as JS
from pnx import run as R

ORDER = ["LN", "U", "E", "JP", "F", "F0"]


def inner(name):
    return json.loads((R.udir(f"inner__{name}") / "record.json").read_text())


def cand(k, arm, beta, uU):
    nm = R.unit_name(k, arm, beta) if arm in ("PN", "LN") else f"nn__s{k}__{arm}__b{beta:g}"
    r = inner(nm)
    ok, sh = JS.gates({int(i): v for i, v in r["utility"].items()}, uU)
    return {"unit": nm, "beta": beta, "gates_ok": ok, "worst_gate_margin": sh,
            **{f"R_{w}": r["recovery"][w] for w in ("v1", "v2", "pair")}}


def _closest(C):
    """Descriptive closest-utility configuration; nonzero beta only (review R6: beta = 0 would be U itself)."""
    nz = [c for c in C if c["beta"] != 0] or C
    return max(nz, key=lambda c: (c["worst_gate_margin"], -c["beta"]))


def select_seed(D, k):
    uU = {int(i): v for i, v in inner(f"nn__s{k}__U")["utility"].items()}
    out = {"seed": k, "arms": {}}
    # R5: U must itself be nontrivially useful (G3) on both tasks, otherwise there is no valid reference
    out["valid_reference"] = all(uU[i]["acc"] - uU[i]["const_acc"] >= 0.03 for i in (0, 1))
    # 1. LN
    C = [cand(k, "LN", b, uU) for b in [0.0] + R.BETAS]
    feas = [c for c in C if c["gates_ok"]]
    if feas:
        s = min(feas, key=lambda c: (max(c["R_v1"], c["R_v2"]), (c["R_v1"] + c["R_v2"]) / 2, c["beta"]))
        out["arms"]["LN"] = {"status": "TASK_ONLY_ALIAS" if s["beta"] == 0 else "NOMINEE", **s, "table": C}
    else:
        out["arms"]["LN"] = {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE", **_closest(C), "table": C}
    if not out["valid_reference"]:
        out["arms"]["LN"]["status"] = "NO_VALID_REFERENCE"
    LN = out["arms"]["LN"]
    ln_ok = LN["status"] in ("NOMINEE", "TASK_ONLY_ALIAS")

    def guarded(c):
        return ln_ok and c["gates_ok"] and c["R_v1"] <= LN["R_v1"] + 0.01 and c["R_v2"] <= LN["R_v2"] + 0.01

    # 2. PN, 3. JP
    for arm, betas in (("PN", [0.0] + R.BETAS), ("JP", R.BETAS)):
        C = [cand(k, arm, b, uU) for b in betas]
        feas = [c for c in C if guarded(c)]
        if feas:
            s = min(feas, key=lambda c: (c["R_pair"], c["beta"]))
            out["arms"][arm] = {"status": "TASK_ONLY_ALIAS" if s["beta"] == 0 else "NOMINEE", **s, "table": C}
        else:
            out["arms"][arm] = {"status": "NO_FEASIBLE_NOMINEE", "descriptive": "INFEASIBLE", **_closest(C), "table": C,
                                "note": "no configuration met the gates and the local allowance versus frozen LN"
                                if ln_ok else "LN has no feasible configuration: local allowance undefined"}
    # 4. U, E
    for arm in ("U", "E"):
        r = inner(f"nn__s{k}__{arm}")
        ok, sh = JS.gates({int(i): v for i, v in r["utility"].items()}, uU)
        out["arms"][arm] = {"status": ("REFERENCE" if arm == "U" else ("NOMINEE" if ok else "GATES_FAILED")),
                            "unit": f"nn__s{k}__{arm}", "beta": 0.0, "gates_ok": ok, "worst_gate_margin": sh,
                            **{f"R_{w}": r["recovery"][w] for w in ("v1", "v2", "pair")}}
    # 5. F (per purpose, declared rule), 6. F0 (frozen budget, own gates)
    fsel = {}
    for i in (0, 1):
        rows = []
        for cfg in range(1, 7):
            r = inner(f"fare__s{k}__p{i}__c{cfg}")
            ok, sh = JS.gate_one(r["utility"], uU[i])
            rows.append({"config": cfg, "unit": f"fare__s{k}__p{i}__c{cfg}", "gate_ok": ok, "margin": sh,
                         "R_local": r["recovery_local"], "acc": r["utility"]["acc"]})
        adm = [x for x in rows if x["gate_ok"]]
        b = min(adm, key=lambda x: (round(x["R_local"], 12), x["config"])) if adm else max(rows, key=lambda x: (x["margin"], -x["config"]))
        fsel[i] = {"status": "NOMINEE" if adm else "NO_FEASIBLE_NOMINEE", **b, "table": rows}
    for arm, tag in (("F", "c"), ("F0", "Z")):
        cfgs = [fsel[0]["config"], fsel[1]["config"]] if arm == "F" else [1, 1]
        units = [f"fare__s{k}__p0__{tag}{cfgs[0]}", f"fare__s{k}__p1__{tag}{cfgs[1]}"]
        r = inner(f"pair__s{k}__{arm}")
        assert r["of"] == units, f"inner pair record for {arm} does not match the frozen units"
        ok, sh = JS.gates({int(i): v for i, v in r["utility"].items()}, uU)
        if arm == "F":
            st = "NOMINEE" if (ok and all(fsel[i]["status"] == "NOMINEE" for i in (0, 1))) else "NO_FEASIBLE_NOMINEE"
            extra = {}
        else:
            st = "NOMINEE" if ok else "GATES_FAILED"
            extra = {"own_gate_status": "PASS" if ok else "FAIL",
                     "pairing_to_F": "zero-fairness twin of grid config 1 for both purposes (budget frozen from the "
                                     "predecessor); F's own per-purpose selection: " +
                                     ", ".join(f"p{i} {fsel[i]['status']} cfg {fsel[i]['config']}" for i in (0, 1))}
        out["arms"][arm] = {"status": st, "units": units, "configs": cfgs, "gates_ok": ok, "worst_gate_margin": sh,
                            **{f"R_{w}": r["recovery"][w] for w in ("v1", "v2", "pair")}, **extra}
    out["fare_selection"] = fsel
    # 7. C*
    cands = []
    for arm in ORDER:
        a = out["arms"][arm]
        feasible = a["status"] in ("NOMINEE", "TASK_ONLY_ALIAS", "REFERENCE") and a.get("gates_ok", False)
        if feasible and ln_ok and a["R_v1"] <= LN["R_v1"] + 0.01 and a["R_v2"] <= LN["R_v2"] + 0.01:
            cands.append((a["R_pair"], ORDER.index(arm), arm))
    if not out["valid_reference"]:
        cands = []
        for arm in ("PN", "JP", "E", "F", "F0"):
            if out["arms"][arm]["status"] in ("NOMINEE", "TASK_ONLY_ALIAS"):
                out["arms"][arm]["status"] = "NO_VALID_REFERENCE"
    win = min(cands)[2] if cands else None
    wa = out["arms"].get(win, {}) if win else {}
    out["comparator"] = {"arm": win, "candidates": [c[2] for c in cands],
                         "status": wa.get("status"), "unit": wa.get("unit", wa.get("units")),
                         "is_task_only_release": bool(win) and (win == "U" or wa.get("status") == "TASK_ONLY_ALIAS"),
                         "rule": "smallest inner coalition recovery among feasible controls within the local allowance "
                                 "versus frozen LN; ties LN, U, E, JP, F, F0"}
    return out


def run_selection(D, seeds):
    res = {k: select_seed(D, k) for k in seeds}
    (R.RUN / "selection.json").write_text(json.dumps(res, indent=1, default=float))
    R.event("selection done", seeds=seeds)
    return res
