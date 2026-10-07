"""Public report tables and figures for lra (lead). Reads COMMITTED result artifacts and private unit records only.
Never refits a mapper, re-solves a decoder for a new purpose, or rewrites a scientific unit (prompt section 16).

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.sema --label A:report -- \\
        env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.report <part|all>

parts: fit (fitting-row tables: DECODER_ONLY_ABLATION fit part, BUDGET_FEASIBILITY, DECODER_CERTIFICATES,
       OPTIMIZATION_RECEIPTS), work (ACTUAL_WORK_ACCOUNTING, RUN_STATUS), later parts are added after the inner and
       assessment stages.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np

from lra import run as R

PKG = R.PKG
FIG = PKG / "figures"


def _jsonable(x):
    return json.loads(json.dumps(R._finite(x), allow_nan=False))


def _write_json(name, body):
    (PKG / name).write_text(json.dumps(_jsonable(body), indent=1, allow_nan=False) + "\n")


def _write_csv(name, rows, header=None):
    header = header or (list(rows[0]) if rows else ["status"])
    with open(PKG / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _rec(unit):
    return R.rec(unit) if R.done(unit) else None


# ------------------------------------------------------------------ fitting-row tables
def decoder_only_ablation_fit():
    """D0 vs D1 on IDENTICAL tokens, fitting rows (OSF_DEFENSE_FIT), every fixed map and seed (dec__ records)."""
    rows = []
    for k in R.SEEDS:
        for cid in R.d1_fixed_ids():
            r = _rec(R.unit_for(k, cid))
            if r is None:
                rows.append({"seed": k, "d1_config": cid, "status": "MISSING"})
                continue
            for i in ("1", "2"):
                f = r["fitting"][i]
                rows.append({"scope": "fitting rows (OSF_DEFENSE_FIT)", "seed": k, "d0_config": r["d0_config"],
                             "d1_config": cid, "recipient": int(i), "tokens_bitwise_equal_d0": r["tokens_bitwise_equal_d0"],
                             "L_U": f["L_U"], "L_D0": f["L_D0"], "L_D1": f["L_D1"], "dL_D1_minus_D0": f["L_D1"] - f["L_D0"],
                             "B_U": f["B_U"], "B_D0": f["B_D0"], "B_D1": f["B_D1"], "dB_D1_minus_D0": f["B_D1"] - f["B_D0"],
                             "fit_budget_ok_D0": f["L_D0"] <= f["L_U"] + 0.005 and f["B_D0"] <= f["B_U"] + 0.003,
                             "fit_budget_ok_D1": f["L_D1"] <= f["L_U"] + 0.005 and f["B_D1"] <= f["B_U"] + 0.003,
                             "tokens": f["tokens"], "occupied_fit": f["occupied_fit"], "I_fit": f["I_fit"],
                             "I12_fit": r["I12_fit"], "status": "OK"})
    return rows


def budget_feasibility():
    """Deployed fitting budgets and local caps of every new mapping unit (C-TASK, weighted, constrained)."""
    rows = []
    for k in R.SEEDS:
        for cid in R.new_fit_ids():
            r = _rec(R.unit_for(k, cid))
            if r is None:
                rows.append({"seed": k, "config": cid, "status": "MISSING"})
                continue
            b, dep = r["budgets"], r["deployed"]
            t = dep["terms"]
            row = {"seed": k, "config": cid, "arm": r["arm"], "lam": r.get("lam"), "mapper_status": r["status"],
                   "budgets_enforced": b["enforced"], "deployed_feasible": dep["feasible"], "parity_ok": dep["parity_ok"],
                   "T": t.get("T"), "Phi": t.get("Phi"), "I12": t.get("I12")}
            for i in ("1", "2"):
                row.update({f"L{i}": t.get(f"L{i}"), f"L{i}_U": b["L_U"][i], f"L{i}_limit": b["L_limit"][i],
                            f"L{i}_slack": b["L_limit"][i] - t.get(f"L{i}"), f"B{i}": t.get(f"B{i}"),
                            f"B{i}_U": b["B_U"][i], f"B{i}_limit": b["B_limit"][i],
                            f"B{i}_slack": b["B_limit"][i] - t.get(f"B{i}"), f"I{i}": t.get(f"I{i}"),
                            f"I{i}_cap": ((r.get("refs") or {}).get("I_ctask") or {}).get(i),
                            f"tokens{i}": int(np.sum(r["token_counts"][i]))})
            rows.append(row)
    return rows


def decoder_certificates():
    from lra import decoder as DEC
    body = {"schema": "lra-decoder-certificates-v1", "tolerances": DEC.TOLERANCES, "units": {}, "aggregate": {}}
    agg = {"units": 0, "tables": 0, "supervised_tokens": 0, "fallback_tokens": 0, "all_converged": True,
           "max_stationarity_rel": 0.0, "max_dual_infeas_rel": 0.0, "max_projection_magnitude": 0.0,
           "max_sum_q_residual": 0.0, "min_margin": float("inf"), "tokens_with_tie": 0}
    for k in R.SEEDS:
        for cid in R.d1_fixed_ids() + R.new_fit_ids():
            u = R.unit_for(k, cid)
            r = _rec(u)
            if r is None:
                body["units"][u] = {"status": "MISSING"}
                continue
            cs = r["certificates"]
            body["units"][u] = {"config": cid, "seed": k, "decoder_sha256": r["decoder_sha256"], "certificates": cs}
            agg["units"] += 1
            for c in cs.values():
                agg["tables"] += 1
                agg["supervised_tokens"] += c["supervised_tokens"]
                agg["fallback_tokens"] += c["fallback_tokens"]
                agg["all_converged"] = agg["all_converged"] and bool(c["all_converged"])
                for key in ("max_stationarity_rel", "max_dual_infeas_rel", "max_projection_magnitude", "max_sum_q_residual"):
                    agg[key] = max(agg[key], float(c[key]))
                agg["min_margin"] = min(agg["min_margin"], float(c["min_margin"]))
                agg["tokens_with_tie"] += int(c["tokens_with_tie"])
    body["aggregate"] = agg
    return body


def optimization_receipts():
    body = {"schema": "lra-optimization-receipts-v1", "units": {}}
    for k in R.SEEDS:
        for cid in R.new_fit_ids():
            u = R.unit_for(k, cid)
            r = _rec(u)
            if r is None:
                body["units"][u] = {"status": "MISSING"}
                continue
            w = r["work"]
            body["units"][u] = {"config": cid, "seed": k, "status": r["status"], "winner": r["winner"], "work": w,
                                "eval_ceiling_hit": w["evals"] >= w["eval_ceiling"],
                                "deployed_feasible": r["deployed"]["feasible"], "parity_ok": r["deployed"]["parity_ok"],
                                "sex_used_in_search": r["sex_used_in_search"], "rules_sha256": r["rules_sha256"],
                                "trace_sha256": r.get("trace_sha256"), "cpu_s_mapper": r.get("cpu_s_mapper")}
    return body


def actual_work_accounting():
    led = [json.loads(l) for l in (R.RUN / "COMPUTE_LEDGER.jsonl").read_text().splitlines() if l.strip()]
    units = sorted(p.name for p in R.UNITS.iterdir() if p.is_dir())
    by = {}
    for u in units:
        by.setdefault(u.split("__")[0], []).append(u)
    done = {p: sum(1 for u in v if R.done(u)) for p, v in by.items()}
    return {"schema": "lra-actual-work-v1",
            "admitted_custody_units": {"total": 189, "note": "verified copies (SOURCE_ADMISSION.json); not new fits"},
            "private_units_by_prefix": {p: {"present": len(v), "complete": done[p]} for p, v in sorted(by.items())},
            "registered": {"d1_fixed_map_decodes": len(R.d1_jobs()), "ctask": 3, "weighted": 72, "constrained": 15,
                           "inner_audit_units": len(R.scored_ids()) * 3},
            "compute_ledger": led}


PARTS = {"fit": lambda: (_write_csv("DECODER_ONLY_ABLATION.csv", decoder_only_ablation_fit()),
                         _write_csv("BUDGET_FEASIBILITY.csv", budget_feasibility()),
                         _write_json("DECODER_CERTIFICATES.json", decoder_certificates()),
                         _write_json("OPTIMIZATION_RECEIPTS.json", optimization_receipts())),
         "work": lambda: _write_json("ACTUAL_WORK_ACCOUNTING.json", actual_work_accounting())}


if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    for p in (PARTS if want == ["all"] else want):
        PARTS[p]()
        print("written", p)
