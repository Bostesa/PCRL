"""D1/D2: recount the INLP / LAFTR / PCRL three-way head-to-head (NeurIPS App. Q,
Table 13; 0.014/0.038/0.270, 56/43/15 of 60) and recompute INLP + LAFTR(Adult)
one-hot R^2 from stored representations.

Inputs: origin/main inlp_results.json, per-cell INLP/LAFTR metrics.json, PCRL
final.pt dominant-axis audits, PCRL per_seed_results (best.pt) task accuracies;
drive member results/inlp_benchmark/*/eval_reps.npz + test_labels.npz
(fl-PCRL-main-results-ignored.tar); LAFTR Adult eval_reps from git history
(5847a401e^, before they were untracked).
Run: /opt/homebrew/bin/python3 D_three_way.py
"""
from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D_common import (SCRATCH, dump, file_bytes, git_bytes, git_json, inputs,  # noqa: E402
                      r2_onehot)

MAIN = "origin/main"
PURPOSES = {
    "adult": ["income_prediction", "employment_analysis", "education_assessment"],
    "hmda": ["underwriting", "pricing_analysis", "fair_lending_audit"],
    "diabetes": ["billing_audit", "quality_research", "clinical_decision_support"],
}
DA = {"adult": "results/v2_adult_ROUND5/dominant_axis_audit.json",
      "hmda": "results/v2_hmda_ROUND5/dominant_axis_audit.json",
      "diabetes": "results/v2_diabetes_ROUND7/dominant_axis_audit.json"}
PSR = {"adult": "results/v2_adult_ROUND5/per_seed_results.json",
       "hmda": "results/v2_hmda_ROUND5/per_seed_results.json",
       "diabetes": "results/v2_diabetes_ROUND7/per_seed_results.json"}

inlp = git_json(MAIN, "results/inlp_benchmark/inlp_results.json")
rows = inlp["rows"]
out: dict = {"n_rows": len(rows), "n_cells": inlp["n_cells"]}

# --- 1. stored-column recount ------------------------------------------------
for m in ("pcrl", "inlp", "laftr"):
    v = np.array([r[f"{m}_r2_onehot"] for r in rows])
    out[f"{m}_mean_r2"] = round(float(v.mean()), 4)
    out[f"{m}_pass_lt_0.05"] = int((v < 0.05).sum())
    out[f"{m}_pass_le_0.05"] = int((v <= 0.05).sum())
    out[f"{m}_stored_pass_flag"] = int(sum(r[f"{m}_strict_pass"] for r in rows))
by_ds = defaultdict(lambda: defaultdict(list))
for r in rows:
    for m in ("pcrl", "inlp", "laftr"):
        by_ds[r["dataset"]][m].append(r[f"{m}_r2_onehot"] < 0.05)
out["pass_by_dataset_lt"] = {d: {m: f"{sum(v)}/{len(v)}" for m, v in mm.items()}
                             for d, mm in by_ds.items()}

# --- 2. PCRL column == final.pt DA audit? ------------------------------------
da_r2 = {}
for ds, p in DA.items():
    a = git_json(MAIN, p)
    for s, blk in a["per_seed"].items():
        for row in blk["rows"]:
            da_r2[(ds, row["purpose"], int(s), row["attribute"])] = row["r2_onehot"]
mm = [r for r in rows if abs(da_r2[(r["dataset"], r["purpose"], r["seed"], r["attribute"])]
                             - r["pcrl_r2_onehot"]) > 5e-6]
out["pcrl_column_vs_finalpt_DA_audit_mismatches"] = len(mm)

# --- 3. INLP column == per-cell metrics.json? and independent R^2 recompute ---
inlp_meta = json.loads((SCRATCH / "inlp_inventory_sha.json").read_text())
recomp, mism_json, max_abs = [], 0, 0.0
for r in rows:
    pi = PURPOSES[r["dataset"]].index(r["purpose"])
    rel = f"results/inlp_benchmark/{r['dataset']}/purpose_{pi}/seed_{r['seed']}"
    mj = git_json(MAIN, f"{rel}/metrics.json")
    v_json = mj["metrics"]["per_attr"][r["attribute"]]["r2_onehot"]
    if abs(v_json - r["inlp_r2_onehot"]) > 1e-9:
        mism_json += 1
    reps_b = file_bytes(SCRATCH / rel / "eval_reps.npz",
                        f"fl-PCRL-main-results-ignored.tar::{rel}/eval_reps.npz")
    lab_b = file_bytes(SCRATCH / rel / "test_labels.npz",
                       f"fl-PCRL-main-results-ignored.tar::{rel}/test_labels.npz")
    assert inlp_meta[f"{rel}/eval_reps.npz"] == __import__("hashlib").sha256(reps_b).hexdigest()
    H = np.load(io.BytesIO(reps_b))["reps"]
    y = np.load(io.BytesIO(lab_b))[f"sensitive_{r['attribute']}"]
    v = r2_onehot(H, y)
    d = abs(v - r["inlp_r2_onehot"])
    max_abs = max(max_abs, d)
    recomp.append({"cell": [r["dataset"], r["purpose"], r["seed"], r["attribute"]],
                   "stored": r["inlp_r2_onehot"], "recomputed_f64": v,
                   "pass_flip": (v < 0.05) != (r["inlp_r2_onehot"] < 0.05)})
out["inlp_column_vs_metrics_json_mismatches"] = mism_json
out["inlp_recomputed_from_drive_reps"] = {
    "n": len(recomp), "max_abs_diff": max_abs,
    "pass_lt_0.05_recomputed": sum(x["recomputed_f64"] < 0.05 for x in recomp),
    "pass_flips": [x["cell"] for x in recomp if x["pass_flip"]],
    "mean_recomputed": round(float(np.mean([x["recomputed_f64"] for x in recomp])), 4)}

# --- 4. LAFTR Adult: column == metrics.json; recompute from git-history reps ---
OLD = "5847a401e^"
lr, lmax, lm = [], 0.0, 0
for r in rows:
    if r["dataset"] != "adult":
        continue
    rel = f"results/laftr_benchmark/adult/{r['purpose']}/seed_{r['seed']}"
    mj = git_json(MAIN, f"{rel}/metrics.json")
    if abs(mj["metrics"]["per_attr"][r["attribute"]]["r2_onehot"] - r["laftr_r2_onehot"]) > 1e-5:
        lm += 1
    H = np.load(io.BytesIO(git_bytes(OLD, f"{rel}/eval_reps.npz")))["reps"]
    y = np.load(io.BytesIO(git_bytes(OLD, f"{rel}/test_labels.npz")))[f"sensitive_{r['attribute']}"]
    v = r2_onehot(H, y)
    lmax = max(lmax, abs(v - r["laftr_r2_onehot"]))
    lr.append(v)
out["laftr_adult"] = {"n_rows": len(lr), "column_vs_metrics_json_mismatches": lm,
                      "recomputed_from_git_history_reps_max_abs_diff": lmax,
                      "recomputed_pass_lt_0.05": int(sum(v < 0.05 for v in lr))}
out["laftr_hmda_diabetes_provenance"] = (
    "values in inlp_results.json were joined from an S3 FINAL_BENCHMARK.csv "
    "(origin/bios-pcrl-layer12-2026-05-05:infra/inlp/aggregate.py:58-80); no per-seed "
    "LAFTR HMDA/Diabetes metrics/reps located in any git ref or drive inventory")

# --- 5. LAFTR native criterion on Adult: disc accuracy vs majority -----------
nat = []
for p in PURPOSES["adult"]:
    for s in range(3):
        mj = git_json(MAIN, f"results/laftr_benchmark/adult/{p}/seed_{s}/metrics.json")
        h = mj["history"]
        be = h.get("best_epoch", -1)
        acc = h["val_disc_acc"][be] if be is not None and be >= 0 else h["val_disc_acc"][-1]
        for a, pa in mj["metrics"]["per_attr"].items():
            maj = max(pa["priors"])
            nat.append({"cell": [p, s, a], "best_epoch": be, "val_disc_acc": acc.get(a),
                        "test_majority": maj,
                        "within_1pp_of_majority": (acc.get(a) - maj) * 100 < 1.0})
out["laftr_adult_native_disc_criterion"] = {
    "n": len(nat), "within_1pp": sum(x["within_1pp_of_majority"] for x in nat), "rows": nat}

# --- 6. task accuracy: INLP vs PCRL (paper: INLP 0.815, 'comparable for PCRL') -
cells = inlp["cells"]
out["inlp_task_acc_mean_27"] = round(float(np.mean([c["task_acc"] for c in cells])), 4)
pc = []
for ds in PSR:
    psr = git_json(MAIN, PSR[ds])
    for blk in psr["per_seed"]:
        for c in cells:
            if c["dataset"] == ds and c["seed"] == blk["seed"]:
                pc.append({"cell": [ds, c["purpose"], c["seed"]], "task": c["task_name"],
                           "inlp": c["task_acc"], "pcrl_bestpt": blk["task_accuracies"][c["task_name"]]})
out["pcrl_task_acc_mean_27_bestpt"] = round(float(np.mean([x["pcrl_bestpt"] for x in pc])), 4)
out["task_acc_pairs"] = pc
out["cells_inlp_gt_pcrl_by_1pp"] = sum(x["inlp"] - x["pcrl_bestpt"] > 0.01 for x in pc)

out["inputs"] = inputs()
p = dump("D_three_way.json", out)
print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "task_acc_pairs",
                                                            "laftr_adult_native_disc_criterion")},
                 indent=1, default=float))
print("native:", out["laftr_adult_native_disc_criterion"]["within_1pp"], "/",
      out["laftr_adult_native_disc_criterion"]["n"])
print("wrote", p)
