#!/usr/bin/env python3
"""Check C - erase-layer pilot, VICReg x5 sweep and rank-8 ablation: per-cell recount (standalone).

For every per-seed file located, recompute per (dataset, purpose, attribute, seed) cell:
  strict R2 (linear_r2 <= 0.05; in-sample OLS on TEST-split representations, from generate_report),
  per_dim_std_mean / effective_rank of the cell's own purpose (TEST-split reps, eval mode),
  cleanly compliant (strict AND std >= 0.5 AND eff_rank >= 2),
  nonlinear auditor channel delta = empirical_best_acc - majority (auditors fit on TRAIN reps,
  scored on TEST; adj_pass = delta < 0.02 AND R2 < 0.05),
  task accuracy (trainer.evaluate(test_loader): TEST split, the split also used for every tuning decision).
Checkpoint: run_v2_dataset.py reloads canonical_iterate.pt > best.pt > final.pt; for these runs best.pt
(Cotter selector; 'kind' recorded per seed).
Baseline = R5/R7 per_seed_results.json (also best.pt).
Usage: python3 C_erase_vicreg.py > ../recount_outputs/C_erase_vicreg.json
"""
import hashlib
import json
import subprocess
from pathlib import Path

REPO = "/Users/nathansamson/PCRL"
TAU, STD_MIN, ER_MIN, AUD = 0.05, 0.5, 2.0, 0.02
INPUTS = []


def load(src):
    if src.startswith("git:"):
        ref, path = src[4:].split("::")
        raw = subprocess.check_output(["git", "-C", REPO, "show", f"{ref}:{path}"])
        loc = f"Bostesa/PCRL@{ref}:{path}"
    else:
        raw = Path(REPO, src).read_bytes()
        loc = f"local-untracked:{REPO}/{src}"
    INPUTS.append({"location": loc, "sha256": hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


RUNS = {
    "baseline_R5R7_bestpt": {
        "adult": "git:origin/main::results/v2_adult_ROUND5/per_seed_results.json",
        "hmda": "git:origin/main::results/v2_hmda_ROUND5/per_seed_results.json",
        "diabetes": "git:origin/main::results/v2_diabetes_ROUND7/per_seed_results.json"},
    "erase_pilot_vicreg1": {
        "adult": "results/rebuttal/erase_layer_pilot_aws/v2_adult_ERASE_PILOT/per_seed_results.json",
        "hmda": "results/rebuttal/erase_layer_pilot_aws/v2_hmda_ERASE_PILOT/per_seed_results.json",
        "diabetes": "results/rebuttal/erase_layer_pilot_aws/v2_diabetes_ERASE_PILOT/per_seed_results.json"},
    "erase_vicreg5": {
        "adult": "git:origin/erase-layer-vicreg-sweep-2026-05-18::results/rebuttal/erase_layer_vicreg_sweep_aws/v2_adult_ERASE_VICREG5/per_seed_results.json",
        "hmda": "git:origin/erase-layer-vicreg-sweep-2026-05-18::results/rebuttal/erase_layer_vicreg_sweep_aws/v2_hmda_ERASE_VICREG5/per_seed_results.json"},
    "erase_rank8_diabetes": {
        "diabetes": "git:origin/rebuttal-evidence::results/rebuttal/erase_rank8_diabetes_cpu/v2_diabetes_ERASE_RANK8/per_seed_results.json"},
}


def mean(x):
    return sum(x) / len(x) if x else None


out = {"definitions": __doc__.strip().splitlines()[2:12], "runs": {}}
for run, dsmap in RUNS.items():
    R = {"per_dataset": {}, "totals": {"n": 0, "strict": 0, "clean": 0, "adj_pass": 0, "aud_delta_ge_0.02": 0}}
    for ds, src in dsmap.items():
        d = load(src)
        cells = []
        for s in d["per_seed"]:
            h = s["per_purpose_health"]
            for r in s["attribute_results"]:
                std, er = h[r["purpose"]]["per_dim_std_mean"], h[r["purpose"]]["effective_rank"]
                strict = r["linear_r2"] <= TAU
                cells.append({"seed": s["seed"], "purpose": r["purpose"], "attribute": r["attribute"],
                              "r2": r["linear_r2"], "strict": strict, "per_dim_std": round(std, 4),
                              "eff_rank": round(er, 3), "clean": strict and std >= STD_MIN and er >= ER_MIN,
                              "aud_delta": r["delta"], "adj_pass_stored": r["adj_pass"],
                              "adj_pass_recomputed": (r["delta"] < AUD and r["linear_r2"] < TAU)})
        tasks = sorted({k for s in d["per_seed"] for k in s["task_accuracies"]})
        std_by_seed_purpose = [h["per_dim_std_mean"] for s in d["per_seed"] for h in s["per_purpose_health"].values()]
        er_by_seed_purpose = [h["effective_rank"] for s in d["per_seed"] for h in s["per_purpose_health"].values()]
        e = {"n_cells": len(cells), "seeds": [s["seed"] for s in d["per_seed"]],
             "strict": sum(c["strict"] for c in cells), "clean": sum(c["clean"] for c in cells),
             "adj_pass": sum(c["adj_pass_recomputed"] for c in cells),
             "adj_pass_mismatch_vs_stored": sum(c["adj_pass_recomputed"] != c["adj_pass_stored"] for c in cells),
             "aud_delta_ge_0.02": sum(c["aud_delta"] >= AUD for c in cells),
             "aud_delta_max": round(max(c["aud_delta"] for c in cells), 4),
             "aud_delta_mean": round(mean([c["aud_delta"] for c in cells]), 4),
             "r2_max": round(max(c["r2"] for c in cells), 4), "r2_mean": round(mean([c["r2"] for c in cells]), 4),
             "per_dim_std_mean_over_seed_purpose": round(mean(std_by_seed_purpose), 4),
             "per_dim_std_max_over_seed_purpose": round(max(std_by_seed_purpose), 4),
             "n_seed_purpose_std_ge_0.5": sum(x >= STD_MIN for x in std_by_seed_purpose),
             "n_seed_purpose": len(std_by_seed_purpose),
             "eff_rank_mean_over_seed_purpose": round(mean(er_by_seed_purpose), 3),
             "eff_rank_min_over_seed_purpose": round(min(er_by_seed_purpose), 3),
             "per_dim_std_mean_over_cells": round(mean([c["per_dim_std"] for c in cells]), 4),
             "eff_rank_mean_over_cells": round(mean([c["eff_rank"] for c in cells]), 3),
             "task_acc_mean_TEST": {t: round(mean([s["task_accuracies"][t] for s in d["per_seed"]]), 4) for t in tasks},
             "best_epochs": [s.get("best_epoch") for s in d["per_seed"]],
             "cotter_kind": [s.get("cotter_selection", {}).get("kind") for s in d["per_seed"]],
             "n_feasible_post_warmup": [s.get("cotter_selection", {}).get("n_feasible_post_warmup") for s in d["per_seed"]],
             "std_scale_needed_for_all_purposes_to_reach_0.5": round(STD_MIN / min(std_by_seed_purpose), 3),
             "cells": cells}
        R["per_dataset"][ds] = e
        for k in ["strict", "clean", "adj_pass", "aud_delta_ge_0.02"]:
            R["totals"][k] += e[k]
        R["totals"]["n"] += e["n_cells"]
    out["runs"][run] = R
# paired per-cell comparison vs the best.pt baseline (same seed, purpose, attribute)
def keyed(run, ds):
    return {(c["seed"], c["purpose"], c["attribute"]): c for c in out["runs"][run]["per_dataset"][ds]["cells"]}


out["paired_vs_baseline"] = {}
for run in ["erase_pilot_vicreg1", "erase_vicreg5", "erase_rank8_diabetes"]:
    for ds in out["runs"][run]["per_dataset"]:
        b, p = keyed("baseline_R5R7_bestpt", ds), keyed(run, ds)
        ks = sorted(set(b) & set(p))
        out["paired_vs_baseline"][f"{run}/{ds}"] = {
            "n": len(ks),
            "aud_delta_rose": sum(p[k]["aud_delta"] > b[k]["aud_delta"] for k in ks),
            "mean_aud_delta_baseline": round(mean([b[k]["aud_delta"] for k in ks]), 4),
            "mean_aud_delta_run": round(mean([p[k]["aud_delta"] for k in ks]), 4),
            "r2_fell": sum(p[k]["r2"] < b[k]["r2"] for k in ks)}
# in-sample OLS null floor on the TEST split: E[R2 | independence] ~ p/(n-1), p = 64 regressors
NTEST = {"adult": 15060, "hmda": 13661, "diabetes": 10728}
out["null_floor_insample_R2_p64"] = {ds: round(64 / (n - 1), 4) for ds, n in NTEST.items()}
out["null_floor_note"] = "n_test from diagnostic_perdim_std.json; pilot mean R2 (0.0078/0.0057/0.0069) sits at ~1.2-1.8x this floor"
out["inputs"] = INPUTS
print(json.dumps(out, indent=1))
