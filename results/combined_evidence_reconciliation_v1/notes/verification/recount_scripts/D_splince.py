"""D3: recount the SPLINCE-vs-PCRL head-to-head (NeurIPS App. R, Table 14:
R2-only 60/60 vs 56/60; +Health 3/60 vs 7/60; +Delta_aud 2/60 vs 3/60).

Also extracts, as a by-product, the health of PCRL's *final.pt* representation:
the SPLINCE driver loads backbone + trained LoRA adapters from
checkpoints/v2_<ds>_ROUND{5,7}_s<seed>/final.pt with no extra projection
(the R5/R7 code has no set_leace_projection; LEACE lives inside the trained
LoRA), so `pre_health` is the health of PCRL's own final.pt h_p on the test
split. Combined with the final.pt dominant-axis R^2 this gives a
single-checkpoint (final.pt) cleanly-compliant count.
Run: /opt/homebrew/bin/python3 D_splince.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D_common import dump, git_json, inputs  # noqa: E402

MAIN = "origin/main"
DA = {"adult": "results/v2_adult_ROUND5/dominant_axis_audit.json",
      "hmda": "results/v2_hmda_ROUND5/dominant_axis_audit.json",
      "diabetes": "results/v2_diabetes_ROUND7/dominant_axis_audit.json"}
PSR = {"adult": "results/v2_adult_ROUND5/per_seed_results.json",
       "hmda": "results/v2_hmda_ROUND5/per_seed_results.json",
       "diabetes": "results/v2_diabetes_ROUND7/per_seed_results.json"}

cells, warm = [], {}
for ds in ("adult", "hmda", "diabetes"):
    for s in range(3):
        m = git_json(MAIN, f"results/splince_benchmark/{ds}_s{s}/metrics.json")
        warm[f"{ds}_s{s}"] = m.get("warm_start_path")
        for p in m["pair_results"]:
            cells.append(dict(ds=ds, seed=s, purpose=p["purpose"], attr=p["attribute"],
                              r2=p["r2_onehot"], delta=p["delta_aud"],
                              post_std=p["post_health"]["per_dim_std_mean"],
                              post_er=p["post_health"]["effective_rank"],
                              pre_std=p["pre_health"]["per_dim_std_mean"],
                              pre_er=p["pre_health"]["effective_rank"],
                              drop=p["task_acc_drop"],
                              fallback=p["splince_fit_info"]["fallback_to_leace"],
                              stored_strict=p["strict_pass"]))
n = len(cells)
r2p = [c["r2"] < 0.05 for c in cells]
hp = [c["post_std"] >= 0.5 and c["post_er"] >= 2.0 for c in cells]
dp = [c["delta"] < 0.02 for c in cells]
out = {
    "n_cells": n,
    "warm_start_paths": warm,
    "splince_r2_only": sum(r2p),
    "splince_r2_and_health": sum(a and b for a, b in zip(r2p, hp)),
    "splince_r2_health_delta": sum(a and b and c for a, b, c in zip(r2p, hp, dp)),
    "splince_r2_and_delta(adj_pass)": sum(a and c for a, c in zip(r2p, dp)),
    "splince_stored_strict_flag": sum(c["stored_strict"] for c in cells),
    "splince_fallbacks_to_leace": sum(c["fallback"] for c in cells),
    "splince_mean_task_drop_pp": round(100 * float(np.mean([c["drop"] for c in cells])), 2),
    "splince_post_per_dim_std_range": [min(c["post_std"] for c in cells),
                                      max(c["post_std"] for c in cells)],
    "splince_cells_with_post_std_gt_10": sum(c["post_std"] > 10 for c in cells),
    "splince_health_passes_detail": [
        [c["ds"], c["seed"], c["purpose"], c["attr"], round(c["post_std"], 3),
         round(c["post_er"], 3), round(c["delta"], 4)]
        for c, h in zip(cells, hp) if h],
}

# ---- PCRL single-checkpoint clean counts --------------------------------------
da = {}
for ds, p in DA.items():
    for s, blk in git_json(MAIN, p)["per_seed"].items():
        for row in blk["rows"]:
            da[(ds, int(s), row["purpose"], row["attribute"])] = row
best_h, best_r2 = {}, {}
for ds, p in PSR.items():
    for blk in git_json(MAIN, p)["per_seed"]:
        for pur, h in blk["per_purpose_health"].items():
            best_h[(ds, blk["seed"], pur)] = (h["per_dim_std_mean"], h["effective_rank"])
        for ar in blk["attribute_results"]:
            best_r2[(ds, blk["seed"], ar["purpose"], ar["attribute"])] = ar["linear_r2"]


def classify(r2, std, er):
    if r2 >= 0.05:
        return "failed"
    return "clean" if (std >= 0.5 and er >= 2.0) else "collapse"


rules = {"final_r2 x best_health (paper rule)": Counter(),
         "final_r2 x final_health (single ckpt final.pt)": Counter(),
         "best_r2 x best_health (single ckpt best.pt)": Counter()}
final_clean_cells = []
for c in cells:
    k = (c["ds"], c["seed"], c["purpose"], c["attr"])
    fr2 = da[k]["r2_onehot"]
    bs, be = best_h[(c["ds"], c["seed"], c["purpose"])]
    rules["final_r2 x best_health (paper rule)"][classify(fr2, bs, be)] += 1
    cl = classify(fr2, c["pre_std"], c["pre_er"])
    rules["final_r2 x final_health (single ckpt final.pt)"][cl] += 1
    if cl == "clean":
        final_clean_cells.append([*k, round(fr2, 4), round(c["pre_std"], 3), round(c["pre_er"], 2)])
    rules["best_r2 x best_health (single ckpt best.pt)"][classify(best_r2[k], bs, be)] += 1
out["pcrl_clean_counts_by_rule"] = {k: dict(v) for k, v in rules.items()}
out["pcrl_final_pt_clean_cells"] = final_clean_cells
out["pcrl_final_pt_health_source"] = (
    "splince_benchmark/<ds>_s<seed>/metrics.json pair_results[].pre_health = repr_health of "
    "backbone+LoRA from final.pt (scripts/run_splince_benchmark.py:299-310), test split")
out["pcrl_final_pt_per_dim_std_range"] = [min(c["pre_std"] for c in cells),
                                          max(c["pre_std"] for c in cells)]
out["inputs"] = inputs()
p = dump("D_splince.json", out)
print(json.dumps({k: v for k, v in out.items() if k != "inputs"}, indent=1, default=float))
print("wrote", p)
