"""D6/D7: per-purpose independent ('separate') encoders (NeurIPS App. P '5/20')
and the R-LACE/LEACE diagnostic (results/rlace_diagnostic.json, 7/8 GREEN).
Run: /opt/homebrew/bin/python3 D_sep_rlace.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D_common import dump, git_json, inputs  # noqa: E402

out: dict = {"separate": {}, "rlace": {}}
tot_pass = tot_r2 = tot = 0
for ds in ("adult", "hmda", "diabetes"):
    d = git_json("origin/main", f"results/{ds}_SEPARATE/per_purpose_results.json")
    rows = [(pp["purpose"], a) for pp in d["per_purpose"] for a in pp["attribute_results"]]
    r2 = [a["linear_r2"] for _, a in rows]
    adj = sum(a["adj_pass"] for _, a in rows)
    rec_adj = sum(a["linear_r2"] < 0.05 and a["delta"] < 0.02 for _, a in rows)
    out["separate"][ds] = {
        "seed": d["summary"]["seed"], "n_pairs": len(rows), "adj_pass_stored": adj,
        "adj_pass_recomputed": rec_adj, "r2_pass_lt_0.05": sum(v < 0.05 for v in r2),
        "r2_range": [round(min(r2), 4), round(max(r2), 4)],
        "comparator_pcrl_paper_pass": f'{d["summary"]["pcrl_paper_pass"]}/{d["summary"]["pcrl_paper_total"]}',
        "eff_rank_by_purpose": {pp["purpose"]: round(pp["health"]["effective_rank"], 2)
                                for pp in d["per_purpose"]},
        "task_acc": {pp["purpose"]: pp["task_accuracies"] for pp in d["per_purpose"]},
    }
    tot_pass += adj
    tot_r2 += sum(v < 0.05 for v in r2)
    tot += len(rows)
out["separate"]["total"] = {"adj_pass": f"{tot_pass}/{tot}", "r2_pass": f"{tot_r2}/{tot}",
                            "n_seeds": 1}

r = git_json("origin/main", "results/rlace_diagnostic.json")
P = r["pairs"]
out["rlace"] = {
    "seed": r["seed"], "n_pairs": len(P), "verdicts": r["verdict_counts"],
    "leace_r2_range": [min(p["leace_r2"] for p in P), max(p["leace_r2"] for p in P)],
    "leace_r2_lt_0.05": sum(p["leace_r2"] < 0.05 for p in P),
    "leace_task_drop_pp_range": [round(100 * min(p["baseline_task_acc"] - p["leace_task_acc"] for p in P), 2),
                                 round(100 * max(p["baseline_task_acc"] - p["leace_task_acc"] for p in P), 2)],
    "leace_mlp_delta_range": [min(p["leace_mlp_delta"] for p in P), max(p["leace_mlp_delta"] for p in P)],
    "leace_mlp_delta_lt_0.02": sum(p["leace_mlp_delta"] < 0.02 for p in P),
    "rlace_r8_r2_lt_0.05": sum(p["rlace_r8_r2"] < 0.05 for p in P),
    "red_pairs": [[p["purpose"], p["attr"], p["best_r2"], p["task_drop_pp"]] for p in P if p["verdict"] == "RED"],
    "note": ("'R-LACE' here is iterative logistic null-space projection at rank 1/4/8 "
             "(experiments/rlace_diagnostic.py:117-150), not the Ravfogel et al. 2022 minimax; "
             "the backbone is a TRAINED StandardEncoder (train_backbone, :221-270), "
             "not PCRL's frozen random backbone; LEACE is concept_erasure.LeaceEraser."),
}
out["inputs"] = inputs()
p = dump("D_sep_rlace.json", out)
print(json.dumps({k: v for k, v in out.items() if k != "inputs"}, indent=1, default=float))
print("wrote", p)
