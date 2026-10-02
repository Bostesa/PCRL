"""D8: LAFTR side of the NeurIPS §5.5 cross-purpose comparison (29/33 absolute,
16/33 incremental) recounted from the stored LAFTR aggregate rows.
Run: /opt/homebrew/bin/python3 D_laftr_crosspurpose.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from D_common import dump, git_json, inputs  # noqa: E402

d = git_json("origin/main", "results/cross_purpose_laftr/cross_purpose_laftr_results.json")
rows = [r for ds in d["laftr"]["aggregate"].values() for r in ds]
out = {"n": len(rows),
       "absolute_concat_minus_majority_gt_1pp": sum(r["concat_delta_mean_pp"] > 1 for r in rows),
       "incremental_concat_minus_best_single_gt_1pp": sum(r["concat_gain_over_best_single_pp_mean"] > 1 for r in rows),
       "by_dataset_abs_incr": {ds: [sum(r["concat_delta_mean_pp"] > 1 for r in v),
                                    sum(r["concat_gain_over_best_single_pp_mean"] > 1 for r in v), len(v)]
                               for ds, v in d["laftr"]["aggregate"].items()},
       "worst_abs": max((r["concat_delta_mean_pp"], r["dataset"], r["attribute"], r["arch"]) for r in rows),
       "note": "aggregate rows only (mean over 3 seeds); LAFTR HMDA/Diabetes encoders/reps not located",
       "inputs": inputs()}
dump("D_laftr_crosspurpose.json", out)
print(json.dumps({k: v for k, v in out.items() if k != "inputs"}, indent=1))
