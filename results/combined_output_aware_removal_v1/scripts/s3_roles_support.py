"""Stage 3: new roles (oar-roles-v1) and frozen class support from labels only (no fits, no outcomes)."""
import json, sys
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from oar.study import CELLS, load_world, role_summary, RUN
THR = {"attacker_fit": 100, "attacker_val": 30, "assessment": 100}
out = {}
for ds in ("adult", "hmda"):
    W = load_world(ds)
    c = CELLS[ds]
    sup = {}
    for what, y, K in (("sensitive", W["s"], c["K_s"]), ("task", W["t"], c["K_t"])):
        counts = {r: np.bincount(y[W["idx"][r]], minlength=K).tolist() for r in ("defense_fit", "cert", *THR)}
        ok = [k for k in range(K) if all(counts[r][k] >= THR[r] for r in THR)]
        sup[what] = {"K": K, "counts": counts, "thresholds": THR, "supported_classes": ok,
                     "supported_pairs": [[i, j] for i in ok for j in ok if i < j],
                     "not_estimable_classes": [k for k in range(K) if k not in ok]}
    out[ds] = {"roles": role_summary(W), "support": sup,
               "exposure_groups_removed": int(len(np.unique(W["unit"][W["role"] == "excluded_exposure"]))),
               "assessment_identical_for_all_methods_and_seeds": True}
RUN.mkdir(parents=True, exist_ok=True)
p = WT / "results/combined_output_aware_removal_v1/ROLES_AND_SUPPORT.json"
p.write_text(json.dumps(out, indent=1))
for ds in out:
    print(ds, {r: v["rows"] for r, v in out[ds]["roles"].items() if isinstance(v, dict)}, out[ds]["support"]["sensitive"]["supported_classes"], out[ds]["support"]["sensitive"]["counts"])
