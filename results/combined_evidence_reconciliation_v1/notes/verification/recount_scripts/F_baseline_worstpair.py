"""F5: worst-class / worst-pair for the published-method rows on the HMDA/race cells, from the
held-out probability arrays that run_tpr_extension.py stored (drive: analysis/tpr_ext_scores/bg_*,
ob_*, lo_*). multiclass_dual_report.json lists these methods as 'not_available'; the arrays exist on
the drive. Only re-scoring; no attacker is fit. FARE has no stored probability arrays.
"""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from F_common import BARS, DRIVE, drive_npz, macro_ovr, per_class_ovr, worst_pair, save

files = sorted(glob.glob(os.path.join(DRIVE, "analysis/tpr_ext_scores/*.npz")))
inputs, rows = [], []
for fp in files:
    name = os.path.basename(fp)[:-4]
    z, inp = drive_npz("analysis/tpr_ext_scores/" + name + ".npz")
    yk = [k for k in z.files if k.startswith("y_")]
    if z[yk[0]].max() < 2:   # binary attribute: worst-pair == AUC, skip
        continue
    inputs.append(inp)
    agg = {}
    for k in z.files:
        m = re.match(r"(XGB|MLP|LoRA|LRT)_(ts\d_ps\d)_prob", k)
        if not m:
            continue
        arch, tag = m.groups()
        y = z["y_" + tag]; p = z[k].astype(np.float64)
        d = agg.setdefault(arch, {"macro": [], "worst_class": [], "pair_all": [], "pair_sup": []})
        d["macro"].append(macro_ovr(y, p))
        d["worst_class"].append(max(v for v in per_class_ovr(y, p).values() if v is not None))
        d["pair_all"].append(worst_pair(y, p)[0])
        d["pair_sup"].append(worst_pair(y, p, keep=[0, 1, 2])[0])
    r = {"file": name}
    for crit in ("macro", "worst_class", "pair_all", "pair_sup"):
        t1 = {a: float(np.mean(v[crit])) for a, v in agg.items() if a != "LRT"}
        r[f"{crit}_tier1_paper"] = max(t1.values())
        if "LRT" in agg:
            r[f"{crit}_tier2_paper"] = max(r[f"{crit}_tier1_paper"], float(np.mean(agg["LRT"][crit])))
    r["n_draws"] = {a: len(v["macro"]) for a, v in agg.items()}
    rows.append(r)
save("F_baseline_worstpair.json", {"inputs": inputs, "rows": rows,
                                   "fare": "no per-class probability arrays stored for FARE certified points "
                                           "(analysis/fare_cells holds embeddings/manifests; rescoring would need attacker refits)"})
for r in rows:
    print(r["file"], {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
