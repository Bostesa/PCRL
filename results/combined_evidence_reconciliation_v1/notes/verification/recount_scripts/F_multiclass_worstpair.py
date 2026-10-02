"""F4: multiclass scoring on the two 5-class HMDA/race headline cells (easy, middle).

Independent re-scoring of stored held-out attacker probabilities (drive: analysis/tpr_scores/*.npz,
written by run_tpr.py; analysis/gate_shards/*.json written by run_gate_shard.py). For each surviving
operating point we compute, per (arch, train seed, probe seed): macro OvR AUC, worst-class OvR AUC
(oriented), worst-pair AUC over all classes (orientation-free, DG convention) and over the supported
classes {0,1,2} (>= 3000 rows in the cell), then aggregate with the paper convention
(max over arch of the mean over seeds) and the strict convention (max over everything).
Unsupported classes are reported with their held-out row counts.
Also: re-aggregates gate_shards against gate_5seed.json (checked against recorded aggregates).
"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from F_common import BARS, DRIVE, drive_npz, dg_json, macro_ovr, per_class_ovr, worst_pair, save, sha

FILES = ["subspace_easy", "subspace_middle", "fullrank_tier1_easy", "fullrank_tier1_middle",
         "fullrank_tier2_easy", "fullrank_tier2_middle", "unprot_easy", "unprot_middle"]
inputs, out = [], {}
for f in FILES:
    z, inp = drive_npz(f"analysis/tpr_scores/{f}.npz")
    inputs.append(inp)
    agg = {}
    counts = None
    for k in z.files:
        m = re.match(r"(rep|out)_(XGB|MLP|LoRA|LRT)_ts(\d)_ps(\d)_prob", k)
        if not m:
            continue
        surf, arch, ts, ps = m.groups()
        y = z[f"y_ps{ps}"]
        p = z[k].astype(np.float64)
        if counts is None:
            counts = np.bincount(y, minlength=p.shape[1]).tolist()
        pc = per_class_ovr(y, p)
        d = agg.setdefault(surf, {}).setdefault(arch, {"macro": [], "worst_class": [], "worst_class_id": [],
                                                       "pair_all": [], "pair_all_arg": [], "pair_sup": []})
        d["macro"].append(macro_ovr(y, p))
        wc = max((v, c) for c, v in pc.items() if v is not None)
        d["worst_class"].append(wc[0]); d["worst_class_id"].append(wc[1])
        a, arg = worst_pair(y, p)
        d["pair_all"].append(a); d["pair_all_arg"].append(arg)
        d["pair_sup"].append(worst_pair(y, p, keep=[0, 1, 2])[0])
    res = {"heldout_class_counts_per_fold": counts,
           "unsupported_classes": {"3": counts[3], "4": counts[4]} if counts else None}
    for surf, archs in agg.items():
        for crit in ("macro", "worst_class", "pair_all", "pair_sup"):
            means = {a: float(np.mean(v[crit])) for a, v in archs.items() if a != "LRT"}
            means_t2 = {a: float(np.mean(v[crit])) for a, v in archs.items()}
            strict = max(max(v[crit]) for a, v in archs.items() if a != "LRT")
            res[f"{surf}_{crit}_tier1_paper"] = max(means.values())
            res[f"{surf}_{crit}_tier1_argmax"] = max(means, key=means.get)
            res[f"{surf}_{crit}_tier1_strict"] = strict
            if "LRT" in archs:
                res[f"{surf}_{crit}_tier2_paper"] = max(means_t2.values())
        res[f"{surf}_worst_class_ids_seen"] = sorted(set(c for v in archs.values() for c in v["worst_class_id"]))
        res[f"{surf}_n_draws_per_arch"] = {a: len(v["macro"]) for a, v in archs.items()}
    for b in BARS:
        res[f"pass_at_{b:.2f}"] = {k: (v <= b) for k, v in res.items()
                                   if isinstance(v, float) and ("tier1_paper" in k or "tier2_paper" in k)}
    out[f] = res

# null: orientation-free max over 10 pairs under no signal is above 0.5 (paper quotes 0.5556 / supported 0.5217)
wn, i_wn = dg_json("results/worstpair_null.json")
inputs.append(i_wn)

# gate shards (5-seed re-certification of the supported-class operating points)
g5, i_g5 = dg_json("results/gate_5seed.json")
inputs.append(i_g5)
gate = {}
for cell in ("easy", "middle"):
    shards = sorted(glob.glob(os.path.join(DRIVE, f"analysis/gate_shards/{cell}_ts*.json")))
    per = {"rep": {}, "out": {}}
    for s in shards:
        inputs.append({"location": "tree-durable-guarantees.tar::durable-guarantees/analysis/gate_shards/"
                       + os.path.basename(s), "sha256": sha(s)})
        d = json.load(open(s))
        for surf in ("rep", "out"):
            for arch, v in d[surf].items():
                for crit in ("macro", "all", "sup"):
                    per[surf].setdefault(crit, {}).setdefault(arch, []).extend(v[crit])
    rec = {}
    for surf in ("rep", "out"):
        for crit, archs in per[surf].items():
            rec[f"{surf}_{crit}_paper"] = max(float(np.mean(v)) for v in archs.values())
            rec[f"{surf}_{crit}_strict"] = max(max(v) for v in archs.values())
    stored = [c for c in g5["cells"] if c["cell"] == cell][0]
    rec["max_abs_delta_vs_gate_5seed"] = max(abs(rec[k] - stored[k]) for k in rec if k in stored)
    rec["n_shards"] = len(shards)
    gate[cell] = rec

summary = {}
for f, r in out.items():
    summary[f] = {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and "tier" in k and "paper" in k}
save("F_multiclass_worstpair.json", {"inputs": inputs, "per_operating_point": out, "summary": summary,
                                     "worstpair_null_stored": wn, "gate_5seed_recount": gate})
for f, s in summary.items():
    print(f, s)
print(out["subspace_easy"]["heldout_class_counts_per_fold"])
print(json.dumps(gate, indent=0)[:1500])
