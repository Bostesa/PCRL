"""F1/F2: recount the AAAI '59 of 67 approved configurations fail' audit.

(a) From stored per-config AUCs (git-tracked JSON): the counts at bars 0.52/0.55/0.60 under
    (i) the stored failure rule max(XGB, MLP) on the representation, (ii) adding the LoRA
    attacker where it is stored (expansion rows only), (iii) adding the Tier-2 Gaussian LRT where
    stored. Attacker decomposition. Surface decomposition (what is stored for the output surface).
(b) Independent recomputation from per-config held-out probability arrays on the external drive
    (analysis/tpr59_scores, analysis/tpr_ext_scores): macro OvR AUC mean over probe seeds, max over
    attackers, compared to the stored values; worst-class / worst-pair for the multiclass row with
    explicit class support.

No attacker is fit here: the probabilities were written by the original run (run_tpr_failing59.py,
run_tpr_extension.py) and are only re-scored.
Run: /opt/homebrew/bin/python3 F_audit59_recount.py
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from F_common import (DG, BARS, dg_json, drive_npz, macro_ovr, per_class_ovr, worst_pair, save, sha)

inputs = []
rows_csv = list(csv.DictReader(open(os.path.join(DG, "results/audit_scatter_rows.csv"))))
inputs.append({"location": "durable-guarantees@956f5c8:results/audit_scatter_rows.csv",
               "sha256": sha(os.path.join(DG, "results/audit_scatter_rows.csv"))})
hr, i1 = dg_json("results/honest_reaudit.json")
er, i2 = dg_json("results/expansion_reaudit.json")
pf, i3 = dg_json("results/expansion_reaudit_paperframing.json")
t59, i4 = dg_json("results/tpr_failing59.json")
ext, i5 = dg_json("results/tpr_extension.json")
inputs += [i1, i2, i3, i4, i5]

assert len(rows_csv) == 67
# ---- join per-attacker stored AUCs ---------------------------------------------------------
er_by = {(r["cell"], r["method"]): r for r in er["master"]}
configs = []
for r in hr["master"]:
    configs.append(dict(group="existing", key=f"{r['experiment']}|{r.get('cell','adult/sex')}|{r['method']}",
                        cell=r.get("cell", "adult/sex"), method=r["method"],
                        xgb=r["xgb_auc"], mlp=r["mlp_auc"], lora=None, lrt=None))
for r in pf["rows"]:
    if not r.get("approved_at_rest"):
        continue
    e = er_by[(r["cell"], r["method"])]
    configs.append(dict(group="new", key=f"expansion|{r['cell']}|{r['method']}", cell=r["cell"],
                        method=r["method"], xgb=e["xgb_auc"], mlp=e["mlp_auc"], lora=e["lora_auc"],
                        lrt=e["lrt_auc"], stored_tier1_max=e["tier1_max"], stored_tier2_max=e["tier2_max"]))
assert len(configs) == 67

# check stored tier1_max definition in expansion_reaudit: is it max(XGB,MLP) or includes LoRA?
t1_def = {"equals_max_xgb_mlp": 0, "equals_max_xgb_mlp_lora": 0, "n": 0}
for c in configs:
    if c["group"] == "new":
        t1_def["n"] += 1
        if abs(c["stored_tier1_max"] - max(c["xgb"], c["mlp"])) < 1e-12:
            t1_def["equals_max_xgb_mlp"] += 1
        if abs(c["stored_tier1_max"] - max(c["xgb"], c["mlp"], c["lora"])) < 1e-12:
            t1_def["equals_max_xgb_mlp_lora"] += 1


def count(rule, bar):
    n = 0
    for c in configs:
        vals = [c["xgb"], c["mlp"]]
        if rule in ("t1_with_lora", "t2") and c["lora"] is not None:
            vals.append(c["lora"])
        if rule == "t2" and c["lrt"] is not None:
            vals.append(c["lrt"])
        n += int(max(vals) > bar)
    return n


stored_counts = {f"{b:.2f}": {"xgb_mlp (paper rule)": count("xgb_mlp", b),
                              "t1_with_lora_where_stored": count("t1_with_lora", b),
                              "t2_with_lora_lrt_where_stored": count("t2", b)} for b in BARS}
csv_fail = sum(int(r["fails_bar"]) for r in rows_csv)

# attacker decomposition at 0.55
dec = {"xgb_only": 0, "mlp_only": 0, "both": 0, "neither": 0}
lora_flips = []
for c in configs:
    x, m = c["xgb"] > 0.55, c["mlp"] > 0.55
    dec["both" if x and m else "xgb_only" if x else "mlp_only" if m else "neither"] += 1
    if not (x or m) and c["lora"] is not None and c["lora"] > 0.55:
        lora_flips.append([c["key"], c["lora"]])
mech = {}
for c, rc in zip(configs, rows_csv):
    k = rc["mechanism"]
    mech.setdefault(k, {"n": 0, "fail@0.55": 0})
    mech[k]["n"] += 1
    mech[k]["fail@0.55"] += int(max(c["xgb"], c["mlp"]) > 0.55)
by_dataset = {}
for c in configs:
    d = c["cell"].split("/")[0]
    by_dataset.setdefault(d, [0, 0])
    by_dataset[d][0] += 1
    by_dataset[d][1] += int(max(c["xgb"], c["mlp"]) > 0.55)

# margins: how close to the bar are the decisions?
margins = sorted([(round(max(c["xgb"], c["mlp"]), 4), c["key"]) for c in configs
                  if 0.50 <= max(c["xgb"], c["mlp"]) <= 0.62])

# ---- independent recomputation from the stored probability arrays --------------------------
def slug_for(c):
    if c["group"] == "existing":
        if c["cell"] == "hmda/race":
            return "tpr59_scores/hmda_noise_s8.npz"
        meth = c["method"]
        if "noise" in meth:
            s = float(meth.replace("σ", "sigma").split("=")[-1])
            if s in (4.0, 8.0):
                return f"tpr_ext_scores/hr_noise_s{s:g}.npz"
            return f"tpr59_scores/adult_noise_s{s:g}.npz"
        fam, r = meth.rsplit(" r=", 1)
        return f"tpr59_scores/adult_{'mmd' if 'MMD' in fam else 'hsic'}_r{r}.npz"
    slug = c["cell"].replace("/", "_") + "_" + c["method"].replace(" ", "").replace("σ=", "s").replace("=", "")
    p59 = f"tpr59_scores/exp59_{slug}.npz"
    from F_common import DRIVE
    if os.path.exists(os.path.join(DRIVE, "analysis", p59)):
        return p59
    return f"tpr_ext_scores/exp_{c['cell'].replace('/', '_')}.npz"


recomp, score_inputs, missing = [], {}, []
for c in configs:
    rel = "analysis/" + slug_for(c)
    try:
        z, inp = drive_npz(rel)
    except FileNotFoundError:
        missing.append([c["key"], rel]); continue
    score_inputs[inp["location"]] = inp["sha256"]
    keys = z.files
    res = {}
    for arch in ("XGB", "MLP", "LoRA", "LRT"):
        pk = sorted(k for k in keys if k.startswith(arch + "_") and k.endswith("_prob"))
        if not pk:
            continue
        aucs, wc, wp = [], [], []
        for k in pk:
            tag = k[len(arch) + 1:-len("_prob")]
            y = z["y_" + tag]
            p = z[k].astype(np.float64)
            aucs.append(macro_ovr(y, p))
            if p.shape[1] > 2:
                pc = per_class_ovr(y, p)
                wc.append(max(v for v in pc.values() if v is not None))
                wp.append(worst_pair(y, p)[0])
        res[arch] = {"macro_mean": float(np.mean(aucs)), "n_draws": len(aucs)}
        if wc:
            res[arch]["worst_class_ovr_mean"] = float(np.mean(wc))
            res[arch]["worst_pair_orientfree_mean"] = float(np.mean(wp))
    t1_xm = max(res[a]["macro_mean"] for a in ("XGB", "MLP"))
    rec = {"key": c["key"], "file": rel, "recomputed": res, "recomputed_max_xgb_mlp": t1_xm,
           "stored_max_xgb_mlp": max(c["xgb"], c["mlp"]),
           "delta": t1_xm - max(c["xgb"], c["mlp"])}
    if "LoRA" in res:
        rec["recomputed_max_with_lora"] = max(t1_xm, res["LoRA"]["macro_mean"])
    if "LRT" in res:
        rec["recomputed_lrt"] = res["LRT"]["macro_mean"]
    recomp.append(rec)

recount_counts = {f"{b:.2f}": sum(int(r["recomputed_max_xgb_mlp"] > b) for r in recomp) for b in BARS}
max_abs_delta = max(abs(r["delta"]) for r in recomp)

# multiclass row: class support in the scored folds
mc = {}
z, _ = drive_npz("analysis/tpr59_scores/hmda_noise_s8.npz")
for s in (0, 1, 2):
    y = z[f"y_ps{s}"]
    p = z[f"XGB_ps{s}_prob"].astype(np.float64)
    mc[f"ps{s}"] = {"test_fold_class_counts": np.bincount(y, minlength=5).tolist(),
                    "per_class_ovr_xgb": per_class_ovr(y, p),
                    "worst_pair_all_classes_xgb": worst_pair(y, p),
                    "worst_pair_supported_0_1_2_xgb": worst_pair(y, p, keep=[0, 1, 2])}

mc_means = {}
for arch in ("XGB", "MLP"):
    ys = [z[f"y_ps{s}"] for s in (0, 1, 2)]
    ps = [z[f"{arch}_ps{s}_prob"].astype(np.float64) for s in (0, 1, 2)]
    mc_means[arch] = {"macro": float(np.mean([macro_ovr(y, q) for y, q in zip(ys, ps)])),
                      "worst_pair_all": float(np.mean([worst_pair(y, q)[0] for y, q in zip(ys, ps)])),
                      "worst_pair_supported_012": float(np.mean([worst_pair(y, q, keep=[0, 1, 2])[0] for y, q in zip(ys, ps)]))}
mc["means_over_probe_seeds"] = mc_means

out = {
    "inputs": inputs + [{"location": k, "sha256": v} for k, v in sorted(score_inputs.items())],
    "n_configs": len(configs), "csv_fails_bar_sum": csv_fail,
    "stored_tier1_max_definition_in_expansion_reaudit": t1_def,
    "stored_counts_fail_by_bar": stored_counts,
    "attacker_decomposition_at_0.55_xgb_mlp": dec,
    "configs_passing_xgb_mlp_but_failing_lora_at_0.55": lora_flips,
    "by_mechanism_at_0.55": mech, "by_dataset_[n,fail]_at_0.55": by_dataset,
    "near_bar_configs_max_xgb_mlp_in_[0.50,0.62]": margins,
    "surface_decomposition": {
        "representation_surface_scored": 67, "output_surface_scored": 0,
        "note": "honest_reaudit.json master stores xgb_auc/mlp_auc on the representation only; "
                "expansion_reaudit.json master stores xgb/mlp/lora/lrt on the representation plus task_lift; "
                "no out_* field exists for any of the 67 approved configurations. The paper's audit question "
                "('on either exposed surface', paper.tex:306-308) is therefore answered on one surface. "
                "Needed per-config file for the output surface: an output-logit battery per audit config "
                "(e.g. an 'out_xgb/out_mlp/out_lora' field in expansion_reaudit.json master and honest_reaudit.json master); "
                "none located in git, the drive tree, or analysis/."},
    "independent_recount_from_scores": {
        "n_configs_rescored": len(recomp), "missing": missing,
        "fail_counts_by_bar_max_xgb_mlp": recount_counts,
        "max_abs_delta_vs_stored": max_abs_delta,
        "rows": recomp},
    "multiclass_row_hmda_race_noise_s8": mc,
}
p = save("F_audit59_recount.json", out)
print(p)
print("stored counts", stored_counts)
print("tier1 def", t1_def, "dec", dec, "lora flips", lora_flips)
print("recount", recount_counts, "max delta", max_abs_delta, "missing", missing)
print("mech", mech, by_dataset)
print("near bar", margins)
print(mc)
