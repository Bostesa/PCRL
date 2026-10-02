#!/usr/bin/env python3
"""Check A - checkpoint selection for the NeurIPS PCRL headline grid (standalone).

Reads committed result files from Bostesa/PCRL origin/main via `git show` (no repo imports) and counts,
SEPARATELY per checkpoint rule:
  * strict R2 pass (<= 0.05 and < 0.05) under the final-iterate rule (final.pt; dominant_axis_audit.json
    r2_onehot, cross-checked against final_vs_best.json 'final' rows) and under the best-validation /
    Cotter-selector rule (best.pt; per_seed_results.json linear_r2, cross-checked against final_vs_best.json
    'best' rows);
  * "cleanly compliant" (R2 <= 0.05 AND per_dim_std_mean >= 0.5 AND effective_rank >= 2 on the cell's own
    purpose) under each rule. Health metrics are stored ONLY for best.pt (per_seed_results.json
    per_purpose_health; run_v2_dataset.py reloads best.pt before computing them), so the final-rule
    clean count is not computable from stored artifacts; the paper's 7/60 is reproduced as the MIXED rule
    (final.pt R2 x best.pt health) used by scripts/identify_collapse_cells.py.
Headline grid composition = Adult R5 + HMDA R5 + Diabetes R7 (as in identify_collapse_cells.py DATASETS).
Diabetes R5 is reported alongside for attribution.

Usage: python3 A_checkpoint_selection.py [REF] > ../recount_outputs/A_checkpoint_selection.json
"""
import hashlib
import json
import subprocess
import sys

REPO = "/Users/nathansamson/PCRL"
REF = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
TAU = 0.05
STD_MIN, ER_MIN = 0.5, 2.0
INPUTS = []


def show(path):
    raw = subprocess.check_output(["git", "-C", REPO, "show", f"{REF}:{path}"])
    INPUTS.append({"location": f"Bostesa/PCRL@{REF}:{path}", "sha256": hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


def rows_from_per_seed(ps):
    out = {}
    for s in ps["per_seed"]:
        for r in s["attribute_results"]:
            out[(s["seed"], r["purpose"], r["attribute"])] = r["linear_r2"]
    return out


def health_from_per_seed(ps):
    out = {}
    for s in ps["per_seed"]:
        for p, h in s["per_purpose_health"].items():
            out[(s["seed"], p)] = (h["per_dim_std_mean"], h["effective_rank"])
    return out


def rows_from_da(da):
    out = {}
    for seed, sd in da["per_seed"].items():
        for r in sd["rows"]:
            out[(int(seed), r["purpose"], r["attribute"])] = r["r2_onehot"]
    return out, {int(k): v["epoch"] for k, v in da["per_seed"].items()}


def rows_from_fvb(fvb, which):
    out = {}
    for seed, sd in fvb["per_seed"].items():
        for r in sd[which]["rows"]:
            out[(int(seed), r["purpose"], r["attribute"])] = r["linear_r2"]
    return out


def count(r2map, op):
    return sum(1 for v in r2map.values() if op(v))


def classify(r2map, health):
    c = {"clean": 0, "collapse": 0, "failed": 0}
    clean_cells = []
    for k, v in sorted(r2map.items()):
        if v > TAU:
            c["failed"] += 1
            continue
        std, er = health[(k[0], k[1])]
        if std >= STD_MIN and er >= ER_MIN:
            c["clean"] += 1
            clean_cells.append(list(k))
        else:
            c["collapse"] += 1
    return c, clean_cells


def maxdiff(a, b):
    ks = set(a) & set(b)
    return {"n_common": len(ks), "n_only_a": len(set(a) - ks), "n_only_b": len(set(b) - ks),
            "max_abs_diff": max(abs(a[k] - b[k]) for k in ks) if ks else None}


GRID = [("adult", "v2_adult_ROUND5", True), ("hmda", "v2_hmda_ROUND5", True),
        ("diabetes", "v2_diabetes_ROUND7", True), ("diabetes_R5", "v2_diabetes_ROUND5", False)]
res = {"ref": REF, "tau": TAU, "health_thresholds": {"per_dim_std_mean_min": STD_MIN, "effective_rank_min": ER_MIN},
       "per_dataset": {}}
tot = {k: 0 for k in ["final_le", "final_lt", "best_le", "best_lt", "n",
                      "mixed_clean", "mixed_collapse", "mixed_failed", "best_clean", "best_collapse", "best_failed"]}
for name, d, in_grid in GRID:
    base = f"results/{d}"
    ps = show(f"{base}/per_seed_results.json")
    best = rows_from_per_seed(ps)
    health = health_from_per_seed(ps)
    entry = {"best_epochs": [s["best_epoch"] for s in ps["per_seed"]],
             "cotter_kind": [s.get("cotter_selection", {}).get("kind") for s in ps["per_seed"]],
             "n_feasible_post_warmup": [s.get("cotter_selection", {}).get("n_feasible_post_warmup") for s in ps["per_seed"]]}
    final = None
    try:
        da = show(f"{base}/dominant_axis_audit.json")
        final, ep = rows_from_da(da)
        entry["final_source"] = "dominant_axis_audit.json r2_onehot (scripts/eval_round4_dominant_axis.py loads final.pt)"
        entry["final_epochs"] = ep
    except subprocess.CalledProcessError:
        pass
    try:
        fvb = show(f"{base}/final_vs_best.json")
        fb, ff = rows_from_fvb(fvb, "best"), rows_from_fvb(fvb, "final")
        entry["xcheck_best_per_seed_vs_fvb_best"] = maxdiff(best, fb)
        if final is not None:
            entry["xcheck_da_final_vs_fvb_final"] = maxdiff(final, ff)
        else:
            final = ff
            entry["final_source"] = "final_vs_best.json final rows (linear_r2)"
    except subprocess.CalledProcessError:
        entry["final_vs_best"] = "absent at ref"
    n = len(best)
    entry["n_cells"] = n
    entry["best_rule"] = {"strict_le": count(best, lambda v: v <= TAU), "strict_lt": count(best, lambda v: v < TAU)}
    bc, bcells = classify(best, health)
    entry["best_rule"].update({"classes_best_r2_x_best_health": bc, "clean_cells": bcells})
    if final is not None:
        entry["final_rule"] = {"strict_le": count(final, lambda v: v <= TAU), "strict_lt": count(final, lambda v: v < TAU),
                               "clean": "not computable: no final.pt health metrics stored"}
        mc, mcells = classify(final, health)
        entry["mixed_rule_final_r2_x_best_health"] = {"classes": mc, "clean_cells": mcells}
        entry["flips_final_vs_best"] = sorted([list(k) + [round(best[k], 4), round(final[k], 4)]
                                               for k in best if (best[k] <= TAU) != (final[k] <= TAU)])
    res["per_dataset"][name] = entry
    if in_grid:
        tot["n"] += n
        tot["final_le"] += entry["final_rule"]["strict_le"]; tot["final_lt"] += entry["final_rule"]["strict_lt"]
        tot["best_le"] += entry["best_rule"]["strict_le"]; tot["best_lt"] += entry["best_rule"]["strict_lt"]
        for k in ["clean", "collapse", "failed"]:
            tot["mixed_" + k] += entry["mixed_rule_final_r2_x_best_health"]["classes"][k]
            tot["best_" + k] += bc[k]
# final.pt health: scripts/run_splince_benchmark.py (origin/main:296-311,355-356) loads final.pt first, eval mode,
# and records repr_health(Z_te) of each purpose BEFORE the SPLINCE projection as 'pre_health'.
final_health = {}
for ds in ["adult", "hmda", "diabetes"]:
    for sd in range(3):
        m = show(f"results/splince_benchmark/{ds}_s{sd}/metrics.json")
        assert m["warm_start_path"].endswith("/final.pt"), m["warm_start_path"]
        for r in m["pair_results"]:
            h = r["pre_health"]
            final_health[(ds, sd, r["purpose"])] = (h["per_dim_std_mean"], h["effective_rank"])
fc = {"clean": 0, "collapse": 0, "failed": 0}
fcells = []
for name, d, in_grid in GRID:
    if not in_grid:
        continue
    da = show(f"results/{d}/dominant_axis_audit.json")
    fr, _ = rows_from_da(da)
    hh = {(k[1], k[2]): v for k, v in final_health.items() if k[0] == name}
    c, cells = classify(fr, hh)
    for k in fc:
        fc[k] += c[k]
    fcells += [[name] + x for x in cells]
    res["per_dataset"][name]["final_rule"]["clean"] = c["clean"]
    res["per_dataset"][name]["final_rule"]["classes_final_r2_x_final_health"] = c
tot["final_clean"], tot["final_collapse"], tot["final_failed"] = fc["clean"], fc["collapse"], fc["failed"]
res["final_rule_clean_cells"] = fcells
res["final_health_source"] = "results/splince_benchmark/<ds>_s<k>/metrics.json pair_results[].pre_health (final.pt, test split, eval mode)"
res["headline_grid_totals"] = tot
res["inputs"] = INPUTS
print(json.dumps(res, indent=1, default=str))
