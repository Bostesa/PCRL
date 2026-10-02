#!/usr/bin/env python3
"""Check B - cross-purpose concatenation attack: absolute vs incremental criteria, per model (standalone).

Criteria (both on (dataset, attribute, auditor-arch) triples; 33 = 11 (dataset, attribute) pairs x {LR, MLP, XGB}):
  ABS  (absolute leakage)      : mean over 3 PCRL seeds of (concat_acc - majority_test) * 100  > 1 pp
  INCR (incremental composition): mean over 3 PCRL seeds of (concat_acc - max_p single_acc_p) * 100 > 1 pp
  where every acc = max over auditor seeds {11,22,33} of TEST accuracy (auditors fit on encoder train split).
Models:
  M0 'submission'  : original PCRL R5/R7 (Adult R5, HMDA R5, Diabetes R7) final.pt; per-seed files
                     origin/main:results/v2_cross_purpose/raw/<ds>_s<k>.json (checkpoint field recorded).
  M1 'rebuttal'    : erase-layer union-LEACE architecture + training-time h_concat linear-R2 constraint
                     (tau_cross=0.10), 200 epochs, best.pt (cross-purpose-rebuttal-2026-05-18);
                     per-seed rows in d39211214:results/rebuttal/cross_purpose/unified_protocol_results.json;
                     absolute-only per-seed files also local (untracked) results/v2_<ds>_CROSS_PURPOSE_*/results.json.
  LAFTR            : origin/main:results/cross_purpose_laftr/cross_purpose_laftr_results.json aggregate only (ABS);
                     INCR only as a recorded count in DUAL_CRITERIA.json.
Usage: python3 B_cross_purpose.py > ../recount_outputs/B_cross_purpose.json
"""
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = "/Users/nathansamson/PCRL"
INPUTS = []


def show(ref, path):
    raw = subprocess.check_output(["git", "-C", REPO, "show", f"{ref}:{path}"])
    INPUTS.append({"location": f"Bostesa/PCRL@{ref}:{path}", "sha256": hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


def local(path):
    raw = Path(REPO, path).read_bytes()
    INPUTS.append({"location": f"local-untracked:{REPO}/{path}", "sha256": hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


def mean(x):
    return sum(x) / len(x)


out = {}
# ---- M0 submission, per-seed raw
cells = defaultdict(lambda: {"abs": [], "incr": [], "single_abs": [], "ckpt": []})
for ds in ["adult", "hmda", "diabetes"]:
    for s in range(3):
        d = show("origin/main", f"results/v2_cross_purpose/raw/{ds}_s{s}.json")
        for r in d["rows"]:
            c = cells[(ds, r["attribute"], r["arch"])]
            c["abs"].append(100 * (r["concat_acc"] - r["majority"]))
            c["incr"].append(100 * (r["concat_acc"] - r["best_single_acc"]))
            c["single_abs"].append(100 * (r["best_single_acc"] - r["majority"]))
            c["ckpt"].append(d["checkpoint"])
m0 = {f"{k[0]}/{k[1]}/{k[2]}": {"abs_pp": round(mean(v["abs"]), 3), "incr_pp": round(mean(v["incr"]), 3),
                               "best_single_abs_pp": round(mean(v["single_abs"]), 3), "n_seeds": len(v["abs"])}
      for k, v in sorted(cells.items())}
ckpts = sorted({c for v in cells.values() for c in v["ckpt"]})
agg = show("origin/main", "results/v2_cross_purpose/aggregate.json")
agg_rows = {f"{r['dataset']}/{r['attribute']}/{r['arch']}": r for ds in agg for r in agg[ds]}
xd = max(max(abs(agg_rows[k]["concat_delta_mean_pp"] - v["abs_pp"]), abs(agg_rows[k]["gain_mean_pp"] - v["incr_pp"]))
         for k, v in m0.items())
out["M0_submission_original_PCRL_finalpt"] = {
    "checkpoints": ckpts,
    "n_triples": len(m0),
    "ABS_gt_1pp": sum(v["abs_pp"] > 1 for v in m0.values()),
    "INCR_gt_1pp": sum(v["incr_pp"] > 1 for v in m0.values()),
    "single_recipient_best_single_minus_majority_gt_1pp": sum(v["best_single_abs_pp"] > 1 for v in m0.values()),
    "aggregate_verdict_FLAG_count(script built-in = INCR)": sum(r["verdict"] == "FLAG" for r in agg_rows.values()),
    "max_abs_diff_vs_aggregate_json_pp": xd,
    "per_dataset": {ds: {"ABS": sum(v["abs_pp"] > 1 for k, v in m0.items() if k.startswith(ds + "/")),
                         "INCR": sum(v["incr_pp"] > 1 for k, v in m0.items() if k.startswith(ds + "/")),
                         "n": sum(k.startswith(ds + "/") for k in m0)} for ds in ["adult", "hmda", "diabetes"]},
    "cells": m0,
}
# ---- M1 rebuttal model, unified-protocol per-seed rows
u = show("origin/cross-purpose-rebuttal-2026-05-18", "results/rebuttal/cross_purpose/unified_protocol_results.json")
cells1 = defaultdict(lambda: {"abs": [], "incr": [], "single_abs": []})
for ds, v in u["per_seed"].items():
    for sd in v["per_seed"]:
        for r in sd["rows"]:
            c = cells1[(ds, r["attribute"], r["arch"])]
            c["abs"].append(100 * (r["concat_acc"] - r["majority"]))
            c["incr"].append(100 * (r["concat_acc"] - r["best_single_acc"]))
            c["single_abs"].append(100 * (r["best_single_acc"] - r["majority"]))
m1 = {f"{k[0]}/{k[1]}/{k[2]}": {"abs_pp": round(mean(v["abs"]), 3), "incr_pp": round(mean(v["incr"]), 3),
                               "best_single_abs_pp": round(mean(v["single_abs"]), 3), "n_seeds": len(v["abs"])}
      for k, v in sorted(cells1.items())}
agg_flags = sum(r["flag_above_1pp"] for ds in u["aggregate"].values() for r in ds.values())
# local absolute-only per-seed files for the same model
loc = {}
for ds, tag in [("adult", "CROSS_PURPOSE_AB"), ("hmda", "CROSS_PURPOSE_AB"), ("diabetes", "CROSS_PURPOSE_DIABETES")]:
    try:
        d = local(f"results/v2_{ds}_{tag}/results.json")
    except FileNotFoundError:
        continue
    acc = defaultdict(list)
    for sd in d["per_seed"]:
        for arch, attrs in sd["attack"].items():
            for a, r in attrs.items():
                acc[(ds, a, arch)].append(r["delta_pp"])
    loc.update({f"{k[0]}/{k[1]}/{k[2]}": round(mean(v), 3) for k, v in acc.items()})
    loc[f"_ckpt_{ds}"] = sorted({sd["checkpoint"] for sd in d["per_seed"]})
common = [k for k in m1 if k in loc]
out["M1_rebuttal_eraselayer_unionLEACE_crosspurp_constraint_bestpt"] = {
    "n_triples": len(m1),
    "ABS_gt_1pp": sum(v["abs_pp"] > 1 for v in m1.values()),
    "INCR_gt_1pp": sum(v["incr_pp"] > 1 for v in m1.values()),
    "INCR_recorded_aggregate_flags": agg_flags,
    "single_recipient_best_single_minus_majority_gt_1pp": sum(v["best_single_abs_pp"] > 1 for v in m1.values()),
    "local_results_json_ABS_gt_1pp": sum(loc[k] > 1 for k in loc if not k.startswith("_")),
    "local_results_json_checkpoints": {k: v for k, v in loc.items() if k.startswith("_")},
    "max_abs_diff_unified_vs_local_abs_pp": max(abs(m1[k]["abs_pp"] - loc[k]) for k in common) if common else None,
    "per_dataset": {ds: {"ABS": sum(v["abs_pp"] > 1 for k, v in m1.items() if k.startswith(ds + "/")),
                         "INCR": sum(v["incr_pp"] > 1 for k, v in m1.items() if k.startswith(ds + "/"))}
                    for ds in ["adult", "hmda", "diabetes"]},
    "cells": m1,
}
# ---- paired comparison M0 vs M1
nl = [k for k in m0 if not k.endswith("/LR")]
out["M0_vs_M1"] = {
    "ABS_rose_on_nonlinear_cells": sum(m1[k]["abs_pp"] > m0[k]["abs_pp"] for k in nl),
    "n_nonlinear": len(nl),
    "mean_ABS_nonlinear_M0": round(mean([m0[k]["abs_pp"] for k in nl]), 2),
    "mean_ABS_nonlinear_M1": round(mean([m1[k]["abs_pp"] for k in nl]), 2),
    "LR_ABS_M1_all_zero": all(abs(m1[k]["abs_pp"]) < 1e-9 for k in m1 if k.endswith("/LR")),
    "INCR_flips_flag_to_ok": sum(m0[k]["incr_pp"] > 1 and m1[k]["incr_pp"] <= 1 for k in m0),
    "INCR_flips_ok_to_flag": sum(m0[k]["incr_pp"] <= 1 and m1[k]["incr_pp"] > 1 for k in m0),
}
# ---- LAFTR (aggregate only)
L = show("origin/main", "results/cross_purpose_laftr/cross_purpose_laftr_results.json")
lab = [100 * (r["concat_acc_mean"] - r["majority"]) for ds in L["laftr"]["aggregate"].values() for r in ds]
dual = show("origin/main", "results/cross_purpose_laftr/DUAL_CRITERIA.json")
out["LAFTR"] = {"ABS_gt_1pp_from_aggregate": sum(x > 1 for x in lab), "n": len(lab),
                "ABS_recorded": L["laftr"]["flagged_above_1pp"],
                "INCR_recorded_DUAL_CRITERIA": dual["criterion_B"]["laftr"]["flagged"],
                "INCR_note": "no per-seed or best-single fields located in the LAFTR aggregate; INCR count not recomputable"}
out["inputs"] = INPUTS
print(json.dumps(out, indent=1))
