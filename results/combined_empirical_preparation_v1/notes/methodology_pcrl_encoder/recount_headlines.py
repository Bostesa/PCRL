#!/usr/bin/env python3
"""Standalone recount of PCRL-encoder headline counts from stored results.

Methodology role, combined-empirical-preparation-v1, 2026-10-01.
No repo imports, no training, no cloud.  Reads committed files via
`git -C /Users/nathansamson/PCRL show <ref>:<path>` and local untracked
files by absolute path.  Prints a JSON blob (entries list) to stdout; the
caller (or a human) writes it into recounts.json.

Usage:  python3 recount_headlines.py > /tmp/recount_out.json
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

REPO = "/Users/nathansamson/PCRL"
MAIN = "origin/main"           # 55e4cb1d1
XP = "d39211214"               # cross-purpose-rebuttal-2026-05-18
RE = "rebuttal-evidence"       # 5739d3bb9
LOCAL = Path(REPO) / "results" / "rebuttal"
SOURCES: dict[str, str] = {}   # label -> sha256


def git_bytes(ref: str, path: str) -> bytes:
    b = subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"],
                       check=True, capture_output=True).stdout
    SOURCES[f"{ref}:{path}"] = hashlib.sha256(b).hexdigest()
    return b


def gj(ref: str, path: str):
    return json.loads(git_bytes(ref, path))


def lj(p: Path):
    b = p.read_bytes()
    SOURCES[str(p)] = hashlib.sha256(b).hexdigest()
    return json.loads(b)


ENTRIES: list[dict] = []


def add(**kw):
    ENTRIES.append(kw)


# ---------------------------------------------------------------- (a) 56/60
CANON = {"adult": "v2_adult_ROUND5", "hmda": "v2_hmda_ROUND5",
         "diabetes": "v2_diabetes_ROUND7"}
da_rows = []     # final.pt test-split OLS audit rows
health = {}      # (ds, seed, purpose) -> health dict from per_seed_results (best.pt)
best_r2 = {}     # (ds, seed, purpose, attr) -> linear_r2 from per_seed_results (best.pt)
for ds, d in CANON.items():
    a = gj(MAIN, f"results/{d}/dominant_axis_audit.json")
    for s, blk in a["per_seed"].items():
        for r in blk["rows"]:
            da_rows.append(dict(ds=ds, seed=int(s), epoch=blk["epoch"], **r))
    p = gj(MAIN, f"results/{d}/per_seed_results.json")
    for s in p["per_seed"]:
        for pu, h in s["per_purpose_health"].items():
            health[(ds, s["seed"], pu)] = h
        for r in s["attribute_results"]:
            best_r2[(ds, s["seed"], r["purpose"], r["attribute"])] = r["linear_r2"]

TAU = 0.05


def strict(rows, key="r2_onehot", tau=TAU, op="lt"):
    f = (lambda v: v < tau) if op == "lt" else (lambda v: v <= tau)
    return sum(1 for r in rows if f(r[key]))


per_ds = {ds: [r for r in da_rows if r["ds"] == ds] for ds in CANON}
add(id="a1_strict_56_60",
    claim="56/60 strict R2_onehot<0.05 (Adult 23/24, HMDA 16/18, Diabetes 17/18)",
    manuscript="pdf23 lines 10,53-54,303,349; Table 1 (lines 301-306)",
    sources=[f"{MAIN}:results/{d}/dominant_axis_audit.json" for d in CANON.values()],
    field="per_seed.<s>.rows[].r2_onehot",
    recomputed={ds: f"{strict(v)}/{len(v)}" for ds, v in per_ds.items()} |
               {"total": f"{strict(da_rows)}/{len(da_rows)}",
                "total_le": f"{strict(da_rows, op='le')}/{len(da_rows)}"},
    reported="56/60 (23/24,16/18,17/18)",
    checkpoint="final.pt (epoch 199 of constrained phase; audit script loads checkpoints/<run>/final.pt)",
    split="in-sample OLS fit+score on TEST split representations (LinearComplianceCertificate.check(test_reprs,test_labels))",
    epochs={ds: sorted({r['epoch'] for r in v}) for ds, v in per_ds.items()})

# failures list
fails = [(r["ds"], r["purpose"], r["attribute"], r["seed"], round(r["r2_onehot"], 4))
         for r in da_rows if r["r2_onehot"] >= TAU]

# tau sweep
taus = [0.01, 0.025, 0.05, 0.1, 0.2]
sweep = {str(t): {"strict_lt": strict(da_rows, tau=t), "strict_le": strict(da_rows, tau=t, op="le"),
                  "da_lt": strict(da_rows, key="r2_da", tau=t),
                  "da_le": strict(da_rows, key="r2_da", tau=t, op="le")} for t in taus}
add(id="a2_tau_sweep", claim="tau sweep strict 37/52/56/59/60; DA 31/49/55/56/59",
    manuscript="pdf23 lines 303-305; Table 5 (lines 997-1005)",
    sources=[f"{MAIN}:results/tier1_analyses/tau_sensitivity.json"] +
            [f"{MAIN}:results/{d}/dominant_axis_audit.json" for d in CANON.values()],
    field="r2_onehot / r2_da",
    recomputed=sweep, reported={"0.01": "37/31", "0.025": "52/49", "0.05": "56/55",
                                "0.1": "59/56", "0.2": "60/59"},
    checkpoint="final.pt", split="test in-sample OLS")
ts = gj(MAIN, "results/tier1_analyses/tau_sensitivity.json")

# hidden-by-onehot
hidden = [(r["ds"], r["purpose"], r["attribute"], r["seed"], round(r["r2_onehot"], 4), round(r["r2_da"], 4))
          for r in da_rows if r["r2_onehot"] < TAU <= r["r2_da"]]
add(id="a3_hidden", claim="1/60 hidden by one-hot (HMDA underwriting/race s1: 0.027 vs 0.288)",
    manuscript="pdf23 lines 13-14, 259-262, Table 1",
    sources=[f"{MAIN}:results/v2_hmda_ROUND5/dominant_axis_audit.json"],
    field="r2_onehot<0.05<=r2_da", recomputed=hidden, reported="1/60",
    checkpoint="final.pt", split="test in-sample OLS")

# amplification ratios (multi-class rows)
mc = [r for r in da_rows if r["is_multiclass"]]
ratios = sorted((r["r2_da"] / r["r2_onehot"], r["ds"], r["purpose"], r["attribute"], r["seed"],
                 r["r2_onehot"], r["r2_da"]) for r in mc if r["r2_onehot"] > 0)
resid = max(abs(r["convex_combo_residual"]) for r in mc)
# independent identity check from per_class_r2 + priors
indep = []
for r in mc:
    w = [p * (1 - p) for p in r["priors"]]
    pred = sum(wi * ri for wi, ri in zip(w, r["per_class_r2"])) / sum(w)
    indep.append(abs(pred - r["r2_onehot"]))
add(id="a4_amplification", claim="median R2_DA/R2_onehot 1.43x, worst 12.7x (HMDA underwriting/race 12.7,10.7,3.2); identity holds 33/33 within 0.01, max residual 0.0021",
    manuscript="pdf23 lines 60, 278-280; Appendix J lines 590-591",
    sources=[f"{MAIN}:results/{d}/dominant_axis_audit.json" for d in CANON.values()],
    field="r2_da/r2_onehot; convex_combo_residual; per_class_r2,priors",
    recomputed={"n_multiclass": len(mc), "median_ratio": round(statistics.median([x[0] for x in ratios]), 3),
                "max_ratio": [round(ratios[-1][0], 2), *ratios[-1][1:]],
                "hmda_underwriting_race": [(s, round(r["r2_da"] / r["r2_onehot"], 2), round(r["r2_onehot"], 4), round(r["r2_da"], 4))
                                           for r in mc for s in [r["seed"]]
                                           if r["ds"] == "hmda" and r["purpose"] == "underwriting" and r["attribute"] == "race"],
                "stored_max_residual": resid, "independent_max_residual": max(indep),
                "n_within_0.01": sum(1 for x in indep if x < 0.01)},
    reported="1.43x median, 12.7x worst, 33/33, 0.0021",
    checkpoint="final.pt", split="test in-sample OLS",
    note="12.7x worst case is s0 where r2_onehot=0.0026 and r2_da=0.0332: both below tau; ratio of two sub-threshold numbers")

# cleanly-compliant
def clean_counts(r2_of):
    out = {"clean": 0, "collapse": 0, "failed": 0}
    per_dp = {}
    for r in da_rows:
        h = health[(r["ds"], r["seed"], r["purpose"])]
        r2 = r2_of(r)
        k = (r["ds"], r["purpose"])
        per_dp.setdefault(k, {"clean": 0, "collapse": 0, "failed": 0})
        if r2 >= TAU:
            cat = "failed"
        elif h["per_dim_std_mean"] >= 0.5 and h["effective_rank"] >= 2.0:
            cat = "clean"
        else:
            cat = "collapse"
        out[cat] += 1
        per_dp[k][cat] += 1
    return out, {f"{a}/{b}": v for (a, b), v in per_dp.items()}


cf, cf_dp = clean_counts(lambda r: r["r2_onehot"])
cb, cb_dp = clean_counts(lambda r: best_r2[(r["ds"], r["seed"], r["purpose"], r["attribute"])])
cb_le = sum(1 for k, v in best_r2.items() if v <= TAU)
clean_cells = [(r["ds"], r["purpose"], r["attribute"], r["seed"]) for r in da_rows
               if r["r2_onehot"] < TAU and health[(r["ds"], r["seed"], r["purpose"])]["per_dim_std_mean"] >= 0.5
               and health[(r["ds"], r["seed"], r["purpose"])]["effective_rank"] >= 2.0]
add(id="a5_clean_7_60", claim="7/60 cleanly compliant, 49 collapse-compliant, 4 failed; Table 6 per-(dataset,purpose)",
    manuscript="pdf23 lines 11,56-57,220-228, Table 1, Table 6 (lines 1017-1031)",
    sources=[f"{MAIN}:results/{d}/per_seed_results.json" for d in CANON.values()] +
            [f"{MAIN}:results/{d}/dominant_axis_audit.json" for d in CANON.values()],
    field="R2 from dominant_axis_audit r2_onehot (final.pt) x per_seed[].per_purpose_health.{per_dim_std_mean,effective_rank} (best.pt-reloaded eval)",
    recomputed={"final_r2_x_besthealth": cf, "per_dataset_purpose": cf_dp, "clean_cells": clean_cells,
                "best_r2_x_besthealth": cb, "best_r2_strict_le_0.05": f"{cb_le}/60"},
    reported="7 clean / 49 collapse / 4 failed",
    checkpoint="MIXED: R2 from final.pt; health metrics from per_seed_results.json, which run_v2_dataset.py computes after reloading canonical_iterate.pt>best.pt>final.pt (best.pt exists for every canonical run per relocated checkpoint inventory)",
    split="health on test-split representations")

# per-cell table check vs manuscript Tables 7-9
ms_path = Path("/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/mpe/manuscript_final_pdf23.txt")
mism = []
nchk = 0
if ms_path.exists():
    import re
    lines = ms_path.read_text().splitlines()
    dsname = None
    for i, ln in enumerate(lines):
        if "Table 7: Per-pair" in ln: dsname = "adult"
        if "Table 8: Per-pair" in ln: dsname = "hmda"
        if "Table 9: Per-pair" in ln: dsname = "diabetes"
        if "Table 10:" in ln: dsname = None
        m = re.match(r"^\s*([a-z_]+)\s+([a-z_]+)\s+(\d)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+(\S+)\s*$", ln)
        if dsname and m:
            pu, at, s, r1, r2, am, dl = m.groups()
            for r in da_rows:
                if r["ds"] == dsname and r["purpose"] == pu and r["attribute"] == at and r["seed"] == int(s):
                    nchk += 1
                    ok = (abs(round(r["r2_onehot"], 4) - float(r1)) < 1.5e-4 and abs(round(r["r2_da"], 4) - float(r2)) < 1.5e-4
                          and int(am) == r["r2_da_argmax_class"])
                    if not ok:
                        mism.append((dsname, pu, at, s, r1, r2, am, round(r["r2_onehot"], 4), round(r["r2_da"], 4), r["r2_da_argmax_class"]))
add(id="a6_tables_7_9", claim="Per-cell R2_onehot / R2_DA / argmax in Tables 7-9",
    manuscript="pdf23 Tables 7-9 (lines 1041-1124)", sources=[f"{MAIN}:results/{d}/dominant_axis_audit.json" for d in CANON.values()],
    field="r2_onehot, r2_da, r2_da_argmax_class", recomputed={"cells_checked": nchk, "mismatches": mism},
    reported="table values", checkpoint="final.pt", split="test in-sample OLS")

# ---------------------------------------------------------------- (b) R4 -> R5/R7
r4 = {"adult": "v2_adult_ROUND4", "hmda": "v2_hmda_ROUND4", "diabetes": "v2_diabetes_ROUND4"}
fvb = {}
for ds, d in list(r4.items()) + [("adult_R5", "v2_adult_ROUND5"), ("hmda_R5", "v2_hmda_ROUND5"), ("diabetes_R5", "v2_diabetes_ROUND5")]:
    j = gj(MAIN, f"results/{d}/final_vs_best.json")
    out = {}
    for tag in ("best", "final"):
        rows = [r for s in j["per_seed"].values() for r in s[tag]["rows"]]
        out[tag] = {"strict_lt": f"{sum(r['linear_r2'] < TAU for r in rows)}/{len(rows)}",
                    "mean_r2": round(sum(r["linear_r2"] for r in rows) / len(rows), 4),
                    "adj_pass": f"{sum(r['adj_pass'] for r in rows)}/{len(rows)}",
                    "best_epochs": sorted({s[tag].get('best_epoch') for s in j['per_seed'].values()})}
    fvb[ds] = out
r4_total_final = sum(int(fvb[k]["final"]["strict_lt"].split("/")[0]) for k in r4)
r4_total_best = sum(int(fvb[k]["best"]["strict_lt"].split("/")[0]) for k in r4)
d7 = gj(MAIN, "results/v2_diabetes_ROUND7/per_seed_results.json")
d7_best = [r["linear_r2"] for s in d7["per_seed"] for r in s["attribute_results"]]
add(id="b1_round4_46_60", claim="Round 4 46/60 strict -> 56/60 post-fix; mean R2 Adult .038->.012, HMDA .045->.020, Diabetes .025->.009",
    manuscript="pdf23 lines 248-249, 616-618 (Appendix K)",
    sources=[f"{MAIN}:results/{d}/final_vs_best.json" for d in list(r4.values()) + ["v2_adult_ROUND5", "v2_hmda_ROUND5", "v2_diabetes_ROUND5"]] +
            [f"{MAIN}:results/v2_diabetes_ROUND7/per_seed_results.json"],
    field="per_seed.<s>.{best,final}.rows[].linear_r2 (generate_report on reloaded checkpoint)",
    recomputed={"per_dataset": fvb, "R4_total_final": f"{r4_total_final}/60", "R4_total_best": f"{r4_total_best}/60",
                "diabetes_R7_bestpt_mean": round(sum(d7_best) / len(d7_best), 4),
                "diabetes_R7_bestpt_strict": f"{sum(v < TAU for v in d7_best)}/{len(d7_best)}",
                "diabetes_R7_finalpt_mean_from_DA_audit": round(sum(r['r2_onehot'] for r in per_ds['diabetes']) / 18, 4),
                "adult_R5_final_mean_DA": round(sum(r['r2_onehot'] for r in per_ds['adult']) / 24, 4),
                "hmda_R5_final_mean_DA": round(sum(r['r2_onehot'] for r in per_ds['hmda']) / 18, 4)},
    reported="46/60 -> 56/60; .038/.045/.025 -> .012/.020/.009",
    checkpoint="R4: final.pt adopted post hoc after Cotter fallback selected epoch ~12 on Adult; R5 final_vs_best.md files are titled 'Round 4' (template alias) but hold R5 numbers",
    note="Diabetes 56/60 component is Round 7 (per-class OvR + rank 24), i.e. two further changes beyond the lambda-floor + warmup-skip 'combined modification' credited in Appendix K")

# ---------------------------------------------------------------- (c) erase-layer pilot
pilot = {}
for ds in ("adult", "hmda", "diabetes"):
    p = lj(LOCAL / "erase_layer_pilot_aws" / f"v2_{ds}_ERASE_PILOT" / "per_seed_results.json")
    rows = [(s["seed"], r) for s in p["per_seed"] for r in s["attribute_results"]]
    hl = {(s["seed"], pu): h for s in p["per_seed"] for pu, h in s["per_purpose_health"].items()}
    st = sum(r["linear_r2"] <= TAU for _, r in rows)
    cl = sum(r["linear_r2"] <= TAU and hl[(sd, r["purpose"])]["per_dim_std_mean"] >= 0.5 and hl[(sd, r["purpose"])]["effective_rank"] >= 2 for sd, r in rows)
    accs = {}
    for s in p["per_seed"]:
        for k, v in s["task_accuracies"].items():
            accs.setdefault(k, []).append(v)
    base = gj(MAIN, f"results/{CANON[ds]}/per_seed_results.json")
    baccs = {}
    for s in base["per_seed"]:
        for k, v in s["task_accuracies"].items():
            baccs.setdefault(k, []).append(v)
    brows = [(s["seed"], r) for s in base["per_seed"] for r in s["attribute_results"]]
    bst = sum(r["linear_r2"] <= TAU for _, r in brows)
    bcl = sum(r["linear_r2"] <= TAU and health[(ds, sd, r["purpose"])]["per_dim_std_mean"] >= 0.5 and health[(ds, sd, r["purpose"])]["effective_rank"] >= 2 for sd, r in brows)
    pilot[ds] = {"pilot_strict_le": f"{st}/{len(rows)}", "pilot_clean": f"{cl}/{len(rows)}",
                 "pilot_r2_max": round(max(r["linear_r2"] for _, r in rows), 4),
                 "baseline_bestpt_strict_le": f"{bst}/{len(brows)}", "baseline_bestpt_clean": f"{bcl}/{len(brows)}",
                 "task_acc_pilot_mean": {k: round(sum(v) / len(v), 4) for k, v in accs.items()},
                 "task_acc_baseline_mean": {k: round(sum(v) / len(v), 4) for k, v in baccs.items()},
                 "pilot_best_epochs": [s["best_epoch"] for s in p["per_seed"]],
                 "pilot_cotter": [s.get("cotter_selection", {}).get("kind") for s in p["per_seed"]]}
lj(LOCAL / "erase_layer_pilot_aws" / "rebuttal" / "comparison.json")
add(id="c1_erase_pilot", claim="erase-layer pilot strict 54/60 -> 60/60; cleanly 5/60 -> 0/60",
    manuscript="not in submitted PDF (rebuttal asset); memory/project notes; rebuttal-evidence HEADLINE",
    sources=[str(LOCAL / "erase_layer_pilot_aws" / f"v2_{d}_ERASE_PILOT" / "per_seed_results.json") for d in ("adult", "hmda", "diabetes")] +
            [str(LOCAL / "erase_layer_pilot_aws" / "rebuttal" / "comparison.json")],
    field="per_seed[].attribute_results[].linear_r2 (<=0.05) and per_purpose_health",
    recomputed=pilot, reported="54/60 -> 60/60; 5/60 -> 0/60",
    checkpoint="both sides best.pt-reloaded evaluation from run_v2_dataset.py (canonical>best>final); the 54/60 baseline is NOT the 56/60 final.pt headline",
    note="pilot is a different architecture (frozen LEACE layer between backbone and repr_proj; LoRA only on repr_proj); large task-accuracy drops on Adult occupation/education")

# ---------------------------------------------------------------- (d) cross-purpose
sub = gj(MAIN, "results/v2_cross_purpose/aggregate.json")
subrows = [r for ds in sub for r in sub[ds]]
crit_a = sum(r["concat_delta_mean_pp"] > 1 for r in subrows)
crit_b = sum(r["gain_mean_pp"] > 1 for r in subrows)
lf = gj(MAIN, "results/cross_purpose_laftr/cross_purpose_laftr_results.json")
lrows = [r for ds in lf["laftr"]["aggregate"] for r in lf["laftr"]["aggregate"][ds]]
la = sum(r["concat_delta_mean_pp"] > 1 for r in lrows)
lb = sum(r["gain_mean_pp"] > 1 for r in lrows) if "gain_mean_pp" in lrows[0] else None
dual = gj(MAIN, "results/cross_purpose_laftr/DUAL_CRITERIA.json")
uni = gj(XP, "results/rebuttal/cross_purpose/unified_protocol_results.json")
ua, ub, nu = 0, 0, 0
abs_cells = {}
for ds, blk in uni["per_seed"].items():
    cells = {}
    for s in blk["per_seed"]:
        for r in s["rows"]:
            cells.setdefault((r["attribute"], r["arch"]), []).append(r)
    for (at, ar), rs in cells.items():
        nu += 1
        g = sum(r["gain_pp"] for r in rs) / len(rs)
        cd = sum(r["concat_delta_pp"] for r in rs) / len(rs)
        sd = sum(r["single_delta_pp"] for r in rs) / len(rs)
        ub += g > 1
        ua += cd > 1
        abs_cells[f"{ds}/{at}/{ar}"] = {"gain_pp": round(g, 2), "concat_minus_majority_pp": round(cd, 2), "best_single_minus_majority_pp": round(sd, 2)}
uagg = sum(v["flag_above_1pp"] for ds in uni["aggregate"].values() for v in ds.values())
loc = lj(LOCAL / "cross_purpose" / "comparison.json")
loc_new = {ds: (v.get("attack_flagged"), v.get("attack_total")) for ds, v in loc.get("new", {}).items()}
sub_abs = {f"{r['dataset']}/{r['attribute']}/{r['arch']}": {"concat_minus_majority_pp": round(r["concat_delta_mean_pp"], 2), "gain_pp": round(r["gain_mean_pp"], 2)} for r in subrows}
nl = [k for k in abs_cells if not k.endswith("/LR")]
nl_cmp = {"n_nonlinear_cells": len(nl),
          "submission_abs_flags_nonlinear": sum(sub_abs[k]["concat_minus_majority_pp"] > 1 for k in nl),
          "crosspurp_abs_flags_nonlinear": sum(abs_cells[k]["concat_minus_majority_pp"] > 1 for k in nl),
          "cells_where_crosspurp_abs_gt_submission_abs": sum(abs_cells[k]["concat_minus_majority_pp"] > sub_abs[k]["concat_minus_majority_pp"] for k in nl),
          "mean_abs_pp_submission_nonlinear": round(sum(sub_abs[k]["concat_minus_majority_pp"] for k in nl) / len(nl), 2),
          "mean_abs_pp_crosspurp_nonlinear": round(sum(abs_cells[k]["concat_minus_majority_pp"] for k in nl) / len(nl), 2),
          "crosspurp_LR_abs_all_zero": all(abs_cells[k]["concat_minus_majority_pp"] == 0 for k in abs_cells if k.endswith("/LR"))}
add(id="d1_cross_purpose", claim="Submission 26/33 absolute, 22/33 incremental (LAFTR 29/33, 16/33); rebuttal 'unified protocol' 22/33 -> 8/33",
    manuscript="pdf23 lines 16-17, 51-52, 317-320 (abstract of source (21) says 22/33)",
    sources=[f"{MAIN}:results/v2_cross_purpose/aggregate.json", f"{MAIN}:results/cross_purpose_laftr/cross_purpose_laftr_results.json",
             f"{MAIN}:results/cross_purpose_laftr/DUAL_CRITERIA.json", f"{XP}:results/rebuttal/cross_purpose/unified_protocol_results.json",
             str(LOCAL / "cross_purpose" / "comparison.json")],
    field="aggregate[].concat_delta_mean_pp (>1 = criterion A), gain_mean_pp (>1 = criterion B); unified per_seed.rows[].{gain_pp,concat_delta_pp}",
    recomputed={"submission_A_abs": f"{crit_a}/{len(subrows)}", "submission_B_incr": f"{crit_b}/{len(subrows)}",
                "laftr_A_abs": f"{la}/{len(lrows)}", "laftr_B_incr_recomputed": lb,
                "dual_criteria_file": {"pcrl_A": dual["criterion_A"]["pcrl"]["flagged"], "pcrl_B": dual["criterion_B"]["pcrl"]["flagged"],
                                       "laftr_A": dual["criterion_A"]["laftr"]["flagged"], "laftr_B": dual["criterion_B"]["laftr"]["flagged"]},
                "crosspurp_model_B_incr_mean_of_seeds": f"{ub}/{nu}", "crosspurp_model_aggregate_flags": f"{uagg}/{nu}",
                "crosspurp_model_A_abs_mean_of_seeds": f"{ua}/{nu}",
                "local_untracked_comparison_new_flags_by_ds": loc_new,
                "nonlinear_comparison": nl_cmp,
                "crosspurp_cells": abs_cells, "submission_cells": sub_abs},
    reported="26/33, 22/33, 29/33, 16/33; 22/33 -> 8/33",
    checkpoint="submission: R5/R7 final? (see run_cross_purpose_attack_v2.py); cross-purpose model: canonical_iterate.pt (Diabetes best_epoch=0 per commit msg)",
    note="8/33 is a NEW model (cross-purpose-constrained h_concat training) scored under the incremental criterion; under the absolute criterion the same new model flags more cells with much larger magnitudes")

# ---------------------------------------------------------------- (e) LAFTR-hard
sm = gj("laftr-hard-r2-2026-05-17", "results/laftr_hard_r2_adult_SMOKE/summary.json")
add(id="e1_laftr_hard_adult", claim="LAFTR-hard-R2 Adult 0/24 strict, mean R2 0.247, COLLAPSED",
    manuscript="not in submitted PDF (rebuttal asset; memory note project_laftr_hard_r2_launch)",
    sources=["laftr-hard-r2-2026-05-17:results/laftr_hard_r2_adult_SMOKE/summary.json"],
    field="summary.pass_count_mean (adj_pass) / STATUS",
    recomputed={"smoke_only": {"STATUS": sm["STATUS"], "n_seeds": sm["n_seeds"], "pass_counts_per_seed": sm["pass_counts_per_seed"], "total_wall_s": sm["total_wall_s"]}},
    reported="0/24 COLLAPSED", match="unverifiable",
    checkpoint="n/a", note="Full Adult per_seed_results.json reportedly only in S3 archive (archive/laftr_hard_r2/results_final/laftr_hard_r2_adult/); not committed, not local, not in relocation inventory. Smoke is 1 seed x 5 epochs.")

# ---------------------------------------------------------------- (f) INLP / LAFTR / PCRL
inlp = gj(MAIN, "results/inlp_benchmark/inlp_results.json")
rows = inlp["rows"]
def mean(k): return round(sum(r[k] for r in rows) / len(rows), 4)
tab = {}
for r in rows:
    k = f"{r['dataset']}/{r['purpose']}"
    tab.setdefault(k, []).append(r)
tab_out = {k: {"laftr_mean": round(sum(x["laftr_r2_onehot"] for x in v) / len(v), 3),
               "inlp_mean": round(sum(x["inlp_r2_onehot"] for x in v) / len(v), 3),
               "pcrl_mean": round(sum(x["pcrl_r2_onehot"] for x in v) / len(v), 3),
               "laftr_pass": f"{sum(x['laftr_strict_pass'] for x in v)}/{len(v)}",
               "inlp_pass": f"{sum(x['inlp_strict_pass'] for x in v)}/{len(v)}",
               "pcrl_pass": f"{sum(x['pcrl_strict_pass'] for x in v)}/{len(v)}"} for k, v in sorted(tab.items())}
# PCRL column consistency with DA audit
pc_mism = 0
for r in rows:
    for d in da_rows:
        if d["ds"] == r["dataset"] and d["purpose"] == r["purpose"] and d["attribute"] == r["attribute"] and d["seed"] == r["seed"]:
            if abs(d["r2_onehot"] - r["pcrl_r2_onehot"]) > 1e-5: pc_mism += 1
# LAFTR Adult metrics.json cross-check
lf_m = 0; lf_n = 0
for r in rows:
    if r["dataset"] != "adult": continue
    m = gj(MAIN, f"results/laftr_benchmark/adult/{r['purpose']}/seed_{r['seed']}/metrics.json")
    lf_n += 1
    if abs(m["metrics"]["per_attr"][r["attribute"]]["r2_onehot"] - r["laftr_r2_onehot"]) > 1e-5: lf_m += 1
# INLP strict pass recomputed (<=0.05)
inlp_pass_re = sum(r["inlp_r2_onehot"] <= TAU for r in rows)
add(id="f1_three_way", claim="Mean R2 PCRL 0.014 / INLP 0.038 / LAFTR 0.270; strict pass 93% / 72% / 25%; Table 13",
    manuscript="pdf23 lines 55-56, 349-350, 728-735, Table 13 (lines 1366-1378); LAFTR 15/60 at line 213-214",
    sources=[f"{MAIN}:results/inlp_benchmark/inlp_results.json"] + [f"{MAIN}:results/laftr_benchmark/adult/<purpose>/seed_<s>/metrics.json"],
    field="rows[].{laftr,inlp,pcrl}_r2_onehot, *_strict_pass",
    recomputed={"n_rows": len(rows), "n_cells": inlp["n_cells"], "mean": {"pcrl": mean("pcrl_r2_onehot"), "inlp": mean("inlp_r2_onehot"), "laftr": mean("laftr_r2_onehot")},
                "pass": {m: f"{sum(r[m + '_strict_pass'] for r in rows)}/{len(rows)}" for m in ("pcrl", "inlp", "laftr")},
                "inlp_pass_recomputed_le_0.05": f"{inlp_pass_re}/{len(rows)}",
                "per_dataset_purpose": tab_out, "pcrl_column_vs_DA_audit_mismatches": pc_mism,
                "laftr_adult_metrics_json_mismatches": f"{lf_m}/{lf_n}"},
    reported="0.014/0.038/0.270; 93/72/25%", checkpoint="PCRL column = final.pt DA audit; LAFTR/INLP = independent per-purpose encoders",
    note="LAFTR per-seed metrics.json committed only for Adult; HMDA/Diabetes LAFTR values exist only inside inlp_results.json rows (writer script not in repo)")

# ---------------------------------------------------------------- (g) rank-8
r8 = lj(LOCAL / "erase_rank8_diabetes_cpu" / "v2_diabetes_ERASE_RANK8" / "per_seed_results.json")
git_bytes(RE, "results/rebuttal/erase_rank8_diabetes_cpu/v2_diabetes_ERASE_RANK8/per_seed_results.json")
r8v = [r["linear_r2"] for s in r8["per_seed"] for r in s["attribute_results"]]
r8acc = {}
for s in r8["per_seed"]:
    for k, v in s["task_accuracies"].items(): r8acc.setdefault(k, []).append(v)
dp = lj(LOCAL / "erase_layer_pilot_aws" / "v2_diabetes_ERASE_PILOT" / "per_seed_results.json")
dpv = [r["linear_r2"] for s in dp["per_seed"] for r in s["attribute_results"]]
add(id="g1_rank8", claim="Diabetes rank-8 ablation 18/18 strict, R2 range [0.0043,0.0087]; rank-24 comparator 18/18 [0.0053,0.0080]",
    manuscript="not in submitted PDF (rebuttal-evidence HEADLINE.md)",
    sources=[f"{RE}:results/rebuttal/erase_rank8_diabetes_cpu/v2_diabetes_ERASE_RANK8/per_seed_results.json",
             str(LOCAL / "erase_rank8_diabetes_cpu/v2_diabetes_ERASE_RANK8/per_seed_results.json"),
             str(LOCAL / "erase_layer_pilot_aws/v2_diabetes_ERASE_PILOT/per_seed_results.json")],
    field="per_seed[].attribute_results[].linear_r2",
    recomputed={"rank8_strict_le": f"{sum(v <= TAU for v in r8v)}/{len(r8v)}", "rank8_range": [round(min(r8v), 4), round(max(r8v), 4)],
                "rank8_task_acc_mean": {k: round(sum(v) / len(v), 4) for k, v in r8acc.items()},
                "rank24_erase_pilot_strict_le": f"{sum(v <= TAU for v in dpv)}/{len(dpv)}", "rank24_erase_pilot_range": [round(min(dpv), 4), round(max(dpv), 4)]},
    reported="18/18 [0.0043,0.0087]", checkpoint="best.pt-reloaded eval (run_v2_dataset.py)",
    note="HEADLINE.md calls the rank-24 comparator 'published Round 7' but the 18/18 rank-24 numbers are the erase-layer pilot; the published Round 7 (LoRA-inside-LEACE) is 17/18 on final.pt. Local file sha256 equals rebuttal-evidence committed copy.")

# ---------------------------------------------------------------- (h) held-out seed 3
ho = gj(MAIN, "results/reviewer_dropins/heldout_s3_compare.json")
pc = ho["per_cell"]
deltas = [c["delta_r2_onehot"] for c in pc]
add(id="h1_heldout_s3", claim="Adult seed-3 held-out: 6/8 vs 7/8 (headline seed 2); clean 1/8 vs 0/8; mean delta +0.0111 (sd 0.0372, range [-0.0395,+0.0824])",
    manuscript="pdf23 lines 364-370",
    sources=[f"{MAIN}:results/reviewer_dropins/heldout_s3_compare.json", f"{MAIN}:results/v2_adult_HELDOUT_S3/dominant_axis_audit.json"],
    field="per_cell[].{held,head}_r2_pass, *_cleanly_compliant, delta_r2_onehot",
    recomputed={"held_pass": f"{sum(c['held_r2_pass'] for c in pc)}/{len(pc)}", "head_pass": f"{sum(c['head_r2_pass'] for c in pc)}/{len(pc)}",
                "held_clean": f"{sum(c['held_cleanly_compliant'] for c in pc)}/{len(pc)}", "head_clean": f"{sum(c['head_cleanly_compliant'] for c in pc)}/{len(pc)}",
                "delta_mean": round(statistics.mean(deltas), 4), "delta_sd_sample": round(statistics.stdev(deltas), 4),
                "delta_sd_pop": round(statistics.pstdev(deltas), 4), "delta_range": [round(min(deltas), 4), round(max(deltas), 4)]},
    reported="6/8 vs 7/8; 1/8 vs 0/8; +0.0111 (0.0372)", checkpoint="final.pt (HELDOUT DA audit epoch 199)",
    note="Seed 3 only shares the schedule; it is a new training seed on the SAME train/test split, not held-out data. 'Head' comparator is seed 2.")

# ---------------------------------------------------------------- (i) CelebA
cel = {}
accs = []
cellwise_viol = []
for s, d in (("0", "v2_celeba_R5_FULL"), ("1", "v2_celeba_R5_seed1"), ("2", "v2_celeba_R5_seed2")):
    t = gj(MAIN, f"results/{d}/train_set_r2.json")
    cp = gj(MAIN, f"results/{d}/cross_purpose.json")
    tl = gj(MAIN, f"results/{d}/training_log.json")
    last = tl["history"][-1]
    accs.append(last["task_acc"])
    for att in ("male", "young"):
        for m in cp["r5_trained"]:
            dlt = cp["r5_trained"][m][f"r2_{att}"] - cp["leace_init_baseline"][m][f"r2_{att}"]
            if dlt > 0: cellwise_viol.append((s, att, m, round(dlt * 100, 1)))
    cel[s] = {"train_r2_male": round(t["train_r2_male"], 4), "train_r2_young": round(t["train_r2_young"], 4),
              "val_r2_male_last_epoch": round(last["r2_male"], 4), "val_r2_young_last_epoch": round(last["r2_young"], 4),
              "val_smiling_acc_last_epoch": round(last["task_acc"], 4), "n_feasible_epochs": tl["selector"]["n_feasible_epochs"],
              "maxmax_delta_pp": {att: round(100 * (max(v[f"r2_{att}"] for v in cp["r5_trained"].values()) - max(v[f"r2_{att}"] for v in cp["leace_init_baseline"].values())), 1) for att in ("male", "young")},
              "mlp_trained_r2": {att: round(cp["r5_trained"]["MLP"][f"r2_{att}"], 3) for att in ("male", "young")}}
add(id="i1_celeba", claim="CelebA train-set R2<=0.005 Male/Young; Smiling acc 0.752+/-0.010; Table 11/12; trained<=LEACE-init 'without exception'",
    manuscript="pdf23 lines 338-340, 682-697, Tables 11-12",
    sources=[f"{MAIN}:results/{d}/{f}" for d in ("v2_celeba_R5_FULL", "v2_celeba_R5_seed1", "v2_celeba_R5_seed2") for f in ("train_set_r2.json", "cross_purpose.json", "training_log.json")],
    field="train_r2_*; history[-1].{task_acc,r2_male,r2_young}; selector.n_feasible_epochs; cross_purpose.{leace_init_baseline,r5_trained}.<arch>.r2_*",
    recomputed={"per_seed": cel, "smiling_acc_mean": round(statistics.mean(accs), 4), "smiling_acc_sd_sample": round(statistics.stdev(accs), 4),
                "cellwise_trained_gt_init": cellwise_viol},
    reported="R2<=0.005; 0.752+/-0.010; worst deltas -6.3 (Male), -8.2 (Young); trained<=init every cell",
    checkpoint="final epoch (selector found 0 feasible epochs on all seeds)",
    note="Table 12 deltas are max-over-attackers(trained) minus max-over-attackers(init), not cellwise; cellwise trained>init occurs. Val-split in-sample linear R2 at final epoch ~0.16 (fails tau). Cross-purpose 'R2' is linear R2 of attacker-predicted probabilities vs labels.")

# ---------------------------------------------------------------- Appendix P
for f in ("results/SEPARATE_SUMMARY.md", "results/LEACE_SUMMARY.md"):
    try:
        git_bytes(MAIN, f)
    except subprocess.CalledProcessError:
        pass

print(json.dumps({"entries": ENTRIES, "failures_final_pt": fails, "sources_sha256": SOURCES}, indent=1, default=str))
