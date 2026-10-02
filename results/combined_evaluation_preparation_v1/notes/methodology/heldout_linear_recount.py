"""Recount of EXISTING held-out linear outputs on defended PCRL representations (no fitting, no new data).

Reads committed git objects only:
  (1) origin/main:results/v2_cross_purpose/raw/{adult,hmda,diabetes}_s{0,1,2}{,_extra}.json
      rows[arch == "LR"].single_per_purpose[p]: LogisticRegression(C=1, lbfgs, max_iter=2000) fit on the
      TRAIN split representation of purpose p and scored (accuracy) on the TEST split
      (experiments/run_cross_purpose_attack_v2.py:167-173, majority on test :403-404). Defenses were fit
      on train. Max over auditor seeds is a no-op for lbfgs LR.
  (2) origin/main:results/v2_adult_CROSSPURP/results.json per_pair[...].r2_onehot_test: ridge(1e-4,
      float32) one-hot linear regression fit on TRAIN and scored on TEST (scripts/crosspurp/run_eval.py:70-90;
      test rows centred with the test mean).
Only purposes that DISALLOW the attribute are counted (pcrl/data/{adult,hmda,diabetes}.py purpose specs).
Run: python3 heldout_linear_recount.py  (from anywhere inside the PCRL repo)
"""
import json
import subprocess

REF = "origin/main"
DISALLOW = {
    "adult": {"income_prediction": ["race", "sex"],
              "employment_analysis": ["race", "age_group", "marital_status"],
              "education_assessment": ["sex", "race", "income"]},
    "hmda": {"underwriting": ["race", "ethnicity"], "pricing_analysis": ["race", "sex"],
             "fair_lending_audit": ["race", "sex"]},
    "diabetes": {"billing_audit": ["race", "gender"], "quality_research": ["race", "age_bucket"],
                 "clinical_decision_support": ["race", "gender"]},
}


def show(path):
    return json.loads(subprocess.check_output(["git", "show", f"{REF}:{path}"]))


out = {"lr_single_release": {}, "crosspurp_heldout_r2": {}}
cells = []
for ds in DISALLOW:
    for s in (0, 1, 2):
        for suf in ("", "_extra"):
            d = show(f"results/v2_cross_purpose/raw/{ds}_s{s}{suf}.json")
            for r in d["rows"]:
                if r.get("arch") != "LR":
                    continue
                for p, acc in r["single_per_purpose"].items():
                    if r["attribute"] in DISALLOW[ds].get(p, []):
                        cells.append({"dataset": ds, "seed": s, "checkpoint": d.get("checkpoint"),
                                      "purpose": p, "attribute": r["attribute"],
                                      "lr_heldout_acc": acc, "test_majority": r["majority"],
                                      "delta": acc - r["majority"]})
seen = {}
for c in cells:                      # de-duplicate rows present in both base and _extra files
    seen[(c["dataset"], c["seed"], c["purpose"], c["attribute"])] = c
cells = list(seen.values())
out["lr_single_release"] = {
    "n_disallowed_pair_seeds": len(cells),
    "delta_gt_0.01": sum(c["delta"] > 0.01 for c in cells),
    "delta_gt_0.02": sum(c["delta"] > 0.02 for c in cells),
    "max_delta": max(c["delta"] for c in cells),
    "by_dataset": {ds: {"n": sum(c["dataset"] == ds for c in cells),
                        "gt_0.02": sum(c["dataset"] == ds and c["delta"] > 0.02 for c in cells)}
                   for ds in DISALLOW},
    "checkpoints": sorted({c["checkpoint"] for c in cells}),
    "top5": sorted(cells, key=lambda c: -c["delta"])[:5],
}
d = show("results/v2_adult_CROSSPURP/results.json")
pairs = [(s["seed"], k, v["r2_onehot_test"]) for s in d["per_seed"] for k, v in s["per_pair"].items()]
out["crosspurp_heldout_r2"] = {
    "model": "Adult cross-purpose-constraint retrain (May 2026), canonical_iterate.pt, 75 epochs",
    "n_pair_seeds": len(pairs), "gt_0.05": sum(r > 0.05 for _, _, r in pairs),
    "failing": [(s, k, round(r, 4)) for s, k, r in pairs if r > 0.05],
    "native_insample_check_for_same_checkpoints": "NOT STORED in git; needs a frozen forward pass",
}
# (3) SPLINCE post-processing of PCRL final.pt (scripts/run_splince_benchmark.py): eraser fit on TRAIN
#     representations, LR auditor (C=1, max_iter 1000) fit on TRAIN (<=20k stratified) scored on TEST.
sp = []
for ds in DISALLOW:
    for s in (0, 1, 2):
        m = show(f"results/splince_benchmark/{ds}_s{s}/metrics.json")
        for r in m["pair_results"]:
            sp.append({"dataset": ds, "seed": s, "purpose": r["purpose"], "attribute": r["attribute"],
                       "r2_insample_test": r["r2_onehot"], "lr_delta": r["auditor_lr_acc"] - r["majority_proportion"]})
ok = [c for c in sp if c["r2_insample_test"] < 0.05]
out["splince_lr_auditor"] = {
    "n_pairs": len(sp), "native_r2_pass": len(ok),
    "lr_delta_gt_0.01_among_native_pass": sum(c["lr_delta"] > 0.01 for c in ok),
    "lr_delta_gt_0.02_among_native_pass": sum(c["lr_delta"] > 0.02 for c in ok),
    "max_lr_delta_among_native_pass": max(c["lr_delta"] for c in ok),
    "note": "majority_proportion as stored by the benchmark",
}
print(json.dumps(out, indent=1))
