"""D5: LEACE-on-raw baseline (NeurIPS App. P, '0/20 pair-seeds pass').

Recomputes the stored in-sample-on-test one-hot R^2 from the stored erased
test features (drive: fl-PCRL-main-results-ignored.tar::results/<ds>_LEACE/
erased_<purpose>.pt, read with a torch-free zip reader) and test labels
(drive INLP test_labels.npz, same test split; alignment is confirmed by
reproducing the stored values). Then measures how much of that R^2 sits in
near-null directions (relative singular value < 1e-3), i.e. numerical residue
of the train-fit eraser rather than a usable linear direction.
Run: /opt/homebrew/bin/python3 D_leace_raw.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D_common import SCRATCH, dump, file_bytes, git_json, inputs, r2_onehot  # noqa: E402
from D_torchzip import load_pt  # noqa: E402

AR = "fl-PCRL-main-results-ignored.tar::"
out = {"pairs": [], "notes": []}
for ds in ("adult", "hmda", "diabetes"):
    stored = git_json("origin/main", f"results/{ds}_LEACE/leace_baseline.json")
    labp = f"results/inlp_benchmark/{ds}/purpose_0/seed_0/test_labels.npz"
    file_bytes(SCRATCH / labp, AR + labp)
    lab = np.load(SCRATCH / labp)
    cache = {}
    for key, v in stored["per_pair"].items():
        pur, attr = key.split("/")
        if pur not in cache:
            rel = f"results/{ds}_LEACE/erased_{pur}.pt"
            file_bytes(SCRATCH / rel, AR + rel)
            cache[pur] = load_pt(str(SCRATCH / rel))["test"]
        H = cache[pur].astype(np.float64)
        y = lab[f"sensitive_{attr}"]
        r_full = r2_onehot(H, y)
        Hc = H - H.mean(0)
        _, s, Vt = np.linalg.svd(Hc, full_matrices=False)
        trunc = {}
        for thr in (1e-3, 1e-5):
            k = int((s > thr * s[0]).sum())
            trunc[str(thr)] = {"rank_kept": k, "r2": r2_onehot(Hc @ Vt[:k].T, y)}
        out["pairs"].append({
            "dataset": ds, "pair": key, "stored_linear_r2": v["linear_r2"],
            "recomputed_f64": r_full, "abs_diff": abs(r_full - v["linear_r2"]),
            "r2_after_dropping_rel_sv_below": trunc,
            "stored_delta_vs_majority": v["delta_vs_majority"],
            "stored_linear_acc_heldout_LR": v["linear_acc"], "majority": v["majority"],
            "stored_pass": v["pass"]})
P = out["pairs"]
out["n_pairs"] = len(P)
out["max_abs_diff_vs_stored"] = max(p["abs_diff"] for p in P)
out["stored_r2_pass_lt_0.05"] = sum(p["stored_linear_r2"] < 0.05 for p in P)
out["r2_pass_after_1e-3_cutoff"] = sum(
    p["r2_after_dropping_rel_sv_below"]["0.001"]["r2"] < 0.05 for p in P)
out["heldout_LR_at_majority(<=0.5pp)"] = sum(
    (p["stored_linear_acc_heldout_LR"] - p["majority"]) * 100 <= 0.5 for p in P)
out["delta_lt_2pp(nonlinear auditors)"] = sum(p["stored_delta_vs_majority"] < 0.02 for p in P)
out["combined_pass_stored"] = sum(p["stored_pass"] for p in P)
out["notes"].append("Stored 'pair-seeds' are single-seed pairs (8+6+6=20), not seeds x pairs.")
out["notes"].append("Erasure is SEQUENTIAL per attribute with concept_erasure.LeaceFitter "
                    "(experiments/run_leace_baseline.py:195-209), not the joint fit PCRL uses.")
out["inputs"] = inputs()
p = dump("D_leace_raw.json", out)
print(json.dumps({k: v for k, v in out.items() if k not in ("inputs", "pairs")}, indent=1))
for x in P:
    print(x["dataset"], x["pair"], round(x["stored_linear_r2"], 4), round(x["recomputed_f64"], 4),
          {k: round(v["r2"], 4) for k, v in x["r2_after_dropping_rel_sv_below"].items()},
          "LRacc-maj=%.4f" % (x["stored_linear_acc_heldout_LR"] - x["majority"]),
          "delta=%.3f" % x["stored_delta_vs_majority"])
print("wrote", p)
