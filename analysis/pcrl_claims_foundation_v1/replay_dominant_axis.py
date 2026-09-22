"""Numerical replay of the original dominant-axis table from committed per-cell scores.

Reads results/v2_{adult_ROUND5,hmda_ROUND5,diabetes_ROUND7}/dominant_axis_audit.json
from the pinned review commit via `git show` (no checkpoint, no dataset). This is a
numerical replay of stored statistics, NOT a model replay: the representations and
fits are not recomputed.
"""
import hashlib
import json
import subprocess
import sys

import numpy as np

PIN = "0176f149e91c02b8e2d202eb25ea9cba8ae019dc"
CELLS = ["adult_ROUND5", "hmda_ROUND5", "diabetes_ROUND7"]


def load(tag):
    raw = subprocess.check_output(["git", "show", f"{PIN}:results/v2_{tag}/dominant_axis_audit.json"])
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


out = {"pin": PIN, "kind": "numerical replay of stored per-class scores (not model replay)", "datasets": {}}
for tag in CELLS:
    d, sha = load(tag)
    rows = []
    for seed, v in d["per_seed"].items():
        for r in v["rows"]:
            if not r["is_multiclass"]:
                continue
            p = np.asarray(r["priors"])
            w = p * (1 - p) / (p * (1 - p)).sum()
            R = np.asarray(r["per_class_r2"])
            k = r["r2_da_argmax_class"]
            rows.append({
                "seed": seed, "epoch": v.get("epoch"), "purpose": r["purpose"], "attribute": r["attribute"],
                "K": r["num_classes"], "r2_onehot_stored": r["r2_onehot"],
                "r2_onehot_float64_reconstruction": float(w @ R), "identity_residual": float(w @ R - r["r2_onehot"]),
                "r2_da": r["r2_da"], "argmax_class": k, "argmax_prior": float(p[k]), "argmax_weight": float(w[k]),
                "argmax_is_rarest": int(np.argmin(p)) == k,
                "bound_aggregate_over_weight": float(r["r2_onehot"] / w[k]),
            })
    res = np.abs([x["identity_residual"] for x in rows])
    out["datasets"][tag] = {
        "source_sha256": sha, "multiclass_pair_seeds": len(rows),
        "mean_r2_onehot_stored": float(np.mean([x["r2_onehot_stored"] for x in rows])),
        "mean_r2_onehot_float64_reconstruction": float(np.mean([x["r2_onehot_float64_reconstruction"] for x in rows])),
        "mean_r2_da": float(np.mean([x["r2_da"] for x in rows])),
        "max_abs_identity_residual": float(res.max()),
        "cells_with_residual_gt_1e-4": int((res > 1e-4).sum()),
        "rows": rows,
    }
json.dump(out, sys.stdout, indent=1)
