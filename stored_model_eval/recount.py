"""Historical recounts from stored arrays / stored values only (no fitting, no network).

probability recount: a spec lists configurations, each pointing at an npz of stored held-out
probabilities written by the original run, with keys "<ARCH>_<tag>_prob" and "y_<tag>" (one tag per
train-seed x probe-seed draw). Per configuration: macro OvR AUC per draw, mean over draws per attacker,
max over the configuration's original attacker suite; counts of configurations above each bar (strict >).
Support-aware multiclass quantities (per-class, worst-class, worst-pair) are reported alongside with
NOT_ESTIMABLE where support is insufficient. Configurations sharing one measurement (same file and same
suite) are also counted once ("distinct measurements").

PCRL strict recount: per-cell stored R2 under the final-iterate rule (dominant_axis_audit.json r2_onehot,
final.pt) and the best-validation rule (per_seed_results.json linear_r2, best.pt), read with `git show`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .metrics import (NotEstimable, is_estimable, macro_ovr_auc, pairwise_auc, per_class_ovr_auc,
                      to_jsonable, worst_class_auc)


def _draws(z, arch):
    tags = sorted(k[len(arch) + 1:-len("_prob")] for k in z.files if k.startswith(arch + "_") and k.endswith("_prob"))
    return [(t, z["y_" + t].astype(np.int64), z[f"{arch}_{t}_prob"].astype(np.float64)) for t in tags]


def recount_probability_configs(spec: dict, base_dir: Path | None = None) -> dict:
    bars = spec.get("bars", [0.52, 0.55, 0.60])
    ms_hist = int(spec.get("historical_min_support", 1))
    ms_sup = int(spec.get("supported_min_support", 100))
    rows, file_hashes, problems = [], {}, []
    for c in spec["configs"]:
        p = Path(c["file"])
        if base_dir is not None and not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            rows.append({"key": c["key"], "status": "PENDING", "file": str(p)})
            continue
        h = file_hashes.get(str(p)) or hashlib.sha256(p.read_bytes()).hexdigest()
        file_hashes[str(p)] = h
        if c.get("file_sha256") and c["file_sha256"] != h:
            problems.append(f"{c['key']}: sha256 mismatch {h} vs {c['file_sha256']}")
            rows.append({"key": c["key"], "status": "REJECTED_HASH", "file": str(p)})
            continue
        z = np.load(p, allow_pickle=False)
        per = {}
        for arch in c["suite"]:
            dr = _draws(z, arch)
            if not dr:
                per[arch] = NotEstimable(f"no stored probabilities for {arch}")
                continue
            vals = [macro_ovr_auc(y, P, ms_hist) for _, y, P in dr]
            ent = {"n_draws": len(dr),
                   "macro_mean": float(np.mean(vals)) if all(is_estimable(v) for v in vals)
                   else NotEstimable("a draw is not estimable", counts={"draws": [to_jsonable(v) for v in vals]}),
                   "n_rows_per_draw": sorted({len(y) for _, y, _ in dr})}
            K = dr[0][2].shape[1]
            if K > 2:
                wc = [worst_class_auc(y, P, ms_sup) for _, y, P in dr]
                wp = [pairwise_auc(y, P, ms_sup) for _, y, P in dr]
                ent["class_counts_per_draw"] = [np.bincount(y, minlength=K).tolist() for _, y, _ in dr]
                ent["supported_min_support"] = ms_sup
                ent["worst_class_supported"] = [to_jsonable(w["value"]) for w in wc]
                ent["worst_class_coverage"] = [list(w["coverage"]) for w in wc]
                ent["worst_pair_supported"] = [to_jsonable(w["max"]) for w in wp]
                ent["worst_pair_argmax"] = [w["argmax"] for w in wp]
                ent["pair_coverage"] = [list(w["coverage"]) for w in wp]
                ent["per_class_ovr_draw0"] = to_jsonable(per_class_ovr_auc(dr[0][1], dr[0][2], ms_sup))
            per[arch] = ent
        est = [per[a]["macro_mean"] for a in c["suite"] if isinstance(per[a], dict) and is_estimable(per[a]["macro_mean"])]
        headline = max(est) if len(est) == len(c["suite"]) else NotEstimable("suite incomplete")
        stored = c.get("stored", {})
        stored_max = max(stored[a] for a in c["suite"]) if all(a in stored for a in c["suite"]) else None
        rows.append({"key": c["key"], "status": "RECOUNTED", "file": str(p), "file_sha256": h,
                     "measurement_id": c.get("measurement_id") or f"{h}|{'+'.join(c['suite'])}",
                     "suite": c["suite"], "per_attacker": per, "recomputed_max": headline,
                     "stored_max": stored_max,
                     "delta_vs_stored": (headline - stored_max) if (stored_max is not None and is_estimable(headline)) else None})
    done = [r for r in rows if r["status"] == "RECOUNTED" and is_estimable(r["recomputed_max"])]
    counts = {f"{b:.2f}": int(sum(r["recomputed_max"] > b for r in done)) for b in bars}
    stored_counts = {f"{b:.2f}": int(sum(r["stored_max"] > b for r in done if r["stored_max"] is not None)) for b in bars}
    distinct = {}
    for r in done:
        distinct.setdefault(r["measurement_id"], r)
    distinct_counts = {f"{b:.2f}": int(sum(r["recomputed_max"] > b for r in distinct.values())) for b in bars}
    deltas = [abs(r["delta_vs_stored"]) for r in done if r["delta_vs_stored"] is not None]
    dup_groups = {}
    for r in done:
        dup_groups.setdefault(r["measurement_id"], []).append(r["key"])
    return to_jsonable({
        "n_configs": len(spec["configs"]), "n_recounted": len(done),
        "n_pending": sum(r["status"] == "PENDING" for r in rows), "problems": problems,
        "fail_counts_recomputed": counts, "fail_counts_from_stored_values": stored_counts,
        "n_distinct_measurements": len(distinct), "fail_counts_distinct_measurements": distinct_counts,
        "shared_measurements": {k: v for k, v in dup_groups.items() if len(v) > 1},
        "max_abs_delta_vs_stored": max(deltas) if deltas else None,
        "near_bar": sorted([[round(r["recomputed_max"], 4), r["key"]] for r in done
                            if min(abs(r["recomputed_max"] - b) for b in bars) < 0.01]),
        "rows": rows})


# --------------------------------------------------------------------------------------------------
# PCRL NeurIPS strict counts
# --------------------------------------------------------------------------------------------------

DEFAULT_GRID = [("adult", "results/v2_adult_ROUND5"), ("hmda", "results/v2_hmda_ROUND5"),
                ("diabetes", "results/v2_diabetes_ROUND7")]


def _git_json(repo, ref, path, inputs):
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{path}"])
    inputs.append({"location": f"{ref}:{path}", "sha256": hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


def recount_pcrl_strict(repo: str | Path, ref: str = "origin/main", grid=None, tau: float = 0.05) -> dict:
    grid = grid or DEFAULT_GRID
    inputs, per_ds = [], {}
    tot = {"n": 0, "final_le": 0, "final_lt": 0, "best_le": 0, "best_lt": 0}
    for name, d in grid:
        ps = _git_json(repo, ref, f"{d}/per_seed_results.json", inputs)
        da = _git_json(repo, ref, f"{d}/dominant_axis_audit.json", inputs)
        best = {(int(s["seed"]), r["purpose"], r["attribute"]): float(r["linear_r2"])
                for s in ps["per_seed"] for r in s["attribute_results"]}
        final = {(int(seed), r["purpose"], r["attribute"]): float(r["r2_onehot"])
                 for seed, sd in da["per_seed"].items() for r in sd["rows"]}
        if set(best) != set(final):
            raise ValueError(f"{name}: final and best cell sets differ")
        e = {"n_cells": len(best),
             "final_le": sum(v <= tau for v in final.values()), "final_lt": sum(v < tau for v in final.values()),
             "best_le": sum(v <= tau for v in best.values()), "best_lt": sum(v < tau for v in best.values()),
             "final_epochs": {k: v.get("epoch") for k, v in da["per_seed"].items()},
             "best_epochs": [s.get("best_epoch") for s in ps["per_seed"]],
             "flips": sorted([[k[0], k[1], k[2], round(best[k], 4), round(final[k], 4)] for k in best
                              if (best[k] <= tau) != (final[k] <= tau)]),
             "final_failing": sorted([[k[0], k[1], k[2], round(final[k], 4)] for k in final if final[k] > tau]),
             "best_failing": sorted([[k[0], k[1], k[2], round(best[k], 4)] for k in best if best[k] > tau])}
        per_ds[name] = e
        tot["n"] += e["n_cells"]
        for k in ("final_le", "final_lt", "best_le", "best_lt"):
            tot[k] += e[k]
    return {"ref": ref, "tau": tau, "grid": [list(g) for g in grid], "per_dataset": per_ds, "totals": tot,
            "inputs": inputs,
            "sources": {"final": "dominant_axis_audit.json per_seed.*.rows[].r2_onehot (final.pt)",
                        "best": "per_seed_results.json per_seed[].attribute_results[].linear_r2 (best.pt)"}}
