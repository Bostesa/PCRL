"""`plan`: enumerate evaluation units, dependencies, estimated CPU time / memory and missing inputs.
Never fits, never loads model weights, never touches the network.

Plan spec (JSON):
{"cells": [{"name": "adult_sex_income", "manifest": "<path or null>", "seeds": [0,1,2],
            "n_fit": 15000, "n_val": 6000, "n_eval": 9000, "d": 64, "n_outputs": 2, "K": 2,
            "pending": [{"archive_member": "...", "sha256": "..."}]}],
 "attackers": ["linear", "gbt", "mlp"], "surfaces": ["rep", "outputs", "rep+outputs"]}

Cost model: seconds(attacker) = c_attacker * (n_fit/1000) * n_grid * (d_surface/64) * k_factor with c_attacker
from a calibration JSON written by `fit-attackers --synthetic --timing-out` (measured on this machine);
bootstrap seconds = c_boot * n_boot * (n_eval/1000) per (surface, attacker, metric). Memory: peak of the
largest dense surface matrix in float64 times a working factor, plus a fixed interpreter overhead.
These are estimates and are labelled as such.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CAL = {"linear": 0.08, "gbt": 0.60, "mlp": 0.90, "bootstrap_per_1k_rows_per_rep": 0.0004,
               "source": "uncalibrated defaults"}
GRID_SIZES = {"linear": 4, "gbt": 4, "mlp": 4}  # default slate; make_plan recomputes from the protocol


def grid_sizes(cfg: dict) -> dict:
    at = cfg.get("attackers", {})
    g = dict(GRID_SIZES)
    if "linear" in at:
        g["linear"] = len(at["linear"].get("C", [None] * 4))
    if "gbt" in at:
        g["gbt"] = len(at["gbt"].get("learning_rate", [0, 0])) * len(at["gbt"].get("max_leaf_nodes", [0, 0]))
    if "mlp" in at:
        g["mlp"] = len(at["mlp"].get("hidden", [0, 0])) * len(at["mlp"].get("alpha", [0, 0]))
    return g
METRICS_PER_ATTACK = 3


def make_plan(spec: dict, cfg: dict, calibration: dict | None = None, base_dir: Path | None = None) -> dict:
    cal = {**DEFAULT_CAL, **(calibration or {})}
    attackers = spec.get("attackers", ["linear", "gbt", "mlp"])
    surfaces = spec.get("surfaces", cfg["surfaces"])
    n_boot = cfg["bootstrap"]["n_boot"]
    gs = grid_sizes(cfg)
    units, missing, deps = [], [], []
    tot_s, peak_mb = 0.0, 0.0
    for c in spec["cells"]:
        man = c.get("manifest")
        if man:
            mp = Path(man) if Path(man).is_absolute() or base_dir is None else base_dir / man
            if not mp.exists():
                missing.append({"cell": c["name"], "input": str(mp), "status": "MISSING"})
        else:
            missing.append({"cell": c["name"], "input": "manifest", "status": "NOT_YET_WRITTEN"})
        for p in c.get("pending", []):
            missing.append({"cell": c["name"], **p, "status": "PENDING (external drive)"})
        d_by_surface = {"rep": c["d"], "outputs": c.get("n_outputs", 2),
                        "rep+outputs": c["d"] + c.get("n_outputs", 2)}
        for seed in c.get("seeds", [0]):
            stage = f"{c['name']}/seed{seed}"
            deps += [[f"admit:{stage}", f"forward:{stage}"], [f"forward:{stage}", f"fit:{stage}"],
                     [f"fit:{stage}", f"score:{stage}"], [f"score:{stage}", f"infer:{stage}"],
                     [f"infer:{stage}", "report"]]
            for s in surfaces:
                ds = d_by_surface[s]
                for a in attackers:
                    K = c.get("K", 2)
                    kf = (K if a == "gbt" else 1.0 + 0.1 * K) if K > 2 else 1.0  # GBT grows one tree per class
                    fit_s = cal[a] * (c["n_fit"] / 1000) * gs[a] * max(ds / 64, 0.25) * kf
                    bf = (K / 2) ** 1.5 if K > 2 else 1.0  # per-class + pairwise statistics in every replicate
                    boot_s = (cal["bootstrap_per_1k_rows_per_rep"] * n_boot * (c["n_eval"] / 1000)
                              * METRICS_PER_ATTACK * bf)
                    mem = 8 * (c["n_fit"] + c["n_val"] + c["n_eval"]) * ds * 6 / 2 ** 20 + 300
                    units.append({"cell": c["name"], "seed": seed, "surface": s, "attacker": a,
                                  "metrics": ["macro_ovr_auc", "worst_class_auc", "worst_pair_auc"],
                                  "est_fit_cpu_s": round(fit_s, 1), "est_bootstrap_cpu_s": round(boot_s, 1),
                                  "est_peak_mem_mb": round(mem, 1),
                                  "requires": "--execute-scientific-fits"})
                    tot_s += fit_s + boot_s
                    peak_mb = max(peak_mb, mem)
    return {"n_units": len(units), "units": units, "dependencies": deps, "missing_inputs": missing,
            "estimate": {"total_cpu_hours": round(tot_s / 3600, 3), "peak_mem_mb": round(peak_mb, 1),
                         "calibration": cal, "grid_sizes": gs,
                         "label": "ESTIMATE (cost model, not a measurement)"},
            "fits_performed": 0}


def load_calibration(path: str | Path | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text())
