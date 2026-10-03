"""FARE timing / memory calibration (method-admission owner, 2026-10-03).

Allowed inputs only: rows of the defense_fit role (features rep_p0, task, protected attribute) of the s0 encoder,
plus synthetic matrices. No attacker, validation or assessment row is read; nothing label-based is reported
except the structural cell count. Models are discarded (nothing is written to the run directory).

    cd <worktree> && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        ~/PCRL/.venv/bin/python results/combined_output_aware_removal_v1/scripts/fare_calibration.py

Writes notes/fare/FARE_CALIBRATION.json.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from oar import fare_official as F  # noqa: E402

NOTES = HERE.parents[1] / "notes" / "fare"
INPUTS = Path("~/PCRL_eval_cache_private/bench_v1/inputs").expanduser()
CELLS = {"adult": {"task": "task_income", "s": "sex", "n_total": 39205},
         "hmda": {"task": "task_loan_decision", "s": "race", "n_total": 77408}}
CERT_SYNTH_N = 1500  # order of the coordinator's certificate role (20% of attacker_fit groups)


def defense_fit_only(ds):
    roles = np.load(INPUTS / f"{ds}_roles.npz")
    keep_ids = roles["defense_fit__row_id"]
    fw = np.load(INPUTS / f"{ds}_s0_forward.npz")
    lab = np.load(INPUTS / f"{ds}_labels.npz")
    assert np.array_equal(fw["row_id"], lab["row_id"])
    m = np.isin(lab["row_id"], keep_ids)
    assert int(m.sum()) == len(keep_ids)
    assert set(np.unique(lab["role"][m]).tolist()) == {"defense_fit"}
    X = np.ascontiguousarray(fw["rep_p0"][m], dtype=np.float64)
    return X, lab[CELLS[ds]["task"]][m].astype(np.int64), lab[CELLS[ds]["s"]][m].astype(np.int64)


def main():
    prop = json.loads((NOTES / "FARE_GRID_PROPOSAL.json").read_text())
    grid = prop["grid"]
    out = {"schema": "oar.fare_calibration/v1", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "machine": {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count()},
           "threads": "OMP/MKL/OPENBLAS/VECLIB_MAXIMUM_THREADS=1 (parent and worker)",
           "rows_used": "defense_fit role of the s0 encoder only; synthetic matrices for full-population encode and "
                        "certificate timing",
           "what_is_measured": {
               "fit_worker_cpu_s": "CPU seconds inside the FARE worker for the official fit (+ medians, counts, "
                                   "row hashes), excluding interpreter start",
               "tree_fit_s": "wall seconds of DecisionTreeClassifier.fit alone",
               "fit_wall_s": "wall seconds of the whole subprocess call (spawn + imports + I/O + fit)",
               "fit_peak_rss_mb": "peak resident memory of the worker process",
               "encode_*": "official apply in a separate worker; defense_fit rows and a synthetic matrix of the "
                           "full per-dataset row count",
               "cert_*": "official certificate on synthetic certificate rows (timing only; value not reported)"},
           "fare_commit": F.FARE_COMMIT, "verify": F.verify_official_fare(), "datasets": {}}
    rng = np.random.RandomState(0)
    for ds, meta in CELLS.items():
        X, t, s = defense_fit_only(ds)
        Xsyn = rng.randn(meta["n_total"], X.shape[1]) * X.std(0) + X.mean(0)
        Xcert = rng.randn(CERT_SYNTH_N, X.shape[1]) * X.std(0) + X.mean(0)
        G = int(s.max()) + 1
        scert = rng.choice(G, CERT_SYNTH_N, p=np.bincount(s, minlength=G) / len(s))
        rows = []
        for cfg in grid + [F.zero_fairness(g) for g in grid]:
            m = F.fit(X, t, s, cfg, seed=0)
            fit_rec = dict(F.LAST_CALL)
            F.encode(m, X)
            enc_df = dict(F.LAST_CALL)
            F.encode(m, Xsyn)
            enc_all = dict(F.LAST_CALL)
            cert = F.certificate(m, Xcert, scert, prop["certificate"]["cert_cfg"])
            cert_rec = dict(F.LAST_CALL)
            rows.append({
                "name": cfg["name"], "max_leaf_nodes": cfg["max_leaf_nodes"],
                "min_samples_leaf": cfg["min_samples_leaf"], "gamma": cfg["gamma"],
                "n_cells": m.n_cells,
                "fit_worker_cpu_s": round(fit_rec["worker_cpu_seconds"], 3),
                "tree_fit_s": round(m.runtime["tree_fit_seconds"], 3),
                "fit_wall_s": round(fit_rec["seconds"], 3),
                "fit_peak_rss_mb": round(fit_rec["peak_rss_bytes"] / 2 ** 20, 1),
                "encode_defense_fit_worker_cpu_s": round(enc_df["worker_cpu_seconds"], 3),
                "encode_defense_fit_wall_s": round(enc_df["seconds"], 3),
                "encode_full_n_synthetic_worker_cpu_s": round(enc_all["worker_cpu_seconds"], 3),
                "encode_full_n_synthetic_wall_s": round(enc_all["seconds"], 3),
                "encode_peak_rss_mb": round(enc_all["peak_rss_bytes"] / 2 ** 20, 1),
                "cert_synthetic_wall_s": round(cert_rec["seconds"], 3),
                "cert_pairs": cert["n_pairs"],
            })
            print(ds, rows[-1], flush=True)
        tot_cpu = sum(r["fit_worker_cpu_s"] + r["encode_full_n_synthetic_worker_cpu_s"] for r in rows)
        tot_wall = sum(r["fit_wall_s"] + r["encode_full_n_synthetic_wall_s"] for r in rows)
        out["datasets"][ds] = {
            "n_defense_fit": int(len(X)), "d": int(X.shape[1]), "n_groups": G, "n_full_encode": meta["n_total"],
            "n_cert_synthetic": CERT_SYNTH_N, "configs": rows,
            "per_encoder_seed_6_grid_plus_6_twins": {"cpu_s": round(tot_cpu, 2), "wall_s": round(tot_wall, 2),
                                                     "max_peak_rss_mb": max(r["fit_peak_rss_mb"] for r in rows)},
        }
    out["projection_full_study"] = {
        "units": "2 datasets x 3 encoder seeds x (6 grid + 1 registered zero-fairness twin) fits and full encodes",
        "cpu_s": round(3 * sum(sum(r["fit_worker_cpu_s"] + r["encode_full_n_synthetic_worker_cpu_s"]
                                   for r in d["configs"][:6]) +
                               max(r["fit_worker_cpu_s"] + r["encode_full_n_synthetic_worker_cpu_s"]
                                   for r in d["configs"][6:])
                               for d in out["datasets"].values()), 1),
        "wall_s_serial_incl_spawn": round(3 * sum(sum(r["fit_wall_s"] + r["encode_full_n_synthetic_wall_s"]
                                                      for r in d["configs"][:6]) +
                                                  max(r["fit_wall_s"] + r["encode_full_n_synthetic_wall_s"]
                                                      for r in d["configs"][6:])
                                                  for d in out["datasets"].values()), 1),
        "note": "excludes attackers/heads; certificates add ~1 s each"}
    out["parent_peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)
    (NOTES / "FARE_CALIBRATION.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["projection_full_study"], indent=1))


if __name__ == "__main__":
    main()
