"""Synthetic timing of one candidate/control 2048-replicate endpoint family.

No Adult arrays, models or labels are opened. This runs 64 representative
paired-group replicates and forecasts a 2048-replicate cost for S3's prelock
resource decision. A science run still uses paired_inference.py exactly.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> None:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("single numerical thread required")
    rng = np.random.Generator(np.random.PCG64DXSM(40260108))
    n = 2235
    groups = np.arange(n, dtype=np.int64)
    groups[-1] = groups[-2]
    group_ids = np.unique(groups)
    positions = {int(g): np.flatnonzero(groups == g) for g in group_ids}
    y1 = rng.integers(0, 2, size=n)
    y2 = rng.integers(0, 6, size=n)
    sex = rng.integers(0, 2, size=n)
    release = {}
    for seed in (0, 1):
        for arm in ("candidate", "control"):
            p1 = rng.uniform(0.1, 0.9, size=n)
            p2 = rng.dirichlet(np.ones(6), size=n)
            readers = {view: rng.random((4, n)) for view in ("r1", "r2", "pair")}
            release[(seed, arm)] = (p1, p2, readers)

    def endpoints(index: np.ndarray) -> np.ndarray:
        out = []
        for seed in (0, 1):
            arm_results = {}
            for arm in ("candidate", "control"):
                p1, p2, readers = release[(seed, arm)]
                p1s = p1[index]
                p2s = p2[index]
                task = [float(np.mean((p1s >= 0.5) == y1[index])),
                        float(np.mean(-np.log(np.maximum(np.where(y1[index] == 1, p1s, 1 - p1s), 1e-12)))),
                        float(np.mean(2 * (p1s - y1[index]) ** 2)),
                        float(np.mean(np.argmax(p2s, axis=1) == y2[index])),
                        float(np.mean(-np.log(np.maximum(p2s[np.arange(len(index)), y2[index]], 1e-12)))),
                        float(np.mean(np.sum((p2s - np.eye(6)[y2[index]]) ** 2, axis=1)))]
                auc = []
                for view in ("r1", "r2", "pair"):
                    vals = [roc_auc_score(sex[index], pred[index]) for pred in readers[view]]
                    auc.append(max(max(v, 1 - v) for v in vals))
                arm_results[arm] = np.array(task + auc)
            out.append(arm_results["candidate"] - arm_results["control"])
        return np.mean(out, axis=0)

    wall, cpu = time.perf_counter(), time.process_time()
    for _ in range(64):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        index = np.concatenate([positions[int(g)] for g in sampled])
        endpoints(index)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    print(json.dumps({"schema": "pcrl-portfolio-bootstrap-smoke-v1", "source": "synthetic only",
                      "replicates_run": 64, "groups": len(group_ids), "paired_model_seeds": 2,
                      "arms": 2, "views": 3, "readers_per_view": 4,
                      "wall_seconds": wall, "cpu_seconds": cpu,
                      "forecast_2048_cpu_seconds": cpu * 32,
                      "forecast_scope": "one candidate/control contrast; scale by registered controls and arm families"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
