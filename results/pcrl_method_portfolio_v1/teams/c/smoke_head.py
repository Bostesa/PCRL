"""Role-sized synthetic timing of the exact proposed Team C head fit.

No Adult data, source archive, role export, or checkpoints are read or written.
S3 must run this under the global queue with numerical threads capped at one.
"""

from __future__ import annotations

import json
import resource
import time

import numpy as np

from prototype import AdultGroupedSchema
from training import HeadSpec, fit_task_head


def main() -> None:
    n = 15_434
    rng = np.random.default_rng(1731)
    X = rng.normal(size=(n, 83)).astype(np.float32)
    signal = 0.45 * X[:, 0] + 0.30 * X[:, 2] - 0.20 * X[:, 4]
    y = np.searchsorted(np.quantile(signal, np.arange(1, 6) / 6), signal).astype(np.int64)
    unit = np.arange(n, dtype=np.int64)
    unit[-6:] = unit[:6]
    schema = AdultGroupedSchema()
    spec = HeadSpec()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    head = fit_task_head(
        X, y, unit, schema.feature_names, tuple(schema.atoms),
        purpose="occupation_group", model_id="synthetic-role-size-timing",
        seed=17, spec=spec,
    )
    report = {
        "schema": "team-c-head-role-size-smoke-v1",
        "data_access": "NONE_SYNTHETIC_ONLY",
        "rows": n,
        "groups": len(np.unique(unit)),
        "columns": X.shape[1],
        "classes": 6,
        "spec": head.receipt["spec"],
        "early_stop_epochs_run": head.receipt["early_stop_epochs_run"],
        "full_refit_epochs": head.receipt["full_refit_epochs"],
        "wall_seconds": time.perf_counter() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
        "max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "task_head_subfits": head.receipt["head_subfits"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
