"""Data-free role-sized timing for the fixed focused inference family.

Uses the exact 2,235/2,234 INNER geometry, two seeds, ten rules, 6+6+18
readers per release, and 2,048 paired-group replicates. It opens no role file.
The output contains only resource timing and shape counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from inference import REPLICATES, RULES, _bootstrap, prepare


LOCAL_READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32",
                 "RAW_LR_C1", "RAW_HGB_15x200")


def synthetic_inputs():
    rng = np.random.Generator(np.random.PCG64DXSM(20261008))
    n = 2235
    row = np.arange(n, dtype=np.int64)
    unit = np.arange(n, dtype=np.int64)
    unit[-1] = unit[-2]
    sex = (unit % 2).astype(np.int64)
    income = rng.integers(0, 2, size=n, dtype=np.int64)
    occupation = rng.integers(0, 6, size=n, dtype=np.int64)

    def probs(labels, width, strength):
        p = np.full((n, width), (1 - strength) / (width - 1), dtype=np.float64)
        p[np.arange(n), labels] = strength
        return p

    ref = {
        "income": {name: probs(income, 2, 0.75) for name in ("U0", "Ucal", "F")},
        "occupation_group": {name: probs(occupation, 6, 0.58)
                             for name in ("U0", "Ucal", "F")},
    }
    seed_data, plans = {}, {}
    for seed in (0, 1):
        rules, selected = {}, {}
        for index, rule_id in enumerate(RULES):
            local = {}
            for view in ("income", "occupation_group"):
                local[view] = {
                    name: sex.astype(np.float64) * (0.04 + j * 0.015)
                          + rng.normal(size=n)
                    for j, name in enumerate(LOCAL_READERS)
                }
            coalition = {
                name: sex.astype(np.float64) * (0.06 + j * 0.015)
                      + rng.normal(size=n)
                for j, name in enumerate(LOCAL_READERS)
            }
            coalition.update({f"IGNORE_R2__{name}": local["income"][name]
                              for name in LOCAL_READERS})
            coalition.update({f"IGNORE_R1__{name}": local["occupation_group"][name]
                              for name in LOCAL_READERS})
            rules[rule_id] = {
                "status": "SELECTED", "pair_id": f"synthetic{index}",
                "prob_income": probs(income, 2, 0.71 + index * 0.001),
                "prob_occupation": probs(occupation, 6, 0.53 + index * 0.001),
                "reader_predictions": {**local, "coalition": coalition},
            }
            selected[rule_id] = {"status": "SELECTED", "pair_id": f"synthetic{index}"}
        plans[seed] = {"schema": "pcrl-focused-m3-control-plan-v1", "rules": selected}
        seed_data[seed] = {
            "truth": {"row_id": row.copy(), "unit": unit.copy(),
                      "income": income.copy(), "occupation": occupation.copy(),
                      "sex": sex.copy()},
            "references": ref,
            "rules": rules,
        }
    return seed_data, plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"synthetic timing requires {key}=1")
    path = Path(args.output).resolve()
    if ".private" not in path.parts or path.exists():
        raise ValueError("timing output must be a new private file")
    before_cpu, before_wall = time.process_time(), time.monotonic()
    inputs, plans = synthetic_inputs()
    prepared = prepare(inputs, plans)
    result = _bootstrap(prepared, hashlib.sha256(b"focused-inference-smoke-v1").digest(),
                        replicates=REPLICATES)
    output = {
        "schema": "pcrl-focused-m3-inference-smoke-v1",
        "scientific_data_access": "NONE",
        "status": result["status"],
        "rows": prepared.rows, "groups": len(np.unique(prepared.groups)),
        "paired_seeds": [0, 1], "rules": len(RULES),
        "reader_count_by_view": prepared.reader_count_by_view,
        "registered_clauses": len(prepared.names), "active_clauses": int(prepared.active.sum()),
        "replicates": REPLICATES,
        "cpu_seconds": time.process_time() - before_cpu,
        "wall_seconds": time.monotonic() - before_wall,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "inference_source_sha256": hashlib.sha256(
            (Path(__file__).parent / "inference.py").read_bytes()).hexdigest(),
        "smoke_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "synthetic resource timing only; no Adult performance estimate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(output, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": output["status"], "cpu_seconds": output["cpu_seconds"],
                      "peak_rss_bytes": output["peak_rss_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
