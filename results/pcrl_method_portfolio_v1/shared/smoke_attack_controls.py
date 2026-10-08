"""Synthetic-only null/shuffled/planted control fit smoke for S3 queue."""

from __future__ import annotations

import json

import numpy as np

from attack_controls import run_control_probes
from attack_eval import AttackView


def fixture(n: int, groups: int, seed: int):
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    row_id = np.arange(n, dtype=np.int64)
    unit = np.arange(n, dtype=np.int64)
    unit[groups:] = np.arange(n - groups, dtype=np.int64)
    sex = rng.integers(0, 2, size=n).astype(np.int64)
    x = rng.normal(size=(n, 3))
    view = AttackView(row_id=row_id, unit=unit, x=x,
                      components=("released_code", "released_probability", "decision"),
                      wire_components=("released_code", "released_probability", "decision", "model_id"),
                      component_columns=(("released_code", 0, 1),
                                         ("released_probability", 1, 2), ("decision", 2, 3)),
                      wire_constants=(("model_id", "toy-v1"),))
    label = {"row_id": row_id, "unit": unit, "sex": sex}
    return view, label


if __name__ == "__main__":
    a, ya = fixture(4064, 4061, 41)
    b, yb = fixture(2235, 2234, 42)
    probe = run_control_probes(a, ya, b, yb)
    out = {"schema": "pcrl-portfolio-synthetic-attack-control-smoke-v1",
           "status": "SYNTHETIC_PASS" if probe["status"] == "PASS" else "SYNTHETIC_FAIL",
           "fixture": "synthetic-role-sized-ATT4064-INNER2235-v1",
           "adult_role_values_accessed": False, "probe": probe}
    print(json.dumps(out, sort_keys=True))
    if out["status"] != "SYNTHETIC_PASS":
        raise SystemExit(1)
