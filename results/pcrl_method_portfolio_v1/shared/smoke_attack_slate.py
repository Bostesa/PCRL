"""Synthetic role-size timing of the four actual shared fresh readers.

Runs one score pair view and one D-style 1024-code pair feature view at the
real ATTACK_FIT_NEW/INNER row and group counts. No Adult file is opened. Every
reader gets a durable private completion receipt from fit_slate so S3 can
charge partial progress if the ceiling interrupts the unit.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from attack_eval import AttackView, fit_slate


ROOT = Path(__file__).resolve().parent / ".private" / "attack_slate_smoke_v1"
READERS = ("LR_C1", "MLP_64x32", "HGB_15x200", "DA_MLP_64x32")


def identities(n: int, ngroups: int):
    row = np.arange(n, dtype=np.int64)
    group = row.copy()
    group[ngroups:] = np.arange(n - ngroups, dtype=np.int64)
    return row, group


def score_view(n: int, ngroups: int, rng) -> AttackView:
    row, group = identities(n, ngroups)
    code1 = rng.integers(0, 16, size=n)
    code2 = rng.integers(0, 321, size=n)
    one1 = np.eye(16, dtype=np.float64)[code1]
    one2 = np.eye(321, dtype=np.float64)[code2]
    prob = rng.dirichlet(np.ones(8), size=n)
    x = np.column_stack((one1, one2, prob))
    da = np.column_stack((x, np.square(prob), np.log(np.maximum(prob, 1e-12))))
    return AttackView(row_id=row, unit=group, x=x,
                      components=("code1", "code2", "probability"),
                      wire_components=("code1", "code2", "probability", "model_id"),
                      component_columns=(("code1", 0, 16), ("code2", 16, 337),
                                         ("probability", 337, 345)),
                      wire_constants=(("model_id", "score-toy-v1"),),
                      x_defense_aware=da)


def feature_view(n: int, ngroups: int, rng) -> AttackView:
    row, group = identities(n, ngroups)
    code1 = rng.integers(0, 1024, size=n)
    code2 = rng.integers(0, 1024, size=n)
    one1 = np.eye(1024, dtype=np.float64)[code1]
    one2 = np.eye(1024, dtype=np.float64)[code2]
    bits1 = ((code1[:, None] >> np.arange(10)) & 1).astype(np.float64)
    bits2 = ((code2[:, None] >> np.arange(10)) & 1).astype(np.float64)
    decoded = rng.normal(size=(n, 32))
    prob = rng.dirichlet(np.ones(8), size=n)
    x = np.column_stack((one1, one2, bits1, bits2, decoded, prob))
    interaction = (bits1[:, :, None] * bits2[:, None, :]).reshape(n, 100)
    da = np.column_stack((x, np.square(decoded), np.log(np.maximum(prob, 1e-12)), interaction))
    return AttackView(row_id=row, unit=group, x=x,
                      components=("code1", "code2", "bits1", "bits2", "decoded", "probability"),
                      wire_components=("code1", "code2", "bits1", "bits2", "decoded", "probability", "model_id"),
                      component_columns=(("code1", 0, 1024), ("code2", 1024, 2048),
                                         ("bits1", 2048, 2058), ("bits2", 2058, 2068),
                                         ("decoded", 2068, 2100), ("probability", 2100, 2108)),
                      wire_constants=(("model_id", "feature-toy-v1"),),
                      x_defense_aware=da,
                      x_code_bits=np.column_stack((bits1, bits2)),
                      cell_key=np.asarray([f"{a}:{b}" for a, b in zip(code1, code2)]))


def run_one(name: str, builder, rng) -> dict:
    att = builder(4064, 4061, rng)
    inner = builder(2235, 2234, rng)
    sex = rng.integers(0, 2, size=4064).astype(np.int64)
    label = {"row_id": att.row_id, "unit": att.unit, "sex": sex}
    wall, cpu = time.perf_counter(), time.process_time()
    fitted = fit_slate(att, label, inner, required_readers=READERS, attacker_seed=0,
                       progress_dir=ROOT / name)
    return {"view": name, "att_rows": 4064, "inner_rows": 2235, "columns": att.x.shape[1],
            "da_columns": att.x_defense_aware.shape[1], "reader_progress": fitted["reader_progress"],
            "wall_seconds": time.perf_counter() - wall, "cpu_seconds": time.process_time() - cpu}


if __name__ == "__main__":
    if os.environ.get("OMP_NUM_THREADS") != "1" or ROOT.exists():
        raise SystemExit("single-thread and write-once private smoke root required")
    ROOT.mkdir(parents=True, mode=0o700)
    rng = np.random.Generator(np.random.PCG64DXSM(20061008))
    for name, builder in (("score_pair", score_view), ("feature_pair", feature_view)):
        result = run_one(name, builder, rng)
        path = ROOT / f"{name}_SUMMARY.json"
        path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        path.chmod(0o600)
        print(json.dumps({"view": name, "cpu_seconds": result["cpu_seconds"],
                          "wall_seconds": result["wall_seconds"], "readers": len(result["reader_progress"])}),
              flush=True)
