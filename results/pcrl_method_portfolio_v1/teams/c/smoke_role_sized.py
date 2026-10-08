"""Queue-only, role-sized synthetic Team C timing; reads no Adult data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from evaluation import attack_views_from_release
from prototype import ATOM_ORDER, AdultGroupedSchema, PurposeRelease, release_pair
from scorecard_builder import build_scorecard
from training import HeadSpec, fit_task_head


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _synthetic(rows: int, seed: int, *, duplicate_groups: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, 83), dtype=np.float32)
    income = (X[:, 0] + 0.2 * X[:, 1] > 0).astype(np.int64)
    occupation = (np.arange(rows) % 6).astype(np.int64)
    sex = (X[:, 2] > 0).astype(np.int64)
    row_id = np.arange(rows, dtype=np.int64)
    unit = row_id.copy()
    if duplicate_groups:
        unit[-duplicate_groups:] = np.arange(duplicate_groups, dtype=np.int64)
    return X, income, occupation, sex, row_id, unit


class _GlobalCalibrator:
    def __init__(self, record, apply):
        self.record = record
        self.apply = apply

    def transform(self, raw):
        return self.apply(raw, self.record)


class _UniformIncomeHead:
    def predict_proba(self, selected):
        return np.full((len(selected), 2), 1 / 2, dtype=np.float64)


class _IdentityCalibrator:
    def transform(self, raw):
        return np.asarray(raw, dtype=np.float64)


def _alignment(row_id, unit):
    return {name: {"row_id": row_id, "unit": unit}
            for name in ("income", "occupation_group", "coalition")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.parent.name != ".private":
        raise ValueError("smoke timing output must be inside a .private directory")
    shared_file = args.shared_dir / "common_eval.py"
    guard_file = args.shared_dir / "safe_data_guard.py"
    if not shared_file.is_file() or not guard_file.is_file():
        raise FileNotFoundError("exact shared evaluator files required")
    sys.path.insert(0, str(args.shared_dir))
    from common_eval import apply_global_temperature, fit_global_temperature

    schema = AdultGroupedSchema()
    fit_X, fit_income, fit_occ, fit_sex, _, fit_unit = _synthetic(
        15434, 410, duplicate_groups=6,
    )
    cal_X, _, cal_occ, _, cal_row, cal_unit = _synthetic(2000, 411)
    phases = {}

    start_wall, start_cpu = time.perf_counter(), time.process_time()
    scorecard = build_scorecard(
        fit_X, schema.feature_names, fit_income, fit_occ, fit_sex, fit_unit,
        base_atoms=ATOM_ORDER, optional_atoms=("relationship",), seed=0,
    )
    phases["two_mask_eight_proxy_fit_scorecard"] = {
        "wall_seconds": time.perf_counter() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
        "fit_counts": scorecard.fit_counts,
        "encoded_columns": [len(schema.column_indices(ATOM_ORDER)),
                            len(schema.column_indices(tuple(a for a in ATOM_ORDER
                                                           if a != "relationship")))],
    }

    selected = tuple(a for a in schema.normalize_atoms(tuple(schema.atoms))
                     if a != "relationship")
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    head = fit_task_head(
        fit_X, fit_occ, fit_unit, schema.feature_names, selected,
        purpose="occupation_group", model_id="team-c-synthetic-smoke", seed=0,
        spec=HeadSpec(),
    )
    phases["one_head_early_stop_plus_full_refit"] = {
        "wall_seconds": time.perf_counter() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
        "best_epoch": head.receipt["best_epoch"],
        "early_stop_epochs_run": head.receipt["early_stop_epochs_run"],
        "full_refit_epochs": head.receipt["full_refit_epochs"],
        "locked_head_epoch_ceiling": 60,
        "head_subfits": head.receipt["head_subfits"],
    }

    start_wall, start_cpu = time.perf_counter(), time.process_time()
    raw_cal = head.predict_proba(cal_X[:, schema.column_indices(selected)])
    cal_record = fit_global_temperature(
        raw_cal, cal_occ, task="occupation", arm_id=head.model_id,
        model_seed=0, row_id=cal_row, unit=cal_unit,
    )
    phases["calibration_2000_rows"] = {
        "wall_seconds": time.perf_counter() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
        "boundary": cal_record["boundary"],
    }

    income_release = PurposeRelease(
        schema, selected, _UniformIncomeHead(), "income", "uniform-synthetic",
        "identity-synthetic", _IdentityCalibrator(),
    )
    occupation_release = PurposeRelease(
        schema, selected, head, "occupation_group", head.model_id,
        "H-GLOBAL-TEMP:synthetic", _GlobalCalibrator(cal_record, apply_global_temperature),
    )
    for name, rows, seed in (("ATTACK_FIT_NEW", 4064, 412),
                             ("INNER_SELECTION", 2235, 413)):
        X, _, _, _, row_id, unit = _synthetic(rows, seed)
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        release = release_pair(X, schema.feature_names, income_release, occupation_release)
        views = attack_views_from_release(release, _alignment(row_id, unit))
        phases[f"forward_complete_views_{name}"] = {
            "wall_seconds": time.perf_counter() - start_wall,
            "cpu_seconds": time.process_time() - start_cpu,
            "rows": rows,
            "matrix_bytes": sum(v.raw_matrix.nbytes + v.complete_matrix.nbytes
                                for v in views.values()),
        }

    result = {
        "schema": "team-c-role-sized-synthetic-smoke-v1",
        "data_access": "NONE", "science_fit": False,
        "fit_rows": 15434, "calibration_rows": 2000,
        "attack_fit_rows": 4064, "inner_rows": 2235,
        "timing_scope": "two wide masks (83 and relationship-removed columns), one six-class occupation head; synthetic dense columns are not realistic one-hot sparsity; observed early stop may be short, so expand to the locked 30+30 epoch ceiling before budget admission",
        "source_hashes": {
            "script": _sha_file(Path(__file__)),
            "shared_common_eval": _sha_file(shared_file),
            "shared_safe_data_guard": _sha_file(guard_file),
        },
        "phases": phases,
        "maxrss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
