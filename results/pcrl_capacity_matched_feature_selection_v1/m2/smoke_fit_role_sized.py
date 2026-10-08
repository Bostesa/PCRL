"""Data-free, role-sized timing of the exact M2 head and raw-critic path.

Run only as an S3 synthetic queue unit. It opens no Adult role export. The
synthetic selector accepts the first pair solely so timing completes even if
random synthetic tasks fail the real utility caps; it does not affect the
number, architecture, calibration, or training rows of any fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from bank import build_shared_bank
from fit_pipeline import _array_sha, run_fit_pipeline
from fit_worker import _search_auc_factory
from head_pool import (_file_sha, _write_once_json, fit_or_restore_task_package,
                       make_head_cache_identity)


FIT_ROWS = 15_434
CAL_ROWS = 2_000
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def run(output: Path, private_root: Path) -> dict:
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise PermissionError("synthetic timing requires one numerical thread")
    project = Path(__file__).resolve().parents[3]
    c_dir = project / "results/pcrl_method_portfolio_v1/teams/c"
    shared = project / "results/pcrl_method_portfolio_v1/shared"
    if str(c_dir) not in sys.path:
        sys.path.insert(0, str(c_dir))
    from prototype import AdultGroupedSchema
    from training import HeadSpec
    from results.pcrl_capacity_matched_feature_selection_v1.m1.capacity import capacity_geometry

    schema = AdultGroupedSchema()
    names = schema.feature_names
    bank = build_shared_bank(capacity_geometry(names)["masks"])
    rng = np.random.default_rng(9041)
    x = rng.normal(size=(FIT_ROWS, 83)).astype(np.float32)
    cx = rng.normal(size=(CAL_ROWS, 83)).astype(np.float32)
    fit_row = np.arange(FIT_ROWS, dtype=np.int64)
    fit_unit = np.arange(FIT_ROWS, dtype=np.int64)
    fit_unit[-6:] = fit_unit[-12:-6]
    cal_row = np.arange(100_000, 100_000 + CAL_ROWS, dtype=np.int64)
    cal_unit = cal_row.copy()
    yi = rng.integers(0, 2, size=FIT_ROWS, dtype=np.int64)
    yo = rng.integers(0, 6, size=FIT_ROWS, dtype=np.int64)
    cyi = rng.integers(0, 2, size=CAL_ROWS, dtype=np.int64)
    cyo = rng.integers(0, 6, size=CAL_ROWS, dtype=np.int64)
    sex = rng.integers(0, 2, size=FIT_ROWS, dtype=np.int64)
    refs = {name: np.full((FIT_ROWS, width), 1 / width, dtype=np.float64)
            for name, width in (("U0_1", 2), ("Ucal_1", 2),
                                ("U0_2", 6), ("Ucal_2", 6))}
    sources = {
        "head_pool.py": _file_sha(Path(__file__).with_name("head_pool.py")),
        **{name: _file_sha(c_dir / name) for name in (
            "training.py", "prototype.py", "scorecard_builder.py",
            "release_package.py")},
        **{name: _file_sha(shared / name) for name in (
            "common_eval.py", "safe_data_guard.py")},
    }
    spec = HeadSpec()
    role_hashes = {
        "fit_features": "1" * 64, "fit_label": "2" * 64,
        "calibration_features": "3" * 64, "calibration_label": "4" * 64,
    }
    counts = {"validation": 0, "final": 0}

    def package(purpose, mask, stage, rows):
        counts[stage] += 1
        fit_y = yi if purpose == "income" else yo
        cal_y = cyi if purpose == "income" else cyo
        identity = make_head_cache_identity(
            mask=mask, purpose=purpose, seed=0,
            campaign_lock_sha256="b" * 64,
            data_manifest_sha256="c" * 64,
            role_file_sha256=role_hashes,
            fit_row_unit_sha256=_array_sha(np.column_stack((fit_row[rows], fit_unit[rows]))),
            calibration_row_unit_sha256=_array_sha(np.column_stack((cal_row, cal_unit))),
            head_spec=spec, source_code_sha256=sources,
        )
        return fit_or_restore_task_package(
            identity, fit_x=x[rows], fit_y=fit_y[rows],
            fit_row_id=fit_row[rows], fit_unit=fit_unit[rows],
            calibration_x=cx, calibration_y=cal_y,
            calibration_row_id=cal_row, calibration_unit=cal_unit,
            feature_names=names, private_root=private_root,
            head_spec=spec, assert_queue_admitted=lambda: None,
        )

    task_seen = []
    def task_sink(table, choice, validation_packages):
        if sum(map(len, validation_packages.values())) != 12:
            raise RuntimeError("synthetic validation package registry is incomplete")
        task_seen.append(table["task_table_sha256"])

    def sex_source():
        if len(task_seen) != 1:
            raise PermissionError("synthetic task phase did not complete before SEX")
        return sex

    start = time.monotonic()
    result = run_fit_pipeline(
        bank=bank, seed=0, fit_x=x, fit_row_id=fit_row, fit_unit=fit_unit,
        fit_income=yi, fit_occupation=yo, fit_references=refs,
        feature_names=names, fit_package=package,
        read_sex=sex_source,
        fit_raw_search_auc=_search_auc_factory(x, 0),
        select_task_only=lambda table, bank: {
            "pair_id": bank.pairs[0].pair_id,
            "income_mask_id": bank.pairs[0].income_mask_id,
            "occupation_mask_id": bank.pairs[0].occupation_mask_id,
        },
        on_task_complete=task_sink,
    )
    elapsed = time.monotonic() - start
    if (counts != {"validation": 12, "final": 12}
            or len(result.pair_table.pairs) != 25):
        raise RuntimeError("role-sized smoke did not execute the full 24-package bank")
    report = {
        "schema": "pcrl-focused-m2-role-sized-timing-v1",
        "data_access": "NONE_SYNTHETIC_ONLY",
        "fit_rows": FIT_ROWS, "calibration_rows": CAL_ROWS,
        "feature_columns": 83, "model_seed": 0,
        "head_spec": vars(spec),
        "head_package_fits": counts,
        "neural_subfits": 48, "calibration_fits": 24,
        "raw_search_reader_fits": 22,
        "shared_masks": len(bank.masks), "pair_rows": len(result.pair_table.pairs),
        "elapsed_inside_worker_s": elapsed,
        "synthetic_selector_note": "first pair; no empirical utility selection",
    }
    _write_once_json(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--private-root", required=True, type=Path)
    args = parser.parse_args(argv)
    args.private_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    print(json.dumps(run(args.output, args.private_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
