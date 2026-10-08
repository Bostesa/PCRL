"""Tiny synthetic end-to-end head/package fit; queue before execution."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from results.pcrl_capacity_matched_feature_selection_v1.m1.capacity import (
    ATOM_ORDER, mask_record, source_groups,
)
from results.pcrl_method_portfolio_v1.teams.c.prototype import AdultGroupedSchema

from head_pool import (
    _array_sha, _file_sha, fit_or_restore_task_package,
    make_head_cache_identity,
)


def _inputs(tmp_path):
    project = Path(__file__).resolve().parents[3]
    c_dir = project / "results/pcrl_method_portfolio_v1/teams/c"
    shared = project / "results/pcrl_method_portfolio_v1/shared"
    if str(c_dir) not in sys.path:
        sys.path.insert(0, str(c_dir))
    from training import HeadSpec

    schema = AdultGroupedSchema()
    full = mask_record(source_groups(schema.feature_names), ATOM_ORDER)
    rng = np.random.default_rng(71)
    x = rng.normal(size=(48, 83)).astype(np.float32)
    cx = rng.normal(size=(16, 83)).astype(np.float32)
    y = np.arange(len(x), dtype=np.int64) % 2
    cy = np.arange(len(cx), dtype=np.int64) % 2
    row = np.arange(100, 148, dtype=np.int64)
    unit = np.arange(1000, 1048, dtype=np.int64)
    crow = np.arange(200, 216, dtype=np.int64)
    cunit = np.arange(2000, 2016, dtype=np.int64)
    spec = HeadSpec(max_epochs=2, patience=1)
    sources = {
        "head_pool.py": _file_sha(Path(__file__).with_name("head_pool.py")),
        **{name: _file_sha(c_dir / name) for name in (
            "training.py", "prototype.py", "scorecard_builder.py",
            "release_package.py",
        )},
        **{name: _file_sha(shared / name) for name in (
            "common_eval.py", "safe_data_guard.py",
        )},
    }
    identity = make_head_cache_identity(
        mask=full, purpose="income", seed=0,
        campaign_lock_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        role_file_sha256={
            "fit_features": "1" * 64, "fit_label": "2" * 64,
            "calibration_features": "3" * 64,
            "calibration_label": "4" * 64,
        },
        fit_row_unit_sha256=_array_sha(np.column_stack((row, unit))),
        calibration_row_unit_sha256=_array_sha(np.column_stack((crow, cunit))),
        head_spec=spec, source_code_sha256=sources,
    )
    private = tmp_path / ".private"
    private.mkdir()
    args = dict(
        identity=identity, fit_x=x, fit_y=y, fit_row_id=row, fit_unit=unit,
        calibration_x=cx, calibration_y=cy, calibration_row_id=crow,
        calibration_unit=cunit, feature_names=schema.feature_names,
        private_root=private, head_spec=spec,
    )
    return args, full


def test_full_f_head_package_fits_once_then_restores_exactly(tmp_path):
    args, full = _inputs(tmp_path)
    first = fit_or_restore_task_package(
        **args, assert_queue_admitted=lambda: None,
    )
    second = fit_or_restore_task_package(
        **args, assert_queue_admitted=lambda: None,
    )
    assert first.fitted_now is True
    assert second.fitted_now is False
    assert first.package_sha256 == second.package_sha256
    assert first.head_sha256 == second.head_sha256
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.restored_release.selected_atoms == tuple(full["atoms"])
    q = first.restored_release.view(
        args["calibration_x"], args["feature_names"]
    )["probabilities"]
    assert q.shape == (16, 2)
    assert np.max(np.abs(q.sum(axis=1) - 1)) < 1e-8


def test_missing_queue_gate_or_wrong_row_hash_refuses_before_fit(tmp_path):
    args, _ = _inputs(tmp_path)
    with pytest.raises(PermissionError, match="queue admission"):
        fit_or_restore_task_package(**args, assert_queue_admitted=None)
    bad = dict(args)
    bad["fit_row_id"] = bad["fit_row_id"].copy()
    bad["fit_row_id"][0] += 1
    with pytest.raises(ValueError, match="FIT row/group"):
        fit_or_restore_task_package(
            **bad, assert_queue_admitted=lambda: None,
        )
