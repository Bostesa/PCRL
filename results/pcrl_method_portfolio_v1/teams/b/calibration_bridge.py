"""Fit Team B's one fixed S2 temperature per purpose on held-out CAL rows.

The input arrays must come from S2 SafeRoleStore's CALIBRATION_HELDOUT role.
This module does not load data, select a calibration family, or touch INNER.
"""

from __future__ import annotations

import hashlib
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from bottleneck import BottleneckPair, ReleaseAdapter, model_fingerprint, schema_fingerprint


@dataclass
class CalibrationResult:
    adapter: ReleaseAdapter
    records: dict[str, dict]
    record_file_sha256: dict[str, str]
    role_pair_sha256: str
    raw_draw_probability_sha256: dict[str, str]
    replay_max_probability_error: dict[str, float]
    paired_calibration_forward_passes: int


def _sha_arr(value) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode() + str(arr.shape).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def fit_calibration(
    model: BottleneckPair, *, arm_id: str, model_seed: int,
    feature_record: Mapping[str, np.ndarray],
    task_label_records: Mapping[str, Mapping[str, np.ndarray]],
    source_id: str, stochastic: bool, sampler_secret: bytes,
    expected_model_sha256: str, expected_schema_sha256: str,
    admitted_cal_pair_sha256: str, private_directory: str | Path,
) -> CalibrationResult:
    """Return a final one-draw release adapter and two private S2 records."""
    from common_eval import (apply_global_temperature, assert_same_people,
                             fit_global_temperature, save_calibrator)

    if not arm_id or not source_id or model_seed not in (0, 1, 2):
        raise ValueError("Locked arm, source and model seed are required")
    if set(task_label_records) != {"income", "occupation"}:
        raise ValueError("CALIBRATION_HELDOUT has both task label records only")
    for record in (feature_record, task_label_records["income"],
                   task_label_records["occupation"]):
        if (np.asarray(record["row_id"]).dtype != np.int64
                or np.asarray(record["unit"]).dtype != np.int64):
            raise ValueError("Admitted calibration person and group IDs must be int64")
    assert_same_people(feature_record, task_label_records["income"],
                       task_label_records["occupation"])
    features = np.asarray(feature_record["X"], dtype=np.float32)
    ids = np.asarray(feature_record["row_id"], dtype=np.int64)
    groups = np.asarray(feature_record["unit"], dtype=np.int64)
    feature_names = tuple(str(name) for name in feature_record["feature_names"])
    if (features.ndim != 2 or features.shape[1] != 83
            or ids.shape != (len(features),) or groups.shape != ids.shape
            or not len(ids) or np.unique(ids).size != len(ids)
            or not np.all(np.diff(ids) > 0)):
        raise ValueError("CALIBRATION_HELDOUT feature/person shape or order differs")
    pair_sha256 = _sha_arr(np.column_stack((ids, groups)).astype(np.int64))
    if pair_sha256 != admitted_cal_pair_sha256:
        raise ValueError("Calibration people differ from S2 admitted role")
    if (model_fingerprint(model) != expected_model_sha256
            or schema_fingerprint(model.feature_names) != expected_schema_sha256
            or feature_names != model.feature_names
            or schema_fingerprint(feature_names) != expected_schema_sha256):
        raise ValueError("Calibration model or ordered schema differs from sealed artifact")
    for purpose in ("income", "occupation"):
        if np.asarray(task_label_records[purpose][purpose]).shape != ids.shape:
            raise ValueError("Calibration task-label row order differs")
    directory = Path(private_directory)
    if ".private" not in directory.parts or directory.exists():
        raise ValueError("A new versioned private calibration directory is required")
    directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if stat.S_IMODE(directory.parent.stat().st_mode) != 0o700:
        raise ValueError("Private calibration parent must have mode 0700")
    directory.mkdir(mode=0o700)

    raw_adapter = ReleaseAdapter(
        model, stochastic=stochastic, sampler_secret=sampler_secret,
        temperatures={"income": 1.0, "occupation": 1.0},
        source_id=source_id, expected_model_sha256=expected_model_sha256,
        expected_schema_sha256=expected_schema_sha256, raw_mode=True,
    )
    raw = {purpose: [] for purpose in ("income", "occupation")}
    raw_z = {purpose: [] for purpose in ("income", "occupation")}
    for index, row in enumerate(ids):
        key = str(int(row))
        for purpose in ("income", "occupation"):
            wire = raw_adapter.release(purpose, key, features[index], columns=model.feature_names)
            raw[purpose].append(wire["probabilities"])
            raw_z[purpose].append(wire["features"])
    raw = {purpose: np.asarray(values, dtype=np.float64) for purpose, values in raw.items()}
    records, hashes, temperatures = {}, {}, {}
    for purpose in ("income", "occupation"):
        record = fit_global_temperature(
            raw[purpose], np.asarray(task_label_records[purpose][purpose]),
            task=purpose, arm_id=arm_id, model_seed=model_seed,
            row_id=ids, unit=groups,
            draw_id=f"{arm_id}:s{model_seed}:cal-one-draw",
        )
        calibrated = apply_global_temperature(raw[purpose], record)
        if calibrated.shape != raw[purpose].shape or not np.isfinite(calibrated).all():
            raise ValueError("S2 calibration output is invalid")
        records[purpose] = record
        hashes[purpose] = save_calibrator(record, directory / f"{purpose}.json")
        temperatures[purpose] = 1.0 / float(record["alpha"])
    replay_adapter = ReleaseAdapter(
        model, stochastic=stochastic, sampler_secret=sampler_secret,
        temperatures=temperatures, source_id=source_id,
        expected_model_sha256=expected_model_sha256,
        expected_schema_sha256=expected_schema_sha256,
        calibration_records=records, calibration_record_sha256=hashes,
    )
    replay_max_error = {}
    for purpose in ("income", "occupation"):
        expected = apply_global_temperature(raw[purpose], records[purpose])
        errors = []
        for index, row in enumerate(ids):
            replay = replay_adapter.release(
                purpose, str(int(row)), features[index], columns=model.feature_names
            )
            if not np.array_equal(replay["features"], raw_z[purpose][index]):
                raise ValueError("Final CAL release changes the fixed sampled representation")
            errors.append(float(np.max(np.abs(replay["probabilities"] - expected[index]))))
        replay_max_error[purpose] = max(errors)
        if replay_max_error[purpose] > 1e-6:
            raise ValueError("Final CAL release differs from S2 fitted calibration law")
    final_adapter = ReleaseAdapter(
        model, stochastic=stochastic, sampler_secret=sampler_secret,
        temperatures=temperatures, source_id=source_id,
        expected_model_sha256=expected_model_sha256,
        expected_schema_sha256=expected_schema_sha256,
        calibration_records=records, calibration_record_sha256=hashes,
    )
    return CalibrationResult(
        adapter=final_adapter, records=records, record_file_sha256=hashes,
        role_pair_sha256=pair_sha256,
        raw_draw_probability_sha256={purpose: _sha_arr(raw[purpose]) for purpose in raw},
        replay_max_probability_error=replay_max_error,
        paired_calibration_forward_passes=2,
    )
