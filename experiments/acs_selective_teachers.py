"""Immutable raw, true-label LEACE and paired-permutation LEACE teachers.

Only raw PCA16 representation-fitting rows and SEX/RAC1P enter construction.
The complete-case permutation and all original input identities are persisted
before either eraser is fitted. Historical LEACE mathematics is reused exactly.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from experiments.acs_protection_maps import (
    FrozenAffineMap, SCHEMA, _concepts, _readonly, covariance_diagnostics,
    fit_joint_leace,
)
from experiments.acs_transfer_data import array_hash


TEACHER_CONFIG = {
    "raw_width": 16, "fit_pool": "representation_fit", "permutation_seed_base": 20260908,
    "sham_permutation": "one shared permutation of complete-case SEX/RAC1P pairs; incomplete rows unchanged",
    "fit_order": ["E", "S"], "teacher_dtype": "float32", "application_compute_dtype": "float64",
    "coordinate_scale": "immutable original representation-fitting PCA16 standard deviations; original constant-coordinate handling",
    "reconstruction_denominator_variance_floor": 1e-12,
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+"\n")


def _raw(value, minimum_rows=1):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 16 or len(raw) < minimum_rows or not np.isfinite(raw).all():
        raise ValueError("Require a nonempty finite matrix of raw PCA16 coordinates")
    if raw.dtype != np.float32:
        raise ValueError("Use the original float32 PCA16 release, without feature restandardization")
    return raw


def original_statistics(raw_fit, original_mean, original_scale):
    """Check saved original statistics; return exact provided first16 values."""
    raw = _raw(raw_fit, minimum_rows=2).astype(np.float64)
    mean, scale = np.asarray(original_mean, np.float64)[:16], np.asarray(original_scale, np.float64)[:16]
    if (mean.shape != (16,) or scale.shape != (16,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Finite original fitting mean and positive coordinate scales required")
    std = raw.std(0)
    if not np.array_equal(mean, raw.mean(0)) or not np.array_equal(scale, np.where(std > 1e-12, std, 1.)):
        raise ValueError("Supplied statistics do not match the original raw PCA16 fitting pool")
    return _readonly(mean, np.float64), _readonly(scale, np.float64)


def paired_permutation(protected, seed, n):
    """Joint permutation preserves masks, complete-case marginals and joint counts."""
    if not isinstance(seed, (int, np.integer)) or int(seed) not in (0, 1, 2):
        raise ValueError("The frozen scientific seeds are 0, 1 and 2")
    _, complete, known, values, coverage = _concepts(protected, n)
    rows = np.flatnonzero(complete)
    if len(rows) < 2:
        raise ValueError("At least two complete protected-label fitting rows required")
    order = np.random.default_rng(20260908+int(seed)).permutation(len(rows))
    full_order = np.arange(n, dtype=np.int64)
    full_order[rows] = rows[order]
    shuffled = {key: _readonly(value[full_order], np.int64) for key, value in values.items()}
    before = np.bincount(values["SEX"][rows]*9+values["RAC1P"][rows], minlength=18)
    after = np.bincount(shuffled["SEX"][rows]*9+shuffled["RAC1P"][rows], minlength=18)
    assert np.array_equal(before, after)
    assert all(np.array_equal(value >= 0, shuffled[key] >= 0) for key, value in values.items())
    return {"protected": MappingProxyType(shuffled), "complete_rows": _readonly(rows, np.int64),
            "permutation": _readonly(order, np.int64), "full_row_permutation": _readonly(full_order, np.int64),
            "complete_mask": _readonly(complete, bool), "known_masks": _readonly(known, bool),
            "metadata": {"seed": int(seed), "permutation_seed": 20260908+int(seed),
                         "input_rows": n, "complete_case_rows": len(rows), "coverage": coverage,
                         "complete_case_joint_counts_SEX_by_RAC1P": before.reshape(2, 9).tolist(),
                         "joint_counts_preserved": True, "missing_masks_preserved": True,
                         "permutation_sha256": array_hash(order), "full_row_permutation_sha256": array_hash(full_order),
                         "complete_rows_sha256": array_hash(rows), "complete_mask_sha256": array_hash(complete),
                         "known_masks_sha256": array_hash(known),
                         "original_label_sha256": {k: array_hash(v) for k, v in values.items()},
                         "permuted_label_sha256": {k: array_hash(v) for k, v in shuffled.items()}}}


def apply_teachers(maps, raw):
    """Frozen raw-coordinate outputs; the maps never receive outcome labels."""
    raw = _raw(raw)
    if set(maps) != {"E", "S"}:
        raise ValueError("Exactly the frozen true-label and sham LEACE maps are required")
    targets = {"R": _readonly(raw, np.float32)}
    for name in ("E", "S"):
        targets[name] = _readonly(maps[name].apply(raw), np.float32)
    return MappingProxyType(targets)


def build_teachers(raw_fit, protected, seed, out, *, original_mean, original_scale,
                   fit_pool="representation_fit", fit_raw_rows=None):
    """Freeze the sham assignment first, then fit each immutable teacher once."""
    if fit_pool != "representation_fit":
        raise ValueError("Teachers may be fitted only on representation_fit")
    raw = _raw(raw_fit, minimum_rows=2)
    mean, scale = original_statistics(raw, original_mean, original_scale)
    permutation = paired_permutation(protected, seed, len(raw))
    if fit_raw_rows is not None:
        fit_raw_rows = np.asarray(fit_raw_rows)
        if fit_raw_rows.shape != (len(raw),) or not np.issubdtype(fit_raw_rows.dtype, np.integer):
            raise ValueError("Original raw-row identifiers must be aligned integer fitting rows")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    permutation_path = out/"paired_permutation.npz"
    with permutation_path.open("xb") as handle:
        np.savez_compressed(handle, complete_rows=permutation["complete_rows"],
            permutation=permutation["permutation"], full_row_permutation=permutation["full_row_permutation"],
            complete_mask=permutation["complete_mask"], known_masks=permutation["known_masks"])
    prefit = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "config": TEACHER_CONFIG, "class_schema": SCHEMA, "permutation": permutation["metadata"],
              "raw_fit_sha256": array_hash(raw), "original_mean": mean.tolist(), "original_scale": scale.tolist(),
              "fit_raw_rows_sha256": array_hash(fit_raw_rows) if fit_raw_rows is not None else None,
              "original_mean_sha256": array_hash(mean), "original_scale_sha256": array_hash(scale),
              "permutation_file_sha256": _sha(permutation_path), "any_teacher_fit_started": False,
              "source_sha256": {"teacher_module": _sha(__file__),
                                "existing_leace_module": _sha(Path(__file__).with_name("acs_protection_maps.py"))}}
    _write_json(out/"teacher_prefit_freeze.json", prefit)
    freeze_hash = _sha(out/"teacher_prefit_freeze.json")
    maps = {}
    for name, labels in (("E", protected), ("S", permutation["protected"])):
        assert _sha(out/"teacher_prefit_freeze.json") == freeze_hash
        assert _sha(permutation_path) == prefit["permutation_file_sha256"]
        maps[name] = fit_joint_leace(raw, labels, fit_split=fit_pool)
        maps[name].save(out/f"map_{name}.npz")
        assert np.array_equal(maps[name].fit_mask, permutation["complete_mask"])
    assert maps["E"].metadata["coverage"] == maps["S"].metadata["coverage"]
    targets = apply_teachers(maps, raw)
    with (out/"fit_targets.npz").open("xb") as handle:
        np.savez_compressed(handle, **targets)
    diagnostics = {}
    for name in ("R", "E", "S"):
        value = targets[name].astype(np.float64)
        moved = (value-raw.astype(np.float64))/scale
        covariance = covariance_diagnostics(value, protected)
        diagnostics[name] = {
            "target_sha256": array_hash(targets[name]), "dimension": 16,
            "movement_from_raw_mean_mse": float(np.mean(moved*moved)),
            "movement_from_raw_per_coordinate_mse": np.mean(moved*moved, axis=0).tolist(),
            "standardized_coordinate_variance": np.var(value/scale, axis=0).tolist(),
            "true_label_complete_case_covariance": covariance,
            "paired_permutation_covariance": covariance_diagnostics(value, permutation["protected"]),
        }
    metadata = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "prefit_freeze_sha256": freeze_hash, "permutation": permutation["metadata"],
                "maps": {key: value.metadata for key, value in maps.items()}, "diagnostics": diagnostics,
                "original_mean": mean.tolist(), "original_scale": scale.tolist(),
                "fit_target_sha256": {key: array_hash(value) for key, value in targets.items()},
                "files_sha256": {path.name: _sha(path) for path in out.iterdir() if path.is_file()},
                "teacher_label_use": {"R": "none", "E": "true representation-fitting protected labels",
                                      "S": "paired permutation of the same complete-case protected labels"},
                "teacher_updated_toward_student": False, "reserved_labels_used": False,
                "scope": "finite label-dependent linear covariance erasure; no privacy or transfer guarantee"}
    _write_json(out/"teachers.json", metadata)
    return {"maps": MappingProxyType(maps), "fit_targets": targets, "metadata": metadata,
            "original_mean": mean, "original_scale": scale}


def load_teachers(directory):
    """Reload and verify existing immutable teacher maps/targets without fitting."""
    directory = Path(directory)
    metadata = json.loads((directory/"teachers.json").read_text())
    for name, digest in metadata["files_sha256"].items():
        if _sha(directory/name) != digest:
            raise ValueError("Teacher artifact changed: "+name)
    maps = {name: FrozenAffineMap.load(directory/f"map_{name}.npz") for name in ("E", "S")}
    for name, eraser in maps.items():
        assert eraser.fingerprint() == metadata["maps"][name]["map_sha256"]
    with np.load(directory/"fit_targets.npz", allow_pickle=False) as arrays:
        targets = {name: _readonly(arrays[name], np.float32) for name in ("R", "E", "S")}
    assert all(array_hash(value) == metadata["fit_target_sha256"][name] for name, value in targets.items())
    return {"maps": MappingProxyType(maps), "fit_targets": MappingProxyType(targets), "metadata": metadata,
            "original_mean": _readonly(metadata["original_mean"], np.float64),
            "original_scale": _readonly(metadata["original_scale"], np.float64)}
