"""Frozen affine recovery of raw, retained and removed teacher components.

All targets use original PCA16 scales. Small erased/residual variances are
reported without whitening or division into apparently large recovery claims.
Affine decoders and their intercepts use representation-fitting examples only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.acs_protection_maps import _readonly
from experiments.acs_selective_teachers import apply_teachers
from experiments.acs_transfer_data import array_hash


RCOND = 1e-12
VARIANCE_FLOOR = 1e-12
BASE_TARGETS = ("raw", "E", "qE")
SHAM_TARGETS = ("S", "qS")


def _matrix(value):
    x = np.asarray(value, np.float64)
    if x.ndim != 2 or x.shape[1] != 16 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Require nonempty finite16-coordinate release or teacher arrays")
    return x


def _statistics(preprocessing):
    mean = np.asarray(preprocessing["mean"], np.float64)[:16]
    scale = np.asarray(preprocessing["scale"], np.float64)[:16]
    if mean.shape != (16,) or scale.shape != (16,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Original finite fitting means and positive scales required")
    return mean, scale


def components(raw, maps, preprocessing, *, include_sham=False):
    """Published float32 teachers, then float64 original-scale decomposition."""
    mean, scale = _statistics(preprocessing)
    targets = apply_teachers(maps, raw)
    r, e = targets["R"].astype(np.float64), targets["E"].astype(np.float64)
    result = {"raw": (r-mean)/scale, "E": (e-mean)/scale, "qE": (r-e)/scale}
    if include_sham:
        s = targets["S"].astype(np.float64)
        result.update(S=(s-mean)/scale, qS=(r-s)/scale)
    return {name: _readonly(value, np.float64) for name, value in result.items()}


def _rank(value):
    singular = np.linalg.svd(value, compute_uv=False)
    cutoff = RCOND*float(singular[0]) if len(singular) else 0.
    return {"rank": int((singular > cutoff).sum()), "singular_values": singular.tolist(),
            "relative_tolerance": RCOND, "absolute_cutoff": cutoff}


def fit_affine(release, standardized_target, *, fit_pool="representation_fit", rcond=RCOND):
    """One fixed float64 solve; no validation/evaluation arguments or selection."""
    if fit_pool != "representation_fit" or rcond != RCOND:
        raise ValueError("Fixed representation_fit pool and rcond1e-12 required")
    x, y = _matrix(release), _matrix(standardized_target)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Aligned fitting release/target rows required")
    design = np.column_stack((x, np.ones(len(x))))
    coefficient, _, rank, singular = np.linalg.lstsq(design, y, rcond=rcond)
    prior = y.mean(0)
    variance = np.mean((y-prior)**2, axis=0)
    metadata = {"fit_pool": fit_pool, "fit_rows": len(x), "intercept_fitted": True,
                "rcond": rcond, "rank": int(rank), "singular_values": singular.tolist(),
                "design_rank": int(rank), "design_singular_values": singular.tolist(),
                "release_centered_rank": _rank(x-x.mean(0)),
                "target_centered_rank": _rank(y-prior), "target_uncentered_rank": _rank(y),
                "fit_target_variance": variance.tolist(), "fit_target_variance_mean": float(variance.mean()),
                "fit_prior": prior.tolist(), "fit_release_sha256": array_hash(x),
                "fit_target_sha256": array_hash(y), "coefficient_sha256": array_hash(coefficient),
                "target_scale": "original raw PCA16 fitting scales; never target/residual variance",
                "variance_floor": VARIANCE_FLOOR, "selection": "none"}
    return {"coefficient": _readonly(coefficient, np.float64), "prior": _readonly(prior, np.float64),
            "fit_variance": _readonly(variance, np.float64), "metadata": metadata}


def evaluate_affine(fitted, release, standardized_target):
    x, y = _matrix(release), _matrix(standardized_target)
    if len(x) != len(y):
        raise ValueError("Aligned frozen release/target rows required")
    coefficient_before = array_hash(fitted["coefficient"])
    prediction = np.column_stack((x, np.ones(len(x)))) @ fitted["coefficient"]
    mse = np.mean((prediction-y)**2, axis=0)
    prior_mse = np.mean((y-fitted["prior"])**2, axis=0)
    variance = np.var(y, axis=0)
    # A shifted evaluation target can have a large prior error despite being
    # constant. Require nonnegligible fitting AND evaluated target variance.
    coordinate_defined = (fitted["fit_variance"] > VARIANCE_FLOOR) & (variance > VARIANCE_FLOOR) & (prior_mse > VARIANCE_FLOOR)
    mean_defined = (float(fitted["fit_variance"].mean()) > VARIANCE_FLOOR
                    and float(variance.mean()) > VARIANCE_FLOOR and float(prior_mse.mean()) > VARIANCE_FLOOR)
    assert array_hash(fitted["coefficient"]) == coefficient_before
    return {"rows": len(x), "mean_mse": float(mse.mean()), "per_coordinate_mse": mse.tolist(),
            "prior_mean_mse": float(prior_mse.mean()), "prior_per_coordinate_mse": prior_mse.tolist(),
            "target_variance_mean": float(variance.mean()), "target_variance_per_coordinate": variance.tolist(),
            "mse_over_prior": float(mse.mean()/prior_mse.mean()) if mean_defined else None,
            "per_coordinate_mse_over_prior": [float(a/b) if defined else None for a, b, defined in zip(mse, prior_mse, coordinate_defined)],
            "ratio_defined": bool(mean_defined), "per_coordinate_ratio_defined": coordinate_defined.tolist(),
            "variance_floor": VARIANCE_FLOOR,
            "ratio_rule": "defined only if fitting/evaluation target variance and evaluation fitting-prior MSE all exceed1e-12",
            "evaluation_target_centered_rank": _rank(y-y.mean(0))}


def direct_errors(release, raw, maps, preprocessing):
    x = _matrix(release)
    targets = apply_teachers(maps, raw)
    _, scale = _statistics(preprocessing)
    if len(x) != len(raw):
        raise ValueError("Aligned release and immutable teachers required")
    result = {}
    for name, target in targets.items():
        error = np.mean(((x-target.astype(np.float64))/scale)**2, axis=0)
        result[name] = {"mean_mse": float(error.mean()), "per_coordinate_mse": error.tolist()}
    return result


def fit_snapshots(releases, pca, preprocessing, maps, directory, teacher_by_release):
    """Freeze every affine fit before accessing diagnostic held-out features."""
    if set(releases) != set(teacher_by_release) or set(teacher_by_release.values())-{"R", "E", "S"}:
        raise ValueError("Every snapshot requires its predeclared R/E/S teacher identity")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw_fit = np.asarray(pca["representation_fit"])[:, :16]
    raw_val = np.asarray(pca["source_validation"])[:, :16]
    # Components are label-free applications of previously frozen maps.
    fit_targets = components(raw_fit, maps, preprocessing, include_sham=True)
    fitted, report = {}, {"snapshots": {}, "rcond": RCOND, "variance_floor": VARIANCE_FLOOR,
        "target_definitions": {"raw": "(raw-mu)/scale", "E": "(publishedE-mu)/scale", "qE": "(raw-publishedE)/scale",
                               "S": "(publishedS-mu)/scale", "qS": "(raw-publishedS)/scale"},
        "scope": "recoverability by independently fitted affine maps; high error does not exclude nonlinear recovery",
        "removed_component_scope": "removed residual is not pure sensitive information; components need not be orthogonal",
        "map_hashes": {name: eraser.fingerprint() for name, eraser in maps.items()},
        "original_mean": np.asarray(preprocessing["mean"])[:16].tolist(),
        "original_scale": np.asarray(preprocessing["scale"])[:16].tolist(),
        "fit_pool": "representation_fit", "selection": "none"}
    for name, pools in releases.items():
        teacher = teacher_by_release[name]
        target_names = (*BASE_TARGETS, *SHAM_TARGETS) if teacher == "S" else BASE_TARGETS
        fitted[name] = {target: fit_affine(pools["representation_fit"], fit_targets[target]) for target in target_names}
        report["snapshots"][name] = {"teacher": teacher, "targets": {}, "direct_teacher_error": {}}
        for target, decoder in fitted[name].items():
            path = directory/f"affine_{name}_{target}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, coefficient=decoder["coefficient"], prior=decoder["prior"], fit_variance=decoder["fit_variance"])
            report["snapshots"][name]["targets"][target] = {**decoder["metadata"],
                "artifact": path.name, "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # No validation inputs entered any target centers, intercepts or coefficients.
    val_targets = components(raw_val, maps, preprocessing, include_sham=True)
    for name, pools in releases.items():
        record = report["snapshots"][name]
        for pool, label, raw, target_arrays in (("representation_fit", "fit", raw_fit, fit_targets),
                                               ("source_validation", "source_validation", raw_val, val_targets)):
            record["direct_teacher_error"][label] = direct_errors(pools[pool], raw, maps, preprocessing)
            for target, decoder in fitted[name].items():
                record["targets"][target][label] = evaluate_affine(decoder, pools[pool], target_arrays[target])
    return fitted, report


def evaluate_snapshots(fitted, report, releases, pca, preprocessing, maps):
    """Post-freeze development-input evaluation; no teacher or decoder refit."""
    raw = np.asarray(pca)[:, :16]
    targets = components(raw, maps, preprocessing, include_sham=True)
    for name, x in releases.items():
        record = report["snapshots"][name]
        record["direct_teacher_error"]["development_evaluation"] = direct_errors(x, raw, maps, preprocessing)
        for target, decoder in fitted[name].items():
            record["targets"][target]["development_evaluation"] = evaluate_affine(decoder, x, targets[target])
    return report
