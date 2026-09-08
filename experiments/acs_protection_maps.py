"""Fixed, rank-stabilized joint SEX/RAC1P LEACE for frozen ACS releases.

Only representation-fitting features and the two protected-label arrays enter
the fitter. Two plus nine one-hot columns are concatenated; this is joint
erasure of the two attribute families, not an 18-category interaction target.
Numerical rank choices below are fixed before any task or attacker evaluation.
Empirical covariance diagnostics do not certify classification accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import hashlib
import inspect
import json
from pathlib import Path

from concept_erasure import LeaceEraser
import numpy as np
import torch


SCHEMA = {"SEX": 2, "RAC1P": 9}
NUMERICAL_POLICY = {
    "input_covariance_rtol": 1e-10,
    "projection_svd_rtol": 1e-10,
    "leace_svd_tol": 1e-10,
    "covariance_atol": 1e-10,
    "covariance_rtol": 1e-8,
}
LEACE_OPTIONS = {
    "method": "leace", "affine": True, "shrinkage": False,
    "constrain_cov_trace": False, "svd_tol": NUMERICAL_POLICY["leace_svd_tol"],
}


def _array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def _readonly(value, dtype):
    """Use an immutable bytes owner; setflags(write=True) cannot undo this."""
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _matrix(value, width=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite matrix with at least one column")
    if width is not None and x.shape[1] != width:
        raise ValueError("Application width differs from fitted representation width")
    return x


def _concepts(labels, n):
    if set(labels) != set(SCHEMA):
        raise ValueError("Protected-label whitelist is exactly SEX and RAC1P")
    values, masks, coverage = {}, [], {}
    for name, size in SCHEMA.items():
        value = labels[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.shape != (n,) or not np.isin(value, np.arange(-1, size)).all():
            raise ValueError(f"{name} requires {n} zero-based schema labels; -1 alone denotes missing")
        value = value.astype(np.int64)
        values[name] = value
        known = value >= 0
        masks.append(known)
        coverage[name] = {
            "schema": list(range(size)), "valid_rows": int(known.sum()),
            "missing_rows": int((~known).sum()),
            "support_all_known": np.bincount(value[known], minlength=size).tolist(),
        }
    masks = np.stack(masks)
    complete = masks.all(0)
    z = np.concatenate([np.eye(size, dtype=np.float64)[values[name][complete]]
                        for name, size in SCHEMA.items()], axis=1)
    for name, size in SCHEMA.items():
        support = np.bincount(values[name][complete], minlength=size)
        coverage[name]["support_complete_cases"] = support.tolist()
        coverage[name]["valid_class_mask"] = ((support > 0) & (support < complete.sum())).tolist()
    return z, complete, masks, values, coverage


def _rank(matrix, relative=1e-10):
    values = np.linalg.svd(matrix, compute_uv=False)
    cutoff = relative * float(values[0]) if len(values) else 0.0
    return {"rank": int((values > cutoff).sum()), "singular_values": values.tolist(),
            "relative_tolerance": relative, "absolute_cutoff": cutoff}


def _mean_and_center(x):
    # Anchoring avoids a nonzero computed variance for exactly repeated rows.
    offsets = x - x[0]
    offset_mean = offsets.mean(0)
    return x[0] + offset_mean, offsets - offset_mean


def _covariance(x, z):
    """Unbiased, independently centered fit-sample covariance, in float64."""
    n = len(x)
    if n < 2:
        raise ValueError("At least two complete fitting rows are required")
    mean_x, xc = _mean_and_center(x)
    mean_z, zc = _mean_and_center(z)
    cross = xc.T @ zc / (n - 1)
    cov = xc.T @ xc / (n - 1)
    variance = (zc * zc).sum(0) / (n - 1)
    valid = (z.sum(0) > 0) & (z.sum(0) < n) & (variance > 0)
    return {
        "n": n, "denominator": n - 1, "dtype": "float64",
        "feature_mean": mean_x.tolist(), "target_mean": mean_z.tolist(),
        "target_support": z.sum(0).astype(int).tolist(),
        "target_variance": variance.tolist(), "valid_column_mask": valid.tolist(),
        "schema_complete": bool(valid.all()),
        "cross_covariance": cross.tolist(),
        "cross_covariance_max_abs": float(np.abs(cross).max()),
        "cross_covariance_max_abs_by_column": np.abs(cross).max(0).tolist(),
        "cross_covariance_frobenius": float(np.linalg.norm(cross)),
        "cross_covariance_rank": _rank(cross),
        "feature_covariance_rank": _rank(cov),
        "feature_covariance_trace": float(np.trace(cov)),
    }


def covariance_diagnostics(x, labels):
    """Describe empirical covariance on complete cases of a supplied split.

    This function fits no map or predictor. An absent/constant schema column is
    explicitly invalid even when its empirical cross-covariance is exactly zero.
    """
    x = _matrix(x)
    z, complete, _, _, coverage = _concepts(labels, len(x))
    return {"coverage": coverage, "complete_rows": int(complete.sum()),
            "excluded_rows": int((~complete).sum()),
            **_covariance(x[complete], z)}


@dataclass(frozen=True)
class FrozenAffineMap:
    """Immutable float64 affine map; published application defaults to float32."""
    matrix: np.ndarray
    mean: np.ndarray
    fit_mask: np.ndarray
    known_masks: np.ndarray
    _metadata_json: str

    def __post_init__(self):
        for name, dtype in (("matrix", np.float64), ("mean", np.float64),
                            ("fit_mask", bool), ("known_masks", bool)):
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype))
        d = len(self.mean)
        if self.matrix.shape != (d, d) or self.known_masks.shape != (2, len(self.fit_mask)):
            raise ValueError("Invalid affine map or mask dimensions")
        if not np.isfinite(self.matrix).all() or not np.isfinite(self.mean).all():
            raise ValueError("Affine map must be finite")
        if not np.array_equal(self.fit_mask, self.known_masks.all(0)):
            raise ValueError("Fit mask differs from complete protected-label cases")
        json.loads(self._metadata_json)

    def fingerprint(self):
        digest = hashlib.sha256(self._metadata_json.encode())
        for name in ("matrix", "mean", "fit_mask", "known_masks"):
            digest.update(name.encode())
            digest.update(_array_hash(getattr(self, name)).encode())
        return digest.hexdigest()

    @property
    def metadata(self):
        # Returning a new dictionary cannot mutate fitted provenance.
        return {**json.loads(self._metadata_json), "map_sha256": self.fingerprint()}

    def apply(self, x, *, dtype=np.float32):
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("Application dtype must be float32 or float64")
        x = _matrix(x, len(self.mean))
        # The full-collapse case contains exact zeros. Every row then receives
        # exactly the same fitted mean, with no residual for a scaler to amplify.
        released = (x - self.mean) @ self.matrix.T + self.mean
        return released.astype(dtype, copy=False)

    def save(self, path):
        """Exclusive creation preserves any existing map artifact."""
        with Path(path).open("xb") as handle:
            np.savez_compressed(handle, matrix=self.matrix, mean=self.mean,
                                fit_mask=self.fit_mask, known_masks=self.known_masks,
                                metadata_json=np.array(self._metadata_json),
                                map_sha256=np.array(self.fingerprint()))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as record:
            result = cls(record["matrix"], record["mean"], record["fit_mask"],
                         record["known_masks"], str(record["metadata_json"]))
            if result.fingerprint() != str(record["map_sha256"]):
                raise ValueError("Saved affine map fingerprint mismatch")
        return result


def fit_joint_leace(x_repr_fit, labels, *, fit_split="representation_fit"):
    """Fit one fixed full-strength joint map, using existing LEACE mathematics.

    Keep covariance eigenvalues strictly above 1e-10 times the largest value,
    fit concept-erasure's float64 affine LEACE in those retained coordinates,
    lift to the original width, and zero projection singular values at or below
    1e-10 * max(1, largest singular value). Preserve the original complete-case
    fitting mean. These numerical thresholds are not erasure-strength settings.
    """
    if fit_split != "representation_fit":
        raise ValueError("Protection maps may be fitted only on representation_fit")
    original = np.asarray(x_repr_fit.detach().cpu().numpy()
                          if isinstance(x_repr_fit, torch.Tensor) else x_repr_fit)
    x = _matrix(original)
    z, complete, known_masks, values, coverage = _concepts(labels, len(x))
    if complete.sum() < 2:
        raise ValueError("At least two complete fitting rows are required")
    fitting = x[complete]
    mean, centered = _mean_and_center(fitting)
    covariance = centered.T @ centered / (len(fitting) - 1)
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    cutoff = max(0.0, float(eigenvalues[-1])) * NUMERICAL_POLICY["input_covariance_rtol"]
    retained = eigenvalues > cutoff
    basis = vectors[:, retained]
    if retained.any():
        reduced = centered @ basis
        eraser = LeaceEraser.fit(torch.from_numpy(reduced), torch.from_numpy(z), **LEACE_OPTIONS)
        projection = basis @ eraser.P.detach().cpu().numpy() @ basis.T
    else:
        projection = np.zeros_like(covariance)
    left, singular, right = np.linalg.svd(projection, full_matrices=False)
    projection_cutoff = NUMERICAL_POLICY["projection_svd_rtol"] * max(1.0, float(singular[0]))
    kept = singular > projection_cutoff
    projection = ((left[:, kept] * singular[kept]) @ right[kept]
                  if kept.any() else np.zeros_like(projection))
    # Match apply()'s operation order when diagnosing the published release.
    released = (fitting - mean) @ projection.T + mean
    before = _covariance(fitting, z)
    after = _covariance(released, z)
    after32 = _covariance(released.astype(np.float32).astype(np.float64), z)
    threshold = (NUMERICAL_POLICY["covariance_atol"]
                 + NUMERICAL_POLICY["covariance_rtol"] * before["cross_covariance_max_abs"])
    for record in (after, after32):
        record["covariance_threshold"] = threshold
        record["empirical_covariance_within_tolerance"] = (
            record["schema_complete"] and record["cross_covariance_max_abs"] <= threshold)
    # An unmodified library fit is a fitting-data numerical diagnostic only. It
    # cannot influence the fixed rank choices or select a method.
    try:
        raw = LeaceEraser.fit(torch.from_numpy(fitting), torch.from_numpy(z), **LEACE_OPTIONS)
        raw_output = raw(torch.from_numpy(fitting)).detach().cpu().numpy()
        if not np.isfinite(raw_output).all():
            raise FloatingPointError("Unstabilized library output is nonfinite")
        library = {"status": "ok", "max_absolute_output_difference": float(np.abs(raw_output - released).max()),
                   "covariance": _covariance(raw_output, z),
                   "projection_rank": _rank(raw.P.detach().cpu().numpy())}
    except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        library = {"status": "numerical_failure", "error": str(error),
                   "max_absolute_output_difference": None}
    package_source = Path(inspect.getfile(LeaceEraser))
    metadata = {
        "method": "rank-stabilized full joint LEACE", "fit_split": fit_split,
        "class_schema": SCHEMA, "concept_columns": [f"{k}={j}" for k, n in SCHEMA.items() for j in range(n)],
        "concept_encoding": "concatenated SEX2 and RAC1P9 one-hot columns, not intersection labels",
        "missing_policy": "-1 is missing; complete cases of both attributes only; fixed schema retained",
        "coverage": coverage, "input_rows": len(x), "fit_rows": len(fitting),
        "excluded_rows": int((~complete).sum()), "input_dim": x.shape[1], "output_dim": x.shape[1],
        "input_array_sha256": _array_hash(original), "fitting_float64_sha256": _array_hash(fitting),
        "label_sha256": {k: _array_hash(v) for k, v in values.items()},
        "fit_mask_sha256": _array_hash(complete), "known_masks_sha256": _array_hash(known_masks),
        "numerical_policy": NUMERICAL_POLICY, "leace_options": LEACE_OPTIONS,
        "centering": "original complete-case representation_fit mean; (x-mean) @ P.T + mean",
        "covariance_definition": "independently centered complete fitting rows, denominator n-1, float64",
        "fit_dtype": "float64", "application_compute_dtype": "float64", "default_output_dtype": "float32",
        "input_covariance_eigenvalues": eigenvalues.tolist(), "input_covariance_cutoff": cutoff,
        "input_retained_rank": int(retained.sum()), "input_discarded_rank": int((~retained).sum()),
        "discarded_covariance_trace": float(np.maximum(eigenvalues[~retained], 0).sum()),
        "projection_singular_values_before_cleanup": singular.tolist(), "projection_cutoff": projection_cutoff,
        "projection_retained_rank": int(kept.sum()), "exact_constant_output": bool(not kept.any()),
        "maximum_mean_preservation_error": float(np.abs(released.mean(0) - mean).max()),
        "before": before, "after_float64": after, "after_float32": after32,
        "unmodified_library_comparison": library,
        "concept_erasure_version": version("concept-erasure"),
        "concept_erasure_source_sha256": hashlib.sha256(package_source.read_bytes()).hexdigest(),
        "module_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "claim_scope": "empirical fitting-sample covariance only; no universal classification guarantee",
    }
    return FrozenAffineMap(projection, mean, complete, known_masks,
                           json.dumps(metadata, sort_keys=True, allow_nan=False))
