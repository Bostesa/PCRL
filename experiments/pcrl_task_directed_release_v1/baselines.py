"""Supervised affine baselines on [standardized PCA32, residence-teacher logit].

Only the caller's designated teacher-fitting rows are accepted. This module has no
raw-data loader, split discovery, H_A/H_B argument, or historical artifact resolver.
The caller freezes/clips the teacher logit before assembling the 33-column input.
H_A is appended unchanged by the release assembler, outside both fitted maps.

LEACE follows Belrose et al., https://arxiv.org/abs/2306.03819 (Thm 4.2/4.3).
SPLINCE follows Holstege et al., https://arxiv.org/html/2506.10703v1 (Thm 1,
Eq 4; that preprint spells the name SPLICE). These are supervised task adaptations,
not the historical A0 fits: SPLINCE preserves residence, income, and employment
covariances simultaneously. Neither method guarantees mutual-information privacy,
nonlinear privacy, population privacy, or privacy of the H_A-augmented release.

In the retained covariance support W is a rectangular pseudoinverse square root,
L=W^+ its inverse, and B spans col(W C_xz). LEACE is I-L B B' W. For SPLINCE,
U spans B-perp and V spans col(W C_xy)+(col(W C_xz)+col(W C_xy))-perp; the map on
the support is L V (U' V)^-1 U' W. Both maps act as identity off covariance support;
this choice leaves all fitting outputs unchanged and avoids inventing an off-support
zeroing intervention. Exact span intersection is infeasibility, but a truncated
numerical rank cannot prove exact intersection. Refusals are numerically unresolved
unless an exact label-relation and nonzero-covariance certificate proves conflict.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

INPUT_WIDTH = 33
PROTECTED = {'SEX': 2, 'RAC1P': 9}
TASKS = ('same_residence', 'income_binary', 'civilian_at_work')
TOLERANCES = {
    'covariance_eigenvalue_rtol': 1e-10,
    'subspace_svd_rtol': 1e-10,
    'subspace_svd_atol': 1e-12,
    'constraint_atol': 1e-9,
    'constraint_rtol': 1e-8,
    'splince_condition_max': 1e6,
    'splince_inverse_norm_max': 1e6,
}
SCOPE = ('Empirical fitting-moment affine guardedness only; not mutual information, '
         'nonlinear protection, population protection, or augmented-release protection.')


def _array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _features(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != INPUT_WIDTH:
        raise ValueError('Auxiliary input must have exactly 33 columns: PCA32 plus teacher logit; no H')
    if not np.isfinite(a).all():
        raise ValueError('Auxiliary features must be finite')
    return a


def _labels(values, schema, n, name):
    if set(values) != set(schema):
        raise ValueError(f'{name} requires exactly {list(schema)}')
    out = {}
    for key, classes in schema.items():
        a = np.asarray(values[key])
        if (a.shape != (n,) or not np.issubdtype(a.dtype, np.number)
                or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all()
                or not np.isin(a, np.arange(-1, classes)).all()):
            raise ValueError(f'{key} requires {n} finite integer schema labels; -1 denotes missing')
        out[key] = a.astype(np.int64)
    return out


def _basis(a):
    """Rank at the fixed absolute/relative numerical resolution; no parameter search."""
    if not a.size:
        return np.empty((a.shape[0], 0)), np.empty(0)
    u, singular, _ = np.linalg.svd(a, full_matrices=False)
    cutoff = max(TOLERANCES['subspace_svd_atol'],
                 TOLERANCES['subspace_svd_rtol'] * singular[0])
    return u[:, singular > cutoff], singular


def _complement(b, dim):
    if not b.shape[1]:
        return np.eye(dim)
    # Full QR supplies the complement directly, without ranking floating-point I-BB'.
    q, _ = np.linalg.qr(b, mode='complete')
    return q[:, b.shape[1]:]


def _maxabs(a):
    return float(np.max(np.abs(a))) if a.size else 0.0


def _bound(reference):
    return TOLERANCES['constraint_atol'] + TOLERANCES['constraint_rtol'] * _maxabs(reference)


def _exact_label_conflict(x, protected, tasks, rows):
    """Sufficient exact conflict certificate; absence does not establish feasibility.

    If a task is exactly a protected one-hot (or its complement), their covariance
    vectors are identical (or negatives). A nonzero coordinate cannot be erased and
    preserved. Rational arithmetic on the actual finite IEEE input values certifies
    that coordinate without an SVD threshold or an approximate equality test.
    """
    n = len(rows)
    for task_name in TASKS:
        target = tasks[task_name][rows]
        for protected_name in PROTECTED:
            labels = protected[protected_name][rows]
            for value in np.unique(labels):
                indicator = labels == value
                relation = ('equal' if np.array_equal(target, indicator) else
                            'complement' if np.array_equal(target, ~indicator) else None)
                if relation is None:
                    continue
                count = int(indicator.sum())
                if count in (0, n):
                    continue  # a constant label has zero covariance with every feature
                for feature in range(INPUT_WIDTH):
                    column = x[rows, feature]
                    positive = sum((Fraction(float(v)) for v in column[indicator]), Fraction())
                    negative = sum((Fraction(float(v)) for v in column[~indicator]), Fraction())
                    covariance = ((n - count) * positive - count * negative) / (n * (n - 1))
                    if covariance:
                        return {
                            'kind': 'exact_binary_label_relation_and_nonzero_covariance',
                            'task': task_name, 'protected': protected_name,
                            'protected_class': int(value), 'task_relation_to_one_hot': relation,
                            'feature_index': feature,
                            'exact_covariance_numerator': str(covariance.numerator),
                            'exact_covariance_denominator': str(covariance.denominator),
                            'arithmetic': 'exact rational values of finite supplied float64 features',
                            'proof': 'task covariance equals plus/minus a nonzero protected covariance; '
                                     'erasing the latter and preserving the former are incompatible',
                        }
    return None


@dataclass(frozen=True)
class AffineEraser:
    """Auxiliary-only, mean-preserving affine map with portable non-pickle persistence."""
    projection: np.ndarray
    mean: np.ndarray
    metadata: dict
    arrays: dict

    def transform(self, x):
        x = _features(x)
        with np.errstate(over='ignore', invalid='ignore'):
            value = (x - self.mean) @ self.projection.T + self.mean
        if not np.isfinite(value).all():
            raise FloatingPointError('Affine transformation produced nonfinite output')
        return np.ascontiguousarray(value)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / 'diagnostics.json').read_text())
        if metadata['status'] != 'fitted':
            raise ValueError('No fitted map exists for this diagnostic')
        path = directory / 'map.npz'
        if _file_hash(path) != metadata['map_sha256']:
            raise ValueError('Serialized affine map hash mismatch')
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {key: bundle[key] for key in bundle.files}
        p, mean = arrays['projection'], arrays['mean']
        if (p.shape != (INPUT_WIDTH, INPUT_WIDTH) or mean.shape != (INPUT_WIDTH,)
                or not np.isfinite(p).all() or not np.isfinite(mean).all()):
            raise ValueError('Invalid or nonfinite serialized affine map')
        return cls(p, mean, metadata, arrays)


def _moments(x, protected, tasks, rows, method):
    if len(rows) < 2:
        raise ValueError(f'{method}: at least two complete fitting rows required')
    xf = x[rows]
    mean = xf.mean(0)
    xc = xf - mean
    observed = {key: np.unique(protected[key][rows]).tolist() for key in PROTECTED}
    z = np.column_stack([protected[key][rows] == value
                         for key in PROTECTED for value in observed[key]]).astype(np.float64)
    z -= z.mean(0)
    # LEACE never imputes or uses task labels in its fitting moments.
    y = (np.column_stack([tasks[key][rows] for key in TASKS]).astype(np.float64)
         if method == 'splince_supervised' else np.empty((len(rows), 0)))
    if y.shape[1]:
        y -= y.mean(0)
    denom = len(rows) - 1
    with np.errstate(over='ignore', invalid='ignore'):
        covariance, cxz, cxy = xc.T @ xc / denom, xc.T @ z / denom, xc.T @ y / denom
    if not all(np.isfinite(a).all() for a in (mean, covariance, cxz, cxy)):
        raise ValueError('Nonfinite empirical covariance; refuse unstable fitting')
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
    cutoff = TOLERANCES['covariance_eigenvalue_rtol'] * max(float(eigenvalues[-1]), 0.)
    keep = eigenvalues > cutoff
    basis, eig = eigenvectors[:, keep], eigenvalues[keep]
    whitener = (basis / np.sqrt(eig)).T
    unwhitener = basis * np.sqrt(eig)
    null_projection = np.eye(INPUT_WIDTH) - basis @ basis.T
    arrays = {'mean': mean, 'covariance_xx': covariance, 'covariance_xz': cxz,
              'covariance_xy': cxy, 'whitener': whitener, 'unwhitener': unwhitener,
              'covariance_eigenvalues': eigenvalues, 'fit_row_indices': rows,
              'covariance_null_projection': null_projection}
    metadata = {
        'method': method, 'status': 'pending', 'scope': SCOPE,
        'input_definition': ['standardized_PCA_0_to_31', 'frozen_clipped_residence_teacher_logit'],
        'input_dim': INPUT_WIDTH, 'output_dim': INPUT_WIDTH, 'H_inputs_accepted': False,
        'fit_scope': 'caller-supplied designated teacher-fitting rows only; no split discovery',
        'provided_rows': len(x), 'fit_rows': len(rows), 'excluded_rows': len(x) - len(rows),
        'missing_label_policy': 'exclude complete cases; never impute labels',
        'fit_row_indices_are': 'indices relative to caller-provided teacher-fitting slice',
        'protected_observed_classes': observed,
        'protected_class_counts': {key: {str(v): int(np.sum(protected[key][rows] == v))
                                        for v in observed[key]} for key in PROTECTED},
        'concept_encoding': 'concatenated one-hot over observed classes of each attribute',
        'preservation_targets': list(TASKS) if method == 'splince_supervised' else [],
        'covariance_denominator': denom, 'covariance_support_rank': int(keep.sum()),
        'covariance_eigenvalue_cutoff': cutoff, 'tolerances': dict(TOLERANCES),
        'off_support_extension': 'identity; empirical least-change problem does not identify it',
        'fitting_input_sha256': _array_hash(xf), 'fit_row_indices_sha256': _array_hash(rows),
        'protected_labels_sha256': {key: _array_hash(protected[key][rows]) for key in PROTECTED},
        'task_labels_sha256': ({key: _array_hash(tasks[key][rows]) for key in TASKS}
                              if method == 'splince_supervised' else {}),
        'fit_weighted': False, 'regularization_sweep': False, 'fallback_used': False,
        'source': ('https://arxiv.org/abs/2306.03819' if method == 'leace_supervised'
                   else 'https://arxiv.org/html/2506.10703v1#S3.SS1'),
    }
    return arrays, metadata


def _finish(projection, arrays, metadata, *, preserve):
    cxz, cxy = arrays['covariance_xz'], arrays['covariance_xy']
    guardedness = _maxabs(projection @ cxz)
    preservation = _maxabs(projection @ cxy - cxy)
    metadata.update(
        cross_covariance_before_max_abs=_maxabs(cxz),
        cross_covariance_after_max_abs=guardedness,
        guardedness_tolerance=_bound(cxz),
        task_covariance_preservation_max_abs_error=preservation,
        preservation_tolerance=_bound(cxy),
        idempotency_max_abs_error=_maxabs(projection @ projection - projection),
        projection_rank=int(np.linalg.matrix_rank(projection, tol=TOLERANCES['subspace_svd_atol'])),
        expected_squared_distortion=float(np.trace((projection - np.eye(INPUT_WIDTH)) @
            arrays['covariance_xx'] @ (projection - np.eye(INPUT_WIDTH)).T)),
    )
    if (not np.isfinite(projection).all() or guardedness > _bound(cxz)
            or (preserve and preservation > _bound(cxy))):
        metadata.update(status='numerically_unresolved', reason='affine_constraint_residual_failed')
        return metadata
    metadata['status'] = 'fitted'
    arrays = {**arrays, 'projection': projection, 'affine_bias': arrays['mean'] - projection @ arrays['mean']}
    return AffineEraser(projection, arrays['mean'], metadata, arrays)


def _fit_one(x, protected, tasks, rows, method):
    arrays, metadata = _moments(x, protected, tasks, rows, method)
    w, l = arrays['whitener'], arrays['unwhitener']
    protected_basis, sz = _basis(w @ arrays['covariance_xz'])
    metadata['protected_covariance_rank'] = protected_basis.shape[1]
    arrays['protected_whitened_singular_values'] = sz
    if method == 'leace_supervised':
        p = np.eye(INPUT_WIDTH) - l @ protected_basis @ protected_basis.T @ w
        return _finish(p, arrays, metadata, preserve=False), arrays

    task_basis, sy = _basis(w @ arrays['covariance_xy'])
    combined_basis, scombined = _basis(np.column_stack([protected_basis, task_basis]))
    intersection = protected_basis.shape[1] + task_basis.shape[1] - combined_basis.shape[1]
    metadata.update(task_covariance_rank=task_basis.shape[1], joint_subspace_rank=combined_basis.shape[1],
                    intersection_dimension=int(intersection),
                    intersection_dimension_interpretation='at declared numerical rank; not an exact certificate',
                    infeasibility_certificate=None,
                    feasibility_rule='exact col(W C_xz) intersect col(W C_xy) = {0}',
                    feasibility_test='numerical subspace rank with optional exact label-conflict certificate')
    arrays.update(task_whitened_singular_values=sy, combined_basis_singular_values=scombined)
    if intersection:
        certificate = _exact_label_conflict(x, protected, tasks, rows)
        metadata.update(infeasibility_certificate=certificate)
        if certificate is not None:
            metadata.update(status='infeasible', reason='exact_protected_task_covariance_conflict')
        else:
            metadata.update(status='numerically_unresolved',
                            reason='numerical_rank_intersection_without_exact_certificate')
        return metadata, arrays
    dim = w.shape[0]
    u = _complement(protected_basis, dim)
    uminus = _complement(combined_basis, dim)
    v, _ = _basis(np.column_stack([task_basis, uminus]))
    inner = u.T @ v
    if inner.shape[0] != inner.shape[1]:
        metadata.update(status='numerically_unresolved', reason='complement_dimension_mismatch')
        return metadata, arrays
    singular = np.linalg.svd(inner, compute_uv=False) if inner.size else np.ones(1)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float('inf')
    inverse_norm = float(1 / singular[-1]) if singular[-1] > 0 else float('inf')
    metadata['condition_UtV'] = condition if np.isfinite(condition) else None
    metadata['inverse_norm_UtV'] = inverse_norm if np.isfinite(inverse_norm) else None
    arrays['UtV_singular_values'] = singular
    # cond([epsilon]) == 1: relative conditioning alone misses a nearly intersecting
    # one-dimensional kernel/range. Also bound the absolute inverse amplification.
    if (not np.isfinite(condition) or condition > TOLERANCES['splince_condition_max']
            or not np.isfinite(inverse_norm)
            or inverse_norm > TOLERANCES['splince_inverse_norm_max']):
        metadata.update(status='numerically_unresolved', reason='condition_gate_exceeded')
        return metadata, arrays
    middle = np.linalg.solve(inner, u.T) if inner.size else np.empty((0, dim))
    p = l @ v @ middle @ w + arrays['covariance_null_projection']
    return _finish(p, arrays, metadata, preserve=True), arrays


def fit_supervised_erasers(x_fit, protected: dict, task_labels: dict, *, out_dir):
    """Fit/save both full-channel adaptations; returned maps transform auxiliary inputs only.

    Missing (-1) protected labels exclude rows from both methods; missing task labels
    additionally exclude rows from SPLINCE. No rows are sampled, expanded or fetched.
    SPLINCE infeasibility/numerical refusal is a diagnostic dict, never a fallback.
    A numerical LEACE failure raises instead of returning an unguarded named baseline.
    Existing named output directories are refused; callers provide a new run location.
    """
    x = _features(x_fit)
    protected = _labels(protected, PROTECTED, len(x), 'protected labels')
    tasks = _labels(task_labels, {name: 2 for name in TASKS}, len(x), 'task labels')
    sensitive_complete = np.stack([a >= 0 for a in protected.values()]).all(0)
    all_complete = sensitive_complete & np.stack([a >= 0 for a in tasks.values()]).all(0)
    names = ('leace_supervised', 'splince_supervised')
    out_dir = Path(out_dir)
    if any((out_dir / name).exists() for name in names):
        raise FileExistsError('Refusing to overwrite an existing fitted-baseline directory')
    results = {}
    pending = []
    for name, mask in zip(names, (sensitive_complete, all_complete)):
        result, arrays = _fit_one(x, protected, tasks, np.flatnonzero(mask), name)
        if name == 'leace_supervised' and isinstance(result, dict):
            raise FloatingPointError(f'LEACE numerical constraint failure: {result}')
        results[name] = result
        pending.append((name, result, arrays))
    for name, result, arrays in pending:
        directory = out_dir / name
        directory.mkdir(parents=True, exist_ok=False)
        metadata = result.metadata if isinstance(result, AffineEraser) else result
        # Even infeasibility keeps the exact centering/covariances and row-mask evidence.
        path = directory / ('map.npz' if isinstance(result, AffineEraser) else 'fit_moments.npz')
        with path.open('xb') as handle:
            np.savez_compressed(handle, **(result.arrays if isinstance(result, AffineEraser) else arrays))
        metadata['map_sha256' if isinstance(result, AffineEraser) else 'fit_moments_sha256'] = _file_hash(path)
        with (directory / 'diagnostics.json').open('x') as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write('\n')
    return results
