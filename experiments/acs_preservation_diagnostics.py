"""Label-free frozen-release diagnostics; no task labels or model selection."""
from pathlib import Path
import numpy as np
from experiments.acs_transfer_data import array_hash


def fit_affine(release, teacher, mean, scale, *, rcond=1e-12, fit_pool='representation_fit'):
    if fit_pool != 'representation_fit' or rcond != 1e-12:
        raise ValueError('Fixed representation-fitting pool and numerical tolerance required')
    x = np.asarray(release, dtype=np.float64)
    y = (np.asarray(teacher, dtype=np.float64) - np.asarray(mean)[:16]) / np.asarray(scale)[:16]
    if x.ndim != 2 or y.shape != (len(x), 16) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite paired release and PCA16 teacher matrices required')
    design = np.column_stack((x, np.ones(len(x))))
    coefficient, _, rank, singular = np.linalg.lstsq(design, y, rcond=rcond)
    return {'coefficient': coefficient, 'prior': y.mean(0), 'rank': int(rank),
            'singular_values': singular, 'rank_tolerance': rcond,
            'fit_release_sha256': array_hash(x), 'fit_teacher_sha256': array_hash(y),
            'fit_pool': fit_pool, 'intercept_fitted': True}


def evaluate_affine(fitted, release, teacher, mean, scale):
    x, raw = np.asarray(release, dtype=np.float64), np.asarray(teacher, dtype=np.float64)
    scale = np.asarray(scale)[:16]
    y = (raw - np.asarray(mean)[:16]) / scale
    prediction = np.column_stack((x, np.ones(len(x)))) @ fitted['coefficient']
    errors = {'direct': ((x - raw) / scale) ** 2,
              'affine': (prediction - y) ** 2, 'prior': (y - fitted['prior']) ** 2}
    return {'rows': len(x), **{f'{k}_mean_mse': float(v.mean()) for k, v in errors.items()},
            **{f'{k}_per_coordinate_mse': v.mean(0).tolist() for k, v in errors.items()}}


def fit_snapshots(releases, pca, preprocessing, directory):
    result, fitted = {}, {}
    mean, scale = preprocessing['mean'], preprocessing['scale']
    for name, pools in releases.items():
        decoder = fit_affine(pools['representation_fit'], pca['representation_fit'][:, :16], mean, scale)
        fitted[name] = decoder
        result[name] = {k: v for k, v in decoder.items() if k not in ('coefficient', 'prior', 'singular_values')}
        result[name]['singular_values'] = decoder['singular_values'].tolist()
        for pool, label in (('representation_fit', 'fit'), ('source_validation', 'source_validation')):
            result[name][label] = evaluate_affine(decoder, pools[pool], pca[pool][:, :16], mean, scale)
        path = Path(directory) / f'affine_{name}.npz'
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, coefficient=decoder['coefficient'], prior=decoder['prior'])
    return fitted, {'snapshots': result, 'selection': 'none; fixed rcond=1e-12; intercept and coefficients representation-fit only'}


def evaluate_snapshots(fitted, report, releases, pca, preprocessing):
    for name, x in releases.items():
        report['snapshots'][name]['development_evaluation'] = evaluate_affine(
            fitted[name], x, pca[:, :16], preprocessing['mean'], preprocessing['scale'])
    return report
