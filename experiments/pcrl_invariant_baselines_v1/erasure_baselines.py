"""Stages 5b and 5c: LEACE and SPLINCE applied to the frozen auxiliary channel.

Both baselines transform the **16-coordinate `A0` auxiliary channel alone** and the
transformed channel is appended to **unchanged `H_A`**. `H_B` is unchanged. The
predecessor specification said to apply erasure to the concatenated `[H_A, Z]` wire;
that is deliberately not done, because LEACE's `P*` is a single oblique projection
over all coordinates and nothing constrains it to act as the identity on `H_A`, so it
would generally alter the published service output.

Neither baseline's guarantee extends to the augmented release. LEACE guarantees that
no affine predictor under any convex loss beats a constant predictor **on the
transformed channel**; `H_A` still discloses, and combined access may recover more.
That is measured, not assumed away.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from concept_erasure import LeaceEraser

# Numerical policy and library overrides are IMPORTED from the verified, regression-
# tested wrapper rather than retyped, so the three dangerous library defaults
# (shrinkage, trace constraint, absolute svd_tol) stay off by construction.
from experiments.acs_protection_maps import LEACE_OPTIONS, NUMERICAL_POLICY
from experiments.pcrl_nonlinear_rank_v1.inputs import (Registry, array_hash, load_pools,
                                                       load_representation_labels, resolve,
                                                       sha_file, write_json)
from experiments.pcrl_nonlinear_rank_v1.run_fit import limit_threads, machine_state

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/pcrl_invariant_baselines_v1'
FIXED_NAME = 'redesign_20260909_acs_fixed_predictions_v1'
POOLS = ('representation_fit', 'source_validation', 'downstream_fit', 'downstream_validation',
         'attacker_fit', 'attacker_validation', 'test')

# The protected schema for these baselines: SEX, race and coverage, as a CONCATENATED
# joint one-hot (not intersection labels), matching the verified wrapper's convention.
PROTECTED_SCHEMA = {'SEX': 2, 'RAC1P': 9, 'public_coverage': 2}

# SPLINCE's preservation target. The authorised TRAINING tasks only.
AUTHORIZED_TASKS = ('income_binary', 'civilian_at_work')
# Never read. Named explicitly so the exclusion is checkable, not merely intended.
RESERVED_TASKS = ('same_residence', 'commute_over20')

H_A_WIDTH = 4
CHANNEL_WIDTH = 16
SPLINCE_CONDITION_MAX = 1e6      # declared before fitting; see BASELINE_ADAPTATIONS.md 4.3


# ------------------------------------------------------------------ inputs
def a0_channel(seed: int, registry: Registry) -> dict:
    """The frozen 16-coordinate `A0` auxiliary channel, per pool, with `H_A` and `H_B`."""
    path = registry.resolve(f'results/{FIXED_NAME}/seed_{seed}/training/A0/releases.npz')
    out = {'channel': {}, 'hA': {}, 'hB': {}, 'source': str(path), 'source_sha256': sha_file(path)}
    with np.load(path, allow_pickle=False) as z:
        for pool in POOLS:
            wire_a = np.asarray(z[f'wire/A/{pool}'], dtype=np.float64)
            if wire_a.shape[1] != H_A_WIDTH + CHANNEL_WIDTH:
                raise ValueError(f'A0 A-wire width {wire_a.shape[1]} is not 4 + 16')
            out['hA'][pool] = wire_a[:, :H_A_WIDTH]
            out['channel'][pool] = np.ascontiguousarray(wire_a[:, H_A_WIDTH:])
            out['hB'][pool] = np.asarray(z[f'wire/B/{pool}'], dtype=np.float64)
    return out


def joint_onehot(labels: dict, n: int):
    """Concatenated one-hot over the protected schema, on complete cases only."""
    masks, values = [], {}
    for name, size in PROTECTED_SCHEMA.items():
        value = np.asarray(labels[name]).astype(np.int64)
        if value.shape != (n,) or not np.isin(value, np.arange(-1, size)).all():
            raise ValueError(f'{name} requires {n} zero-based schema labels; -1 denotes missing')
        values[name] = value
        masks.append(value >= 0)
    complete = np.stack(masks).all(0)
    z = np.concatenate([np.eye(size, dtype=np.float64)[values[name][complete]]
                        for name, size in PROTECTED_SCHEMA.items()], axis=1)
    coverage = {name: {'schema': list(range(size)),
                       'valid_rows': int((values[name] >= 0).sum()),
                       'support_complete_cases': np.bincount(
                           values[name][complete], minlength=size).tolist()}
                for name, size in PROTECTED_SCHEMA.items()}
    return z, complete, coverage


def authorized_task_matrix(seed: int, registry: Registry, n: int):
    """`[income_binary, civilian_at_work]` on the representation-fitting rows.

    The frozen pipeline withholds task labels from ``representation_fit`` as a blanket
    guard. They are reconstructed here for the **authorised training tasks only**, via
    the same ``all_labels`` helper every other pool uses. The reserved residence and
    commute labels are never read; the assertion below makes that checkable rather
    than merely intended.
    """
    from experiments.run_acs_coalition import all_labels
    from experiments.run_acs_residual_spectral import load_labels

    frame, pools, _labels, _weights = load_labels(Path('/Users/nathansamson/PCRL'), seed)
    rows = pools['representation_fit']
    every = all_labels(frame.iloc[rows], [])
    for reserved in RESERVED_TASKS:
        if reserved in AUTHORIZED_TASKS:
            raise AssertionError('a reserved task appears in the authorised list')
    columns, coverage = [], {}
    for task in AUTHORIZED_TASKS:
        y = np.asarray(every[task]).astype(float)
        if y.shape != (n,):
            raise ValueError(f'{task} has {y.shape} rows, expected {n}')
        valid = y >= 0
        centred = np.where(valid, y, np.nan)
        coverage[task] = {'valid_rows': int(valid.sum()), 'positive_rate':
                          float(np.nanmean(centred)) if valid.any() else None}
        columns.append(np.where(valid, y, np.nanmean(centred)))
    return np.column_stack(columns), coverage


# ------------------------------------------------------------------ LEACE
def _stabilized_leace(x: np.ndarray, z: np.ndarray) -> dict:
    """Rank-stabilised LEACE, generalising the verified two-attribute wrapper.

    Same recipe as ``experiments.acs_protection_maps.fit_joint_leace`` -- eigendecompose
    the covariance, keep eigenvalues above ``1e-10 * lambda_max``, fit affine LEACE in
    the retained coordinates with the three library defaults overridden, lift back, and
    clean the projection's singular values -- generalised only in that the protected
    schema may carry more than two attributes. A regression fixture asserts that
    restricting the schema to ``{SEX, RAC1P}`` reproduces the verified wrapper.
    """
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / (len(x) - 1)
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    cutoff = max(0.0, float(eigenvalues[-1])) * NUMERICAL_POLICY['input_covariance_rtol']
    retained = eigenvalues > cutoff
    basis = vectors[:, retained]
    if retained.any():
        eraser = LeaceEraser.fit(torch.from_numpy(centered @ basis), torch.from_numpy(z),
                                 **LEACE_OPTIONS)
        projection = basis @ eraser.P.detach().cpu().numpy() @ basis.T
    else:
        projection = np.zeros_like(covariance)
    left, singular, right = np.linalg.svd(projection, full_matrices=False)
    projection_cutoff = NUMERICAL_POLICY['projection_svd_rtol'] * max(1.0, float(singular[0]))
    kept = singular > projection_cutoff
    projection = ((left[:, kept] * singular[kept]) @ right[kept]
                  if kept.any() else np.zeros_like(projection))
    return {'projection': projection, 'mean': mean,
            'input_covariance_eigenvalues': eigenvalues.tolist(),
            'input_retained_rank': int(retained.sum()),
            'projection_singular_values': singular.tolist(),
            'projection_retained_rank': int(kept.sum()),
            'exact_constant_output': bool(not kept.any())}


def apply_affine(x: np.ndarray, projection: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """``r(x) = mean + (x - mean) @ P.T`` -- the mean-preserving convention of both papers."""
    return (x - mean) @ projection.T + mean


def cross_covariance_max_abs(x: np.ndarray, z: np.ndarray) -> float:
    xc = x - x.mean(0)
    zc = z - z.mean(0)
    return float(np.max(abs(xc.T @ zc / (len(x) - 1))))


def fit_leace(seed: int, registry: Registry) -> dict:
    channel = a0_channel(seed, registry)
    labels, _ = load_representation_labels(seed, registry)
    x = channel['channel']['representation_fit']
    z, complete, coverage = joint_onehot(labels, len(x))
    fitted = _stabilized_leace(x[complete], z)
    released = apply_affine(x[complete], fitted['projection'], fitted['mean'])
    before = cross_covariance_max_abs(x[complete], z)
    after = cross_covariance_max_abs(released, z)
    metadata = {
        'method': 'LEACE (Belrose et al. 2023, Thm 4.2/4.3), rank-stabilised',
        'applied_to': 'the frozen 16-coordinate A0 auxiliary channel ALONE',
        'appended_to': 'UNCHANGED H_A (4 coordinates); H_B unchanged',
        'protected_schema': PROTECTED_SCHEMA,
        'concept_encoding': 'concatenated one-hot columns, not intersection labels',
        'expected_rank_loss': sum(k - 1 for k in PROTECTED_SCHEMA.values()),
        'realised_projection_rank': fitted['projection_retained_rank'],
        'channel_width': CHANNEL_WIDTH,
        'fit_rows': int(complete.sum()), 'excluded_rows': int((~complete).sum()),
        'coverage': coverage,
        'cross_covariance_max_abs_before': before,
        'cross_covariance_max_abs_after': after,
        'mean_preservation_max_abs_error': float(
            np.abs(released.mean(0) - x[complete].mean(0)).max()),
        'idempotent_max_abs_error': float(np.max(abs(
            fitted['projection'] @ fitted['projection'] - fitted['projection']))),
        'numerical_policy': NUMERICAL_POLICY, 'leace_options': LEACE_OPTIONS,
        'input_retained_rank': fitted['input_retained_rank'],
        'guarantee_scope': (
            'No AFFINE predictor under any nonnegative convex loss beats the best constant '
            'predictor, on the TRANSFORMED CHANNEL and for the moments it was fitted on. This '
            'is a statement about optimal loss, not thresholded accuracy. It says nothing about '
            'the augmented release, which still contains H_A, and nothing about the nonlinear '
            'attacker slate this study scores it against. The paper itself conjectures that '
            'nondestructive editing against a general nonlinear adversary is intractable, and '
            'warns that multiclass softmax outputs can leak the removed information when '
            'another classifier is stacked on top.'),
        'source': channel['source'], 'source_sha256': channel['source_sha256'],
    }
    return {'projection': fitted['projection'], 'mean': fitted['mean'], 'metadata': metadata,
            'channel': channel}


# ------------------------------------------------------------------ SPLINCE
def fit_splince(seed: int, registry: Registry) -> dict:
    """SPLINCE (Holstege et al. 2025, Thm 1) on the same channel.

    The repository's ``pcrl/baselines/splince.py`` implements Thm 1 faithfully but
    **silently falls back to LEACE** when the constraints are jointly infeasible. That
    fallback is disabled here: the protocol requires an infeasible arm to be reported
    as ``SCOPED INFEASIBLE``, never replaced by a LEACE fit wearing the SPLINCE name.
    """
    from pcrl.baselines.splince import fit_splince as splince_fit

    channel = a0_channel(seed, registry)
    labels, _ = load_representation_labels(seed, registry)
    x = channel['channel']['representation_fit']
    z, complete, coverage = joint_onehot(labels, len(x))
    y, task_coverage = authorized_task_matrix(seed, registry, len(x))

    concepts = [np.asarray(labels[name])[complete] for name in PROTECTED_SCHEMA]
    projection, mean, info = splince_fit(x[complete], concepts, y[complete],
                                         cond_max=SPLINCE_CONDITION_MAX)
    record = info.to_dict() if hasattr(info, 'to_dict') else dict(info.__dict__)

    infeasible = bool(record.get('fallback_to_leace')) or not record.get('feasible', True)
    metadata = {
        'method': 'SPLINCE (Holstege, Ravfogel, Wouters 2025, Thm 1) -- ADAPTATION',
        'applied_to': 'the frozen 16-coordinate A0 auxiliary channel ALONE',
        'appended_to': 'UNCHANGED H_A (4 coordinates); H_B unchanged',
        'protected_schema': PROTECTED_SCHEMA,
        'preservation_target': list(AUTHORIZED_TASKS),
        'preservation_target_never_used': list(RESERVED_TASKS),
        'adaptation_note': (
            'The predecessor specification had SPLINCE preserve covariance with the RESIDENCE '
            'label. That is the held-out task, so using it would defeat the held-out-task test. '
            'The preservation target here is the authorised TRAINING tasks income_binary and '
            'civilian_at_work. This is named an ADAPTATION, not a verbatim reproduction.'),
        'label_access_asymmetry': (
            'No other arm in this study uses task labels during representation fitting: the '
            'spectral and repaired arms optimise against the residualised teacher R. SPLINCE '
            'therefore sees two authorised task labels that its comparators do not. This is a '
            'fairness asymmetry in SPLINCE\'s favour on utility and is disclosed, not resolved.'),
        'task_coverage': task_coverage, 'coverage': coverage,
        'fit_rows': int(complete.sum()),
        'condition_gate': SPLINCE_CONDITION_MAX,
        'silent_leace_fallback': 'DISABLED for this study',
        'feasibility_condition': (
            'colsp(W Sigma_xz) intersect colsp(W Sigma_xy) = {0} after whitening. The paper\'s '
            'prose ("do not perfectly overlap") is looser than its own formal condition and is '
            'accurate only in the binary/binary case; the formal version is used. The paper '
            'states NO infeasibility theorem and offers no fallback -- the joint-infeasibility '
            'argument is this study\'s derivation and is labelled as such.'),
        'theorem_2_caveat': (
            'SPLINCE Thm 2: two projections with the same KERNEL but different ranges give '
            'IDENTICAL predictions after re-fitting a strictly convex UNREGULARIZED model with a '
            'unique minimiser. SPLINCE and LEACE share the kernel colsp(Sigma_xz) and so have '
            'the same rank. This study\'s attacker slate is regularised logistic regression, '
            'restarted MLPs, boosted trees and ridge kernel features, none of which is that '
            'case, so a difference CAN appear -- but any observed difference is attributable to '
            'attacker regularisation and nonlinearity, NOT to a stronger erasure guarantee. '
            'Stated before any number is read.'),
        'splince_fit_info': record,
        'status': 'SCOPED INFEASIBLE' if infeasible else 'FITTED',
        'source': channel['source'], 'source_sha256': channel['source_sha256'],
    }
    if not infeasible:
        released = apply_affine(x[complete], projection, mean)
        metadata['cross_covariance_max_abs_before'] = cross_covariance_max_abs(x[complete], z)
        metadata['cross_covariance_max_abs_after'] = cross_covariance_max_abs(released, z)
        yc = y[complete] - y[complete].mean(0)
        before_xy = (x[complete] - x[complete].mean(0)).T @ yc / (len(yc) - 1)
        after_xy = (released - released.mean(0)).T @ yc / (len(yc) - 1)
        metadata['target_covariance_preservation_max_abs_error'] = float(
            np.max(abs(after_xy - before_xy)))
        metadata['target_covariance_frobenius'] = float(np.linalg.norm(before_xy))
        metadata['idempotent_max_abs_error'] = float(
            np.max(abs(projection @ projection - projection)))
        metadata['realised_projection_rank'] = int(np.linalg.matrix_rank(projection, tol=1e-10))
        metadata['mean_preservation_max_abs_error'] = float(
            np.abs(released.mean(0) - x[complete].mean(0)).max())
    return {'projection': None if infeasible else projection,
            'mean': None if infeasible else mean,
            'metadata': metadata, 'channel': channel, 'infeasible': infeasible}


# ------------------------------------------------------------------ releases
def build_releases(name: str, seed: int, fitted: dict, out: Path) -> dict:
    """Append the transformed channel to UNCHANGED `H_A`; `H_B` untouched."""
    channel = fitted['channel']
    path = out / f'seed_{seed}' / 'releases' / name / 'releases.npz'
    if path.exists():
        return {'release_sha256': sha_file(path), 'reused': True, 'parity': []}
    cache, parity = {}, []
    for pool in POOLS:
        ha = channel['hA'][pool]
        hb = channel['hB'][pool]
        transformed = apply_affine(channel['channel'][pool], fitted['projection'], fitted['mean'])
        wa = np.column_stack((ha, transformed))
        wab = np.column_stack((wa, hb))
        assert wa.dtype == np.float64 and np.array_equal(wa[:, :H_A_WIDTH], ha)
        assert np.array_equal(wab[:, -2:], hb) and np.array_equal(wab[:, :wa.shape[1]], wa)
        assert np.isfinite(wa).all() and np.isfinite(wab).all()
        cache[f'wire/A/{pool}'] = wa
        cache[f'wire/B/{pool}'] = hb
        cache[f'wire/AB/{pool}'] = wab
        parity.append({'arm': name, 'pool': pool, 'anchor_A_exact': True, 'anchor_B_exact': True,
                       'A_width': int(wa.shape[1]), 'AB_width': int(wab.shape[1]),
                       'channel_hash': array_hash(transformed)})
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    os.close(fd)
    np.savez_compressed(tmp, **cache)
    os.replace(tmp, path)
    return {'release_sha256': sha_file(path), 'reused': False, 'parity': parity}


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    summary = {}
    for seed in seeds:
        registry = Registry.new()
        tick = time.perf_counter()
        seed_record = {}

        leace = fit_leace(seed, registry)
        release = build_releases('leace_A0', seed, leace, out)
        seed_record['leace_A0'] = {**leace['metadata'], **release, 'status': 'FITTED'}
        print('LEACE', seed, 'rank', leace['metadata']['realised_projection_rank'],
              'xcov', f"{leace['metadata']['cross_covariance_max_abs_after']:.3e}", flush=True)

        splince = fit_splince(seed, registry)
        if splince['infeasible']:
            seed_record['splince_A0'] = splince['metadata']
            print('SPLINCE', seed, 'SCOPED INFEASIBLE', flush=True)
        else:
            release = build_releases('splince_A0', seed, splince, out)
            seed_record['splince_A0'] = {**splince['metadata'], **release}
            print('SPLINCE', seed, 'rank', splince['metadata']['realised_projection_rank'],
                  'cov-preserve err',
                  f"{splince['metadata']['target_covariance_preservation_max_abs_error']:.3e}",
                  flush=True)

        seed_record['runtime_seconds'] = time.perf_counter() - tick
        seed_record['inputs'] = registry.dump()
        write_json(out / f'seed_{seed}' / 'erasure_baselines.json', seed_record)
        summary[str(seed)] = seed_record
    write_json(out / 'ERASURE_BASELINES.json', {'seeds': summary, 'machine': machine_state()})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()
