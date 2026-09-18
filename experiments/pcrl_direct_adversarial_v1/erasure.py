"""New erasure controls: LEACE and SPLINCE on each width's no-protection channel.

Refit **only** because the input channel is new. The method, the numerical policy and
the three dangerous library overrides are imported from the verified, regression-tested
wrapper of `73903b7f` rather than retyped. The **historical** `leace_A0` / `splince_A0`
are kept and are never replaced by these.

Erasure operates on `Z` alone; `H_A` and `H_B` are appended unchanged, for the reason
recorded in `BASELINE_ADAPTATIONS.md` §4: LEACE's `P*` is a single oblique projection
over all coordinates, and nothing constrains it to act as the identity on `H_A`.

**Width matching is not claimed when the effective rank drops.** Ambient width, realised
projection rank and a compression sensitivity are recorded for every arm.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.pcrl_invariant_baselines_v1.erasure_baselines import (
    AUTHORIZED_TASKS, PROTECTED_SCHEMA, RESERVED_TASKS, SPLINCE_CONDITION_MAX, _stabilized_leace,
    apply_affine, cross_covariance_max_abs, joint_onehot)

from .inputs import (H_A_WIDTH, H_B_WIDTH, OUT, POOLS, Registry, array_hash, arrays,
                     load_frozen_state, read_json, representation_labels, sha_file, write_json)
from .run_fit import WIDTHS, limit_threads, machine_state


def source_channel(out: Path, seed: int, width: int) -> dict:
    """The fitted no-protection channel of this width, read back from its release."""
    path = Path(out) / f'seed_{seed}' / 'releases' / f'dax{width}_none' / 'releases.npz'
    data = arrays(path)
    channel, ha, hb = {}, {}, {}
    for pool in POOLS:
        wire = np.asarray(data[f'wire/A/{pool}'], np.float64)
        if wire.shape[1] != H_A_WIDTH + width:
            raise AssertionError(f'{path} A-wire width {wire.shape[1]} is not {H_A_WIDTH}+{width}')
        ha[pool] = wire[:, :H_A_WIDTH]
        channel[pool] = np.ascontiguousarray(wire[:, H_A_WIDTH:])
        hb[pool] = np.asarray(data[f'wire/B/{pool}'], np.float64)
    return {'channel': channel, 'hA': ha, 'hB': hb, 'source': str(path),
            'source_sha256': sha_file(path), 'width': width}


def authorised_task_matrix(seed: int, n: int):
    from .inputs import authorised_source_labels
    labels = authorised_source_labels(seed, n)
    columns, coverage = [], {}
    for task in AUTHORIZED_TASKS:
        y = np.asarray(labels[task]).astype(float)
        valid = y >= 0
        mean = float(y[valid].mean()) if valid.any() else 0.0
        coverage[task] = {'valid_rows': int(valid.sum()), 'positive_rate': mean}
        columns.append(np.where(valid, y, mean))
    return np.column_stack(columns), coverage


def compression_sensitivity(released: np.ndarray, ambient: int) -> dict:
    """Ambient width is not effective width. Report the realised spectrum."""
    centred = released - released.mean(0)
    singular = np.linalg.svd(centred, compute_uv=False)
    total = float((singular ** 2).sum())
    energy = (singular ** 2) / total if total > 0 else singular * 0
    cumulative = np.cumsum(energy)
    return {'ambient_width': int(ambient),
            'numerical_rank_1e-10': int(np.linalg.matrix_rank(centred, tol=1e-10)),
            'singular_values': singular.tolist(),
            'components_for_99pct_variance': int(np.searchsorted(cumulative, 0.99) + 1),
            'components_for_999pct_variance': int(np.searchsorted(cumulative, 0.999) + 1),
            'note': ('effective rank, not ambient width, is what an attacker sees; a matched '
                     'ambient width with a lower effective rank is NOT a width match')}


def fit_leace(out: Path, seed: int, width: int, registry: Registry) -> dict:
    source = source_channel(out, seed, width)
    labels = representation_labels(seed, registry)['protected']
    x = source['channel']['representation_fit']
    z, complete, coverage = joint_onehot(labels, len(x))
    fitted = _stabilized_leace(x[complete], z)
    released = apply_affine(x[complete], fitted['projection'], fitted['mean'])
    metadata = {
        'method': 'LEACE (Belrose et al. 2023, Thm 4.2/4.3), rank-stabilised',
        'applied_to': f'the fitted no-protection channel dax{width}_none ALONE',
        'appended_to': f'UNCHANGED H_A ({H_A_WIDTH} coordinates); H_B unchanged',
        'protected_schema': PROTECTED_SCHEMA,
        'channel_width': width,
        'expected_rank_loss': sum(k - 1 for k in PROTECTED_SCHEMA.values()),
        'realised_projection_rank': fitted['projection_retained_rank'],
        'input_retained_rank': fitted['input_retained_rank'],
        'fit_rows': int(complete.sum()), 'excluded_rows': int((~complete).sum()),
        'coverage': coverage,
        'cross_covariance_max_abs_before': cross_covariance_max_abs(x[complete], z),
        'cross_covariance_max_abs_after': cross_covariance_max_abs(released, z),
        'mean_preservation_max_abs_error': float(
            np.abs(released.mean(0) - x[complete].mean(0)).max()),
        'idempotent_max_abs_error': float(np.max(abs(
            fitted['projection'] @ fitted['projection'] - fitted['projection']))),
        'compression_sensitivity': compression_sensitivity(released, width),
        'guarantee_scope': (
            'No AFFINE predictor under any nonnegative convex loss beats the best constant '
            'predictor, on the TRANSFORMED CHANNEL and for the moments it was fitted on. It '
            'says nothing about the augmented release, which still contains H_A, and nothing '
            'about the nonlinear attacker slate this study scores it against.'),
        'source': source['source'], 'source_sha256': source['source_sha256']}
    return {'projection': fitted['projection'], 'mean': fitted['mean'], 'metadata': metadata,
            'channel': source, 'infeasible': False}


def fit_splince(out: Path, seed: int, width: int, registry: Registry) -> dict:
    from pcrl.baselines.splince import fit_splince as splince_fit

    source = source_channel(out, seed, width)
    labels = representation_labels(seed, registry)['protected']
    x = source['channel']['representation_fit']
    z, complete, coverage = joint_onehot(labels, len(x))
    y, task_coverage = authorised_task_matrix(seed, len(x))
    concepts = [np.asarray(labels[name])[complete] for name in PROTECTED_SCHEMA]
    projection, mean, info = splince_fit(x[complete], concepts, y[complete],
                                         cond_max=SPLINCE_CONDITION_MAX)
    record = info.to_dict() if hasattr(info, 'to_dict') else dict(info.__dict__)
    infeasible = bool(record.get('fallback_to_leace')) or not record.get('feasible', True)
    metadata = {
        'method': 'SPLINCE (Holstege, Ravfogel, Wouters 2025, Thm 1) -- ADAPTATION',
        'applied_to': f'the fitted no-protection channel dax{width}_none ALONE',
        'appended_to': f'UNCHANGED H_A ({H_A_WIDTH} coordinates); H_B unchanged',
        'protected_schema': PROTECTED_SCHEMA, 'channel_width': width,
        'preservation_target': list(AUTHORIZED_TASKS),
        'preservation_target_never_used': list(RESERVED_TASKS),
        'label_access_asymmetry': (
            'SPLINCE sees two authorised task labels during representation fitting that the '
            'new neural arms also see (through their Z-only source heads) but that the '
            'spectral family did not. Disclosed, not resolved.'),
        'task_coverage': task_coverage, 'coverage': coverage,
        'fit_rows': int(complete.sum()), 'condition_gate': SPLINCE_CONDITION_MAX,
        'silent_leace_fallback': 'DISABLED for this study',
        'theorem_2_caveat': (
            'SPLINCE and LEACE share the kernel colsp(Sigma_xz) and so have the same rank. Any '
            'observed difference here is attributable to attacker regularisation and '
            'nonlinearity, NOT to a stronger erasure guarantee.'),
        'splince_fit_info': record,
        'status': 'SCOPED INFEASIBLE' if infeasible else 'FITTED',
        'source': source['source'], 'source_sha256': source['source_sha256']}
    if not infeasible:
        released = apply_affine(x[complete], projection, mean)
        yc = y[complete] - y[complete].mean(0)
        before_xy = (x[complete] - x[complete].mean(0)).T @ yc / (len(yc) - 1)
        after_xy = (released - released.mean(0)).T @ yc / (len(yc) - 1)
        metadata.update({
            'cross_covariance_max_abs_before': cross_covariance_max_abs(x[complete], z),
            'cross_covariance_max_abs_after': cross_covariance_max_abs(released, z),
            'target_covariance_preservation_max_abs_error': float(np.max(abs(after_xy - before_xy))),
            'realised_projection_rank': int(np.linalg.matrix_rank(projection, tol=1e-10)),
            'idempotent_max_abs_error': float(np.max(abs(projection @ projection - projection))),
            'mean_preservation_max_abs_error': float(
                np.abs(released.mean(0) - x[complete].mean(0)).max()),
            'compression_sensitivity': compression_sensitivity(released, width)})
    return {'projection': None if infeasible else projection,
            'mean': None if infeasible else mean, 'metadata': metadata,
            'channel': source, 'infeasible': infeasible}


def build_release(out: Path, name: str, seed: int, fitted: dict) -> dict:
    source = fitted['channel']
    path = Path(out) / f'seed_{seed}' / 'releases' / name / 'releases.npz'
    if path.exists():
        return {'sha256': sha_file(path), 'reused': True}
    cache, parity = {}, []
    for pool in POOLS:
        ha, hb = source['hA'][pool], source['hB'][pool]
        transformed = apply_affine(source['channel'][pool], fitted['projection'], fitted['mean'])
        wa = np.column_stack((ha, transformed))
        wab = np.column_stack((wa, hb))
        if not (np.array_equal(wa[:, :H_A_WIDTH], ha) and np.array_equal(wab[:, -H_B_WIDTH:], hb)):
            raise AssertionError(f'anchor parity failed for {name} at {pool}')
        if not (np.isfinite(wa).all() and np.isfinite(wab).all()):
            raise AssertionError(f'non-finite erasure release for {name} at {pool}')
        cache[f'wire/A/{pool}'] = wa
        cache[f'wire/B/{pool}'] = hb.copy()
        cache[f'wire/AB/{pool}'] = wab
        parity.append({'pool': pool, 'anchor_A_exact': True, 'anchor_B_exact': True,
                       'channel_hash': array_hash(transformed)})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)
    return {'sha256': sha_file(path), 'reused': False, 'parity': parity,
            'A_width': int(cache['wire/A/test'].shape[1])}


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    summary = {}
    for seed in seeds:
        registry = Registry.new()
        record = {}
        for width in WIDTHS:
            tick = time.perf_counter()
            leace = fit_leace(out, seed, width, registry)
            name = f'leace_dax{width}_none'
            record[name] = {**leace['metadata'], 'release': build_release(out, name, seed, leace),
                            'runtime_seconds': time.perf_counter() - tick}
            print('LEACE', seed, width, 'rank',
                  leace['metadata']['realised_projection_rank'],
                  'xcov', f"{leace['metadata']['cross_covariance_max_abs_after']:.3e}", flush=True)

            tick = time.perf_counter()
            splince = fit_splince(out, seed, width, registry)
            name = f'splince_dax{width}_none'
            entry = {**splince['metadata'], 'runtime_seconds': time.perf_counter() - tick}
            if not splince['infeasible']:
                entry['release'] = build_release(out, name, seed, splince)
            record[name] = entry
            print('SPLINCE', seed, width, entry['status'], flush=True)
        write_json(out / f'seed_{seed}' / 'new_erasure.json',
                   {**record, 'inputs': registry.dump()})
        summary[str(seed)] = record
    write_json(out / 'NEW_ERASURE.json', {'seeds': summary, 'machine': machine_state(),
                                          'note': ('Refit only because the input channel is new. '
                                                   'The historical leace_A0 and splince_A0 are '
                                                   'kept and are never replaced.')})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()
