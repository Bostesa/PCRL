"""Track E fitting: coalition-conditioned partial projections of frozen A0 and J.

Every quantity here is fitted on **representation-training rows only**. No attacker,
no utility probe, no downstream or test outcome, and no residence or commute label is
read. The output of this stage is a set of release wires and serialised affine maps;
nothing is scored.

Per anchor seed:

1. recover frozen `A0` (bit-exact, via the predecessor loader) and frozen `J`
   (reconstructed from its checkpoint and asserted bit-exact against its released wire
   on every pool);
2. cross-fit the five service-only nuisances ONCE, with household folds, shared by both
   channels and every policy;
3. per channel: whiten, build bases, compute every `G_j`, and derive the L / C / LX
   policy maps, the marginal partial eraser, rank-matched PCA and random controls,
   full-span diagnostics, and (for `J` only) ordinary LEACE and SPLINCE.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_direct_adversarial_v1.attackers import (baseline_logits,
                                                              fit_service_baselines,
                                                              service_view)
from experiments.pcrl_invariant_baselines_v1.erasure_baselines import (
    AUTHORIZED_TASKS, PROTECTED_SCHEMA, SPLINCE_CONDITION_MAX, _stabilized_leace, apply_affine,
    cross_covariance_max_abs, joint_onehot)

from . import projection as pj
from .common import OUT, limit_threads, read_json, sha_file, utcnow, write_json_atomic

CHANNELS = ('A0', 'J')
ROLE_ORDER = dax.ROLE_ORDER
LOCAL_ROLES = dax.LOCAL_ROLES
COALITION_ROLES = dax.COALITION_ROLES
ROLE_CLASSES = dax.ROLE_CLASSES
CROSSFIT_FOLDS = 5
CROSSFIT_SALT = 'pcrl_competitive_method_v1/nuisance_crossfit/v1'


# ------------------------------------------------------------------ frozen channels
def load_mapper(seed: int, registry, arm: str) -> dict:
    path = registry.resolve(dax.fixed_path(seed, f'training/{arm}/final.pt'))
    state = torch.load(path, map_location='cpu', weights_only=False)['model_state']
    mapper = nn.Sequential(nn.Linear(dax.PCA_WIDTH, 64), nn.ReLU(), nn.Linear(64, dax.A0_WIDTH))
    mapper.load_state_dict({k[len('branch.mapper.'):]: v.detach().clone()
                            for k, v in state.items() if k.startswith('branch.mapper.')})
    heads = {}
    for name in dax.SOURCE_TASKS:
        head = nn.Linear(dax.A0_WIDTH, 1)
        head.load_state_dict({'weight': state[f'branch.heads.{name}.weight'].detach().clone(),
                              'bias': state[f'branch.heads.{name}.bias'].detach().clone()})
        heads[name] = head
    return {'mapper': mapper.eval(), 'heads': heads,
            'input_mean': state['input_mean'].numpy().copy(),
            'input_scale': state['input_scale'].numpy().copy(),
            'checkpoint': str(path), 'checkpoint_sha256': sha_file(path)}


def frozen_channel(seed: int, registry, state: dict, arm: str) -> dict:
    """Reconstruct `arm`'s channel on every pool and prove it bit-exact to its release."""
    model = load_mapper(seed, registry, arm)
    released = dax.arrays(registry.resolve(dax.fixed_path(seed, f'training/{arm}/releases.npz')))
    channel, parity = {}, {}
    with torch.no_grad():
        for pool in dax.POOLS:
            x = dax.standardize(state['pca'][pool], model['input_mean'], model['input_scale'])
            value = model['mapper'](x).numpy().astype(np.float64)
            stored = np.asarray(released[f'wire/A/{pool}'][:, dax.H_A_WIDTH:], np.float64)
            if stored.shape[1] != dax.A0_WIDTH:
                raise AssertionError(f'{arm} width {stored.shape[1]} != {dax.A0_WIDTH}')
            same = bool(np.array_equal(value, stored))
            parity[pool] = {'bitwise_identical': same,
                            'max_abs_difference': float(np.abs(value - stored).max())}
            if not same:
                raise AssertionError(f'{arm} reconstruction differs from its release at {pool}')
            if not np.array_equal(released[f'wire/A/{pool}'][:, :dax.H_A_WIDTH],
                                  state['anchors'][f'{pool}/A']):
                raise AssertionError(f'{arm} H_A anchor mismatch at {pool}')
            channel[pool] = value
    return {'channel': channel, 'parity': parity, 'model': model}


# ------------------------------------------------------------------ nuisance cross-fitting
def crossfit_folds(serials) -> np.ndarray:
    fold = np.empty(len(serials), np.int64)
    for i, s in enumerate(np.asarray(serials).astype(str)):
        digest = hashlib.sha256(f'{CROSSFIT_SALT}|{s}'.encode()).hexdigest()
        fold[i] = int(digest[:8], 16) % CROSSFIT_FOLDS
    return fold


def crossfit_residuals(seed: int, ha: np.ndarray, hb: np.ndarray, protected: dict,
                       serials) -> dict:
    """Out-of-fold `e_j = onehot(s) - p_j(h_j)` for the five roles, household cross-fitted."""
    ta = torch.tensor(ha, dtype=torch.float32)
    tb = torch.tensor(hb, dtype=torch.float32)
    views = {role: service_view(role, ta, tb) for role in ROLE_ORDER}
    labels = {role: torch.from_numpy(np.asarray(protected[role.split('/')[1]], np.int64))
              for role in ROLE_ORDER}
    fold = crossfit_folds(serials)
    probs = {role: np.full((len(fold), ROLE_CLASSES[role]), np.nan) for role in ROLE_ORDER}
    history = []
    for f in range(CROSSFIT_FOLDS):
        train = np.flatnonzero(fold != f)
        held = np.flatnonzero(fold == f)
        fitted = fit_service_baselines(views, labels, train, 7000 + 10 * seed + f)
        logits = baseline_logits(fitted, {r: views[r][held] for r in ROLE_ORDER})
        for role in ROLE_ORDER:
            probs[role][held] = torch.softmax(logits[role].double(), 1).numpy()
        history.append({'fold': f, 'train_rows': int(len(train)), 'held_rows': int(len(held)),
                        'fit_history': {r: fitted[r]['history'] for r in ROLE_ORDER}})
    out = {}
    for role in ROLE_ORDER:
        y = np.asarray(protected[role.split('/')[1]], np.int64)
        p = probs[role]
        if not np.isfinite(p).all():
            raise AssertionError(f'cross-fit left rows unpredicted for {role}')
        valid = y >= 0
        onehot = np.zeros_like(p)
        onehot[np.flatnonzero(valid), y[valid]] = 1.0
        e = onehot - p
        e[~valid] = 0.0
        out[role] = {'residual': e, 'valid': valid,
                     'class_support': np.bincount(y[valid], minlength=p.shape[1]).tolist(),
                     'oof_cross_entropy': float(-np.log(np.clip(p[valid, y[valid]], 1e-12, 1))
                                                .mean())}
    return {'roles': out, 'fold_sizes': np.bincount(fold, minlength=CROSSFIT_FOLDS).tolist(),
            'history': history, 'folds': fold}


# ------------------------------------------------------------------ per-channel maps
def service_bases(ha: np.ndarray, hb: np.ndarray) -> dict:
    hab = np.column_stack([ha, hb])
    kept_a = pj.nonredundant_columns(ha)
    kept_ab = pj.nonredundant_columns(hab)
    b_a, scale_a = pj.polynomial_basis(ha, kept_a)
    b_ab, scale_ab = pj.polynomial_basis(hab, kept_ab)
    rff = pj.rff_parameters(ha, kept_a)
    f_a, scale_rff = pj.rff_features(ha, kept_a, rff['weights'], rff['offsets'])
    return {'kept_A': kept_a, 'kept_AB': kept_ab, 'b_A': b_a, 'b_AB': b_ab,
            'b_LX': np.column_stack([b_a, f_a]),
            'scale_A': scale_a, 'scale_AB': scale_ab, 'scale_rff': scale_rff,
            'rff': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rff.items()},
            'dims': {'A': int(b_a.shape[1]), 'AB': int(b_ab.shape[1]),
                     'LX': int(b_a.shape[1] + f_a.shape[1])}}


def policy_kernels(v: np.ndarray, bases: dict, nuisance: dict, protected: dict) -> dict:
    roles = nuisance['roles']
    G, energy = {}, {}
    for role in ROLE_ORDER:
        basis = bases['b_A'] if role in LOCAL_ROLES else bases['b_AB']
        G[role] = pj.cross_moment(v, basis, roles[role]['residual'], roles[role]['valid'])
        energy[role] = float((G[role] ** 2).sum())
    for role in ('A/SEX', 'A/RAC1P'):
        key = f'LX:{role}'
        G[key] = pj.cross_moment(v, bases['b_LX'], roles[role]['residual'], roles[role]['valid'])
        energy[key] = float((G[key] ** 2).sum())
    # marginal: b = 1, globally centred one-hot target (no nuisance), three targets
    ones = np.ones((len(v), 1))
    for name in PROTECTED_SCHEMA:
        y = np.asarray(protected[name], np.int64)
        valid = y >= 0
        onehot = np.zeros((len(y), PROTECTED_SCHEMA[name]))
        onehot[np.flatnonzero(valid), y[valid]] = 1.0
        centred = onehot - onehot[valid].mean(0)
        centred[~valid] = 0.0
        key = f'marginal:{name}'
        G[key] = pj.cross_moment(v, ones, centred, valid)
        energy[key] = float((G[key] ** 2).sum())
    eps = pj.EPS_REL * max(energy[r] for r in ROLE_ORDER)
    K = {key: pj.role_kernel(g, eps)[0] for key, g in G.items()}
    k_local = np.mean([K[r] for r in LOCAL_ROLES], 0)
    k_coal = np.mean([K[r] for r in COALITION_ROLES], 0)
    k_lx = np.mean([K['A/public_coverage'], K['LX:A/SEX'], K['LX:A/RAC1P']], 0)
    k_marg = np.mean([K[f'marginal:{n}'] for n in PROTECTED_SCHEMA], 0)
    near_zero = {key: e for key, e in energy.items() if e <= eps}
    return {'policy': {'L': k_local, 'C': k_local + k_coal, 'LX': k_lx, 'marginal': k_marg},
            'coalition_only': k_coal, 'energy': energy, 'eps': eps, 'near_zero': near_zero,
            'G_shapes': {k: list(g.shape) for k, g in G.items()}}


def spectrum(kernel: np.ndarray) -> list:
    values = np.linalg.eigvalsh(0.5 * (kernel + kernel.T))[::-1]
    return [float(x) for x in values]


def full_span_rank(kernel: np.ndarray, rel: float = 1e-10) -> int:
    values = np.array(spectrum(kernel))
    return int((values > rel * max(values[0], 1e-300)).sum())


def write_release(out: Path, seed: int, name: str, channel: dict, anchors: dict,
                  map_record: dict, maps: dict) -> dict:
    path = out / f'seed_{seed}' / 'releases' / name / 'releases.npz'
    cache = dax.build_wires(channel, anchors)
    for key, value in cache.items():
        if not np.isfinite(value).all() or value.dtype != np.float64:
            raise AssertionError(f'invalid release {name} {key}')
    if path.exists():
        existing = dax.arrays(path)
        if not all(np.array_equal(existing[k], cache[k]) for k in cache):
            raise AssertionError(f'{name} release exists with different content')
        release = {'path': str(path), 'sha256': sha_file(path), 'reused': True}
    else:
        release = {**dax.save_release(path, cache), 'reused': False}
    release['channel_hash'] = {p: dax.array_hash(channel[p]) for p in dax.POOLS}
    maps[name] = map_record
    return release


def projected_channel(whitening: pj.Whitening, source: dict, P: np.ndarray) -> dict:
    return {pool: pj.release(whitening, source[pool], P) for pool in dax.POOLS}


def realised_rank(values: np.ndarray) -> int:
    centred = values - values.mean(0)
    top = np.linalg.norm(centred, 2)
    return int(np.linalg.matrix_rank(centred, tol=1e-9 * top)) if top > 0 else 0


def fit_channel(out: Path, seed: int, name: str, source: dict, anchors: dict, bases: dict,
                nuisance: dict, protected: dict, maps: dict, ledger: list):
    fit_rows = source['representation_fit']
    whitening = pj.fit_whitening(fit_rows)
    v = whitening.forward(fit_rows)
    kernels = policy_kernels(v, bases, nuisance, protected)
    r = whitening.rank
    sensitivity = {f'{t:g}': int(pj.fit_whitening(fit_rows, t).rank)
                   for t in (1e-14, 1e-12, 1e-10, 1e-8, 1e-6)}
    context = {'channel': name, 'whitening': whitening.report(),
               'whitening_rank_by_tolerance': sensitivity,
               'energy': kernels['energy'], 'eps': kernels['eps'],
               'near_zero_blocks': kernels['near_zero'], 'G_shapes': kernels['G_shapes'],
               'spectra': {p: spectrum(k) for p, k in kernels['policy'].items()},
               'coalition_only_spectrum': spectrum(kernels['coalition_only'])}

    def emit(unit, P, family, policy, k, extra=None):
        channel = projected_channel(whitening, source, P)
        record = {'kind': 'whitened_projection', 'mu': whitening.mu, 'R': whitening.R,
                  'P': P, 'S': whitening.S}
        rel = write_release(out, seed, unit, channel, anchors, record, maps)
        fitted = channel['representation_fit']
        entry = {'seed': seed, 'unit': unit, 'channel': name, 'family': family,
                 'policy': policy, 'k': k, 'supported_rank': r,
                 'requested_retained_rank': None if k is None else r - k,
                 'realised_rank_representation_fit': realised_rank(fitted),
                 'distortion_normalised': float(((fitted - fit_rows) ** 2).mean()
                                                / fit_rows.var(0).mean()),
                 'idempotence_max_abs': float(np.abs(P @ P - P).max()),
                 'mean_preservation_max_abs': float(np.abs(fitted.mean(0) - whitening.mu).max()),
                 'release': rel, **(extra or {})}
        # residual declared moments after projection, per role
        v_out = v @ P
        entry['moment_energy_after'] = {
            role: float((pj.cross_moment(v_out, bases['b_A'] if role in LOCAL_ROLES
                                         else bases['b_AB'],
                                         nuisance['roles'][role]['residual'],
                                         nuisance['roles'][role]['valid']) ** 2).sum())
            for role in ROLE_ORDER}
        ledger.append(entry)
        print('E_MAP', seed, unit, 'rank', entry['realised_rank_representation_fit'], flush=True)

    for policy in ('L', 'C', 'LX', 'marginal'):
        kernel = kernels['policy'][policy]
        family = 'marginal_partial_erasure' if policy == 'marginal' else 'coalition_projection'
        for k in pj.REMOVAL_RANKS:
            lead = pj.leading_directions(kernel, k)
            emit(f'E_{name}_{policy}_k{k}', pj.projector(lead['U'], r), family, policy, k,
                 {'boundary_gap': lead['boundary_gap'],
                  'boundary_degenerate': lead['boundary_degenerate']})
        if policy != 'marginal':
            full = full_span_rank(kernel)
            lead = pj.leading_directions(kernel, full)
            emit(f'E_{name}_{policy}_full', pj.projector(lead['U'], r), 'full_span_diagnostic',
                 policy, full, {'full_span_rank': full})
    for k in pj.REMOVAL_RANKS:
        # PCA retaining r-k directions, ORIGINAL metric
        pca = pj.pca_retaining(fit_rows, r - k)
        channel = {pool: pca['mu'] + (source[pool] - pca['mu']) @ pca['projector']
                   for pool in dax.POOLS}
        unit = f'E_{name}_pca_k{k}'
        rel = write_release(out, seed, unit, channel, anchors,
                            {'kind': 'original_metric_projection', 'mu': pca['mu'],
                             'Q': pca['projector']}, maps)
        fitted = channel['representation_fit']
        ledger.append({'seed': seed, 'unit': unit, 'channel': name, 'family': 'pca',
                       'policy': 'pca', 'k': k, 'supported_rank': r,
                       'requested_retained_rank': r - k,
                       'realised_rank_representation_fit': realised_rank(fitted),
                       'explained_variance_ratio': pca['explained_variance_ratio'],
                       'distortion_normalised': float(((fitted - fit_rows) ** 2).mean()
                                                      / fit_rows.var(0).mean()),
                       'release': rel})
        print('E_MAP', seed, unit, flush=True)
        # random subspace, whitened metric, paired draw shared by A0 and J
        draw = pj.random_subspace(r, r - k, 20269500 + 1000 * seed + k)
        emit(f'E_{name}_rand_k{k}', draw['projector'], 'random', 'random', k,
             {'random_seed': draw['seed']})
    return context


# ------------------------------------------------------------------ ordinary erasure on J
def fit_erasure_on_j(out: Path, seed: int, source: dict, anchors: dict, protected: dict,
                     maps: dict, ledger: list):
    from experiments.pcrl_direct_adversarial_v1.erasure import authorised_task_matrix
    from pcrl.baselines.splince import fit_splince as splince_fit
    x = source['representation_fit']
    z, complete, coverage = joint_onehot(protected, len(x))
    fitted = _stabilized_leace(x[complete], z)
    results = {}

    def emit(unit, projection, mean, meta):
        channel = {pool: apply_affine(source[pool], projection, mean) for pool in dax.POOLS}
        rel = write_release(out, seed, unit, channel, anchors,
                            {'kind': 'affine_erasure', 'projection': projection, 'mean': mean},
                            maps)
        released = channel['representation_fit'][complete]
        ledger.append({'seed': seed, 'unit': unit, 'channel': 'J', 'family': unit.split('_')[0],
                       'policy': unit.split('_')[0], 'k': None,
                       'realised_rank_representation_fit': realised_rank(
                           channel['representation_fit']),
                       'cross_covariance_max_abs_before': cross_covariance_max_abs(x[complete], z),
                       'cross_covariance_max_abs_after': cross_covariance_max_abs(released, z),
                       'distortion_normalised': float(((channel['representation_fit'] - x) ** 2)
                                                      .mean() / x.var(0).mean()),
                       'release': rel, **meta})
        print('E_MAP', seed, unit, flush=True)

    emit('leace_J', fitted['projection'], fitted['mean'],
         {'realised_projection_rank': fitted['projection_retained_rank'], 'status': 'FITTED'})
    y, _task = authorised_task_matrix(seed, len(x))
    concepts = [np.asarray(protected[name])[complete] for name in PROTECTED_SCHEMA]
    projection, mean, info = splince_fit(x[complete], concepts, y[complete],
                                         cond_max=SPLINCE_CONDITION_MAX)
    record = info.to_dict() if hasattr(info, 'to_dict') else dict(info.__dict__)
    infeasible = bool(record.get('fallback_to_leace')) or not record.get('feasible', True)
    if infeasible:
        ledger.append({'seed': seed, 'unit': 'splince_J', 'channel': 'J', 'family': 'splince',
                       'status': 'SCOPED INFEASIBLE', 'splince_fit_info': str(record)})
        print('E_MAP', seed, 'splince_J SCOPED INFEASIBLE', flush=True)
    else:
        emit('splince_J', projection, mean, {'status': 'FITTED',
                                              'splince_fit_info': str(record)})


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    summary = {}
    for seed in seeds:
        marker = out / f'seed_{seed}' / 'track_e_complete.json'
        if marker.exists():
            summary[str(seed)] = read_json(marker)
            print('E_REUSED', seed, flush=True)
            continue
        tick = time.perf_counter()
        registry = dax.Registry.new()
        state = dax.load_frozen_state(seed, registry)
        labels = dax.representation_labels(seed, registry)
        protected = labels['protected']
        channels = {'A0': state['channel']}
        j = frozen_channel(seed, registry, state, 'J')
        channels['J'] = j['channel']
        ha = np.asarray(state['anchors']['representation_fit/A'], np.float64)
        hb = np.asarray(state['anchors']['representation_fit/B'], np.float64)
        nuisance = crossfit_residuals(seed, ha, hb, protected, labels['serials'])
        bases = service_bases(ha, hb)
        maps, ledger, contexts = {}, [], {}
        for name in CHANNELS:
            contexts[name] = fit_channel(out, seed, name, channels[name], state['anchors'], bases,
                                         nuisance, protected, maps, ledger)
        fit_erasure_on_j(out, seed, channels['J'], state['anchors'], protected, maps, ledger)
        joblib.dump({'maps': maps, 'bases': {k: v for k, v in bases.items()
                                             if k not in ('b_A', 'b_AB', 'b_LX')}},
                    out / f'seed_{seed}' / 'track_e_maps.joblib')
        record = {'seed': seed, 'generated_utc': utcnow(),
                  'J_parity': j['parity'], 'J_checkpoint_sha256': j['model']['checkpoint_sha256'],
                  'A0_parity': state['a0_parity'],
                  'nuisance': {'fold_sizes': nuisance['fold_sizes'],
                               'roles': {r: {k: v for k, v in d.items()
                                             if k not in ('residual', 'valid')}
                                         for r, d in nuisance['roles'].items()}},
                  'basis': {k: v for k, v in bases.items()
                            if k not in ('b_A', 'b_AB', 'b_LX', 'rff')} | {
                      'rff_bandwidth': bases['rff']['bandwidth']},
                  'contexts': contexts, 'units': ledger,
                  'maps_sha256': sha_file(out / f'seed_{seed}' / 'track_e_maps.joblib'),
                  'runtime_seconds': time.perf_counter() - tick,
                  'outcomes_read': False}
        write_json_atomic(out / f'seed_{seed}' / 'track_e_fit.json', record)
        write_json_atomic(marker, {'seed': seed, 'units': len(ledger),
                                   'runtime_seconds': record['runtime_seconds'],
                                   'fit_sha256': sha_file(out / f'seed_{seed}' / 'track_e_fit.json')})
        summary[str(seed)] = read_json(marker)
        print('E_SEED_DONE', seed, len(ledger), round(record['runtime_seconds'], 1), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out, tuple(args.seeds))


if __name__ == '__main__':
    main()
