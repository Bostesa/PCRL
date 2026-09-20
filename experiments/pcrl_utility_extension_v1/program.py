"""Matrix, fit/audit jobs, deduplication, pilot gate and Tier-3 selection.

Every unit is resumable: a completed marker is reused, never refitted to fill a count.
Fitted weights (extension, decoder, checkpoints) and fitted audit attackers are KEPT; this
study never compacts away attacker weights (the predecessor's compaction did, and a later
transport needed them).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_direct_adversarial_v1 import report as rep

from . import extension as ext

STUDY = 'pcrl_utility_extension_v1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('PCRL_UX_OUT', ROOT / 'results' / STUDY))
SEEDS = (0, 1, 2)
POLICIES = ('L1', 'L2', 'C1')
BETAS = (1, 3, 10, 30)
PILOT_R = (2,)
FULL_R = (2, 4, 8)
WEIGHTS = ('unweighted', 'person_weighted')
SENSITIVE4 = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')
SOURCE = ('income_binary', 'civilian_at_work', 'public_coverage')
SOURCE_ALLOWANCE = 0.01
GATE_RECON_REDUCTION = 0.10
GATE_MEAN_INC = 0.001
GATE_ANCHOR_INC = 0.003
MAIN_BUDGET, MAIN_SCOPE = 360, 'kernel_expanded_independent'
MAX_RETRIES = 3


def unit_spec(name: str) -> dict:
    kind, r, *rest = name.split('_')
    r = int(r[1:])
    if kind == 'X':
        return {'kind': 'protected', 'r': r, 'policy': rest[0], 'beta': float(rest[1][1:])}
    if kind == 'U':
        return {'kind': 'unprotected', 'r': r, 'policy': 'C1', 'beta': 0.0}
    if kind == 'P':
        return {'kind': 'pca', 'r': r}
    if kind == 'LEACE':
        return {'kind': 'leace_on_U', 'r': r, 'source': f'U_r{r}'}
    raise ValueError(name)


def units(rs, leace=False) -> list:
    out = [f'X_r{r}_{p}_b{b:03d}' for r in rs for p in POLICIES for b in BETAS]
    out += [f'U_r{r}' for r in rs] + [f'P_r{r}' for r in rs]
    if leace:
        out += [f'LEACE_r{r}' for r in rs]
    return out


def config_id(name):
    return name                                  # configurations are common across anchors


# ------------------------------------------------------------------ IO
def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp{os.getpid()}')
    with open(tmp, 'w') as fh:
        json.dump(payload, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def release_identity(path: Path) -> str:
    data = np.load(path)
    d = hashlib.sha256()
    for key in sorted(data.files):
        a = np.ascontiguousarray(data[key])
        d.update(key.encode()); d.update(str(a.dtype).encode()); d.update(str(a.shape).encode())
        d.update(a.tobytes())
    return d.hexdigest()


# ------------------------------------------------------------------ fit jobs
_CTX = {}


def context(seed, config):
    if seed not in _CTX:
        _CTX[seed] = ext.seed_context(seed, config)
    return _CTX[seed]


def references(seed, config):
    """ref_J / ref_A0 wires rebuilt from the STORED release arrays; asserted equal to them."""
    ctx = context(seed, config)
    for name, channel in (('ref_J', ctx['zj']), ('ref_A0', ctx['za0'])):
        path = OUT / f'seed_{seed}' / 'releases' / name / 'releases.npz'
        if path.exists():
            continue
        wires = dax.build_wires(channel, ctx['state']['anchors'])
        arm = 'J' if name == 'ref_J' else 'A0'
        stored = dax.arrays(dax.Registry.new().resolve(dax.fixed_path(seed, f'training/{arm}/releases.npz')))
        for k, v in wires.items():
            if k in stored and not np.array_equal(v, np.asarray(stored[k], np.float64)):
                raise AssertionError(f'{name} identity failed at {k}')
        dax.save_release(path, wires)
    path = OUT / f'seed_{seed}' / 'releases' / 'ref_leace_A0' / 'releases.npz'
    if not path.exists():
        src = dax.resolve(f'results/pcrl_invariant_baselines_v1/seed_{seed}/releases/leace_A0/releases.npz')
        stored = dax.arrays(src)
        wires = {k: np.asarray(stored[k], np.float64) for k in stored if k.startswith('wire/')}
        for p in dax.POOLS:
            if not (np.array_equal(wires[f'wire/A/{p}'][:, :ext.HA], ctx['state']['anchors'][f'{p}/A'])
                    and np.array_equal(wires[f'wire/B/{p}'], ctx['state']['anchors'][f'{p}/B'])):
                raise AssertionError(f'leace_A0 anchor parity failed at {p}')
        dax.save_release(path, wires)


def fit_job(seed: int, name: str, config: ext.Config) -> dict:
    dest = OUT / f'seed_{seed}' / 'fits' / name
    marker = dest / 'complete.json'
    if marker.exists():
        return read_json(marker)
    spec = unit_spec(name)
    ctx = context(seed, config)
    tick = time.perf_counter()
    record = {'seed': seed, 'unit': name, 'spec': spec, 'config': config.as_dict(),
              'd0': ctx['d0_record'], 'portability': ctx['portability']}
    if ctx['degenerate_target']:
        record.update(status='DEGENERATE_RESIDUAL_TARGET')
        write_json(marker, record)
        return record
    dest.mkdir(parents=True, exist_ok=True)
    if spec['kind'] in ('protected', 'unprotected'):
        fit = ext.fit_unit(ctx, spec['r'], spec['policy'], spec['beta'], config)
        torch.save({'selected_state': fit['model'].state_dict(),
                    'checkpoint_states': {c['step']: c['state'] for c in fit['checkpoints']},
                    'selected_step': fit['selected_step']}, dest / 'model.pt')
        values = ext.learned_extension_values(ctx, fit['model'])
        record.update(selected_step=fit['selected_step'], monitor_scores=fit['monitor_scores'],
                      trace=fit['trace'], refreshes=fit['refreshes'], fit_seconds=fit['runtime_seconds'])
    elif spec['kind'] == 'pca':
        values, meta = ext.pca_extension(ctx, spec['r'])
        np.savez(dest / 'pca_map.npz', mu=meta['mu'], components=meta['components'])
        record.update(pca_singular_values=meta['singular_values'],
                      deployed_function='R = (Z_A0(x) - D0_final(H_A, Z_J(x)) - mu) @ V_r; mu, V_r fitted on mapper_fit')
    elif spec['kind'] == 'leace_on_U':
        src = OUT / f'seed_{seed}' / 'releases' / spec['source'] / 'releases.npz'
        if not src.exists():
            raise FileNotFoundError(f'LEACE source {src} missing')
        values, meta = leace_on_extension(ctx, dax.arrays(src), spec['r'])
        np.savez(dest / 'leace_map.npz', projection=meta.pop('projection'), mean=meta.pop('mean'))
        record.update(leace=meta)
    wires = ext.build_release(ctx, values)
    ext.assert_parity(ctx, wires)
    record['validation_reconstruction'] = ext.validation_reconstruction(ctx, values, config)
    r_rf = np.asarray(values['representation_fit'])
    record['extension_std'] = r_rf.std(0).tolist()
    record['extension_constant'] = bool(np.all(r_rf.std(0) == 0))
    rpath = OUT / f'seed_{seed}' / 'releases' / name / 'releases.npz'
    record['release'] = dax.save_release(rpath, wires)
    record['release_identity'] = release_identity(rpath)
    record['seconds'] = time.perf_counter() - tick
    record['status'] = 'FITTED'
    write_json(dest / 'fit_record.json', record)
    write_json(marker, {k: v for k, v in record.items() if k not in ('trace', 'monitor_scores', 'refreshes')})
    return record


def leace_on_extension(ctx, source_release, r):
    """Ordinary rank-stabilised LEACE (predecessor wrapper) applied ONLY to the unprotected
    extension R, never to H or Z_J. Measured projection rank and class support are recorded."""
    from experiments.pcrl_invariant_baselines_v1.erasure_baselines import (
        _stabilized_leace, apply_affine, joint_onehot, cross_covariance_max_abs)
    rf = 'representation_fit'
    base = ext.HA + ext.Z_WIDTH
    x = {p: np.asarray(source_release[f'wire/A/{p}'], np.float64)[:, base:base + r] for p in dax.POOLS}
    labels = ctx['labels']['protected']
    z, complete, coverage = joint_onehot(labels, len(x[rf]))
    fitted = _stabilized_leace(x[rf][complete], z)
    values = {p: apply_affine(x[p], fitted['projection'], fitted['mean']) for p in dax.POOLS}
    meta = {'realised_projection_rank': fitted['projection_retained_rank'],
            'input_retained_rank': fitted['input_retained_rank'], 'coverage': coverage,
            'fit_rows': int(complete.sum()),
            'cross_covariance_max_abs_before': cross_covariance_max_abs(x[rf][complete], z),
            'cross_covariance_max_abs_after': cross_covariance_max_abs(values[rf][complete], z),
            'rank_zero': int(fitted['projection_retained_rank']) == 0,
            'projection': fitted['projection'], 'mean': fitted['mean']}
    return values, meta


# ------------------------------------------------------------------ audit jobs
def audit_job(seed: int, name: str) -> dict:
    """Unchanged predecessor audit (run_dev_2018.evaluate_seed), exact-identity dedup, bounded
    retries with quarantine. No compaction, no deletion of fitted attacker weights."""
    from experiments.pcrl_direct_adversarial_v1 import run_dev_2018 as dev
    from . import portability
    portability.install(OUT / 'PORTABILITY_AMENDMENT_1.json')
    seed_dir = OUT / f'seed_{seed}'
    rpath = seed_dir / 'releases' / name / 'releases.npz'
    ident = release_identity(rpath)
    reg_dir = seed_dir / 'audit_identity'
    reg_dir.mkdir(parents=True, exist_ok=True)
    claim = reg_dir / f'{ident}.json'
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as fh:
            json.dump({'canonical': name}, fh)
    except FileExistsError:
        canonical = read_json(claim)['canonical']
        if canonical != name:
            write_json(seed_dir / name / 'duplicate.json', {'duplicate_of': canonical, 'identity': ident})
            return {'duplicate_of': canonical}
    if (seed_dir / name / 'complete.json').exists():
        return read_json(seed_dir / name / 'complete.json')
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = dev.evaluate_seed(OUT, seed, [name], dax.Registry.new())[name]
            res['identity'] = ident
            res['portability'] = portability.record()
            return res
        except Exception as exc:
            q = seed_dir / 'quarantine' / f'{name}_attempt{attempt}'
            q.parent.mkdir(parents=True, exist_ok=True)
            if (seed_dir / name).exists():
                shutil.move(str(seed_dir / name), str(q))
            q.mkdir(parents=True, exist_ok=True)
            write_json(q / 'quarantine.json', {'seed': seed, 'unit': name, 'attempt': attempt,
                                               'error': repr(exc), 'traceback': traceback.format_exc()})
    return {'failed': True}


# ------------------------------------------------------------------ points
_PRIOR = {}


def prior(seed):
    if seed not in _PRIOR:
        reg = dax.Registry.new()
        _PRIOR[seed] = {r['target']: r['scores'] for r in
                        read_json(reg.resolve(rep.PRIOR_PATH.format(seed=seed)))['raw_metrics']
                        if r['condition'] == 'prior'}
    return _PRIOR[seed]


def canonical(seed, name):
    dup = OUT / f'seed_{seed}' / name / 'duplicate.json'
    return read_json(dup)['duplicate_of'] if dup.exists() else name


def point(seed, name, split, weight):
    """Historical H is read through the predecessor resolver; everything else is this study's."""
    if name == 'H':
        path = rep.condition_file(Path('/nonexistent'), seed, 'H', 'metrics.json')
    else:
        path = OUT / f'seed_{seed}' / canonical(seed, name) / 'metrics.json'
    records = read_json(path)['raw_metrics']
    return rep.point_from_records(records, prior(seed), split, weight, MAIN_BUDGET, MAIN_SCOPE)


def epca_source(seed, split, weight):
    from experiments.pcrl_competitive_method_v1 import evidence as ev
    return ev.epca_source(seed, split, weight)


# ------------------------------------------------------------------ pilot gate / screen
def screen(names, seeds=SEEDS) -> dict:
    """Validation-only screen. Reads no test split, no residence, no commute."""
    rows = {}
    for name in names:
        rec = {'config': name, 'assessable': True, 'reasons': []}
        recon, src_ok, inc = [], True, {e: {w: [] for w in WEIGHTS} for e in SENSITIVE4}
        for seed in seeds:
            fr = OUT / f'seed_{seed}' / 'fits' / name / 'fit_record.json'
            done = OUT / f'seed_{seed}' / canonical(seed, name) / 'complete.json'
            if not fr.exists() or not done.exists():
                rec['assessable'] = False
                rec['reasons'].append(f'seed {seed} missing fit or audit')
                continue
            f = read_json(fr)
            recon.append(f['validation_reconstruction']['relative_reduction'])
            for w in WEIGHTS:
                p = point(seed, name, 'validation', w)
                j = point(seed, 'ref_J', 'validation', w)
                parent = epca_source(seed, 'validation', w)
                for t in SOURCE:
                    if p['utility'][t] - parent[t] > SOURCE_ALLOWANCE + 1e-12:
                        src_ok = False
                for e in SENSITIVE4:
                    if e not in p['gains'] or e not in j['gains']:
                        rec['assessable'] = False
                        rec['reasons'].append(f'role {e} unsupported')
                        continue
                    inc[e][w].append(p['gains'][e] - j['gains'][e])
        if not rec['assessable']:
            rec['pass'] = None
            rows[name] = rec
            continue
        recon_ok = sum(v >= GATE_RECON_REDUCTION for v in recon) >= 2
        mean_ok = all(np.mean(inc[e][w]) <= GATE_MEAN_INC for e in SENSITIVE4 for w in WEIGHTS)
        anchor_ok = all(max(inc[e][w]) <= GATE_ANCHOR_INC for e in SENSITIVE4 for w in WEIGHTS)
        rec.update(recon_reduction_by_anchor=recon, recon_mean=float(np.mean(recon)),
                   recon_ok=recon_ok, source_ok=src_ok, mean_increment_ok=mean_ok,
                   anchor_increment_ok=anchor_ok,
                   increments_vs_J={e: {w: inc[e][w] for w in WEIGHTS} for e in SENSITIVE4},
                   pass_=bool(recon_ok and src_ok and mean_ok and anchor_ok))
        rec['pass'] = rec.pop('pass_')
        rows[name] = rec
    return rows


def select_tier3(rows: dict) -> list:
    """At most two COALITION (C1) configurations passing the validation screen, ranked by mean
    validation reconstruction reduction; tie-break lower width then config ID."""
    ok = [r for n, r in rows.items() if n.startswith('X_') and '_C1_' in n and r.get('pass')]
    ok.sort(key=lambda r: (-round(r['recon_mean'], 6), unit_spec(r['config'])['r'], r['config']))
    return [r['config'] for r in ok[:2]]
