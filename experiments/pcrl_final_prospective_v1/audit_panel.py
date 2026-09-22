"""Frozen-recipe auditors and utility probes for the locked panel (fitting pools only).

One unit = (release, anchor, role). The fitted slate is the predecessor's unchanged
``catchup`` slate (logistic C=1, two exact-expanded and two sampled-token
HistGB, one 360-epoch MLP trajectory with nested 120/360 selections). Candidate
routes and their validation replay are the predecessor's ``routed_probabilities``.
Legal ancestors (decided by what is on the wire, never by preference):
  * every non-H release role: the same role's H-only candidates (wire H);
  * AB roles: all candidates of the same release's A role plus H's B role;
  * utility: the frozen fixed decoder when the release defines one, and the
    frozen public H_A global-offset recalibrators.
J is never an ancestor of a replacement release. Validation selection is the
balanced 0.5*U + 0.5*PWGTP expected log loss, lexical tie; the same selected
predictions serve both reporting weightings.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import joblib
import numpy as np

from experiments.pcrl_task_directed_release_v1.audits import (expected_token_loss, fit_slate, load_candidate,
                                                              loss_scores, select_losses)
from experiments.pcrl_task_directed_release_v1.evaluation import ROLE_SPECS, routed_probabilities, _wire
from .common import (ANCHORS, B_ROLES, OUT, PANEL, PRIMARY_ROLES, PRIVATE, TASK_ROLE, array_hash, atomic_json,
                     now, sha)
from .prepare import load_pools
from .releases import FrozenReleases

H_ROLES = ('attack:A/SEX', 'attack:A/RAC1P', 'attack:B/SEX', 'attack:B/RAC1P', TASK_ROLE,
           'attack:AB/SEX', 'attack:AB/RAC1P')
RELEASE_ROLES = ('attack:A/SEX', 'attack:A/RAC1P', TASK_ROLE, 'attack:AB/SEX', 'attack:AB/RAC1P')
SEED_BASE = 2016000


def roles_for(short):
    return H_ROLES if short == 'H' else RELEASE_ROLES


def parse(role):
    kind, rest = role.split(':')
    view, target = rest.split('/')
    return kind, view, target


def role_seed(anchor, role):
    index = [f'{k}:{r}' for k, r in ROLE_SPECS].index(role)
    return SEED_BASE+10000*int(anchor)+37*index


def dependencies(short, anchor, role):
    kind, view, target = parse(role)
    deps = []
    if short != 'H':
        deps.append(('H', anchor, role))
    if view == 'AB':
        deps += [(short, anchor, f'attack:A/{target}'), ('H', anchor, f'attack:B/{target}')]
    return list(dict.fromkeys(deps))


def all_units():
    return [(s, a, r) for s in PANEL for a in ANCHORS for r in roles_for(s)]


def unit_dir(root, short, anchor, role):
    return Path(root)/'units'/f'anchor_{anchor}'/short/role.replace(':', '__').replace('/', '__')


def _relative(path, root):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def load_unit(root, short, anchor, role):
    d = unit_dir(root, short, anchor, role)
    marker = d/'COMPLETE.json'
    if not marker.exists():
        raise FileNotFoundError(f'Missing completed unit {short}/{anchor}/{role}')
    record = json.loads(marker.read_text())
    if sha(d/'registry.joblib') != record['registry_sha256']:
        raise ValueError('Unit registry changed after completion')
    return joblib.load(d/'registry.joblib')


def materialize(record, root, cache):
    """Attach the candidate object for model routes (loaded from its saved directory)."""
    if record['route']['kind'] != 'model':
        return record
    path = Path(root)/record['model_path']
    if path not in cache:
        cache[path] = load_candidate(path)
        if cache[path].metadata.get('candidate_id') is None:
            raise ValueError('Candidate without identity')
    return {**record, 'candidate': cache[path]}


def _subset(data, mask):
    return {k: ({t: y[mask] for t, y in v.items()} if k == 'labels' else v[mask]) for k, v in data.items()}


def _subset_release(rel, mask):
    return {k: (None if v is None else np.asarray(v)[mask]) for k, v in rel.items()}


def build(dataset, short, anchor, pools, *, with_labels=True):
    data, enc = load_pools(dataset, anchor, pools, with_labels=with_labels)
    frozen = FrozenReleases(anchor)
    release = frozen.release(short, {p: {k: data[p][k] for k in ('x', 'ha', 'J')} for p in pools}, enc)
    return data, release


def fit_unit(root, dataset, short, anchor, role):
    root = Path(root)
    d = unit_dir(root, short, anchor, role)
    if (d/'COMPLETE.json').exists():
        return json.loads((d/'COMPLETE.json').read_text())
    if d.exists():
        quarantine = root/'quarantine'/f'{d.name}-{short}-{anchor}-{time.time_ns()}'
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        os.replace(d, quarantine)
    d.mkdir(parents=True)
    tick = time.perf_counter()
    kind, view, target = parse(role)
    val_pool = 'attack_val' if kind == 'attack' else 'task_val'
    data, release = build(dataset, short, anchor, ('fit', val_pool))
    n_classes = 9 if target == 'RAC1P' else 2
    is_h = short == 'H'
    own_wire = 'H' if view == 'B' or is_h else 'release'
    fm = data['fit']['labels'][target] >= 0
    vm = data[val_pool]['labels'][target] >= 0
    fit, fit_rel = _subset(data['fit'], fm), _subset_release(release['fit'], fm)
    val, val_rel = _subset(data[val_pool], vm), _subset_release(release[val_pool], vm)
    hfit, pfit = _wire(fit, fit_rel, view, own_wire)
    hval, pval = _wire(val, val_rel, view, own_wire)
    seed = role_seed(anchor, role)
    fitted = fit_slate(hfit, pfit, fit['labels'][target], fit['weights'], hval, pval, val['labels'][target],
                       val['weights'], n_classes, seed, d/'models', slate='catchup')
    candidates = {cid: {'route': {'kind': 'model', 'source_view': view, 'wire': own_wire},
                        'model_path': _relative(c.metadata['directory'], root), 'origin': 'independent',
                        'source_role': role}
                  for cid, c in fitted['candidates'].items()}
    independent_ids = sorted(candidates)
    ancestors = []
    if not is_h:
        h_reg = load_unit(root, 'H', anchor, role)
        for cid, rec in h_reg['candidates'].items():
            if rec['route']['kind'] != 'model':
                continue
            if rec['route']['wire'] != 'H':
                raise ValueError('An H ancestor attempted to read release data')
            candidates['H__'+cid] = {**rec, 'origin': 'H_baseline', 'inherited_from': f'H/{anchor}/{role}/{cid}'}
        ancestors.append(f'H/{anchor}/{role}')
    if view == 'AB':
        for anc_short, anc_view in ((short, 'A'), ('H', 'B')):
            src = f'attack:{anc_view}/{target}'
            reg = load_unit(root, anc_short, anchor, src)
            for cid, rec in reg['candidates'].items():
                candidates[f'ancestor_{anc_view}__{cid}'] = {**rec, 'origin': f'{anc_view}_ancestor',
                                                           'inherited_from': f'{anc_short}/{anchor}/{src}/{cid}'}
            ancestors.append(f'{anc_short}/{anchor}/{src}')
    if kind == 'utility':
        if 'fixed_probabilities' in val_rel and val_rel['fixed_probabilities'] is not None:
            candidates['fixed_decoder'] = {'route': {'kind': 'fixed_decoder', 'source_view': 'A', 'wire': 'release'},
                                           'origin': 'deployment_decoder', 'source_role': role}
        offsets = val_rel.get('global_offsets')
        if offsets is not None:
            n_actions = offsets.shape[1]
            for action in range(n_actions):
                candidates[f'global_offset_{action:04d}'] = {
                    'route': {'kind': 'global_offset', 'source_view': 'A', 'wire': 'H', 'action_index': action,
                              'n_actions': n_actions}, 'origin': 'public_H_recalibration', 'source_role': role}
    cache = {}
    losses = {}
    for cid, rec in candidates.items():
        q, p = routed_probabilities(materialize(rec, root, cache), val, val_rel)
        losses[cid] = expected_token_loss(q, p, val['labels'][target])
    choice = select_losses(losses, val['weights'])
    independent = select_losses({c: losses[c] for c in independent_ids}, val['weights'])['selection']
    ids = sorted(candidates)
    np.savez_compressed(d/'val_losses.npz', ids=val['ids'], households=val['households'], weights=val['weights'],
                        y=val['labels'][target], candidate_ids=np.asarray(ids),
                        losses=np.column_stack([losses[c] for c in ids]))
    registry = {'schema': 1, 'dataset': dataset, 'release': short, 'family': PANEL[short], 'anchor': anchor,
                'role': role, 'view': view, 'target': target, 'kind': kind, 'validation_pool': val_pool,
                'candidates': candidates, 'selection': choice['selection'], 'independent_selection': independent,
                'validation_scores': choice['scores'], 'fit_rows': int(len(fit['ids'])),
                'validation_rows': int(len(val['ids'])), 'seed': seed, 'ancestors': ancestors,
                'fit_hashes': {'h': array_hash(hfit), 'tokens': array_hash(pfit), 'y': array_hash(fit['labels'][target]),
                               'weights': array_hash(fit['weights'])},
                'validation_hashes': {'h': array_hash(hval), 'tokens': array_hash(pval),
                                      'y': array_hash(val['labels'][target]), 'weights': array_hash(val['weights'])}}
    joblib.dump(registry, d/'registry.joblib', compress=3)
    artifacts = {_relative(p, root): sha(p) for p in sorted(d.rglob('*')) if p.is_file()}
    record = {'created_utc': now(), 'dataset': dataset, 'release': short, 'anchor': anchor, 'role': role,
              'selection': choice['selection'], 'independent_selection': independent,
              'selected_validation': choice['scores'][choice['selection']],
              'independent_validation': choice['scores'][independent],
              'standard_selection': fitted['standard_selection'],
              'registry_sha256': sha(d/'registry.joblib'), 'seconds': time.perf_counter()-tick,
              'peak_rss_bytes': _peak_rss(), 'n_tokens': int(pfit.shape[1]),
              'expanded_fit_rows': int(np.count_nonzero(pfit)), 'fit_rows': int(len(fit['ids'])),
              'artifact_count': len(artifacts), 'artifacts_sha256': sha_of(artifacts)}
    atomic_json(d/'ARTIFACTS.json', artifacts)
    atomic_json(d/'COMPLETE.json', record)
    return record


def sha_of(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _peak_rss():
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if os.uname().sysname == 'Darwin' else int(r)*1024


# ------------------------------------------------------------------ scoring
def score_unit(root, dataset, short, anchor, role, *, pool='final'):
    """Replay frozen selections on the scoring pool; no fitting or selection."""
    root = Path(root)
    d = unit_dir(root, short, anchor, role)
    out = d/f'score_{pool}.npz'
    marker = d/f'SCORED_{pool}.json'
    if marker.exists():
        return json.loads(marker.read_text())
    reg = load_unit(root, short, anchor, role)
    kind, view, target = parse(role)
    data, release = build(dataset, short, anchor, (pool,))
    mask = data[pool]['labels'][target] >= 0
    rows, rel = _subset(data[pool], mask), _subset_release(release[pool], mask)
    cache = {}
    y = rows['labels'][target]
    q, p = routed_probabilities(materialize(reg['candidates'][reg['selection']], root, cache), rows, rel)
    _validate_probabilities(q, p, len(y))
    loss = expected_token_loss(q, p, y)
    q2, p2 = routed_probabilities(materialize(reg['candidates'][reg['selection']], root, {}), rows, rel)
    if not (np.array_equal(q, q2) and np.array_equal(p, p2)):
        raise FloatingPointError('Selected prediction differs between two in-process computations')
    accuracy = np.sum(p*(q.argmax(2) == y[:, None]), axis=1)
    qi, pi = routed_probabilities(materialize(reg['candidates'][reg['independent_selection']], root, cache), rows, rel)
    independent = expected_token_loss(qi, pi, y)
    arrays = {'ids': rows['ids'], 'households': rows['households'], 'weights': rows['weights'], 'y': y,
              'loss': loss, 'accuracy': accuracy, 'independent_loss': independent}
    if 'fixed_decoder' in reg['candidates']:
        qf, pf = routed_probabilities(reg['candidates']['fixed_decoder'], rows, rel)
        arrays['fixed_decoder_loss'] = expected_token_loss(qf, pf, y)
    np.savez_compressed(out, **arrays)
    np.savez_compressed(d/f'pred_{pool}.npz', probabilities=q, token_probs=p)
    record = {'created_utc': now(), 'pool': pool, 'release': short, 'anchor': anchor, 'role': role,
              'selection': reg['selection'], 'rows': int(len(y)), 'score_sha256': sha(out),
              'prediction_sha256': sha(d/f'pred_{pool}.npz'),
              'ce': loss_scores(loss, rows['weights']), 'independent_ce': loss_scores(independent, rows['weights']),
              'fixed_decoder_ce': (loss_scores(arrays['fixed_decoder_loss'], rows['weights'])
                                   if 'fixed_decoder_loss' in arrays else None),
              'loss_hash': array_hash(loss), 'prediction_hash': array_hash(q), 'token_hash': array_hash(p)}
    atomic_json(marker, record)
    return record


def _validate_probabilities(q, p, n):
    if q.ndim != 3 or q.shape[0] != n or p.shape != q.shape[:2]:
        raise ValueError('Unexpected prediction shape')
    if not np.isfinite(q).all() or (q < 0).any():
        raise FloatingPointError('Nonfinite or negative probability')
    sums = q.sum(2)
    if (sums == 0).any():
        raise FloatingPointError('Zero-sum probability row')
    if not np.allclose(sums, 1, atol=1e-6):
        raise FloatingPointError('Unnormalized probability row')
    if not np.allclose(p.sum(1), 1, atol=1e-8) or (p < 0).any():
        raise FloatingPointError('Invalid token law')


# --------------------------------------------------------------- scheduling
def run_all(root, dataset, workers, *, phase='fit', only=None):
    """Dependency-aware process pool; a unit starts when its dependencies completed."""
    import multiprocessing as mp
    units = all_units() if only is None else only
    done, failed, running = set(), {}, {}
    status_path = Path(root)/f'{phase.upper()}_STATUS.json'
    for u in units:
        m = unit_dir(root, *u)/('COMPLETE.json' if phase == 'fit' else 'SCORED_final.json')
        if m.exists():
            done.add(u)
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(workers, maxtasksperchild=1)
    attempts = {}
    target = score_worker if phase == 'score' else fit_worker
    try:
        while len(done)+len(failed) < len(units):
            for u in units:
                if u in done or u in running or u in failed:
                    continue
                deps = dependencies(*u) if phase == 'fit' else []
                if any(dep in failed for dep in deps):
                    failed[u] = 'dependency failed'
                    continue
                if all(dep in done for dep in deps) and len(running) < workers:
                    attempts[u] = attempts.get(u, 0)+1
                    running[u] = pool.apply_async(target, (str(root), dataset, *u))
            for u, res in list(running.items()):
                if res.ready():
                    del running[u]
                    try:
                        res.get()
                        done.add(u)
                    except Exception as error:
                        if attempts[u] < 3:
                            print('retrying', u, repr(error), flush=True)
                        else:
                            failed[u] = repr(error)
            atomic_json(status_path, {'updated_utc': now(), 'phase': phase, 'dataset': dataset, 'units': len(units),
                                      'done': len(done), 'running': [list(u) for u in running],
                                      'failed': {'|'.join(map(str, k)): v for k, v in failed.items()}})
            time.sleep(2)
    finally:
        pool.close()
        pool.join()
    return {'done': len(done), 'failed': failed}


def fit_worker(root, dataset, short, anchor, role):
    _limit_threads()
    try:
        return fit_unit(root, dataset, short, anchor, role)
    except Exception:
        log = Path(root)/'logs'/f'fit-{short}-{anchor}-{role.replace(":", "_").replace("/", "_")}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as f:
            f.write(traceback.format_exc())
        raise


def score_worker(root, dataset, short, anchor, role):
    _limit_threads()
    return score_unit(root, dataset, short, anchor, role)


def _limit_threads():
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'
    import torch
    torch.set_num_threads(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=('acs2016', 'emulated', 'smoke'))
    ap.add_argument('--root', required=True)
    ap.add_argument('--phase', choices=('fit', 'score'), default='fit')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    print(json.dumps(run_all(a.root, a.dataset, a.workers, phase=a.phase), indent=1, default=str))
