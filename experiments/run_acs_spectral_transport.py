"""Resumable runner for the locked 2017 transport study.

Phases: replay (2018 bitwise pipeline identity), releases (2017 fitting and
validation partitions only), fit (Mode B), lock, score (sealed final partition,
Modes A and B). Each unit writes a completion record and is skipped on resume.
"""
from __future__ import annotations
import argparse, os, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import joblib
import numpy as np
from experiments import acs_spectral_transport_eval as ev
from experiments import acs_fixed_predictions_audits as old
from experiments import acs_spectral_audits as spec
from experiments.acs_transfer_data import array_hash, sha_file, write_json
from experiments.acs_transfer_heads import load_candidate, metrics

_CACHE = {}


def _init():
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    _CACHE['limits'] = threadpool_limits(limits=1)


def frozen(seed):
    if ('frozen', seed) not in _CACHE: _CACHE['frozen', seed] = ev.FrozenSeed(seed)
    return _CACHE['frozen', seed]


def fit_inputs(out, seed):
    key = ('fit_inputs', seed)
    if key not in _CACHE:
        rel = {p: ev.load_releases(Path(out)/f'seed_{seed}'/'releases_2017'/f'{p}.npz') for p in ev.FIT_PARTITIONS}
        labels = {p: ev.load_fit_partition(p)['labels'] for p in ev.FIT_PARTITIONS}
        _CACHE[key] = (rel, labels)
    return _CACHE[key]


def events(out, record):
    path = Path(out)/'RUNTIME_EVENTS.json'
    log = ev.read(path) if path.exists() else []
    log.append(record); write_json(path, log)


# ------------------------------------------------------------------ phases
def phase_replay(out, seeds):
    for seed in seeds:
        path = Path(out)/f'seed_{seed}'/'REPLAY_2018.json'
        if path.exists(): continue
        tick = time.perf_counter(); report = ev.replay_2018(frozen(seed), seed)
        report['runtime_seconds'] = time.perf_counter()-tick
        write_json(path, report); print('REPLAY', seed, report['all_bitwise_equal'], flush=True)
        if not report['all_bitwise_equal']: raise SystemExit('2018 pipeline identity failed; transport blocked')


def phase_releases(out, seeds):
    for seed in seeds:
        for part in ev.FIT_PARTITIONS:
            path = Path(out)/f'seed_{seed}'/'releases_2017'/f'{part}.npz'
            if path.exists(): continue
            data = ev.load_fit_partition(part)
            ev.save_releases(path, ev.build_releases(frozen(seed), data['frame']))
        support_path = Path(out)/f'seed_{seed}'/'feature_support_fit_partitions.json'
        if not support_path.exists():
            write_json(support_path, {p: ev.unseen_category_counts(frozen(seed).pre, ev.load_fit_partition(p)['frame']) for p in ev.FIT_PARTITIONS})
        print('RELEASES', seed, flush=True)


def _fit_job(out, seed, condition):
    tick = time.perf_counter()
    try:
        rel, labels = fit_inputs(out, seed)
        if condition == 'reference': r = ev.fit_reference(out, seed, rel, labels)
        else: r = ev.fit_unit(out, seed, condition, rel, labels, frozen(seed))
        return {'seed': seed, 'condition': condition, 'ok': True, 'seconds': time.perf_counter()-tick, 'record': r}
    except Exception:
        return {'seed': seed, 'condition': condition, 'ok': False, 'seconds': time.perf_counter()-tick, 'error': traceback.format_exc()}


def run_pool(out, jobs, workers, label):
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futures = {pool.submit(fn, out, *args): args for fn, args in jobs}
        for f in as_completed(futures):
            r = f.result(); results.append(r)
            events(out, {'phase': label, 'utc': ev.now(), **{k: v for k, v in r.items() if k != 'record'}})
            print(label, r['seed'], r['condition'], 'ok' if r['ok'] else 'FAILED', round(r['seconds'], 1), flush=True)
            if not r['ok']: print(r['error'], flush=True)
    return results


def phase_fit(out, seeds, workers, only=None):
    first = [(_fit_job, (s, c)) for s in seeds for c in ('H', 'reference') if only is None or c in only]
    run_pool(out, first, workers, 'FIT')
    rest = [(_fit_job, (s, c)) for s in seeds for c in ev.INTERFACES if c != 'H' and (only is None or c in only)]
    run_pool(out, rest, workers, 'FIT')


def phase_lock(out, seeds):
    out = Path(out)
    assert not ev.lock_path(out).exists(), 'Lock already written'
    missing = [(s, c) for s in seeds for c in (*ev.INTERFACES, 'reference') if not (out/f'seed_{s}'/c/'fit_complete.json').exists()]
    if missing: raise SystemExit(f'Incomplete Mode B fitting: {missing}')
    files = {}
    def add(p):
        p = Path(p).resolve()
        key = str(p.relative_to(ev.ROOT)) if p.is_relative_to(ev.ROOT) else str(p)
        files[key] = sha_file(p)
    for p in (out/'PROTOCOL.md', out/'COMPARISONS.json', out/'PROTOCOL_FREEZE.json'): add(p)
    for p in ('experiments/acs_spectral_transport_eval.py', 'experiments/run_acs_spectral_transport.py',
              'scripts/report_acs_spectral_transport.py', 'experiments/acs_spectral_audits.py',
              'experiments/acs_fixed_predictions_audits.py', 'experiments/acs_fixed_predictions_training.py',
              'experiments/acs_preservation_audits.py', 'experiments/acs_transfer_heads.py',
              'experiments/acs_transfer_data.py', 'experiments/acs_residual_spectral.py'):
        add(ev.ROOT/p)
    for s in seeds:
        fz = frozen(s)
        for p in fz.object_files(): add(p)
        for p in sorted((out/f'seed_{s}'/'releases_2017').glob('*.npz')): add(p)
        add(out/f'seed_{s}'/'REPLAY_2018.json')
        for c in (*ev.INTERFACES, 'reference'):
            d = out/f'seed_{s}'/c
            for p in sorted(d.rglob('*')):
                if p.is_file() and p.name != '.DS_Store': add(p)
        for c in ev.INTERFACES:
            add(ev.DEV/f'seed_{s}'/c/'audits'/'audit_selection.json'); add(ev.DEV/f'seed_{s}'/c/'selection_before_test.json')
    for p in ('2017_attacker_fit.npz', '2017_attacker_validation.npz', '2017_task_fit.npz', '2017_task_validation.npz', '2017_final_evaluation.npz'):
        add(ev.LOCAL/p)
    final_hash = sha_file(ev.LOCAL/'2017_final_evaluation.npz')
    lock = {'created_utc': ev.now(), 'protocol_sha256': sha_file(out/'PROTOCOL.md'), 'comparisons_sha256': sha_file(out/'COMPARISONS.json'),
            'final_partition_sha256': final_hash, 'final_outcomes_scored_before_lock': 0, 'seeds': list(seeds),
            'interfaces': list(ev.INTERFACES), 'file_count': len(files), 'files': files}
    write_json(ev.lock_path(out), lock)
    print('LOCK', len(files), sha_file(ev.lock_path(out)), flush=True)


# ----------------------------------------------------------------- scoring
def _score_rows(y, w, preds, k):
    valid = y >= 0
    return {'test': metrics(y[valid], preds, k), 'test_person_weighted': metrics(y[valid], preds, k, w[valid])}


def _final_inputs(out, seed):
    key = ('final', seed)
    if key not in _CACHE:
        final = ev.load_final(out)
        path = Path(out)/f'seed_{seed}'/'releases_2017'/'final_evaluation.npz'
        if not path.exists(): ev.save_releases(path, ev.build_releases(frozen(seed), final['frame']))
        _CACHE[key] = (final, ev.load_releases(path))
    return _CACHE[key]


class Store:
    def __init__(self): self.arrays, self.keys = {}, {}

    def put(self, key, p):
        p = np.ascontiguousarray(p, dtype=np.float64); h = array_hash(p)
        self.arrays.setdefault(h, p); self.keys[key] = h

    def save(self, path):
        np.savez_compressed(path, **{'p/'+h: a for h, a in self.arrays.items()},
                            keys=np.asarray(list(self.keys)), ids=np.asarray(list(self.keys.values())))


def _bundle(rel, condition, view, valid):
    b = {'wire': rel['wire'][condition][view][valid]}
    if rel['derived'][condition] is not None: b['derived'] = rel['derived'][condition][view][valid]
    return b


def score_mode_b(out, seed, condition):
    dest = Path(out)/f'seed_{seed}'/condition/'mode_B'
    if (dest/'complete.json').exists(): return
    tick = time.perf_counter(); dest.mkdir(parents=True, exist_ok=True)
    final, rel = _final_inputs(out, seed); y, w = final['labels'], final['weights']
    record = ev.read(Path(out)/f'seed_{seed}'/condition/'audit_selection.json')
    util = ev.read(Path(out)/f'seed_{seed}'/condition/'utility_selection.json')
    cache = ev.Loaded(); store = Store(); rows = []
    for role, cs in util['candidates'].items():
        view, t = role.split('/'); valid = y[t] >= 0
        for cid, c in cs.items():
            p = load_candidate(c['dir']).predict_proba(rel['wire'][condition][view][valid])
            store.put(f'utility/{role}/{cid}', p)
            rows.append({'role': 'utility', 'view': view, 'target': t, 'candidate_id': cid,
                         'selected': util['selection'][role] == cid, 'scores': _score_rows(y[t], w, p, 2)})
    for b, roles in record['candidates'].items():
        for role, cs in roles.items():
            view, t = role.split('/'); valid = y[t] >= 0; bundle = _bundle(rel, condition, view, valid)
            chosen = record['selections'][b][role]
            for cid, r in cs.items():
                p = cache.predict(r, bundle, 'final'); store.put(f'audit/{b}/{role}/{cid}', p)
                rows.append({'role': 'audit', 'view': view, 'target': t, 'audit_budget': int(b), 'candidate_id': cid,
                             'transport_origin': r['transport_origin'], 'space': r['space'], 'family': r.get('family'),
                             'anchor_ancestor': r['anchor_ancestor'], 'inherited_singleton': r['inherited_singleton'],
                             'diagnostic_only': r['diagnostic_only'],
                             'selected_scopes': [s for s, c in chosen.items() if c == cid],
                             'validation_log_loss': record['validation_scores'][b][role][cid]['log_loss'],
                             'scores': _score_rows(y[t], w, p, old.CLASSES[t])})
    store.save(dest/'predictions.npz')
    write_json(dest/'metrics.json', {'seed': seed, 'condition': condition, 'mode': 'B', 'raw_metrics': rows})
    write_json(dest/'complete.json', {'utc': ev.now(), 'runtime_seconds': time.perf_counter()-tick,
        'metrics_sha256': sha_file(dest/'metrics.json'), 'predictions_sha256': sha_file(dest/'predictions.npz'), 'rows': len(rows)})


def _dev_utilities(seed, condition):
    dev = ev.DEV/f'seed_{seed}'/condition
    if condition in ev.SPECTRAL:
        u = joblib.load(dev/'utility_state.joblib')
        return u['selection'], {role: dict(cs) for role, cs in u['candidates'].items()}
    sel = ev.read(ev.FIXED/f'seed_{seed}'/condition/'selection_before_test.json')['utility']
    cands = {}
    for role in sel:
        parent = ev.FIXED/f'seed_{seed}'/('H' if role.startswith('B/') else condition)
        cands[role] = {cid: spec.load_base(parent/'fitted/utility'/role/cid) for cid in ('logistic', 'mlp')}
    return sel, cands


def score_mode_a(out, seed, condition):
    dest = Path(out)/f'seed_{seed}'/condition/'mode_A'
    if (dest/'complete.json').exists(): return
    tick = time.perf_counter(); dest.mkdir(parents=True, exist_ok=True)
    final, rel = _final_inputs(out, seed); y, w = final['labels'], final['weights']
    store = Store(); rows = []; pcache = {}
    sel, cands = _dev_utilities(seed, condition)
    assert sel == ev.read(ev.DEV/f'seed_{seed}'/condition/'selection_before_test.json')['utility']
    for role, cs in cands.items():
        view, t = role.split('/'); valid = y[t] >= 0
        for cid, c in cs.items():
            p = c.predict_proba(rel['wire'][condition][view][valid]); store.put(f'utility/{role}/{cid}', p)
            rows.append({'role': 'utility', 'view': view, 'target': t, 'candidate_id': cid, 'selected': sel[role] == cid,
                         'scores': _score_rows(y[t], w, p, 2)})
    audits = spec.load_audits(ev.DEV/f'seed_{seed}'/condition/'audits')
    for b, roles in audits['candidates'].items():
        for role, cs in roles.items():
            view, t = role.split('/'); valid = y[t] >= 0; bundle = _bundle(rel, condition, view, valid)
            chosen = audits['selection'][b][role]
            for cid, c in cs.items():
                x = bundle[c.space]; x = x[:, c.columns] if c.columns is not None else x; x = np.ascontiguousarray(x)
                key = (c.metadata['base_candidate_directory'], array_hash(x))
                if key not in pcache: pcache[key] = c.base.predict_proba(x)
                p = pcache[key]; store.put(f'audit/{b}/{role}/{cid}', p)
                rows.append({'role': 'audit', 'view': view, 'target': t, 'audit_budget': int(b), 'candidate_id': cid,
                             'space': c.space, 'family': c.metadata.get('family'), 'candidate_origin': c.metadata.get('candidate_origin'),
                             'anchor_ancestor': bool(c.metadata.get('anchor_ancestor')), 'inherited_singleton': bool(c.metadata.get('inherited_singleton')),
                             'diagnostic_only': bool(c.metadata.get('diagnostic_only')),
                             'selected_scopes': [s for s, sc in chosen.items() if sc == cid],
                             'scores': _score_rows(y[t], w, p, old.CLASSES[t])})
    store.save(dest/'predictions.npz')
    write_json(dest/'metrics.json', {'seed': seed, 'condition': condition, 'mode': 'A', 'raw_metrics': rows})
    write_json(dest/'complete.json', {'utc': ev.now(), 'runtime_seconds': time.perf_counter()-tick,
        'metrics_sha256': sha_file(dest/'metrics.json'), 'predictions_sha256': sha_file(dest/'predictions.npz'), 'rows': len(rows)})


def score_seed_context(out, seed):
    """Priors, reference probes (both modes), service quality and moment diagnostics."""
    dest = Path(out)/f'seed_{seed}'/'context'
    if (dest/'complete.json').exists(): return
    tick = time.perf_counter(); dest.mkdir(parents=True, exist_ok=True)
    final, rel = _final_inputs(out, seed); y, w = final['labels'], final['weights']; store = Store(); rows = []
    ref = ev.read(Path(out)/f'seed_{seed}'/'reference'/'selection.json')
    for key, r in ref['reference_probes'].items():
        name, t = key.split('/'); valid = y[t] >= 0
        x = rel['pca'] if name == 'E_pca' else rel['banks'][name]
        for cid, d in r['dirs'].items():
            p = load_candidate(d).predict_proba(x[valid]); store.put(f'B/reference/{key}/{cid}', p)
            rows.append({'mode': 'B', 'kind': 'reference', 'release': name, 'target': t, 'candidate_id': cid, 'selected': cid == r['selected'], 'scores': _score_rows(y[t], w, p, 2)})
    hsel = ev.read(ev.PROTECTION/f'seed_{seed}'/'selection_before_test.json')['head_selections']
    for name, tasks in (('E_pca', ev.TASKS), ('B_rich_bank', ('same_residence',)), ('C_tree_bank', ('same_residence',))):
        for t in tasks:
            valid = y[t] >= 0; x = rel['pca'] if name == 'E_pca' else rel['banks'][name]
            cid = hsel[f'transfer/{name}/{t}']
            p = load_candidate(ev.PROTECTION/f'seed_{seed}'/'fitted/transfer'/name/t/cid).predict_proba(x[valid])
            store.put(f'A/reference/{name}/{t}/{cid}', p)
            rows.append({'mode': 'A', 'kind': 'reference', 'release': name, 'target': t, 'candidate_id': cid, 'selected': True, 'scores': _score_rows(y[t], w, p, 2)})
    for t in ev.TARGETS:
        valid = y[t] >= 0; k = old.CLASSES[t]
        for mode, path in (('B', ref['priors'][t]), ('A', ev.COALITION/f'seed_{seed}'/'controls/fitted/prior'/t)):
            p = load_candidate(path).predict_proba(np.zeros((int(valid.sum()), 1)))
            store.put(f'{mode}/prior/{t}', p)
            rows.append({'mode': mode, 'kind': 'prior', 'target': t, 'scores': _score_rows(y[t], w, p, k)})
    for t, cols in (('income_binary', rel['anchors']['A'][:, 0:2]), ('civilian_at_work', rel['anchors']['A'][:, 2:4]), ('public_coverage', rel['anchors']['B'])):
        valid = y[t] >= 0; p = cols[valid]; store.put(f'service/{t}', p)
        rows.append({'mode': 'both', 'kind': 'service', 'target': t, 'scores': _score_rows(y[t], w, p, 2)})
    model = frozen(seed).spectral
    moments = {}
    for part in (*ev.FIT_PARTITIONS, ev.FINAL):
        r = _final_inputs(out, seed)[1] if part == ev.FINAL else ev.load_releases(Path(out)/f'seed_{seed}'/'releases_2017'/f'{part}.npz')
        labs = final['labels'] if part == ev.FINAL else ev.load_fit_partition(part)['labels']
        moments[part] = model.moment_diagnostics(r['pca'], r['anchors']['A'], r['anchors']['B'], {k: labs[k] for k in ('SEX', 'RAC1P', 'public_coverage')})
    write_json(dest/'transport_moments.json', moments)
    store.save(dest/'predictions.npz')
    write_json(dest/'metrics.json', {'seed': seed, 'rows': rows})
    write_json(dest/'feature_support_final.json', ev.unseen_category_counts(frozen(seed).pre, final['frame']))
    write_json(dest/'complete.json', {'utc': ev.now(), 'runtime_seconds': time.perf_counter()-tick, 'metrics_sha256': sha_file(dest/'metrics.json')})


def _score_job(out, seed, condition):
    tick = time.perf_counter()
    try:
        if condition == 'context': score_seed_context(out, seed)
        else:
            score_mode_a(out, seed, condition); score_mode_b(out, seed, condition)
        return {'seed': seed, 'condition': condition, 'ok': True, 'seconds': time.perf_counter()-tick}
    except Exception:
        return {'seed': seed, 'condition': condition, 'ok': False, 'seconds': time.perf_counter()-tick, 'error': traceback.format_exc()}


def phase_score(out, seeds, workers):
    if ev.DRY_RUN_PARTITION is None:
        ev.verify_lock(out); os.environ['PCRL_TRANSPORT_LOCK_VERIFIED'] = sha_file(ev.lock_path(out))
    final = ev.load_final(out)
    labels = Path(out)/'final_labels.npz'
    if not labels.exists():
        np.savez_compressed(labels, **{'y/'+t: v for t, v in final['labels'].items()}, weights=final['weights'],
                            serialno=final['serialno'], sporder=final['sporder'], raw_rows=final['raw_rows'])
    jobs = [(_score_job, (s, c)) for s in seeds for c in ('context', *ev.INTERFACES)]
    run_pool(out, jobs, workers, 'SCORE')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ev.OUT)
    p.add_argument('--phase', choices=['replay', 'releases', 'fit', 'lock', 'score'], required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--only', nargs='+')
    a = p.parse_args(); _init(); tick = time.perf_counter(); error = None
    try:
        if a.phase == 'replay': phase_replay(a.out, a.seeds)
        elif a.phase == 'releases': phase_releases(a.out, a.seeds)
        elif a.phase == 'fit': phase_fit(a.out, a.seeds, a.workers, a.only)
        elif a.phase == 'lock': phase_lock(a.out, a.seeds)
        else: phase_score(a.out, a.seeds, a.workers)
    except BaseException as exc:
        error = repr(exc); raise
    finally:
        events(a.out, {'phase': a.phase, 'utc': ev.now(), 'elapsed_seconds': time.perf_counter()-tick, 'seeds': a.seeds, 'error': error, 'process': True})


if __name__ == '__main__':
    main()
