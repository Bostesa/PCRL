"""INNER reporting: opened once after the committed REPORTING_LOCK.

Primary simultaneous family (fixed at SELECTION_LOCK): candidate J_TRUST's 24
original INNER utility clauses plus three privacy endpoints per point-eligible
comparison. Separate diagnostic family for every other declared contrast.
Inherited paired duplicate-group bootstrap (2048 draws, fixed seed, shared row
multiplicities for every service and both source seeds); interval = point +/-
Student-t(1-alpha/(2K), G-1) * bootstrap SD (ddof=1). Statuses: PASS,
ESTABLISHED_VIOLATION, UNRESOLVED, NOT_ESTIMABLE.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BOOTSTRAP_SEED = 2026101301
REPLICATES = 2048
ALPHA = 0.05
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
PRIVACY = {'pair': ('benefit', 0.020), 'r1': ('excess', 0.010), 'r2': ('excess', 0.010)}
TASKS = ('income', 'occupation')


def envelope_keys(view, names):
    keys = [f'{view}__{n}' for n in names]
    if view == 'pair':
        keys += [f'pair__IGNORE_{o}__{n}' for o in ('r1', 'r2') for n in names]
    return keys


def classify(point, low, high, threshold, direction):
    """direction 'at_most' (excess <= cap) or 'greater_than' (benefit lower bound > threshold)."""
    if direction == 'at_most':
        return 'PASS' if high <= threshold else ('ESTABLISHED_VIOLATION' if low > threshold else 'UNRESOLVED')
    if direction == 'greater_than':
        return 'PASS' if low > threshold else ('ESTABLISHED_VIOLATION' if high < threshold else 'UNRESOLVED')
    raise ValueError('direction')


def diagnostic_classify(low, high):
    return 'REDUCTION_ESTABLISHED' if low > 0 else ('INCREASE_ESTABLISHED' if high < 0 else 'UNRESOLVED')


def critical(K, groups):
    from scipy.stats import t
    return float(t.ppf(1 - ALPHA / (2 * K), df=groups - 1))


def fixtures():
    """Data-free validation of the bound arithmetic (run before INNER opens)."""
    import numpy as np
    from compressed_income_safeguard.reporting import weighted_auc_draws, bootstrap_weights
    checks = {}
    y = np.array([0, 0, 1, 1]); s = np.array([.2, .5, .5, .8])
    checks['tie_aware_weighted_auc'] = bool(np.allclose(weighted_auc_draws(y, s, np.array([[1., 1, 1, 1], [1, 2, 3, 1]])), [.875, .75], atol=1e-15, rtol=0))
    w = np.array([[1., 1, 1, 1], [10, 1, 1, 10], [1, 10, 10, 1]])
    a = weighted_auc_draws(y, np.array([0, 2, 1, 3]), w); b = weighted_auc_draws(y, np.array([2, 0, 3, 1]), w)
    checks['envelope_reselected_inside_each_draw'] = bool(a[1] > b[1] and b[2] > a[2])
    g = np.array([11, 11, 22, 33]); m = bootstrap_weights(g, B=128, seed=BOOTSTRAP_SEED)
    checks['duplicate_group_shared_multiplicity'] = bool(np.array_equal(m[:, 0], m[:, 1]))
    checks['draws_reproducible_across_seeds_and_services'] = bool(np.array_equal(m, bootstrap_weights(g, B=128, seed=BOOTSTRAP_SEED)))
    checks['critical_value_bonferroni_t'] = abs(critical(39, 2234) - 3.2240705) < 1e-6 and critical(39, 2234) > 3.2199675 and critical(1, 2234) < critical(39, 2234)
    checks['classification'] = (classify(.005, 0., .010, .010, 'at_most') == 'PASS' and classify(.012, .011, .013, .010, 'at_most') == 'ESTABLISHED_VIOLATION'
                                and classify(.009, .007, .011, .010, 'at_most') == 'UNRESOLVED'
                                and classify(.03, .021, .04, .02, 'greater_than') == 'PASS' and classify(.03, .020, .04, .02, 'greater_than') == 'UNRESOLVED'
                                and classify(.005, -.01, .019, .02, 'greater_than') == 'ESTABLISHED_VIOLATION'
                                and diagnostic_classify(.001, .01) == 'REDUCTION_ESTABLISHED' and diagnostic_classify(-.01, -.001) == 'INCREASE_ESTABLISHED')
    keys = envelope_keys('pair', ['A', 'B'])
    checks['pair_envelope_includes_ignore_aliases'] = len(keys) == 6 and 'pair__IGNORE_r2__A' in keys
    return checks


def _committed(spec, path, sha):
    root = Path(spec['repo_root'])
    rel = Path(path).resolve().relative_to(root.resolve()).as_posix()
    r = subprocess.run(['git', 'show', 'HEAD:' + rel], cwd=root, capture_output=True, check=False)
    if r.returncode or hashlib.sha256(r.stdout).hexdigest() != sha:
        raise PermissionError('lock must be committed at HEAD')


def fixture_worker(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'admission')
    checks = fixtures()
    out = {'schema': 'MTA1-reporting-fixtures-v1', 'queue_unit_id': active['unit_id'], 'checks': checks,
           'status': 'PASS' if all(checks.values()) else 'FAIL', 'scientific_arrays_opened': False, 'INNER_truth_opened_or_hashed': False}
    with Path(cfg['output']).open('x') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    if out['status'] != 'PASS':
        raise AssertionError('reporting fixtures failed')
    return out


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'attacks', allow_inner_truth=True)
    rp = verify(cfg['reporting_lock'], cfg['reporting_lock_sha256']); _committed(cfg, rp, cfg['reporting_lock_sha256'])
    lock = json.loads(rp.read_text())
    sp = verify(cfg['selection_lock'], cfg['selection_lock_sha256']); _committed(cfg, sp, cfg['selection_lock_sha256'])
    sel = json.loads(sp.read_text())
    if lock.get('status') != 'ALL_READERS_FROZEN' or lock.get('selection_lock_sha256') != cfg['selection_lock_sha256']:
        raise PermissionError('reporting lock does not bind the selection lock')
    import numpy as np
    from compressed_income_safeguard.reporting import weighted_auc_draws, bootstrap_weights, task_vectors
    from .reader_worker import slate
    names = slate()
    # Verify every frozen reader receipt and file named by the reporting lock before any truth read.
    banks = {}
    for key, b in cfg['banks'].items():
        rec = json.loads(verify(b['receipt']['path'], b['receipt']['sha256']).read_text())
        if lock['reader_receipts'].get(key) != b['receipt']['sha256'] or rec.get('status') != 'FROZEN_COMPLETE':
            raise PermissionError('reader receipt differs from REPORTING_LOCK')
        verify(b['release']['path'], rec['inner_release_sha256']); verify(b['scores']['path'], rec['inner_scores_sha256'])
        with np.load(b['scores']['path'], allow_pickle=False) as z:
            sc = {k: z[k].copy() for k in z.files}
        with np.load(b['release']['path'], allow_pickle=False) as z:
            relz = {k: z[k].copy() for k in z.files}
        banks[key] = {'receipt': rec, 'scores': sc, 'release': relz}
    # INNER truth: exact committed public hashes.
    truth = {}
    for name in ('income', 'occupation', 'sex'):
        item = cfg['inner_truth'][name]
        if lock['INNER_label_sha256'][name] != item['sha256']:
            raise PermissionError('INNER label hash differs from committed lock')
        with np.load(verify(item['path'], item['sha256']), allow_pickle=False) as z:
            truth[name] = {k: z[k].copy() for k in z.files}
    ids = {'row_id': truth['sex']['row_id'], 'unit': truth['sex']['unit']}
    for name in truth:
        if not (np.array_equal(truth[name]['row_id'], ids['row_id']) and np.array_equal(truth[name]['unit'], ids['unit'])):
            raise PermissionError('INNER truth pairing differs')
    y = {k: truth[k][k].astype(np.int64) for k in truth}
    refs = {}
    for seed in (0, 1):
        with np.load(verify(cfg['inner_references'][str(seed)]['path'], cfg['inner_references'][str(seed)]['sha256']), allow_pickle=False) as z:
            refs[seed] = {k: z[k].copy() for k in z.files}
        if not np.array_equal(refs[seed]['row_id'], ids['row_id']):
            raise PermissionError('reference pairing differs')
    for b in banks.values():
        if not (np.array_equal(b['scores']['row_id'], ids['row_id']) and np.array_equal(b['release']['row_id'], ids['row_id'])):
            raise PermissionError('bank person order differs from INNER truth')
    rows = len(ids['row_id']); groups = len(np.unique(ids['unit']))
    if rows != 2235 or groups != 2234:
        raise PermissionError('INNER rows/groups differ')
    weights = np.vstack((np.ones((1, rows)), bootstrap_weights(ids['unit'], B=REPLICATES, seed=BOOTSTRAP_SEED)))
    total = weights.sum(1)
    cache = {}

    def curve(scores):
        k = hashlib.sha256(np.ascontiguousarray(scores, dtype=np.float64).tobytes()).hexdigest()
        if k not in cache:
            cache[k] = weighted_auc_draws(y['sex'], scores, weights)
        return cache[k]

    def service_scores(label, seed):
        r1 = banks[f'R1__S{seed}']['scores']; sv = banks[f'{label}__S{seed}']['scores']
        out = {k: r1[k] for k in r1 if k.startswith('r1__')}
        out.update({k: sv[k] for k in sv if k.startswith(('r2__', 'pair__'))})
        for n in names:
            out[f'pair__IGNORE_r2__{n}'] = out[f'r1__{n}']; out[f'pair__IGNORE_r1__{n}'] = out[f'r2__{n}']
        return out

    envelopes = {}
    def envelope(label, seed, view):
        key = (label, seed, view)
        if key not in envelopes:
            sc = service_scores(label, seed)
            envelopes[key] = np.maximum.reduce([curve(sc[k]) for k in envelope_keys(view, names)])
        return envelopes[key]

    # Control gates (scoring validity) per seed and view.
    from task_conditioned_representation.attacks import group_shuffle, INNER_NULL_SEED
    null_truth = group_shuffle(y['sex'], ids['unit'], INNER_NULL_SEED)
    gates = {}
    for seed in (0, 1):
        c = banks[f'CONTROLS__S{seed}']['scores']
        for view in ('r1', 'r2', 'pair'):
            sh = [weighted_auc_draws(null_truth, c[f'{view}__SHUFFLED__{n}'], weights[:1])[0] for n in names]
            if view == 'pair':
                sh += [weighted_auc_draws(null_truth, c[f'{v}__SHUFFLED__{n}'], weights[:1])[0] for v in ('r1', 'r2') for n in names]
            planted = [weighted_auc_draws(y['sex'], np.where(y['sex'] == 1, c[f'{view}__PLANTED_1__{n}'], c[f'{view}__PLANTED_0__{n}']), weights[:1])[0] for n in names]
            values = {'null_auc': float(weighted_auc_draws(y['sex'], np.full(rows, .5), weights[:1])[0]),
                      'shuffled_envelope_max_two_sided': float(max(max(a, 1 - a) for a in sh)),
                      'planted_envelope': float(max(planted)), 'planted_weakest_reader': float(min(planted))}
            g = {'null_exact': values['null_auc'] == .5, 'shuffle_at_most_0.60': values['shuffled_envelope_max_two_sided'] <= .60,
                 'planted_at_least_0.99': values['planted_envelope'] >= .99}
            gates[f'S{seed}__{view}'] = {'values': values, 'gates': g, 'passed': all(g.values())}
    gates_ok = all(v['passed'] for v in gates.values())
    # Utility vectors on INNER from the actual released probabilities.
    def utility(label, seed):
        rel = banks[f'{label}__S{seed}']['release']; out = {}
        for i, task in enumerate(TASKS):
            k = 2 if i == 0 else 6; order = np.arange(k)
            wire = rel[f'r{i + 1}']; p = wire[:, 16 + k:16 + 2 * k]
            if not np.array_equal(wire[:, -1], order[p.argmax(1)]):
                raise PermissionError('released decision differs from own argmax')
            own = task_vectors(y[task], p, order)
            for ref in ('U0', 'Ucal'):
                rv = task_vectors(y[task], refs[seed][f'{ref}_{i + 1}'], order)
                out[(task, ref)] = {'accuracy_loss': rv['accuracy'] - own['accuracy'], 'nll_excess': own['nll'] - rv['nll'],
                                    'brier_excess': own['brier'] - rv['brier'], 'own': own}
        return out

    services = sel['attack_roster']['services']
    service_table = []
    for label in services:
        for seed in (0, 1):
            u = utility(label, seed)
            row = {'service': label, 'seed': seed, 'status': sel['attack_roster']['status'][label]}
            for (task, ref), d in u.items():
                for metric in CAPS:
                    row[f'{task}__{ref}__{metric}'] = float(d[metric].mean())
                    row[f'{task}__{ref}__{metric}__point_pass'] = bool(d[metric].mean() <= CAPS[metric])
                row[f'{task}__accuracy'] = float(d['own']['accuracy'].mean()); row[f'{task}__nll'] = float(d['own']['nll'].mean())
                row[f'{task}__brier'] = float(d['own']['brier'].mean())
            for view in ('r1', 'r2', 'pair'):
                row[f'{view}__complete_auc'] = float(envelope(label, seed, view)[0])
            service_table.append(row)
    # Primary family.
    prim = sel['primary_family']; K = prim['K']; crit = critical(K, groups) if K else None
    primary_rows = []
    if K:
        cand = prim['candidate']
        for seed in (0, 1):
            u = utility(cand, seed)
            for task in TASKS:
                for ref in ('U0', 'Ucal'):
                    for metric, cap in CAPS.items():
                        vals = (weights @ u[(task, ref)][metric]) / total
                        point = float(vals[0]); se = float(vals[1:].std(ddof=1))
                        lo, hi = point - crit * se, point + crit * se
                        primary_rows.append({'endpoint_id': f'utility__{cand}__S{seed}__{task}__{ref}__{metric}', 'kind': 'utility',
                                             'point': point, 'se': se, 'lower': lo, 'upper': hi, 'threshold': cap, 'direction': 'at_most',
                                             'status': classify(point, lo, hi, cap, 'at_most'), 'gap_to_threshold': point - cap, 'K': K, 'critical': crit})
        for control in prim['roster']:
            for view, (kind, thr) in PRIVACY.items():
                eid = f'privacy__{cand}__vs__{control}__{view}_{kind}'
                if not gates_ok:
                    primary_rows.append({'endpoint_id': eid, 'kind': 'privacy', 'status': 'NOT_ESTIMABLE', 'reason': 'control gate failure', 'K': K})
                    continue
                parts = [(envelope(control, s, view) - envelope(cand, s, view)) if kind == 'benefit' else (envelope(cand, s, view) - envelope(control, s, view)) for s in (0, 1)]
                vals = np.mean(parts, axis=0); point = float(vals[0]); se = float(vals[1:].std(ddof=1)); lo, hi = point - crit * se, point + crit * se
                direction = 'greater_than' if kind == 'benefit' else 'at_most'
                primary_rows.append({'endpoint_id': eid, 'kind': 'privacy', 'view': view, 'control': control, 'point': point, 'se': se,
                                     'lower': lo, 'upper': hi, 'threshold': thr, 'direction': direction,
                                     'status': classify(point, lo, hi, thr, direction),
                                     'gap_to_threshold': (thr - point) if kind == 'benefit' else (point - thr), 'K': K, 'critical': crit})
        if len(primary_rows) != K:
            raise AssertionError('primary family size differs from locked K')
    # Diagnostic family.
    diag = sel['diagnostic_family']; Kd = diag['K']; critd = critical(Kd, groups)
    diag_rows = []
    for c in diag['contrasts']:
        for view in ('r1', 'r2', 'pair'):
            parts = [envelope(c['control'], s, view) - envelope(c['candidate'], s, view) for s in (0, 1)]
            vals = np.mean(parts, axis=0); point = float(vals[0]); se = float(vals[1:].std(ddof=1)); lo, hi = point - critd * se, point + critd * se
            per_seed = [float(p[0]) for p in parts]
            diag_rows.append({'endpoint_id': f'diagnostic__{c["id"]}__{view}', 'contrast': c['id'], 'question': c['question'],
                              'candidate': c['candidate'], 'control': c['control'], 'view': view,
                              'recovery_reduction_point': point, 'per_seed_reduction': per_seed, 'se': se, 'lower': lo, 'upper': hi,
                              'status': diagnostic_classify(lo, hi) if gates_ok else 'NOT_ESTIMABLE', 'K': Kd, 'critical': critd,
                              'family': 'historical/diagnostic (separate from primary)'})
    if len(diag_rows) != Kd:
        raise AssertionError('diagnostic family size differs from locked K')
    # Ordinary descriptive (unadjusted 95%) per-seed intervals for each service AUC.
    descriptive = []
    for label in services:
        for seed in (0, 1):
            for view in ('r1', 'r2', 'pair'):
                e = envelope(label, seed, view)
                sd = float(e[1:].std(ddof=1))
                descriptive.append({'service': label, 'seed': seed, 'view': view, 'auc': float(e[0]), 'se': sd,
                                    'lower95': float(e[0] - 1.959964 * sd), 'upper95': float(e[0] + 1.959964 * sd)})
    out = Path(cfg['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private report directory required')
    out.mkdir(parents=True, mode=0o700)
    result = {'schema': 'MTA1-inner-report-v1', 'queue_unit_id': active['unit_id'], 'reporting_lock_sha256': cfg['reporting_lock_sha256'],
              'selection_lock_sha256': cfg['selection_lock_sha256'], 'rows': rows, 'groups': groups,
              'bootstrap': {'seed': BOOTSTRAP_SEED, 'replicates': REPLICATES, 'alpha': ALPHA,
                            'construction': 'point +/- t(1-alpha/(2K),G-1)*SD(ddof=1); shared group multiplicities for all services and both seeds'},
              'control_gates': gates, 'control_gates_all_passed': gates_ok, 'service_table': service_table,
              'primary': {'K': K, 'candidate': prim.get('candidate'), 'roster': prim['roster'], 'endpoints': primary_rows,
                          'status_counts': {s: sum(r['status'] == s for r in primary_rows) for s in ('PASS', 'ESTABLISHED_VIOLATION', 'UNRESOLVED', 'NOT_ESTIMABLE')}},
              'diagnostic': {'K': Kd, 'endpoints': diag_rows}, 'descriptive_per_seed_auc': descriptive,
              'cached_reader_curves': len(cache), 'INNER_opened_once': True,
              'precision_note': 'conditional on fitted models and readers; both model seeds reuse the same people; exposed development history outside the intervals'}
    with (out / 'INNER_REPORT.json').open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, default=float)
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, default=float)
    return {'status': 'REPORTED', 'primary_K': K, 'diagnostic_K': Kd}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    parser.add_argument('--phase', choices=('report', 'fixture'), default='report')
    a = parser.parse_args()
    print(json.dumps(fixture_worker(a.spec) if a.phase == 'fixture' else run(a.spec), default=float))
