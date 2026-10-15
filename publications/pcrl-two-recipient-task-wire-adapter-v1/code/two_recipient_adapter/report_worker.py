"""INNER reporting: opened once after the committed REPORTING_LOCK.

Primary simultaneous family (fixed at SCIENCE_LOCK, roster status frozen at
SELECTION_LOCK): candidate J_BOTH's 24 original INNER utility clauses plus three
privacy endpoints per registered comparator (RAW_PROJ, T_BOTH, L_BOTH, S12,
S21), K = 39 always; an ineligible comparator keeps its visible slot with status
INCOMPLETE_COMPARATOR_INELIGIBLE. Separate diagnostic family (K = 45) for the
registered contrasts, including the J_BOTH wire-substitution hybrids.
Inherited paired duplicate-group bootstrap (2048 draws, fixed seed, shared row
multiplicities for every service and both source seeds); interval = point +/-
Student-t(1-alpha/(2K), G-1) * bootstrap SD (ddof=1). Statuses: PASS,
ESTABLISHED_VIOLATION, UNRESOLVED, NOT_ESTIMABLE, INCOMPLETE_*.
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
PRIMARY_CANDIDATE = 'J_BOTH'
PRIMARY_ROSTER = ('RAW_PROJ', 'T_BOTH', 'L_BOTH', 'S12', 'S21')
PRIMARY_K = 24 + 3 * len(PRIMARY_ROSTER)
DIAGNOSTIC_CONTRASTS = (
    ('D01', 'J_BOTH', 'RAW_PROJ', 'joint two-recipient adaptation vs immutable start'),
    ('D02', 'J_BOTH', 'T_BOTH', 'privacy objective beyond equally trained task-only two-recipient adaptation'),
    ('D03', 'J_BOTH', 'L_BOTH', 'coalition term beyond separate local objectives'),
    ('D04', 'J_BOTH', 'S12', 'simultaneous joint vs learned sequential income-then-occupation'),
    ('D05', 'J_BOTH', 'S21', 'simultaneous joint vs learned sequential occupation-then-income'),
    ('D06', 'J_BOTH', 'O_PREVIOUS', 'adapting both vs inherited occupation-only component (not equally trained)'),
    ('D07', 'J_BOTH', 'I_ONLY', 'adding occupation adaptation to income-only adaptation (retrained, not a substitution)'),
    ('D08', 'I_ONLY', 'RAW_PROJ', 'income-only adaptation vs start'),
    ('D09', 'L_BOTH', 'RAW_PROJ', 'local two-recipient adaptation vs start'),
    ('D10', 'T_BOTH', 'RAW_PROJ', 'task-only two-recipient adaptation vs start'),
    ('D11', 'HYB_JI_RO', 'RAW_PROJ', 'J_BOTH income wire substituted alone (fixed ablation)'),
    ('D12', 'HYB_RI_JO', 'RAW_PROJ', 'J_BOTH occupation wire substituted alone (fixed ablation)'),
    ('D13', 'J_BOTH', 'HYB_JI_RO', 'adding the changed occupation wire given the changed income wire'),
    ('D14', 'J_BOTH', 'HYB_RI_JO', 'adding the changed income wire given the changed occupation wire'),
    ('D15', 'O_PREVIOUS', 'RAW_PROJ', 'replication of the previous occupation-only component'),
)
DIAGNOSTIC_K = 3 * len(DIAGNOSTIC_CONTRASTS)


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


def scientific_label(*, nominee, comparators_eligible, gates_ok, primary_rows, movement_ok, verification_complete):
    """Registered label rule (SCIENCE_LOCK)."""
    if not gates_ok:
        return 'NOT_ESTIMABLE_CONTROL_GATE'
    if not nominee:
        return 'NO_POINT_FEASIBLE_NOMINEE'
    util = [r for r in primary_rows if r['kind'] == 'utility']
    priv = [r for r in primary_rows if r['kind'] == 'privacy']
    elig = [r for r in priv if r['control'] in comparators_eligible]
    competitive = (all(r['status'] == 'PASS' for r in util) and elig and all(r['status'] == 'PASS' for r in elig)
                   and movement_ok and verification_complete)
    if competitive and len(comparators_eligible) == len(PRIMARY_ROSTER):
        return 'TWO_RECIPIENT_COMPETITIVE_CRITERION_ESTABLISHED'
    if competitive:
        return 'COMPETITIVE_CLAIM_INCOMPLETE_COMPARATOR_INELIGIBLE'
    return 'POINT_FEASIBLE_NO_COMPETITIVE_CRITERION'


def fixtures():
    """Data-free validation of the bound arithmetic and label rule (run before INNER opens)."""
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
    checks['primary_K_39_and_diagnostic_K_45'] = PRIMARY_K == 39 and DIAGNOSTIC_K == 45
    checks['critical_value_bonferroni_t'] = abs(critical(39, 2234) - 3.2240705) < 1e-6 and abs(critical(45, 2234) - critical(45, 2234)) == 0 and critical(1, 2234) < critical(39, 2234) < critical(45, 2234)
    checks['classification'] = (classify(.005, 0., .010, .010, 'at_most') == 'PASS' and classify(.012, .011, .013, .010, 'at_most') == 'ESTABLISHED_VIOLATION'
                                and classify(.009, .007, .011, .010, 'at_most') == 'UNRESOLVED'
                                and classify(.03, .021, .04, .02, 'greater_than') == 'PASS' and classify(.03, .020, .04, .02, 'greater_than') == 'UNRESOLVED'
                                and classify(.005, -.01, .019, .02, 'greater_than') == 'ESTABLISHED_VIOLATION'
                                and diagnostic_classify(.001, .01) == 'REDUCTION_ESTABLISHED' and diagnostic_classify(-.01, -.001) == 'INCREASE_ESTABLISHED')
    keys = envelope_keys('pair', ['A', 'B'])
    checks['pair_envelope_includes_ignore_aliases'] = len(keys) == 6 and 'pair__IGNORE_r2__A' in keys and 'pair__IGNORE_r1__B' in keys
    ok = [{'kind': 'utility', 'status': 'PASS'}] * 24 + [{'kind': 'privacy', 'control': c, 'status': 'PASS'} for c in PRIMARY_ROSTER for _ in range(3)]
    full = list(PRIMARY_ROSTER)
    checks['label_rule'] = (scientific_label(nominee=True, comparators_eligible=full, gates_ok=True, primary_rows=ok, movement_ok=True, verification_complete=True) == 'TWO_RECIPIENT_COMPETITIVE_CRITERION_ESTABLISHED'
                            and scientific_label(nominee=True, comparators_eligible=full[:4], gates_ok=True, primary_rows=ok, movement_ok=True, verification_complete=True) == 'COMPETITIVE_CLAIM_INCOMPLETE_COMPARATOR_INELIGIBLE'
                            and scientific_label(nominee=True, comparators_eligible=full, gates_ok=True, primary_rows=ok, movement_ok=False, verification_complete=True) == 'POINT_FEASIBLE_NO_COMPETITIVE_CRITERION'
                            and scientific_label(nominee=True, comparators_eligible=full, gates_ok=True, primary_rows=ok[:-1] + [{'kind': 'privacy', 'control': 'S21', 'status': 'UNRESOLVED'}], movement_ok=True, verification_complete=True) == 'POINT_FEASIBLE_NO_COMPETITIVE_CRITERION'
                            and scientific_label(nominee=False, comparators_eligible=full, gates_ok=True, primary_rows=ok, movement_ok=True, verification_complete=True) == 'NO_POINT_FEASIBLE_NOMINEE'
                            and scientific_label(nominee=True, comparators_eligible=full, gates_ok=False, primary_rows=ok, movement_ok=True, verification_complete=True) == 'NOT_ESTIMABLE_CONTROL_GATE')
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
    out = {'schema': 'TRA1-reporting-fixtures-v1', 'queue_unit_id': active['unit_id'], 'checks': checks,
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
        sv = banks[f'{label}__S{seed}']['scores']
        out = {k: sv[k] for k in sv if k.startswith(('r1__', 'r2__', 'pair__'))}
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

    crit = critical(PRIMARY_K, groups)

    def utility_rows(label):
        out = []
        for seed in (0, 1):
            u = utility(label, seed)
            for task in TASKS:
                for ref in ('U0', 'Ucal'):
                    for metric, cap in CAPS.items():
                        vals = (weights @ u[(task, ref)][metric]) / total
                        point = float(vals[0]); se = float(vals[1:].std(ddof=1))
                        lo, hi = point - crit * se, point + crit * se
                        out.append({'endpoint_id': f'utility__{label}__S{seed}__{task}__{ref}__{metric}', 'kind': 'utility', 'service': label,
                                    'seed': seed, 'task': task, 'reference': ref, 'metric': metric,
                                    'point': point, 'se': se, 'lower': lo, 'upper': hi, 'threshold': cap, 'direction': 'at_most',
                                    'status': classify(point, lo, hi, cap, 'at_most'), 'gap_to_threshold': point - cap, 'K': PRIMARY_K, 'critical': crit})
        return out

    services = sel['attack_roster']['services']
    status = sel['attack_roster']['status']
    service_table = []
    for label in services:
        for seed in (0, 1):
            u = utility(label, seed)
            row = {'service': label, 'seed': seed, 'status': status[label]}
            for (task, ref), d in u.items():
                for metric in CAPS:
                    row[f'{task}__{ref}__{metric}'] = float(d[metric].mean())
                    row[f'{task}__{ref}__{metric}__point_pass'] = bool(d[metric].mean() <= CAPS[metric])
                row[f'{task}__accuracy'] = float(d['own']['accuracy'].mean()); row[f'{task}__nll'] = float(d['own']['nll'].mean())
                row[f'{task}__brier'] = float(d['own']['brier'].mean())
            for view in ('r1', 'r2', 'pair'):
                row[f'{view}__complete_auc'] = float(envelope(label, seed, view)[0])
            service_table.append(row)
    # Primary family (K = 39 always; ineligible comparators keep visible slots).
    prim = sel['primary_family']
    if prim['K'] != PRIMARY_K or prim['candidate'] != PRIMARY_CANDIDATE or tuple(prim['roster']) != PRIMARY_ROSTER:
        raise PermissionError('primary family differs from the registered roster')
    cand = PRIMARY_CANDIDATE
    primary_rows = utility_rows(cand)
    eligible = prim['comparators_point_eligible']
    for control in PRIMARY_ROSTER:
        for view, (kind, thr) in PRIVACY.items():
            eid = f'privacy__{cand}__vs__{control}__{view}_{kind}'
            if not gates_ok:
                primary_rows.append({'endpoint_id': eid, 'kind': 'privacy', 'view': view, 'control': control, 'status': 'NOT_ESTIMABLE', 'reason': 'control gate failure', 'K': PRIMARY_K})
                continue
            parts = [(envelope(control, s, view) - envelope(cand, s, view)) if kind == 'benefit' else (envelope(cand, s, view) - envelope(control, s, view)) for s in (0, 1)]
            vals = np.mean(parts, axis=0); point = float(vals[0]); se = float(vals[1:].std(ddof=1)); lo, hi = point - crit * se, point + crit * se
            direction = 'greater_than' if kind == 'benefit' else 'at_most'
            st = classify(point, lo, hi, thr, direction)
            row = {'endpoint_id': eid, 'kind': 'privacy', 'view': view, 'control': control, 'point': point, 'se': se,
                   'per_seed': [float(p[0]) for p in parts], 'lower': lo, 'upper': hi, 'threshold': thr, 'direction': direction,
                   'interval_status': st, 'status': st if control in eligible else 'INCOMPLETE_COMPARATOR_INELIGIBLE',
                   'comparator_point_eligible': control in eligible,
                   'gap_to_threshold': (thr - point) if kind == 'benefit' else (point - thr), 'K': PRIMARY_K, 'critical': crit}
            primary_rows.append(row)
    if len(primary_rows) != PRIMARY_K:
        raise AssertionError('primary family size differs from locked K')
    diag = sel['diagnostic_family']; Kd = diag['K']; critd = critical(Kd, groups)
    if Kd != DIAGNOSTIC_K or [c['id'] for c in diag['contrasts']] != [c[0] for c in DIAGNOSTIC_CONTRASTS]:
        raise PermissionError('diagnostic family differs from the registered contrasts')
    diag_rows = []
    for c in diag['contrasts']:
        for view in ('r1', 'r2', 'pair'):
            parts = [envelope(c['control'], s, view) - envelope(c['candidate'], s, view) for s in (0, 1)]
            vals = np.mean(parts, axis=0); point = float(vals[0]); se = float(vals[1:].std(ddof=1)); lo, hi = point - critd * se, point + critd * se
            diag_rows.append({'endpoint_id': f'diagnostic__{c["id"]}__{view}', 'contrast': c['id'], 'question': c['question'],
                              'candidate': c['candidate'], 'control': c['control'], 'view': view,
                              'candidate_status': status[c['candidate']], 'control_status': status[c['control']],
                              'recovery_reduction_point': point, 'per_seed_reduction': [float(p[0]) for p in parts], 'se': se, 'lower': lo, 'upper': hi,
                              'status': diagnostic_classify(lo, hi) if gates_ok else 'NOT_ESTIMABLE', 'K': Kd, 'critical': critd,
                              'family': 'diagnostic (separate from primary)'})
    if len(diag_rows) != Kd:
        raise AssertionError('diagnostic family size differs from locked K')
    # Descriptive complementarity (unadjusted 95%; not a family member): (RAW-J) - (RAW-HYB_JI) - (RAW-HYB_JO).
    interaction = []
    for view in ('r1', 'r2', 'pair'):
        parts = [(envelope('RAW_PROJ', s, view) - envelope(cand, s, view)) - (envelope('RAW_PROJ', s, view) - envelope('HYB_JI_RO', s, view))
                 - (envelope('RAW_PROJ', s, view) - envelope('HYB_RI_JO', s, view)) for s in (0, 1)]
        vals = np.mean(parts, axis=0); sd = float(vals[1:].std(ddof=1))
        interaction.append({'view': view, 'point': float(vals[0]), 'se': sd, 'lower95': float(vals[0] - 1.959964 * sd), 'upper95': float(vals[0] + 1.959964 * sd),
                            'meaning': 'positive = combined changed wires reduce recovery more than the sum of each substitution alone'})
    # Descriptive simultaneous-width utility bounds for every attacked service (same K=39 critical value; not a family).
    utility_all = {label: utility_rows(label) for label in services}
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
    result = {'schema': 'TRA1-inner-report-v1', 'queue_unit_id': active['unit_id'], 'reporting_lock_sha256': cfg['reporting_lock_sha256'],
              'selection_lock_sha256': cfg['selection_lock_sha256'], 'rows': rows, 'groups': groups,
              'bootstrap': {'seed': BOOTSTRAP_SEED, 'replicates': REPLICATES, 'alpha': ALPHA,
                            'construction': 'point +/- t(1-alpha/(2K),G-1)*SD(ddof=1); shared group multiplicities for all services and both seeds'},
              'control_gates': gates, 'control_gates_all_passed': gates_ok, 'service_table': service_table,
              'primary': {'K': PRIMARY_K, 'candidate': cand, 'candidate_status': status[cand], 'roster': list(PRIMARY_ROSTER),
                          'comparators_point_eligible': eligible, 'endpoints': primary_rows,
                          'status_counts': {s: sum(r['status'] == s for r in primary_rows) for s in
                                            ('PASS', 'ESTABLISHED_VIOLATION', 'UNRESOLVED', 'NOT_ESTIMABLE', 'INCOMPLETE_COMPARATOR_INELIGIBLE')}},
              'diagnostic': {'K': Kd, 'endpoints': diag_rows}, 'descriptive_interaction': interaction,
              'utility_bounds_all_services': utility_all, 'descriptive_per_seed_auc': descriptive,
              'cached_reader_curves': len(cache), 'INNER_opened_once': True,
              'precision_note': 'conditional on fitted models and readers; both model seeds reuse the same people; exposed development history outside the intervals'}
    with (out / 'INNER_REPORT.json').open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, default=float)
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, default=float)
    return {'status': 'REPORTED', 'primary_K': PRIMARY_K, 'diagnostic_K': Kd}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    parser.add_argument('--phase', choices=('report', 'fixture'), default='report')
    a = parser.parse_args()
    print(json.dumps(fixture_worker(a.spec) if a.phase == 'fixture' else run(a.spec), default=float))
