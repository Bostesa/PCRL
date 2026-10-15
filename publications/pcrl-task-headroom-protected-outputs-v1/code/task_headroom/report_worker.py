"""The single INNER report, after the committed REPORTING_LOCK (all readers frozen).

Implements SCIENCE_LOCK endpoints exactly: primary reading = INNER AUC of the
ATT-validation winner per service/seed/view; paired duplicate-group bootstrap
(2048 draws, seed 2026101501) shared across services and both seeds;
Student-t Bonferroni over the K frozen at SELECTION_LOCK; status rules
PASS / UNRESOLVED / ESTABLISHED_VIOLATION and ESTABLISHED / UNRESOLVED /
ESTABLISHED_SHORTFALL. Diagnostics use unadjusted 95% percentile intervals.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

TASKS = ('income', 'occupation')
VIEWS = ('income', 'occupation', 'pair')
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}


def run(spec_path):
    from .evaluation_queue import queue_config
    from .data import load_npz, task_vectors, summary, K
    from .inference import group_weights, weighted_mean, critical, interval, percentile, status_upper, status_benefit
    from task_conditioned_representation.inference import auc, bootstrap_auc
    active, cfg, verify, guard = queue_config(spec_path, 'verification', allow_inner_truth=True)
    rl = json.loads(verify(cfg['reporting_lock'], cfg['reporting_lock_sha256']).read_text())
    sel = json.loads(verify(cfg['selection_lock'], cfg['selection_lock_sha256']).read_text())
    if rl.get('status') != 'ALL_READERS_FROZEN':
        raise PermissionError('reporting lock not frozen')
    truth = {k: load_npz(cfg['inner_truth'][k], verify, ('row_id', 'unit', k)) for k in ('income', 'occupation', 'sex')}
    row_id, unit = truth['sex']['row_id'], truth['sex']['unit']
    y = {k: truth[k][k].astype(np.int64) for k in truth}
    for k in truth:
        if not np.array_equal(truth[k]['row_id'], row_id):
            raise PermissionError('INNER truth pairing differs')
    W, G = group_weights(unit)
    refs = {}
    for s in (0, 1):
        r = load_npz(cfg['inner_references'][str(s)], verify, ('row_id', 'unit', 'U0_1', 'U0_2', 'Ucal_1', 'Ucal_2', 'd_1', 'd_2'))
        if not np.array_equal(r['row_id'], row_id):
            raise PermissionError('reference pairing differs')
        refs[s] = {'U0': [r['U0_1'], r['U0_2']], 'Ucal': [r['Ucal_1'], r['Ucal_2']]}
    receipts, scores, releases = {}, {}, {}
    for s in (0, 1):
        b = cfg['banks'][str(s)]
        receipts[s] = json.loads(verify(b['receipt']['path'], b['receipt']['sha256']).read_text())
        with np.load(verify(b['scores']['path'], b['scores']['sha256'])) as z:
            if not np.array_equal(z['row_id'], row_id):
                raise PermissionError('score pairing differs')
            scores[s] = {k: z[k].copy() for k in z.files if k not in ('row_id', 'unit')}
        releases[s] = {}
        for label, item in b['releases'].items():
            with np.load(verify(item['path'], item['sha256'])) as z:
                releases[s][label] = [z['p1'].copy(), z['p2'].copy()]
    labels = list(cfg['service_labels'])
    # Recovery: winner AUC point + draws per service/seed/view.
    rec = {}
    for s in (0, 1):
        for lab in labels:
            for v in VIEWS:
                guard()
                win = receipts[s]['services'][lab]['winner'][v]['reader']
                sc = scores[s][f'{lab}__{v}__{win}']
                env = max(auc(y['sex'], scores[s][k]) for k in scores[s] if k.startswith(f'{lab}__{v}__'))
                rec[(lab, s, v)] = {'reader': win, 'point': auc(y['sex'], sc), 'draws': bootstrap_auc(y['sex'], sc, W), 'envelope': env}
    # Utility vectors.
    vec = {}
    for s in (0, 1):
        for name in labels + ['U0', 'Ucal']:
            probs = refs[s][name] if name in ('U0', 'Ucal') else releases[s][name]
            vec[(name, s)] = [task_vectors(y[t], probs[i], K[t]) for i, t in enumerate(TASKS)]

    def clause_rows(name, crit):
        out = []
        for s in (0, 1):
            for i, t in enumerate(TASKS):
                for ref in ('U0', 'Ucal'):
                    a, r = vec[(name, s)][i], vec[(ref, s)][i]
                    for metric, d in (('accuracy_loss', r['accuracy'] - a['accuracy']), ('nll_excess', a['nll'] - r['nll']), ('brier_excess', a['brier'] - r['brier'])):
                        iv = interval(d.mean(), weighted_mean(d, W), crit)
                        out.append({'endpoint': f'A|{name}|s{s}|{t}|{ref}|{metric}', 'service': name, 'seed': s, 'task': t, 'reference': ref, 'metric': metric,
                                    **iv, 'cap': CAPS[metric], 'point_pass': bool(iv['point'] <= CAPS[metric]), 'status': status_upper(iv, CAPS[metric])})
        return out

    def recovery_diff(a, b, view):
        """mean over seeds of AUC(a) - AUC(b)."""
        pt = np.mean([rec[(a, s, view)]['point'] - rec[(b, s, view)]['point'] for s in (0, 1)])
        dr = np.mean([rec[(a, s, view)]['draws'] - rec[(b, s, view)]['draws'] for s in (0, 1)], axis=0)
        per = {str(s): rec[(a, s, view)]['point'] - rec[(b, s, view)]['point'] for s in (0, 1)}
        return pt, dr, per
    primary = sel['primary_service']
    Kf = int(sel['K_primary_family']); crit = critical(Kf, G)
    family = []
    alias = {lab: all(receipts[s]['services'][lab]['wire_sha256']['INNER'] == receipts[s]['services'][primary]['wire_sha256']['INNER'] for s in (0, 1))
             for lab in labels} if primary else {}

    def bc_rows(P, comps, margin, prefix):
        rows = []
        for c in comps:
            if alias.get(c) and P == primary:
                rows.append({'endpoint': f'{prefix}|{P}_vs_{c}|alias', 'status': 'EXCLUDED_EXACT_ALIAS'}); continue
            pt, dr, per = recovery_diff(c, P, 'pair'); iv = interval(pt, dr, crit)
            rows.append({'endpoint': f'{prefix}|{P}_vs_{c}|pair_benefit', 'comparator': c, **iv, 'per_seed': per, 'margin': margin,
                         'status': status_benefit(iv, margin), 'status_at_0.020': status_benefit(iv, 0.020)})
            for v in ('income', 'occupation'):
                pt, dr, per = recovery_diff(P, c, v); iv = interval(pt, dr, crit)
                rows.append({'endpoint': f'{prefix}|{P}_vs_{c}|{v}_excess', 'comparator': c, **iv, 'per_seed': per, 'cap': 0.010, 'status': status_upper(iv, 0.010)})
        return rows
    if primary:
        family += clause_rows(primary, crit)
        family += bc_rows(primary, sel['B_required'], 0.020, 'B')
    J = 'J_TREE'
    family += bc_rows(J, sel['C_roster'], 0.0, 'C')
    if primary != J:
        family += clause_rows(J, crit)
    counted = [r for r in family if r.get('status') != 'EXCLUDED_EXACT_ALIAS']
    # Diagnostics: every service vs RAW_PROJ (95% percentile), all-service utility at the family critical value.
    diag = []
    for lab in labels:
        if lab == 'RAW_PROJ':
            continue
        for v in VIEWS:
            pt, dr, per = recovery_diff('RAW_PROJ', lab, v)
            diag.append({'endpoint': f'D|RAW_PROJ_minus_{lab}|{v}', **percentile(pt, dr), 'per_seed': per})
    for lab in labels:
        if lab != primary:
            pt, dr, per = recovery_diff('T_TREE', lab, 'pair')
            diag.append({'endpoint': f'D|T_TREE_minus_{lab}|pair', **percentile(pt, dr), 'per_seed': per})
    utility_all = {lab: clause_rows(lab, crit) for lab in labels}
    # Controls.
    from task_conditioned_representation.attacks import group_shuffle, INNER_NULL_SEED
    shuffled = group_shuffle(y['sex'], unit, INNER_NULL_SEED); controls = {}
    for s in (0, 1):
        for v in VIEWS:
            sh = auc(shuffled, scores[s][f'CONTROL__{v}__SHUFFLED']); pl = auc(y['sex'], np.where(y['sex'] == 1, scores[s][f'CONTROL__{v}__PLANTED_1'], scores[s][f'CONTROL__{v}__PLANTED_0']))
            controls[f's{s}|{v}'] = {'shuffled_auc_abs': max(sh, 1 - sh), 'planted_auc': pl, 'constant_auc': auc(y['sex'], np.full(len(row_id), .5)),
                                     'pass': bool(max(sh, 1 - sh) <= 0.60 and pl >= 0.99)}
    # Labels.
    A = [r for r in family if r['endpoint'].startswith(f'A|{primary}|')] if primary else []
    Bp = [r for r in family if r['endpoint'].startswith('B|')]
    Cr = [r for r in family if r['endpoint'].startswith('C|') and r.get('status') != 'EXCLUDED_EXACT_ALIAS']
    if not primary:
        sci = 'NO_POINT_FEASIBLE_PROTECTED_NOMINEE'
    else:
        a_point = all(r['point_pass'] for r in A)
        b_ok = all((r['status'] == 'ESTABLISHED') if r['endpoint'].endswith('pair_benefit') else (r['status'] == 'PASS') for r in Bp if r.get('status') != 'EXCLUDED_EXACT_ALIAS')
        if sel.get('B_unavailable'):
            sci = 'COMPETITIVE_PROTECTION_INCOMPLETE'
        elif b_ok and a_point:
            sci = 'COMPETITIVE_PROTECTION_ESTABLISHED'
        elif a_point:
            sci = 'POINT_FEASIBLE_NO_COMPETITIVE_CRITERION'
        else:
            sci = 'NOT_POINT_FEASIBLE_ON_INNER_NO_COMPETITIVE_CRITERION'
    j_internal = sel['families']['J_TREE']['status'] == 'POINT_FEASIBLE_NOMINEE'
    c_ok = j_internal and all((r['status'] == 'ESTABLISHED') if r['endpoint'].endswith('pair_benefit') else (r['status'] == 'PASS') for r in Cr)
    table = []
    for lab in labels:
        for s in (0, 1):
            row = {'service': lab, 'seed': s, 'nominee_key': sel['attack_roster'][lab]}
            for v in VIEWS:
                row[f'{v}_auc'] = rec[(lab, s, v)]['point']; row[f'{v}_reader'] = rec[(lab, s, v)]['reader']; row[f'{v}_envelope'] = rec[(lab, s, v)]['envelope']
            for i, t in enumerate(TASKS):
                m = summary(y[t], releases[s][lab][i], K[t])
                row[f'{t}_accuracy'], row[f'{t}_nll'], row[f'{t}_brier'], row[f'{t}_balanced_accuracy'] = m['accuracy'], m['nll'], m['brier'], m['balanced_accuracy']
            us = [r for r in utility_all[lab] if r['seed'] == s]
            row['utility_points_pass_of_12'] = sum(r['point_pass'] for r in us); row['utility_simultaneous_pass_of_12'] = sum(r['status'] == 'PASS' for r in us)
            table.append(row)
    for name in ('U0', 'Ucal'):
        for s in (0, 1):
            row = {'service': name, 'seed': s, 'nominee_key': 'reference (not attacked)'}
            for i, t in enumerate(TASKS):
                m = summary(y[t], refs[s][name][i], K[t])
                row[f'{t}_accuracy'], row[f'{t}_nll'], row[f'{t}_brier'], row[f'{t}_balanced_accuracy'] = m['accuracy'], m['nll'], m['brier'], m['balanced_accuracy']
            table.append(row)
    result = {'schema': 'THP1-inner-report-v1', 'queue_unit_id': active['unit_id'], 'primary_service': primary,
              'scientific_label': sci, 'joint_label': 'JOINT_COMPONENT_ESTABLISHED' if c_ok else 'JOINT_COMPONENT_NOT_ESTABLISHED',
              'K': Kf, 'K_counted_rows': len(counted), 'critical_value': crit, 'groups': G, 'draws': int(W.shape[0]),
              'family': family, 'diagnostic': diag, 'utility_all_services': utility_all, 'controls': controls,
              'all_controls_pass': all(c['pass'] for c in controls.values()), 'service_table': table, 'aliases_of_primary': alias,
              'INNER_opened_once': True, 'interval_caveat': 'conditional on fitted models and readers; does not cover the repeated development history; two seeds share the same INNER people'}
    out = Path(cfg['private_output']); out.mkdir(parents=True, exist_ok=False, mode=0o700)
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, default=float)
    def write_csv(path, rows):
        keys = sorted({k for r in rows for k in r})
        with Path(path).open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()})
    write_csv(out / 'PRIMARY_ENDPOINTS.csv', family); write_csv(out / 'DIAGNOSTIC_ENDPOINTS.csv', diag); write_csv(out / 'INNER_SERVICE_TABLE.csv', table)
    write_csv(out / 'UTILITY_BOUNDS_ALL_SERVICES.csv', [r for rows in utility_all.values() for r in rows])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
