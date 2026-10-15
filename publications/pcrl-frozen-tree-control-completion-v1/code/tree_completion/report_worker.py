"""Completion report on INNER (already exposed; opened after the committed COMPLETION_REPORTING_LOCK).

Implements COMPLETION_LOCK endpoints exactly. Candidate L_TREE and the original comparators use their
STORED frozen winner scores and releases (original reader banks); the new controls use the completion
banks. Same INNER bootstrap resamples as the original report (task_headroom.inference.group_weights,
seed 2026101501, 2048 draws), mean over seeds, Student-t Bonferroni with K = 11.
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
    from task_headroom.data import load_npz, task_vectors, summary, K
    from task_headroom.inference import group_weights, weighted_mean, critical, interval, percentile, status_upper, status_benefit
    from task_conditioned_representation.inference import auc, bootstrap_auc
    from task_conditioned_representation.attacks import group_shuffle, INNER_NULL_SEED
    active, cfg, verify, guard = queue_config(spec_path, 'verification', allow_inner_truth=True)
    rl = json.loads(verify(cfg['reporting_lock'], cfg['reporting_lock_sha256']).read_text())
    sel = json.loads(verify(cfg['selection_lock'], cfg['selection_lock_sha256']).read_text())
    if rl.get('status') != 'ALL_READERS_FROZEN' or sel.get('status') != 'COMPLETION_ROSTER_FROZEN':
        raise PermissionError('completion reporting lock not frozen')
    truth = {k: load_npz(cfg['inner_truth'][k], verify, ('row_id', 'unit', k)) for k in ('income', 'occupation', 'sex')}
    row_id, unit = truth['sex']['row_id'], truth['sex']['unit']
    y = {k: truth[k][k].astype(np.int64) for k in truth}
    W, G = group_weights(unit)
    refs = {}
    for s in (0, 1):
        r = load_npz(cfg['inner_references'][str(s)], verify, ('row_id', 'unit', 'U0_1', 'U0_2', 'Ucal_1', 'Ucal_2', 'd_1', 'd_2'))
        refs[s] = {'U0': [r['U0_1'], r['U0_2']], 'Ucal': [r['Ucal_1'], r['Ucal_2']]}
    # Banks: 'old' (frozen original) and 'new' (completion).
    rec, rel, banks = {}, {}, {}
    for bank in ('old', 'new'):
        for s in (0, 1):
            b = cfg['banks'][bank][str(s)]
            receipt = json.loads(verify(b['receipt']['path'], b['receipt']['sha256']).read_text())
            with np.load(verify(b['scores']['path'], b['scores']['sha256'])) as z:
                if not np.array_equal(z['row_id'], row_id):
                    raise PermissionError('score pairing differs')
                scores = {k: z[k].copy() for k in z.files if k not in ('row_id', 'unit')}
            banks[(bank, s)] = (receipt, scores)
    services = cfg['services']   # label -> {'bank': 'old'|'new', 'bank_label': label in that bank}
    for lab, where in services.items():
        for s in (0, 1):
            receipt, scores = banks[(where['bank'], s)]
            bl = where['bank_label']
            item = cfg['releases'][lab][str(s)]
            with np.load(verify(item['path'], item['sha256'])) as z:
                rel[(lab, s)] = [z['p1'].copy(), z['p2'].copy()]
            for v in VIEWS:
                win = receipt['services'][bl]['winner'][v]['reader']
                sc = scores[f'{bl}__{v}__{win}']
                env = max(auc(y['sex'], scores[k]) for k in scores if k.startswith(f'{bl}__{v}__'))
                rec[(lab, s, v)] = {'reader': win, 'point': auc(y['sex'], sc), 'draws': bootstrap_auc(y['sex'], sc, W), 'envelope': env}
    # Replay control: T_TREE re-attacked in the new banks must reproduce the stored scores bitwise.
    replay = {}
    for s in (0, 1):
        ro, so = banks[('old', s)]; rn, sn = banks[('new', s)]
        keys = [k for k in so if k.startswith('T_TREE__')]
        replay[str(s)] = {'keys': len(keys), 'all_bitwise': all(np.array_equal(so[k], sn[k]) for k in keys),
                          'winners_equal': ro['services']['T_TREE']['winner'] == rn['services']['T_TREE']['winner']}
    vec = {}
    for s in (0, 1):
        for name in list(services) + ['U0', 'Ucal']:
            probs = refs[s][name] if name in ('U0', 'Ucal') else rel[(name, s)]
            vec[(name, s)] = [task_vectors(y[t], probs[i], K[t]) for i, t in enumerate(TASKS)]
    Kc = int(cfg['K']); crit = critical(Kc, G)

    def clause_rows(name):
        out = []
        for s in (0, 1):
            for i, t in enumerate(TASKS):
                for ref in ('U0', 'Ucal'):
                    a, r = vec[(name, s)][i], vec[(ref, s)][i]
                    for metric, d in (('accuracy_loss', r['accuracy'] - a['accuracy']), ('nll_excess', a['nll'] - r['nll']), ('brier_excess', a['brier'] - r['brier'])):
                        iv = interval(d.mean(), weighted_mean(d, W), crit)
                        out.append({'endpoint': f'U|{name}|s{s}|{t}|{ref}|{metric}', 'service': name, 'seed': s, 'task': t, 'reference': ref, 'metric': metric,
                                    **iv, 'cap': CAPS[metric], 'point_pass': bool(iv['point'] <= CAPS[metric]), 'status': status_upper(iv, CAPS[metric])})
        return out

    def diff(a, b, view):
        pt = np.mean([rec[(a, s, view)]['point'] - rec[(b, s, view)]['point'] for s in (0, 1)])
        dr = np.mean([rec[(a, s, view)]['draws'] - rec[(b, s, view)]['draws'] for s in (0, 1)], axis=0)
        return pt, dr, {str(s): rec[(a, s, view)]['point'] - rec[(b, s, view)]['point'] for s in (0, 1)}

    def comparison(eid, cand, comp, margin, with_excess=True):
        rows = []
        pt, dr, per = diff(comp, cand, 'pair'); iv = interval(pt, dr, crit)
        rows.append({'endpoint': f'{eid}|{cand}_vs_{comp}|pair_benefit', 'comparator': comp, **iv, 'per_seed': per, 'margin': margin,
                     'status': status_benefit(iv, margin), 'status_at_0': status_benefit(iv, 0.0), 'status_at_0.020': status_benefit(iv, 0.020)})
        if with_excess:
            for v in ('income', 'occupation'):
                pt, dr, per = diff(cand, comp, v); iv = interval(pt, dr, crit)
                rows.append({'endpoint': f'{eid}|{cand}_vs_{comp}|{v}_excess', 'comparator': comp, **iv, 'per_seed': per, 'cap': 0.010, 'status': status_upper(iv, 0.010)})
        return rows
    req = sel['required_status']
    family = []
    family += comparison('C1', 'L_TREE', 'T_REFIT_E20', 0.0)
    family += comparison('C2', 'L_TREE', 'RAW_J_NEQ', 0.020 if req['RAW_J_NEQ']['required_comparator_for_L_TREE'] else 0.0)
    family += comparison('C3', 'RAW_J_NEQ', 'T_NEURAL', 0.0, with_excess=False)
    family += comparison('C4', 'L_TREE', 'FARE_TC', 0.020 if req['FARE_TC']['required_comparator_for_L_TREE'] else 0.0)
    family += comparison('C5', 'FARE_TC', 'FARE_TC_Z', 0.0, with_excess=False)
    if len(family) != Kc:
        raise AssertionError('family size differs from the frozen K')
    diag = []
    for a, b in (('T_TREE', 'L_TREE'), ('T_REFIT_E5', 'L_TREE'), ('T_TREE', 'T_REFIT_E20'), ('T_REFIT_E5', 'T_REFIT_E20'),
                 ('RAW_PROJ', 'RAW_J_NEQ'), ('RAW_PROJ', 'FARE_TC'), ('H_SEQ21_CODE', 'RAW_J_NEQ'), ('H_JOINT_CODE', 'RAW_J_NEQ'),
                 ('N_EQ', 'RAW_J_NEQ'), ('N_EQ', 'T_NEURAL'), ('RAW_PROJ', 'L_TREE'), ('H_SEQ21_CODE', 'L_TREE'), ('H_JOINT_CODE', 'L_TREE'), ('H_DPC_JOINT', 'L_TREE')):
        for v in VIEWS:
            pt, dr, per = diff(a, b, v)
            diag.append({'endpoint': f'D|{a}_minus_{b}|{v}', **percentile(pt, dr), 'per_seed': per})
    utility = {lab: clause_rows(lab) for lab in services}
    sh = group_shuffle(y['sex'], unit, INNER_NULL_SEED); controls = {}
    for s in (0, 1):
        _, sn = banks[('new', s)]
        for v in VIEWS:
            a = auc(sh, sn[f'CONTROL__{v}__SHUFFLED']); pl = auc(y['sex'], np.where(y['sex'] == 1, sn[f'CONTROL__{v}__PLANTED_1'], sn[f'CONTROL__{v}__PLANTED_0']))
            controls[f's{s}|{v}'] = {'shuffled_auc_abs': max(a, 1 - a), 'planted_auc': pl, 'pass': bool(max(a, 1 - a) <= 0.60 and pl >= 0.99)}
    labels = {}
    by = lambda pre: [r for r in family if r['endpoint'].startswith(pre)]
    c1 = by('C1|')[0]; labels['C1'] = 'PRIVACY_TRAINING_EFFECT_ESTABLISHED' if c1['status_at_0'] == 'ESTABLISHED' else 'PRIVACY_TRAINING_EFFECT_NOT_ESTABLISHED'

    def margin_label(rows, required):
        b = rows[0]; ex = rows[1:]
        if b['upper'] < 0:
            return 'CONTROL_LOWER_RECOVERY'
        if required and b['status_at_0.020'] == 'ESTABLISHED' and all(e['status'] == 'PASS' for e in ex):
            return 'MARGIN_ESTABLISHED'
        if b['status_at_0'] == 'ESTABLISHED':
            return 'SUPERIORITY_ONLY' if required else 'SUPERIORITY_DESCRIPTIVE_COMPARATOR'
        return 'NOT_DISTINGUISHABLE'
    labels['C2'] = margin_label(by('C2|'), req['RAW_J_NEQ']['required_comparator_for_L_TREE'])
    labels['C3'] = 'NEURAL_PRIVACY_EFFECT_ESTABLISHED' if by('C3|')[0]['status_at_0'] == 'ESTABLISHED' else 'NEURAL_PRIVACY_EFFECT_NOT_ESTABLISHED'
    labels['C4'] = margin_label(by('C4|'), True) if req['FARE_TC']['required_comparator_for_L_TREE'] else 'FARE_TC_DESCRIPTIVE_ONLY (not internally point-feasible): ' + margin_label(by('C4|'), False)
    labels['C5'] = 'FARE_FAIRNESS_EFFECT_ESTABLISHED' if by('C5|')[0]['status_at_0'] == 'ESTABLISHED' else 'FARE_FAIRNESS_EFFECT_NOT_ESTABLISHED'
    for pre, n in (('C1|', 3), ('C2|', 3), ('C3|', 1), ('C4|', 3), ('C5|', 1)):
        if len(by(pre)) != n or not by(pre)[0]['endpoint'].endswith('pair_benefit'):
            raise AssertionError('endpoint grouping differs')
    table = []
    for lab in services:
        for s in (0, 1):
            row = {'service': lab, 'seed': s, 'key': cfg['service_keys'][lab]}
            for v in VIEWS:
                row[f'{v}_auc'] = rec[(lab, s, v)]['point']; row[f'{v}_reader'] = rec[(lab, s, v)]['reader']; row[f'{v}_envelope'] = rec[(lab, s, v)]['envelope']
            for i, t in enumerate(TASKS):
                m = summary(y[t], rel[(lab, s)][i], K[t])
                for kk in ('accuracy', 'nll', 'brier', 'balanced_accuracy'):
                    row[f'{t}_{kk}'] = m[kk]
            us = [r for r in utility[lab] if r['seed'] == s]
            row['utility_points_pass_of_12'] = sum(r['point_pass'] for r in us); row['utility_simultaneous_pass_of_12'] = sum(r['status'] == 'PASS' for r in us)
            table.append(row)
    result = {'schema': 'TC1-completion-report-v1', 'queue_unit_id': active['unit_id'], 'K': Kc, 'critical_value': crit, 'groups': G, 'draws': int(W.shape[0]),
              'family': family, 'labels': labels, 'diagnostic': diag, 'utility_new_services': {k: v for k, v in utility.items() if services[k]['bank'] == 'new'},
              'utility_all_services': utility, 'controls_new_banks': controls, 'all_controls_pass': all(c['pass'] for c in controls.values()),
              'replay_T_TREE': replay, 'service_table': table, 'required_status': req,
              'interval_caveat': 'exploratory; INNER is an exposed development role; conditional on fitted models and readers; seeds share INNER people'}
    out = Path(cfg['private_output']); out.mkdir(parents=True, exist_ok=False, mode=0o700)
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, default=float)

    def write_csv(path, rows):
        keys = sorted({k for r in rows for k in r})
        with Path(path).open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()})
    write_csv(out / 'CONTROL_COMPLETION_RESULTS.csv', family + diag)
    write_csv(out / 'COMPLETION_SERVICE_TABLE.csv', table)
    write_csv(out / 'COMPLETION_UTILITY_BOUNDS.csv', [r for rows in utility.values() for r in rows])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
