"""Stage A report (both seeds): task-only recipe selection, internal prerequisite and headroom on FIT2.

Selection (frozen in SCIENCE_LOCK): per recipient, the registry depth with the
lowest internal-fold NLL summed over both seeds (tie: depth 4); shared across
seeds. Task performance only; FIT2 does not select the recipe.
Internal prerequisite: the selected teacher meets all 12 point clauses
(2 tasks x 3 metrics x 2 seeds) against the equally exposed neural reference
N_EQ on FIT2. Headroom: N_EQ minus teacher, paired group-bootstrap 95%
percentile intervals over FIT2 groups (diagnostic, unadjusted).
U0/Ucal are in-sample on FIT2 (trained on all FIT) and are reported only as a
labelled diagnostic, never as the internal reference.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def run(spec_path):
    from .evaluation_queue import queue_config
    from .data import load_role, clauses, summary, task_vectors, K
    from .inference import group_weights, weighted_mean, percentile
    from . import trees
    active, cfg, verify, guard = queue_config(spec_path, 'witness')
    F = load_role('FIT2', cfg['roles']['FIT2'], verify, ('features', 'income', 'occupation', 'references_s0', 'references_s1'))
    labels = {'income': F['income'], 'occupation': F['occupation']}
    seeds = {}
    for s in (0, 1):
        rec = json.loads(verify(cfg['stage_a'][str(s)]['record']['path'], cfg['stage_a'][str(s)]['record']['sha256']).read_text())
        out = {}
        with np.load(verify(cfg['stage_a'][str(s)]['neq']['path'], cfg['stage_a'][str(s)]['neq']['sha256'])) as z:
            out['N_EQ'] = [z['FIT2__income'], z['FIT2__occupation']]
        for task in ('income', 'occupation'):
            for d in trees.REGISTRY['depth']:
                item = cfg['stage_a'][str(s)]['trees'][f'{task}_d{d}']
                with np.load(verify(item['path'], item['sha256'])) as z:
                    out[f'{task}_d{d}'] = z['FIT2']
        seeds[s] = {'record': rec, 'probs': out}
    selection = {}
    for task in ('income', 'occupation'):
        totals = {d: sum(seeds[s]['record']['trees'][f'{task}_d{d}']['internal_best_nll'] for s in (0, 1)) for d in trees.REGISTRY['depth']}
        chosen = min(totals, key=lambda d: (totals[d], d))
        selection[task] = {'depth': chosen, 'internal_nll_sum_by_depth': {str(d): v for d, v in totals.items()},
                           'tree_count_by_seed': {str(s): seeds[s]['record']['trees'][f'{task}_d{chosen}']['tree_count'] for s in (0, 1)}}
    w, G = group_weights(F['unit'])
    table, internal, diagnostic, headroom = {}, [], [], {}
    for s in (0, 1):
        P = seeds[s]['probs']
        teacher = [P[f'income_d{selection["income"]["depth"]}'], P[f'occupation_d{selection["occupation"]["depth"]}']]
        refs = {'N_EQ': P['N_EQ'], 'U0': [F[f'references_s{s}']['U0_1'], F[f'references_s{s}']['U0_2']],
                'Ucal': [F[f'references_s{s}']['Ucal_1'], F[f'references_s{s}']['Ucal_2']]}
        internal += clauses(labels, teacher, refs, s, ('N_EQ',))
        diagnostic += clauses(labels, teacher, refs, s, ('U0', 'Ucal'))
        for name, pr in [('T_TREE', teacher), *[(k, v) for k, v in refs.items()]] + [(f'{t}_d{d}', None) for t in ('income', 'occupation') for d in trees.REGISTRY['depth']]:
            if pr is None:
                t = name.split('_d')[0]; i = 0 if t == 'income' else 1
                table[f'{s}|{name}'] = {t: summary(labels[t], P[name], K[t])}
            else:
                table[f'{s}|{name}'] = {t: summary(labels[t], pr[i], K[t]) for i, t in enumerate(('income', 'occupation'))}
        for i, t in enumerate(('income', 'occupation')):
            vt = task_vectors(labels[t], teacher[i], K[t]); vn = task_vectors(labels[t], refs['N_EQ'][i], K[t])
            for m in ('nll', 'brier', 'accuracy'):
                sign = 1 if m != 'accuracy' else -1   # positive = tree better
                diff = sign * (vn[m] - vt[m])
                headroom.setdefault(f'{t}|{m}', {})[str(s)] = {'diff': diff}
    for key, per in headroom.items():
        pooled = (per['0']['diff'] + per['1']['diff']) / 2
        for s in ('0', '1'):
            d = per[s].pop('diff'); per[s] = percentile(d.mean(), weighted_mean(d, w))
        per['mean_over_seeds'] = percentile(pooled.mean(), weighted_mean(pooled, w))
    prereq = all(c['point_pass'] for c in internal)
    occ = headroom['occupation|nll']['mean_over_seeds']
    result = {'schema': 'THP1-stage-a-report-v1', 'queue_unit_id': active['unit_id'], 'selection': selection,
              'internal_clauses_vs_N_EQ': internal, 'diagnostic_clauses_vs_in_sample_U0_Ucal': diagnostic,
              'task_metrics_FIT2': table, 'headroom_N_EQ_minus_tree_positive_is_tree_better': headroom,
              'prerequisite': 'TASK_MODEL_PREREQUISITE_MET' if prereq else 'TASK_MODEL_PREREQUISITE_NOT_MET',
              'occupation_nll_practical_target_0.010': {'point': occ['point'], 'met_at_point': bool(occ['point'] >= 0.010),
                                                       'met_at_lower_95': bool(occ['lower'] >= 0.010)},
              'bootstrap': {'draws': int(w.shape[0]), 'groups': G, 'seed': 2026101501, 'interval': '95% percentile, unadjusted diagnostic'},
              'U0_Ucal_caveat': 'U0/Ucal trained on all FIT including FIT2: in-sample on FIT2, diagnostic only',
              'INNER_opened': False}
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
