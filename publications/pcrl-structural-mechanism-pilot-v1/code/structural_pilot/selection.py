"""Frozen MODEL_SELECT rule applied to this campaign's fitted arms (copied from acs_replication.selection, unchanged in substance).

Rule: (1) a configuration is FEASIBLE iff all 24 point utility clauses (2 tasks x {U0, Ucal} x {accuracy loss <= 0.010, NLL
excess <= 0.010, summed Brier excess <= 0.005} x 2 seeds) pass on MODEL_SELECT; (2) among feasible configurations, minimise
the mean over seeds of the recovery proxy: coalition AUC on MODEL_SELECT of a fixed OUT_LR reader (LogisticRegression C=1 on
the released pair wire, finite scaler) trained on the MODEL_FIT pair wire with MODEL_FIT SEX; ties by the fixed grid order;
(3) if no configuration is feasible, the nominee is NONE (the pre-declared diagnostic rule in ANALYSIS_LOCK applies instead).
One common configuration across both seeds. No DEV_REPORT, ATTACK or SEALED label is read.
"""
from __future__ import annotations
import warnings

import numpy as np

from . import common as C


def pair_out(svc_role):
    cols = []
    for t in C.TASKS:
        p = np.asarray(svc_role[t], dtype=np.float64)
        cols += [p, p.argmax(1)[:, None].astype(np.float64)]
    return np.column_stack(cols)


def proxy_auc(out_fit, out_sel, sex_fit, sex_sel, seed):
    from task_conditioned_representation.attacks import fit_reader, predict_reader, MODEL_SEED_BASE
    from task_conditioned_representation.inference import auc
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rec = fit_reader(pair_out(out_fit), sex_fit, 'OUT_LR', MODEL_SEED_BASE + seed)
    return float(auc(sex_sel, predict_reader(rec, pair_out(out_sel))))


def clauses(labels, probs, refs, seed):
    """Per seed: task x reference x (accuracy loss, NLL excess, summed Brier excess), inclusive caps (acs_replication.data.clauses)."""
    out = []
    for task in C.TASKS:
        k = C.K[task]; own = C.task_vectors(labels[task], probs[task], k)
        for ref in ('U0', 'Ucal'):
            r = C.task_vectors(labels[task], refs[ref][task], k)
            diff = {'accuracy_loss': r['accuracy'].mean() - own['accuracy'].mean(), 'nll_excess': own['nll'].mean() - r['nll'].mean(),
                    'brier_excess': own['brier'].mean() - r['brier'].mean()}
            for metric, cap in C.CAPS.items():
                out.append({'seed': seed, 'task': task, 'reference': ref, 'metric': metric, 'value': float(diff[metric]), 'cap': cap,
                            'normalized': float(diff[metric] / cap), 'point_pass': bool(diff[metric] <= cap)})
    return out


def select(grid, per_seed):
    """grid: ordered configuration names; per_seed[s][cfg] = {'clauses': [...], 'proxy_auc': float}."""
    table = []
    for c in grid:
        cl = per_seed[0][c]['clauses'] + per_seed[1][c]['clauses']
        table.append({'configuration': c, 'feasible': all(x['point_pass'] for x in cl), 'max_normalized': max(x['normalized'] for x in cl),
                      'proxy_auc_mean': float(np.mean([per_seed[s][c]['proxy_auc'] for s in (0, 1)])),
                      'proxy_auc_per_seed': [per_seed[s][c]['proxy_auc'] for s in (0, 1)],
                      'failed_clauses': [f"s{x['seed']}|{x['task']}|{x['reference']}|{x['metric']}" for x in cl if not x['point_pass']]})
    feas = [r for r in table if r['feasible']]
    if feas:
        win = min(feas, key=lambda r: (r['proxy_auc_mean'], grid.index(r['configuration'])))
        return {'status': 'NOMINEE_SELECTED', 'configuration': win['configuration'], 'rule': 'feasible; minimum mean MODEL_SELECT proxy AUC', 'table': table}
    closest = min(table, key=lambda r: (r['max_normalized'], grid.index(r['configuration'])))
    return {'status': 'NO_FEASIBLE_CONFIGURATION', 'configuration': None, 'closest_failing_configuration': closest['configuration'],
            'rule': 'no feasible configuration; closest failing = minimum maximum normalised clause value (diagnostic only)', 'table': table}
