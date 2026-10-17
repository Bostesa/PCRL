"""Development evaluation on MODEL_SELECT (selection feedback only; every label read is logged by amd.common.load_role).

For one seed's release on MODEL_SELECT (income and occupation float64 probability vectors; decisions = argmax):
- the 12 original point clauses against the frozen U0 / Ucal MODEL_SELECT outputs (accuracy loss <= 0.010, NLL excess
  <= 0.010, summed Brier excess <= 0.005; scm_bench.selection.clauses, unchanged),
- the within-SEX occupation summed-Brier excess against U0 for each SEX group (guard 0.005),
- task metrics by SEX and per-class occupation NLL (diagnostics),
- household-separated reader proxies: MODEL_SELECT split into the source's two household halves (fixed seed), each reader
  fitted on one half and scored on the other in both directions; views pair / income / occupation; readers OUT_LR, OUT_HGB,
  LOG_MLP (source reader constructors) and, on the pair, the exact-wire LOOKUP. `source_proxy` = max(OUT_LR, OUT_HGB, LOOKUP)
  on the pair, the source definition; `max` = maximum over every proxy reader of the view.
- distinct-output counts (exact float64 rows) per view.
These proxies are development diagnostics on previously used records, not fresh confirmation and not population-optimal.
"""
from __future__ import annotations
import warnings

import numpy as np

from acs_replication import trees
from scm_bench.readers import kinds, lookup_keys, fit_lookup, predict_lookup
from scm_bench.selection import clauses
from task_conditioned_representation.attacks import fit_reader, predict_reader, MODEL_SEED_BASE
from task_conditioned_representation.inference import auc
from . import common as C

PROXY_READERS = ('OUT_LR', 'OUT_HGB', 'LOG_MLP')


def halves(unit):
    groups = np.unique(unit)
    perm = np.random.default_rng(trees.seed_for('SCM1proxysplit')).permutation(len(groups))
    a = np.isin(unit, groups[perm[:len(groups) // 2]])
    return a, ~a


def view_tables(outs):
    a, b = kinds(outs['income'], 2), kinds(outs['occupation'], 5)
    return {'pair': {'out': np.column_stack((a['out'], b['out'])), 'log': np.column_stack((a['log'], b['log']))},
            'income': {'out': a['out'], 'log': a['log']}, 'occupation': {'out': b['out'], 'log': b['log']}}


def proxies(outs, sex, unit, seed, readers=PROXY_READERS, views=('pair', 'income', 'occupation')):
    hs = halves(unit); tabs = view_tables(outs); res = {}
    for view in views:
        tab = tabs[view]; res[view] = {}
        for name in readers:
            kind = 'log' if name.startswith('LOG') else 'out'
            aucs = []
            for tr, te in ((hs[0], hs[1]), (hs[1], hs[0])):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m = fit_reader(tab[kind][tr], sex[tr], name, MODEL_SEED_BASE + seed)
                aucs.append(float(auc(sex[te], predict_reader(m, tab[kind][te]))))
            res[view][name] = float(np.mean(aucs))
        if view == 'pair':
            keys = np.array(lookup_keys(outs['income'], outs['occupation']), dtype=object)
            aucs = []
            for tr, te in ((hs[0], hs[1]), (hs[1], hs[0])):
                m = fit_lookup(list(keys[tr]), sex[tr]); aucs.append(float(auc(sex[te], predict_lookup(m, list(keys[te])))))
            res[view]['LOOKUP'] = float(np.mean(aucs))
            res[view]['source_proxy'] = max(res[view][n] for n in ('OUT_LR', 'OUT_HGB', 'LOOKUP') if n in res[view])
        res[view]['max'] = max(v for k, v in res[view].items() if k not in ('max', 'source_proxy'))
    return res


def distinct(outs):
    d = {t: int(np.unique(np.ascontiguousarray(outs[t]), axis=0).shape[0]) for t in C.TASKS}
    d['pair'] = int(np.unique(np.column_stack((outs['income'], outs['occupation'])), axis=0).shape[0])
    return d


def check_outs(outs, n):
    for t in C.TASKS:
        p = np.asarray(outs[t], dtype=np.float64)
        if p.shape != (n, C.K[t]) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
            raise ValueError(f'{t}: release violates the probability-vector contract')


def measure(seed, outs, S, refs, with_proxy=True):
    """outs: {'income': (n,2), 'occupation': (n,5)} on MODEL_SELECT rows in S order; refs: {'U0': {...}, 'Ucal': {...}}."""
    check_outs(outs, len(S['row_id']))
    labels = {t: S[t] for t in C.TASKS}
    cl = clauses(labels, outs, refs, seed)
    sex = S['sex']
    own = {t: C.task_vectors(labels[t], outs[t], C.K[t]) for t in C.TASKS}
    u0 = {t: C.task_vectors(labels[t], refs['U0'][t], C.K[t]) for t in C.TASKS}
    guard = {f'sex{g}': float((own['occupation']['brier'] - u0['occupation']['brier'])[sex == g].mean()) for g in (0, 1)}
    by_sex = {f'sex{g}': {f'{t}_{m}_excess_vs_U0': float((own[t][m] - u0[t][m])[sex == g].mean())
                          for t in C.TASKS for m in ('nll', 'brier')} for g in (0, 1)}
    y = labels['occupation']
    classwise = {str(k): float((own['occupation']['nll'] - u0['occupation']['nll'])[y == k].mean()) for k in range(5)}
    out = {'clauses': cl, 'point_pass_12': all(c['point_pass'] for c in cl), 'max_normalized_12': max(c['normalized'] for c in cl),
           'guard_occupation_brier_excess_vs_U0': guard, 'guard_pass': all(v <= C.GUARD for v in guard.values()),
           'by_sex': by_sex, 'occupation_classwise_nll_excess_vs_U0': classwise,
           'task_summary': {t: {m: float(v.mean()) for m, v in own[t].items()} for t in C.TASKS},
           'distinct_outputs': distinct(outs)}
    if with_proxy:
        out['proxy'] = proxies(outs, sex, S['unit'], seed)
    return out


def conditions(m, seed, margin=1.0):
    """(name, value, limit) for the 12 clauses and the two guards at `margin` x cap."""
    out = [(f"s{seed}|{c['task']}|{c['reference']}|{c['metric']}", c['value'], margin * c['cap']) for c in m['clauses']]
    out += [(f's{seed}|guard|{g}', v, margin * C.GUARD) for g, v in m['guard_occupation_brier_excess_vs_U0'].items()]
    return out


def eligibility(per_seed, margin=1.0):
    conds = [c for s, m in per_seed.items() for c in conditions(m, s, margin)]
    failed = [n for n, v, lim in conds if v > lim]
    return {'eligible': not failed, 'failed': failed, 'worst_normalized': max(v / lim for _, v, lim in conds)}


def ms_context(seed, purpose):
    """MODEL_SELECT data (labels + SEX, logged) and the frozen U0 / Ucal references, row-aligned."""
    S = C.load_role('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), purpose=purpose)
    bank, ids = C.frozen_bank(seed, roles=('MODEL_SELECT',))
    if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
        raise AssertionError('frozen bank MODEL_SELECT row order differs')
    refs = {r: {t: bank[r]['MODEL_SELECT'][t] for t in C.TASKS} for r in ('U0', 'Ucal')}
    return S, refs, bank
