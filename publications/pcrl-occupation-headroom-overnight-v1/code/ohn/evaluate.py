"""Development utility evaluation of a COMPLETE release (income anchor + occupation output) on MODEL_SELECT, plus the
occupation-specific diagnostics of the overnight occupation-headroom campaign (selection feedback only; every MODEL_SELECT label
read is logged by ohn.common.load_role).

Per seed (unchanged definitions, ihs.evaluate.utility / scm_bench.selection.clauses): the 12 point clauses (income and occupation
x U0 and Ucal x accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005), each with a household-clustered
paired SE over the 5,306 MODEL_SELECT households, its margin in metric and SE units; the men's (sex0) and women's (sex1) occupation
summed-Brier excess against U0 (guard 0.005) with linearized clustered SEs.

Occupation diagnostics (brief section 8; descriptive, never gates by themselves):
- per-class decomposition of the occupation NLL and summed-Brier excess against U0 and Ucal (contribution of the people whose true
  group is c to the total excess, with a clustered SE);
- per-SEX occupation accuracy loss, NLL and Brier excess against U0 and Ucal;
- paired comparisons with the equally exposed task-only references T_TREE and TASK_REFIT (and the inherited JOINT_TREE output):
  candidate minus reference occupation accuracy loss, NLL and Brier, with clustered SEs;
- household cross-fitting checks (if CROSSFIT.npz exists): the cross-fitted (out-of-fold) MODEL_FIT utility of the pipeline in
  absolute terms and against U0 / T_TREE on MODEL_FIT (both IN-SAMPLE there: exposure flagged), and the deployed-versus-fold-average
  difference on MODEL_SELECT (paired, clustered SEs; max absolute probability difference; decision agreement).
"""
from __future__ import annotations

import numpy as np

from ihs.evaluate import utility, cluster_se, subgroup_se      # unchanged clause / SE definitions
from . import common as C

K_OCC = 5


def task_vec(y, p, k):
    return C.task_vectors(y, p, k)


def ms_context(seed, purpose):
    """MODEL_SELECT data (labels + SEX, logged) and the frozen U0 / Ucal / T_TREE / TASK_REFIT / JOINT references, row-aligned."""
    S = C.load_role('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), purpose=purpose)
    bank, ids = C.frozen_bank(seed, roles=('MODEL_SELECT',))
    if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
        raise AssertionError('frozen bank MODEL_SELECT row order differs')
    refs = {r: {t: bank[r]['MODEL_SELECT'][t] for t in C.TASKS} for r in ('U0', 'Ucal')}
    others = {'T_TREE': bank['T_TREE']['MODEL_SELECT']['occupation'], 'TASK_REFIT': bank['TASK_REFIT']['MODEL_SELECT']['occupation'],
              'JOINT': bank['H_LJ']['MODEL_SELECT']['occupation']}
    return S, refs, others


def _excess(y, p, ref, k):
    a, b = task_vec(y, p, k), task_vec(y, ref, k)
    return {'accuracy_loss': b['accuracy'] - a['accuracy'], 'nll_excess': a['nll'] - b['nll'], 'brier_excess': a['brier'] - b['brier']}


def per_class(y, p, refs, unit):
    out = {}
    for r, ref in refs.items():
        d = _excess(y, p, ref['occupation'], K_OCC)
        n = len(y)
        for m in ('nll_excess', 'brier_excess'):
            for c in range(K_OCC):
                v = np.where(y == c, d[m], 0.0)          # contribution of group c to the total mean excess
                out[f'{r}|{m}|class{c}'] = {'contribution': float(v.mean()), 'se': cluster_se(v, unit), 'n_class': int((y == c).sum()),
                                            'mean_within_class': float(d[m][y == c].mean())}
        out[f'{r}|share'] = [float((y == c).mean()) for c in range(K_OCC)]
    return out


def per_sex(y, p, refs, sex, unit):
    out = {}
    for r, ref in refs.items():
        d = _excess(y, p, ref['occupation'], K_OCC)
        for g, name in ((0, 'men'), (1, 'women')):
            mask = sex == g
            for m, v in d.items():
                out[f'{name}|{r}|{m}'] = {'value': float(v[mask].mean()), 'se': subgroup_se(v, mask, unit)}
    return out


def versus(y, p, others, unit):
    """Candidate minus reference occupation loss (positive = candidate worse), paired, clustered SEs."""
    out = {}
    for name, q in others.items():
        a, b = task_vec(y, p, K_OCC), task_vec(y, q, K_OCC)
        for m, d in (('accuracy_loss', b['accuracy'] - a['accuracy']), ('nll_diff', a['nll'] - b['nll']), ('brier_diff', a['brier'] - b['brier'])):
            out[f'{name}|{m}'] = {'value': float(d.mean()), 'se': cluster_se(d, unit)}
    return out


def crossfit_diag(seed, cf, occ_ms, S, purpose):
    """Cross-fitted MODEL_FIT utility and deployed-versus-fold-average difference on MODEL_SELECT."""
    if cf is None:
        return None
    F = C.load_role('MODEL_FIT', ('features', 'occupation', 'sex'), purpose=purpose)
    oof = cf['MODEL_FIT__oof']
    if oof.shape != (len(F['row_id']), K_OCC) or not np.array_equal(cf['fold'], C.crossfit_folds(F['unit'])):
        raise AssertionError('cross-fitting arrays do not match the registered MODEL_FIT folds')
    sa = C.stage_a_outputs(seed)
    if not np.array_equal(sa['row_id__MODEL_FIT'], F['row_id']):
        raise AssertionError('stored Stage A MODEL_FIT row order differs')
    y = F['occupation']
    u0 = sa['U0__MODEL_FIT__occupation']; tt = sa['T_TREE__MODEL_FIT__occupation']
    own = task_vec(y, oof, K_OCC)
    res = {'oof_absolute': {m: float(v.mean()) for m, v in own.items()}, 'people': int(len(y)),
           'exposure_note': 'U0 and T_TREE were fitted on all MODEL_FIT people: their MODEL_FIT outputs are in-sample, so the '
                            'comparisons below favour the references; the OOF predictions are out of fold for every label-using step '
                            'the family refitted per fold (see OCC.json for any frozen in-sample layer)'}
    for name, q in (('U0_in_sample', u0), ('T_TREE_in_sample', tt)):
        d = _excess(y, oof, q, K_OCC)
        res[f'oof_vs_{name}'] = {m: {'value': float(v.mean()), 'se': cluster_se(v, F['unit'])} for m, v in d.items()}
    fa = cf['MODEL_SELECT__foldavg']
    ys = S['occupation']
    a, b = task_vec(ys, occ_ms, K_OCC), task_vec(ys, fa, K_OCC)
    res['deployed_minus_foldavg_on_MODEL_SELECT'] = {
        'accuracy_loss': {'value': float((b['accuracy'] - a['accuracy']).mean()), 'se': cluster_se(b['accuracy'] - a['accuracy'], S['unit'])},
        'nll_diff': {'value': float((a['nll'] - b['nll']).mean()), 'se': cluster_se(a['nll'] - b['nll'], S['unit'])},
        'brier_diff': {'value': float((a['brier'] - b['brier']).mean()), 'se': cluster_se(a['brier'] - b['brier'], S['unit'])},
        'max_abs_prob_diff': float(np.abs(occ_ms - fa).max()), 'decision_agreement': float((occ_ms.argmax(1) == fa.argmax(1)).mean())}
    res['oof_minus_deployed_ms_absolute'] = {m: float(own[m].mean() - a[m].mean()) for m in own}
    res['oof_by_sex'] = {name: {m: float(own[m][F['sex'] == g].mean()) for m in own} for g, name in ((0, 'men'), (1, 'women'))}
    return res


def complete_eval(seed, inc, occ, S, refs, others, cf=None, purpose=None):
    n = len(S['row_id'])
    for t, p, k in (('income', inc, 2), ('occupation', occ, 5)):
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (n, k) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
            raise ValueError(f'{t}: release violates the probability-vector contract')
    labels = {t: S[t] for t in C.TASKS}
    outs = {'income': inc, 'occupation': occ}
    u = utility(labels, outs, refs, S['sex'], S['unit'], seed)
    y = S['occupation']
    out = {'utility': u,
           'occupation_per_class': per_class(y, occ, refs, S['unit']),
           'occupation_per_sex': per_sex(y, occ, refs, S['sex'], S['unit']),
           'occupation_versus_task_only_references': versus(y, occ, others, S['unit']),
           'wire_sha256': {t: C.wire_sha(outs[t]) for t in C.TASKS},
           'distinct_vectors': {t: int(np.unique(np.ascontiguousarray(outs[t]), axis=0).shape[0]) for t in C.TASKS}}
    if cf is not None:
        out['crossfit'] = crossfit_diag(seed, cf, occ, S, purpose or 'cross-fitting diagnostic')
    return out
