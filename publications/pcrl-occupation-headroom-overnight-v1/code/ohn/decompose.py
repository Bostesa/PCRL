"""Paired cost decomposition (fallback programme): where does a candidate occupation output spend task quality relative to a base
output, by occupation class and by SEX group, with household-clustered SEs on MODEL_SELECT (labels and SEX read and logged).

    python -m ohn.decompose --pair CAND=BASE [--pair ...] --seed s --out results/<campaign>/decomposition/DECOMP_S<s>.json

CAND / BASE are '<FAMILY>/<CONFIG>' occupation releases or reference names (T_TREE, TASK_REFIT, JOINT, O1). Per pair: total occupation
NLL and summed-Brier difference (candidate minus base) with SE; each class's contribution to that total (people whose true group is
c) with SE; per-SEX mean differences with linearized SEs; and the change in each class's mean predicted probability, overall and by
SEX (label-free: where the probability mass moved).
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from ihs.evaluate import cluster_se, subgroup_se
from . import common as C
from . import devrun as D


def occ(name, seed):
    if '/' in name:
        fam, cfg = name.split('/')
        return D.load_occupation(fam, cfg, seed, roles=('MODEL_SELECT',))[0]['MODEL_SELECT']
    return D.source_occupation(name, seed, 'MODEL_SELECT')


def pair_decomp(y, sex, unit, p, q):
    a, b = C.task_vectors(y, p, 5), C.task_vectors(y, q, 5)
    out = {}
    for m in ('nll', 'brier'):
        d = a[m] - b[m]
        out[f'{m}_total'] = {'value': float(d.mean()), 'se': cluster_se(d, unit)}
        out[f'{m}_by_class'] = {f'class{c}': {'contribution': float(np.where(y == c, d, 0.0).mean()),
                                             'se': cluster_se(np.where(y == c, d, 0.0), unit), 'n': int((y == c).sum())} for c in range(5)}
        out[f'{m}_by_sex'] = {name: {'value': float(d[sex == g].mean()), 'se': subgroup_se(d, sex == g, unit)}
                              for g, name in ((0, 'men'), (1, 'women'))}
    dp = p - q
    out['mean_probability_shift'] = {'all': dp.mean(0).tolist(), 'men': dp[sex == 0].mean(0).tolist(), 'women': dp[sex == 1].mean(0).tolist()}
    out['decision_change_rate'] = float((p.argmax(1) != q.argmax(1)).mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', action='append', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    S = C.load_role('MODEL_SELECT', ('features', 'occupation', 'sex'), purpose=f'fallback cost decomposition (paired, by class and SEX), seed {a.seed}')
    y, sex, unit = S['occupation'], S['sex'], S['unit']
    res = {'schema': 'OHN1-decomposition-v1', 'seed': a.seed, 'utc': C.now(), 'pairs': {},
           'note': 'MODEL_SELECT (adaptively used); descriptive; candidate minus base; positive = candidate worse'}
    for spec in a.pair:
        cand, base = spec.split('=')
        res['pairs'][spec] = pair_decomp(y, sex, unit, occ(cand, a.seed), occ(base, a.seed))
    C.write_json(a.out, res)
    for spec, r in res['pairs'].items():
        print(spec, 'NLL %+.4f (%.4f)' % (r['nll_total']['value'], r['nll_total']['se']),
              'by class', ' '.join('%+.4f' % r['nll_by_class'][f'class{c}']['contribution'] for c in range(5)),
              'men %+.4f women %+.4f' % (r['nll_by_sex']['men']['value'], r['nll_by_sex']['women']['value']))


if __name__ == '__main__':
    main()
