"""Verifier checks for designer 2's exact finite laws (8 cells x=(a,b,c), mass 1/8; income frozen at its posterior).

Independent re-implementation from the proposal's law definitions. Exhaustive over all Bell(8) = 4140 partitions for the
occupation release (calibrated block posterior). Arms as in the proposal (MIN, COND, COND_PP, JOINTSUM, INJ), under
(i) the proposal's Brier-only budgets and (ii) the binding clause set (accuracy, NLL, summed Brier) at the cap and at
cap + headroom. Also reports how the INJECTIVE counter-shift (the literal RCCS map class) changes Bayes recovery
relative to the task pair.
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
from fractions import Fraction as Fr
from itertools import product
from pathlib import Path

from mechanism_design_independent.exact import Cell, check_law, bayes_auc, auc_from_x, excess, partitions, block_posterior

OUT = Path(__file__).resolve().parent.parent / 'results' / 'pcrl_mechanism_design_v1' / 'roles' / 'math_verifier' / 'V_RCCS_FINITE_V1.json'


def law(pi_fn):
    cells = []
    for a, b, c in product((0, 1), (0, 1), (0, 1)):
        e1 = Fr(3, 10) + Fr(4, 10) * a
        e2 = Fr(30, 100) + Fr(5, 100) * a + Fr(30, 100) * b
        q1 = ((1 - e1, e1), (1 - e1, e1))
        q2 = ((1 - e2, e2), (1 - e2, e2))
        cells.append(Cell(Fr(1, 8), pi_fn(a, b, c), q1, q2, name=(a, b, c)))
    check_law(cells)
    return cells


LAWS = {
    'feasible': lambda a, b, c: Fr(1, 2) - Fr(15, 100) * a - Fr(15, 100) * b + Fr(15, 100) * c,
    'negative': lambda a, b, c: Fr(1, 2) - Fr(15, 100) * a - Fr(30, 100) * b,
    'redundant_only': lambda a, b, c: Fr(1, 2) - Fr(25, 100) * a,
    'coin_control': lambda a, b, c: Fr(1, 2) - Fr(15, 100) * a - Fr(15, 100) * b,
}
BRIER_BUDGETS = [Fr(x, 1000) for x in (0, 5, 9, 10, 15, 20, 25, 30, 40, 50)]
CLAUSE_BUDGETS = {'cap': (0.010, 0.010, 0.005), 'cap_plus_headroom': (0.016, 0.020, 0.009)}


def analyse(name, cells):
    inc = [c.eta(1) for c in cells]
    task = [c.eta(2) for c in cells]
    base_cls = [(inc[i], task[i]) for i in range(8)]
    rec = {'law': name, 'AUC_star_X': float(auc_from_x(cells)),
           'income_only_auc': float(bayes_auc(cells, lambda i: inc[i])),
           'TASK_occ_auc': float(bayes_auc(cells, lambda i: task[i])),
           'TASK_pair_auc': float(bayes_auc(cells, lambda i: (inc[i], task[i]))),
           'INJ_pair_auc': float(bayes_auc(cells, lambda i: (inc[i], task[i], cells[i].name[2])))}
    rows = []
    for p in partitions(8):
        r = block_posterior(cells, 2, p)
        ex = excess(cells, 2, r, task)
        blk_of = {}
        for bi, blk in enumerate(p):
            for i in blk:
                blk_of[i] = bi
        cls_blocks = {}
        for i in range(8):
            cls_blocks.setdefault(base_cls[i], set()).add(blk_of[i])
        rows.append({'part': p, 'brier': ex['brier_excess'], 'nll': ex['nll_excess'], 'acc': ex['accuracy_loss'],
                     'occ': bayes_auc(cells, lambda i: r[i]), 'pair': bayes_auc(cells, lambda i: (inc[i], r[i])),
                     'pp': all(len(v) == 1 for v in cls_blocks.values())})

    def pick(pool, key):
        pool = list(pool)
        return min(pool, key=lambda r: (key(r), r['brier'])) if pool else None

    def fmt(r):
        if r is None:
            return None
        return {'pair': round(float(r['pair']), 4), 'occ': round(float(r['occ']), 4), 'brier': round(float(r['brier']), 5),
                'nll': round(float(r['nll']), 5), 'acc_loss': round(float(r['acc']), 4),
                'blocks': [[cells[i].name for i in blk] for blk in r['part']]}
    rec['brier_only'] = []
    for B in BRIER_BUDGETS:
        feas = [r for r in rows if r['brier'] <= B]
        rec['brier_only'].append({'B': float(B), 'MIN': fmt(pick(feas, lambda r: r['occ'])),
                                  'COND': fmt(pick(feas, lambda r: r['pair'])),
                                  'COND_PP': fmt(pick([r for r in feas if r['pp']], lambda r: r['pair'])),
                                  'JOINTSUM': fmt(pick(feas, lambda r: r['occ'] + r['pair']))})
    rec['all_clauses'] = {}
    for bn, (ba, bnll, bb) in CLAUSE_BUDGETS.items():
        feas = [r for r in rows if float(r['acc']) <= ba + 1e-12 and r['nll'] <= bnll + 1e-12 and float(r['brier']) <= bb + 1e-12]
        rec['all_clauses'][bn] = {'MIN': fmt(pick(feas, lambda r: r['occ'])), 'COND': fmt(pick(feas, lambda r: r['pair'])),
                                  'COND_PP': fmt(pick([r for r in feas if r['pp']], lambda r: r['pair'])),
                                  'JOINTSUM': fmt(pick(feas, lambda r: r['occ'] + r['pair']))}
    return rec


def main():
    out = {}
    for name, fn in LAWS.items():
        out[name] = analyse(name, law(fn))
        r = out[name]
        print(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
        for row in r['brier_only']:
            print('  B=%.3f' % row['B'], {a: (None if row[a] is None else (row[a]['pair'], row[a]['occ'], row[a]['brier'], row[a]['nll'], row[a]['acc_loss'])) for a in ('MIN', 'COND', 'COND_PP', 'JOINTSUM')})
        for bn, v in r['all_clauses'].items():
            print('  ALL-CLAUSES', bn, {a: (None if v[a] is None else (v[a]['pair'], v[a]['occ'], v[a]['brier'], v[a]['nll'], v[a]['acc_loss'])) for a in v})
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
