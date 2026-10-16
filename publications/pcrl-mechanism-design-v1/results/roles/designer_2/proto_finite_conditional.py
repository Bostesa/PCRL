"""D2 data-free check 2: exact finite laws (no real data; exact Bayes AUC from the released view).

Cells x = (a, b, c), each binary and independent with mass 1/8. Income Y1 depends on a only, so the frozen income
release (its exact posterior) discloses a. Occupation Y2 (K2 = 2) depends on (a, b). S depends on (a, b, c); c is
task-irrelevant (an 'S-only' input such as a class-of-worker category that predicts SEX but not the tasks).
S, Y1, Y2 are conditionally independent given x.

Arms (income frozen at its exact posterior; the occupation release is the calibrated posterior of a partition):
  TASK    identity occupation release (cells sharing (a, b) are automatically pooled: same posterior);
  MIN     partition minimising OCCUPATION-ONLY Bayes AUC under the occupation budget (local target);
  COND    partition minimising PAIR Bayes AUC under the same budget (release-conditional target, income frozen);
  COND_PP COND restricted to post-processing of the base pair: blocks never split cells with identical base releases
          (so it cannot use c, the information the propensity driver supplies);
  JOINTSUM partition minimising occupation-only AUC + pair AUC (the own-view-plus-pair structure of RAW-J / JOINT_TREE,
          with income frozen as in every arm here);
  INJ     an unpooled counter-shift (distinct tiny shifts by c): shows an injective release cannot lower Bayes AUC.
Budget: summed occupation Brier excess vs TASK (exact). Exhaustive search over all 4140 set partitions of 8 cells.
"""
from mechanism_design import no_real_data
no_real_data.install()

import json
from fractions import Fraction as Fr
from itertools import product

from mechanism_design.prototype.finite_law import (FiniteLaw, bayes_auc, clause_excess, posterior_release,
                                                   set_partitions, task_metrics)


def make_law(eta1, eta2, pi):
    cells = list(product((0, 1), (0, 1), (0, 1)))
    w, joint = [], []
    for (a, b, c) in cells:
        p1, p2, ps = eta1(a), eta2(a, b), pi(a, b, c)
        tab = [[[Fr(0), Fr(0)], [Fr(0), Fr(0)]], [[Fr(0), Fr(0)], [Fr(0), Fr(0)]]]
        for s in (0, 1):
            for y1 in (0, 1):
                for y2 in (0, 1):
                    tab[s][y1][y2] = (ps if s else 1 - ps) * (p1 if y1 else 1 - p1) * (p2 if y2 else 1 - p2)
        w.append(Fr(1, 8))
        joint.append(tuple(tuple(tuple(r) for r in t) for t in tab))
    return FiniteLaw(w=tuple(w), joint=tuple(joint), k1=2, k2=2, names=tuple(cells)), cells


def analyse(law, cells, budgets, label):
    inc = {c: tuple(law.p_y(c, 1)) for c in law.cells}
    task_occ = {c: tuple(law.p_y(c, 2)) for c in law.cells}
    base_key = {c: (inc[c], task_occ[c]) for c in law.cells}
    task_pair = bayes_auc(law, lambda c: (inc[c], task_occ[c]))
    task_occ_auc = bayes_auc(law, lambda c: task_occ[c])
    inc_auc = bayes_auc(law, lambda c: inc[c])
    x_auc = bayes_auc(law, lambda c: c)
    rows = []
    for part in set_partitions(list(law.cells)):
        q = posterior_release(law, 2, part)
        ex = clause_excess(law, 2, lambda c: q[c], lambda c: task_occ[c])
        # post-processing: every base-release class lies inside one block
        cls = {}
        for blk_id, blk in enumerate(part):
            for c in blk:
                cls.setdefault(base_key[c], set()).add(blk_id)
        pp_ok = all(len(v) == 1 for v in cls.values())
        rows.append({'part': part, 'brier': ex['brier_excess'], 'nll': ex['nll_excess'], 'acc_loss': ex['accuracy_loss'],
                     'occ_auc': bayes_auc(law, lambda c: q[c]), 'pair_auc': bayes_auc(law, lambda c: (inc[c], q[c])),
                     'pp': pp_ok})
    out = {'label': label, 'AUC_star_X': x_auc, 'income_only_auc': inc_auc, 'TASK_occ_auc': task_occ_auc,
           'TASK_pair_auc': task_pair, 'by_budget': []}
    for B in budgets:
        feas = [r for r in rows if r['brier'] <= B]
        def pick(key, pool):
            pool = list(pool)
            return min(pool, key=lambda r: (r[key], r['brier'])) if pool else None
        m = pick('occ_auc', feas); cnd = pick('pair_auc', feas); pp = pick('pair_auc', [r for r in feas if r['pp']])
        for r in feas:
            r['sum_auc'] = r['occ_auc'] + r['pair_auc']
        js = pick('sum_auc', feas)   # analogue of the RAW-J structure (own-view + pair terms), income frozen
        fmt = lambda r: None if r is None else {'pair_auc': round(r['pair_auc'], 4), 'occ_auc': round(r['occ_auc'], 4),
                                               'brier': round(r['brier'], 5), 'nll': round(r['nll'], 5),
                                               'acc_loss': round(r['acc_loss'], 4),
                                               'blocks': [[cells[c] for c in blk] for blk in r['part']]}
        out['by_budget'].append({'budget': B, 'MIN': fmt(m), 'COND': fmt(cnd), 'COND_PP': fmt(pp), 'JOINTSUM': fmt(js)})
    # injective counter-shift: distinct tiny shifts by c -> every cell distinguishable
    q_inj = {c: (task_occ[c][0] - Fr(1, 1000) * (2 * law.names[c][2] - 1), task_occ[c][1] + Fr(1, 1000) * (2 * law.names[c][2] - 1))
             for c in law.cells}
    out['INJ_pair_auc'] = bayes_auc(law, lambda c: (inc[c], q_inj[c]))
    out['INJ_equals_AUC_star_X'] = out['INJ_pair_auc'] == x_auc
    out['exact_arithmetic'] = 'fractions.Fraction throughout (task NLL in float)'
    return out


def main():
    res = {}
    budgets = [Fr(x, 1000) for x in (0, 5, 10, 15, 20, 25, 30, 40, 50)]
    # Feasible case: occupation needs b (gap 0.30); a is disclosed by income; c is S-only.
    law, cells = make_law(lambda a: Fr(3, 10) + Fr(4, 10) * a, lambda a, b: Fr(30, 100) + Fr(5, 100) * a + Fr(30, 100) * b,
                          lambda a, b, c: Fr(1, 2) - Fr(15, 100) * a - Fr(15, 100) * b + Fr(15, 100) * c)
    res['feasible_case'] = analyse(law, cells, budgets, 'feasible: S-only input c available to the driver')
    # Negative case (required disclosure): no S-only input; S follows the occupation-relevant b exactly as the task needs.
    law2, cells2 = make_law(lambda a: Fr(3, 10) + Fr(4, 10) * a, lambda a, b: Fr(30, 100) + Fr(5, 100) * a + Fr(30, 100) * b,
                            lambda a, b, c: Fr(1, 2) - Fr(15, 100) * a - Fr(30, 100) * b)
    res['negative_case'] = analyse(law2, cells2, budgets, 'negative: S tracks the occupation signal; no S-only input')
    # Redundancy-only case: S depends on a only (fully disclosed by income) -> occupation protection is pointless.
    law3, cells3 = make_law(lambda a: Fr(3, 10) + Fr(4, 10) * a, lambda a, b: Fr(30, 100) + Fr(5, 100) * a + Fr(30, 100) * b,
                            lambda a, b, c: Fr(1, 2) - Fr(25, 100) * a)
    res['redundant_only_case'] = analyse(law3, cells3, budgets, 'S depends on a only (income discloses it)')
    # Coin control: the feasible law with c made S-irrelevant (c can still split cells, acting as a deterministic coin).
    law4, cells4 = make_law(lambda a: Fr(3, 10) + Fr(4, 10) * a, lambda a, b: Fr(30, 100) + Fr(5, 100) * a + Fr(30, 100) * b,
                            lambda a, b, c: Fr(1, 2) - Fr(15, 100) * a - Fr(15, 100) * b)
    res['coin_control_case'] = analyse(law4, cells4, budgets, 'feasible law with c S-irrelevant (coin only)')
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
