"""Verifier check 3c (DRF1; label-free): reader-grid validity of a registered grid by my reading of RULES.md 9 (and NRM1 RULES.md 6),
from the grid's per-recipe SCORE-score fingerprints (k0 vs k1 per member, view, version and repetition) and my own MODEL_SELECT
distinct-input counts (check 2), compared with the reading's DECISIONS.json.

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check3c_grid_validity.py --plan PLAN_DRF_R1 --bank DRF1_R1 \
        --check2 OUT2 --out FILE

My reading of the rule: a seed-dependent recipe (ET5, ET25, MLP64, MLP6432, OUT_MLP, LOG_MLP) whose k0 and k1 SCORE scores are
bitwise equal (equal fingerprints) must be explained, else the grid is invalid:
  - off the minimum set, by NRM1 A-SC1 (iii): (a) another seed-dependent recipe differs on the same wire, view, version and
    repetition; (b) the same recipe differs on another view of the same wire (same version and repetition); (c) the view's
    distinct-input count is recorded label-free;
  - on the minimum set (candidate, twin, saved CTRL points, OBJ_MI, SUCC), only ET5 / ET25 by the extra-trees support
    explanation: (iii) (a)-(c) and the view's distinct inputs <= 1600 (ET5) / <= 320 (ET25);
  - anywhere, by the few-valued explanation: the view's distinct-input count <= 320 and the same recipe differs on another wire or
    view in the same version, repetition and seed pair;
  - anywhere, by the finite-pair extension: a pair view with 320 < n, the same wire's income and occupation counts both <= 320,
    n <= their product <= 1024, and the same recipe differs on the pair view of another wire in the same version, repetition and
    seed pair.
Seed-invariant recipes whose fingerprints differ are counted as NONDETERMINISTIC_RECIPE (recorded, not fatal).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

CAMPAIGN = 'pcrl_direct_recovery_frontier_v1'
RES = REPO / 'results' / CAMPAIGN
PRIV = REPO / '.private' / CAMPAIGN
SEED_DEP = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
SUBJECT = {'B2TWIN': 'REF:B2TWIN', 'H_LJ': 'REF:H_LJ', 'NEURAL_RAW': 'REF:NEURAL_RAW', 'RETIRED_B3CTL': 'REF:RETIRED_B3CTL',
           'SUCC': 'REF:SUCC', 'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE', 'TWJ': 'REF:TWJ', 'U0': 'REF:U0', 'Ucal': 'REF:Ucal',
           'W': 'W', 'TWB_C64': 'TWB_C64'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='PLAN_DRF_R1')
    ap.add_argument('--bank', default='DRF1_R1')
    ap.add_argument('--check2', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    items = []
    plan = json.loads((RES / 'claims' / f'{a.plan}.json').read_text())
    c2 = json.loads(Path(a.check2).read_text())
    minimum = {plan['cand'], plan['twin']['id'], plan['obj_mi']['config_id'], 'SUCC'} | set(plan['paths'].get('CTRL') or [])
    fp, members = {}, set()
    for v in (0, 1):
        g = json.loads((PRIV / 'runs' / 'GRID' / a.bank / f'S{v}' / 'GRID.json').read_text())
        for cell, cres in g['cells'].items():
            vv, r, k = V.parse_cell(cell)
            for m, ent in cres.items():
                members.add(m)
                for view, e in ent.items():
                    for rec, h in e['extras']['score_sha256'].items():
                        fp[(vv, r, k, m, view, rec)] = h
    distinct = {}
    for m in sorted(members):
        sid = SUBJECT.get(m, m)
        for v in (0, 1):
            distinct[(m, v)] = c2['per_subject'][sid][str(v)]['distinct_inputs']
    recipes = sorted({key[5] for key in fp})
    views = ('income', 'occupation', 'pair')

    def differs(v, r, m, view, rec):
        a_, b_ = fp.get((v, r, 0, m, view, rec)), fp.get((v, r, 1, m, view, rec))
        return None if a_ is None or b_ is None else a_ != b_

    equalities, nondet, unexplained = [], 0, []
    n_checks = 0
    for v in (0, 1):
        for r in (1, 2):
            for m in sorted(members):
                for view in views:
                    for rec in recipes:
                        d = differs(v, r, m, view, rec)
                        if d is None:
                            continue
                        n_checks += 1
                        if rec not in SEED_DEP:
                            nondet += int(d)
                            continue
                        if d:
                            continue
                        n = distinct[(m, v)][view]
                        a_ = any(differs(v, r, m, view, o) for o in SEED_DEP if o != rec)
                        b_ = any(differs(v, r, m, w, rec) for w in views if w != view)
                        c_ = n is not None
                        iii = a_ and b_ and c_
                        few = n <= 320 and (any(differs(v, r, m2, view, rec) for m2 in members if m2 != m) or b_)
                        ni, no = distinct[(m, v)]['income'], distinct[(m, v)]['occupation']
                        pair_ext = (view == 'pair' and n > 320 and ni <= 320 and no <= 320 and n <= ni * no <= 1024 and
                                    any(differs(v, r, m2, 'pair', rec) for m2 in members if m2 != m))
                        on_min = m in minimum
                        if on_min:
                            et = rec in ('ET5', 'ET25') and iii and n <= (1600 if rec == 'ET5' else 320)
                            expl = 'ER_EXTRA_TREES_SUPPORT' if et else 'FEW_VALUED' if few else 'FINITE_PAIR' if pair_ext else None
                        else:
                            expl = 'NRM1_III' if iii else 'FEW_VALUED' if few else 'FINITE_PAIR' if pair_ext else None
                        item = {'v': v, 'R': f'R{r}', 'wire': m, 'view': view, 'recipe': rec, 'minimum_set': on_min, 'distinct': n,
                                'explanation': expl}
                        equalities.append(item)
                        if expl is None:
                            unexplained.append(item)
    valid = not unexplained
    dec = json.loads((RES / 'claims' / plan['plan_id'] / 'DECISIONS.json').read_text())
    gv = dec['grid_validity']
    items.append(V.cmp_eq('G:grid_valid', valid, gv.get('effective')))
    items.append(V.cmp_eq('G:seed_checks_count', n_checks, dec.get('grid_validity_checks')))
    items.append(V.cmp_eq('G:explained_equality_count', len(equalities) - len(unexplained), gv['explained_equalities']['count']))
    key = lambda i: (i['v'], i['R'], i['wire'], i['view'], i['recipe'])
    mine = {key(i): i['explanation'] for i in equalities}
    theirs = {(i['v'], i['R'], i['wire'], i['view'], i['recipe']): i['explanation'] for i in gv['explained_equalities']['items']}
    items.append(V.cmp_eq('G:same_equality_set', sorted(map(list, mine)), sorted(map(list, theirs))))
    items.append(V.cmp_eq('G:same_explanations', {'|'.join(map(str, k)): x for k, x in sorted(mine.items())},
                          {'|'.join(map(str, k)): x for k, x in sorted(theirs.items())}))
    rd = dec.get('distinct_inputs') or {}
    for (m, v), d in sorted(distinct.items()):
        items.append(V.cmp_eq(f'G:distinct_inputs:{m}|V{v}', d, rd.get(f'{m}|V{v}')))
    out = {'schema': 'DRF1-verifier-check3c-v1', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
           'info': {'plan': a.plan, 'bank': a.bank, 'check2': a.check2, 'minimum_set': sorted(minimum)},
           'grid_valid': valid, 'n_seed_checks': n_checks, 'equalities': equalities, 'unexplained': unexplained,
           'nondeterministic_seed_invariant_recipes': nondet, 'overall': V.summarize(items), 'items': items,
           'summary': {'G': V.summarize(items)}}
    if a.out:
        V.dump(a.out, out)
    print(json.dumps(V.jsonable({'overall': out['overall'], 'grid_valid': valid, 'n_checks': n_checks, 'n_equalities': len(equalities),
                                 'unexplained': unexplained, 'nondeterministic': nondet}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
