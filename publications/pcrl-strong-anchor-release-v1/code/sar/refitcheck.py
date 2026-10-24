"""Descriptive breakdown of the refit's reader-seed equalities (coordinator; label-free; reads saved grid outputs only).

    python3 -m sar.refitcheck --rob SAR1_ROB --out results/<campaign>/claims/STABILITY/REFIT_EQUALITIES.json

sar.claims applies sar.rules.refit_effectiveness (SAR1 rule: the extra-trees explanation with its support bound is registered
prospectively, RULES.md 9; there is no er1 argument). This module is descriptive and changes no label.
This module only describes the k-pair equalities that rule counted: by recipe, view, wire, version and repetition, and whether each
would meet the NRM1 (iii) explanation (a) + (b) + (c) that ER1 applies to the registered grid. It changes no label.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter

from nrm import rules as NR
from . import claims as CL
from . import common as C


def checks_of(grid):
    """The same k-pair checks as sar.claims.stability (k0 / k1 and k2 / k3 per recipe, view, wire, version, repetition)."""
    out = []
    for cell, cres in grid['cells'].items():
        if '@' in cell:
            continue
        v, rep, k = NR.parse_cell(cell)
        if k not in (0, 2):
            continue
        other = cell[:-1] + str(k + 1)
        for member, ent in cres.items():
            for view, e in ent.items():
                f0 = (e.get('extras') or {}).get('score_sha256') or {}
                f1 = (((grid['cells'].get(other) or {}).get(member) or {}).get(view, {}).get('extras') or {}).get('score_sha256') or {}
                for recipe in NR.SEED_DEPENDENT + NR.SEED_INVARIANT:
                    if recipe in f0 and recipe in f1:
                        out.append({'recipe': recipe, 'view': view, 'wire': member, 'v': v, 'R': rep, 'k_pair': [k, k + 1],
                                    'equal': f0[recipe] == f1[recipe]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rob', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    grid = CL.load_grid(a.rob)
    ch = checks_of(grid)
    dep = [c for c in ch if c['recipe'] in NR.SEED_DEPENDENT]
    eq = [c for c in dep if c['equal']]
    distinct = CL.distinct_inputs(a.rob)
    explained, unexplained = [], []
    for c in eq:
        same_block = [d for d in dep if (d['wire'], d['view'], d['v'], d['R'], tuple(d['k_pair'])) ==
                      (c['wire'], c['view'], c['v'], c['R'], tuple(c['k_pair']))]
        a_ok = any(not d['equal'] for d in same_block if d['recipe'] != c['recipe'])
        b_ok = any(not d['equal'] for d in dep if (d['wire'], d['recipe'], d['v'], d['R'], tuple(d['k_pair'])) ==
                   (c['wire'], c['recipe'], c['v'], c['R'], tuple(c['k_pair'])) and d['view'] != c['view'])
        c_ok = f"{c['wire']}|V{c['v']}" in distinct
        (explained if (a_ok and b_ok and c_ok) else unexplained).append({**c, 'a': a_ok, 'b': b_ok, 'c': c_ok})
    rec = {'schema': 'SAR1-refit-equalities-v1', 'utc': C.now(), 'bank': a.rob, 'labels_read': False,
           'rule_applied_by_claims': 'sar.rules.refit_effectiveness (SAR1, RULES.md 9); this record changes no label',
           'checks_seed_dependent': len(dep), 'equal_seed_dependent': len(eq),
           'by_recipe_view': dict(Counter(f"{c['recipe']}|{c['view']}" for c in eq)),
           'by_wire': dict(Counter(c['wire'] for c in eq)),
           'by_k_pair': dict(Counter(f"k{c['k_pair'][0]}k{c['k_pair'][1]}" for c in eq)),
           'distinct_inputs': distinct,
           'nrm1_iii_explanation': {'explained': len(explained), 'unexplained': len(unexplained),
                                    'unexplained_items': unexplained[:50]},
           'seed_dependent_recipes_differing': dict(Counter(f"{c['recipe']}|{c['view']}" for c in dep if not c['equal']))}
    C.write_json_x(a.out, rec)
    print(json.dumps({k: rec[k] for k in ('checks_seed_dependent', 'equal_seed_dependent', 'by_recipe_view', 'nrm1_iii_explanation')},
                     indent=1, default=str)[:3000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
