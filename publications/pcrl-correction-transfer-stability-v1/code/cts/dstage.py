"""Development-recipe selection after the D0 / D1 diagnosis (RULES.md section 8; cts.rules.select_recipe). Label-free: it reads
the saved selections, realized outcomes, witness evaluations and reader-grid results; it opens no repair panel.

    python3 -m cts.dstage gates                       G1-G3 for every recipe (routing x arm x deployment x rule) on D0 and D1
                                                      -> results/<campaign>/diagnosis/RECIPE_GATES.json
    python3 -m cts.dstage bank --bank-id ID           reader bank: every recipe passing G1-G3 on both D partitions and its matched
                                                      CONT control (CONT at the strength the same rule selects on its own GUARD
                                                      costs, same routing and deployment), plus the witness
    python3 -m cts.dstage select --bank-id ID         select_recipe with the grid's pair benefits vs matched CONT
                                                      -> results/<campaign>/diagnosis/RECIPE_SELECTION.json
"""
from __future__ import annotations
import argparse
import json
import sys

from . import common as C
from . import rules as RL
from . import trackf as TF
from . import releases as REL

DPARTS = ('D0', 'D1')
ROUTINGS = ('REFIT', 'FROZEN')
ARMS = ('PAIR', 'LOC')
DEPS = ('full', 'ens')
RULES = ('SIMPLE', 'UNCERTAINTY', 'OLD-B50', 'OLD-B80')
DEPNAME = {'full': 'full', 'ens': 'ensemble'}


def _sel(P, did):
    return json.loads((TF.pdir(P) / 'SELECT' / f'{did}.json').read_text())


def _real(P, did):
    return json.loads((TF.pdir(P) / 'SELECT' / f'{did}__REALIZED.json').read_text())


def income_point_pass(P):
    ev = json.loads((TF.pdir(P) / 'EVAL' / 'WITNESS_EVAL_MS.json').read_text())
    return all(c['point_pass'] for c in ev['eval']['utility']['clauses'] if c['task'] == 'income')


def recipe_id(routing, arm, dep, rule):
    return f'{routing}_{arm}@{rule}|{dep}'


def gates():
    out, survivors = {}, []
    entry = 0
    for routing in ROUTINGS:
        for arm in ARMS:
            did = f'{routing}_{arm}'
            for dep in DEPS:
                for rule in RULES:
                    entry += 1
                    rid = recipe_id(routing, arm, dep, rule)
                    key = f'{rule}|{dep}'
                    rec = {'recipe': rid, 'routing': routing, 'arm': arm, 'deployment': DEPNAME[dep], 'rule': rule, 'entry': entry, 'D': {}}
                    for P in DPARTS:
                        s, r = _sel(P, did), _real(P, did)
                        sel = float(s['selected'][key])
                        oc = 'outcome' if dep == 'full' else 'outcome_complete'       # ens: complete release vs W's margins
                        ms = r['realized']['MS'][key][oc]; ck = r['realized']['CHECK'][key][oc]
                        base = s['baseline']
                        ratios = [(base[c]['slack'] - ms['conditions'][c]['cost']) / base[c]['se_used'] for c in RL.OCC_CONDITIONS
                                  if base[c].get('se_used')]
                        rec['D'][P] = {'selected': sel, 'reserve_crossing_check': bool(ck['reserve_crossing']),
                                       'reserve_crossing_ms': bool(ms['reserve_crossing']),
                                       'point_pass_ms': bool((not ms['point_cap_crossing']) and income_point_pass(P)),
                                       'ms_reserve_ratios': ratios, 'cont_grid': None,
                                       'cont_strength': float(_sel(P, f'{routing}_CONT')['selected'][key])}
                    g = {P: [n for n, ok in (('G1', rec['D'][P]['selected'] > 0), ('G2', not (rec['D'][P]['reserve_crossing_check'] or
                                                                                               rec['D'][P]['reserve_crossing_ms'])),
                                             ('G3', rec['D'][P]['point_pass_ms'])) if not ok] for P in DPARTS}
                    rec['failed_gates_G1_G3'] = g
                    rec['passes_G1_G3_both'] = not any(g.values())
                    out[rid] = rec
                    if rec['passes_G1_G3_both']:
                        survivors.append(rid)
    res = {'schema': 'CTS1-recipe-gates-v1', 'recipes': out, 'survivors_G1_G3': survivors, 'utc': C.now(),
           'note': 'G4 (reader grid vs matched CONT) and the score follow from the D reader bank; FROZEN routing results carry '
                   'A1_NOT_SUPPORTED (exchangeable-household assumption of UNCERTAINTY fails there)'}
    C.write_json(C.RESULTS / 'diagnosis' / 'RECIPE_GATES.json', res)
    print(json.dumps({'recipes': len(out), 'survivors': survivors,
                      'gate_failures_by_partition': {P: sorted({x for r in out.values() for x in r['failed_gates_G1_G3'][P]}) for P in DPARTS}}))
    return res


def bank(bank_id):
    g = json.loads((C.RESULTS / 'diagnosis' / 'RECIPE_GATES.json').read_text())
    cands, comps = [], ['W']
    for rid in g['survivors_G1_G3']:
        r = g['recipes'][rid]
        dep = 'full' if r['deployment'] == 'full' else 'ens'
        cands.append(rid)
        comps.append(f"{r['routing']}_CONT@{r['rule']}|{dep}")
    comps = sorted(set(comps), key=comps.index)
    if not cands:
        print(json.dumps({'bank': None, 'reason': 'no recipe passes G1-G3 on both D partitions'}))
        return None
    p, spec = REL.bank('D', bank_id, cands, comps, references=False, purpose='D-stage recipe selection: pair benefit vs matched CONT')
    print(json.dumps({'spec': str(p.relative_to(C.ROOT)), 'candidates': cands, 'comparators': comps}))
    return spec


def select(bank_id):
    g = json.loads((C.RESULTS / 'diagnosis' / 'RECIPE_GATES.json').read_text())
    grids = {}
    for P in DPARTS:
        s = TF.seed_of(P)
        f = C.RUNS / 'GRID' / bank_id / f'S{s}' / 'GRID.json'
        grids[P] = json.loads(f.read_text()) if f.is_file() else None
    cands = []
    for rid, r in g['recipes'].items():
        c = dict(r)
        dep = 'full' if r['deployment'] == 'full' else 'ens'
        cont = f"{r['routing']}_CONT@{r['rule']}|{dep}"
        for P in DPARTS:
            gr = grids[P]
            if gr and rid in gr['paired'] and cont in gr['paired'][rid]:
                pv = gr['paired'][rid][cont]['pair']
                c['D'][P]['cont_grid'] = {'cells': pv['cells'], 'cell_se': pv['cell_se']}
        cands.append(c)
    res = RL.select_recipe(cands)
    res.update({'schema': 'CTS1-recipe-selection-v1', 'bank_id': bank_id, 'utc': C.now(),
                'note': 'chosen before any repair CHECK panel label is opened; every frozen rule is still evaluated on the repair '
                        'panels; the selected recipe is never switched to the winner there'})
    C.write_json(C.RESULTS / 'diagnosis' / 'RECIPE_SELECTION.json', res)
    print(json.dumps({k: res.get(k) for k in ('outcome', 'selected', 'near_tie_set', 'best_score')}))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('gates')
    for c in ('bank', 'select'):
        x = sub.add_parser(c); x.add_argument('--bank-id', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'gates':
        gates()
    elif a.cmd == 'bank':
        bank(a.bank_id)
    else:
        select(a.bank_id)
    return 0


if __name__ == '__main__':
    sys.exit(main())
