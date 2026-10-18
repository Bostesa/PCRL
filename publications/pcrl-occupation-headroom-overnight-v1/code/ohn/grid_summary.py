"""Merge the two release seeds' development reader-grid results into the registered 8-cell structure and apply the frozen
stability rule (ohn.rules.stability), with learning-curve cells kept separate (diagnostic only).

    python -m ohn.grid_summary --bank BANK_ID [--bank BANK_ID ...] --out results/<campaign>/grid/<NAME>.json

Each bank directory is runs/GRID/<BANK_ID>/S<seed>/GRID.json (written by `ohn.devreaders run`). For every (candidate, comparator)
pair present in both seeds: the 8 full-size cells (S0R1k0 ... S1R2k1) of the pair benefit (comparator - candidate) and of the
income / occupation recipient differences (candidate - comparator) with their household-bootstrap SEs, the stability label, the
recipient point gate, the material-target outcome against NEURAL_RAW / H_LJ, winner changes and validation near ties per release,
and the learning-curve means (1/4, 1/2, full FIT) where present.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from . import common as C
from . import rules as RU

CELLS = tuple(f'S{s}{r}k{k}' for s in C.SEEDS for r in ('R1', 'R2') for k in (0, 1))


def load_bank(bank_id):
    out = {}
    for s in C.SEEDS:
        p = C.RUNS / 'GRID' / bank_id / f'S{s}' / 'GRID.json'
        if p.is_file():
            out[s] = json.loads(p.read_text())
    return out


def merge(banks):
    """{(cand, comp): {'pair'|'income'|'occupation': {'cells', 'cell_se'}}, ...} over the 8 full cells, plus learning curves."""
    entries, lc, winners = {}, {}, {}
    for bank_id in banks:
        per_seed = load_bank(bank_id)
        for s, g in per_seed.items():
            for cand, comps in g['paired'].items():
                for comp, views in comps.items():
                    e = entries.setdefault((cand, comp), {v: {'cells': {}, 'cell_se': {}} for v in ('pair', 'income', 'occupation')})
                    for v in ('pair', 'income', 'occupation'):
                        for cell, val in views[v]['cells'].items():
                            if '@' in cell:
                                frac = cell.split('@')[1]
                                lc.setdefault((cand, comp), {}).setdefault(frac, []).append(val) if v == 'pair' else None
                                continue
                            e[v]['cells'][cell] = val; e[v]['cell_se'][cell] = views[v]['cell_se'][cell]
                        if v == 'pair':
                            full = [x for c, x in views[v]['cells'].items() if '@' not in c]
                            lc.setdefault((cand, comp), {}).setdefault('1.0', []).extend(full)
            for k, w in g['winner_changes'].items():
                winners.setdefault(k, {'distinct_winners': set(), 'n_cells': 0, 'near_ties': 0})
                winners[k]['distinct_winners'] |= set(w['distinct_winners']); winners[k]['n_cells'] += w['n_cells']
                winners[k]['near_ties'] += w['near_ties']
    return entries, lc, winners


def summarize(banks, out):
    entries, lc, winners = merge(banks)
    rules = RU.load_rules()
    res = {}
    for (cand, comp), e in sorted(entries.items()):
        complete = all(c in e['pair']['cells'] for c in CELLS)
        st = RU.stability(e, rules, comparator=comp) if complete else {'benefit': {'label': 'INCOMPLETE'}, 'cells_present': sorted(e['pair']['cells'])}
        curve = {f: float(np.mean(v)) for f, v in lc.get((cand, comp), {}).items() if v}
        res.setdefault(cand, {})[comp] = {'stability': st, 'pair_cells': e['pair']['cells'], 'pair_cell_se': e['pair']['cell_se'],
                                         'income_cells': e['income']['cells'], 'occupation_cells': e['occupation']['cells'],
                                         'learning_curve_mean_pair_benefit': curve,
                                         'learning_curve_flag': RU.learning_curve_flag({float(k): v for k, v in curve.items()}, rules)
                                         if {'0.25', '1.0'} <= set(curve) else None}
    rec = {'schema': 'OHN1-grid-summary-v1', 'utc': C.now(), 'banks': list(banks), 'cells': list(CELLS), 'entries': res,
           'winner_changes': {k: {'distinct_winners': sorted(v['distinct_winners']), 'n_cells': v['n_cells'], 'near_ties': v['near_ties']}
                              for k, v in winners.items()},
           'note': 'adaptive development evidence on MODEL_SELECT; reader-relative; people and reader fits are counted separately'}
    C.write_json(out, rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', action='append', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    rec = summarize(a.bank, Path(a.out))
    brief = {cand: {comp: (v['stability']['benefit']['label'], round(float(np.mean(list(v['pair_cells'].values()))), 4) if v['pair_cells'] else None)
                    for comp, v in comps.items()} for cand, comps in rec['entries'].items()}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    main()
