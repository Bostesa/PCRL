"""Reader-stability and screen tables from saved reader-grid results (label-free; cts.rules definitions).

    python3 -m cts.tables stability --bank-id ID [--pairs CAND:COMP,...] [--out STEM]

For a bank scored on both release seeds (GRID.json + BOOT.npz under runs/GRID/<bank>/S0 and S1), every requested (candidate,
comparator) pair gets, per view: the 8 cell differences (pair: comparator minus candidate AUC, positive favours the candidate;
income / occupation views: candidate minus comparator), household-bootstrap cell SEs, per-seed means, the source stability label
(cts.rules.stability_summary, z = 2) for the pair view, and for the single-output views the OLD per-cell recipient gate and the AGG1
screen side by side (exact aliases labelled). Winner near ties are counted from the grid's winner_changes.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys

from . import common as C
from . import rules as RL

VIEWS = ('pair', 'income', 'occupation')


def _grid(bank_id):
    out = {}
    for s in (0, 1):
        f = C.RUNS / 'GRID' / bank_id / f'S{s}' / 'GRID.json'
        if f.is_file():
            out[s] = json.loads(f.read_text())
    return out


def stability(bank_id, pairs=None, stem=None):
    g = _grid(bank_id)
    if not g:
        raise FileNotFoundError(f'no grid results for {bank_id}')
    seeds = sorted(g)
    anyg = g[seeds[0]]
    if pairs is None:
        pairs = [(c, k) for c in anyg['paired'] for k in anyg['paired'][c]]
    rows, recs = [], {}
    phi = RL.load_rules()['screens']['aggregate_paired_rule']['phi_prior']
    for cand, comp in pairs:
        for view in VIEWS:
            cells, ses = {}, {}
            for s in seeds:
                p = g[s]['paired'].get(cand, {}).get(comp, {}).get(view)
                if p is None:
                    continue
                for cell, v in p['cells'].items():
                    if '@' in cell:
                        continue
                    cells[cell] = v; ses[cell] = p['cell_se'][cell]
            if not cells:
                continue
            rec = {'candidate': cand, 'comparator': comp, 'view': view, 'n_cells': len(cells),
                   'mean': sum(cells.values()) / len(cells),
                   'per_seed_mean': {s: sum(v for c, v in cells.items() if c.startswith(f'S{s}')) /
                                     max(1, sum(1 for c in cells if c.startswith(f'S{s}'))) for s in seeds}}
            if view == 'pair' and len(seeds) == 2 and len(cells) == 8:
                st = RL.stability_summary(cells, ses)
                rec.update({'stability': st['label'], 'sd_total': st['sd_total'], 'min_lower': st['min_lower']})
            elif view != 'pair' and len(seeds) == 2 and len(cells) == 8:
                gate = RL.per_cell_gate(cells)
                alias = all(v == 0.0 for v in cells.values())
                agg = RL.aggregate_paired_screen(cells, ses, phi[view], alias=alias)
                rec.update({'old_gate': gate['label'], 'old_gate_max': gate.get('max'), 'agg1': agg['label'], 'agg1_U': agg.get('U')})
            recs[f'{cand}|{comp}|{view}'] = rec
            rows.append([cand, comp, view, len(cells), rec['mean'], rec['per_seed_mean'].get(0), rec['per_seed_mean'].get(1),
                         rec.get('stability'), rec.get('sd_total'), rec.get('min_lower'), rec.get('old_gate'), rec.get('old_gate_max'),
                         rec.get('agg1'), rec.get('agg1_U')])
    stem = stem or f'READER_STABILITY_{bank_id}'
    out = C.RESULTS / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f'{stem}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['candidate', 'comparator', 'view', 'n_cells', 'mean', 'seed0_mean', 'seed1_mean', 'stability', 'sd_total',
                    'min_lower', 'old_gate', 'old_gate_max', 'agg1', 'agg1_U'])
        w.writerows(rows)
    nt = {s: {k: v for k, v in g[s].get('winner_changes', {}).items() if v.get('near_ties')} for s in seeds}
    C.write_json(out / f'{stem}.json', {'schema': 'CTS1-reader-stability-v1', 'bank_id': bank_id, 'seeds': seeds, 'records': recs,
                                        'winner_near_ties': nt, 'utc': C.now(),
                                        'note': 'pair benefit = comparator AUC minus candidate AUC (positive favours the candidate); '
                                                'single-output views = candidate minus comparator (recipient conditions)'})
    print(json.dumps({'stem': stem, 'records': len(recs)}))
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    x = sub.add_parser('stability'); x.add_argument('--bank-id', required=True); x.add_argument('--pairs', default=None)
    x.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    pairs = [tuple(p.split(':', 1)) for p in a.pairs.split(',')] if a.pairs else None
    stability(a.bank_id, pairs, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
