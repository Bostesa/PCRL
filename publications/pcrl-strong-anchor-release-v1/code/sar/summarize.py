"""Label-free summaries of saved records for the coordinator (no label read): registered-cell AUC levels of a bank's members and the
MODEL_SELECT utility of saved releases.

    python3 -m sar.summarize bank --bank SAR1_REFS
    python3 -m sar.summarize utility --release FB/FB_PFO [--release ...]
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np

from nrm import rules as NR
from . import common as C

VIEWS = ('pair', 'income', 'occupation')


def bank_levels(bank):
    cells, boot = {}, {}
    for v in C.VERSIONS:
        d = C.RUNS / 'GRID' / bank / f'S{v}'
        g = json.loads((d / 'GRID.json').read_text())
        cells.update(g['cells'])
        with np.load(d / 'BOOT.npz') as z:
            for k in z.files:
                boot[k] = z[k].copy()
    members = sorted({m for c in cells.values() for m in c})
    out = {}
    for m in members:
        row = {}
        for view in VIEWS:
            e = {'cells': {}, 'cell_se': {}}
            for c in NR.REGISTERED:
                if c in cells and m in cells[c]:
                    e['cells'][c] = float(cells[c][m][view]['score_auc'])
                    e['cell_se'][c] = float(np.std(boot[f'{c}__{m}__{view}'], ddof=1))
            a = NR.aggregate(e)
            row[view] = {'mean': a.get('mean'), 'se_agg': a.get('se_agg'),
                         'v': {str(v): a['per_version'][v]['mean'] for v in C.VERSIONS} if a.get('complete') else None}
        out[m] = row
    return out


def utility(spec):
    fam, cfg = spec.split('/')
    r = json.loads((C.RUNS / fam / cfg / 'EVAL_MS.json').read_text())
    out = {}
    for v, e in r['versions'].items():
        if e.get('missing'):
            out[v] = None
            continue
        g = {x['group']: round(x['value'], 5) for x in e['utility']['guards']}
        out[v] = {'pass28': e['point']['point_pass_12_plus_2'], 'max_over_cap': round(e['point']['max_value_over_cap'], 3),
                  'failed': e['point'].get('failed'), 'men_guard': g.get('sex0'), 'women_guard': g.get('sex1'),
                  'reserve_min': round(e['reserve_without_mc']['min_whole_ratio'], 2),
                  'binding': e['reserve_without_mc'].get('whole_binding'),
                  'costs': {k: {m: round(x['value'], 5) for m, x in (c.get('costs') or {}).items()
                                if m in ('total|nll', 'occupation|nll', 'income|nll', 'occupation|men_brier', 'occupation|women_brier')}
                            for k, c in (e.get('paired_costs') or {}).items()}}
    return out


def decisions(reading):
    """A compact view of a reading's decision record: labels, and per comparison the pair benefit with its se_agg and version means."""
    d = json.loads((C.RESULTS / 'claims' / reading / 'DECISIONS.json').read_text())
    dec = d['decisions']
    out = {'labels': {k: (dec[k].get('label') if isinstance(dec.get(k), dict) else dec.get(k))
                      for k in ('D1', 'UM', 'D2', 'D2R', 'D3', 'D4', 'D5', 'D6', 'outcome_class') if k in dec},
           'grid_effective': (d.get('grid_validity') or {}).get('effective')}

    def walk(o, path=''):
        if isinstance(o, dict):
            if 'mean' in o and 'se_agg' in o and isinstance(o.get('mean'), (int, float)):
                pv = o.get('per_version') or {}
                yield path, round(o['mean'], 4), round(o['se_agg'], 4), {str(k): round(v.get('mean'), 4) for k, v in pv.items()
                                                                      if isinstance(v, dict) and isinstance(v.get('mean'), (int, float))}
            for k, v in o.items():
                if k not in ('cells',):
                    yield from walk(v, f'{path}/{k}')
    rows = list(walk(dec))
    return out, rows, dec


def main_decisions(reading, full=False):
    out, rows, dec = decisions(reading)
    print(json.dumps(out, indent=1))
    for r in rows:
        if full or '/benefit/aggregate' in r[0] or r[0].endswith('/aggregate'):
            print(r)


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('bank'); b.add_argument('--bank', required=True)
    u = sub.add_parser('utility'); u.add_argument('--release', action='append', required=True)
    dd = sub.add_parser('decisions'); dd.add_argument('--reading', required=True); dd.add_argument('--full', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'decisions':
        main_decisions(a.reading, a.full)
        return 0
    if a.cmd == 'bank':
        for m, row in bank_levels(a.bank).items():
            print(f"{m:18s} " + '  '.join(f"{v} {row[v]['mean']:.4f} ({row[v]['se_agg']:.4f})" for v in VIEWS if row[v]['mean'] is not None))
    else:
        for s in a.release:
            print(s, json.dumps(utility(s), indent=None))
    return 0


if __name__ == '__main__':
    sys.exit(main())
