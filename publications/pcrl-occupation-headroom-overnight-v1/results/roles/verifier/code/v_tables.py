"""Derived-table consistency (light; aggregate numbers only): UTILITY_FRONTIER.csv against the verifier's own utility tables and
per-seed reserves; GRID_RECOVERY.csv against GRID.json winner SCORE AUCs; READER_STABILITY.csv (when present) against the
verifier's own stability labels (record VER_GRID_PAIRED.json)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402

RULES = V.load_rules()


def my_tables():
    out = {}
    refs = json.loads((V.RECORDS / 'VER_UTILITY_REFS.json').read_text())
    for name, per in refs['tables'].items():
        for s, t in per.items():
            out[(name, int(s))] = t
    db = json.loads((V.RECORDS / 'VER_UTILITY_CONFIGS.json').read_text())
    for key, e in db['entries'].items():
        fam, cfg, s, anchor = key.split('/')
        out[(f'{fam}/{cfg}@{anchor}', int(s[1:]))] = e['table']
    return out


def frontier():
    T = my_tables()
    p = V.RES / 'UTILITY_FRONTIER.csv'
    rows = list(csv.DictReader(open(p)))
    res = {'n_rows': len(rows), 'n_checked': 0, 'not_found': [], 'mismatches': []}
    for r in rows:
        key = (r['configuration'], int(r['seed']))
        if key not in T:
            res['not_found'].append(f'{key[0]}/S{key[1]}')
            continue
        t = T[key]
        res['n_checked'] += 1
        for task, pre in (('occupation', 'occ'), ('income', 'inc')):
            for ref in ('U0', 'Ucal'):
                for m in ('accuracy_loss', 'nll_excess', 'brier_excess'):
                    c = t[f'{task}|{ref}|{m}']
                    for col, val in ((f'{pre}_{m}_{ref}', c['value']), (f'{pre}_{m}_{ref}_se', c['se'])):
                        if r.get(col, '') != '' and abs(float(r[col]) - val) > 1.5e-6:
                            res['mismatches'].append(f'{key}: {col} {r[col]} vs {val:.8f}')
        for col, cond in (('men_guard', 'guard|sex0'), ('women_guard', 'guard|sex1')):
            for c2, v2 in ((col, t[cond]['value']), (col + '_se', t[cond]['se'])):
                if r.get(c2, '') != '' and abs(float(r[c2]) - v2) > 1.5e-6:
                    res['mismatches'].append(f'{key}: {c2} {r[c2]} vs {v2:.8f}')
        rs = V.reserve_seed(t, key[1], RULES, 5306)
        mn = min(v['ratio'] for v in rs.values())
        mo = min(v['ratio'] for k, v in rs.items() if k in V.OCC_SIDE)
        mi = min(v['ratio'] for k, v in rs.items() if k not in V.OCC_SIDE)
        for col, val in (('min_ratio', mn), ('min_occupation_side_ratio', mo), ('min_income_ratio', mi)):
            if r.get(col, '') != '' and abs(float(r[col]) - val) > 1.5e-3:
                res['mismatches'].append(f'{key}: {col} {r[col]} vs {val:.4f}')
    return res


def recovery():
    p = V.RES / 'GRID_RECOVERY.csv'
    rows = list(csv.DictReader(open(p)))
    res = {'n_rows': len(rows), 'n_checked': 0, 'mismatches': []}
    grids = {}
    for g in V.RUNS.glob('GRID/*/S*/GRID.json'):
        grids[(g.parts[-3], int(g.parts[-2][1:]))] = json.loads(g.read_text())
    for r in rows:
        g = grids.get((r['bank'], int(r['seed'])))
        if g is None:
            res['mismatches'].append(f"{r['bank']}/{r['seed']}: grid not found")
            continue
        full = [c for c in g['cells'] if '@' not in c]
        res['n_checked'] += 1
        for view in ('pair', 'income', 'occupation'):
            vals = [g['cells'][c][r['release']][view]['score_auc'] for c in full]
            for col, val in ((f'{view}_mean', np.mean(vals)), (f'{view}_min', np.min(vals)), (f'{view}_max', np.max(vals))):
                if abs(float(r[col]) - float(val)) > 1.5e-6:
                    res['mismatches'].append(f"{r['bank']}/{r['seed']}/{r['release']}: {col} {r[col]} vs {val:.7f}")
            wins = ';'.join(sorted({g['cells'][c][r['release']][view]['winner'] for c in full}))
            if r[f'{view}_winners'] != wins:
                res['mismatches'].append(f"{r['bank']}/{r['seed']}/{r['release']}: {view} winners {r[f'{view}_winners']} vs {wins}")
            nt = sum(bool(g['cells'][c][r['release']][view]['near_tie']) for c in full)
            if int(r[f'{view}_near_ties']) != nt:
                res['mismatches'].append(f"{r['bank']}/{r['seed']}/{r['release']}: {view} near ties {r[f'{view}_near_ties']} vs {nt}")
        for frac in ('0.25', '0.5'):
            col = f'pair_mean_at_{frac}'
            cells = [c for c in g['cells'] if c.endswith('@' + frac)]
            if cells and r.get(col, '') != '':
                val = np.mean([g['cells'][c][r['release']]['pair']['score_auc'] for c in cells])
                if abs(float(r[col]) - float(val)) > 1.5e-6:
                    res['mismatches'].append(f"{r['bank']}/{r['seed']}/{r['release']}: {col} {r[col]} vs {val:.7f}")
    return res


def stability():
    """READER_STABILITY.csv against the verifier's own labels. Change 07:58Z (records/CHECK_CHANGES.json): each row is compared within
    ITS summary's banks (column 'summary' -> runs/GRID/<summary>/SUMMARY.json 'banks'), INCOMPLETE rows are compared as labels and
    present cells instead of being reported as missing; complete rows also compare min, max, per-seed means, the 8 pair cells and the
    recipient / target labels (6-decimal table rounding)."""
    import v_grid as G
    p = V.RES / 'READER_STABILITY.csv'
    if not p.is_file():
        return {'status': 'NOT_PRESENT'}
    rows = list(csv.DictReader(open(p)))
    per_bank, per_bank_lc, _ = G.collect_cells()
    scopes = {}
    res = {'n_rows': len(rows), 'n_checked': 0, 'n_incomplete_rows': 0, 'mismatches': [], 'columns': list(rows[0].keys()) if rows else []}

    def near(a, b):
        return abs(float(a) - float(b)) <= 1.5e-6

    for r in rows:
        summ = r.get('summary')
        if summ not in scopes:
            sp = V.RUNS / 'GRID' / str(summ) / 'SUMMARY.json'
            banks = json.loads(sp.read_text()).get('banks') if sp.is_file() else [summ]
            scopes[summ] = G.build_labels(per_bank, per_bank_lc, list(banks or [summ]))
        mine = scopes[summ]
        cand, comp = r.get('candidate'), r.get('comparator')
        m = mine.get(cand, {}).get(comp)
        tag = f'[{summ}] {cand} vs {comp}'
        if m is None:
            res['mismatches'].append(f'{tag}: not in my labels for this summary scope')
            continue
        res['n_checked'] += 1
        if r['label'] != m['benefit']:
            res['mismatches'].append(f"{tag}: label {r['label']} vs mine {m['benefit']}")
            continue
        if m['benefit'] == 'INCOMPLETE':
            res['n_incomplete_rows'] += 1
            present = [c for c in G.CELLS if r.get(f'pair_{c}', '') != '']
            if sorted(present) != sorted(m.get('cells_present', [])):
                res['mismatches'].append(f'{tag}: present cells differ')
            continue
        checks = [('mean', m['mean']), ('min', m['min']), ('sd_total', m['sd_total']),
                  ('mean_s0', m['per_seed_mean'][0]), ('mean_s1', m['per_seed_mean'][1]),
                  ('lower_s0', m['lower'][0]), ('lower_s1', m['lower'][1]),
                  ('income_recipient_max', m['recipient_income_max']), ('occupation_recipient_max', m['recipient_occupation_max'])]
        checks += [(f'pair_{c}', m['pair_cells'][c]) for c in G.CELLS]
        for col, val in checks:
            if r.get(col, '') != '' and not near(r[col], val):
                res['mismatches'].append(f'{tag}: {col} {r[col]} vs {val:.7f}')
        for col, val in (('income_recipient_label', m['recipient_income']), ('occupation_recipient_label', m['recipient_occupation']),
                         ('target_label', m.get('target') or G.target_label(m['pair_cells'], m['pair_cell_se']))):
            if r.get(col, '') != '' and r[col] != val:
                res['mismatches'].append(f'{tag}: {col} {r[col]} vs {val}')
    return res


def main():
    rec = {'check': 'VER_TABLES', 'utc': V.utc(), 'utility_frontier': frontier(), 'grid_recovery': recovery(), 'reader_stability': stability()}
    ok = all(not v.get('mismatches') for v in (rec['utility_frontier'], rec['grid_recovery'], rec['reader_stability']))
    rec['status'] = 'PASS' if ok else 'ATTENTION'
    rec['summary'] = {k: {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items() if kk != 'columns'}
                      for k, v in rec.items() if isinstance(v, dict)}
    V.write_record('VER_TABLES', rec)
    print(json.dumps({'status': rec['status'], 'summary': rec['summary']}, indent=1))
    for k in ('utility_frontier', 'grid_recovery', 'reader_stability'):
        for m in (rec[k].get('mismatches') or [])[:10]:
            print('MISMATCH', k, m)
        for m in (rec[k].get('not_found') or [])[:10]:
            print('NOT_FOUND', k, m)


if __name__ == '__main__':
    main()
