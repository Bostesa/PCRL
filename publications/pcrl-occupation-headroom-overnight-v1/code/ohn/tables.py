"""Comparison tables of the overnight occupation-headroom campaign (no label read; stored evaluations and grid results only).

    python -m ohn.tables

-> results/<campaign>/UTILITY_FRONTIER.csv   one row per scored configuration x seed (and every frozen reference): occupation
   accuracy / NLL / Brier excess against U0 and Ucal, both SEX-group guards with SEs, the whole-release reserve ratios (minimum,
   occupation side, income), point pass, differences against the equally exposed task-only references, cross-fitting checks;
-> results/<campaign>/READER_STABILITY.csv   one row per (candidate, comparator) of every grid summary under runs/GRID/*/SUMMARY.json:
   the 8 SCORE-cell pair benefits, mean, min, per-seed means, sd_total, the frozen stability label, recipient maxima and labels,
   the material-target label and the learning-curve means where present;
-> results/<campaign>/GRID_RECOVERY.csv       one row per release x seed x grid bank: mean / min / max SCORE AUC of the pair, income
   and occupation winners over the full-size cells, winner identities and validation near ties.
"""
from __future__ import annotations
import csv
import json

import numpy as np

from . import common as C
from . import devrun as D


def _w(path, rows):
    if not rows:
        return 0
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    return len(rows)


def _r(x, k=6):
    return '' if x is None else round(float(x), k)


def utility_row(name, seed, anchor, m, rec=None):
    u = m['utility']
    by = {(c['task'], c['reference'], c['metric']): c for c in u['clauses']}
    g = {x['group']: x for x in u['guards']}
    rs = m.get('reserve') or {}
    row = {'configuration': name, 'seed': seed, 'anchor': anchor, 'kind': (rec or {}).get('kind', 'reference'),
           'point_pass_12': u['point_pass_12'], 'guard_pass': u['guard_pass']}
    for t in ('occupation', 'income'):
        for r in ('U0', 'Ucal'):
            for mm in ('accuracy_loss', 'nll_excess', 'brier_excess'):
                c = by[(t, r, mm)]
                row[f'{t[:3]}_{mm}_{r}'] = _r(c['value']); row[f'{t[:3]}_{mm}_{r}_se'] = _r(c['se'])
    row.update({'men_guard': _r(g['sex0']['value']), 'men_guard_se': _r(g['sex0']['se']), 'women_guard': _r(g['sex1']['value']),
                'women_guard_se': _r(g['sex1']['se']), 'min_ratio': _r(rs.get('min_ratio'), 3),
                'min_occupation_side_ratio': _r(rs.get('min_occupation_side_ratio'), 3), 'min_income_ratio': _r(rs.get('min_income_ratio'), 3),
                'binding_condition': rs.get('binding_condition', '')})
    v = m.get('occupation_versus_task_only_references') or {}
    for ref in ('T_TREE', 'TASK_REFIT', 'JOINT'):
        for mm in ('nll_diff', 'brier_diff', 'accuracy_loss'):
            k = f'{ref}|{mm}'
            if k in v:
                row[f'occ_{mm}_vs_{ref}'] = _r(v[k]['value']); row[f'occ_{mm}_vs_{ref}_se'] = _r(v[k]['se'])
    cf = m.get('crossfit')
    if cf:
        row['oof_nll_abs'] = _r(cf['oof_absolute']['nll']); row['oof_brier_abs'] = _r(cf['oof_absolute']['brier'])
        d = cf['deployed_minus_foldavg_on_MODEL_SELECT']
        row['deployed_minus_foldavg_nll'] = _r(d['nll_diff']['value']); row['deployed_minus_foldavg_nll_se'] = _r(d['nll_diff']['se'])
        row['deployed_minus_foldavg_brier'] = _r(d['brier_diff']['value'])
    if rec:
        for k in ('n_model_fits', 'n_calibrator_fits', 'n_crossfit_fits', 'fit_cpu_s', 'python'):
            row[k] = rec.get(k, '')
    return row


def utility_rows():
    rows = []
    for s in C.SEEDS:
        p = C.RUNS / 'REF' / f'S{s}' / 'EVAL_MS_REFERENCES.json'
        if p.is_file():
            for name, m in json.loads(p.read_text()).items():
                rows.append(utility_row(name, s, '', m))
    for fam_dir in sorted(p for p in C.RUNS.iterdir() if p.is_dir() and p.name not in ('REF', 'GRID', 'DEPLOY', 'DEPLOY_TEST', 'devgrid_cache')):
        for cdir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            for s in C.SEEDS:
                d = cdir / f'S{s}'
                rp = d / 'OCC.json' if (d / 'OCC.json').is_file() else d / 'RELEASE.json'   # RELEASE.json: income-branch complete release
                if not rp.is_file():
                    continue
                rec = json.loads(rp.read_text())
                for ev in sorted(d.glob('EVAL_MS_*.json')):
                    anchor = ev.stem.split('_')[-1]
                    rows.append(utility_row(f'{fam_dir.name}/{cdir.name}@{anchor}', s, anchor, json.loads(ev.read_text()), rec))
    return rows


def stability_rows():
    rows = []
    for summ in sorted((C.RUNS / 'GRID').glob('*/SUMMARY.json')):
        g = json.loads(summ.read_text())
        for cand, comps in g['entries'].items():
            for comp, v in comps.items():
                st = v['stability']; b = st.get('benefit', {})
                row = {'summary': summ.parent.name, 'candidate': cand, 'comparator': comp, 'label': b.get('label'),
                       'mean': _r(b.get('mean')), 'min': _r(b.get('min')), 'max': _r(b.get('max')), 'sd_total': _r(b.get('sd_total')),
                       'cells_positive': b.get('cells_above')}
                for s in C.SEEDS:
                    row[f'mean_s{s}'] = _r((b.get('per_seed_mean') or {}).get(s, (b.get('per_seed_mean') or {}).get(str(s))))
                    row[f'lower_s{s}'] = _r((b.get('lower_per_seed') or {}).get(s, (b.get('lower_per_seed') or {}).get(str(s))))
                for cell, val in sorted(v.get('pair_cells', {}).items()):
                    row[f'pair_{cell}'] = _r(val)
                rec = st.get('recipient') or {}
                for view in ('income', 'occupation'):
                    rv = rec.get(view) or {}
                    row[f'{view}_recipient_max'] = _r(rv.get('max')); row[f'{view}_recipient_label'] = rv.get('label')
                row['target_label'] = (st.get('target') or {}).get('label')
                for f, val in sorted((v.get('learning_curve_mean_pair_benefit') or {}).items()):
                    row[f'lc_mean_pair_benefit_{f}'] = _r(val)
                rows.append(row)
    return rows


def recovery_rows():
    rows = []
    for gp in sorted((C.RUNS / 'GRID').glob('*/S*/GRID.json')):
        g = json.loads(gp.read_text())
        cells = [c for c in g['cells'] if '@' not in c]
        for name in g['spec']['members']:
            row = {'bank': g['bank_id'], 'seed': g['seed'], 'release': name}
            for view in ('pair', 'income', 'occupation'):
                vals = [g['cells'][c][name][view]['score_auc'] for c in cells]
                row[f'{view}_mean'] = _r(np.mean(vals)); row[f'{view}_min'] = _r(np.min(vals)); row[f'{view}_max'] = _r(np.max(vals))
                row[f'{view}_winners'] = ';'.join(sorted({g['cells'][c][name][view]['winner'] for c in cells}))
                row[f'{view}_near_ties'] = sum(bool(g['cells'][c][name][view]['near_tie']) for c in cells)
            for c in g['cells']:
                if '@' in c:
                    f = c.split('@')[1]
                    row.setdefault(f'pair_mean_at_{f}', [])
                    row[f'pair_mean_at_{f}'].append(g['cells'][c][name]['pair']['score_auc'])
            for k in [k for k in row if k.startswith('pair_mean_at_')]:
                row[k] = _r(np.mean(row[k]))
            rows.append(row)
    return rows


def main():
    n1 = _w(C.RESULTS / 'UTILITY_FRONTIER.csv', utility_rows())
    n2 = _w(C.RESULTS / 'READER_STABILITY.csv', stability_rows())
    n3 = _w(C.RESULTS / 'GRID_RECOVERY.csv', recovery_rows())
    print(json.dumps({'utility_rows': n1, 'stability_rows': n2, 'recovery_rows': n3}))


if __name__ == '__main__':
    main()
