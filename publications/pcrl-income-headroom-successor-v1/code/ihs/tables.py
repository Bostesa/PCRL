"""Unit IHS_TABLES: comparison tables for families, revisions and references from the ledger and the private EVAL_MS records
(clause- and proxy-level numbers only; no person-level value).

Outputs (results/<campaign>/):
- COMPARISON_TABLE.csv: every evaluated configuration and reference, both seeds where run: the 12 income-side clause values
  with SEs and margins in SE units, worst occupation clause, guards with SEs, reserve status, development proxies (mean and
  the four-cell range), paired benefits and recipient differences against NEURAL_RAW and H_LJ (mean and minimum over cells).
- FRONTIER.csv: one row per configuration-seed: income summed-Brier excess vs Ucal (worst of U0 / Ucal), its margin in SE
  units, and the pair proxy mean, for the utility-recovery frontier.
"""
from __future__ import annotations
import csv
import json

from . import common as C
from . import devrun as D
from . import ledger as L


def eval_paths():
    for p in sorted(C.RUNS.glob('*/*/S*/EVAL_MS.json')):
        fam, cfg, s = p.parts[-4], p.parts[-3], int(p.parts[-2][1:])
        yield fam, cfg, s, p


def row_of(fam, cfg, s, m, cmp=None):
    u = m['utility']; cl = {(c['task'], c['reference'], c['metric']): c for c in u['clauses']}
    g = {x['group']: x for x in u['guards']}
    out = {'family': fam, 'config_id': cfg, 'seed': s}
    for r in ('U0', 'Ucal'):
        for mm in ('accuracy_loss', 'nll_excess', 'brier_excess'):
            c = cl[('income', r, mm)]
            out[f'inc_{mm}_{r}'] = c['value']; out[f'inc_{mm}_{r}_se'] = c['se']; out[f'inc_{mm}_{r}_margin_se'] = c['margin_se_units']
    occ = [c for c in u['clauses'] if c['task'] == 'occupation']
    w = max(occ, key=lambda c: c['normalized'])
    out.update({'occ_worst': f"{w['reference']}|{w['metric']}", 'occ_worst_value': w['value'], 'occ_worst_margin_se': w['margin_se_units'],
                'occ_min_margin_se': min(c['margin_se_units'] for c in occ),
                'guard_men': g['sex0']['value'], 'guard_men_se': g['sex0']['se'], 'guard_men_margin_se': g['sex0']['margin_se_units'],
                'guard_women': g['sex1']['value'], 'guard_women_se': g['sex1']['se'],
                'point_pass_12': u['point_pass_12'], 'guard_pass': u['guard_pass'],
                'reserve_status': (m.get('reserve') or {}).get('status', '')})
    if 'proxy' in m:
        sm = m['proxy']['summary']
        for v in ('pair', 'income', 'occupation'):
            out[f'proxy_{v}_mean'] = sm[v]['mean']; out[f'proxy_{v}_min'] = sm[v]['min']; out[f'proxy_{v}_max'] = sm[v]['max']
    cmp = cmp if cmp is not None else m.get('comparisons')
    if cmp:
        for ref in ('NEURAL_RAW', 'H_LJ'):
            if ref in cmp:
                out[f'benefit_vs_{ref}_mean'] = cmp[ref]['pair']['mean']; out[f'benefit_vs_{ref}_min'] = cmp[ref]['pair']['min']
                out[f'inc_recipient_vs_{ref}_max'] = cmp[ref]['income']['max']; out[f'occ_recipient_vs_{ref}_max'] = cmp[ref]['occupation']['max']
    return out


def run():
    rows = []
    for s in C.SEEDS:
        res = D.reference_eval(s)
        for name in D.REFERENCE_SET:
            m = res[name]
            rows.append(row_of('REF', name, s, m, D.compare(m, res) if 'proxy' in m else None))
    for fam, cfg, s, p in eval_paths():
        rows.append(row_of(fam, cfg, s, json.loads(p.read_text())))
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('family', 'config_id', 'seed'), k))
    with open(C.RESULTS / 'COMPARISON_TABLE.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})
    with open(C.RESULTS / 'FRONTIER.csv', 'w', newline='') as f:
        cols2 = ['family', 'config_id', 'seed', 'income_brier_excess_worst_ref', 'income_brier_margin_se', 'income_nll_excess_worst_ref',
                 'income_nll_margin_se', 'proxy_pair_mean', 'proxy_pair_min', 'proxy_pair_max', 'guard_men', 'reserve_status']
        w = csv.DictWriter(f, fieldnames=cols2); w.writeheader()
        for r in rows:
            if 'proxy_pair_mean' not in r:
                continue
            b = max(('U0', 'Ucal'), key=lambda x: r[f'inc_brier_excess_{x}'])
            n = max(('U0', 'Ucal'), key=lambda x: r[f'inc_nll_excess_{x}'])
            w.writerow({'family': r['family'], 'config_id': r['config_id'], 'seed': r['seed'],
                        'income_brier_excess_worst_ref': r[f'inc_brier_excess_{b}'], 'income_brier_margin_se': r[f'inc_brier_excess_{b}_margin_se'],
                        'income_nll_excess_worst_ref': r[f'inc_nll_excess_{n}'], 'income_nll_margin_se': r[f'inc_nll_excess_{n}_margin_se'],
                        'proxy_pair_mean': r['proxy_pair_mean'], 'proxy_pair_min': r['proxy_pair_min'], 'proxy_pair_max': r['proxy_pair_max'],
                        'guard_men': r['guard_men'], 'reserve_status': r['reserve_status']})
    print(json.dumps({'rows': len(rows)}))


if __name__ == '__main__':
    run()
