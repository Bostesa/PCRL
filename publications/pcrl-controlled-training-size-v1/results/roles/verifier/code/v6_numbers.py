"""Verifier stage V6c helper (label-free): print the verifier's own numbers that the final texts quote (RESEARCH_DECISION.md,
ADVISOR_BRIEF.md, START_HERE.md), from the verifier's records only (V4 size, V2 trigger, V5 readers, V6 utility) plus the
production COND_UTILITY realized costs (replayed exactly by V6.COND_UTILITY) and EQUAL_RESERVE rows (replayed by V6.TABLES).

    python3 .../v6_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

R = V.RECORDS
m = json.loads((R / 'V4_SIZE_MINE__V1.json').read_text())['mine']
fits = m['fits']
out = {}
for f in ('50', '70'):
    for v in (0, 1):
        ks = [k for k in fits if k.startswith(f'A1|S{f}_') and k.endswith(f'|V{v}')]
        g0 = [fits[k]['conditions']['guard|sex0']['value'] for k in ks]
        nll = [fits[k]['conditions']['occupation|U0|nll_excess']['value'] for k in ks]
        out[f'A1 S{f} V{v}'] = {'men_excess_mean_1e3': round(1e3 * sum(g0) / 4, 3), 'men_excess_range_1e3': [round(1e3 * min(g0), 3), round(1e3 * max(g0), 3)],
                                'nll_excess_mean_1e3': round(1e3 * sum(nll) / 4, 3)}
for v in (0, 1):
    f = fits[f'A1|S100|V{v}']
    out[f'A1 S100 V{v}'] = {'men_excess_1e3': round(1e3 * f['conditions']['guard|sex0']['value'], 3),
                            'nll_excess_1e3': round(1e3 * f['conditions']['occupation|U0|nll_excess']['value'], 3),
                            'rho_men_1e3': round(1e3 * f['conditions']['guard|sex0']['rho'], 3)}
    for ctl in ('TR', 'TT'):
        fc = fits[f'{ctl}|S100|V{v}']
        out[f'{ctl} S100 V{v}'] = {'min_occ_ratio': round(fc['min_occ_ratio'], 3), 'income_min_ratio': round(fc['income_min_ratio'], 3),
                                   'whole_min': round(min(fc['min_occ_ratio'], fc['income_min_ratio']), 3)}
size = m['size']['A1']
for c in ('50->100',):
    for mm in ('acc_loss', 'nll', 'brier', 'men_brier', 'women_brier'):
        out[f'A1 {c} {mm}'] = {v: (round(1e3 * size[c][mm][v]['summary']['mean'], 2), round(1e3 * size[c][mm][v]['summary']['se_comb'], 2),
                                   size[c][mm][v]['summary']['label']) for v in ('0', '1')}
for c in ('50->70', '70->100'):
    out[f'A1 {c} labels'] = {mm: size[c][mm]['across'] for mm in ('acc_loss', 'nll', 'brier', 'men_brier', 'women_brier')}
vd = m['version_difference']
out['version_difference_men_max_abs_1e3'] = round(1e3 * max(abs(d['men_brier']['value']) for d in vd.values()), 3)
enc = m['encoding']
out['encoding'] = {mm: (round(1e3 * enc[mm]['summary']['mean'], 3), round(1e3 * enc[mm]['summary']['se_comb'], 3), enc[mm]['label'],
                        enc[mm]['share_of_50_100_improvement_v0']) for mm in enc}
for ctl in ('TR', 'TT'):
    lev = m['control'][ctl]['level']['100']
    out[f'level100 vs {ctl}'] = {mm: {v: (round(1e3 * lev[mm][v]['summary']['mean'], 2), round(1e3 * lev[mm][v]['summary']['se_comb'], 2),
                                         lev[mm][v]['summary']['label']) for v in ('0', '1')} for mm in ('acc_loss', 'nll', 'brier', 'men_brier', 'women_brier')}
    did = m['control'][ctl]['did']['50->100']
    out[f'did 50->100 {ctl}'] = {mm: {'across': did[mm]['across'], **{v: (round(1e3 * did[mm][v]['summary']['mean'], 2), did[mm][v]['summary']['label'])
                                                                       for v in ('0', '1')}} for mm in did}
labels = next(c for c in json.loads((R / 'V5_READERS__V1.json').read_text())['checks'] if c['id'].startswith('V5.READERS.LABELS'))['observed']
for key in ('LOC_SEL|W', 'LOC_SEL|CONT_AT_LOC', 'PAIR_SEL|W', 'PAIR_SEL|CONT_AT_PAIR', 'PAIR_SEL|LOC_SEL', 'CONT_AT_LOC|W'):
    x = labels[key]
    out[f'pair {key}'] = (round(1e3 * x['mean'], 2), round(1e3 * x['min'], 2), round(1e3 * x['max'], 2), x['stability'])
refs = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
cands = ('LOC_SEL', 'PAIR_SEL', 'CONT_AT_LOC', 'CONT_AT_PAIR', 'W')
vals = [labels[f'{c}|{r}']['mean'] for c in cands for r in refs]
out['candidates_vs_refs_mean_range_1e3'] = [round(1e3 * min(vals), 2), round(1e3 * max(vals), 2)]
out['candidates_vs_refs_labels'] = sorted({labels[f'{c}|{r}']['stability'] for c in cands for r in refs})
out['occupation_gate_fails_vs_every_ref'] = all(labels[f'{c}|{r}']['recipient_max']['occupation'] > 0.010 for c in cands for r in refs)
out['income_gate_failures'] = sorted({r for c in cands for r in refs if labels[f'{c}|{r}']['recipient_max']['income'] > 0.010})
cu = json.loads((V.RESULTS / 'conditional' / 'COND_UTILITY.json').read_text())
rc = {}
for arm in ('LOC', 'PAIR', 'CONT'):
    for v in ('0', '1'):
        q = cu['realized_minus_predicted'][arm][v]['0.3']
        rc[f'{arm} V{v}'] = {j: (round(1e3 * q[j]['realized'], 2), round(1e3 * q[j]['predicted'], 2)) for j in ('men_brier', 'nll')}
out['realized_vs_predicted_at_0.3_1e3'] = rc
cl = json.loads((V.RESULTS / 'conditional' / 'CLAIMS.json').read_text())
out['equal_reserve_whole_refs'] = {p['id']: round(p['min_ratio'], 3) for p in cl['equal_reserve']['whole_release_28']['points']}
print(json.dumps(out, indent=1, default=str))
