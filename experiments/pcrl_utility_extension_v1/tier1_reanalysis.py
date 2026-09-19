"""Tier 1a: exploratory UTILITY-FIRST reanalysis of the completed 372-slot study.

NEW exploratory criterion, not a retroactive success rule. The original verdict of
`pcrl_competitive_method_v1` (no competitive tradeoff under its registered criterion)
is preserved unchanged and quoted alongside.

Input: the completed study's stored paired-bootstrap estimates and standard errors
(`INTERVALS_X.csv`, families `grid_vs_J` and `grid_vs_leace_A0`: every one of the 124
searched configurations, seed-averaged, 5 endpoints x 2 weightings). Nothing is refitted.

The old family-adjusted critical value (m = 3900) belongs to the old search family and is
NOT reused. The new search family is every searched configuration x {J, leace_A0} x
5 endpoints x 2 weightings; the studentized Bonferroni critical value is recomputed for
that realised size (the same adjustment form as the original study, PROTOCOL section 5).

Utility-first criterion (declared here, before this table was computed):
  vs J:        residence loss difference <= -0.003 AND adjusted upper bound < 0
               (improvement established); for each of the four sensitive endpoints the
               ONE-sided adjusted upper bound on (candidate - J) recovery <= +0.001 nats;
  vs leace_A0: the same residence improvement AND the same sensitive bound
               (comparable-or-better protection than the strong external comparator);
  both weightings. Zero-margin results are reported too.
Selection note: this is a search over 124 configurations whose outcomes were already
seen; any candidate is a hypothesis for prospective evaluation, never a result.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

SENS = ('recovery/A/SEX', 'recovery/AB/SEX', 'recovery/A/RAC1P', 'recovery/AB/RAC1P')
RES = 'utility/same_residence'
WEIGHTS = ('unweighted', 'person_weighted')
FAMILIES = {'grid_vs_J': 'J', 'grid_vs_leace_A0': 'leace_A0'}
ALPHA = 0.05
RES_MARGIN = 0.003
SENS_TOL = 0.001


def run(intervals_csv: Path, out_json: Path, out_md: Path):
    rows = [r for r in csv.DictReader(open(intervals_csv)) if r['comparison_family'] in FAMILIES]
    m = len(rows)
    z_two = NormalDist().inv_cdf(1 - ALPHA / (2 * m))     # two-sided simultaneous
    z_one = NormalDist().inv_cdf(1 - ALPHA / m)           # one-sided bounds
    table = defaultdict(dict)
    for r in rows:
        est, se = float(r['estimate']), float(r['bootstrap_se'])
        table[(r['left'], FAMILIES[r['comparison_family']], r['weight'])][r['endpoint']] = {
            'estimate': est, 'se': se, 'upper_one_sided': est + z_one * se,
            'low_two_sided': est - z_two * se, 'high_two_sided': est + z_two * se,
            'old_family_size': int(r['family_size'])}
    configs = sorted({k[0] for k in table})
    verdict = {}
    for c in configs:
        per = {}
        for comp in ('J', 'leace_A0'):
            for w in WEIGHTS:
                e = table.get((c, comp, w))
                if not e or RES not in e or any(s not in e for s in SENS):
                    per[f'{comp}/{w}'] = {'assessable': False}
                    continue
                res = e[RES]
                res_ok = res['estimate'] <= -RES_MARGIN and res['high_two_sided'] < 0
                sens_ok = all(e[s]['upper_one_sided'] <= SENS_TOL for s in SENS)
                sens_zero = all(e[s]['upper_one_sided'] <= 0 for s in SENS)
                per[f'{comp}/{w}'] = {'assessable': True, 'residence_improved': res_ok,
                                      'sensitive_within_001': sens_ok, 'sensitive_within_0': sens_zero,
                                      'residence_estimate': res['estimate'],
                                      'residence_interval': [res['low_two_sided'], res['high_two_sided']],
                                      'max_sensitive_upper': max(e[s]['upper_one_sided'] for s in SENS)}
        vs_j = all(per[f'J/{w}'].get('residence_improved') and per[f'J/{w}'].get('sensitive_within_001') for w in WEIGHTS)
        vs_l = all(per[f'leace_A0/{w}'].get('residence_improved') and per[f'leace_A0/{w}'].get('sensitive_within_001') for w in WEIGHTS)
        verdict[c] = {'passes_vs_J': vs_j, 'passes_vs_leace_A0': vs_l, 'passes_both': vs_j and vs_l,
                      'residence_point_better_than_J_both_weightings':
                          all(per[f'J/{w}'].get('residence_estimate', 1) < 0 for w in WEIGHTS),
                      'detail': per}
    near = sorted(configs, key=lambda c: max(verdict[c]['detail'][f'J/{w}'].get('residence_estimate', 9) for w in WEIGHTS))
    summary = {
        'status': 'EXPLORATORY utility-first reanalysis of stored development evidence (2018, repeatedly used pools). '
                  'Not a retroactive success; the original negative verdict and criterion stand.',
        'new_family_size': m, 'z_two_sided': z_two, 'z_one_sided': z_one,
        'old_family_size_not_reused': 3900, 'configurations_searched': len(configs),
        'n_pass_vs_J': sum(v['passes_vs_J'] for v in verdict.values()),
        'n_pass_vs_leace_A0': sum(v['passes_vs_leace_A0'] for v in verdict.values()),
        'n_pass_both': sum(v['passes_both'] for v in verdict.values()),
        'n_residence_point_better_than_J': sum(v['residence_point_better_than_J_both_weightings'] for v in verdict.values()),
        'passing': [c for c in configs if verdict[c]['passes_both']],
        'closest_by_worst_weighting_residence_vs_J': [
            {'config': c, 'residence_vs_J': {w: verdict[c]['detail'][f'J/{w}'].get('residence_estimate') for w in WEIGHTS},
             'max_sensitive_upper_vs_J': {w: verdict[c]['detail'][f'J/{w}'].get('max_sensitive_upper') for w in WEIGHTS}}
            for c in near[:8]],
        'limitations': [
            'selection over 124 already-observed configurations; any near miss is a hypothesis only',
            'J-relative increments near zero do not show no information remains; negative increments do not undo disclosure',
            'intervals are conditional development sampling uncertainty on repeatedly used 2018 pools',
            '2016 not opened'],
        'per_config': verdict}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_json, 'w'), indent=1)
    lines = ['# TIER1_REANALYSIS — exploratory utility-first criterion (new search family)', '',
             summary['status'], '',
             f"New family size m = {m} (124 configs x {{J, leace_A0}} x 5 endpoints x 2 weightings); "
             f"two-sided z = {z_two:.3f}, one-sided z = {z_one:.3f}. The old m = 3900 adjustment is not reused.", '',
             f"* configurations passing vs J: **{summary['n_pass_vs_J']}**",
             f"* passing vs leace_A0: **{summary['n_pass_vs_leace_A0']}**",
             f"* passing both (useful candidate): **{summary['n_pass_both']}**",
             f"* residence point estimate better than J under both weightings: {summary['n_residence_point_better_than_J']}", '',
             '| config (closest to J on residence) | residence vs J unw | pw | max sens. upper vs J unw | pw |', '|---|---|---|---|---|']
    for n in summary['closest_by_worst_weighting_residence_vs_J']:
        r, s = n['residence_vs_J'], n['max_sensitive_upper_vs_J']
        f = lambda v: 'NA' if v is None else f'{v:+.4f}'
        lines.append(f"| {n['config']} | {f(r['unweighted'])} | {f(r['person_weighted'])} | {f(s['unweighted'])} | {f(s['person_weighted'])} |")
    lines += ['', 'Limitations: ' + '; '.join(summary['limitations']) + '.']
    Path(out_md).write_text('\n'.join(lines) + '\n')
    return summary


if __name__ == '__main__':
    s = run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print({k: s[k] for k in ('new_family_size', 'n_pass_vs_J', 'n_pass_vs_leace_A0', 'n_pass_both', 'n_residence_point_better_than_J')})
