"""Unit IHS_INHERITED_V1: inherited occupation-side margins and the existing paired variances, from EXISTING development
artifacts only (before any new fit; brief section 4).

Inputs (no new label read here):
- the published DEV_REPORT endpoints of the validation (results/pcrl_frozen_finalist_validation_v1/PRIMARY_ENDPOINTS.csv): the
  retired finalist's occupation output is bitwise the JOINT_TREE output of H_LJ, so its occupation utility clauses and within-SEX
  guards are the inherited occupation constraints (reader-free; identical in both scorecards);
- this campaign's MODEL_SELECT reference evaluation (units IHS_REFERENCES_S0_V1 / S1_V1, private runs/REF), summarized to
  clause level only (no person-level value leaves the private tree).
Outputs: REFERENCE_MS_TABLE.csv (every reference x seed x condition: value, cap, household-clustered SE, margin, margin / SE),
INHERITED_OCCUPATION_MARGINS.json (DEV and MODEL_SELECT margins of the JOINT_TREE occupation output in metric and SE units,
with the same SEs projected to other household counts by sqrt(G_measured / G_target)).
"""
from __future__ import annotations
import csv
import json
import math

from . import common as C
from . import devrun as D

SCALES = {'MODEL_SELECT': 5306, 'DEV_REPORT': 3955, 'SAME_POPULATION_UNTESTED_FL2019': 9853 + 7657}


def ms_rows():
    out = []
    for s in C.SEEDS:
        res = D.reference_eval(s)
        for name in D.REFERENCE_SET:
            u = res[name]['utility']
            for c in u['clauses']:
                out.append({'reference': name, 'seed': s, 'condition': f"{c['task']}|{c['reference']}|{c['metric']}", 'value': c['value'],
                            'cap': c['cap'], 'se': c['se'], 'margin': c['margin'], 'margin_se_units': c['margin_se_units'], 'households': u['households']})
            for g in u['guards']:
                out.append({'reference': name, 'seed': s, 'condition': f"guard|occupation_brier_vs_U0|{g['group']}", 'value': g['value'],
                            'cap': g['cap'], 'se': g['se'], 'margin': g['margin'], 'margin_se_units': g['margin_se_units'], 'households': u['households']})
            if 'proxy' in res[name]:
                for view in ('pair', 'income', 'occupation'):
                    sm = res[name]['proxy']['summary'][view]
                    out.append({'reference': name, 'seed': s, 'condition': f'proxy|{view}|mean_over_2x2_cells', 'value': sm['mean'], 'cap': '',
                                'se': '', 'margin': '', 'margin_se_units': '', 'households': u['households'],
                                'cell_min': sm['min'], 'cell_max': sm['max']})
    return out


def dev_rows():
    rows = list(csv.DictReader(open(C.FFV_RESULTS / 'PRIMARY_ENDPOINTS.csv')))
    out = {}
    for r in rows:
        if r['scorecard'] != 'ORIGINAL':
            continue
        k = r['key']
        if k.startswith('UTILITY|occupation|') or k.startswith('SUBGROUP|occupation_brier_excess_vs_U0'):
            out[k] = {'value': float(r['point']), 'se': float(r['se']), 'cap': float(r['margin']), 'flags': r['flags'], 'seed': int(r['seed'])}
    return out


def project(margin, se, g_from, g_to):
    se_to = se * math.sqrt(g_from / g_to)
    return {'se': se_to, 'margin_se_units': margin / se_to if se_to > 0 else math.inf}


def run():
    ms = ms_rows()
    with open(C.RESULTS / 'REFERENCE_MS_TABLE.csv', 'w', newline='') as f:
        cols = ['reference', 'seed', 'condition', 'value', 'cap', 'se', 'margin', 'margin_se_units', 'households', 'cell_min', 'cell_max']
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in ms:
            w.writerow({k: r.get(k, '') for k in cols})
    dev = dev_rows()
    inh = {'DEV_REPORT_published_validation': {}, 'MODEL_SELECT_this_campaign': {}}
    for k, v in sorted(dev.items()):
        m = v['cap'] - v['value']
        inh['DEV_REPORT_published_validation'][k] = {
            'value': v['value'], 'cap': v['cap'], 'se': v['se'], 'margin': m, 'margin_se_units_at_DEV': m / v['se'],
            'projected': {n: project(m, v['se'], SCALES['DEV_REPORT'], g) for n, g in SCALES.items()}, 'flags': v['flags']}
    for r in ms:
        if r['reference'] == 'H_LJ' and (r['condition'].startswith('occupation|') or r['condition'].startswith('guard|')):
            inh['MODEL_SELECT_this_campaign'][f"s{r['seed']}|{r['condition']}"] = {
                'value': r['value'], 'cap': r['cap'], 'se': r['se'], 'margin': r['margin'], 'margin_se_units_at_MS': r['margin_se_units'],
                'projected': {n: project(r['margin'], r['se'], SCALES['MODEL_SELECT'], g) for n, g in SCALES.items()}}
    below = {scale: sorted(k for k, v in inh['DEV_REPORT_published_validation'].items() if v['projected'][scale]['margin_se_units'] < 2.0)
             for scale in SCALES}
    below_ms = {scale: sorted(k for k, v in inh['MODEL_SELECT_this_campaign'].items() if v['projected'][scale]['margin_se_units'] < 2.0)
                for scale in SCALES}
    rec = {'schema': 'IHS1-inherited-occupation-margins-v1', 'utc': C.now(),
           'what': ('the occupation output of H_LJ (JOINT_TREE occupation, E20 corrections on the frozen T_TREE routing), which the '
                    'retired finalist released bitwise; its utility clauses and within-SEX guards cannot be changed by any income-side '
                    'mechanism that keeps it'),
           'household_scales': SCALES, 'margins': inh,
           'conditions_below_2_SE': {'from_DEV_REPORT_SEs': below, 'from_MODEL_SELECT_SEs': below_ms},
           'note': ('margins are point estimates minus caps divided by household-clustered paired SEs; projections assume the same '
                    'per-household variance at another household count (iid households from the same population, unweighted). '
                    'They are planning arithmetic, not confidence statements, and DEV_REPORT has been read 12 times before this campaign.')}
    C.write_json(C.RESULTS / 'INHERITED_OCCUPATION_MARGINS.json', rec)
    print(json.dumps(rec['conditions_below_2_SE'], indent=1))


if __name__ == '__main__':
    run()
