"""Independent verifier, check 10 helper: extract the aggregate numbers quoted in RESEARCH_DECISION.md / START_HERE.md from the
written DEV_REPORT_RESULTS.json, NOMINATION.json and EXPERIMENT_LEDGER.csv (aggregate outputs only; no person-level array, no
label is read). Public-use ACS 2019 Florida records; readers estimate the recorded SEX column from released predictions.
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
RES = ROOT / 'results' / 'pcrl_income_headroom_successor_v1'


def main():
    d = json.loads((RES / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    c1 = d['critical']['c1']
    out = {'c1': c1}
    tab = {}
    for svc in ('SUCCESSOR', 'H_LJ', 'NEURAL_RAW', 'TW', 'CC', 'B2TWIN', 'RETIRED_B3CTL', 'PUB_IBSI'):
        row = {}
        for s in (0, 1):
            sv = d['per_seed'][f'S{s}']['services'][svc]
            u = sv['utility']
            row[f's{s}'] = {'pair_E': sv['recovery']['ENHANCED']['pair']['auc'], 'pair_O': sv['recovery']['ORIGINAL']['pair']['auc'],
                            'inc_E': sv['recovery']['ENHANCED']['income']['auc'], 'inc_O': sv['recovery']['ORIGINAL']['income']['auc'],
                            'occ_E': sv['recovery']['ENHANCED']['occupation']['auc'],
                            'inc_brier_vs_Ucal': u['income|Ucal|brier_excess']['point'], 'inc_acc_loss_U0': u['income|U0|accuracy_loss']['point'],
                            'inc_margin_SE': {k: (v['cap'] - v['point']) / v['se'] for k, v in u.items() if k.startswith('income|')},
                            'occ_margin_SE': {k: (v['cap'] - v['point']) / v['se'] for k, v in u.items() if k.startswith('occupation|')}}
        tab[svc] = row
    out['table'] = tab
    E = d['scorecards']['ENHANCED']['endpoints']; O = d['scorecards']['ORIGINAL']['endpoints']
    pick = {}
    for k in E:
        if k.startswith(('BEN|', 'NW|', 'DESC|BEN|')):
            e = E[k]
            pick[k] = {'ENH_point': e.get('point'), 'ENH_se': e.get('se'), 'ENH_L1': e.get('point') - c1 * e.get('se') if e.get('se') is not None else None,
                       'ENH_U1': e.get('point') + c1 * e.get('se') if e.get('se') is not None else None,
                       'ORI_point': O.get(k, {}).get('point'), 'ORI_U1': (O[k]['point'] + c1 * O[k]['se']) if k in O and O[k].get('se') is not None else None,
                       'ORI_L1': (O[k]['point'] - c1 * O[k]['se']) if k in O and O[k].get('se') is not None else None}
    out['endpoints'] = pick
    out['labels'] = {b: {c: v.get('label') for c, v in d['scorecards'][b]['claims'].items()} for b in ('ENHANCED', 'ORIGINAL')}
    cl = d['scorecards']['ENHANCED']['claims']
    out['E_utility_confidence'] = cl['E']['utility_confidence']
    out['R'] = {k: cl['R'][k] for k in ('R2', 'R3', 'label')}
    out['R2_detail'] = cl['R']['R2_detail']
    out['O_below_rule'] = cl['O']['below_rule']
    out['refit_roster_S0_SUCCESSOR'] = d['per_seed']['S0']['refit_roster']['SUCCESSOR']
    out['refit_roster_S1_SUCCESSOR'] = d['per_seed']['S1']['refit_roster']['SUCCESSOR']
    out['reader_sensitive'] = d['reader_sensitive']
    # configuration counts per family (EXPERIMENT_LEDGER.csv: first evaluation rows with counts_toward_36 = 1)
    rows = list(csv.DictReader(open(RES / 'EXPERIMENT_LEDGER.csv')))
    fams = Counter(); seen = set()
    for r in rows:
        if str(r.get('counts_toward_36')) == '1':
            key = (r['family'], r['config_id'], r['seed'])
            if key not in seen:
                seen.add(key); fams[r['family']] += 1
    out['configs_counts_toward_36_by_family'] = dict(fams); out['configs_total'] = sum(fams.values())
    nom = json.loads((RES / 'NOMINATION.json').read_text())
    out['nomination_keys'] = list(nom)[:30]
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
