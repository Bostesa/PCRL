"""After the one logged DEV_REPORT evaluation: public aggregate tables and the forecast scores (no person-level data; no new label
read; everything comes from report/DEV_REPORT_RESULTS.json and the private development evaluations).

UTILITY_TABLE.csv  : DEV_REPORT utility vector (12 clauses, guards) of every release in the final bank, per seed.
RECOVERY_TABLE.csv : DEV_REPORT recovery (winner AUC, SE, winner) per release, seed and view, beside the MODEL_SELECT proxies.
PRIMARY_ENDPOINTS.csv, SECONDARY_ENDPOINTS.csv : every frozen endpoint with its point, SE, bounds, status and flags.
FORECAST_SCORES.json : the registered forecasts (FORECASTS.json, FORECASTS_2.json) scored against the outcomes.
"""
from __future__ import annotations
import csv
import json

from . import common as C

R = C.RESULTS
ROLE_SOURCE = {'FINALIST': ('B', 'B3CTL_UNCOUPLED_MIX50_C6A'), 'TASKONLY': ('B', 'B3CTL_TASKONLY_TR_C6A'), 'B2TWIN': ('B', 'B2CTL_UNCOUPLED_C6A')}
BASE_NAME = {'NEURAL_RAW': 'NEURAL_RAW', 'H_LJ': 'H_LJ', 'H_JJ': 'H_JJ', 'T_TREE': 'T_TREE', 'PUB_IBSI': 'IBSI_NLLM'}


def ms_proxy(role, s):
    if role in ROLE_SOURCE:
        fam, cid = ROLE_SOURCE[role]
        m = json.loads((C.RUNS / fam / cid / f'S{s}' / 'EVAL_MS.json').read_text())
    else:
        m = json.loads((C.RUNS / 'BASELINES' / f'BASELINES_MS_S{s}.json').read_text())[BASE_NAME[role]][str(s)]
    return {v: m['proxy'][v]['max'] for v in ('pair', 'income', 'occupation')}


def main():
    d = json.loads((R / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    util_rows, rec_rows = [], []
    for s in C.SEEDS:
        ps = d['per_seed'][f'S{s}']
        for role, svc in ps['service_of_role'].items():
            v = ps['services'][svc]
            row = {'seed': s, 'release': role, 'dev_point_feasible_12': v['utility_point_feasible_12_this_seed'], 'guard_point_pass': v['guard_point_pass_this_seed']}
            for k, x in v['utility'].items():
                row[f'dev_{k.replace("|", "_")}'] = round(x['point'], 5)
            for g, x in v['subgroup'].items():
                row[f'dev_guard_{g}'] = round(x['occupation_brier_excess_vs_U0'], 5)
            util_rows.append(row)
            msp = ms_proxy(role, s)
            for view in ('pair', 'income', 'occupation'):
                r = v['recovery'][view]
                rec_rows.append({'seed': s, 'release': role, 'view': view, 'dev_auc': round(r['auc'], 4), 'dev_auc_se': round(r['auc_se'], 4),
                                 'dev_marginal95_lower': round(r['marginal95'][0], 4), 'dev_marginal95_upper': round(r['marginal95'][1], 4),
                                 'dev_winner': r['winner'], 'attack_validation_auc': round(r['attack_validation_auc'], 4),
                                 'model_select_proxy': round(msp[view], 4),
                                 'distinct_attack_fit_vectors': ps['coverage_attack_roles'][svc][view]['distinct_vectors_ATTACK_FIT'],
                                 'median_attack_fit_people_per_vector': ps['coverage_attack_roles'][svc][view]['median_ATTACK_FIT_people_per_vector']})
    for name, rows in (('UTILITY_TABLE.csv', util_rows), ('RECOVERY_TABLE.csv', rec_rows)):
        with open(R / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for fam, name in (('P', 'PRIMARY_ENDPOINTS.csv'), ('S', 'SECONDARY_ENDPOINTS.csv')):
        rows = []
        for k, e in d['endpoints'].items():
            if e['family'] != fam:
                continue
            rows.append({'key': k, 'claim': e.get('claim'), 'kind': e['kind'], 'margin': e['margin'], 'point': round(e['point'], 5), 'se': round(e['se'], 5),
                         'clause_L1': None if e.get('clause_L1') is None else round(e['clause_L1'], 5),
                         'clause_U1': None if e.get('clause_U1') is None else round(e['clause_U1'], 5),
                         'family_lower': round(e['family_lower'], 5), 'family_upper': round(e['family_upper'], 5), 'status': e['status'],
                         'category': e['category'], 'flags': ';'.join(e['flags']), 'availability': e['availability']})
        with open(R / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # forecast scoring
    ep = d['endpoints']; cl = d['claims']
    pair_keys = [f'COMPETITIVE|pair_benefit_vs_{X}|s{s}' for X in ('NEURAL_RAW', 'H_LJ') for s in C.SEEDS]
    util_ok = d['E_fields']['dev_point_status']['utility_24'] == 24
    outcome = {
        'F1_utility_24_point': util_ok,
        'F2_guards_4_point': d['E_fields']['dev_point_status']['guards_4'] == 4,
        'F3_income_recipient': all(ep[f'COMPETITIVE|income_worsening_vs_{X}|s{s}']['clause_pass'] for X in ('NEURAL_RAW', 'H_LJ') for s in C.SEEDS),
        'F4_occupation_recipient': all(ep[f'COMPETITIVE|occupation_worsening_vs_{X}|s{s}']['clause_pass'] for X in ('NEURAL_RAW', 'H_LJ') for s in C.SEEDS),
        'F5_pair_vs_NEURAL_RAW': all(ep[f'COMPETITIVE|pair_benefit_vs_NEURAL_RAW|s{s}']['clause_pass'] for s in C.SEEDS),
        'F6_pair_vs_H_LJ': all(ep[f'COMPETITIVE|pair_benefit_vs_H_LJ|s{s}']['clause_pass'] for s in C.SEEDS),
        'F7_point_0020': all(ep[k]['point'] >= 0.020 for k in pair_keys),
        'F8_established_0020': all(ep[k]['clause_L1'] > 0.020 for k in pair_keys),
        'F9_vs_taskonly': all(ep[f'MECH_TO|pair_benefit_vs_TASKONLY|s{s}']['clause_pass'] for s in C.SEEDS),
        'F10_claim_B_full': cl['B']['holds'],
        'F11_claim_E': cl['E']['holds'],
        'F12_claim_B': cl['B']['holds'],
        'F13_claim_B_material': cl['B']['nested_0020']['label'] == 'MATERIAL_SUPERIORITY_ESTABLISHED',
        'F14_claim_M_TO': cl['M']['steps'][0]['holds'],
        'F15_claim_M_full': cl['M']['holds'],
        'F16_claim_J': cl['J']['holds'],
        'F17_control_validity': d['control_validity']['valid']}
    probs = {}
    for fn in ('FORECASTS.json', 'FORECASTS_2.json'):
        probs.update({k: v['p'] for k, v in json.loads((R / fn).read_text())['forecasts'].items()})
    scored = {k: {'p': probs[k], 'outcome': bool(outcome[k]), 'brier': (probs[k] - float(outcome[k])) ** 2} for k in probs}
    rec = {'schema': 'AMD1-forecast-scores-v1', 'utc': C.now(), 'scored': scored, 'mean_brier': sum(x['brier'] for x in scored.values()) / len(scored),
           'note': 'forecasts registered before any final reader fit (FORECASTS.json committed in ad37c120c; FORECASTS_2.json in 345e0adb0)'}
    (R / 'FORECAST_SCORES.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({'mean_brier': rec['mean_brier'], 'outcomes': {k: v['outcome'] for k, v in scored.items()}}))


if __name__ == '__main__':
    main()
