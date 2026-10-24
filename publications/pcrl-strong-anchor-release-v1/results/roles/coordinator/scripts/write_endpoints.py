# Coordinator script (label-free) that wrote ENDPOINTS.json; saved from the session scratchpad at closeout.
"""ENDPOINTS.json: the compact machine-readable endpoint table, built from the saved records only (label-free)."""
import json
import os
import sys

ROOT = str(__import__('pathlib').Path(__file__).resolve().parents[5])   # the repository root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np  # noqa: E402
from sar import claims as CL, common as C  # noqa: E402
from nrm import rules as NR  # noqa: E402

R = C.RESULTS
ORDER = ('FA_REG', 'FB_REG', 'FB_R1', 'FA_R1', 'FS_CPI', 'FA_R2')
BANK = {'FB_R1': 'SAR1_FB_R1', 'FA_R1': 'SAR1_FA_R1', 'FA_R2': 'SAR1_FA_R2'}
summ = json.loads((R / 'claims' / 'CAMPAIGN_SUMMARY.json').read_text())['summary']
stm = {x['card']: x['outcomes_statement'] for x in summ['non_registered']}
stm.update({x['card']: x['outcomes_statement'] for x in summ['registered'].values()})


def lab(x):
    return x.get('label') if isinstance(x, dict) else x


def r4(x):
    return None if x is None else round(float(x), 4)


def views(grid, m):
    out = {}
    for view in ('pair', 'income', 'occupation'):
        for v in C.VERSIONS:
            cells = [c for c in NR.REGISTERED if c.startswith(f'S{v}')]
            xs = [grid['cells'][c][m][view]['score_auc'] for c in cells if c in grid['cells'] and m in grid['cells'][c]]
            out[f'{view}_v{v}'] = r4(np.mean(xs)) if len(xs) == len(cells) else None
    return out


def utility(spec):
    fam, cfg = spec.split('/')
    p = C.RUNS / fam / cfg / 'EVAL_MS.json'
    if not p.is_file():
        return None
    ev = json.loads(p.read_text())['versions']
    out = {}
    for v in ('0', '1'):
        e = ev[v]
        g = {('men_sex0' if c['group'] == 'sex0' else 'women_sex1'): {'value': round(float(c['value']), 5), 'se': round(float(c['se']), 5),
                                                                       'cap': c['cap']} for c in e['utility'].get('guards', [])}
        out[v] = {'point_pass_12_plus_2': e['point']['point_pass_12_plus_2'], 'max_value_over_cap': round(e['point']['max_value_over_cap'], 3),
                  'guards': g}
    return out


rows = {}
for rd in ORDER:
    dec = json.loads((R / 'claims' / rd / 'DECISIONS.json').read_text())
    d = dec['decisions']
    plan = json.loads((R / 'claims' / f'PLAN_{rd}.json').read_text())
    row = {'reading': rd, 'family': d.get('family'), 'card': d.get('card'), 'revision': d.get('revision'), 'start': d.get('start'),
           'candidate': plan.get('cand'), 'adaptive_tag': d.get('adaptive_tag'), 'outcome_class': lab(d['outcome_class']),
           'labels': {k: lab(d[k]) for k in ('D1', 'UM', 'D2', 'D3', 'D4', 'D5', 'D6') if k in d},
           'three_outcomes_statement': stm.get(d.get('card'))}
    if rd in BANK:
        grid = CL.load_grid(BANK[rd])
        cand = plan['cand']
        row['recovery_registered_cells'] = {cand: views(grid, cand), d.get('start'): views(grid, d.get('start'))}
        row['confidence_conjunction'] = {k: (v or {}).get('label') for k, v in (d['D1'].get('confidence') or {}).items()}
        d2 = d['D2'].get('comparisons') or {}
        row['D2_comparisons'] = {k: {'point': v.get('point') or (v.get('frontier') or {}).get('bracketing_control'),
                                     'benefit_mean': r4(v.get('benefit_mean')), 'benefit_se': r4(v.get('benefit_se')),
                                     'pass': v.get('pass'), 'label': v.get('label') or v.get('benefit')} for k, v in d2.items()}
        row['D3_per_reference'] = {k: {'mean': r4(v.get('mean')), 'se_agg': r4(v.get('se_agg')), 'benefit': v.get('benefit'),
                                       'recipient_gate_pass': v.get('recipient_gate_pass')}
                                   for k, v in (d['D3'].get('per_reference') or {}).items()}
        row['D3_passed'] = d['D3'].get('passed')
        row['D4_per_reference'] = {k: lab(v) for k, v in (d['D4'].get('per_reference') or {}).items()}
        fam = 'FA' if rd.startswith('FA') else 'FB'
        row['utility_MODEL_SELECT'] = utility(f'{fam}/{cand}')
        dp = R / 'claims' / f'{rd}_D2R' / 'D2R.json'
        if dp.is_file():
            x = json.loads(dp.read_text())['d2r']
            row['D2R'] = {'label': x['label'], 'refit': x.get('refit'), 'sign_changes': x.get('sign_changes'),
                          'headline_caveat': x.get('headline_caveat'),
                          'per_comparator': {k: {'label': v.get('label'), 'registered_mean': r4(v.get('registered_mean')),
                                                 'robustness_mean': r4(v.get('robustness_mean')), 'robustness_se': r4(v.get('robustness_se')),
                                                 'sign_change': v.get('sign_change'), 'decisive': v.get('decisive')}
                                             for k, v in x['per_comparator'].items() if v.get('label') != 'NOT_RUN'}}
        else:
            row['D2R'] = None
        if row['D2R'] and row.get('three_outcomes_statement'):
            cav = row['D2R'].get('headline_caveat')
            row['three_outcomes_statement'] = (
                row['three_outcomes_statement'].replace('(D2R NOT_RUN)', f"(D2R {row['D2R']['label']} from claims/{rd}_D2R/D2R.json"
                                                        + (f'; {cav}' if cav else '') + ')'))
            row['statement_note'] = ('the write-once DECISIONS.json statement predates the D2R record and reads "D2R NOT_RUN"; '
                                     'the D2R label here comes from the D2R record')
    rows[rd] = row
refs = {}
grid = CL.load_grid('SAR1_REFS')
for m in ('NEURAL_RAW', 'H_LJ', 'SUCC', 'RETIRED_B3CTL', 'B2TWIN', 'DDR_PTC_R1', 'TWJ', 'U0'):
    if m in grid['members']['0'] if isinstance(grid['members'], dict) and '0' in grid['members'] else True:
        try:
            refs[m] = views(grid, m)
        except KeyError:
            refs[m] = None
rec = {'schema': 'SAR1-endpoints-v1', 'utc': C.now(),
       'note': 'built by a coordinator script from claims/*/DECISIONS.json, claims/*_D2R/D2R.json, claims/CAMPAIGN_SUMMARY.json, the '
               'saved grids and the MODEL_SELECT evaluation records; development evidence on adaptively reused MODEL_SELECT people; '
               'benefit = comparator AUC - candidate AUC (positive favours the candidate); intervals do not correct the adaptive search',
       'material_target': 'a resolved 0.020 linked-pair benefit against NEURAL_RAW and H_LJ with the recipient gates (mean - 2 se_agg > '
                          '0.020 and both version means > 0.020)',
       'readings': rows, 'reference_levels_SAR1_REFS': refs,
       'campaign_summary': {**{k: summ[k] for k in ('d2_looks', 'd2_met', 'd3_met', 'd4_met', 'flags')},
                            'statement': summ['statement'],
                            'registration_status': 'both D2 looks (FB_R1, FA_R2) are non-registered adaptive revisions; the registered '
                                                   'readings FA_REG and FB_REG found no feasible strength'},
       'deployment': {'packages': 13, 'equivalence': 'bitwise for the full evaluated batch; within 6e-16 per probability for other '
                                                     'request batch sizes, identical decisions (deploy/DEPLOYMENT_TOLERANCE.json)'},
       'specifications': {'used': 28, 'limit': 36, 'candidates': 8, 'task_twins': 3, 'matched_controls': 17}}
C.write_json(R / 'ENDPOINTS.json', rec)
print(json.dumps({k: (v['outcome_class'], v['labels'].get('D2'), v['labels'].get('D4'), (v.get('D2R') or {}).get('label'))
                  for k, v in rows.items()}, indent=1))
print(json.dumps(refs, indent=1))
