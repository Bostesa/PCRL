#!/usr/bin/env python3
"""RFA1 verifier, wave 2, task 4c: check the D1 / D2 / D3 / D4 labels of claims/RFA_v1/DECISIONS.json against the rules applied to the
verifier's own recomputed inputs (no label read here; release / bank arrays only for the income-wire characterization).

Leg 1 (own): the verifier's labels and numbers (V2_GRID.json from unit RFA_VER_GRID_v1; V2_USEFULNESS.json from RFA_VER_USEFULNESS_v1)
vs DECISIONS.json: D1 and its confidence conjunction, every D2 comparison (START, CONT, the STOPFIT / ALTCRIT / RAWJ frontiers: Q*,
label, benefit mean / se_agg, secondary metrics, c2, c3, targeting), D3 per reference, D4 target screens, the outcome class.
Leg 2 (frozen rules): the campaign's record rebuilt by rfa.rfclaims.build (reader-budget flags taken from DECISIONS.json instead of
re-reading release arrays) and rfa.rules.decisions applied (a) unchanged (replay) and (b) with the verifier's recomputed grid
entries and non-table winner benefits substituted.
Income wire: float32 representability of the frozen bank's NEURAL_RAW MODEL_SELECT income vs RF_FULL's released income.

    python3 -m rfa.jobqueue run --unit RFA_VER_DECISIONS_v1 --stage verify --rss-gib 2 -- \\
        <repository>/.venv/bin/python results/pcrl_reader_fit_aware_release_v1/roles/verifier/code/v2_decisions.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

OUT = HERE.parent / 'out' / 'V2_DECISIONS.json'
RES = ROOT / 'results' / 'pcrl_reader_fit_aware_release_v1'
RUNS = ROOT / '.private' / 'pcrl_reader_fit_aware_release_v1' / 'runs'
TOL = 1e-12


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b, tol=TOL):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def confidence_label(conf_by_v, keys_fmt):
    """From V2_USEFULNESS per-version per-condition labels: (label, n_pass, unresolved names, violations)."""
    unresolved, viol, n = [], [], 0
    for v, labs in conf_by_v.items():
        for k, lab in labs.items():
            part, name = k.split('|', 1)
            nm = keys_fmt(v, part, name)
            if lab == 'PASS':
                n += 1
            elif lab == 'UNRESOLVED':
                unresolved.append(nm)
            else:
                viol.append(nm)
    label = 'CONFIDENCE_VIOLATION' if viol else ('CONFIDENCE_UNRESOLVED' if unresolved else 'CONFIDENCE_ALL_PASS')
    return label, n, sorted(unresolved), sorted(viol)


def main():
    t0 = time.process_time()
    dec_file = RES / 'claims' / 'RFA_v1' / 'DECISIONS.json'
    D = json.loads(dec_file.read_text())
    d = D['decisions']
    G = json.loads((HERE.parent / 'out' / 'V2_GRID.json').read_text())
    U = json.loads((HERE.parent / 'out' / 'V2_USEFULNESS.json').read_text())
    res = {'schema': 'RFA1-verifier-decisions-v2', 'unit_id': os.environ.get('RFA_UNIT_ID'), 'decisions_sha256': sha256(dec_file),
           'inputs_sha256': {'V2_GRID.json': sha256(HERE.parent / 'out' / 'V2_GRID.json'),
                             'V2_USEFULNESS.json': sha256(HERE.parent / 'out' / 'V2_USEFULNESS.json')}, 'checks': {}, 'compare': {}}
    chk, cmpd = res['checks'], res['compare']
    mine = G['decisions_verifier']
    # ---- leg 1: labels
    for k in ('D1', 'D2', 'D3', 'D4'):
        cmpd[k] = {'decisions': d[k]['label'], 'verifier': mine[k]}
        chk[f'{k}_label_equals_verifier'] = d[k]['label'] == mine[k]
    oc_mine = ('TASK_INFEASIBLE' if mine['D1'] != 'TASK_FEASIBLE' else
               'COMPETITIVE_CANDIDATE' if mine['D2'].startswith('COMPONENT_GAIN') and mine['D3'] == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' else
               'MECHANISM_COMPONENT_NOT_COMPETITIVE' if mine['D2'].startswith('COMPONENT_GAIN') else
               'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED' if mine['D3'] == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' else 'BOUNDED_NEGATIVE_EXPERIMENT')
    cmpd['outcome_class'] = {'decisions': d['outcome_class']['label'], 'partial': d['outcome_class'].get('partial'), 'verifier': oc_mine,
                             'verifier_partial': mine['D2']}
    chk['outcome_class_equals_verifier'] = d['outcome_class']['label'] == oc_mine and d['outcome_class'].get('partial') == mine['D2']
    # D1 confidence (CAND = RF_FULL, START = NEURAL_RAW, CONT = RF_TASK)
    fmt = lambda v, part, name: f'v{v}|{name}' if part == 'clauses' else f'v{v}|guard|{name}'
    for role, src in (('cand', U['configs']['RF_FULL']), ('start', U['references']['NEURAL_RAW']), ('cont', U['configs']['RF_TASK'])):
        lab, n, unr, viol = confidence_label({v: pv['confidence'] for v, pv in src['per_version'].items()}, fmt)
        rec = d['D1']['confidence'][role]
        cmpd[f'D1_confidence_{role}'] = {'decisions': rec, 'verifier': {'label': lab, 'n_pass': n, 'unresolved': unr, 'violations': viol}}
        chk[f'D1_confidence_{role}_equals_verifier'] = rec['label'] == lab and rec['n_pass'] == n and sorted(rec['unresolved']) == unr \
            and sorted(rec['violations']) == viol
    # D2 comparisons
    c2 = d['D2']['comparisons']
    gc = G['comparisons']
    for name, comp in (('START', 'NEURAL_RAW'), ('CONT', 'RF_TASK')):
        x, m = c2[name], gc[comp]
        cmpd[f'D2_{name}'] = {'decisions': {k: x.get(k) for k in ('label', 'benefit_mean', 'benefit_se', 'recipient_gate_pass', 'pass')},
                              'verifier': {'label': m['benefit_label'], 'mean': m.get('mean'), 'se_agg': m.get('se_agg'),
                                           'recipient_gate_pass': m['recipient_gate_pass'], 'pass': m['passes']}}
        chk[f'D2_{name}_equals_verifier'] = (x['label'] == m['benefit_label'] and close(x['benefit_mean'], m.get('mean'))
                                             and close(x['benefit_se'], m.get('se_agg')) and x['recipient_gate_pass'] == m['recipient_gate_pass']
                                             and x['pass'] == m['passes'])
    for key in ('STOPFIT', 'ALTCRIT', 'RAWJ'):
        x, f = c2[key], G['frontiers'][key]
        lab_mine = 'NOT_SEPARATED' if f.get('reason') == 'NO_BRACKET' else f['label']
        cmpd[f'D2_{key}'] = {'decisions': {k: x.get(k) for k in ('label', 'frontier_label', 'point', 'benefit_mean', 'benefit_se', 'reason',
                                                                 'recipient_gate_vs_bracket', 'dominated_by_cheaper', 'secondary_more_spending', 'pass')},
                             'verifier': {'label': lab_mine, 'bracket': f.get('bracket'), 'benefit_label': f.get('benefit_label'),
                                          'mean': f.get('benefit_mean'), 'se_agg': f.get('benefit_se_agg'), 'c2': f.get('c2_recipient_gate_vs_Q'),
                                          'c3': f.get('c3_cheaper_points_resolved_less_recoverable'),
                                          'secondary_more_spending': f.get('secondary_more_spending'), 'pass': f.get('passes'),
                                          'primary_cost_differences': f.get('primary')}}
        chk[f'D2_{key}_equals_verifier'] = (x['label'] == lab_mine and x['point'] == f.get('bracket') and close(x['benefit_mean'], f.get('benefit_mean'))
                                            and close(x['benefit_se'], f.get('benefit_se_agg')) and x['recipient_gate_vs_bracket'] == f.get('c2_recipient_gate_vs_Q')
                                            and sorted(x['dominated_by_cheaper'] or []) == sorted(f.get('c3_cheaper_points_resolved_less_recoverable') or [])
                                            and sorted(x['secondary_more_spending'] or []) == sorted(f.get('secondary_more_spending') or [])
                                            and x['pass'] == bool(f.get('passes')))
    tg_d, tg_m = d['D2']['targeting'], G['targeting']
    cmpd['D2_targeting'] = {'decisions': tg_d, 'verifier': tg_m}
    chk['D2_targeting_equals_verifier'] = sorted(tg_d) == sorted(tg_m) and all(
        tg_d[q]['pass'] == tg_m[q]['pass'] and all(close(tg_d[q]['version_means'][str(v)], tg_m[q]['version_means'][str(v)], 1e-12)
                                                  for v in (0, 1)) for q in tg_d)
    # D3 and D4
    pr = d['D3']['per_reference']
    ok3 = True
    for r, x in pr.items():
        m = gc[r]
        ok3 &= (x['benefit'] == m['benefit_label'] and close(x['mean'], m.get('mean')) and close(x['se_agg'], m.get('se_agg'))
                and x['recipient_gate_pass'] == m['recipient_gate_pass'])
    cmpd['D3'] = {'decisions_passed': d['D3']['passed'], 'decisions_failing': d['D3']['failing'], 'verifier_passed': mine['D3_passed'],
                  'verifier_failing': mine['D3_failing']}
    chk['D3_per_reference_equals_verifier'] = bool(ok3) and len(pr) == 9
    chk['D3_lists_equal_verifier'] = sorted(d['D3']['passed']) == sorted(mine['D3_passed']) and sorted(d['D3']['failing']) == sorted(mine['D3_failing'])
    chk['reader_budget_all_comparable (credit rule inactive)'] = all(x == 'READER_BUDGET_COMPARABLE' for x in (D.get('reader_budget') or {}).values())
    t4 = d['D4']['per_reference']
    cmpd['D4'] = {'decisions': t4, 'verifier': mine['D4_target_screens']}
    chk['D4_target_screens_equal_verifier'] = all(t4[r]['label'] == mine['D4_target_screens'][r]['label'] for r in ('NEURAL_RAW', 'H_LJ'))
    # ---- leg 2: frozen rules on the campaign record, unchanged and with the verifier's inputs
    from rfa import claims as CL
    from rfa import rfclaims as RFC
    from rfa import rules as RU
    plan = json.loads((RES / 'claims' / 'PLAN_RFA_v1.json').read_text())
    CL.reader_budget_flags = lambda bank, cand, refs=CL.REFS: dict(D['reader_budget'])        # no release-array re-read
    grid = CL.load_grid('RFA1_MAIN')
    lc_grid = CL.load_grid('RFA1_LC')
    gv = bool(D['grid_validity']['effective'])
    rec = RFC.build(plan, grid, 'RFA1_MAIN', lc_grid=lc_grid, lc_bank='RFA1_LC', gv_effective=gv)
    rec['grid_valid'] = gv
    rep = RU.decisions(copy.deepcopy(rec))
    labels = lambda x: {k: (x[k] or {}).get('label') for k in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'LC')} | {'class': x['outcome_class']['label']}
    cmpd['frozen_rules_replay'] = labels(rep)
    chk['frozen_rules_replay_equals_DECISIONS'] = labels(rep) == labels(d)
    sub = copy.deepcopy(rec)
    ent, nt = G['inputs']['entries'], G['inputs']['nontable']
    sub['grid'] = {k: ent[k] for k in rec['grid']}
    sub['nolookup'] = {k: (nt[k] if k in nt else rec['nolookup'][k]) for k in rec['nolookup']}
    missing = [k for k in rec['grid'] if k not in ent] + [k for k in rec['nolookup'] if k not in nt]
    with_mine = RU.decisions(sub)
    cmpd['frozen_rules_on_verifier_inputs'] = labels(with_mine)
    cmpd['substitution_missing_keys'] = missing
    chk['frozen_rules_on_verifier_inputs_equal_DECISIONS'] = labels(with_mine) == labels(d) and not missing
    # input agreement (campaign grid entries vs the verifier's): max abs differences of the substituted cells
    worst = 0.0
    for k in rec['grid']:
        for view, e in rec['grid'][k].items():
            for c, x in e['cells'].items():
                worst = max(worst, abs(x - ent[k][view]['cells'][c]), abs(e['cell_se'][c] - ent[k][view]['cell_se'][c]))
    wn = 0.0
    for k, e in rec['nolookup'].items():
        if e and k in nt:
            for c, x in e['cells'].items():
                wn = max(wn, abs(x - nt[k]['cells'][c]))
    res['campaign_vs_verifier_inputs_max_abs'] = {'grid_entries': worst, 'nontable': wn}
    chk['campaign_record_inputs_equal_verifier_inputs (abs <= 1e-12)'] = worst <= TOL and wn <= TOL
    # ---- income wire characterization (arrays; no label)
    from ohn import common as OC
    inc = {}
    for v in (0, 1):
        with np.load(OC.MIRROR / f'admitted/bank/FROZEN_BANK_S{v}.npz', allow_pickle=False) as z:
            nr = np.asarray(z['NEURAL_RAW__MODEL_SELECT__income'], dtype=np.float64)
            nr_dtype = str(z['NEURAL_RAW__MODEL_SELECT__income'].dtype)
        with np.load(RUNS / 'FAM' / 'RF_FULL' / f'V{v}' / 'RELEASE.npz', allow_pickle=False) as z:
            rf = np.asarray(z['MODEL_SELECT__income'], dtype=np.float64)
        f32 = lambda a: float(np.mean(a.astype(np.float32).astype(np.float64) == a))
        inc[str(v)] = {'bank_dtype': nr_dtype, 'bank_NEURAL_RAW_float32_representable_share': f32(nr), 'RF_FULL_float32_representable_share': f32(rf),
                       'max_abs_bank_minus_RF_FULL': float(np.max(np.abs(nr - rf))),
                       'max_abs_bank_minus_float32_rounded_RF_FULL': float(np.max(np.abs(nr - rf.astype(np.float32).astype(np.float64)))),
                       'decision_agreement': float(np.mean(nr.argmax(1) == rf.argmax(1)))}
    res['income_wire'] = inc
    res['all_pass'] = all(bool(x) for x in chk.values())
    res['cpu_s'] = time.process_time() - t0
    res['code_sha256'] = {f: sha256(HERE / f) for f in ('v2_decisions.py',)}
    if OUT.exists():
        raise PermissionError(f'{OUT.name} exists (write-once)')
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'all_pass': res['all_pass'], 'failed': [k for k, x in chk.items() if not x], 'labels': {k: cmpd[k] for k in ('D1', 'D2', 'D3', 'D4')},
                      'income_wire': inc, 'cpu_s': res['cpu_s']}, indent=1, default=str))
    return 0 if res['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
