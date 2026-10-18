"""Verifier replay of the development-recipe gates G1-G3 (RULES.md section 8) and of the recipe-selection outcome, from the saved
selections, the verifier's own realized costs and its own complete-release point check on MODEL_SELECT.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
G1 nonzero selected strength; G2 no realized reserve crossing on D CHECK and on MODEL_SELECT at the selected strength (the released
vector against the deployed witness W, whose MODEL_SELECT margins define rho: full r_s vs W; ens e_s vs W as amended in
PROTOCOL_DEVIATIONS item 6); G3 the complete release (H3 income + corrected occupation) passes its seed's 12 clauses and 2 guards at
the point on MODEL_SELECT (occupation side: witness value + realized increment <= cap; income side: H3 income clauses recomputed
here from MODEL_SELECT income labels and the frozen-bank U0 / Ucal income outputs). Labels via cts.common.load('verify').

    python3 .../v_gates.py --tag V1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402
from v_costs import frozen_refs, my_conditions, npz  # noqa: E402  (verifier's own helpers)

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402

TRACKF = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1' / 'runs' / 'TRACKF'
RES = ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'
DPARTS = ('D0', 'D1')
RULES = ('SIMPLE', 'UNCERTAINTY', 'OLD-B50', 'OLD-B80')


def labels(P):
    chk = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'verify',
                 f'independent verifier: replay of recipe gates G2 / G3 on {P} diagnostic CHECK', partition=P, prole='CHECK')
    return chk


def ms_labels():
    return C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
                  'independent verifier: replay of recipe gates G2 / G3 on MODEL_SELECT (occupation increments, H3 income clauses)')


def income_clauses(seed, inc, S):
    from ohn import common as OC
    bank, ids = OC.frozen_bank(seed, roles=('MODEL_SELECT',))
    if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
        raise AssertionError('row order')
    y = S['income']
    own = metrics2(y, inc)
    out = {}
    for ref in ('U0', 'Ucal'):
        r = metrics2(y, np.asarray(bank[ref]['MODEL_SELECT']['income'], dtype=np.float64))
        for m, d in (('accuracy_loss', r['accuracy'] - own['accuracy']), ('nll_excess', own['nll'] - r['nll']),
                     ('brier_excess', own['brier'] - r['brier'])):
            out[f'income|{ref}|{m}'] = {'value': float(d.mean()), 'cap': V.CAP[m], 'pass': bool(d.mean() <= V.CAP[m])}
    return out


def metrics2(y, p):
    y = np.asarray(y, dtype=np.int64); p = np.asarray(p, dtype=np.float64)
    idx = np.arange(len(y))
    onehot = np.zeros_like(p); onehot[idx, y] = 1.0
    return {'accuracy': (np.argmax(p, 1) == y).astype(np.float64), 'nll': -np.log(np.maximum(p[idx, y], V.NLL_FLOOR)),
            'brier': np.sum((p - onehot) ** 2, 1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    spec = json.loads((RES / 'PARTITIONS.json').read_text())
    rules = json.loads((RES / 'roles' / 'statistical_reviewer' / 'RULES.json').read_text())
    gates_rec = json.loads((RES / 'diagnosis' / 'RECIPE_GATES.json').read_text())
    sel_rec = json.loads((RES / 'diagnosis' / 'RECIPE_SELECTION.json').read_text())
    S = ms_labels()
    per_part = {}
    for P in DPARTS:
        seed = int(spec['partitions'][P]['associated_source_seed'])
        L = labels(P)
        W = npz(TRACKF / P / 'WITNESS' / 'WITNESS.npz')
        refs = frozen_refs(seed, S['row_id'])
        conds = my_conditions(W['dep__MODEL_SELECT'], refs, S['occupation'], S['sex'], S['unit'])
        base = V.baseline_from_conditions(conds, rules['reserve']['floor_se'][str(seed)], len(np.unique(S['unit'])))
        from ohn import devrun as OD
        inc = OD.anchor_income('H3', seed, 'MODEL_SELECT')
        ic = income_clauses(seed, inc, S)
        per_part[P] = {'L': L, 'W': W, 'conds': conds, 'base': base, 'income_pass': all(v['pass'] for v in ic.values()),
                       'income_clauses': ic}
    mine, mism = {}, []
    for routing in ('REFIT', 'FROZEN'):
        for arm in ('PAIR', 'LOC'):
            did = f'{routing}_{arm}'
            for dep in ('full', 'ens'):
                for rule in RULES:
                    rid = f'{did}@{rule}|{dep}'
                    rec = {'D': {}}
                    for P in DPARTS:
                        pp = per_part[P]
                        s = float(json.loads((TRACKF / P / 'SELECT' / f'{did}.json').read_text())['selected'][f'{rule}|{dep}'])
                        D = npz(TRACKF / P / 'DIR' / did / 'DELTA.npz')
                        W, L, base = pp['W'], pp['L'], pp['base']
                        out = {}
                        for name in ('CHECK', 'MS'):
                            if name == 'CHECK':
                                r = L['rows']; X = L
                                pw, dl, pf, df = W['dep__MODEL_FIT'][r], D['full__MODEL_FIT'][r], W['fold__MODEL_FIT'][:, r], D['folds__MODEL_FIT'][:, r]
                            else:
                                X = S
                                pw, dl, pf, df = W['dep__MODEL_SELECT'], D['full__MODEL_SELECT'], W['fold__MODEL_SELECT'], D['folds__MODEL_SELECT']
                            p = V.release_full(pw, dl, s) if dep == 'full' else V.release_ensemble(pf, df, s)
                            c = V.incremental_costs(p, pw, X['occupation'], X['sex'], X['unit'])
                            out[name] = (c, V.outcome(c['stats'], base))
                        cms, oms = out['MS']
                        point_occ = all(pp['conds'][c]['value'] + cms['stats'][V.STAT_FOR[c]] <= pp['conds'][c]['cap'] for c in V.CONDITIONS)
                        ratios = sorted((base[c]['rho'] - cms['stats'][V.STAT_FOR[c]]) / base[c]['se_used'] for c in V.CONDITIONS)
                        rec['D'][P] = {'selected': s, 'G1': s > 0,
                                       'G2': not (out['CHECK'][1]['reserve_crossing'] or oms['reserve_crossing']),
                                       'G3': bool(point_occ and pp['income_pass']), 'G3_occupation_point': bool(point_occ),
                                       'min_ms_reserve_ratio': ratios[0],
                                       'crossing_conditions': sorted(set(out['CHECK'][1]['reserve_crossing_conditions']) |
                                                                     set(oms['reserve_crossing_conditions']))}
                        g = gates_rec['recipes'][rid]['D'][P]
                        rg = gates_rec['recipes'][rid]['failed_gates_G1_G3'][P]
                        my_fail = [n for n in ('G1', 'G2', 'G3') if not rec['D'][P][n]]
                        if my_fail != rg:
                            mism.append(f'{rid} {P}: mine {my_fail} vs recorded {rg}')
                        if abs(g['selected'] - s) > 0:
                            mism.append(f'{rid} {P}: selected')
                        rr = sorted(g['ms_reserve_ratios'])
                        if len(rr) != 8 or max(abs(x - y) for x, y in zip(rr, ratios)) > 1e-8:
                            mism.append(f'{rid} {P}: MS reserve ratios')
                    rec['passes_G1_G3_both'] = all(rec['D'][P][n] for P in DPARTS for n in ('G1', 'G2', 'G3'))
                    if rec['passes_G1_G3_both'] != gates_rec['recipes'][rid]['passes_G1_G3_both']:
                        mism.append(f'{rid}: survivor flag')
                    mine[rid] = rec
    survivors = sorted(r for r, v in mine.items() if v['passes_G1_G3_both'])
    outcome = 'NO_RECIPE' if not survivors else 'NEEDS_G4_AND_SCORE'
    sel_ok = (outcome == 'NO_RECIPE' and sel_rec['outcome'] == 'NO_RECIPE' and sel_rec['selected'] is None) or \
        (outcome != 'NO_RECIPE' and sel_rec['outcome'] != 'NO_RECIPE')
    g2_neg = {P: sorted(c for c in V.CONDITIONS if per_part[P]['base'][c]['rho'] <= 0) for P in DPARTS}
    g2_only_identity = sorted(f'{rid} {P}' for rid, v in mine.items() for P in DPARTS
                              if not v['D'][P]['G2'] and v['D'][P]['selected'] == 0.0)
    chk = {'id': f'C6.1.RECIPE_GATES_AND_SELECTION__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': 'recipe gates G1-G3 for all 32 recipes (routing x arm x deployment x rule) on D0 and D1, the MODEL_SELECT reserve '
                   'ratios of the tie-break, the survivor list and the recipe-selection outcome',
           'method': 'own realized increments (released vector against the deployed witness W) on diagnostic CHECK and MODEL_SELECT, own '
                     'crossing / point tests against the own witness baseline, own H3 income clauses; compared with RECIPE_GATES.json and '
                     'RECIPE_SELECTION.json',
           'status': 'PASS' if (not mism and sel_ok and survivors == sorted(gates_rec['survivors_G1_G3'])) else 'FAIL',
           'numbers': {'mismatches': mism[:40], 'survivors_mine': survivors, 'survivors_recorded': gates_rec['survivors_G1_G3'],
                       'selection_outcome_recorded': sel_rec['outcome'], 'selection_outcome_implied': outcome,
                       'income_clauses_pass': {P: per_part[P]['income_pass'] for P in DPARTS},
                       'income_clause_values': {P: {k: round(v['value'], 6) for k, v in per_part[P]['income_clauses'].items()} for P in DPARTS},
                       'non_positive_reserve_slack_conditions': g2_neg,
                       'G2_failures_at_identity_selections': g2_only_identity,
                       'per_recipe': {rid: {P: {k: v['D'][P][k] for k in ('selected', 'G1', 'G2', 'G3', 'crossing_conditions')}
                                            for P in DPARTS} for rid, v in mine.items()}}}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__, HERE / 'v_costs.py')
    V.write_record(f'V_GATES__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'mismatches': mism[:10], 'survivors': survivors,
                      'G2_identity_failures': len(g2_only_identity), 'neg_slack': g2_neg}))


if __name__ == '__main__':
    main()
