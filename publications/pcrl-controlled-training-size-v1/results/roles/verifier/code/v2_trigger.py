"""Verifier stage V2: independent trigger inputs (non-gating cross-check of results/pcrl_controlled_training_size_v1/trigger/TRIGGER.json).

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records;
MODEL_SELECT is adaptively used development data, not confirmation).

From MODEL_SELECT labels (income, occupation, SEX; read once through tsz.common.load operation 'verify'), the frozen references
U0_S<v> / Ucal_S<v> (ohn.common.frozen_bank, row order checked), the frozen H3 income S<v> (ohn.devrun.anchor_income; its wire hash
recomputed by the source wire definition and compared with the admitted RELEASE.json) and the W100_v occupation release
(runs/SIZE/A1__S100__V<v>/RELEASE.npz, hash-checked against FIT.json), the verifier's own code computes the 12 clauses + 2 guards per
version (values, household-clustered paired SEs, linearized subgroup SEs), the reserve on all 14 conditions with the OHN1 H_LJ floor
table (k = 2), the 8 occupation-side ratios, minimum and binding condition, the income side separately, and the trigger decision
T0-T3 (T3 from the verifier's own DEPLOY_W100.json), then compares everything with TRIGGER.json.

    python3 -m tsz.jobqueue run --unit VER_TRIGGER_V1 --stage verification --rss-gib 2 -- python3 .../v2_trigger.py --tag V1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (the logged label-access layer; operation 'verify')

RES = V.RESULTS
PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
MIRROR = ROOT / '.private' / 'pcrl_income_headroom_successor_v1'


def wire_sha(p):
    """Source definition of one recipient's wire hash: SHA-256 of float64 probability bytes, then int64 argmax bytes."""
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes())
    h.update(np.ascontiguousarray(p.argmax(1).astype(np.int64)).tobytes())
    return h.hexdigest()


def w100_release(v):
    d = PRIV / 'runs' / 'SIZE' / f'A1__S100__V{v}'
    rec = json.loads((d / 'FIT.json').read_text())
    if V.file_sha(d / 'RELEASE.npz') != rec['files_sha256']['RELEASE.npz']:
        raise PermissionError('RELEASE.npz differs from FIT.json')
    with np.load(d / 'RELEASE.npz') as z:
        return z['MODEL_SELECT'].copy(), rec, [d / 'RELEASE.npz', d / 'FIT.json']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
               'independent verifier V2: recompute the W100 trigger inputs (24 clauses + 4 guards, reserve, trigger) with own metric code')
    from ohn import common as OC
    from ohn import devrun as OD
    floors = json.loads((ROOT / V.OHN1_RULES).read_text())['reserve']['floor']['se']
    if V.file_sha(ROOT / V.OHN1_RULES) != V.OHN1_RULES_SHA:
        raise SystemExit('OHN1 RULES.json hash differs')
    trig_p = RES / 'trigger' / 'TRIGGER.json'
    trig = json.loads(trig_p.read_text())
    dep_p = V.VER_DIR / 'DEPLOY_W100.json'
    dep = json.loads(dep_p.read_text())
    unit = np.asarray(S['unit']); sex = np.asarray(S['sex'])
    labels = {'income': np.asarray(S['income']), 'occupation': np.asarray(S['occupation'])}
    G = len(np.unique(unit))
    mine, inputs, cmp_rows, t0_ok, res_by_v, h3 = {}, [trig_p, dep_p, ROOT / V.OHN1_RULES], {}, {}, {}, {}
    for v in (0, 1):
        bank, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('frozen bank MODEL_SELECT row order differs')
        refs = {r: {t: np.asarray(bank[r]['MODEL_SELECT'][t], dtype=np.float64) for t in ('income', 'occupation')} for r in ('U0', 'Ucal')}
        inc = np.asarray(OD.anchor_income('H3', v, 'MODEL_SELECT'), dtype=np.float64)
        rel_json = json.loads((MIRROR / 'runs' / 'H' / 'H3_STRONG_LJDIR_PB' / f'S{v}' / 'RELEASE.json').read_text())
        with np.load(MIRROR / 'runs' / 'H' / 'H3_STRONG_LJDIR_PB' / f'S{v}' / 'RELEASE.npz') as z:
            inc_file = z['MODEL_SELECT__income'].copy()
        h3[v] = {'wire_sha256_mine': wire_sha(inc), 'wire_sha256_admitted_RELEASE_json': rel_json['wire_sha256']['MODEL_SELECT__income'],
                 'anchor_income_equals_release_file': bool(np.array_equal(inc, inc_file)),
                 'release_npz_sha256': V.file_sha(MIRROR / 'runs' / 'H' / 'H3_STRONG_LJDIR_PB' / f'S{v}' / 'RELEASE.npz'),
                 'arr_sha_mine': V.arr_sha(inc), 'arr_sha_trigger_record': trig['h3_income_wire_sha256'][str(v)]}
        h3[v]['ok'] = (h3[v]['wire_sha256_mine'] == h3[v]['wire_sha256_admitted_RELEASE_json'] and h3[v]['anchor_income_equals_release_file']
                       and h3[v]['arr_sha_mine'] == h3[v]['arr_sha_trigger_record'])
        occ, rec, files = w100_release(v)
        inputs += files + [MIRROR / 'runs' / 'H' / 'H3_STRONG_LJDIR_PB' / f'S{v}' / 'RELEASE.npz']
        conds = V.clauses_and_guards(labels, {'income': inc, 'occupation': occ}, refs, sex, unit)
        res_all = V.reserve(conds, floors[str(v)], G, names=V.ALL_CONDITIONS)
        res_occ = V.reserve(conds, floors[str(v)], G, names=V.OCC_SIDE)
        res_inc = V.reserve(conds, floors[str(v)], G, names=V.INCOME_SIDE)
        res_by_v[v] = res_occ
        t0_ok[v] = bool(len(conds) == 14 and all(math.isfinite(c['value']) and math.isfinite(c['se']) for c in conds.values()) and h3[v]['ok'])
        mine[v] = {'conditions': conds, 'occupation_side': res_occ, 'income_side': {'min_ratio': res_inc['min_ratio'], 'binding': res_inc['binding'],
                                                                                     'met': res_inc['met'], 'failed': res_inc['failed']},
                   'whole_release_met': res_all['met'], 'whole_release_min_ratio': res_all['min_ratio'],
                   'w100_MODEL_SELECT_arr_sha256': V.arr_sha(occ), 'T_star': rec['T_star'], 'households': G, 'people': int(len(unit))}
        # comparison with the coordinator's record
        u = trig['evaluations'][str(v)]['utility']
        theirs = {f"{c['task']}|{c['reference']}|{c['metric']}": c for c in u['clauses']}
        theirs.update({f"guard|{g['group']}": g for g in u['guards']})
        diffs = {c: {'value_abs_diff': abs(conds[c]['value'] - float(theirs[c]['value'])), 'se_abs_diff': abs(conds[c]['se'] - float(theirs[c]['se']))}
                 for c in conds}
        occ_t = trig['trigger']['parts']['T2']['detail'][str(v)]
        cmp_rows[v] = {'max_value_abs_diff': max(x['value_abs_diff'] for x in diffs.values()),
                       'max_se_abs_diff': max(x['se_abs_diff'] for x in diffs.values()),
                       'min_occ_ratio_mine': res_occ['min_ratio'], 'min_occ_ratio_record': occ_t['min_ratio'],
                       'binding_mine': res_occ['binding'], 'binding_record': occ_t['binding'],
                       'failed_mine': res_occ['failed'], 'failed_record': occ_t['failed'],
                       'point_failed_mine': res_occ['point_failed'], 'point_failed_record': trig['trigger']['parts']['T1']['detail'][str(v)],
                       'release_sha_mine': V.arr_sha(occ), 'release_sha_record': trig['witness'][str(v)]['release_sha256_MODEL_SELECT'],
                       'T_star_mine_record': [rec['T_star'], trig['witness'][str(v)]['T_star']]}
    dep_by_v = {v: dep.get(str(v)) for v in (0, 1)}
    t = V.trigger(res_by_v, t0_ok, dep_by_v)
    rec_dec = trig['trigger']['decision']
    tol = 1e-12
    agree = {v: (cmp_rows[v]['max_value_abs_diff'] <= tol and cmp_rows[v]['max_se_abs_diff'] <= tol
                 and abs(cmp_rows[v]['min_occ_ratio_mine'] - cmp_rows[v]['min_occ_ratio_record']) <= 1e-9
                 and cmp_rows[v]['binding_mine'] == cmp_rows[v]['binding_record'] and cmp_rows[v]['failed_mine'] == cmp_rows[v]['failed_record']
                 and cmp_rows[v]['point_failed_mine'] == cmp_rows[v]['point_failed_record'] and cmp_rows[v]['release_sha_mine'] == cmp_rows[v]['release_sha_record']
                 and cmp_rows[v]['T_star_mine_record'][0] == cmp_rows[v]['T_star_mine_record'][1]) for v in (0, 1)}
    parts_agree = {k: t['parts'][k] == trig['trigger']['parts'][k]['pass'] for k in ('T0', 'T1', 'T2', 'T3')}
    code = V.code_hashes(__file__, V.__file__)
    occ_table = {v: {c: {k: mine[v]['occupation_side']['conditions'][c][k] for k in ('value', 'own_se', 'floor_se', 'se_used', 'margin', 'ratio', 'rho', 'met', 'point_pass', 'se_source')}
                     for c in V.OCC_SIDE} for v in (0, 1)}
    checks = [
        {'id': f'V2.TRIGGER.CONDITIONS_AND_RESERVE__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'W100_v (v = 0, 1) with the frozen H3 income S<v> on MODEL_SELECT: the 12 clauses and 2 guards per version (values and '
                 'household-clustered / linearized subgroup SEs), the reserve (k = 2, OHN1 H_LJ floors, 5,306 households), the 8 '
                 'occupation-side ratios, minimum and binding condition, recomputed with the verifier\'s own code and compared with '
                 'TRIGGER.json (tolerance 1e-12 on values and SEs, 1e-9 on ratios; identical binding, failing and point-failing lists, '
                 'identical W100 release hash and T*)',
         'inputs': V.inputs_sha(inputs), 'expected': {str(v): {k: cmp_rows[v][k.replace('_mine', '_record')] if k.endswith('_mine') else None
                                                              for k in ('min_occ_ratio_mine', 'binding_mine', 'failed_mine')} for v in (0, 1)},
         'observed': {'comparison': cmp_rows, 'occupation_side': occ_table,
                      'income_side': {v: mine[v]['income_side'] for v in (0, 1)},
                      'whole_release': {v: {'met': mine[v]['whole_release_met'], 'min_ratio': mine[v]['whole_release_min_ratio']} for v in (0, 1)}},
         'status': 'PASS' if all(agree.values()) else 'FAIL', 'code_sha256': code},
        {'id': f'V2.TRIGGER.H3_INCOME_WIRE__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'T0 income part: the H3 income S<v> MODEL_SELECT wire used by the evaluation equals the admitted wire (source wire hash '
                 'definition: SHA-256 of float64 probabilities then int64 argmax, vs the admitted RELEASE.json wire_sha256), equals the '
                 'admitted RELEASE.npz array, and its array hash equals the one recorded in TRIGGER.json',
         'inputs': V.inputs_sha([trig_p]), 'expected': 'all equal', 'observed': h3,
         'status': 'PASS' if all(h3[v]['ok'] for v in (0, 1)) else 'FAIL', 'code_sha256': code},
        {'id': f'V2.TRIGGER.DECISION__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
         'what': 'the trigger decision T0-T3 recomputed by the verifier (T0: 14 finite conditions per version and H3 wire ok; T1: 8 '
                 'occupation-side point caps per version; T2: 8 occupation-side reserve conditions per version; T3: the verifier\'s own '
                 'DEPLOY_W100.json) vs TRIGGER.json',
         'inputs': V.inputs_sha([trig_p, dep_p]), 'expected': {'decision': rec_dec, 'parts': {k: trig['trigger']['parts'][k]['pass'] for k in t['parts']}},
         'observed': {'decision': t['decision'], 'parts': t['parts'], 'parts_agree': parts_agree,
                      'min_occ_ratio': {v: res_by_v[v]['min_ratio'] for v in (0, 1)}, 'binding': {v: res_by_v[v]['binding'] for v in (0, 1)},
                      'usable_slack_rho': {v: {c: res_by_v[v]['conditions'][c]['rho'] for c in V.OCC_SIDE} for v in (0, 1)},
                      'deployment_record_sha256': V.file_sha(dep_p), 'deployment_record_sha_in_trigger': trig['deployment_record']['sha256']},
         'status': 'PASS' if (t['decision'] == rec_dec and all(parts_agree.values()) and V.file_sha(dep_p) == trig['deployment_record']['sha256']) else 'FAIL',
         'code_sha256': code},
    ]
    V.write_record(f'V2_TRIGGER__{a.tag}.json', {'checks': checks, 'mine': mine, 'utc': V.utc()})
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({c['id']: c['status'] for c in checks}))
    print(json.dumps({'decision': t['decision'], 'parts': t['parts'], 'cmp': cmp_rows, 'h3_ok': {v: h3[v]['ok'] for v in (0, 1)}}, default=str)[:3000])


if __name__ == '__main__':
    main()
