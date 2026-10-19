"""Verifier stage V5a: the nested cost tables and the strength selection, replayed with the verifier's own code.

PCRL controlled training-size campaign (TSZ1; exploratory development on previously used ACS 2019 Florida PUMS person records).
For each version and arm with a nested table (.private/<campaign>/runs/NESTED/V<v>/NESTED_<arm>.json + .npz, npz hash-checked):
  - the pooled out-of-fold witness and correction arrays (witness_oof, delta_oof) and the outer-fold ids are read; the fold ids are
    compared with the verifier's own TSZ1_ncv10 folds; row ids with MODEL_FIT;
  - MODEL_FIT occupation and SEX (all 20,000 people, subset S100) are read through tsz.common.load operation 'verify';
  - per registered strength s: own release r_s = softmax(centred log p_W + s Delta) (s = 0: p_W exactly), own per-person increments
    against s = 0 (summed Brier, NLL with the 1e-12 floor, accuracy loss), own pooled means and household-clustered SEs (subgroup SE
    for the SEX groups), compared with the table (tolerance 1e-12);
  - own fixed-sequence selection (U = estimate + 2 SE <= allowance; RULES.json allowance of version v) on the table's numbers AND on
    the verifier's own numbers, compared with results/<campaign>/conditional/SELECTION_V<v>.json (selected strength, first failing
    strength, PASS_AFTER_FAIL), and the outer-fold training sizes with the verifier's reconstruction.

    python3 -m tsz.jobqueue run --unit VER_SELECT_V1 --stage verification --rss-gib 3 -- python3 .../v5_select.py --tag V1
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

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from tsz import common as C  # noqa: E402   (the logged label-access layer; operation 'verify')

RES = V.RESULTS
PRIV = ROOT / '.private' / 'pcrl_controlled_training_size_v1'
NESTED = PRIV / 'runs' / 'NESTED'


def skey(s):
    return repr(float(s))


def release(pw, dn, s):
    if float(s) == 0.0:
        return np.array(pw, dtype=np.float64, copy=True)
    lp = np.log(np.maximum(np.asarray(pw, dtype=np.float64), 1e-300))
    lp = lp - lp.mean(axis=1, keepdims=True)
    return V.softmax(lp + float(s) * np.asarray(dn, dtype=np.float64))


def increments(p, p0, y):
    a, b = V.person_vectors(y, p, 5), V.person_vectors(y, p0, 5)
    return {'brier': a['brier'] - b['brier'], 'nll': a['nll'] - b['nll'], 'acc_loss': b['accuracy'] - a['accuracy']}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--arms', default='LOC,PAIR,CONT')
    a = ap.parse_args()
    F = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'verify',
               'independent verifier V5: replay of the nested out-of-fold cost tables and the fixed-sequence strength rule with own code',
               subset='S100')
    y, sex, unit, row_id = np.asarray(F['occupation']), np.asarray(F['sex']), np.asarray(F['unit']), np.asarray(F['row_id'])
    men, women = sex == 0, sex == 1
    my_ncv = V.folds(unit, 10, 'TSZ1_ncv10')
    rules = json.loads((RES / 'roles' / 'statistical_reviewer' / 'RULES.json').read_text())
    subsets = json.loads((RES / 'SUBSETS.json').read_text())
    checks, inputs_all = [], []
    for v in (0, 1):
        allow = {j: float(rules['selection']['allowance'][str(v)][j]) for j in V.STATS5}
        sel_p = RES / 'conditional' / f'SELECTION_V{v}.json'
        sel = json.loads(sel_p.read_text()) if sel_p.is_file() else None
        for arm in [x for x in a.arms.split(',') if x]:
            jp, zp = NESTED / f'V{v}' / f'NESTED_{arm}.json', NESTED / f'V{v}' / f'NESTED_{arm}.npz'
            if not jp.is_file():
                continue
            rec = json.loads(jp.read_text())
            if V.file_sha(zp) != rec['npz_sha256']:
                raise PermissionError(f'NESTED_{arm}.npz differs from its record')
            with np.load(zp) as z:
                pw, dn, ncv, rid = z['witness_oof'].copy(), z['delta_oof'].copy(), z['ncv_fold'].copy(), z['row_id'].copy()
            ok_rows = bool(np.array_equal(rid, row_id)) and bool(np.array_equal(ncv, my_ncv))
            strengths = [float(s) for s in rec['strengths']]
            p0 = release(pw, dn, 0.0)
            est, se, diffs = {}, {}, {}
            for s in strengths:
                inc = increments(release(pw, dn, s), p0, y)
                est[s] = {'men_brier': float(inc['brier'][men].mean()), 'women_brier': float(inc['brier'][women].mean()),
                          'brier': float(inc['brier'].mean()), 'nll': float(inc['nll'].mean()), 'acc_loss': float(inc['acc_loss'].mean())}
                se[s] = {'men_brier': V.subgroup_se(inc['brier'], men, unit), 'women_brier': V.subgroup_se(inc['brier'], women, unit),
                         'brier': V.cluster_se(inc['brier'], unit), 'nll': V.cluster_se(inc['nll'], unit), 'acc_loss': V.cluster_se(inc['acc_loss'], unit)}
                row = rec['per_strength'][skey(s)]
                diffs[skey(s)] = {'max_est_abs_diff': max(abs(est[s][j] - row['stats'][j]) for j in V.STATS5),
                                  'max_se_abs_diff': max(abs(se[s][j] - row['se'][j]) for j in V.STATS5)}
            tab_est = {float(k): row['stats'] for k, row in rec['per_strength'].items()}
            tab_se = {float(k): row['se'] for k, row in rec['per_strength'].items()}
            mine_on_table = V.select_strength(tab_est, tab_se, allow, strengths=tuple(strengths))
            mine_own = V.select_strength(est, se, allow, strengths=tuple(strengths))
            train_sizes = {str(k): {'people': int((my_ncv != k).sum()), 'households': int(len(np.unique(unit[my_ncv != k])))} for k in range(10)}
            rec_train = {k: f['training'] for k, f in rec.get('folds', {}).items()}
            sizes_ok = rec_train == train_sizes and [train_sizes[str(k)]['people'] for k in range(10)] == subsets['nested_selection_folds']['training_people_per_outer_fold']
            max_e = max(x['max_est_abs_diff'] for x in diffs.values()); max_s = max(x['max_se_abs_diff'] for x in diffs.values())
            obs = {'rows_and_folds_equal': ok_rows, 'strengths': strengths, 'strengths_registered': tuple(strengths) == V.STRENGTHS,
                   'identity_row_zero': all(rec['per_strength'][skey(0.0)]['stats'][j] == 0.0 for j in V.STATS5),
                   'max_est_abs_diff': max_e, 'max_se_abs_diff': max_s, 'per_strength_diffs': diffs,
                   'selection_on_table_numbers': {k: mine_on_table[k] for k in ('selected', 'first_failing_strength', 'binding_statistic', 'flags',
                                                                                  'pass_after_fail_strengths')},
                   'selection_on_own_numbers': {k: mine_own[k] for k in ('selected', 'first_failing_strength', 'binding_statistic', 'flags')},
                   'trail_on_own_numbers': {skey(s): {j: {kk: mine_own['trail'][s][j][kk] for kk in ('estimate', 'se', 'U', 'U_over_allowance', 'pass')}
                                                      for j in V.STATS5} for s in strengths if s > 0},
                   'outer_training_sizes_equal': sizes_ok, 'targets': rec.get('targets'), 'guard': rec.get('guard'), 'n_pooled': rec.get('pooled')}
            exp = None
            if sel is not None and arm in sel.get('selected', {}):
                aud = sel['audit'][arm]
                exp = {'selected': sel['selected'][arm], 'first_failing_strength': aud.get('first_failing_strength'),
                       'pass_after_fail_strengths': aud.get('pass_after_fail_strengths'), 'flags': aud.get('flags'),
                       'nested_json_sha256': sel['nested_tables'][arm]['json_sha256']}
                obs['selection_json_sha_matches_table'] = sel['nested_tables'][arm]['json_sha256'] == V.file_sha(jp)
            agree_sel = exp is None or (mine_on_table['selected'] == exp['selected'] and mine_own['selected'] == exp['selected']
                                        and mine_on_table['first_failing_strength'] == exp['first_failing_strength']
                                        and mine_on_table['pass_after_fail_strengths'] == exp['pass_after_fail_strengths']
                                        and obs.get('selection_json_sha_matches_table', True))
            ok = (ok_rows and obs['strengths_registered'] and obs['identity_row_zero'] and max_e <= 1e-12 and max_s <= 1e-12 and sizes_ok
                  and agree_sel and rec.get('guard') == 'none' and rec.get('targets') == {'men': 0.005, 'women': 0.005})
            near = (ok_rows and max_e <= 1e-9 and max_s <= 1e-9 and agree_sel)
            inputs = [jp, zp] + ([sel_p] if sel is not None else [])
            inputs_all += inputs
            checks.append({'id': f'V5.NESTED_AND_SELECTION.V{v}.{arm}__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
                           'what': f'version {v}, arm {arm}: the pooled nested out-of-fold cost table (estimates and household-clustered SEs at '
                                   'every registered strength) recomputed from the stored out-of-fold witness and correction arrays with own '
                                   'release, metric and SE code; own fixed-sequence selection (allowance of version v, z = 2) on the table and '
                                   'on own numbers vs SELECTION_V<v>.json' + ('' if arm != 'CONT' else ' (CONT: descriptive, no selection)'),
                           'inputs': V.inputs_sha(inputs), 'expected': exp, 'observed': obs,
                           'status': 'PASS' if ok else ('ATTENTION' if near else 'FAIL'), 'code_sha256': V.code_hashes(__file__, V.__file__)})
    V.write_record(f'V5_SELECT__{a.tag}.json', {'checks': checks, 'utc': V.utc()})
    print(json.dumps(V.append_checks(checks)))
    print(json.dumps({c['id']: [c['status'], c['observed']['selection_on_table_numbers']['selected'], c['observed']['selection_on_own_numbers']['selected'],
                                (c['expected'] or {}).get('selected'), c['observed']['max_est_abs_diff'], c['observed']['max_se_abs_diff']] for c in checks}, indent=0))


if __name__ == '__main__':
    main()
