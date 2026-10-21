"""V5d revision R1 (UNT) bracket replay (label-free), own code.

From results/<campaign>/fa/UNT_SELECTION_V<v>.json, the P1 record SELECTION_V<v>.json and the saved models:
- the registered f_O grid (card FA_FRC_R1, typed here) and hash tag FRC_UNT_S<v>;
- the twin's nested means equal P1's (inc / occ NLL, <= 1e-12) and the target equals P1's PRC_S2 bracket target;
- PRC_S2's own income / occupation re-routing rates on the MODEL_FIT out-of-fold statistics relative to the twin, recomputed with
  my own routing (v5b_protected_tables.route_book) of the cached out-of-fold statistics through the saved PRC_S2 and twin books,
  and the ratio f_I / f_O = rate_income / rate_occupation;
- the bracket = the first f_O in grid order whose recorded nested total-NLL spend over the twin is >= the target (else
  CONTROL_DID_NOT_REACH_SPEND, nothing saved); the saved UNT_BR book carries f_O, f_I = min(1, f_O x ratio), the ratio and the tag;
- the recorded nested LOOKUP entries are internally consistent (diff = AUC_twin - AUC_entry).
The nested spends and AUCs themselves are not recomputed (no person-level R1 output is saved).

    python3 .../v5d_unt_selection.py --tag V1
    python3 .../v5d_unt_selection.py --selftest
"""
from __future__ import annotations
import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402
import v5b_protected_tables as T  # noqa: E402

GRID_F_O = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
TOL = 1e-12


def bracket(spend, target, grid=GRID_F_O):
    for f in grid:
        key = f'UNT@{f}'
        if key in spend and spend[key] >= target:
            return key
    return 'CONTROL_DID_NOT_REACH_SPEND'


def load_model(cfg, v):
    md = V.RUNS / 'FA' / cfg / f'V{v}'
    buf = (md / 'MODEL.pkl').read_bytes()
    ok = V.bytes_sha(buf) == json.loads((md / 'MODEL.json').read_text()).get('model_sha256')
    return pickle.loads(buf), ok


def run(tag):
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import frc.common as C
    from ohn import fam_a as FA
    F = C.features('MODEL_FIT')
    Z = FA.compact_po(np.asarray(F['X'], dtype=np.float64), tuple(map(str, F['feature_names'])))
    checks, record = [], {'tag': tag, 'utc': V.utc(), 'per': {}}
    for v in (0, 1):
        p = V.RESULTS / 'fa' / f'UNT_SELECTION_V{v}.json'
        if not p.is_file():
            continue
        u = json.loads(p.read_text())
        s = json.loads((V.RESULTS / 'fa' / f'SELECTION_V{v}.json').read_text())
        mm, proc = [], []
        if [float(x) for x in u['grid_f_O']] != list(GRID_F_O):
            proc.append('f_O grid differs from the registered grid')
        if u.get('hash_tag') != f'FRC_UNT_S{v}':
            proc.append('hash tag differs')
        tw = s['nested_means']['PRC_TWIN']
        d_tw = max(abs(u['twin_nested_means'][k] - tw[k]) for k in ('inc_nll', 'occ_nll'))
        if d_tw > TOL or u.get('twin_nested_means_equal_p1') is not True:
            mm.append(f'twin nested means differ from P1 by {d_tw}')
        if abs(u['target_total_nll_spend_prc_s2'] - s['brackets']['target_total_nll_spend']) > TOL:
            mm.append('target differs from the P1 PRC_S2 target')
        # PRC_S2's own re-routing rates on the out-of-fold statistics (own routing)
        task = pickle.loads((V.SRC_FA_CACHE / f'V{v}' / 'TASK.pkl').read_bytes())
        ss = pickle.loads((V.FA_CACHE / f'V{v}' / 'SEXSIDE.pkl').read_bytes())
        E = {'qI': np.asarray(task['oof']['inc'], dtype=np.float64), 'qO': np.asarray(task['oof']['occ'], dtype=np.float64), 'Z': Z,
             'pi': np.minimum(np.maximum(np.asarray(ss['oof']['sex'], dtype=np.float64)[:, 1], 0.01), 0.99),
             'qO0': np.asarray(ss['oof']['occ_s'][0], dtype=np.float64), 'qO1': np.asarray(ss['oof']['occ_s'][1], dtype=np.float64)}
        Mp, okp = load_model('PRC_S2', v)
        Mt, okt = load_model('TWB_C64', v)
        if not (okp and okt):
            proc.append('a model file differs from its record')
        kp, lp, _ = T.route_book(E, Mp['book'])
        kt, lt, _ = T.route_book(E, Mt['book'])
        rates = {'income': float((kp != kt).mean()), 'occupation': float((lp != lt).mean())}
        ratio = rates['income'] / rates['occupation']
        d_rates = max(abs(rates[t] - u['prc_s2_reroute_rates_oof'][t]) for t in ('income', 'occupation'))
        if d_rates > TOL or abs(ratio - u['ratio_f_I_over_f_O']) > TOL:
            mm.append(f'PRC_S2 re-routing rates / ratio differ (rates {rates}, ratio {ratio})')
        br = bracket(u['spend_over_twin'], u['target_total_nll_spend_prc_s2'])
        if br != u.get('bracket'):
            mm.append(f'bracket mine {br} recorded {u.get("bracket")}')
        saved_ok = (u.get('saved') == 'UNT_BR') == (br != 'CONTROL_DID_NOT_REACH_SPEND')
        book_rec = {}
        if br != 'CONTROL_DID_NOT_REACH_SPEND' and (V.RUNS / 'FA' / 'UNT_BR' / f'V{v}' / 'MODEL.pkl').is_file():
            Mu, oku = load_model('UNT_BR', v)
            b = Mu['book']
            f_o = float(br.split('@')[1])
            book_rec = {'unt_f_O': b.get('unt_f_O'), 'unt_f_I': b.get('unt_f_I'), 'unt_ratio': b.get('unt_ratio'), 'unt_tag': b.get('unt_tag'),
                        'expected_f_I': min(1.0, f_o * ratio)}
            if not (oku and b.get('frc_route') == 'unt' and b.get('unt_f_O') == f_o and abs(b.get('unt_f_I') - min(1.0, f_o * ratio)) <= TOL
                    and abs(b.get('unt_ratio') - ratio) <= TOL and b.get('unt_tag') == f'FRC_UNT_S{v}'
                    and np.array_equal(b['CI'], Mt['book']['CI']) and np.array_equal(b['CO'], Mt['book']['CO'])):
                mm.append(f'UNT_BR book differs from the bracket / ratio / twin codewords: {book_rec}')
        elif not saved_ok:
            mm.append('saved flag inconsistent with the bracket')
        look = u.get('nested_lookup_pair') or {}
        lw = max((abs(r['diff_twin_minus_entry'] - (look['TWIN']['auc'] - r['auc'])) for k, r in look.items() if k != 'TWIN'),
                 default=0.0)
        if lw > TOL:
            mm.append(f'nested LOOKUP entries inconsistent by {lw}')
        if abs(look.get('TWIN', {}).get('auc', math.nan) - s['nested_lookup_pair']['PRC_TWIN']['auc']) > TOL:
            mm.append('UNT twin LOOKUP AUC differs from P1')
        res = {'bracket_mine': br, 'bracket_recorded': u.get('bracket'), 'target': u['target_total_nll_spend_prc_s2'],
               'rates_mine': rates, 'ratio_mine': ratio, 'max_abs_twin_means_diff': d_tw, 'max_abs_rates_diff': d_rates,
               'unt_book': book_rec, 'lookup_consistency': lw, 'mismatches': mm}
        record['per'][f'V{v}'] = res
        checks.append({'id': f'V5D.UNT_SELECTION_V{v}__{tag}',
                       'what': f'R1 version {v}: registered f_O grid and tag, twin nested means and target vs P1, PRC_S2 re-routing '
                               f'rates and the f_I / f_O ratio from my own routing, bracket rule, saved UNT_BR book parameters, LOOKUP '
                               f'entry consistency (spends and AUCs as recorded; not recomputed)',
                       'inputs': V.inputs_sha([p, V.RESULTS / 'fa' / f'SELECTION_V{v}.json']), 'result': res,
                       'numerical_status': 'PASS' if not mm else 'FAIL', 'procedural_status': 'PASS' if not proc else 'FAIL',
                       'procedural_findings': proc})
    V.write_record(f'V5D_UNT__{tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['procedural_status'], c['result']['bracket_mine'], c['result']['rates_mine'],
              c['result']['mismatches'])
    return 0


def selftest():
    sp = {f'UNT@{f}': 0.001 * i for i, f in enumerate(GRID_F_O)}
    ok = {'first_reaching': bracket(sp, 0.0025) == 'UNT@0.08', 'exact_tie_reaches': bracket(sp, 0.004) == 'UNT@0.1',
          'none_reaches': bracket(sp, 1.0) == 'CONTROL_DID_NOT_REACH_SPEND', 'grid_order_not_dict_order':
          bracket({'UNT@0.3': 1.0, 'UNT@0.02': 1.0}, 0.5) == 'UNT@0.02'}
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())
