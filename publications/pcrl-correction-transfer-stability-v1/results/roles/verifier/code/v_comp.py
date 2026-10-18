"""Verifier replay of the ordinary compression control (cts.compression; matched GUARD utilization) on one partition.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
For every saved code COMP__<DIR>__<rule>: the released vectors are the nearest centroid (squared Euclidean, ties to the first) of the
deployed witness vector (own assignment, compared bitwise); its GUARD incremental costs and utilization (max over the five statistics of
cost / statistic reserve slack) and the candidate's target utilization at the matched strength are recomputed; the MATCHED /
TARGET_NOT_REACHED_LARGEST_K flag is checked against them; the realized CHECK and MODEL_SELECT costs (COMP_REALIZED.json) are recomputed.
Labels via cts.common.load(operation='verify'); repair panels only after the committed REPAIR_FREEZE. Nothing person-level is written.

    python3 .../v_comp.py --partition R0 --tag V1
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
from cts import common as C  # noqa: E402

TRACKF = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1' / 'runs' / 'TRACKF'
COMP_MAX_K = 2048
TOL = 1e-10


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def labels(P):
    out = {r: C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'verify',
                     f'independent verifier: compression-control replay on {P} {r}', partition=P, prole=r) for r in ('GUARD', 'CHECK')}
    out['MS'] = C.load('MODEL_SELECT', ('features', 'occupation', 'sex'), 'verify',
                       f'independent verifier: compression-control replay on MODEL_SELECT ({P})')
    return out


def nearest(P_, cents, chunk=256):
    out = np.empty(len(P_), dtype=np.int64)
    for i in range(0, len(P_), chunk):
        d = np.sum((P_[i:i + chunk, None, :] - cents[None, :, :]) ** 2, axis=2)
        out[i:i + chunk] = np.argmin(d, axis=1)
    return out


def utilization(stats, sslack):
    u = 0.0
    for k in V.STATS:
        b = sslack[k]
        u = max(u, stats[k] / b if b > 0 else (math.inf if stats[k] > b else 0.0))
    return u


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--partition', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    P = a.partition
    L = labels(P)
    W = npz(TRACKF / P / 'WITNESS' / 'WITNESS.npz')
    real = json.loads((TRACKF / P / 'COMP' / 'COMP_REALIZED.json').read_text()) if (TRACKF / P / 'COMP' / 'COMP_REALIZED.json').is_file() else None
    G, CK, S = L['GUARD'], L['CHECK'], L['MS']
    res, probs = {}, []
    for d in sorted((TRACKF / P / 'COMP').glob('COMP__*')):
        rec = json.loads((d / 'COMP.json').read_text())
        Z = npz(d / 'CODE.npz')
        sha_ok = V.file_sha(d / 'CODE.npz') == rec['npz_sha256']
        cents = Z['centroids']
        rel_mf = cents[nearest(W['dep__MODEL_FIT'], cents)]
        rel_ms = cents[nearest(W['dep__MODEL_SELECT'], cents)]
        rel_ok = bool(np.array_equal(rel_mf, Z['rel__MODEL_FIT']) and np.array_equal(rel_ms, Z['rel__MODEL_SELECT']))
        did = rec['matched_to']['direction']; s = float(rec['matched_to']['strength'])
        sel = json.loads((TRACKF / P / 'SELECT' / f'{did}.json').read_text())
        rho = {c: sel['baseline'][c]['slack'] for c in V.CONDITIONS}
        sslack = {k: min(rho[c] for c in V.CONDITIONS if V.STAT_FOR[c] == k) for k in V.STATS}
        Dd = npz(TRACKF / P / 'DIR' / did / 'DELTA.npz')
        r = G['rows']; pw = W['dep__MODEL_FIT'][r]
        cand = V.incremental_costs(V.release_full(pw, Dd['full__MODEL_FIT'][r], s), pw, G['occupation'], G['sex'], G['unit'])
        code = V.incremental_costs(rel_mf[r], pw, G['occupation'], G['sex'], G['unit'])
        tgt, ach = utilization(cand['stats'], sslack), utilization(code['stats'], sslack)
        flag_mine = 'MATCHED' if ach <= tgt else ('TARGET_NOT_REACHED_LARGEST_K' if rec['K'] == COMP_MAX_K else 'NOT_MATCHED_SMALLER_K')
        dcost = max(abs(code['stats'][k] - rec['guard_cost']['stats'][k]) for k in V.STATS)
        dtgt = abs(tgt - rec['matched_to']['guard_utilization']); dach = abs(ach - rec['guard_utilization'])
        row = {'K': rec['K'], 'flag_recorded': rec['flag'], 'flag_mine': flag_mine, 'target_utilization': tgt, 'achieved_utilization': ach,
               'diff_guard_cost': dcost, 'diff_target': dtgt, 'diff_achieved': dach, 'code_sha_ok': sha_ok, 'release_bitwise_equal': rel_ok}
        if real and d.name in real['codes']:
            rr = real['codes'][d.name]
            ck = V.incremental_costs(rel_mf[CK['rows']], W['dep__MODEL_FIT'][CK['rows']], CK['occupation'], CK['sex'], CK['unit'])
            ms = V.incremental_costs(rel_ms, W['dep__MODEL_SELECT'], S['occupation'], S['sex'], S['unit'])
            row['diff_realized_CHECK'] = max(abs(ck['stats'][k] - rr['CHECK']['stats'][k]) for k in V.STATS)
            row['diff_realized_MS'] = max(abs(ms['stats'][k] - rr['MS']['stats'][k]) for k in V.STATS)
            row['realized_MS_men_brier'] = ms['stats']['men_brier']
            row['realized_MS_max_slack_use'] = max(ms['stats'][V.STAT_FOR[c]] / rho[c] if rho[c] > 0 else
                                                   (math.inf if ms['stats'][V.STAT_FOR[c]] > rho[c] else 0.0) for c in V.CONDITIONS)
        else:
            row['realized'] = 'absent'
        ok = (sha_ok and rel_ok and dcost <= TOL and dtgt <= 1e-9 and dach <= 1e-9 and flag_mine == rec['flag'] and
              row.get('diff_realized_CHECK', 0.0) <= TOL and row.get('diff_realized_MS', 0.0) <= TOL)
        if not ok:
            probs.append(d.name)
        res[d.name] = row
    flags = sorted({v['flag_recorded'] for v in res.values()})
    chk = {'id': f'C3.COMP.{P}__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'{P} compression controls: code releases (nearest centroid of the deployed witness vector), GUARD cost and utilization '
                   'of each code, the candidate target utilization at the matched strength, the MATCHED / TARGET_NOT_REACHED flag, and the '
                   'realized CHECK / MODEL_SELECT costs',
           'method': 'own squared-Euclidean assignment (ties to the first centroid), own metrics and utilization (max over statistics of '
                     'cost / statistic reserve slack, the SELECT baseline); tolerances 1e-10 (costs), 1e-9 (utilizations)',
           'status': 'PASS' if not probs else 'FAIL',
           'numbers': {'codes': res, 'problems': probs, 'flags_present': flags,
                       'note': ('codes flagged TARGET_NOT_REACHED_LARGEST_K are not utilization-matched controls; a matched comparison '
                                'against them is not available at those strengths') if 'TARGET_NOT_REACHED_LARGEST_K' in flags else ''}}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V_COMP__{P}__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'codes': {k: [v['K'], v['flag_recorded'], round(v['target_utilization'], 3),
                                                              round(v['achieved_utilization'], 3)] for k, v in res.items()}}, indent=1))


if __name__ == '__main__':
    main()
