"""Descriptive variance detail of question-A units (adjudication A5, condition 4; no criterion, no label change): D, the bootstrap SE and
q05 under the registered scheme (one household resample per set and replicate) and under the superseded per-statistic scheme (both
with the registered tag CTS1_transfer_boot, B = 2000, and the draw orders of the respective implementations), the household influence
on D (linearization of D in the set means), the share of the linearized variance carried by the top 10 households, and D / q05 after
dropping the 1, 5 and 10 households with the largest absolute influence. Household identifiers are not written (ranks, set, size and
influence only).

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records). Labels via
cts.common.load(operation='verify'); the R1 panel C1 is read after the committed REPAIR_FREEZE.

    python3 .../v_qa_influence.py --partition R1 --records REFIT_LOC__full__OOF,REFIT_PAIR_TR__full__OOF --tag V1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402
from v_questiona import stat_arrays, B  # noqa: E402
from v_qa_boot import build_sets  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402

TRACKF = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1' / 'runs' / 'TRACKF'
TAG = b'CTS1_transfer_boot'
SETS = ('OOF', 'GUARD', 'CHECK')


def labels(P):
    return {r: C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'verify',
                      f'independent verifier: descriptive variance / household influence of question A on {P} {r} (adjudication A5)',
                      partition=P, prole=r) for r in ('FIT', 'GUARD', 'CHECK')}


def means(sets):
    out = {}
    for name, st in sets.items():
        for k, (arr, mask) in stat_arrays(st).items():
            m = np.ones(len(arr), dtype=bool) if mask is None else mask
            out[name, k] = arr[m].mean(0)
    return out


def D_of(mn, ks, ss):
    g = np.mean([np.abs(mn['CHECK', k] - mn['GUARD', k]) / ss[k] for k in ks], axis=0).mean()
    o = np.mean([np.abs(mn['CHECK', k] - mn['OOF', k]) / ss[k] for k in ks], axis=0).mean()
    return float(o - g)


def boot(sets, ks, ss, per_statistic, return_mean=False):
    """Registered: one weight matrix per set (order OOF, GUARD, CHECK). Superseded: a fresh matrix for every (set, statistic), all five
    statistics in the order men_brier, women_brier, brier, nll, acc_loss (as cts.rules.prediction_comparison drew them)."""
    rng = np.random.default_rng(int(hashlib.sha256(TAG).hexdigest()[:15], 16))
    reps = {}
    for name in SETS:
        st = sets[name]
        _, inv = np.unique(np.asarray(st['unit']), return_inverse=True)
        inv = inv.reshape(-1); G = int(inv.max()) + 1
        W = None if per_statistic else rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)
        for k, (arr, mask) in stat_arrays(st).items():
            Wk = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64) if per_statistic else W
            m = np.ones(len(arr), dtype=bool) if mask is None else mask
            S = np.zeros((G, arr.shape[1])); np.add.at(S, inv[m], arr[m])
            cnt = np.bincount(inv[m], minlength=G).astype(np.float64)
            reps[name, k] = (Wk @ S) / (Wk @ cnt)[:, None]
    g = np.mean([np.abs(reps['CHECK', k] - reps['GUARD', k]) / ss[k] for k in ks], axis=0).mean(1)
    o = np.mean([np.abs(reps['CHECK', k] - reps['OOF', k]) / ss[k] for k in ks], axis=0).mean(1)
    Db = o - g
    if return_mean:
        return float(Db.std(ddof=1)), float(np.quantile(Db, 0.05)), float(Db.mean())
    return float(Db.std(ddof=1)), float(np.quantile(Db, 0.05))


def influence(sets, ks, ss):
    """Linearized household influence on D: IF_h = sum_{k,s} dD/dmean_{X,k,s} * (household h's share of the centred sum)."""
    mn = means(sets)
    n_cells = 10 * len(ks)
    grad = {}
    for k in ks:
        sgo = np.sign(mn['CHECK', k] - mn['OOF', k]); sgg = np.sign(mn['CHECK', k] - mn['GUARD', k])
        grad['OOF', k] = -sgo / (ss[k] * n_cells)
        grad['GUARD', k] = sgg / (ss[k] * n_cells)
        grad['CHECK', k] = (sgo - sgg) / (ss[k] * n_cells)
    rows = []
    var_lin = 0.0
    for name in SETS:
        st = sets[name]
        groups, inv = np.unique(np.asarray(st['unit']), return_inverse=True)
        inv = inv.reshape(-1); G = len(groups)
        IF = np.zeros(G); per_stat = {k: np.zeros(G) for k in ks}
        for k, (arr, mask) in stat_arrays(st).items():
            if k not in ks:
                continue
            m = np.ones(len(arr), dtype=bool) if mask is None else mask
            centred = np.where(m[:, None], arr - arr[m].mean(0), 0.0) / m.sum()        # person share of the mean's deviation
            contrib = centred @ grad[name, k]                                        # (n,)
            hk = np.bincount(inv, weights=contrib, minlength=G)
            IF += hk; per_stat[k] += hk
        size = np.bincount(inv, minlength=G)
        var_lin += G / (G - 1) * float(np.sum(IF ** 2))
        for h in range(G):
            rows.append({'set': name, 'h_index': h, 'IF': float(IF[h]), 'size': int(size[h]),
                         'dominant_statistic': max(ks, key=lambda kk: abs(per_stat[kk][h]))})
    return rows, var_lin


def drop(sets, rows_drop):
    out = {}
    for name in SETS:
        st = sets[name]
        groups, inv = np.unique(np.asarray(st['unit']), return_inverse=True)
        inv = inv.reshape(-1)
        bad = {r['h_index'] for r in rows_drop if r['set'] == name}
        keep = ~np.isin(inv, list(bad))
        out[name] = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in st.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition', required=True); ap.add_argument('--records', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    P = a.partition
    L = labels(P)
    res = {}
    for recname in [r for r in a.records.split(',') if r]:
        did, dep, oof = recname.split('__')
        sel = json.loads((TRACKF / P / 'SELECT' / f'{did}.json').read_text())
        rho = {c: sel['baseline'][c]['slack'] for c in V.CONDITIONS}
        ss = {k: min(rho[c] for c in V.CONDITIONS if V.STAT_FOR[c] == k) for k in V.STATS}
        ks = [k for k in V.STATS if ss[k] > 0]
        sets = build_sets(P, L, did, dep, oof)
        D = D_of(means(sets), ks, ss)
        se_r, q_r, mu_r = boot(sets, ks, ss, per_statistic=False, return_mean=True)
        se_s, q_s, mu_s = boot(sets, ks, ss, per_statistic=True, return_mean=True)
        tails = {name: {'max_abs_nll_increment': float(np.max(np.abs(sets[name]['dn']))),
                        'persons_with_abs_nll_increment_gt_1_nat': int(np.sum(np.max(np.abs(sets[name]['dn']), axis=1) > 1.0)),
                        'persons': int(len(sets[name]['dn']))} for name in SETS}
        rec_reg = json.loads((TRACKF / P / 'QUESTION_A' / f'{recname}__REG.json').read_text())['result']
        rec_old = json.loads((TRACKF / P / 'QUESTION_A' / f'{recname}.json').read_text())['result']
        rows, var_lin = influence(sets, ks, ss)
        order = sorted(rows, key=lambda r: -abs(r['IF']))
        tot = sum(r['IF'] ** 2 for r in rows)
        top10_share = sum(r['IF'] ** 2 for r in order[:10]) / tot
        drops = {}
        for n in (1, 5, 10):
            red = drop(sets, order[:n])
            Dn = D_of(means(red), ks, ss)
            se_n, q_n = boot(red, ks, ss, per_statistic=False)
            drops[str(n)] = {'D': Dn, 'q05_registered': q_n, 'se_registered': se_n,
                             'label_registered': 'PREDICTION_IMPROVED' if (Dn > 0 and q_n > 0) else 'NOT_IMPROVED'}
        res[recname] = {'D': D, 'stats_used': ks,
                        'registered': {'boot_se': se_r, 'q05': q_r, 'boot_mean': mu_r, 'recorded_q05': rec_reg['q05'],
                                       'recorded_se': rec_reg['boot_se'], 'normal_approx_q05_D_minus_z_se': D - V.Z_REGISTERED * se_r},
                        'superseded_per_statistic': {'boot_se': se_s, 'q05': q_s, 'boot_mean': mu_s, 'recorded_q05': rec_old['q05'],
                                                     'recorded_se': rec_old['boot_se'],
                                                     'normal_approx_q05_D_minus_z_se': D - V.Z_REGISTERED * se_s},
                        'q05_shift_vs_z_times_se_change': {'q05_shift': q_s - q_r, 'z_x_se_change': V.Z_REGISTERED * (se_r - se_s)},
                        'nll_tails': tails,
                        'linearized_se': float(np.sqrt(var_lin)),
                        'top10_household_share_of_linearized_variance': top10_share,
                        'top10_households': [{'rank': i + 1, 'set': r['set'], 'size': r['size'], 'IF': r['IF'],
                                              'dominant_statistic': r['dominant_statistic']} for i, r in enumerate(order[:10])],
                        'after_dropping_top_households': drops,
                        'households_per_set': {s: int(sum(1 for r in rows if r['set'] == s)) for s in SETS}}
    chk = {'id': f'C3.QA.{P}.VARIANCE_DETAIL__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'{P}: descriptive variance detail of the scheme-dependent question-A units (adjudication A5 condition 4; no criterion)',
           'method': 'own per-person increments; both bootstrap schemes with the registered tag and B = 2000 (draw orders of the registered '
                     'and the superseded implementations); linearized household influence of D; exact D and registered-scheme q05 after '
                     'dropping the households with the largest absolute influence',
           'status': 'ATTENTION', 'numbers': res}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__, HERE / 'v_questiona.py', HERE / 'v_qa_boot.py')
    V.write_record(f'V_QA_INFLUENCE__{P}__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main()
