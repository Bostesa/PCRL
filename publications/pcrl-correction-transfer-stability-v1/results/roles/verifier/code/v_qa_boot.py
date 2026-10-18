"""Verifier diagnostic of the question A bootstrap scheme (RULES.md section 6): the recorded implementation
(cts.rules.prediction_comparison) draws a separate household resample for EACH statistic of a set, so the five statistics that enter
D (men / women / all summed-Brier increase, NLL increase, accuracy loss; computed on the same households) are resampled
independently; the verifier's reading draws ONE household resample per set per replicate and recomputes D on it. This script
recomputes both schemes with the verifier's own code over several seeds for selected question A records, to show whether the
difference in the bootstrap SE / 5% quantile / label is systematic (scheme) or Monte Carlo noise.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records). Labels via
cts.common.load(operation='verify'); nothing person-level is printed or written.

    python3 .../v_qa_boot.py --partition D0 --records REFIT_CONT__ens__OOF,REFIT_PAIR__full__OOF --seeds 5 --tag V1
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
from v_questiona import incr_matrix, stat_arrays, npz, S10, B  # noqa: E402  (verifier's own helpers)

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402

TRACKF = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1' / 'runs' / 'TRACKF'


def labels(P):
    return {r: C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'verify',
                      f'independent verifier: question A bootstrap-scheme diagnostic on {P} {r}', partition=P, prole=r)
            for r in ('FIT', 'GUARD', 'CHECK')}


def build_sets(P, L, did, dep, oof):
    W, O = npz(TRACKF / P / 'WITNESS' / 'WITNESS.npz'), npz(TRACKF / P / 'WITNESS' / 'OOF.npz')
    T, D = npz(TRACKF / P / 'DIR' / did / 'TRAIN.npz'), npz(TRACKF / P / 'DIR' / did / 'DELTA.npz')
    F = L['FIT']; sets = {}
    if oof == 'OOF':
        z0 = V.centred_log(O['oof']); p0 = V.softmax_rows(z0); ps = [V.softmax_rows(z0 + s * T['delta_oof']) for s in S10]
    else:
        nz = npz(TRACKF / P / 'NESTED' / f'NESTED_{did.split("_", 1)[1]}.npz')
        p0 = nz['witness_oof']; ps = [V.release_full(nz['witness_oof'], nz['delta_oof'], s) for s in S10]
    db, dn, da = incr_matrix(ps, p0, F['occupation'])
    sets['OOF'] = {'db': db, 'dn': dn, 'da': da, 'unit': F['unit'], 'sex': F['sex']}
    for name in ('GUARD', 'CHECK'):
        X = L[name]; r = X['rows']
        if dep == 'full':
            pw, dl = W['dep__MODEL_FIT'][r], D['full__MODEL_FIT'][r]; p0 = pw; ps = [V.release_full(pw, dl, s) for s in S10]
        else:
            pf, df = W['fold__MODEL_FIT'][:, r], D['folds__MODEL_FIT'][:, r]
            p0 = V.release_ensemble(pf, df, 0.0); ps = [V.release_ensemble(pf, df, s) for s in S10]
        db, dn, da = incr_matrix(ps, p0, X['occupation'])
        sets[name] = {'db': db, 'dn': dn, 'da': da, 'unit': X['unit'], 'sex': X['sex']}
    return sets


def boot_D(sets, ks, ss, rng, per_statistic):
    reps = {}
    for name, st in sets.items():
        _, inv = np.unique(np.asarray(st['unit']), return_inverse=True)
        inv = inv.reshape(-1); G = int(inv.max()) + 1
        Wshared = None if per_statistic else rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)
        reps[name] = {}
        for k, (arr, mask) in stat_arrays(st).items():
            if k not in ks:
                continue
            Wk = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64) if per_statistic else Wshared
            m = np.ones(len(arr), dtype=bool) if mask is None else mask
            Ssum = np.zeros((G, arr.shape[1])); np.add.at(Ssum, inv[m], arr[m])
            cnt = np.bincount(inv[m], minlength=G).astype(np.float64)
            reps[name][k] = (Wk @ Ssum) / (Wk @ cnt)[:, None]
    e_g = np.mean([np.abs(reps['CHECK'][k] - reps['GUARD'][k]) / ss[k] for k in ks], axis=0).mean(1)
    e_o = np.mean([np.abs(reps['CHECK'][k] - reps['OOF'][k]) / ss[k] for k in ks], axis=0).mean(1)
    return e_o - e_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition', required=True); ap.add_argument('--records', required=True)
    ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    P = a.partition
    L = labels(P)
    res = {}
    for recname in [r for r in a.records.split(',') if r]:
        did, dep, oof = recname.split('__')
        rec = json.loads((TRACKF / P / 'QUESTION_A' / f'{recname}.json').read_text())['result']
        sel = json.loads((TRACKF / P / 'SELECT' / f'{did}.json').read_text())
        rho = {c: sel['baseline'][c]['slack'] for c in V.CONDITIONS}
        ss = {k: min(rho[c] for c in V.CONDITIONS if V.STAT_FOR[c] == k) for k in V.STATS}
        ks = [k for k in V.STATS if ss[k] > 0]
        sets = build_sets(P, L, did, dep, oof)
        out = {}
        for scheme, per in (('one_resample_per_set', False), ('separate_resample_per_statistic', True)):
            q05s, ses, labs = [], [], []
            for j in range(a.seeds):
                rng = np.random.default_rng(int(hashlib.sha256(f'CTS1_verifier_qa_boot|{scheme}|{j}'.encode()).hexdigest()[:15], 16))
                Db = boot_D(sets, ks, ss, rng, per)
                q05s.append(float(np.quantile(Db, 0.05))); ses.append(float(Db.std(ddof=1)))
                labs.append('PREDICTION_IMPROVED' if (rec['D'] > 0 and q05s[-1] > 0) else 'NOT_IMPROVED')
            out[scheme] = {'q05': q05s, 'boot_se': ses, 'labels': labs, 'se_mean': float(np.mean(ses)), 'q05_mean': float(np.mean(q05s))}
        out['recorded'] = {'D': rec['D'], 'q05': rec['q05'], 'boot_se': rec['boot_se'], 'label': rec['label']}
        out['se_ratio_recorded_over_one_resample'] = rec['boot_se'] / out['one_resample_per_set']['se_mean']
        res[recname] = out
    flips = {r: v for r, v in res.items() if set(v['one_resample_per_set']['labels']) != {v['recorded']['label']}}
    chk = {'id': f'C3.QA.{P}.BOOTSTRAP_SCHEME__{a.tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'question A on {P}: does the recorded bootstrap (separate household resample per statistic within a set) understate '
                   'the SE of D relative to one resample per set per replicate (D recomputed on each replicate), and does the label depend on it',
           'method': f'both schemes implemented by the verifier, B = 2000, {a.seeds} seeds each, on the verifier\'s own per-person increments',
           'status': 'ATTENTION' if (flips or any(v['se_ratio_recorded_over_one_resample'] < 0.95 for v in res.values())) else 'PASS',
           'numbers': {'records': res, 'labels_differing_under_one_resample_per_set': sorted(flips),
                       'reading': 'RULES.md section 6 says households are resampled independently within FIT, GUARD and CHECK and D is '
                                  'recomputed on every replicate; the verifier reads this as one resample per set per replicate. Under '
                                  'positive correlation between the statistics (they share households and people), separate per-statistic '
                                  'resampling shrinks the bootstrap variance of D (anti-conservative 5% quantile). For adjudication by the '
                                  'statistical reviewer / coordinator; not a widened criterion.'}}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__, HERE / 'v_questiona.py')
    V.write_record(f'V_QA_BOOT__{P}__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({r: {'rec_se': v['recorded']['boot_se'], 'one_se': v['one_resample_per_set']['se_mean'],
                          'sep_se': v['separate_resample_per_statistic']['se_mean'], 'rec_q05': v['recorded']['q05'],
                          'one_q05': v['one_resample_per_set']['q05'], 'sep_q05': v['separate_resample_per_statistic']['q05'],
                          'one_labels': v['one_resample_per_set']['labels'], 'rec_label': v['recorded']['label']} for r, v in res.items()},
                     indent=1))


if __name__ == '__main__':
    main()
