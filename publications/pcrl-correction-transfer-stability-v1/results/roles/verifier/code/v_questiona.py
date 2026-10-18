"""Verifier replay of question A (RULES.md section 6): does the GUARD estimate predict the realized CHECK increment better than
the out-of-fold (OOF) estimate? Point errors, MAE_rel, D = MAE_rel(OOF) - MAE_rel(GUARD) and a paired household bootstrap,
recomputed with the verifier's own code from the saved arrays and verify-operation labels, compared with QUESTION_A/*.json.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
Bootstrap reading used here (from the RULES.md text): in every replicate ONE household resample is drawn for each set (FIT people
of the OOF estimator, GUARD, CHECK) and all statistics and strengths of that set are recomputed on it; the sets are resampled
independently. The recorded implementation is replayed as well for comparison (its own documented draw order is not imitated;
only the point estimate, the label and the bootstrap SE are compared). Nothing person-level is printed or written.

    python3 .../v_questiona.py --partition D0 --direction REFIT_PAIR --deployment full --oof OOF --tag V1
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

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402

CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
TRACKF = CAMP / 'runs' / 'TRACKF'
RES = ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'
S10 = V.LADDER[1:]
B = 2000


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def labels(P, kind):
    out = {}
    for r in ('FIT', 'GUARD', 'CHECK'):
        out[r] = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'verify',
                        f'independent verifier: question A replay on {P} {r} (per-person increments; own bootstrap)',
                        partition=P, prole=r)
    return out


def incr_matrix(ps, p0, y):
    db, dn, da = [], [], []
    for p in ps:
        b, n, a = V.increments(p, p0, y)
        db.append(b); dn.append(n); da.append(a)
    return np.column_stack(db), np.column_stack(dn), np.column_stack(da)


def stat_arrays(st):
    sex = np.asarray(st['sex'])
    return {'men_brier': (st['db'], sex == 0), 'women_brier': (st['db'], sex == 1), 'brier': (st['db'], None),
            'nll': (st['dn'], None), 'acc_loss': (st['da'], None)}


def one_resample_boot(sets, ks, ss, rng):
    """Replicates of D with one household resample per set per replicate (shared by all statistics and strengths)."""
    reps = {}
    for name, st in sets.items():
        _, inv = np.unique(np.asarray(st['unit']), return_inverse=True)
        inv = inv.reshape(-1); G = int(inv.max()) + 1
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)       # (B, G)
        out = {}
        for k, (arr, mask) in stat_arrays(st).items():
            if k not in ks:
                continue
            m = np.ones(len(arr), dtype=bool) if mask is None else mask
            Ssum = np.zeros((G, arr.shape[1])); np.add.at(Ssum, inv[m], arr[m])
            cnt = np.bincount(inv[m], minlength=G).astype(np.float64)
            out[k] = (W @ Ssum) / (W @ cnt)[:, None]                                   # (B, 10)
        reps[name] = out
    e_g = np.mean([np.abs(reps['CHECK'][k] - reps['GUARD'][k]) / ss[k] for k in ks], axis=0).mean(1)
    e_o = np.mean([np.abs(reps['CHECK'][k] - reps['OOF'][k]) / ss[k] for k in ks], axis=0).mean(1)
    return e_o - e_g


def replay(P, L, W, O, did, dep, oof, tag):
    rec = json.loads((TRACKF / P / 'QUESTION_A' / f'{did}__{dep}__{oof}.json').read_text())['result']
    sel = json.loads((TRACKF / P / 'SELECT' / f'{did}.json').read_text())
    rho = {c: sel['baseline'][c]['slack'] for c in V.CONDITIONS}
    ss = {k: min(rho[c] for c in V.CONDITIONS if V.STAT_FOR[c] == k) for k in V.STATS}
    ks = [k for k in V.STATS if ss[k] > 0]
    T, D = npz(TRACKF / P / 'DIR' / did / 'TRAIN.npz'), npz(TRACKF / P / 'DIR' / did / 'DELTA.npz')
    sets = {}
    F = L['FIT']
    if oof == 'OOF':
        z0 = V.centred_log(O['oof']); p0 = V.softmax_rows(z0)
        ps = [V.softmax_rows(z0 + s * T['delta_oof']) for s in S10]
    else:
        nz = npz(TRACKF / P / 'NESTED' / f'NESTED_{did.split("_", 1)[1]}.npz')
        p0 = nz['witness_oof']; ps = [V.release_full(nz['witness_oof'], nz['delta_oof'], s) for s in S10]
    db, dn, da = incr_matrix(ps, p0, F['occupation'])
    sets['OOF'] = {'db': db, 'dn': dn, 'da': da, 'unit': F['unit'], 'sex': F['sex']}
    for name in ('GUARD', 'CHECK'):
        X = L[name]; r = X['rows']
        if dep == 'full':
            pw, dl = W['dep__MODEL_FIT'][r], D['full__MODEL_FIT'][r]
            p0 = pw; ps = [V.release_full(pw, dl, s) for s in S10]
        else:
            pf, df = W['fold__MODEL_FIT'][:, r], D['folds__MODEL_FIT'][:, r]
            p0 = V.release_ensemble(pf, df, 0.0); ps = [V.release_ensemble(pf, df, s) for s in S10]
        db, dn, da = incr_matrix(ps, p0, X['occupation'])
        sets[name] = {'db': db, 'dn': dn, 'da': da, 'unit': X['unit'], 'sex': X['sex']}
    point = {}
    for name, st in sets.items():
        for k, (arr, mask) in stat_arrays(st).items():
            m = np.ones(len(arr), dtype=bool) if mask is None else mask
            point[name, k] = arr[m].mean(0)
    mae_g = float(np.mean([np.abs(point['CHECK', k] - point['GUARD', k]) / ss[k] for k in ks], axis=0).mean())
    mae_o = float(np.mean([np.abs(point['CHECK', k] - point['OOF', k]) / ss[k] for k in ks], axis=0).mean())
    Dp = mae_o - mae_g
    rng = np.random.default_rng(int(hashlib.sha256(b'CTS1_verifier_questionA').hexdigest()[:15], 16))
    Db = one_resample_boot(sets, ks, ss, rng)
    q05, q95, se = float(np.quantile(Db, 0.05)), float(np.quantile(Db, 0.95)), float(Db.std(ddof=1))
    label = 'PREDICTION_IMPROVED' if (Dp > 0 and q05 > 0) else 'NOT_IMPROVED'
    wins = int(sum((np.abs(point['CHECK', k] - point['GUARD', k]) < np.abs(point['CHECK', k] - point['OOF', k])).sum() for k in ks))
    num = {'recorded': {k: rec[k] for k in ('MAE_rel_GUARD', 'MAE_rel_OOF', 'D', 'boot_se', 'q05', 'q95', 'label', 'guard_closer_cells',
                                            'cells', 'stats_used')},
           'mine': {'MAE_rel_GUARD': mae_g, 'MAE_rel_OOF': mae_o, 'D': Dp, 'boot_se_one_resample_per_set': se, 'q05': q05, 'q95': q95,
                    'label': label, 'guard_closer_cells': wins, 'cells': 10 * len(ks), 'stats_used': ks},
           'point_abs_diff': {'MAE_rel_GUARD': abs(mae_g - rec['MAE_rel_GUARD']), 'MAE_rel_OOF': abs(mae_o - rec['MAE_rel_OOF']),
                              'D': abs(Dp - rec['D'])},
           'men_brier_signed_error_by_strength': {
               'GUARD': {str(s): float(point['CHECK', 'men_brier'][j] - point['GUARD', 'men_brier'][j]) for j, s in enumerate(S10)},
               'OOF': {str(s): float(point['CHECK', 'men_brier'][j] - point['OOF', 'men_brier'][j]) for j, s in enumerate(S10)}},
           'stat_slack': ss}
    point_ok = all(v <= 1e-9 for v in num['point_abs_diff'].values()) and wins == rec['guard_closer_cells'] and ks == rec['stats_used']
    label_ok = label == rec['label']
    status = 'FAIL' if not point_ok else ('PASS' if label_ok else 'ATTENTION')
    chk = {'id': f'C3.QA.{P}.{did}.{dep}.{oof}__{tag}', 'kind': 'NUMERICAL_REPLAY',
           'what': f'question A on {P} {did} ({dep}, {oof}): transfer errors e_X = Delta_CHECK - Delta_X, MAE_rel (stat slack > 0), '
                   'D = MAE_rel(OOF) - MAE_rel(GUARD), bootstrap quantiles and the PREDICTION_IMPROVED label',
           'method': 'own per-person increments at the 10 nonzero strengths; point estimates compared at 1e-9; own household bootstrap (B = '
                     '2000, one resample per set per replicate shared by all statistics, own seed) for the label; the recorded bootstrap '
                     'draws a separate household resample for each statistic of a set (cts.rules.prediction_comparison), so quantiles '
                     'are compared only through the label and the SE',
           'status': status, 'numbers': num}
    chk['code_sha256'] = V.code_hashes(__file__, V.__file__)
    V.write_record(f'V_QUESTIONA__{P}__{did}__{dep}__{oof}__{tag}.json', chk)
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition', required=True); ap.add_argument('--direction', default=None)
    ap.add_argument('--deployment', default='full', choices=('full', 'ens')); ap.add_argument('--oof', default='OOF', choices=('OOF', 'NESTED'))
    ap.add_argument('--all', action='store_true', help='every QUESTION_A record of the partition')
    ap.add_argument('--skip-done-tag', default=None, help='skip records already replayed under this tag')
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    P = a.partition
    if a.all:
        jobs = []
        for f in sorted((TRACKF / P / 'QUESTION_A').glob('*.json')):
            did, dep, oof = f.stem.split('__')
            if a.skip_done_tag and (V.VER_DIR / 'records' / f'V_QUESTIONA__{P}__{did}__{dep}__{oof}__{a.skip_done_tag}.json').is_file():
                continue
            jobs.append((did, dep, oof))
    else:
        jobs = [(a.direction, a.deployment, a.oof)]
    L = labels(P, C.PARTITIONS[P]['kind'])
    W, O = npz(TRACKF / P / 'WITNESS' / 'WITNESS.npz'), npz(TRACKF / P / 'WITNESS' / 'OOF.npz')
    out = [replay(P, L, W, O, did, dep, oof, a.tag) for did, dep, oof in jobs]
    print(json.dumps(V.append_checks(out)))
    print(json.dumps({c['id']: [c['status'], c['numbers']['mine']['label'], c['numbers']['recorded']['label'],
                                round(c['numbers']['mine']['D'], 4), round(c['numbers']['mine']['q05'], 4),
                                round(c['numbers']['recorded']['q05'], 4)] for c in out}, indent=1))


if __name__ == '__main__':
    main()
