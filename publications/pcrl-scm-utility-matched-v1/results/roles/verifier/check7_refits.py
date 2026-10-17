"""Verifier check 7: predefined independent reader refits (own code; sklearn estimators with the source settings).

Predefined set (PROTOCOL.md section 8): the PAIR winners of SCM, ORD_NLLM, IBSI_NLLM and NEURAL_RAW on both seeds. Additionally
(not predefined, verification only) the income- and occupation-recipient winners of the same services.
Wires are rebuilt independently: SCM / ORD_NLLM / IBSI_NLLM from the locked release tables (bitwise-verified in check 2) and the
admitted routing (re-derived in check 1); NEURAL_RAW from the admitted frozen bank; each compared bitwise with the final
SERVICE_BANK. Readers: own finite scaler (population moments; zero scale -> 1), sklearn LogisticRegression(C=1, lbfgs, 500),
MLPClassifier((64, 32), relu, adam, alpha 1e-4, 300 iterations, no early stopping, random_state 2026101205 + s),
HistGradientBoostingClassifier(15 leaves, 200 iterations, learning rate 0.05, no early stopping, random_state as above), own
exact-vector lookup ((n_female + 2 pi)/(n + 2), unseen -> pi). Fit on ATTACK_FIT SEX, orientation by own ATTACK_VALIDATION AUC,
DEV_REPORT scores compared with the stored DEV_REPORT_SCORES.npz (label-free comparison). Winner choice re-applied from the stored
candidate AUCs. No DEV_REPORT label is read by this check.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

MODEL_SEED_BASE = 2026101205
SLATE = (('OUT_LR', 'out'), ('OUT_MLP', 'out'), ('OUT_HGB', 'out'), ('LOG_LR', 'log'), ('LOG_MLP', 'log'), ('DEC_LR', 'dec'), ('CONF_HGB', 'conf'))
KIND = dict(SLATE)
ROLES3 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def kinds(p, k):
    p = np.asarray(p, dtype=np.float64); d = np.argmax(p, axis=1); one = np.zeros((len(p), k)); one[np.arange(len(p)), d] = 1.0
    return {'out': np.column_stack((p, d.astype(np.float64))), 'log': np.column_stack((np.log(np.maximum(p, 1e-300)), one)),
            'dec': one, 'conf': p.max(axis=1, keepdims=True)}


class Scaler:
    def fit(self, x):
        self.mu = x.mean(axis=0); sd = x.std(axis=0); self.sd = np.where(sd > 0, sd, 1.0); return self

    def tr(self, x):
        return (x - self.mu) / self.sd


def fit(x, y, name, seed):
    sc = Scaler().fit(x)
    if 'MLP' in name:
        m = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=1e-4, max_iter=300, early_stopping=False, random_state=seed)
    elif 'HGB' in name:
        m = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False, random_state=seed)
    else:
        m = LogisticRegression(C=1, max_iter=500, solver='lbfgs')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(sc.tr(x), y)
    assert list(m.classes_) == [0, 1]
    return lambda z: m.predict_proba(sc.tr(z))[:, 1].astype(np.float64)


def lookup(keys_fit, y):
    tab = {}
    for kk, v in zip(keys_fit, y):
        a, b = tab.get(kk, (0, 0)); tab[kk] = (a + int(v), b + 1)
    pi = float(np.mean(y))
    return lambda keys: np.array([(tab[kk][0] + 2.0 * pi) / (tab[kk][1] + 2.0) if kk in tab else pi for kk in keys])


def keys_of(*vs):
    z = np.ascontiguousarray(np.column_stack([np.asarray(v, dtype=np.float64) for v in vs]))
    return [r.tobytes() for r in z]


def main():
    t0 = time.process_time()
    sel = V.jload(V.RES / 'SELECTION_LOCK.json'); sbm = V.jload(V.RES / 'SERVICE_BANK_MANIFEST.json'); rl = V.jload(V.RES / 'REPORTING_LOCK.json')
    fbm = V.jload(V.HYB_RES / 'FROZEN_BANK_MANIFEST.json')['bank_files_sha256']
    A = V.export('ATTACK_FIT', 'features'); As = V.export('ATTACK_FIT', 'sex')
    Vf = V.export('ATTACK_VALIDATION', 'features'); Vs = V.export('ATTACK_VALIDATION', 'sex'); R = V.export('DEV_REPORT', 'features')
    assert np.array_equal(As['row_id'], A['row_id']) and np.array_equal(Vs['row_id'], Vf['row_id'])
    yA = As['sex'].astype(np.int64); yV = Vs['sex'].astype(np.int64)
    rows = {'ATTACK_FIT': A, 'ATTACK_VALIDATION': Vf, 'DEV_REPORT': R}
    out = {'check': 7, 'title': 'predefined independent reader refits', 'per_seed': {}, 'hashes': {}}
    for s in V.SEEDS:
        rdir = V.PRIV / 'runs' / f'READERS_{s}'
        hs = {'READERS_SEED.json': V.sha_file(rdir / 'READERS_SEED.json') == rl['readers'][f'S{s}']['readers_seed_sha256'],
              'DEV_REPORT_SCORES.npz': V.sha_file(rdir / 'DEV_REPORT_SCORES.npz') == rl['readers'][f'S{s}']['dev_report_scores_sha256'],
              'SERVICE_BANK (manifest)': V.sha_file(V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz') == sbm['bank_files_sha256'][f'S{s}'],
              'SERVICE_BANK (reporting lock)': sbm['bank_files_sha256'][f'S{s}'] == rl['bank_files_sha256'][f'S{s}']}
        out['hashes'][f'S{s}'] = hs
        RS = V.jload(rdir / 'READERS_SEED.json'); SC = V.npz(rdir / 'DEV_REPORT_SCORES.npz'); SB = V.npz(V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz')
        assert np.array_equal(SC['row_id'], R['row_id'])
        RTAB = V.npz(V.PRIV / 'runs' / f'TRAJ_{s}' / 'RELEASE_TABLES.npz'); RT = V.npz(V.SRC / 'routing' / f'ROUTING_S{s}.npz')
        FB = V.npz(V.SRC / 'bank' / f'FROZEN_BANK_S{s}.npz'); assert V.sha_file(V.SRC / 'bank' / f'FROZEN_BANK_S{s}.npz') == fbm[f'S{s}']
        for r in ROLES3:
            assert np.array_equal(SB[f'row_id__{r}'], rows[r]['row_id']) and np.array_equal(RT[f'row_id__{r}'], rows[r]['row_id'])
        svc_of = sel['services'][str(s)]['service_of_role']
        res = {}
        for role in ('SCM', 'ORD_NLLM', 'IBSI_NLLM', 'NEURAL_RAW'):
            svc = svc_of[role]
            wire = {}
            for r in ROLES3:
                if role == 'NEURAL_RAW':
                    wire[r] = {t: FB[f'NEURAL_RAW__{r}__{t}'] for t in V.TASKS}
                else:
                    rid = sel['roles'][str(s)][role]['release']; init = RT[f'init__{r}']
                    wire[r] = {'income': RTAB[f'{rid}__v1'][RTAB[f'{rid}__group_of_initial'][init]], 'occupation': RTAB[f'{rid}__v2'][RTAB[f'{rid}__cell_of_initial'][init]]}
            bank_equal = all(np.array_equal(wire[r][t], SB[f'{svc}__{r}__{t}']) and SB[f'{svc}__{r}__{t}'].dtype == np.float64 for r in ROLES3 for t in V.TASKS)
            srec = RS['services'][svc]
            # winner logic re-applied from the stored candidate AUCs (pair: 8 pair + 16 ignore-one; recipients: 8)
            cands = list(srec['pair_candidates_attack_validation_auc'].items())
            wi = max(range(len(cands)), key=lambda i: (cands[i][1], -i))
            winner_logic_ok = cands[wi][0] == srec['winner']['pair']['candidate']
            items = {}
            for view in ('pair', 'income', 'occupation'):
                w = srec['winner'][view]; cand = w['candidate']; rid_ = w['reader_id']
                if view == 'pair' and cand.startswith('IGNORE_'):
                    other, name = cand[len('IGNORE_'):].split('__')
                    feat_view = 'income' if other == 'occupation' else 'occupation'
                else:
                    name = cand; feat_view = view
                if name == 'LOOKUP':
                    ks = {r: (keys_of(wire[r]['income'], wire[r]['occupation']) if feat_view == 'pair' else keys_of(wire[r][feat_view])) for r in ROLES3}
                    pred = lookup(ks['ATTACK_FIT'], yA); sv = pred(ks['ATTACK_VALIDATION']); sr = pred(ks['DEV_REPORT'])
                else:
                    def X(r):
                        if feat_view == 'pair':
                            a = kinds(wire[r]['income'], 2)[KIND[name]]; b = kinds(wire[r]['occupation'], 5)[KIND[name]]
                            return np.column_stack((a, b))
                        return kinds(wire[r][feat_view], V.KCLS[feat_view])[KIND[name]]
                    pred = fit(X('ATTACK_FIT'), yA, name, MODEL_SEED_BASE + s); sv = pred(X('ATTACK_VALIDATION')); sr = pred(X('DEV_REPORT'))
                a_raw = V.auc_mw(yV, sv); sign = 1 if a_raw >= 0.5 else -1
                if sign == -1:
                    sv, sr = 1 - sv, 1 - sr
                av = V.auc_mw(yV, sv)
                stored = SC[rid_]
                items[view] = {'candidate': cand, 'reader_id': rid_, 'fitted_on': feat_view, 'reader': name, 'predefined': view == 'pair',
                               'own_attack_validation_auc': av, 'stored_attack_validation_auc': w['attack_validation_auc'],
                               'abs_diff_attack_validation_auc': abs(av - w['attack_validation_auc']),
                               'own_orientation': sign, 'dev_scores_bitwise_equal': bool(np.array_equal(sr, stored)),
                               'dev_scores_max_abs_diff': float(np.max(np.abs(sr - stored))),
                               'dev_scores_rank_corr_ok': bool(np.corrcoef(sr, stored)[0, 1] > 0.999999) if np.std(sr) > 0 and np.std(stored) > 0 else None}
            res[role] = {'service': svc, 'wire_equal_to_service_bank': bank_equal, 'pair_winner_logic_ok': winner_logic_ok, 'refits': items}
        out['per_seed'][f'S{s}'] = res
    pre = [it for r in out['per_seed'].values() for x in r.values() for v, it in x['refits'].items() if it['predefined']]
    extra = [it for r in out['per_seed'].values() for x in r.values() for v, it in x['refits'].items() if not it['predefined']]
    out['summary'] = {'predefined_refits': len(pre), 'predefined_dev_bitwise': sum(i['dev_scores_bitwise_equal'] for i in pre),
                      'predefined_max_dev_abs_diff': max(i['dev_scores_max_abs_diff'] for i in pre),
                      'predefined_max_av_auc_abs_diff': max(i['abs_diff_attack_validation_auc'] for i in pre),
                      'extra_refits': len(extra), 'extra_dev_bitwise': sum(i['dev_scores_bitwise_equal'] for i in extra),
                      'extra_max_dev_abs_diff': max(i['dev_scores_max_abs_diff'] for i in extra),
                      'extra_max_av_auc_abs_diff': max(i['abs_diff_attack_validation_auc'] for i in extra),
                      'wires_equal_bank': all(x['wire_equal_to_service_bank'] for r in out['per_seed'].values() for x in r.values()),
                      'winner_logic_ok': all(x['pair_winner_logic_ok'] for r in out['per_seed'].values() for x in r.values()),
                      'hashes_ok': all(all(v.values()) for v in out['hashes'].values())}
    sm = out['summary']
    out['status'] = 'PASS' if (sm['predefined_max_dev_abs_diff'] <= 1e-9 and sm['predefined_max_av_auc_abs_diff'] <= 1e-9 and sm['wires_equal_bank']
                               and sm['winner_logic_ok'] and sm['hashes_ok']) else 'FAIL_OR_REVIEW'
    out['cpu_s'] = round(time.process_time() - t0, 2)
    V.write_json('check7_refits.json', out)
    print(json.dumps({'status': out['status'], 'summary': sm, 'cpu_s': out['cpu_s'],
                      'detail': {s: {role: {v: (it['candidate'], round(it['own_attack_validation_auc'], 6), round(it['stored_attack_validation_auc'], 6), it['dev_scores_bitwise_equal'],
                                               it['dev_scores_max_abs_diff']) for v, it in x['refits'].items()} for role, x in r.items()} for s, r in out['per_seed'].items()}},
                     indent=1, default=str))


if __name__ == '__main__':
    main()
