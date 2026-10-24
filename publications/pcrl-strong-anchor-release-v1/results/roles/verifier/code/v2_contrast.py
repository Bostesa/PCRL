"""Verifier wave 2 unit (label read: MODEL_FIT occupation and SEX, operation 'verify'): recompute FA_TFRB_R2's out-of-fold guard
contrast at the selected rho* (0.6 / 1.0) from the saved bag (its five fold residual networks and feature maps in MODEL.pkl), without
importing sar.fa. Each MODEL_FIT person is released by the one fold model that did not see them (folds SAR_xf_FA_S<v>, the verifier's
own implementation of the SHA-256 household dealing); d = summed occupation Brier(bag fold release) - Brier(SUCC); C = men's mean d -
women's mean d; per-fold contrasts and their between-fold SE (ddof 1, / sqrt 5).

Inputs (loaders / label-free building blocks only): ihs.fam_h.predict (SUCC's deployment function, as the fit used), the nrm task cache
TASK.pkl (hash-checked against MODEL.pkl's task_cache_sha256: the teachers' stored MODEL_FIT out-of-fold outputs), ohn.fam_a.compact_po
(the compact fields), acs_replication.trees.probabilities (the release law). The feature map, network forward pass, logits and
metrics are the verifier's own code from the stated definitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

sys.path.insert(0, str(V.ROOT))
import numpy as np  # noqa: E402
from sar import common as C  # noqa: E402

CLIP = 1e-300
RHO_STAR = {0: 0.6, 1: 1.0}


def logodds(p):
    p = np.asarray(p, np.float64)
    return np.log(np.clip(p[:, 1], CLIP, None)) - np.log(np.clip(p[:, 0], CLIP, None))


def clogits(p):
    L = np.log(np.clip(np.asarray(p, np.float64), CLIP, None))
    return L - L.mean(1, keepdims=True)


def phi(P, A):
    Zc = A[:, 12:18]
    num = (Zc[:, [0, 1, 3, 5]] - P['num_mu']) / P['num_sd']
    oh = [(Zc[:, 2][:, None] == P['levels_COW'][None, :]).astype(np.float64),
          (Zc[:, 4][:, None] == P['levels_MAR'][None, :]).astype(np.float64)]
    return np.column_stack([num] + oh + [(A[:, 0:12] - P['lg_mu']) / P['lg_sd']])


def forward(W, F):
    h = np.maximum(F @ W[0].T + W[1], 0.0)
    h = np.maximum(h @ W[2].T + W[3], 0.0)
    return h @ W[4].T + W[5]


def brier(y, p):
    n = len(y)
    oh = np.zeros_like(p)
    oh[np.arange(n), y] = 1.0
    return ((p - oh) ** 2).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    S = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'verify',
               'independent verifier wave 2: FA_TFRB_R2 out-of-fold guard contrast at rho* recomputed from the saved bag (no sar.fa import)')
    X, names, unit = np.asarray(S['X'], np.float64), tuple(map(str, S['feature_names'])), np.asarray(S['unit'])
    yO, sex = np.asarray(S['occupation'], np.int64), np.asarray(S['sex'])
    del S
    from ihs import fam_h
    from ohn.fam_a import compact_po
    from acs_replication.trees import probabilities
    out = {'schema': 'SAR1-verifier-contrast-v1', 'utc_start': C.now(), 'versions': {}, 'inputs_sha256': {}}
    Zc = compact_po(X, names)
    men = sex == 0
    for v in (0, 1):
        d = V.RUNS / 'FA' / 'FA_TFRB_R2' / f'V{v}'
        buf = (d / 'MODEL.pkl').read_bytes()
        meta = json.loads((d / 'MODEL.json').read_text())
        if hashlib.sha256(buf).hexdigest() != meta['model_sha256']:
            raise PermissionError('MODEL.pkl differs from MODEL.json')
        P = pickle.loads(buf)
        tb = (C.NRM_FA_CACHE / f'V{v}' / 'TASK.pkl').read_bytes()
        if hashlib.sha256(tb).hexdigest() != P['task_cache_sha256']:
            raise PermissionError('TASK.pkl differs from the model\'s recorded task cache hash')
        T = pickle.loads(tb)
        out['inputs_sha256'][str((d / 'MODEL.pkl').relative_to(V.ROOT))] = meta['model_sha256']
        out['inputs_sha256'][str(C.NRM_FA_CACHE / f'V{v}' / 'TASK.pkl')] = P['task_cache_sha256']
        succ = fam_h.predict(v, 'H3_STRONG_LJDIR_PB', X, names)
        A = np.column_stack([logodds(succ['income']), clogits(succ['occupation']), logodds(T['oof']['inc']), clogits(T['oof']['occ']),
                             np.asarray(Zc, np.float64)])
        folds = V.household_parts(unit, 5, f'SAR_xf_FA_S{v}')
        occ = np.zeros((len(unit), 5))
        for j in range(5):
            te = folds == j
            occ[te] = probabilities('occupation', A[te, 1:6] + forward(P['nets'][j], phi(P['phis'][j], A[te])))
        dd = brier(yO, occ) - brier(yO, np.asarray(succ['occupation'], np.float64))
        mf_men, mf_women = float(dd[men].mean()), float(dd[~men].mean())
        per = [float(dd[(folds == j) & men].mean() - dd[(folds == j) & ~men].mean()) for j in range(5)]
        se_b = float(np.std(per, ddof=1) / np.sqrt(5))
        rec = json.loads((V.RES / 'fa' / f'FIT_R2_V{v}.json').read_text())['selection']['per_rho'][str(RHO_STAR[v])]['guard_estimate']
        out['versions'][str(v)] = {'rho_star': RHO_STAR[v], 'mf_men': mf_men, 'mf_women': mf_women, 'C_mf': mf_men - mf_women,
                                   'C_per_fold': per, 'se_C_between': se_b,
                                   'record': {k: rec[k] for k in ('mf_men', 'mf_women', 'C_mf', 'C_per_fold', 'se_C_between')},
                                   'max_abs_diff': max(abs(mf_men - rec['mf_men']), abs(mf_women - rec['mf_women']),
                                                       abs(mf_men - mf_women - rec['C_mf']), abs(se_b - rec['se_C_between']),
                                                       max(abs(x - y) for x, y in zip(per, rec['C_per_fold']))),
                                   'bag_size': len(P['nets'])}
        print(json.dumps(out['versions'][str(v)]), flush=True)
    out['cpu_s'] = time.process_time() - t0
    out['utc_end'] = C.now()
    V.jdump(Path(a.out), out)


if __name__ == '__main__':
    main()
