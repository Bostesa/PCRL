"""Verifier check 5: development reader grids (GRID.json / BOOT.npz). Recomputes, with the verifier's own code,
  (a) every paired cell difference and its household-bootstrap SE from the recorded SCORE AUCs and bootstrap draws;
  (b) the new release arrays the grid read (file SHA-256, recomposition from the partition's witness and correction at the stated
      strength, wire hashes);
  (c) a few deterministic readers refitted from scratch (OUT_LR and LOG_LR, logistic regression on standardized released vectors /
      decisions) for two releases in one cell, compared with the grid's cached reader scores and SCORE AUCs (tolerance 1e-6), and
      the recorded winner SCORE AUCs recomputed from the cached winner scores.
PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records). SEX of
MODEL_SELECT through cts.common.load(operation='verify'). Nothing person-level is printed or written.

    python3 .../v_grid.py --bank D_FRONTIER --stage D --tag V1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402

CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
TRACKF = CAMP / 'runs' / 'TRACKF'
GRID = CAMP / 'runs' / 'GRID'
OWN_CACHE = CAMP / 'runs' / 'devgrid_cache'
SRC_CACHE = ROOT / '.private' / 'pcrl_occupation_headroom_overnight_v1' / 'runs' / 'devgrid_cache'
GRID_VERSION = 'OHN1-devgrid-v1'
MODEL_SEED_BASE = 2026101205
REP_TAG = {'R1': 'OHN1_devgrid_R1', 'R2': 'OHN1_devgrid_R2'}
STAGE_PARTS = {'D': {0: 'D0', 1: 'D1'}, 'R': {0: 'R0', 1: 'R1'}}


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def ms_sex():
    return C.load('MODEL_SELECT', ('features', 'sex'), 'verify',
                  'independent verifier: replay of development reader grid cells (paired differences, deterministic reader refits)')


def my_auc(y, s):
    y = np.asarray(y); r = rankdata(np.asarray(s, dtype=np.float64), method='average')
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def wire_sha(p):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes()); h.update(np.ascontiguousarray(np.argmax(p, 1).astype(np.int64)).tobytes())
    return h.hexdigest()


def view_hash(view, inc, occ):
    h = hashlib.sha256(view.encode())
    for a in ((inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def cache_entry(rep, seed, k, view, inc, occ, sex_sha):
    rs = MODEL_SEED_BASE + 300 + 10 * seed + k
    key = hashlib.sha256(f'{GRID_VERSION}|{rep}|{rs}|1.0|{view_hash(view, inc, occ)}|{sex_sha}'.encode()).hexdigest()
    for d in (OWN_CACHE, SRC_CACHE):
        p = d / f'{key}.npz'
        if p.is_file():
            return npz(p), str(p.parent.name if d == OWN_CACHE else 'source_cache')
    return None, None


def features(kind, p, k):
    p = np.asarray(p, dtype=np.float64); d = np.argmax(p, 1)
    if kind == 'out':
        return np.column_stack((p, d.astype(np.float64)))
    if kind == 'log':
        return np.column_stack((np.log(np.maximum(p, 1e-300)), np.eye(k)[d]))
    raise KeyError(kind)


def lr_reader(X, y, fit, val):
    from sklearn.linear_model import LogisticRegression
    mu = X[fit].mean(0); sd = X[fit].std(0); sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs').fit(Z[fit], y[fit])
    s = m.predict_proba(Z)[:, 1].astype(np.float64)
    sign = 1 if my_auc(y[val], s[val]) >= 0.5 else -1
    return (s if sign == 1 else 1 - s), sign


def release_of(stage, seed, name):
    """Own recomposition of a Track F release on MODEL_SELECT: W, or <DIR>@s<x> = softmax(log p_W + x Delta_full)."""
    P = STAGE_PARTS[stage][seed]
    W = npz(TRACKF / P / 'WITNESS' / 'WITNESS.npz')
    if name == 'W':
        return W['dep__MODEL_SELECT']
    m = re.match(r'^([A-Z_]+)@s([0-9.]+)$', name)
    if not m:
        return None
    D = npz(TRACKF / P / 'DIR' / m.group(1) / 'DELTA.npz')
    return V.release_full(W['dep__MODEL_SELECT'], D['full__MODEL_SELECT'], float(m.group(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True); ap.add_argument('--stage', required=True, choices=('D', 'R'))
    ap.add_argument('--tag', required=True); ap.add_argument('--refit', default='W,REFIT_PAIR@s1.0')
    a = ap.parse_args()
    S = ms_sex()
    sex = S['sex']; unit = S['unit']
    sex_sha = V.arr_sha(np.asarray(sex))
    from ohn import devrun as OD                      # admitted, hash-checked frozen anchor income (label-free)
    out = []
    for sd in sorted((GRID / a.bank).glob('S*')):
        seed = int(sd.name[1:])
        g = json.loads((sd / 'GRID.json').read_text())
        B = npz(sd / 'BOOT.npz')
        # (a) paired differences
        worst_d, worst_se, n, miss = 0.0, 0.0, 0, []
        for cand, comps in g['paired'].items():
            for comp, views in comps.items():
                for view, rec in views.items():
                    for cell, d_rec in rec['cells'].items():
                        A, Bc = g['cells'][cell][cand][view], g['cells'][cell][comp][view]
                        ba, bb = B[f'{cell}__{cand}__{view}'], B[f'{cell}__{comp}__{view}']
                        if view == 'pair':
                            d, db = Bc['score_auc'] - A['score_auc'], bb - ba
                        else:
                            d, db = A['score_auc'] - Bc['score_auc'], ba - bb
                        se = float(np.std(db, ddof=1))
                        worst_d = max(worst_d, abs(d - d_rec)); worst_se = max(worst_se, abs(se - rec['cell_se'][cell])); n += 1
        out.append({'id': f'C5.{a.bank}.S{seed}.PAIRED_DIFFERENCES', 'kind': 'NUMERICAL_REPLAY',
                    'what': f'{a.bank} seed {seed}: every paired cell difference (pair view: comparator - candidate, a benefit; single '
                            'views: candidate - comparator) and its household-bootstrap SE (paired draws)',
                    'method': 'recomputed from GRID.json SCORE AUCs and BOOT.npz draws (SD with ddof 1); tolerance 1e-12',
                    'status': 'PASS' if (worst_d <= 1e-12 and worst_se <= 1e-12 and not miss) else 'FAIL',
                    'numbers': {'paired_cells_checked': n, 'max_abs_diff_cell': worst_d, 'max_abs_diff_se': worst_se}})
        # (b) release materialization
        inc = OD.anchor_income('H3', seed, 'MODEL_SELECT')
        rel_prob, rels, stored = [], {}, {}
        for name, mem in g['members'].items():
            occ_src = mem.get('occupation') if isinstance(mem, dict) else None
            if not (isinstance(occ_src, dict) and 'path' in occ_src):
                continue
            p = ROOT / occ_src['path']
            ok_sha = V.file_sha(p) == occ_src['sha256']
            arr = npz(p)[occ_src['key']]
            stored[name] = arr if ok_sha else None
            mine = release_of(a.stage, seed, name)
            diff = float(np.max(np.abs(mine - arr))) if mine is not None else None
            ws = wire_sha(arr) == mem['wire_sha256']['occupation']
            wi = (wire_sha(inc) == mem['wire_sha256']['income']) if mem.get('income', {}).get('frozen_anchor') == 'H3' else None
            rels[name] = {'file_sha_ok': ok_sha, 'recomposition_max_abs_diff': diff, 'occupation_wire_sha_ok': ws, 'income_wire_sha_ok': wi}
            if not ok_sha or not ws or wi is False or diff is None or diff > 1e-12:
                rel_prob.append(name)
        out.append({'id': f'C5.{a.bank}.S{seed}.RELEASE_MATERIALIZATION', 'kind': 'NUMERICAL_REPLAY',
                    'what': f'{a.bank} seed {seed}: the new release arrays read by the grid equal the verifier\'s recomposition from the '
                            f'{STAGE_PARTS[a.stage][seed]} witness and full-FIT correction at the stated strength; file and wire hashes',
                    'method': 'own softmax(centred log p_W + s Delta_full) on MODEL_SELECT (tolerance 1e-12); own wire hash (float64 vector + '
                              'int64 argmax)',
                    'status': 'PASS' if not rel_prob else 'FAIL', 'numbers': {'releases': rels, 'problems': rel_prob}})
        # (c) reader refits in cell S<seed>R1k0 + winner AUCs from cached winner scores
        rep, k = 'R1', 0
        part = V.deal(unit, 10, REP_TAG[rep])
        fit, val, sc = part <= 4, (part >= 5) & (part <= 6), part >= 7
        cell = f'S{seed}{rep}k{k}'
        refit, win, prob = {}, {}, []
        for name in [x for x in a.refit.split(',') if x]:
            # readers read the exact stored release bytes (hash- and recomposition-checked in RELEASE_MATERIALIZATION above); a
            # recomposed array differs in the last bits (<= 5e-16) and would change the cache key (CHECK_CHANGES.json, change 1)
            occ = stored.get(name)
            if occ is None or name not in g['cells'][cell]:
                prob.append(f'{name}: no hash-checked stored release'); continue
            for view in ('pair', 'income', 'occupation'):
                ent, where = cache_entry(rep, seed, k, view, inc, occ, sex_sha)
                if ent is None:
                    prob.append(f'{name}/{view}: no cache entry'); continue
                info = json.loads(str(ent['info']))
                # winner score AUC from the cached winner scores
                wname = g['cells'][cell][name][view]['winner']
                wkey = wname.split('__', 1)[1] if '__' in wname else wname
                if wname.startswith('IGNORE_'):
                    other = 'income' if wname.startswith('IGNORE_occupation') else 'occupation'
                    ent2, _ = cache_entry(rep, seed, k, other, inc, occ, sex_sha)
                    ws_ = ent2[f'score__{wkey}'] if ent2 is not None else None
                else:
                    ws_ = ent[f'score__{wkey}']
                wa = my_auc(sex[sc], ws_) if ws_ is not None else None
                win[f'{name}/{view}'] = {'winner': wname, 'recorded_score_auc': g['cells'][cell][name][view]['score_auc'],
                                         'auc_of_cached_winner_scores': wa,
                                         'abs_diff': abs(wa - g['cells'][cell][name][view]['score_auc']) if wa is not None else None}
                if wa is None or abs(wa - g['cells'][cell][name][view]['score_auc']) > 1e-12:
                    prob.append(f'{name}/{view}: winner AUC')
                if view != 'pair':
                    continue
                for recipe, kind in (('OUT_LR', 'out'), ('LOG_LR', 'log')):
                    X = np.column_stack((features(kind, inc, 2), features(kind, occ, 5)))
                    s, sign = lr_reader(X, sex, fit, val)
                    cached = ent[f'score__{recipe}']
                    r = {'cache': where, 'orientation_mine': sign, 'orientation_cached': info[recipe].get('orientation'),
                         'score_max_abs_diff': float(np.max(np.abs(s[sc] - cached))),
                         'score_auc_mine': my_auc(sex[sc], s[sc]), 'score_auc_cached': my_auc(sex[sc], cached),
                         'val_auc_mine': my_auc(sex[val], s[val]), 'val_auc_cached': info[recipe]['val_auc']}
                    r['score_auc_abs_diff'] = abs(r['score_auc_mine'] - r['score_auc_cached'])
                    r['val_auc_abs_diff'] = abs(r['val_auc_mine'] - r['val_auc_cached'])
                    refit[f'{name}/{view}/{recipe}'] = r
                    if r['score_auc_abs_diff'] > 1e-6 or r['val_auc_abs_diff'] > 1e-6 or r['orientation_mine'] != r['orientation_cached']:
                        prob.append(f'{name}/{view}/{recipe}')
        out.append({'id': f'C5.{a.bank}.S{seed}.READER_REFIT_{cell}', 'kind': 'NUMERICAL_REPLAY',
                    'what': f'{a.bank} seed {seed}, cell {cell}: deterministic logistic readers (OUT_LR, LOG_LR) on the pair view refitted '
                            'from scratch for releases ' + a.refit + '; winner SCORE AUCs of all three views recomputed from the cached '
                            'winner scores',
                    'method': 'own household dealing of MODEL_SELECT (OHN1_devgrid_R1: FIT parts 0-4, VALIDATION 5-6, SCORE 7-9), own '
                              'features (out: vector + argmax; log: log vector + one-hot decision), own standardization, scikit-learn '
                              'LogisticRegression(C=1, lbfgs, 500), orientation on VALIDATION, own rank AUC; tolerance 1e-6 (refits) / '
                              '1e-12 (cached winners)',
                    'status': 'PASS' if not prob else 'FAIL', 'numbers': {'refits': refit, 'winners': win, 'problems': prob}})
    code = V.code_hashes(__file__, V.__file__)
    for c in out:
        c['id'] = f"{c['id']}__{a.tag}"
        c['code_sha256'] = code
    V.write_record(f'V_GRID__{a.bank}__{a.tag}.json', {'checks': out, 'utc': V.utc()})
    print(json.dumps(V.append_checks(out)))
    print(json.dumps({c['id']: c['status'] for c in out}, indent=1))


if __name__ == '__main__':
    main()
