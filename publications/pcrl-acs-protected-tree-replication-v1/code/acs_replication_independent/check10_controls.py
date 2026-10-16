"""Independent check 10: real-data controls on the T_TREE wire (shuffled-SEX null and planted SEX bit), per view, per seed.

PRIMARY (required): own Mann-Whitney AUC recomputed on the lead's STORED DEV_REPORT control scores
  shuffled: AUC(SEX, CONTROL__<view>__SHUFFLED)
  planted : AUC(SEX, score with the person's own SEX bit planted) = PLANTED_1 where SEX==1 else PLANTED_0
compared with DEV_REPORT_RESULTS per_seed[s].controls (tolerance 1e-12), plus registered forecast ranges
F13 (all shuffled-null AUCs in [0.47, 0.53]) and F14 (all planted-bit AUCs > 0.95).
SECONDARY (extra, independent refit): re-estimate both controls from released T_TREE ATTACK_FIT outputs with
OUT_HGB (registered hyperparameters), the registered null permutation rng (2026101602 + seed) and the planted column,
then compare refit DEV_REPORT scores with the stored scores and refit ATTACK_VALIDATION AUCs with the receipt.
Aggregates only; DEV_REPORT SEX is a verification re-read of an already-opened role; SEALED_ASSESSMENT untouched.
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import hashlib
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
PRIV = ROOT / '.private' / 'acs_replication'
RES = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
OUT = RES / 'independent'
MODEL_SEED_BASE = 2026101205
NULL_SEED = 2026101602
VIEWS = ('income', 'occupation', 'pair')


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def mw_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    pos = y == 1; neg = y == 0
    if y.ndim != 1 or s.shape != y.shape or not (pos | neg).all() or not np.isfinite(s).all():
        raise ValueError('aligned binary labels and finite scores required')
    n1, n0 = int(pos.sum()), int(neg.sum())
    r = rankdata(s, method='average')
    return (float(r[pos].sum()) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def load_export(role, field):
    rec = json.loads((PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())['files'][role][field]
    p = ROOT / rec['path']
    if 'SEALED' in str(p):
        raise PermissionError('sealed assessment closed')
    if sha(p) != rec['sha256']:
        raise ValueError(f'export hash mismatch {role}/{field}')
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def out_kind(p, k):
    p = np.asarray(p, dtype=np.float64)
    if p.shape[1] != k:
        raise ValueError('width differs')
    return np.hstack([p, np.argmax(p, axis=1)[:, None].astype(np.float64)])


def view_x(P, view):
    a, b = out_kind(P[0], 2), out_kind(P[1], 5)
    return a if view == 'income' else b if view == 'occupation' else np.hstack([a, b])


class OwnScaler:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64); self.mu = x.mean(axis=0); sd = x.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0); return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mu) / self.sd


def hgb(x, y, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    sc = OwnScaler().fit(x)
    m = HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.fit(sc.transform(x), y)
    if list(m.classes_) != [0, 1]:
        raise RuntimeError('class order differs')
    return lambda z: m.predict_proba(sc.transform(z))[:, 1].astype(np.float64)


def refit_seed_view(job):
    seed, view = job
    sA = load_export('ATTACK_FIT', 'sex')['sex'].astype(np.int64)
    sV = load_export('ATTACK_VALIDATION', 'sex')['sex'].astype(np.int64)
    with np.load(PRIV / 'runs' / f'STAGE_A_{seed}' / 'STAGE_A_OUTPUTS.npz') as z:
        P = {r: (z[f'T_TREE__{r}__income'].copy(), z[f'T_TREE__{r}__occupation'].copy()) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')}
    x, xv, xr = view_x(P['ATTACK_FIT'], view), view_x(P['ATTACK_VALIDATION'], view), view_x(P['DEV_REPORT'], view)
    ms = MODEL_SEED_BASE + seed
    shuffled = np.random.default_rng(NULL_SEED + seed).permutation(sA)
    f_s = hgb(x, shuffled, ms)
    f_p = hgb(np.hstack([x, sA[:, None].astype(np.float64)]), sA, ms)
    s_shuf = f_s(xr)
    s_p0 = f_p(np.hstack([xr, np.zeros((len(xr), 1))])); s_p1 = f_p(np.hstack([xr, np.ones((len(xr), 1))]))
    with np.load(PRIV / 'runs' / f'READERS_{seed}' / 'DEV_REPORT_SCORES.npz') as z:
        st = {k: z[f'CONTROL__{view}__{k}'].copy() for k in ('SHUFFLED', 'PLANTED_0', 'PLANTED_1')}
    return {'seed': seed, 'view': view,
            'shuffled_label_fraction_changed': float(np.mean(shuffled != sA)),
            'refit_validation_auc': {'shuffled': mw_auc(sV, f_s(xv)), 'planted': mw_auc(sV, f_p(np.hstack([xv, sV[:, None].astype(np.float64)])))},
            'max_abs_score_diff_vs_stored': {'SHUFFLED': float(np.max(np.abs(s_shuf - st['SHUFFLED']))),
                                             'PLANTED_0': float(np.max(np.abs(s_p0 - st['PLANTED_0']))),
                                             'PLANTED_1': float(np.max(np.abs(s_p1 - st['PLANTED_1'])))}}


def main():
    t0 = time.time()
    dev = load_export('DEV_REPORT', 'sex'); feat = load_export('DEV_REPORT', 'features')
    if not (np.array_equal(dev['row_id'], feat['row_id']) and np.array_equal(dev['unit'], feat['unit'])):
        raise RuntimeError('DEV_REPORT truth/feature pairing differs')
    sex = dev['sex'].astype(np.int64)
    rep = json.loads((RES / 'DEV_REPORT_RESULTS.json').read_text())
    primary = {}; diffs = []; shuf_vals = []; plant_vals = []; align = {}
    for seed in (0, 1):
        with np.load(PRIV / 'runs' / f'READERS_{seed}' / 'DEV_REPORT_SCORES.npz') as z:
            align[str(seed)] = bool(np.array_equal(z['row_id'], dev['row_id']) and np.array_equal(z['unit'], dev['unit']))
            S = {k: z[k].copy() for k in z.files if k.startswith('CONTROL__')}
        primary[str(seed)] = {}
        for view in VIEWS:
            a_s = mw_auc(sex, S[f'CONTROL__{view}__SHUFFLED'])
            a_p = mw_auc(sex, np.where(sex == 1, S[f'CONTROL__{view}__PLANTED_1'], S[f'CONTROL__{view}__PLANTED_0']))
            r = rep['per_seed'][str(seed)]['controls'][view]
            d_s, d_p = abs(a_s - r['shuffled_null_auc']), abs(a_p - r['planted_bit_auc'])
            diffs += [d_s, d_p]; shuf_vals.append(a_s); plant_vals.append(a_p)
            primary[str(seed)][view] = {'own_shuffled_null_auc': a_s, 'reported_shuffled_null_auc': r['shuffled_null_auc'], 'abs_diff_shuffled': d_s,
                                        'own_planted_bit_auc': a_p, 'reported_planted_bit_auc': r['planted_bit_auc'], 'abs_diff_planted': d_p}
    with ProcessPoolExecutor(max_workers=6) as ex:
        refits = list(ex.map(refit_seed_view, [(s, v) for s in (0, 1) for v in VIEWS]))
    receipts = {s: json.loads((PRIV / 'runs' / f'READERS_{s}' / 'READERS_SEED.json').read_text())['controls'] for s in (0, 1)}
    for r in refits:
        st = receipts[r['seed']][r['view']]
        r['stored_validation_auc'] = {'shuffled': st['shuffled_attack_validation_auc'], 'planted': st['planted_attack_validation_auc']}
        r['validation_auc_abs_diff'] = {k: abs(r['refit_validation_auc'][k] - r['stored_validation_auc'][k]) for k in ('shuffled', 'planted')}
    max_primary = max(diffs)
    f13_by_seed = {str(s): all(0.47 <= primary[str(s)][v]['own_shuffled_null_auc'] <= 0.53 for v in VIEWS) for s in (0, 1)}
    f14_by_seed = {str(s): all(primary[str(s)][v]['own_planted_bit_auc'] > 0.95 for v in VIEWS) for s in (0, 1)}
    max_refit_score = max(max(r['max_abs_score_diff_vs_stored'].values()) for r in refits)
    max_refit_val = max(max(r['validation_auc_abs_diff'].values()) for r in refits)
    primary_ok = max_primary <= 1e-12 and all(align.values())
    status = 'PASS' if primary_ok else 'FAIL'
    out = {
        'id': 10, 'name': 'CONTROLS (shuffled-SEX null and planted SEX bit on the T_TREE wire, DEV_REPORT)', 'status': status,
        'evidence': {
            'primary_stored_score_recompute': primary, 'max_abs_diff_vs_reported': max_primary,
            'dev_report_scores_row_alignment': align,
            'forecast_F13_all_shuffled_in_[0.47,0.53]': {'by_seed': f13_by_seed, 'both_seeds': all(f13_by_seed.values()),
                                                          'min': min(shuf_vals), 'max': max(shuf_vals)},
            'forecast_F14_all_planted_gt_0.95': {'by_seed': f14_by_seed, 'both_seeds': all(f14_by_seed.values()), 'min': min(plant_vals)},
            'secondary_independent_refit': {'max_abs_score_diff_vs_stored': max_refit_score,
                                            'max_validation_auc_abs_diff_vs_receipt': max_refit_val,
                                            'refit_reproduces_stored_scores_bitwise': max_refit_score == 0.0,
                                            'per_seed_view': refits},
        },
        'notes': ('PASS/FAIL governed by the stored-score recompute (tolerance 1e-12) and row alignment; the refit is an extra '
                  'independent re-estimation of the 12 control readers (registered OUT_HGB, null rng 2026101602+seed). '
                  'F13/F14 are registered forecasts (scored, not gates). Control interpretation: shuffled-null near 0.5 shows '
                  'no reader/wire leakage of SEX by construction; planted bit = 1.0 shows the reader slate can find a SEX signal when present.'),
        'DEV_REPORT_truth_reread': True, 'sealed_assessment_touched': False, 'wall_s': round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'check10_controls.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': status, 'max_abs_diff_vs_reported': max_primary, 'F13': out['evidence']['forecast_F13_all_shuffled_in_[0.47,0.53]'],
                      'F14': out['evidence']['forecast_F14_all_planted_gt_0.95'], 'refit_max_score_diff': max_refit_score,
                      'refit_max_val_auc_diff': max_refit_val}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
