"""Independent check 9: refit the predeclared sample of pair-view WINNING readers and compare with the report.

Scope (predeclared, 5 refits): pair-view winners of LOCAL_TREE s0/s1, TASK_REFIT s0/s1 and NEURAL_RAW_rho1.0_E2 s0.
Everything metric-related here is written independently of acs_replication: own Mann-Whitney AUC (average ranks,
ties counted 1/2), own feature construction, own zero-sd-guarded scaler, own orientation and own winner rule.
sklearn estimators are instantiated with the registered hyperparameters (task_conditioned_representation/attacks.py
fit_reader, read but not imported).

Two comparisons per refit:
  REPLAY: own AUC on the lead's STORED DEV_REPORT scores (runs/READERS_s/DEV_REPORT_SCORES.npz) vs the reported winner AUC.
  REFIT:  own refit (ATTACK_FIT outputs + ATTACK_FIT SEX), orientation on ATTACK_VALIDATION, DEV_REPORT scores from
          released outputs alone; own AUC vs reported AUC, plus max |refit score - stored score|.
DEV_REPORT truth (SEX) is a verification re-read of an already-opened role. SEALED_ASSESSMENT is never touched.
No person-level values are printed or written; only aggregates.
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
SLATE = (('OUT_LR', 'out'), ('OUT_MLP', 'out'), ('OUT_HGB', 'out'), ('LOG_LR', 'log'), ('LOG_MLP', 'log'), ('DEC_LR', 'dec'), ('CONF_HGB', 'conf'))
KIND = dict(SLATE)
SAMPLE = (('LOCAL_TREE', 0), ('LOCAL_TREE', 1), ('TASK_REFIT', 0), ('TASK_REFIT', 1), ('NEURAL_RAW_rho1.0_E2', 0))
OUTPUT_FILE = {'LOCAL_TREE': 'STAGE_B_{s}/STAGE_B_OUTPUTS_E20.npz', 'TASK_REFIT': 'STAGE_B_{s}/STAGE_B_OUTPUTS_E20.npz',
               'NEURAL_RAW_rho1.0_E2': 'NEURAL_RAW_{s}/NEURAL_RAW_OUTPUTS.npz'}


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def mw_auc(y, s):
    """Mann-Whitney AUC with ties (average ranks); positive class = SEX code 1."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    if y.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError('aligned finite scores required')
    pos = y == 1; neg = y == 0
    if not (pos | neg).all():
        raise ValueError('binary labels required')
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        raise ValueError('both classes required')
    r = rankdata(s, method='average')
    u = float(r[pos].sum()) - n1 * (n1 + 1) / 2.0
    return u / (n1 * n0)


def load_export(role, field):
    rec = json.loads((PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())['files'][role][field]
    p = ROOT / rec['path']
    if role == 'SEALED_ASSESSMENT' or 'SEALED' in str(p):
        raise PermissionError('sealed assessment closed')
    h = sha(p)
    if h != rec['sha256']:
        raise ValueError(f'export hash mismatch {role}/{field}')
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}, h


def kind_features(p, k):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != k:
        raise ValueError('probability width differs')
    d = np.argmax(p, axis=1)
    onehot = np.zeros((len(p), k), dtype=np.float64); onehot[np.arange(len(p)), d] = 1.0
    return {'out': np.hstack([p, d[:, None].astype(np.float64)]),
            'log': np.hstack([np.log(np.maximum(p, 1e-300)), onehot]),
            'dec': onehot,
            'conf': p.max(axis=1)[:, None]}


def pair_features(p_inc, p_occ, kind):
    return np.hstack([kind_features(p_inc, 2)[kind], kind_features(p_occ, 5)[kind]])


class OwnScaler:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('finite matrix required')
        self.mu = x.mean(axis=0); sd = x.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float64) - self.mu) / self.sd


def make_model(name, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    if 'MLP' in name:
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=1e-4, max_iter=300,
                             early_stopping=False, random_state=seed)
    if 'HGB' in name:
        return HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=0.05, early_stopping=False,
                                              random_state=seed)
    return LogisticRegression(C=1, max_iter=500, solver='lbfgs')


def winner_rule(val_pair, val_income, val_occ):
    """Own re-implementation: candidates = 7 pair readers (slate order), then IGNORE_occupation__* (income view),
    then IGNORE_income__* (occupation view); highest ATTACK_VALIDATION AUC, ties to the earliest candidate."""
    cands = [(n, val_pair[n]) for n, _ in SLATE]
    cands += [(f'IGNORE_occupation__{n}', val_income[n]) for n, _ in SLATE]
    cands += [(f'IGNORE_income__{n}', val_occ[n]) for n, _ in SLATE]
    best = None
    for n, a in cands:
        if best is None or a > best[1]:
            best = (n, a)
    return best, len(cands)


def one_refit(job):
    svc, seed = job
    t0 = time.time()
    rs = json.loads((PRIV / 'runs' / f'READERS_{seed}' / 'READERS_SEED.json').read_text())
    srec = rs['services'][svc]
    win = srec['winner']['pair']['reader']
    if win.startswith('IGNORE_'):
        raise RuntimeError('alias winner: refit would be a single-view reader (not expected in this sample)')
    kind = KIND[win]
    fA, _ = load_export('ATTACK_FIT', 'features'); sA, _ = load_export('ATTACK_FIT', 'sex')
    fV, _ = load_export('ATTACK_VALIDATION', 'features'); sV, _ = load_export('ATTACK_VALIDATION', 'sex')
    fR, _ = load_export('DEV_REPORT', 'features'); tR, _ = load_export('DEV_REPORT', 'sex')
    align = {}
    for role, f, s in (('ATTACK_FIT', fA, sA), ('ATTACK_VALIDATION', fV, sV), ('DEV_REPORT', fR, tR)):
        align[role] = bool(np.array_equal(f['row_id'], s['row_id']) and np.array_equal(f['unit'], s['unit']))
    # Output-row alignment witness: STAGE_A stores row ids per role; they must equal the export order.
    with np.load(PRIV / 'runs' / f'STAGE_A_{seed}' / 'STAGE_A_OUTPUTS.npz') as z:
        for role, f in (('ATTACK_FIT', fA), ('ATTACK_VALIDATION', fV), ('DEV_REPORT', fR)):
            align[f'stage_a_row_id_{role}'] = bool(np.array_equal(z[f'row_id__{role}'], f['row_id']))
    with np.load(PRIV / 'runs' / OUTPUT_FILE[svc].format(s=seed)) as z:
        P = {r: (z[f'{svc}__{r}__income'].copy(), z[f'{svc}__{r}__occupation'].copy()) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')}
    for r, f in (('ATTACK_FIT', fA), ('ATTACK_VALIDATION', fV), ('DEV_REPORT', fR)):
        align[f'output_rows_{r}'] = bool(len(P[r][0]) == len(f['row_id']) == len(P[r][1]))
    xA = pair_features(*P['ATTACK_FIT'], kind); xV = pair_features(*P['ATTACK_VALIDATION'], kind); xR = pair_features(*P['DEV_REPORT'], kind)
    yA = sA['sex'].astype(np.int64); yV = sV['sex'].astype(np.int64); yR = tR['sex'].astype(np.int64)
    sc = OwnScaler().fit(xA)
    model = make_model(win, MODEL_SEED_BASE + seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(sc.transform(xA), yA)
    if list(model.classes_) != [0, 1]:
        raise RuntimeError('class order differs')
    raw_v = model.predict_proba(sc.transform(xV))[:, 1].astype(np.float64)
    a_v_raw = mw_auc(yV, raw_v)
    sign = 1 if a_v_raw >= 0.5 else -1
    orient = (lambda s: s) if sign == 1 else (lambda s: 1.0 - s)
    val_auc = mw_auc(yV, orient(raw_v))
    score_r = orient(model.predict_proba(sc.transform(xR))[:, 1].astype(np.float64))
    with np.load(PRIV / 'runs' / f'READERS_{seed}' / 'DEV_REPORT_SCORES.npz') as z:
        stored = z[f'{svc}__pair__{win}'].copy(); s_row = z['row_id'].copy(); s_unit = z['unit'].copy()
    align['scores_row_id_DEV_REPORT'] = bool(np.array_equal(s_row, fR['row_id']) and np.array_equal(s_unit, fR['unit']))
    rep = json.loads((RES / 'DEV_REPORT_RESULTS.json').read_text())['per_seed'][str(seed)]['services'][svc]['recovery']['pair']
    replay_auc = mw_auc(yR, stored)
    refit_auc = mw_auc(yR, score_r)
    # Winner rule replay from the stored ATTACK_VALIDATION AUCs (own rule).
    v = srec['validation']
    (w_name, w_auc), ncand = winner_rule({n: v['pair'][n]['fixed_orientation_auc'] for n, _ in SLATE},
                                         {n: v['income'][n]['fixed_orientation_auc'] for n, _ in SLATE},
                                         {n: v['occupation'][n]['fixed_orientation_auc'] for n, _ in SLATE})
    margin_to_runner_up = None
    allc = sorted([v['pair'][n]['fixed_orientation_auc'] for n, _ in SLATE] + [v['income'][n]['fixed_orientation_auc'] for n, _ in SLATE]
                  + [v['occupation'][n]['fixed_orientation_auc'] for n, _ in SLATE], reverse=True)
    margin_to_runner_up = allc[0] - allc[1]
    return {
        'service': svc, 'seed': seed, 'winner_reader': win, 'winner_is_ignore_alias': win.startswith('IGNORE_'), 'feature_kind': kind,
        'feature_width': int(xA.shape[1]), 'model_seed': MODEL_SEED_BASE + seed,
        'rows': {'ATTACK_FIT': int(len(yA)), 'ATTACK_VALIDATION': int(len(yV)), 'DEV_REPORT': int(len(yR))},
        'alignment': align,
        'reported': {'reader': rep['reader'], 'auc': rep['auc']},
        'stored_validation': {'orientation': v['pair'][win]['orientation'], 'fixed_orientation_auc': v['pair'][win]['fixed_orientation_auc']},
        'winner_rule_replay': {'own_winner': w_name, 'own_winner_validation_auc': w_auc, 'candidates': ncand,
                               'matches_stored_winner': w_name == win and w_name == rep['reader'],
                               'validation_auc_margin_to_runner_up': margin_to_runner_up},
        'replay': {'own_auc_on_stored_scores': replay_auc, 'abs_diff_vs_reported': abs(replay_auc - rep['auc'])},
        'refit': {'orientation': sign, 'validation_auc': val_auc,
                  'validation_auc_abs_diff_vs_stored': abs(val_auc - v['pair'][win]['fixed_orientation_auc']),
                  'orientation_matches_stored': sign == v['pair'][win]['orientation'],
                  'dev_report_auc': refit_auc, 'abs_diff_vs_reported': abs(refit_auc - rep['auc']),
                  'max_abs_score_diff_vs_stored': float(np.max(np.abs(score_r - stored))),
                  'scores_bitwise_equal_stored': bool(np.array_equal(score_r, stored)),
                  'n_iter': int(getattr(model, 'n_iter_', -1))},
        'wall_s': round(time.time() - t0, 2),
    }


def view_features(P, view, kind):
    if view == 'income':
        return kind_features(P[0], 2)[kind]
    if view == 'occupation':
        return kind_features(P[1], 5)[kind]
    return pair_features(P[0], P[1], kind)


def winner_choice_refit(job):
    """Extra: refit all 21 views x readers of a sampled service on ATTACK_FIT, orient and score ATTACK_VALIDATION with
    own code, apply own winner rule (pair readers + 14 IGNORE aliases) and compare with the stored validation AUCs/winner."""
    svc, seed = job
    rs = json.loads((PRIV / 'runs' / f'READERS_{seed}' / 'READERS_SEED.json').read_text())['services'][svc]
    sA, _ = load_export('ATTACK_FIT', 'sex'); sV, _ = load_export('ATTACK_VALIDATION', 'sex')
    yA = sA['sex'].astype(np.int64); yV = sV['sex'].astype(np.int64)
    with np.load(PRIV / 'runs' / OUTPUT_FILE[svc].format(s=seed)) as z:
        P = {r: (z[f'{svc}__{r}__income'].copy(), z[f'{svc}__{r}__occupation'].copy()) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION')}
    val = {}; max_diff = 0.0; orient_mismatch = 0
    for view in ('income', 'occupation', 'pair'):
        val[view] = {}
        for name, kind in SLATE:
            xa = view_features(P['ATTACK_FIT'], view, kind); xv = view_features(P['ATTACK_VALIDATION'], view, kind)
            sc = OwnScaler().fit(xa); m = make_model(name, MODEL_SEED_BASE + seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.fit(sc.transform(xa), yA)
            raw = m.predict_proba(sc.transform(xv))[:, 1].astype(np.float64)
            sign = 1 if mw_auc(yV, raw) >= 0.5 else -1
            a = mw_auc(yV, raw if sign == 1 else 1.0 - raw)
            val[view][name] = a
            max_diff = max(max_diff, abs(a - rs['validation'][view][name]['fixed_orientation_auc']))
            orient_mismatch += int(sign != rs['validation'][view][name]['orientation'])
    (w, wa), ncand = winner_rule(val['pair'], val['income'], val['occupation'])
    return {'service': svc, 'seed': seed, 'fits': 21, 'own_refit_winner': w, 'own_refit_winner_validation_auc': wa,
            'stored_winner': rs['winner']['pair']['reader'], 'matches': w == rs['winner']['pair']['reader'],
            'max_validation_auc_abs_diff_vs_stored_all_21': max_diff, 'orientation_mismatches': orient_mismatch,
            'own_view_winners': {v: max(SLATE, key=lambda nk: (val[v][nk[0]], -[n for n, _ in SLATE].index(nk[0])))[0] for v in ('income', 'occupation')},
            'stored_view_winners': {v: rs['winner'][v]['reader'] for v in ('income', 'occupation')}}


def main():
    import sklearn
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(one_refit, SAMPLE))
        wc = list(ex.map(winner_choice_refit, SAMPLE))
    winner_refit_ok = all(r['matches'] and r['max_validation_auc_abs_diff_vs_stored_all_21'] <= 1e-6 and r['orientation_mismatches'] == 0 for r in wc)
    replay_ok = all(r['replay']['abs_diff_vs_reported'] <= 1e-12 for r in results)
    refit_ok = all(r['refit']['abs_diff_vs_reported'] <= 1e-6 for r in results)
    winner_ok = all(r['winner_rule_replay']['matches_stored_winner'] for r in results)
    align_ok = all(all(r['alignment'].values()) for r in results)
    orient_ok = all(r['refit']['orientation_matches_stored'] for r in results)
    status = 'PASS' if (replay_ok and refit_ok and winner_ok and align_ok and orient_ok and winner_refit_ok) else 'FAIL'
    rs = {s: json.loads((PRIV / 'runs' / f'READERS_{s}' / 'READERS_SEED.json').read_text()) for s in (0, 1)}
    scores_sha = {s: sha(PRIV / 'runs' / f'READERS_{s}' / 'DEV_REPORT_SCORES.npz') for s in (0, 1)}
    out = {
        'id': 9, 'name': 'READER REFITS (predeclared sample: pair-view winners)', 'status': status,
        'evidence': {
            'sample': [f'{s}|s{k}' for s, k in SAMPLE],
            'criteria': {'replay_abs_diff_le_1e-12': replay_ok, 'refit_abs_diff_le_1e-6': refit_ok,
                         'winner_rule_replay_matches': winner_ok, 'row_alignment_all_true': align_ok,
                         'refit_orientation_matches_stored': orient_ok,
                         'winner_choice_refit_all_21_candidates_matches': winner_refit_ok},
            'winner_choice_refit': wc,
            'max_replay_abs_diff': max(r['replay']['abs_diff_vs_reported'] for r in results),
            'max_refit_abs_diff': max(r['refit']['abs_diff_vs_reported'] for r in results),
            'max_refit_score_abs_diff_vs_stored': max(r['refit']['max_abs_score_diff_vs_stored'] for r in results),
            'all_refit_scores_bitwise_equal_stored': all(r['refit']['scores_bitwise_equal_stored'] for r in results),
            'dev_report_scores_sha256': {str(s): scores_sha[s] for s in (0, 1)},
            'dev_report_scores_sha256_matches_receipt': {str(s): scores_sha[s] == rs[s]['dev_report_scores_sha256'] for s in (0, 1)},
            'lead_sklearn_version': {str(s): rs[s].get('sklearn_version') for s in (0, 1)},
            'own_sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
            'refits': results,
        },
        'scope': ('REPLAY = own AUC recomputed on the lead\'s stored DEV_REPORT reader scores; REFIT = own re-estimation of '
                  'the 5 predeclared pair-view winning readers from released ATTACK_FIT outputs + ATTACK_FIT SEX with the '
                  'registered sklearn algorithm/hyperparameters and own scaler/features/orientation; winner choice replayed '
                  'from stored ATTACK_VALIDATION AUCs AND re-derived from an extra refit of all 21 view x reader fits per '
                  'sampled service (7 pair readers + 14 single-view readers used as IGNORE aliases), oriented and scored on '
                  'ATTACK_VALIDATION with own code. Services outside the sample (T_TREE, U0, JOINT_TREE, FAGTB, FARE, '
                  'NEURAL_RAW s1) are covered by replay only (parent check 8).'),
        'DEV_REPORT_truth_reread': True, 'sealed_assessment_touched': False,
        'wall_s': round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'check09_reader_refits.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': status, **{k: out['evidence'][k] for k in ('criteria', 'max_replay_abs_diff', 'max_refit_abs_diff', 'max_refit_score_abs_diff_vs_stored')},
                      'per_refit': [(r['service'], r['seed'], r['winner_reader'], round(r['reported']['auc'], 6), round(r['refit']['dev_report_auc'], 6), r['wall_s']) for r in results]}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
