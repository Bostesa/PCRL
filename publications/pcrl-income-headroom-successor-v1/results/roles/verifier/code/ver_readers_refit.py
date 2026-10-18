"""Independent verifier, check 8a: refit the READER_PLAN verifier refit roster (PAIR__{SUCCESSOR,NEURAL_RAW,H_LJ}__{ET5,HGB31,
MLP6432,CATI1}), the ENHANCED and ORIGINAL pair winners of SUCCESSOR, NEURAL_RAW and H_LJ, and (extension) every other pair
recipe of those three services, for one seed; compare AV AUCs and the oriented AV / DEV_REPORT score arrays with the stored ones.

Context: public-use U.S. Census ACS 2019 1-Year PUMS Florida person records (development roles). Readers estimate the dataset's
recorded SEX column from the released income-class and occupation-group probability vectors (each recipient's float64 vector and
argmax decision; the linked pair). Fit on ATTACK_FIT SEX, orientation on ATTACK_VALIDATION SEX (logged loader, after the
committed FINALIST_LOCK), DEV_REPORT scored from released outputs only (no DEV_REPORT label is opened here).

Own code: driver, feature tables for the new recipes (rich), standardization, LOOKUP table, AUC, orientation rule, comparisons.
Allowed constructors used: task_conditioned_representation.attacks.fit_reader / predict_reader (source recipes) with
scm_bench.readers.kinds features; ffv.readers.make_model (new recipes, estimator construction only); ffv.features.CatiDesign
(categorical-interaction design; ffv.readers.design_for is NOT called). Interpreter: the READER_PLAN .venv.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata

from ihs import common as C                                         # admitted, hash-checked loaders only
from scm_bench.readers import kinds                                  # source feature kinds (allowed)
from task_conditioned_representation.attacks import MODEL_SEED_BASE, fit_reader, predict_reader   # source recipes (allowed)
from ffv.readers import make_model                                   # new-recipe estimator construction (allowed)
from ffv import features as F                                        # CatiDesign (allowed); rich_view used only as a cross-check

SLATE_KIND = {'OUT_LR': 'out', 'OUT_MLP': 'out', 'OUT_HGB': 'out', 'LOG_LR': 'log', 'LOG_MLP': 'log', 'DEC_LR': 'dec', 'CONF_HGB': 'conf'}
SRC8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
ROSTER = ('SUCCESSOR', 'NEURAL_RAW', 'H_LJ')
PLAN_RECIPES = ('ET5', 'HGB31', 'MLP6432', 'CATI1')
ROLES3 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
OUT = ROOT / '.private' / C.CAMPAIGN / 'verifier'
PSEUDO = 2.0


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def mw_auc(y, s):
    y = np.asarray(y).astype(np.int64); r = rankdata(np.asarray(s, dtype=np.float64), method='average')
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def my_orient(y_val, raw_val):
    return 1 if mw_auc(y_val, raw_val) >= 0.5 else -1


def apply_sign(s, sign):
    return np.asarray(s) if sign == 1 else 1 - np.asarray(s)


def rich(p):
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack((p, np.log(np.maximum(p, 1e-300)), np.eye(p.shape[1])[p.argmax(1)]))


def rich_tab(view, inc, occ):
    return rich(inc) if view == 'income' else rich(occ) if view == 'occupation' else np.column_stack((rich(inc), rich(occ)))


def src_tab(view, inc, occ, kind):
    a, b = kinds(inc, 2)[kind], kinds(occ, 5)[kind]
    return a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))


def lookup_keys(view, inc, occ):
    vec = [inc] if view == 'income' else [occ] if view == 'occupation' else [inc, occ]
    z = np.ascontiguousarray(np.column_stack([np.asarray(v, dtype=np.float64) for v in vec]))
    return [row.tobytes() for row in z]


class Scaler:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64); self.m = X.mean(0); sd = X.std(0); self.s = np.where(sd > 0, sd, 1.0); return self

    def __call__(self, X):
        return (np.asarray(X, dtype=np.float64) - self.m) / self.s


def load_bank(seed):
    man = json.loads((C.RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    p = C.RUNS / 'FINAL' / f'SERVICE_BANK_S{seed}.npz'
    h = sha256(p)
    if h != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('bank hash differs from SERVICE_BANK_MANIFEST')
    bank, ids = {}, {}
    with np.load(p) as z:
        for k in z.files:
            parts = k.split('__')
            if parts[0] in ('row_id', 'unit'):
                ids[k] = z[k].copy()
            elif len(parts) == 3:
                bank.setdefault(parts[0], {}).setdefault(parts[1], {})[parts[2]] = z[k].copy()
    return bank, ids, h


def wire_name(bank, svc, t):
    for s in sorted(bank):
        if all(np.array_equal(bank[s][r][t], bank[svc][r][t]) for r in ROLES3):
            return f'{s}|{t}'


def refit_one(name, view, trip, yA, yV, seed_model):
    t0 = time.process_time(); info = {}
    if name == 'LOOKUP':
        k3 = [lookup_keys(view, i, o) for i, o in trip]
        pi = float(np.asarray(yA).mean()); table = {}
        for k, v in zip(k3[0], yA):
            n1, n = table.get(k, (0, 0)); table[k] = (n1 + int(v), n + 1)
        pred = lambda keys: np.array([(table[k][0] + PSEUDO * pi) / (table[k][1] + PSEUDO) if k in table else pi for k in keys], dtype=np.float64)
        raw_v, raw_d = pred(k3[1]), pred(k3[2])
    elif name in SLATE_KIND:
        X3 = [src_tab(view, i, o, SLATE_KIND[name]) for i, o in trip]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = fit_reader(X3[0], yA, name, seed_model)
        raw_v, raw_d = predict_reader(m, X3[1]), predict_reader(m, X3[2])
    else:
        if name.startswith('CATI'):
            d = F.CatiDesign(view).fit(trip[0][0], trip[0][1]); X3 = [d.transform(i, o) for i, o in trip]
            info['design_columns'] = int(X3[0].shape[1])
        else:
            X3 = [rich_tab(view, i, o) for i, o in trip]
            info['rich_equals_ffv_features'] = bool(all(np.array_equal(X3[j], F.rich_view(view, *trip[j])) for j in range(3)))
        sc = Scaler().fit(X3[0]); m = make_model(name, seed_model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m.fit(sc(X3[0]), yA)
        assert np.array_equal(m.classes_, [0, 1])
        info['n_iter'] = int(np.max(getattr(m, 'n_iter_', [-1])))
        info['convergence_warning'] = any(type(x.message).__name__ == 'ConvergenceWarning' for x in w)
        raw_v = m.predict_proba(sc(X3[1]))[:, 1].astype(np.float64); raw_d = m.predict_proba(sc(X3[2]))[:, 1].astype(np.float64)
    sign = my_orient(yV, raw_v)
    sv, sd = apply_sign(raw_v, sign), apply_sign(raw_d, sign)
    info.update({'orientation': sign, 'av_auc': mw_auc(yV, sv), 'cpu_s': round(time.process_time() - t0, 3)})
    return sv, sd, info


def run(seed):
    vers = {'numpy': np.__version__, 'sklearn': sklearn.__version__, 'scipy': scipy.__version__}
    assert vers == {'numpy': '2.4.2', 'sklearn': '1.8.0', 'scipy': '1.17.1'}, vers
    plan = json.loads((C.RESULTS / 'READER_PLAN.json').read_text())
    assert plan['versions'] == vers
    A = C.load_role('ATTACK_FIT', ('features', 'sex'), purpose=f'independent verification: final reader refits (IHS verifier), seed {seed}')
    V = C.load_role('ATTACK_VALIDATION', ('features', 'sex'), purpose=f'independent verification: final reader refit orientation (IHS verifier), seed {seed}')
    R = C.load_role('DEV_REPORT', ('features',))                     # label-free: ids only, for the row-order check
    bank, ids, bank_sha = load_bank(seed)
    row_ok = {r: bool(np.array_equal(ids[f'row_id__{r}'], X['row_id']) and np.array_equal(ids[f'unit__{r}'], X['unit']))
              for r, X in (('ATTACK_FIT', A), ('ATTACK_VALIDATION', V), ('DEV_REPORT', R))}
    assert all(row_ok.values()), row_ok
    yA, yV = A['sex'], V['sex']
    d = C.RUNS / f'READERS_S{seed}'
    rec = json.loads((d / 'READERS_SEED.json').read_text())
    assert sha256(d / 'AV_SCORES.npz') == rec['av_scores_sha256'] and sha256(d / 'DEV_REPORT_SCORES.npz') == rec['dev_report_scores_sha256']
    with np.load(d / 'AV_SCORES.npz') as z:
        av = {k: z[k].copy() for k in z.files}
    with np.load(d / 'DEV_REPORT_SCORES.npz') as z:
        dv = {k: z[k].copy() for k in z.files}
    seed_model = MODEL_SEED_BASE + seed
    assert seed_model == rec['reader_model_seed']
    jobs = {}                                       # stored key -> (svc, recipe, view, why)
    for svc in ROSTER:
        for n in SRC8 + NEW8:
            why = ['READER_PLAN verifier roster'] if n in PLAN_RECIPES else ['extension: all pair recipes of the roster services']
            jobs[f'PAIR__{svc}__{n}'] = [svc, n, 'pair', why]
        for bk in ('original', 'enhanced'):
            w = rec['winners'][svc][bk]['pair']; cand = w['candidate']; name = cand.split('__')[-1]
            if cand.startswith('IGNORE_'):
                view = 'income' if cand.startswith('IGNORE_occupation') else 'occupation'
                key = f'MARG__{wire_name(bank, svc, view)}__{name}'
                assert key == w['reader_key'], (key, w['reader_key'])
                jobs.setdefault(key, [svc, name, view, []])[3].append(f'{bk} pair winner ({cand})')
            else:
                assert w['reader_key'] == f'PAIR__{svc}__{name}'
                jobs[f'PAIR__{svc}__{name}'][3].append(f'{bk} pair winner')
    out = {'schema': 'IHS1-verifier-reader-refits-v1', 'seed': seed, 'versions': vers, 'bank_sha256': bank_sha,
           'bank_sha_matches_manifest': True, 'row_order_ok': row_ok, 'reader_model_seed': seed_model, 'readers': {}}
    for key, (svc, n, view, why) in jobs.items():
        trip = [(bank[svc][r]['income'], bank[svc][r]['occupation']) for r in ROLES3]
        sv, sd, info = refit_one(n, view, trip, yA, yV, seed_model)
        if key.startswith('PAIR__'):
            st = rec['pairs'][svc][n]
        else:
            st = rec['marginal_banks'][key.split('__')[1]]['readers'][n]
        r = {'service': svc, 'recipe': n, 'view': view, 'why': why,
             'stored_av_auc': st['attack_validation_auc'], 'refit_av_auc': info['av_auc'],
             'av_auc_diff': info['av_auc'] - st['attack_validation_auc'],
             'orientation_refit': info['orientation'], 'orientation_stored': st.get('orientation'),
             'av_scores_bitwise': bool(np.array_equal(sv, av[key])), 'av_scores_max_abs_diff': float(np.max(np.abs(sv - av[key]))),
             'dev_scores_bitwise': bool(np.array_equal(sd, dv[key])), 'dev_scores_max_abs_diff': float(np.max(np.abs(sd - dv[key]))),
             'cpu_s': info['cpu_s']}
        for k in ('n_iter', 'convergence_warning', 'design_columns', 'rich_equals_ffv_features'):
            if k in info:
                r[k] = info[k]
        if 'n_iter' in st:
            r['stored_n_iter'] = st['n_iter']; r['stored_converged'] = st.get('converged')
        if n.startswith('CATI') and key in rec['designs']:
            r['stored_design_columns'] = rec['designs'][key].get('n_columns')
        out['readers'][key] = r
    rs = out['readers'].values()
    out['summary'] = {'n_refits': len(out['readers']),
                      'av_bitwise': sum(r['av_scores_bitwise'] for r in rs), 'dev_bitwise': sum(r['dev_scores_bitwise'] for r in rs),
                      'max_abs_av_auc_diff': max(abs(r['av_auc_diff']) for r in rs),
                      'max_abs_dev_score_diff': max(r['dev_scores_max_abs_diff'] for r in rs),
                      'orientation_mismatch': [k for k, r in out['readers'].items() if r['orientation_stored'] is not None and r['orientation_refit'] != r['orientation_stored']],
                      'not_bitwise': [k for k, r in out['readers'].items() if not (r['av_scores_bitwise'] and r['dev_scores_bitwise'])]}
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    (OUT / f'READERS_REFIT_S{seed}.json').write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'seed': seed, 'summary': out['summary'],
                      'readers': {k: [r['stored_av_auc'], r['refit_av_auc'], r['av_scores_bitwise'], r['dev_scores_bitwise'],
                                      r['dev_scores_max_abs_diff'], r.get('n_iter'), r.get('stored_n_iter'), r['why']]
                                  for k, r in out['readers'].items()}}, indent=1, default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True)
    run(ap.parse_args().seed)
