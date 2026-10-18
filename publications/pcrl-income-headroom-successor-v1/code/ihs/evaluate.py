"""Development evaluation on MODEL_SELECT (selection feedback only; every label read is logged by ihs.common.load_role).

For one seed's release on MODEL_SELECT (income (n, 2) and occupation (n, 5) float64 probability vectors; decisions = argmax):

Utility (original caps, unchanged): the 12 point clauses per seed against the frozen U0 / Ucal MODEL_SELECT outputs (accuracy
loss <= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005; scm_bench.selection.clauses), each with a
household-clustered paired standard error (the per-person paired difference against the reference, cluster sums over the
5,306 MODEL_SELECT households, G/(G-1) correction), its margin (cap - value) in metric units and in SE units. The within-SEX
occupation summed-Brier excess against U0 for each SEX group (guard 0.005) with a linearized household-clustered SE.

Reader proxies (PROTOCOL.md section 6; frozen before any new fit): the fixed development grid of two household-split
repetitions x two reader initialization seeds, shared by every method:
- repetition A: the source campaign's household halves of MODEL_SELECT (trees.seed_for('SCM1proxysplit'));
- repetition B: new household halves (SHA-256 order of household ids, tag 'IHS1_dev_proxy_rep_B');
- reader seeds k = 0, 1: random_state = MODEL_SEED_BASE + 100 + k (only LOG_MLP and ET25 depend on it; the seed-invariant
  readers are fitted once per repetition and reported for both seeds);
- each reader is fitted on one half (about 4,000 people, about 2,650 households: smaller than the 8,001-person ATTACK_FIT of
  the final readers) and scored on the other, in both directions; the cell value is the mean AUC of the two directions.
Readers (development slate): OUT_LR, OUT_HGB, LOG_MLP (source constructors on the source 'out' / 'log' features), ET25,
HGB31 (validation recipes on released p, log p, one-hot decision), CATI1 (validation categorical-interaction design, fitted
label-free on the training half), LOOKUP (exact released vectors). Views: income, occupation, pair. The pair view's maximum
always includes the single-view readers (alternatives that ignore occupation or ignore income). Readers see only the released
probability vectors and decisions: never raw inputs, hidden scores, teacher outputs, parameters or labels other than SEX.
Results are cached by (view wire hash, reader, repetition, seed), so identical released views are fitted once.

These are development diagnostics on previously used records, not fresh confirmation and not population-optimal.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import warnings

import numpy as np

from acs_replication import trees
from scm_bench.readers import kinds, lookup_keys, fit_lookup, predict_lookup
from scm_bench.selection import clauses
from task_conditioned_representation.attacks import fit_reader, predict_reader, MODEL_SEED_BASE
from task_conditioned_representation.inference import auc
from ffv import readers as FR
from ffv import features as FF
from . import common as C

DEV_READERS = ('OUT_LR', 'OUT_HGB', 'LOG_MLP', 'ET25', 'HGB31', 'CATI1', 'LOOKUP')
SEED_DEPENDENT = ('LOG_MLP', 'ET25')
REPS = ('A', 'B')
READER_SEEDS = (0, 1)
READER_SEED_OFFSET = 100
VIEWS = ('income', 'occupation', 'pair')
CACHE = C.RUNS / 'proxy_cache'
PROXY_VERSION = 'IHS1-dev-proxy-v1'


# ---------------------------------------------------------------- household-clustered standard errors
def cluster_se(d, unit):
    """SE of the mean of d with household clustering (cluster sums of centred values, G/(G-1) correction)."""
    d = np.asarray(d, dtype=np.float64); n = len(d)
    _, inv = np.unique(unit, return_inverse=True); G = inv.max() + 1
    s = np.bincount(inv, weights=d - d.mean(), minlength=G)
    return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


def subgroup_se(d, mask, unit):
    """Linearized household-clustered SE of the subgroup mean of d over `mask` (a ratio of two sample means)."""
    d = np.asarray(d, dtype=np.float64); mask = np.asarray(mask, dtype=bool); n = len(d)
    m = d[mask].mean(); p = mask.mean()
    z = np.where(mask, (d - m) / p, 0.0)
    _, inv = np.unique(unit, return_inverse=True); G = inv.max() + 1
    s = np.bincount(inv, weights=z, minlength=G)
    return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


def utility(labels, outs, refs, sex, unit, seed):
    """The 12 clauses (scm_bench definitions) plus SEs and margins, and the two guards with SEs."""
    cl = clauses(labels, outs, refs, seed)
    own = {t: C.task_vectors(labels[t], outs[t], C.K[t]) for t in C.TASKS}
    G = len(np.unique(unit))
    for c in cl:
        r = C.task_vectors(labels[c['task']], refs[c['reference']][c['task']], C.K[c['task']])
        o = own[c['task']]
        d = {'accuracy_loss': r['accuracy'] - o['accuracy'], 'nll_excess': o['nll'] - r['nll'], 'brier_excess': o['brier'] - r['brier']}[c['metric']]
        if abs(d.mean() - c['value']) > 1e-12:
            raise AssertionError('clause value and paired vector disagree')
        se = cluster_se(d, unit)
        c.update({'se': se, 'margin': c['cap'] - c['value'], 'margin_se_units': (c['cap'] - c['value']) / se if se > 0 else (np.inf if c['cap'] > c['value'] else -np.inf)})
    u0 = C.task_vectors(labels['occupation'], refs['U0']['occupation'], 5)
    d = own['occupation']['brier'] - u0['brier']
    guards = []
    for g, name in ((0, 'sex0'), (1, 'sex1')):
        mask = sex == g; v = float(d[mask].mean()); se = subgroup_se(d, mask, unit)
        guards.append({'seed': seed, 'group': name, 'value': v, 'cap': C.GUARD, 'se': se, 'margin': C.GUARD - v,
                       'margin_se_units': (C.GUARD - v) / se if se > 0 else np.inf, 'point_pass': bool(v <= C.GUARD)})
    by_sex = {f'sex{g}': {f'{t}_{m}_excess_vs_U0': float((own[t][m] - C.task_vectors(labels[t], refs['U0'][t], C.K[t])[m])[sex == g].mean())
                          for t in C.TASKS for m in ('nll', 'brier')} for g in (0, 1)}
    return {'clauses': cl, 'guards': guards, 'by_sex': by_sex, 'households': G, 'people': int(len(unit)),
            'point_pass_12': all(c['point_pass'] for c in cl), 'guard_pass': all(g['point_pass'] for g in guards),
            'task_summary': {t: {m: float(v.mean()) for m, v in own[t].items()} for t in C.TASKS}}


# ---------------------------------------------------------------- reader proxies on the fixed 2 x 2 development grid
def halves(unit, rep):
    if rep == 'A':
        groups = np.unique(unit)
        perm = np.random.default_rng(trees.seed_for('SCM1proxysplit')).permutation(len(groups))
        a = np.isin(unit, groups[perm[:len(groups) // 2]])
    elif rep == 'B':
        a = C.household_folds(unit, 2, 'IHS1_dev_proxy_rep_B') == 0
    else:
        raise ValueError(rep)
    return a, ~a


def _view_arrays(view, inc, occ):
    return {'income': (inc,), 'occupation': (occ,), 'pair': (inc, occ)}[view]


def view_hash(view, inc, occ):
    h = hashlib.sha256(view.encode())
    for a in _view_arrays(view, inc, occ):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def _src_features(view, inc, occ, kind):
    a, b = kinds(inc, 2), kinds(occ, 5)
    return {'income': a[kind], 'occupation': b[kind], 'pair': np.column_stack((a[kind], b[kind]))}[view]


def _fit_score(name, view, inc, occ, sex, tr, te, seed):
    rs = MODEL_SEED_BASE + READER_SEED_OFFSET + seed
    if name == 'LOOKUP':
        keys = np.array(lookup_keys(*_view_arrays(view, inc, occ)), dtype=object)
        m = fit_lookup(list(keys[tr]), sex[tr]); return predict_lookup(m, list(keys[te]))
    if name in ('OUT_LR', 'OUT_HGB', 'LOG_MLP'):
        X = _src_features(view, inc, occ, 'log' if name.startswith('LOG') else 'out')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = fit_reader(X[tr], sex[tr], name, rs)
        return predict_reader(m, X[te])
    if name == 'CATI1':
        d = FF.CatiDesign(view).fit(inc[tr], occ[tr]); X = d.transform(inc, occ)
    else:
        X = FF.rich_view(view, inc, occ)
    rec, _ = FR.fit_new(name, X[tr], sex[tr], rs)
    return FR.predict_new(rec, X[te])


FINGERPRINT_FILES = ('ffv/readers.py', 'ffv/features.py', 'scm_bench/readers.py', 'task_conditioned_representation/attacks.py',
                     'task_conditioned_representation/inference.py', 'ihs/evaluate.py')
FINGERPRINT_PATH = C.RESULTS / 'DEV_EVALUATOR_FINGERPRINT.json'


def fingerprint():
    """Reader code and library versions of the development grid (reviewer request S2)."""
    import platform
    import scipy
    import sklearn
    return {'files_sha256': {f: C.sha(C.ROOT / f) for f in FINGERPRINT_FILES}, 'python': platform.python_version(),
            'numpy': np.__version__, 'sklearn': sklearn.__version__, 'scipy': scipy.__version__, 'proxy_version': PROXY_VERSION}


def check_fingerprint():
    """Refuse to compute development proxies if the reader code or library versions differ from the registered fingerprint
    (cached AUCs would otherwise be silently stale). Not part of the cache key, so the registered reference cells stay valid."""
    if not FINGERPRINT_PATH.is_file():
        return
    reg = json.loads(FINGERPRINT_PATH.read_text())
    cur = fingerprint()
    diff = [k for k in cur if reg.get(k) != cur[k]]
    if diff:
        raise PermissionError(f'development evaluator fingerprint differs from the registration in {diff}; use the registered '
                              f"interpreter ({reg['python']}, numpy {reg['numpy']}) and unchanged reader code")


def _read_cache(p):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def proxy_cell(view, name, rep, seed, inc, occ, sex, unit):
    """Mean AUC over both directions for one (view, reader, repetition, reader seed); cached by released-view hash."""
    eff_seed = seed if name in SEED_DEPENDENT else 0
    key = hashlib.sha256(f'{PROXY_VERSION}|{view_hash(view, inc, occ)}|{name}|{rep}|{eff_seed}|{C.arr_sha(np.asarray(sex))}|{C.arr_sha(np.asarray(unit))}'.encode()).hexdigest()
    p = CACHE / f'{key}.json'
    hit = _read_cache(p)
    if hit is not None:
        return hit
    a, b = halves(unit, rep); aucs = []
    t0 = time.process_time()
    for tr, te in ((a, b), (b, a)):
        aucs.append(float(auc(sex[te], _fit_score(name, view, inc, occ, sex, tr, te, eff_seed))))
    rec = {'view': view, 'reader': name, 'rep': rep, 'reader_seed': eff_seed, 'auc_dirs': aucs, 'auc': float(np.mean(aucs)),
           'cpu_s': time.process_time() - t0, 'n_fits': 2}
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f'.{key}.{os.getpid()}.tmp'                 # atomic write (reviewer request S1): two queue slots may share a view
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, p)
    return rec


def proxies(outs, sex, unit):
    """Per cell (rep x reader seed): per-view reader AUCs, view max (pair max includes single-view readers)."""
    check_fingerprint()
    inc, occ = outs['income'], outs['occupation']
    cells = {}
    n_fits = 0
    for rep in REPS:
        for seed in READER_SEEDS:
            res = {}
            for view in VIEWS:
                res[view] = {}
                for name in DEV_READERS:
                    r = proxy_cell(view, name, rep, seed, inc, occ, sex, unit)
                    res[view][name] = r['auc']; n_fits += r['n_fits']
                res[view]['view_max'] = max(res[view][n] for n in DEV_READERS)
            res['pair']['max_incl_single'] = max(res['pair']['view_max'], res['income']['view_max'], res['occupation']['view_max'])
            cells[f'{rep}{seed}'] = res
    summ = {}
    for view, key in (('pair', 'max_incl_single'), ('income', 'view_max'), ('occupation', 'view_max')):
        v = [cells[c][view][key] for c in cells]
        summ[view] = {'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v)), 'cells': {c: cells[c][view][key] for c in cells}}
    return {'version': PROXY_VERSION, 'cells': cells, 'summary': summ, 'n_reader_fits_requested': n_fits}


def benefits(cand, ref):
    """Paired per-cell differences ref - cand (positive = candidate reads lower) for the pair; candidate - ref for recipients."""
    out = {}
    for view, key, sign in (('pair', 'max_incl_single', 'ref_minus_cand'), ('income', 'view_max', 'cand_minus_ref'),
                            ('occupation', 'view_max', 'cand_minus_ref')):
        d = {c: (ref['cells'][c][view][key] - cand['cells'][c][view][key]) if sign == 'ref_minus_cand'
             else (cand['cells'][c][view][key] - ref['cells'][c][view][key]) for c in cand['cells']}
        out[view] = {'definition': sign, 'cells': d, 'mean': float(np.mean(list(d.values()))), 'min': float(np.min(list(d.values()))),
                     'max': float(np.max(list(d.values())))}
    return out


def distinct(outs):
    d = {t: int(np.unique(np.ascontiguousarray(outs[t]), axis=0).shape[0]) for t in C.TASKS}
    d['pair'] = int(np.unique(np.column_stack((outs['income'], outs['occupation'])), axis=0).shape[0])
    return d


def check_outs(outs, n):
    for t in C.TASKS:
        p = np.asarray(outs[t], dtype=np.float64)
        if p.shape != (n, C.K[t]) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
            raise ValueError(f'{t}: release violates the probability-vector contract')


def ms_context(seed, purpose):
    """MODEL_SELECT data (labels + SEX, logged) and the frozen U0 / Ucal references, row-aligned."""
    S = C.load_role('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), purpose=purpose)
    bank, ids = C.frozen_bank(seed, roles=('MODEL_SELECT',))
    if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
        raise AssertionError('frozen bank MODEL_SELECT row order differs')
    refs = {r: {t: bank[r]['MODEL_SELECT'][t] for t in C.TASKS} for r in ('U0', 'Ucal')}
    return S, refs, bank


def measure(seed, outs, S, refs, with_proxy=True):
    check_outs(outs, len(S['row_id']))
    labels = {t: S[t] for t in C.TASKS}
    out = {'utility': utility(labels, outs, refs, S['sex'], S['unit'], seed), 'distinct_outputs': distinct(outs),
           'wire_sha256': {t: C.wire_sha(outs[t]) for t in C.TASKS}}
    if with_proxy:
        out['proxy'] = proxies(outs, S['sex'], S['unit'])
    return out
