"""Grid extras of the PCRL dual-design release campaign: a campaign copy of nrm.readerextra (unchanged code; the copy exists so that
its relative import `from . import devreaders` resolves to ddr.devreaders and its DDR1 splits, not to nrm.devreaders and NRM1's splits).
Source docstring follows. Proposed grid extras of the PCRL new-release-mechanisms campaign (reader-robustness investigator; a PROPOSAL to the coordinator,
written before any candidate outcome; it changes nothing until the coordinator integrates it into nrm.devreaders before the grid
code lock and adds this file to LOCK_FILES). EXPLORATORY DEVELOPMENT code; readers estimate the recorded SEX column from released
outputs only; lower recovery for these readers is never protection against every reader.

One call per (cell, release) inside nrm.devreaders.evaluate_bank, after the 16 locked recipes of that cell are fitted (or read from
the cache), returns three records per view. None of them changes the registered winner, the registered cells or any decision:

1. `score_sha256` {recipe: SHA-256 of its oriented SCORE scores} and `recipe_val_auc` {recipe: oriented VALIDATION AUC}. With them the
   A-SC1 grid-validity check of the frozen rules (RULES.md section 6; nrm.rules.grid_validity: k0 / k1 bitwise equality per
   seed-dependent recipe, view, wire, version and repetition) can be computed from GRID.json alone, without re-reading SEX.
2. `nolookup`: the winner chosen exactly as the registered winner (max oriented VALIDATION AUC, exact ties to the earlier candidate,
   pair candidates = pair readers then the ignore-occupation and ignore-income readers, source recipes first) but with the
   table readers excluded (LOOKUP and, if enabled, TABLE_CV). Its SCORE AUC and household-bootstrap draws (the cell's weights, so
   paired across releases) say whether a benefit survives without frequency-table readers (the challenger's reader-targeting
   requirement: "credit an advantage only if it also holds for smooth readers").
3. `table_cv` and `enhanced`: TABLE_CV, a frequency-table reader with cross-validated smoothing for identifiable finite output cells
   (fitted on FIT, smoothing chosen by household-grouped 5-fold cross-validation INSIDE FIT, oriented on VALIDATION, scored on
   SCORE; deterministic, no seed), and the winner over the 16 locked recipes plus TABLE_CV (TABLE_CV candidates last, so exact ties
   keep the locked recipe). It addresses the sparse-reading bias: the locked LOOKUP reads 16 x 64 = 1,024 joint values with about
   4,000 FIT people (about 4 per value, many unseen), and no locked recipe has one free parameter per released cell for the
   occupation view inside the pair.
     single view: P(S=1 | key) = (n1 + a pi) / (n + a), a from A_GRID by CV log loss (LOOKUP with a CV-chosen pseudo-count);
     pair view:   additive logit on one-hot(income key) + one-hot(occupation key) (keys with >= 2 FIT people; others all-zero),
                  L2 strength C from C_GRID by CV; then the joint table shrunk towards it, P = (n1_joint + a p_add) / (n_joint + a),
                  a from A_GRID (inf = additive only) by CV; unseen joint keys get p_add, unseen marginal keys the intercept.
   Keys are the exact float64 bytes of the released vectors (the LOOKUP key). Continuous releases have almost no repeated keys, so
   TABLE_CV returns about the FIT prior there (it never helps a reader of continuous outputs and costs about one second).

Integration (about eight lines in nrm.devreaders; see roles/reader_robustness/PROPOSAL_GRID_EXTRAS.md for the exact diff):
    ex, eb = RX.cell_extras(rd, inc, occ, sex, unit, rep, frac, W, views)      # after `w = winners(...)`
    entry[view]['extras'] = ex[view]; entry[view]['extras_boot'] = eb[view]    # per view
  strip() drops 'extras_boot'; run_bank writes BOOT_EXTRAS.npz with keys f'{cell}__{name}__{view}__{NOLOOKUP|TABLE_CV|ENHANCED}'.

    python3 -m ddr.readerextra selftest        invented data only (no real data, no label)
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
import time
import warnings

import numpy as np

SCHEMA = 'DDR1-readerextra-v1'
TABLE_READERS = ('LOOKUP', 'TABLE_CV')
A_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, math.inf)
C_GRID = (0.01, 0.1, 1.0, 10.0)
CV_K = 5
CV_TAG = 'DDR1_tablecv'
MIN_KEY_COUNT = 2
EPS = 1e-6


# ---------------------------------------------------------------- keys
def _ids(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    v = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    _, inv = np.unique(v, return_inverse=True)
    return np.asarray(inv, dtype=np.int64).ravel()


def view_ids(view, inc, occ):
    """Exact-byte key ids (equal iff scm_bench.readers.lookup_keys are equal) of a view and, for the pair, of each recipient."""
    if view == 'income':
        return {'key': _ids(inc)}
    if view == 'occupation':
        return {'key': _ids(occ)}
    return {'key': _ids(np.column_stack((np.asarray(inc, np.float64), np.asarray(occ, np.float64)))), 'inc': _ids(inc), 'occ': _ids(occ)}


# ---------------------------------------------------------------- TABLE_CV (deterministic; FIT-internal cross-validation)
def _logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _shrunk(ids_tr, y_tr, ids_te, prior_te, a):
    """(n1 + a prior) / (n + a) on the test keys with counts from the training rows; a = inf -> prior; unseen -> prior."""
    K = int(max(ids_tr.max(initial=-1), ids_te.max(initial=-1))) + 1
    n = np.bincount(ids_tr, minlength=K).astype(np.float64)
    n1 = np.bincount(ids_tr, weights=y_tr.astype(np.float64), minlength=K)
    nt, n1t = n[ids_te], n1[ids_te]
    if math.isinf(a):
        return np.asarray(prior_te, dtype=np.float64).copy()
    return (n1t + a * prior_te) / (nt + a)


def _onehot(ids_list, fit_rows):
    """Sparse one-hot of each recipient's key, columns only for keys with >= MIN_KEY_COUNT people among fit_rows."""
    from scipy import sparse
    blocks = []
    for ids in ids_list:
        cnt = np.bincount(ids[fit_rows], minlength=int(ids.max()) + 1)
        keep = np.flatnonzero(cnt >= MIN_KEY_COUNT)
        col = np.full(int(ids.max()) + 1, -1, dtype=np.int64); col[keep] = np.arange(len(keep))
        c = col[ids]; rows = np.flatnonzero(c >= 0)
        blocks.append(sparse.csr_matrix((np.ones(len(rows)), (rows, c[rows])), shape=(len(ids), max(len(keep), 1))))
    return sparse.hstack(blocks).tocsr()


def _additive(ids_inc, ids_occ, y, train_rows, pred_rows, Cval):
    """Additive logit on recipient-key indicators, fitted on train_rows (indicator columns from train_rows only)."""
    from sklearn.linear_model import LogisticRegression
    X = _onehot([ids_inc, ids_occ], train_rows)
    yt = y[train_rows]
    if np.unique(yt).size < 2:
        return np.full(int(np.sum(pred_rows)) if pred_rows.dtype == bool else len(pred_rows), float(yt.mean()))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = LogisticRegression(C=Cval, solver='lbfgs', max_iter=2000).fit(X[train_rows], yt)
    return m.predict_proba(X[pred_rows])[:, 1].astype(np.float64)


def fit_table_cv(view, inc, occ, sex, unit, fit, folds=None):
    """TABLE_CV raw scores for ALL rows (fitted on FIT rows only) and its chosen smoothing. sex is read on FIT rows only."""
    from ohn.common import household_folds
    y = np.asarray(sex, dtype=np.int64)
    fit = np.asarray(fit, dtype=bool)
    ids = view_ids(view, inc, occ)
    fit_idx = np.flatnonzero(fit)
    fold = household_folds(np.asarray(unit)[fit_idx], CV_K, CV_TAG) if folds is None else np.asarray(folds)
    pi = float(y[fit].mean())
    yf = y[fit_idx]
    info = {'view': view, 'cv_folds': CV_K, 'cv_tag': CV_TAG, 'fit_prior': pi}
    if view in ('income', 'occupation'):
        key = ids['key']
        losses = []
        for a in A_GRID[:-1]:
            pred = np.empty(len(fit_idx))
            for f in range(CV_K):
                tr, te = fit_idx[fold != f], fit_idx[fold == f]
                pr = float(y[tr].mean())
                pred[fold == f] = _shrunk(key[tr], y[tr], key[te], np.full(len(te), pr), a)
            losses.append(_logloss(yf, pred))
        j = int(np.argmin(losses))
        a = A_GRID[j]
        raw = _shrunk(key[fit_idx], yf, key, np.full(len(key), pi), a)
        info.update({'a': a, 'cv_logloss': dict(zip(map(str, A_GRID[:-1]), losses)), 'model': 'shrunk frequency table'})
        return raw, info
    # pair: additive back-off + joint table
    ki, ko, kj = ids['inc'], ids['occ'], ids['key']
    fold_pred = {c: np.empty(len(fit_idx)) for c in C_GRID}
    for f in range(CV_K):
        trm = np.zeros(len(y), bool); trm[fit_idx[fold != f]] = True
        tem = np.zeros(len(y), bool); tem[fit_idx[fold == f]] = True
        for c in C_GRID:
            fold_pred[c][fold == f] = _additive(ki, ko, y, trm, tem, c)
    closs = {c: _logloss(yf, fold_pred[c]) for c in C_GRID}
    cstar = min(C_GRID, key=lambda c: (closs[c], C_GRID.index(c)))
    aloss = []
    for a in A_GRID:
        pred = np.empty(len(fit_idx))
        for f in range(CV_K):
            tr, te = fit_idx[fold != f], fit_idx[fold == f]
            pred[fold == f] = _shrunk(kj[tr], y[tr], kj[te], fold_pred[cstar][fold == f], a)
        aloss.append(_logloss(yf, pred))
    a = A_GRID[int(np.argmin(aloss))]
    allm = np.ones(len(y), bool)
    p_add = _additive(ki, ko, y, fit, allm, cstar)
    raw = _shrunk(kj[fit_idx], yf, kj, p_add, a)
    info.update({'C': cstar, 'a': a if math.isfinite(a) else 'inf', 'cv_logloss_C': {str(c): closs[c] for c in C_GRID},
                 'cv_logloss_a': dict(zip(map(str, A_GRID), aloss)), 'model': 'additive key logit + joint table back-off'})
    return raw, info


# ---------------------------------------------------------------- restricted / enhanced winners (devreaders ordering)
def _cands(rd, view, allowed, extra=None):
    """Candidate list in the registered order (nrm.devreaders.winners), restricted to allowed recipes; `extra` = {view: (val_auc)}
    candidates appended last (exact ties keep the locked recipe)."""
    from .devreaders import SRC8, NEW8
    out = []
    if view in ('income', 'occupation'):
        for group, org in ((SRC8, 'source'), (NEW8, 'new')):
            out += [(n, rd[view][n][1]['val_auc'], f'{view}__{n}', org) for n in group if n in allowed]
        if extra and view in extra:
            out.append(('TABLE_CV', extra[view], f'{view}__TABLE_CV', 'extra'))
        return out
    for group, org in ((SRC8, 'source'), (NEW8, 'new')):
        out += [(n, rd['pair'][n][1]['val_auc'], f'pair__{n}', org) for n in group if n in allowed]
        out += [(f'IGNORE_occupation__{n}', rd['income'][n][1]['val_auc'], f'income__{n}', org) for n in group if n in allowed]
        out += [(f'IGNORE_income__{n}', rd['occupation'][n][1]['val_auc'], f'occupation__{n}', org) for n in group if n in allowed]
    if extra:
        for v, nm in (('pair', 'TABLE_CV'), ('income', 'IGNORE_occupation__TABLE_CV'), ('occupation', 'IGNORE_income__TABLE_CV')):
            if v in extra:
                out.append((nm, extra[v], f'{v}__TABLE_CV', 'extra'))
    return out


def score_sha(s):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(s, dtype=np.float64)).tobytes()).hexdigest()


def cell_extras(rd, inc, occ, sex, unit, rep, frac, W, views=('income', 'occupation', 'pair'), table_cv=True):
    """rd = {view: {recipe: (oriented SCORE scores, info with 'val_auc')}} for the 16 locked recipes of one cell and release (as
    nrm.devreaders.cell_readers returns them); W = the cell's household-bootstrap weights (B x n_SCORE). Returns (record, boots):
    record[view] = {'score_sha256', 'recipe_val_auc', 'nolookup', 'table_cv', 'enhanced'}; boots[view] = {'NOLOOKUP', 'TABLE_CV',
    'ENHANCED'} arrays of B bootstrap AUCs. Reads SEX only as the grid already does (FIT for fitting, VALIDATION for orientation,
    SCORE for the AUC)."""
    from . import devreaders as DR
    from ffv.readers import pick
    from task_conditioned_representation.attacks import orientation, orient
    from task_conditioned_representation.inference import auc
    t0 = time.process_time()
    roles = DR.roles_of(unit, rep)
    fit = DR.lc_fit(unit, roles['FIT'], frac); val = roles['VALIDATION']; sc = roles['SCORE']
    y = np.asarray(sex)[sc]
    rec, boots, tab = {}, {}, {}
    if table_cv:
        for view in views:
            raw, info = fit_table_cv(view, inc, occ, sex, unit, fit)
            sign = orientation(np.asarray(sex)[val], raw[val])
            s = orient(raw, sign)
            tab[view] = (np.ascontiguousarray(s[sc], dtype=np.float64), {**info, 'val_auc': float(auc(np.asarray(sex)[val], s[val])),
                                                                      'orientation': int(sign)})
    locked = list(rd[views[0]].keys())
    allowed = [n for n in locked if n not in TABLE_READERS]
    for view in views:
        if view == 'pair' and not all(v in rd for v in ('income', 'occupation', 'pair')):
            raise ValueError('the pair extras need the income and occupation readers (ignore-one alternatives)')
        r = {'score_sha256': {n: score_sha(rd[view][n][0]) for n in locked},
             'recipe_val_auc': {n: float(rd[view][n][1]['val_auc']) for n in locked}}
        b = {}
        def score_of(key):
            v, name = key.split('__', 1)
            return tab[v][0] if name == 'TABLE_CV' else rd[v][name][0]
        w = pick(_cands(rd, view, allowed))
        s = score_of(w['reader_key'])
        r['nolookup'] = {'winner': w['candidate'], 'val_auc': w['attack_validation_auc'], 'near_tie': w['AV_NEAR_TIE'],
                         'gap_to_runner_up': w['gap_to_runner_up'], 'score_auc': float(auc(y, s))}
        b['NOLOOKUP'] = DR.wauc_draws(y, s, W)
        if table_cv:
            s = tab[view][0]
            r['table_cv'] = {**{k: x for k, x in tab[view][1].items() if not k.startswith('cv_logloss')},
                             'cv_logloss': {k: x for k, x in tab[view][1].items() if k.startswith('cv_logloss')},
                             'score_auc': float(auc(y, s))}
            b['TABLE_CV'] = DR.wauc_draws(y, s, W)
            ext = {v: tab[v][1]['val_auc'] for v in tab}
            w = pick(_cands(rd, view, locked, ext if view == 'pair' else {view: ext[view]}))
            s = score_of(w['reader_key'])
            r['enhanced'] = {'winner': w['candidate'], 'val_auc': w['attack_validation_auc'], 'near_tie': w['AV_NEAR_TIE'],
                             'score_auc': float(auc(y, s))}
            b['ENHANCED'] = DR.wauc_draws(y, s, W)
        rec[view], boots[view] = r, b
    for view in views:
        rec[view]['extras_cpu_s'] = round(time.process_time() - t0, 3)
    return rec, boots


# ---------------------------------------------------------------- invented-data selftest (no real data, no label)
def _invented_finite(n, seed, n_inc=16, n_occ=64, strength=0.35):
    """Finite released cells with an ADDITIVE SEX signal: sparse joint values, informative marginal cells."""
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n // 2), 2)[:n]
    sex = (rng.random(n) < 0.47).astype(np.int64)
    eff_i = rng.normal(scale=strength * 3, size=n_inc); eff_o = rng.normal(scale=strength * 3, size=n_occ)
    # cell membership depends on SEX through additive log-odds tilts
    pi_ = np.exp(np.outer(2 * sex - 1, eff_i) * 0.5); pi_ /= pi_.sum(1, keepdims=True)
    po_ = np.exp(np.outer(2 * sex - 1, eff_o) * 0.5); po_ /= po_.sum(1, keepdims=True)
    ci = np.array([rng.choice(n_inc, p=p) for p in pi_]); co = np.array([rng.choice(n_occ, p=p) for p in po_])
    inc_lv = rng.dirichlet(np.ones(2) * 3, n_inc); occ_lv = rng.dirichlet(np.ones(5) * 2, n_occ)
    return unit, sex, inc_lv[ci], occ_lv[co]


def selftest():
    from . import devreaders as DR
    from scm_bench.readers import lookup_keys, fit_lookup, predict_lookup
    from task_conditioned_representation.attacks import orientation, orient
    from task_conditioned_representation.inference import auc
    chk = {}
    n = 8000
    unit, sex, inc, occ = _invented_finite(n, 7)
    r = DR.roles_of(unit, 'R1')
    fit, val, sc = r['FIT'], r['VALIDATION'], r['SCORE']
    # ids equal the LOOKUP key partition
    keys = lookup_keys(inc, occ); ids = view_ids('pair', inc, occ)['key']
    m = {}
    chk['ids_match_lookup_keys'] = all(m.setdefault(i, k) == k for i, k in zip(ids.tolist(), keys)) and len(m) == len(set(keys))

    def oriented_auc(raw):
        sign = orientation(sex[val], raw[val]); s = orient(raw, sign)
        return float(auc(sex[sc], s[sc]))
    lk = predict_lookup(fit_lookup([keys[i] for i in np.flatnonzero(fit)], sex[fit]), keys)
    t0 = time.process_time()
    raw, info = fit_table_cv('pair', inc, occ, sex, unit, fit)
    chk['table_cv_cpu_s'] = round(time.process_time() - t0, 3)
    a_lk, a_tab = oriented_auc(lk), oriented_auc(raw)
    chk['lookup_pair_auc'], chk['table_cv_pair_auc'] = round(a_lk, 4), round(a_tab, 4)
    chk['table_cv_beats_sparse_lookup'] = a_tab > a_lk + 0.01
    chk['table_cv_chosen'] = {'C': info['C'], 'a': info['a']}
    raw2, _ = fit_table_cv('pair', inc, occ, sex, unit, fit)
    chk['table_cv_deterministic'] = bool(np.array_equal(raw, raw2))
    # FIT-only: changing SEX outside FIT does not change the fitted scores
    sex_b = sex.copy(); sex_b[~fit] = 1 - sex_b[~fit]
    raw3, _ = fit_table_cv('pair', inc, occ, sex_b, unit, fit)
    chk['table_cv_reads_fit_labels_only'] = bool(np.array_equal(raw, raw3))
    # single view = shrunk table; on a continuous release TABLE_CV is about the prior
    raw_o, info_o = fit_table_cv('occupation', inc, occ, sex, unit, fit)
    chk['single_view_shrunk_table'] = info_o['model'] == 'shrunk frequency table' and info_o['a'] in A_GRID
    rng = np.random.default_rng(3)
    ci, co = rng.dirichlet(np.ones(2), n), rng.dirichlet(np.ones(5), n)
    raw_c, info_c = fit_table_cv('pair', ci, co, sex, unit, fit)
    chk['continuous_release_near_prior'] = abs(oriented_auc(raw_c) - 0.5) < 0.03
    # restricted / enhanced winners with fake locked readers (selection logic only; no reader fit)
    from .devreaders import RECIPES, SRC8, NEW8
    ns = int(sc.sum())
    rdf = {}
    for v in ('income', 'occupation', 'pair'):
        rdf[v] = {nm: (rng.random(ns), {'val_auc': 0.55 + 0.001 * j}) for j, nm in enumerate(RECIPES)}
    rdf['pair']['LOOKUP'] = (rng.random(ns), {'val_auc': 0.99})          # LOOKUP would win everything
    W = DR.boot_weights(unit[sc], 0, 'R1')
    ex, eb = cell_extras(rdf, inc, occ, sex, unit, 'R1', 1.0, W)
    chk['nolookup_excludes_table_readers'] = all('LOOKUP' not in ex[v]['nolookup']['winner'] for v in ex)
    chk['nolookup_pair_order_as_registered'] = ex['pair']['nolookup']['winner'] == 'CATI1'          # highest non-table val AUC
    chk['enhanced_contains_lookup_or_table'] = ex['pair']['enhanced']['winner'] in ('LOOKUP', 'TABLE_CV', 'IGNORE_income__TABLE_CV',
                                                                                      'IGNORE_occupation__TABLE_CV')
    chk['fingerprints_16'] = all(len(ex[v]['score_sha256']) == 16 for v in ex)
    chk['boots_shapes'] = all(b.shape == (W.shape[0],) for v in eb for b in eb[v].values())
    # exact tie: TABLE_CV equal to the best locked recipe keeps the locked recipe
    cands = _cands(rdf, 'income', list(RECIPES), {'income': max(x[1]['val_auc'] for x in rdf['income'].values())})
    from ffv.readers import pick
    chk['exact_tie_keeps_locked'] = pick(cands)['candidate'] != 'TABLE_CV'
    chk['all_ok'] = all(v for v in chk.values() if isinstance(v, bool))
    return chk


def main(argv=None):
    a = (argv or sys.argv[1:])
    if a[:1] == ['selftest']:
        chk = selftest(); print(json.dumps(chk, indent=1, default=str)); return 0 if chk['all_ok'] else 1
    print(__doc__); return 2


if __name__ == '__main__':
    sys.exit(main())
