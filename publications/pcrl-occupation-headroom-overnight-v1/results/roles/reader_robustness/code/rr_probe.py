"""Diagnostic reader probe beyond the frozen development slate (task 4 of READER_ROBUSTNESS_BRIEF.md). NEVER used for selection.

Public-use ACS 2019 1-Year PUMS Florida person records (MODEL_SELECT development role, adaptively used). Readers estimate the
recorded SEX column (sex0 = men, sex1 = women) from released outputs only: each recipient's float64 probability vector and argmax
decision (income view, occupation view) and the two outputs aligned to the same person (pair view). Same access and the same
registered roles as the development grid: FIT / VALIDATION / SCORE of repetitions R1 and R2 (`ohn.devreaders.roles_of`; FIT
fraction via `lc_fit`), reader seeds k0 / k1, release seeds 0 / 1 (8 cells). Labels: MODEL_SELECT SEX through
`ohn.common.load_role(..., purpose='reader robustness: ...')` (logged). Census inputs are dropped right after loading.

Additional designs (fitted on FIT, oriented and selected on VALIDATION, scored on SCORE; never on SCORE):
- MLP_BIG: MLP (256, 128), ReLU, Adam lr 1e-3, alpha 1e-3, batch 128, up to 400 epochs with early stopping on a stratified 15%
  inner part of FIT (patience 25); features = probabilities, clipped log-probabilities, one-hot decision (as the slate's rich view),
  standardized on FIT.
- GB_LONG: histogram gradient boosting, 31 leaves, learning rate 0.02, up to 3000 iterations with early stopping on a 15% inner
  part of FIT (patience 100, log loss), min 20 per leaf, l2 1.0; same features (a longer-trained, slower-learning booster than
  the slate's 200 x 0.05).
- KNN_LOGP: k-nearest neighbours (uniform, Euclidean) on standardized log max(p, 1e-6) of the view's vector(s); k chosen from
  (15, 50, 150, 400) by oriented VALIDATION AUC (ties to the smaller k).
Pair candidates = pair designs + IGNORE_occupation (income-view designs) + IGNORE_income (occupation-view designs), as in the grid.

The frozen slate's 16 recipes are NOT refitted: their oriented SCORE scores and VALIDATION AUCs are read (read-only) from the grid's
reader cache (key re-derived from DEV_GRID.json); a missing entry is recorded, never fitted. This gives, per cell: the frozen
winner (re-derived; checked against the grid), the probe winner, the UNION winner (slate + probe, max VALIDATION AUC), every
fixed recipe (candidate and comparator read by the SAME recipe: removes the recipe-family confound) and the SCORE-max oracle
(optimistic, not a reader). Paired benefits use the grid's household bootstrap (re-implemented with the same constants: SCORE
households, B = 400, seed 2026101806 + 100 s + rep, readers fixed). Lower recovery for these readers is not protection against
every possible reader.

    python rr_probe.py selftest
    python rr_probe.py run --spec <probe spec.json> --seed <s>
    python rr_probe.py summarize --probe <PROBE_ID> [--probe ...] --tag <TAG>
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from ohn import common as C                                      # noqa: E402
from ohn.devreaders import roles_of, lc_fit, load_member          # noqa: E402  (registered role masks and loading only)

PROBE_VERSION = 'OHN1-rr-probe-v1'
DESIGNS = ('MLP_BIG', 'GB_LONG', 'KNN_LOGP')
KNN_K = (15, 50, 150, 400)
KNN_CLIP = 1e-6
VIEWS = ('income', 'occupation', 'pair')
REPS = ('R1', 'R2')
READER_SEEDS = (0, 1)
NEAR_TIE = 0.010
GRID = json.loads((C.RESULTS / 'DEV_GRID.json').read_text())
GRID_VERSION = GRID['schema']
B_BOOT = int(GRID['bootstrap']['B'])
BOOT_SEED = int(GRID['bootstrap']['seed'])
SRC8 = tuple(GRID['recipes']['source8'])
NEW8 = tuple(GRID['recipes']['new8'])
SLATE16 = SRC8 + NEW8
PROBE_SEED_OFFSET = 400
RUN_ROOT = C.RUNS / 'RR_PROBE'
PROBE_CACHE = RUN_ROOT / 'cache'
SLATE_CACHE = C.RUNS / 'devgrid_cache'
OUT_RES = C.RESULTS / 'roles' / 'reader_robustness'
STATS = {'probe_fits': 0, 'probe_cache_hits': 0, 'slate_cache_hits': 0, 'slate_cache_missing': 0, 'probe_cpu_s': 0.0}
SUMMARY_KEYS = ('slate_score_auc', 'probe_score_auc', 'union_score_auc', 'oracle_slate_score_max', 'oracle_union_score_max')


# ---------------------------------------------------------------- primitives
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind='stable'); ss = s[order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    pos = np.add.reduceat((y[order] == 1).astype(np.float64), starts); neg = np.add.reduceat((y[order] == 0).astype(np.float64), starts)
    return float((pos * (np.cumsum(neg) - 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def boot_weights(unit_sc, rel_seed, rep):
    """The grid's household bootstrap of SCORE (same constants and draw order as the registered grid)."""
    g, inv = np.unique(unit_sc, return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED + 100 * int(rel_seed) + (0 if rep == 'R1' else 1))
    W = np.empty((B_BOOT, len(unit_sc)))
    for b in range(B_BOOT):
        W[b] = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))[inv]
    return W


def wauc_draws(y, s, W):
    u, inv = np.unique(s, return_inverse=True)
    pos = (y == 1).astype(np.float64); neg = 1.0 - pos
    out = np.empty(W.shape[0])
    for b in range(W.shape[0]):
        wp = np.bincount(inv, weights=W[b] * pos, minlength=len(u)); wn = np.bincount(inv, weights=W[b] * neg, minlength=len(u))
        out[b] = np.sum(wp * (np.cumsum(wn) - 0.5 * wn)) / (wp.sum() * wn.sum())
    return out


def _log(p, clip=1e-300):
    return np.log(np.maximum(np.asarray(p, dtype=np.float64), clip))


def rich(p):
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack((p, _log(p), np.eye(p.shape[1])[p.argmax(1)]))


def view_features(design, view, inc, occ):
    f = (lambda p: _log(p, KNN_CLIP)) if design == 'KNN_LOGP' else rich
    return f(inc) if view == 'income' else f(occ) if view == 'occupation' else np.column_stack((f(inc), f(occ)))


def wire_hash(view, inc, occ):
    """Identical to the grid's wire hash (view name, then the float64 arrays the view reads)."""
    h = hashlib.sha256(view.encode())
    for a in ((inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def grid_reader_seed(rel_seed, k):
    return int(GRID['reader_seeds'][f'S{rel_seed}k{k}'])


# ---------------------------------------------------------------- probe designs
def fit_design(design, view, inc, occ, y, fit, val, sc, seed):
    X = view_features(design, view, inc, occ)
    mu = X[fit].mean(0); sd = X[fit].std(0); sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    t0 = time.process_time()
    info = {}
    if design == 'MLP_BIG':
        m = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', alpha=1e-3, batch_size=128,
                          learning_rate_init=1e-3, max_iter=400, early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
                          random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.fit(Z[fit], y[fit])
        rv, rs = m.predict_proba(Z[val])[:, 1], m.predict_proba(Z[sc])[:, 1]
        info = {'n_iter': int(m.n_iter_), 'inner_best_validation_accuracy': float(m.best_validation_score_)}
    elif design == 'GB_LONG':
        m = HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.02, max_iter=3000, min_samples_leaf=20,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                           n_iter_no_change=100, scoring='loss', random_state=seed)
        m.fit(Z[fit], y[fit])
        rv, rs = m.predict_proba(Z[val])[:, 1], m.predict_proba(Z[sc])[:, 1]
        info = {'n_iter': int(m.n_iter_)}
    elif design == 'KNN_LOGP':
        best, by_k = None, {}
        nfit = int(fit.sum())
        for k in KNN_K:
            kk = min(k, nfit - 1)
            m = KNeighborsClassifier(n_neighbors=kk, weights='uniform', algorithm='auto').fit(Z[fit], y[fit])
            v = m.predict_proba(Z[val])[:, 1]
            a = auc(y[val], v); oa = max(a, 1 - a); by_k[str(k)] = oa
            if best is None or oa > best[0] + 1e-15:
                best = (oa, kk, m, v)
        _, kk, m, rv = best
        rs = m.predict_proba(Z[sc])[:, 1]
        info = {'k': int(kk), 'oriented_val_auc_by_k': by_k}
    else:
        raise ValueError(design)
    sign = 1 if auc(y[val], rv) >= 0.5 else -1
    ov, os_ = (rv, rs) if sign == 1 else (1 - rv, 1 - rs)
    info.update({'orientation': sign, 'val_auc': auc(y[val], ov), 'cpu_s': round(time.process_time() - t0, 3)})
    return ov.astype(np.float64), os_.astype(np.float64), info


def probe_cell(view, inc, occ, y, unit, rel_seed, rep, k, frac, cache=None):
    """The three probe designs for one view of one wire in one cell (own cache by wire). Returns {design: (SCORE scores, info)}."""
    cache = cache or PROBE_CACHE
    r = roles_of(unit, rep)
    fit = lc_fit(unit, r['FIT'], frac); val = r['VALIDATION']; sc = r['SCORE']
    seed = grid_reader_seed(rel_seed, k) + PROBE_SEED_OFFSET
    key = hashlib.sha256(f'{PROBE_VERSION}|{",".join(DESIGNS)}|{rep}|{seed}|{frac}|{wire_hash(view, inc, occ)}|{C.arr_sha(np.asarray(y))}'.encode()).hexdigest()
    p = Path(cache) / f'{key}.npz'
    if p.is_file():
        with np.load(p, allow_pickle=False) as z:
            info = json.loads(str(z['info']))
            STATS['probe_cache_hits'] += 1
            return {d: (z[f'score__{d}'].copy(), info[d]) for d in DESIGNS}
    out, info = {}, {}
    t0 = time.process_time()
    for d in DESIGNS:
        _, s, inf = fit_design(d, view, inc, occ, y, fit, val, sc, seed)
        out[d] = (s, inf); info[d] = inf
        STATS['probe_fits'] += 1
    Path(cache).mkdir(parents=True, exist_ok=True)
    tmp = Path(cache) / f'.{key}.{os.getpid()}.npz'
    with open(tmp, 'wb') as f:
        np.savez(f, info=json.dumps(info), **{f'score__{d}': out[d][0] for d in DESIGNS})
    os.replace(tmp, p)
    STATS['probe_cpu_s'] += time.process_time() - t0
    return out


def slate_cell(view, inc, occ, y, rel_seed, rep, k, frac, cache=None):
    """Read-only access to the grid's cached slate readers of this wire and cell; None if absent (never fitted here)."""
    cache = cache or SLATE_CACHE
    rs = grid_reader_seed(rel_seed, k)
    key = hashlib.sha256(f'{GRID_VERSION}|{rep}|{rs}|{frac}|{wire_hash(view, inc, occ)}|{C.arr_sha(np.asarray(y))}'.encode()).hexdigest()
    p = Path(cache) / f'{key}.npz'
    if not p.is_file():
        STATS['slate_cache_missing'] += 1
        return None
    with np.load(p, allow_pickle=False) as z:
        info = json.loads(str(z['info']))
        STATS['slate_cache_hits'] += 1
        return {n: (z[f'score__{n}'].copy(), info[n]) for n in SLATE16}


def pick(cands):
    """[(name, val_auc, scores)]: max VALIDATION AUC, exact ties to the earlier candidate; near tie < 0.010 flagged."""
    win = max(range(len(cands)), key=lambda i: (cands[i][1], -i))
    others = sorted((c[1] for i, c in enumerate(cands) if i != win), reverse=True)
    gap = float(cands[win][1] - others[0]) if others else None
    return {'name': cands[win][0], 'val_auc': float(cands[win][1]), 'scores': cands[win][2], 'gap': gap,
            'near_tie': bool(gap is not None and gap < NEAR_TIE)}


def slate_candidates(view, sl):
    if view != 'pair':
        return [(n, sl[view][n][1]['val_auc'], sl[view][n][0]) for n in SLATE16]
    c = []
    for group in (SRC8, NEW8):
        c += [(n, sl['pair'][n][1]['val_auc'], sl['pair'][n][0]) for n in group]
        c += [(f'IGNORE_occupation__{n}', sl['income'][n][1]['val_auc'], sl['income'][n][0]) for n in group]
        c += [(f'IGNORE_income__{n}', sl['occupation'][n][1]['val_auc'], sl['occupation'][n][0]) for n in group]
    return c


def probe_candidates(view, pr):
    if view != 'pair':
        return [(f'P:{d}', pr[view][d][1]['val_auc'], pr[view][d][0]) for d in DESIGNS]
    return ([(f'P:{d}', pr['pair'][d][1]['val_auc'], pr['pair'][d][0]) for d in DESIGNS]
            + [(f'P:IGNORE_occupation__{d}', pr['income'][d][1]['val_auc'], pr['income'][d][0]) for d in DESIGNS]
            + [(f'P:IGNORE_income__{d}', pr['occupation'][d][1]['val_auc'], pr['occupation'][d][0]) for d in DESIGNS])


# ---------------------------------------------------------------- one release seed
def evaluate(bank, rel_seed, sex, unit, frac=1.0, probe_cache=None, slate_cache=None, probe_members=None):
    """bank: {name: {'income', 'occupation'}} (MODEL_SELECT rows). probe_members: names that get the new designs (None = all);
    the others are read with the cached frozen slate only (fixed-recipe comparisons at no fit cost). Returns per cell and
    release: views (AUC records) and schemes (oriented SCORE scores per reading scheme)."""
    res = {}
    for rep in REPS:
        sc = roles_of(unit, rep)['SCORE']; y = sex[sc]
        W = boot_weights(unit[sc], rel_seed, rep)
        for k in READER_SEEDS:
            cell = f'S{rel_seed}{rep}k{k}'
            cres = {}
            for name, rel in bank.items():
                inc, occ = rel['income'], rel['occupation']
                do_probe = probe_members is None or name in probe_members
                pr = {v: probe_cell(v, inc, occ, sex, unit, rel_seed, rep, k, frac, probe_cache) for v in VIEWS} if do_probe else None
                sl = {v: slate_cell(v, inc, occ, sex, rel_seed, rep, k, frac, slate_cache) for v in VIEWS}
                have_slate = all(sl[v] is not None for v in VIEWS)
                ent = {'views': {}, 'schemes': {}, 'slate_cached': have_slate, 'probed': do_probe}
                for v in VIEWS:
                    e = {}
                    if do_probe:
                        pw = pick(probe_candidates(v, pr))
                        e.update({'probe_winner': pw['name'], 'probe_val_auc': pw['val_auc'], 'probe_near_tie': pw['near_tie'],
                                  'probe_score_auc': auc(y, pw['scores']),
                                  'designs': {d: {'val_auc': pr[v][d][1]['val_auc'], 'score_auc': auc(y, pr[v][d][0]),
                                                  **{kk: vv for kk, vv in pr[v][d][1].items() if kk not in ('val_auc',)}} for d in DESIGNS}})
                        ent['schemes'][f'{v}|PROBE_WINNER'] = pw['scores']
                        for d in DESIGNS:
                            ent['schemes'][f'{v}|PROBE:{d}'] = pr[v][d][0]
                    if have_slate:
                        slc = slate_candidates(v, sl)
                        sw = pick(slc)
                        sc_sl = [auc(y, c[2]) for c in slc]
                        rec_auc = {n: auc(y, sl[v][n][0]) for n in SLATE16}
                        e.update({'slate_winner': sw['name'], 'slate_val_auc': sw['val_auc'], 'slate_score_auc': auc(y, sw['scores']),
                                  'slate_near_tie': sw['near_tie'], 'slate_gap': sw['gap'],
                                  'oracle_slate_score_max': float(max(sc_sl)), 'oracle_slate_candidate': slc[int(np.argmax(sc_sl))][0],
                                  'slate_recipe_score_auc': rec_auc,
                                  'slate_recipe_val_auc': {n: sl[v][n][1]['val_auc'] for n in SLATE16},
                                  'slate_score_auc_sd_across_16': float(np.std(list(rec_auc.values()), ddof=1)),
                                  'slate_val_auc_sd_across_16': float(np.std([sl[v][n][1]['val_auc'] for n in SLATE16], ddof=1))})
                        ent['schemes'][f'{v}|SLATE_WINNER'] = sw['scores']
                        for n in SLATE16:
                            ent['schemes'][f'{v}|SLATE:{n}'] = sl[v][n][0]
                        if do_probe:
                            allc = slc + probe_candidates(v, pr)
                            uw = pick(allc)
                            sc_all = [auc(y, c[2]) for c in allc]
                            e.update({'union_winner': uw['name'], 'union_val_auc': uw['val_auc'], 'union_score_auc': auc(y, uw['scores']),
                                      'union_near_tie': uw['near_tie'], 'oracle_union_score_max': float(max(sc_all)),
                                      'oracle_union_candidate': allc[int(np.argmax(sc_all))][0]})
                            ent['schemes'][f'{v}|UNION_WINNER'] = uw['scores']
                    ent['views'][v] = e
                cres[name] = ent
            res[cell] = {'releases': cres, 'W': W, 'y': y}
    return res


_DRAWS = {}


def _draws(res, cell, name, key):
    k = (id(res), cell, name, key)
    if k not in _DRAWS:
        c = res[cell]
        _DRAWS[k] = wauc_draws(c['y'], c['releases'][name]['schemes'][key], c['W'])
    return _DRAWS[k]


def paired(res, cand, comp):
    """Per cell and scheme: pair benefit (comparator - candidate SCORE AUC, positive favours the candidate) with bootstrap SE;
    recipient differences (candidate - comparator) for the single views."""
    out = {}
    for cell, c in res.items():
        a, b = c['releases'][cand], c['releases'][comp]
        for key in sorted(set(a['schemes']) & set(b['schemes'])):
            v = key.split('|')[0]
            da = _draws(res, cell, cand, key); db = _draws(res, cell, comp, key)
            pa, pb = auc(c['y'], a['schemes'][key]), auc(c['y'], b['schemes'][key])
            d, dd = (pb - pa, db - da) if v == 'pair' else (pa - pb, da - db)
            out.setdefault(key, {'cells': {}, 'cell_se': {}})
            out[key]['cells'][cell] = float(d); out[key]['cell_se'][cell] = float(np.std(dd, ddof=1))
        for ok in ('oracle_slate_score_max', 'oracle_union_score_max'):
            if a['views']['pair'].get(ok) is not None and b['views']['pair'].get(ok) is not None:
                key = 'pair|' + ok.upper().replace('_SCORE_MAX', '')
                out.setdefault(key, {'cells': {}, 'cell_se': {}})
                out[key]['cells'][cell] = float(b['views']['pair'][ok] - a['views']['pair'][ok])
    return out


def similarity(bank):
    """Label-free similarity of the released outputs between members (MODEL_SELECT rows): income P(class 1) Pearson / Spearman
    correlation and decision agreement; occupation mean total-variation distance and decision agreement; distinct vectors."""
    from scipy.stats import spearmanr
    names = sorted(bank)
    out = {'distinct_vectors': {n: {t: int(len(np.unique(np.ascontiguousarray(bank[n][t]), axis=0))) for t in ('income', 'occupation')} for n in names},
           'pairs': {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ia, ib = bank[a]['income'][:, 1], bank[b]['income'][:, 1]
            oa, ob = bank[a]['occupation'], bank[b]['occupation']
            same_inc = bool(np.array_equal(bank[a]['income'], bank[b]['income']))
            same_occ = bool(np.array_equal(oa, ob))
            out['pairs'][f'{a}|{b}'] = {
                'income_identical': same_inc, 'occupation_identical': same_occ,
                'income_pearson': None if same_inc else float(np.corrcoef(ia, ib)[0, 1]),
                'income_spearman': None if same_inc else float(spearmanr(ia, ib).statistic),
                'income_decision_agreement': float(np.mean(bank[a]['income'].argmax(1) == bank[b]['income'].argmax(1))),
                'income_mean_abs_diff': float(np.mean(np.abs(ia - ib))),
                'occupation_mean_tv': float(np.mean(0.5 * np.abs(oa - ob).sum(1))),
                'occupation_decision_agreement': float(np.mean(oa.argmax(1) == ob.argmax(1)))}
    return out


def grid_consistency(cells, seed):
    """The re-derived frozen winners and SCORE AUCs must equal the registered grid's (any bank holding the release)."""
    grids = {}
    for p in sorted((C.RUNS / 'GRID').glob(f'*/S{seed}/GRID.json')):
        g = json.loads(p.read_text())
        for cell, cres in g['cells'].items():
            if '@' in cell:
                continue
            for name, ent in cres.items():
                grids.setdefault((cell, name), (p.parent.parent.name, ent))
    checked, mismatches = 0, []
    for cell, cres in cells.items():
        for name, e in cres.items():
            if not e['slate_cached'] or (cell, name) not in grids:
                continue
            bank, ent = grids[(cell, name)]
            for v in VIEWS:
                checked += 1
                mine = e['views'][v]
                if mine['slate_winner'] != ent[v]['winner'] or abs(mine['slate_score_auc'] - ent[v]['score_auc']) > 1e-12:
                    mismatches.append({'cell': cell, 'release': name, 'view': v, 'bank': bank, 'grid': [ent[v]['winner'], ent[v]['score_auc']],
                                       'rederived': [mine['slate_winner'], mine['slate_score_auc']]})
    return {'checked': checked, 'mismatches': mismatches, 'ok': checked > 0 and not mismatches}


def run(spec_path, seed):
    spec = json.loads(Path(spec_path).read_text())
    out = RUN_ROOT / spec['probe_id'] / f'S{seed}'
    if out.exists():
        raise PermissionError('fresh output directory required')
    S = C.load_role('MODEL_SELECT', ('features', 'sex'),
                    purpose=f"reader robustness: diagnostic reader probe beyond the frozen slate ({spec['probe_id']}, seed {seed}); never selection")
    sex, unit = S['sex'], S['unit']
    del S['X'], S['feature_names']                                   # readers never see Census inputs
    bank = {name: load_member(m, seed) for name, m in spec['members'].items()}
    for name, rel in bank.items():
        if rel['income'].shape != (len(sex), 2) or rel['occupation'].shape != (len(sex), 5):
            raise ValueError(f'{name}: release shape differs from MODEL_SELECT')
    t0 = time.process_time()
    pm = spec.get('probe_members')
    res = evaluate(bank, seed, sex, unit, float(spec.get('fraction', 1.0)), probe_members=set(pm) if pm is not None else None)
    pairs = {f'{cand}|{comp}': paired(res, cand, comp) for cand in spec['candidates'] for comp in spec['comparators'] if cand != comp}
    out.mkdir(parents=True, mode=0o700)
    cells = {cell: {name: {'views': e['views'], 'slate_cached': e['slate_cached']} for name, e in c['releases'].items()} for cell, c in res.items()}
    rec = {'schema': 'OHN1-rr-probe-result-v1', 'probe_id': spec['probe_id'], 'seed': seed, 'spec': spec, 'probe_version': PROBE_VERSION,
           'designs': DESIGNS, 'cells': cells, 'paired': pairs, 'stats': dict(STATS), 'cpu_s': round(time.process_time() - t0, 1),
           'grid_consistency': grid_consistency(cells, seed), 'label_free_similarity': similarity(bank),
           'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV),
           'note': 'diagnostic reader probe (MODEL_SELECT development role); never used for selection; reader-relative'}
    C.write_json(out / 'PROBE.json', rec)
    try:
        from ohn import devrun as D
        D.unit_stats(reader_fits=STATS['probe_fits'], cache_hits=STATS['probe_cache_hits'])
    except Exception:                                                # noqa: BLE001  (counters are bookkeeping only)
        pass
    print(json.dumps({'probe_id': spec['probe_id'], 'seed': seed, 'stats': STATS, 'cpu_s': rec['cpu_s']}))
    return rec


# ---------------------------------------------------------------- fixed-recipe learning curves from the cached slate (no fit)
LC_FRACS = (0.25, 0.5, 1.0)


def lc_fixed(spec_path, seed):
    """Learning curves per FIXED slate recipe: the grid's cached readers at FIT fractions 1/4, 1/2 and full (bank specs with
    'fractions' [1.0, 0.5, 0.25] already ran through the locked grid), read-only; candidate and comparator read by the same recipe
    in the same cell. Separates 'larger readers recover more' from VALIDATION-winner selection noise. Never selection."""
    spec = json.loads(Path(spec_path).read_text())
    out_dir = RUN_ROOT / f"LCFIX_{spec['bank_id']}" / f'S{seed}'
    if out_dir.exists():
        raise PermissionError('fresh output directory required')
    S = C.load_role('MODEL_SELECT', ('features', 'sex'),
                    purpose=f"reader robustness: fixed-recipe learning curves from the cached grid readers ({spec['bank_id']}, seed {seed}); never selection")
    sex, unit = S['sex'], S['unit']
    del S['X'], S['feature_names']
    bank = {name: load_member(m, seed) for name, m in spec['members'].items()}
    aucs, missing = {}, []
    for rep in REPS:
        sc = roles_of(unit, rep)['SCORE']; y = sex[sc]
        for k in READER_SEEDS:
            for frac in LC_FRACS:
                cell = f'S{seed}{rep}k{k}@{frac}'
                for name, rel in bank.items():
                    sl = {v: slate_cell(v, rel['income'], rel['occupation'], sex, seed, rep, k, frac) for v in VIEWS}
                    if any(x is None for x in sl.values()):
                        missing.append([cell, name]); continue
                    ent = {v: {n: auc(y, sl[v][n][0]) for n in SLATE16} for v in VIEWS}
                    for v in VIEWS:
                        w = pick(slate_candidates(v, sl))
                        ent[v]['WINNER'] = auc(y, w['scores'])
                    aucs.setdefault(cell, {})[name] = ent
    pairs = {}
    for cand in spec['candidates']:
        for comp in spec['comparators']:
            if cand == comp:
                continue
            rec = {}
            for key in SLATE16 + ('WINNER',):
                by = {}
                for cell, cres in aucs.items():
                    if cand in cres and comp in cres:
                        by.setdefault(cell.split('@')[1], []).append(cres[comp]['pair'][key] - cres[cand]['pair'][key])
                if set(by) == {str(f) for f in LC_FRACS}:
                    rec[key] = {fr: float(np.mean(v)) for fr, v in by.items()}
            pairs[f'{cand}|{comp}'] = rec
    out_dir.mkdir(parents=True, mode=0o700)
    res = {'schema': 'OHN1-rr-lcfix-v1', 'bank_id': spec['bank_id'], 'seed': seed, 'aucs': aucs, 'pair_benefit_by_fraction': pairs,
           'missing': missing, 'stats': dict(STATS), 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV),
           'note': 'cached frozen-slate readers only (no fit); diagnostic; never selection'}
    C.write_json(out_dir / 'LCFIX.json', res)
    print(json.dumps({'bank_id': spec['bank_id'], 'seed': seed, 'missing': len(missing), 'stats': STATS}))
    return res


def lc_fixed_summary(bank_ids, tag):
    """Merge LCFIX seeds; per pair and recipe: mean benefit by fraction and the frozen learning-curve flag (descriptive)."""
    from ohn import rules as RU
    rules = RU.load_rules()
    merged = {}
    for b in bank_ids:
        for s in (0, 1):
            p = RUN_ROOT / f'LCFIX_{b}' / f'S{s}' / 'LCFIX.json'
            if not p.is_file():
                continue
            g = json.loads(p.read_text())
            for cell, cres in g['aucs'].items():
                for name, ent in cres.items():
                    merged.setdefault(name, {})[cell] = ent
    out = {}
    pairs = set()
    for b in bank_ids:
        for s in (0, 1):
            p = RUN_ROOT / f'LCFIX_{b}' / f'S{s}' / 'LCFIX.json'
            if p.is_file():
                pairs |= set(json.loads(p.read_text())['pair_benefit_by_fraction'])
    for pk in sorted(pairs):
        cand, comp = pk.split('|')
        cells = sorted(set(merged.get(cand, {})) & set(merged.get(comp, {})))
        rec = {}
        for key in SLATE16 + ('WINNER',):
            by = {}
            for cell in cells:
                by.setdefault(cell.split('@')[1], []).append(merged[comp][cell]['pair'][key] - merged[cand][cell]['pair'][key])
            if set(by) == {str(f) for f in LC_FRACS}:
                curve = {fr: float(np.mean(v)) for fr, v in by.items()}
                flag = RU.learning_curve_flag({float(k): v for k, v in curve.items()}, rules)['flag']
                rec[key] = {'curve': curve, 'n_cells_per_fraction': {fr: len(v) for fr, v in by.items()}, 'flag': flag,
                            'full_minus_quarter': curve['1.0'] - curve['0.25']}
        inf = {k: v for k, v in rec.items() if k not in ('LOOKUP', 'WINNER')}
        out[pk] = {'by_recipe': rec,
                   'n_recipes_flagged_shrink': sum(v['flag'] == 'BENEFIT_SHRINKS_WITH_READER_DATA' for v in inf.values()),
                   'n_recipes_informative': len(inf),
                   'median_full_minus_quarter': float(np.median([v['full_minus_quarter'] for v in inf.values()])) if inf else None,
                   'winner_flag': rec.get('WINNER', {}).get('flag')}
    # per release and recipe: mean pair AUC by fraction
    rel = {}
    for name, cs in merged.items():
        by = {}
        for cell, ent in cs.items():
            fr = cell.split('@')[1]
            for key in SLATE16 + ('WINNER',):
                by.setdefault(key, {}).setdefault(fr, []).append(ent['pair'][key])
        rel[name] = {key: {fr: float(np.mean(v)) for fr, v in d.items()} for key, d in by.items()}
    C.write_json(OUT_RES / f'{tag}__LC_FIXED_RECIPE.json', {'banks': bank_ids, 'utc': C.now(), 'pairs': out, 'release_pair_auc_by_fraction': rel,
                                                            'note': 'cached frozen-slate readers (no fit); LOOKUP excluded from counts (non-informative on continuous pair vectors); diagnostic only'})
    return out


# ---------------------------------------------------------------- selftest (invented data only; no real data, no label)
def selftest():
    rng = np.random.default_rng(777)
    n = 4000
    unit = np.sort(rng.integers(0, 2650, n))
    sex = (rng.random(n) < 0.47).astype(np.int64)
    z = rng.normal(size=(n, 5)); z[:, 1] += 0.7 * sex; z[:, 3] -= 0.5 * sex
    leaky = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    z2 = z.copy(); z2[:, 1] -= 0.6 * sex; z2[:, 3] += 0.45 * sex
    safe = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
    t = 1 / (1 + np.exp(-(rng.normal(size=n) + 0.15 * sex))); inc = np.column_stack((1 - t, t))
    bank = {'CAND': {'income': inc, 'occupation': safe}, 'COMP': {'income': inc, 'occupation': leaky}}
    tmp = Path(tempfile.mkdtemp(prefix='rr_probe_selftest_'))
    res = evaluate(bank, 0, sex, unit, 1.0, probe_cache=tmp / 'probe', slate_cache=tmp / 'slate')
    fits1 = STATS['probe_fits']
    p = paired(res, 'CAND', 'COMP')
    res2 = evaluate(bank, 0, sex, unit, 1.0, probe_cache=tmp / 'probe', slate_cache=tmp / 'slate')
    W = res['S0R1k0']['W']; y = res['S0R1k0']['y']
    s = res['S0R1k0']['releases']['COMP']['schemes']['pair|PROBE_WINNER']
    chk = {
        'cells': sorted(res),
        'probe_fits_first_call': fits1, 'second_call_all_cached': STATS['probe_fits'] == fits1,
        'slate_absent_recorded': STATS['slate_cache_missing'] > 0 and not res['S0R1k0']['releases']['CAND']['slate_cached'],
        'pair_benefit_positive_every_cell_every_design': all(v > 0 for key in ('pair|PROBE:MLP_BIG', 'pair|PROBE:GB_LONG', 'pair|PROBE:KNN_LOGP', 'pair|PROBE_WINNER')
                                                             for v in p[key]['cells'].values()),
        'income_view_identical_wire_zero_difference': all(abs(v) < 1e-12 for v in p['income|PROBE_WINNER']['cells'].values()),
        'bootstrap_se_positive': all(v > 0 for v in p['pair|PROBE_WINNER']['cell_se'].values()),
        'unweighted_draw_equals_auc': abs(wauc_draws(y, s, np.ones((1, len(y))))[0] - auc(y, s)) < 1e-12,
        'scores_reproduce': all(np.array_equal(res[c]['releases'][m]['schemes'][k], res2[c]['releases'][m]['schemes'][k])
                                for c in res for m in bank for k in res[c]['releases'][m]['schemes']),
        'oriented_val_auc_ge_half': all(res[c]['releases'][m]['views'][v]['designs'][d]['val_auc'] >= 0.5
                                        for c in res for m in bank for v in VIEWS for d in DESIGNS),
        'pair_benefit_mlp_big': {c: round(v, 4) for c, v in p['pair|PROBE:MLP_BIG']['cells'].items()},
        'cpu_s_per_fit': round(STATS['probe_cpu_s'] / max(1, fits1), 3),
    }
    for k in ('second_call_all_cached', 'slate_absent_recorded', 'pair_benefit_positive_every_cell_every_design',
              'income_view_identical_wire_zero_difference', 'bootstrap_se_positive', 'unweighted_draw_equals_auc',
              'scores_reproduce', 'oriented_val_auc_ge_half'):
        chk[k] = bool(chk[k])
    chk['all_ok'] = all(chk[k] for k in ('second_call_all_cached', 'slate_absent_recorded', 'pair_benefit_positive_every_cell_every_design',
                                         'income_view_identical_wire_zero_difference', 'bootstrap_se_positive', 'unweighted_draw_equals_auc',
                                         'scores_reproduce', 'oriented_val_auc_ge_half'))
    return chk


# ---------------------------------------------------------------- summary over seeds (label-free: reads PROBE.json only)
def summarize(probe_ids, tag):
    cells_all, pairs_all = {}, {}
    for pid in probe_ids:
        for s in (0, 1):
            p = RUN_ROOT / pid / f'S{s}' / 'PROBE.json'
            if not p.is_file():
                continue
            g = json.loads(p.read_text())
            for cell, cres in g['cells'].items():
                for name, e in cres.items():
                    cells_all.setdefault(name, {}).setdefault(cell, e)
            for pk, schemes in g['paired'].items():
                for key, d in schemes.items():
                    tgt = pairs_all.setdefault(pk, {}).setdefault(key, {'cells': {}, 'cell_se': {}})
                    for c, v in d['cells'].items():
                        tgt['cells'].setdefault(c, v)
                        if c in d.get('cell_se', {}):
                            tgt['cell_se'].setdefault(c, d['cell_se'][c])
    # per release: mean SCORE AUC per view and scheme over the available cells
    rel = {}
    for name, cs in cells_all.items():
        r = {}
        for v in VIEWS:
            vals = {}
            for cell, e in cs.items():
                ve = e['views'][v]
                for k in SUMMARY_KEYS:
                    if ve.get(k) is not None:
                        vals.setdefault(k, []).append(ve[k])
                for d in DESIGNS:
                    if 'designs' in ve:
                        vals.setdefault(f'design:{d}', []).append(ve['designs'][d]['score_auc'])
                for n, x in (ve.get('slate_recipe_score_auc') or {}).items():
                    vals.setdefault(f'slate:{n}', []).append(x)
            r[v] = {k: {'mean': float(np.mean(x)), 'n_cells': len(x)} for k, x in vals.items()}
            r[v]['probe_winners'] = sorted({e['views'][v]['probe_winner'] for e in cs.values() if 'probe_winner' in e['views'][v]})
            r[v]['union_winners'] = sorted({e['views'][v].get('union_winner') for e in cs.values() if e['views'][v].get('union_winner')})
            r[v]['union_picks_probe_cells'] = sum(1 for e in cs.values() if str(e['views'][v].get('union_winner', '')).startswith('P:'))
            r[v]['slate_score_auc_sd_across_16_mean'] = float(np.mean([e['views'][v]['slate_score_auc_sd_across_16'] for e in cs.values()
                                                                       if e['views'][v].get('slate_score_auc_sd_across_16') is not None] or [float('nan')]))
            r[v]['n_cells'] = len(cs)
        rel[name] = r
    psum = {}
    for pk, schemes in pairs_all.items():
        psum[pk] = {}
        for key, d in schemes.items():
            vals = list(d['cells'].values())
            by = {s: [x for c, x in d['cells'].items() if c.startswith(f'S{s}')] for s in (0, 1)}
            psum[pk][key] = {'mean': float(np.mean(vals)), 'min': float(np.min(vals)), 'max': float(np.max(vals)), 'n_cells': len(vals),
                             'cells_le0': int(sum(x <= 0 for x in vals)),
                             'per_seed_mean': {s: float(np.mean(x)) for s, x in by.items() if x},
                             'rms_cell_se': float(np.sqrt(np.mean(np.square(list(d['cell_se'].values()))))) if d.get('cell_se') else None}
            if ':' not in key.split('|', 1)[1] or key.split('|', 1)[1].startswith('PROBE:'):
                psum[pk][key]['cells'] = d['cells']                    # per-cell detail for winners, oracle and probe designs only
    # the frozen stability rule applied to each reading scheme (8 complete cells with SEs only; descriptive, never a gate)
    from ohn import rules as RU
    from scipy.stats import kendalltau
    rules = RU.load_rules()
    cells8 = tuple(f'S{s}{r}k{k}' for s in (0, 1) for r in REPS for k in READER_SEEDS)
    for pk, schemes in pairs_all.items():
        for key, d in schemes.items():
            if set(d['cells']) == set(cells8) and set(d.get('cell_se', {})) == set(cells8):
                ent = {'cells': {c: d['cells'][c] for c in cells8}, 'cell_se': {c: d['cell_se'][c] for c in cells8}}
                if key.startswith('pair|'):
                    b = RU.benefit_summary(ent, float(rules['stability']['z']), 0.0)
                    psum[pk][key].update({'rule_label': b['label'], 'sd_total': b['sd_total'], 'sd_reader_split': b['sd_reader_split'],
                                          'rms_cell_se_rule': b['rms_cell_se'], 'flip_p': b['predicted_flip_probability_per_seed']})
                else:
                    rr = RU.recipient_summary(ent, float(rules['stability']['z']))
                    psum[pk][key].update({'rule_label': rr['label'], 'cells_above_limit': rr['cells_above_limit']})
    # fixed-recipe robustness of the pair-benefit sign: candidate and comparator read by the SAME recipe
    fixed = {}
    for pk, schemes in psum.items():
        vals = {key.split('|', 1)[1]: v['mean'] for key, v in schemes.items()
                if key.startswith('pair|SLATE:') or key.startswith('pair|PROBE:')}
        if vals:
            # the pair LOOKUP reader is non-informative on continuous released vectors (every pair vector is distinct, so it
            # returns the prior: AUC 0.5 for every release); it is excluded from the sign count and kept in by_recipe
            inf = {k: v for k, v in vals.items() if k != 'SLATE:LOOKUP'}
            arr = np.array(list(inf.values()))
            fixed[pk] = {'n_recipes_informative': len(inf), 'min': float(arr.min()), 'median': float(np.median(arr)), 'max': float(arr.max()),
                         'n_positive': int((arr > 0).sum()), 'excluded': ['SLATE:LOOKUP (pair view, non-informative on continuous outputs)'],
                         'by_recipe': vals}
    # ordering of mean pair recovery per scheme (releases with all 8 cells only) and agreement with the frozen-slate ordering
    order, tau = {}, {}
    ncmax = max((r['pair'].get('slate_score_auc', {}).get('n_cells', 0) for r in rel.values()), default=0)
    full = [n for n, r in rel.items() if r['pair'].get('slate_score_auc', {}).get('n_cells') == ncmax]
    for scheme in SUMMARY_KEYS + tuple(f'design:{d}' for d in DESIGNS) + tuple(f'slate:{n}' for n in SLATE16):
        m = {name: (rel[name]['pair'][scheme]['mean'], rel[name]['pair'][scheme]['n_cells']) for name in full
             if scheme in rel[name]['pair'] and rel[name]['pair'][scheme]['n_cells'] == ncmax}
        order[scheme] = [[n, round(x, 4), nc] for n, (x, nc) in sorted(m.items(), key=lambda kv: kv[1][0])]
        common = [n for n in m if 'slate_score_auc' in rel[n]['pair']]
        if len(common) >= 3 and scheme != 'slate_score_auc':
            t = kendalltau([rel[n]['pair']['slate_score_auc']['mean'] for n in common], [m[n][0] for n in common]).statistic
            tau[scheme] = {'kendall_tau_vs_frozen_slate': float(t), 'n_releases': len(common)}
    rec = {'schema': 'OHN1-rr-probe-summary-v1', 'utc': C.now(), 'probes': probe_ids, 'release_mean_score_auc': rel, 'pairs': psum,
           'fixed_recipe_pair_benefit_sign': fixed, 'ordering_low_to_high_pair_recovery': order, 'ordering_agreement': tau,
           'note': 'diagnostic reader probe on MODEL_SELECT (development role, adaptively used); never selection; reader-relative: '
                   'lower recovery for these readers is not protection against every possible reader'}
    OUT_RES.mkdir(parents=True, exist_ok=True)
    C.write_json(OUT_RES / f'{tag}__PROBE_SUMMARY.json', rec)
    import csv
    with open(OUT_RES / f'{tag}__PROBE_PAIRS.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['candidate', 'comparator', 'view', 'scheme', 'mean', 'min', 'max', 'n_cells', 'cells_le0', 'mean_s0', 'mean_s1',
                    'rms_cell_se', 'rule_label', 'sd_total', 'sd_reader_split'])
        for pk, schemes in sorted(psum.items()):
            cand, comp = pk.split('|')
            for key, v in sorted(schemes.items()):
                view, scheme = key.split('|', 1)
                w.writerow([cand, comp, view, scheme, v['mean'], v['min'], v['max'], v['n_cells'], v['cells_le0'],
                            v['per_seed_mean'].get(0), v['per_seed_mean'].get(1), v.get('rms_cell_se'), v.get('rule_label'),
                            v.get('sd_total'), v.get('sd_reader_split')])
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    r = sub.add_parser('run'); r.add_argument('--spec', required=True); r.add_argument('--seed', type=int, required=True)
    s = sub.add_parser('summarize'); s.add_argument('--probe', action='append', required=True); s.add_argument('--tag', required=True)
    lf = sub.add_parser('lcfixed'); lf.add_argument('--spec', required=True); lf.add_argument('--seed', type=int, required=True)
    ls = sub.add_parser('lcfixed_summary'); ls.add_argument('--bank', action='append', required=True); ls.add_argument('--tag', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'lcfixed':
        lc_fixed(a.spec, a.seed)
        return
    if a.cmd == 'lcfixed_summary':
        out = lc_fixed_summary(a.bank, a.tag)
        print(json.dumps({k: [v['winner_flag'], v['n_recipes_flagged_shrink'], v['n_recipes_informative'], v['median_full_minus_quarter']]
                          for k, v in out.items()}, indent=1))
        return
    if a.cmd == 'selftest':
        chk = selftest()
        OUT_RES.mkdir(parents=True, exist_ok=True)
        C.write_json(OUT_RES / 'checks' / 'RR_PROBE_SELFTEST.json', chk)
        print(json.dumps(chk, indent=1))
        if not chk['all_ok']:
            raise SystemExit(1)
    elif a.cmd == 'run':
        run(a.spec, a.seed)
    elif a.cmd == 'summarize':
        rec = summarize(a.probe, a.tag)
        print(json.dumps(rec['ordering_low_to_high_pair_recovery'], indent=1))


if __name__ == '__main__':
    main()
