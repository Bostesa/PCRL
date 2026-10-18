"""Development reader grid of the PCRL correction-transfer campaign (`pcrl_correction_transfer_stability_v1`; EXPLORATORY
DEVELOPMENT evidence on adaptively used MODEL_SELECT records, never confirmation).

A minimal adaptation of the source grid `ohn.devreaders` (occupation-headroom campaign; DEV_GRID.json schema OHN1-devgrid-v1;
publication 1435c395d). Readers estimate the recorded SEX column (sex 0 = men, sex 1 = women) of public-use ACS 2019 1-Year PUMS
Florida person records from released predictions only: each recipient's float64 probability vector and argmax decision (income
view, occupation view) and the two outputs aligned to the same person (the linked pair). Readers never see raw fields, hidden
scores, codes or labels other than SEX. Pool: MODEL_SELECT (8,001 people, 5,306 households). SEX is read ONLY through
`cts.common.load('MODEL_SELECT', ('features', 'sex'), operation='readers', purpose=...)` (logged); the Census inputs are dropped
right after loading. ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT labels are never touched (the access layer refuses them).

Unchanged from the source (so the source's cached readers stay valid and every number is comparable):
- repetitions R1, R2: MODEL_SELECT households ordered by SHA-256('OHN1_devgrid_<R>|<household>') and dealt round-robin into 10
  parts; parts 0-4 = FIT (50% of households), 5-6 = VALIDATION (20%), 7-9 = SCORE (30%); household-disjoint;
- reader seeds k0, k1 = MODEL_SEED_BASE + 300 + 10 x release seed + k; a cell = (release seed, repetition, reader seed); 8 cells
  per configuration over release seeds 0 and 1;
- the 16 recipes (source 8: OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB, LOOKUP; new 8: ET5, ET25, HGB15, HGB31,
  MLP64, MLP6432, CATI01, CATI1) from the locked recipe code (scm_bench.readers, task_conditioned_representation.attacks /
  .inference, ffv.readers / .features; file hashes and library versions must equal the source DEVGRID_CODE_LOCK.json), fitted on
  FIT, oriented and selected on VALIDATION (max oriented VALIDATION AUC, exact ties to the earlier candidate; pair candidates =
  pair readers, then the ignore-occupation (income-view) and ignore-income (occupation-view) readers, source recipes first; near
  tie < 0.010 flagged), scored on SCORE; SCORE is never used for selection;
- household bootstrap of SCORE (B = 400, seed 2026101806 + 100 x release seed + repetition index R1 0 / R2 1 / R3 2, readers and
  orientations fixed); every release in a cell shares the same weights, so differences of draws are paired;
- nested learning curves: FIT households ordered by SHA-256('OHN1_devgrid_LC|<household>') into 4 parts; fractions 1/4 (part 0),
  1/2 (parts 0-1), 1 (all); VALIDATION and SCORE unchanged (cells 'S<s><R>k<k>@0.25' / '@0.5').
Changed or added here:
- releases come from a bank spec (below): frozen source releases through the admitted copies exactly as `ohn.devreaders` loads them,
  new releases of this campaign as MODEL_SELECT probability arrays named by path + SHA-256 (the file is re-hashed before use);
- reader cache, key verified against `ohn.devreaders.cell_readers`:
      SHA-256('OHN1-devgrid-v1|<rep>|<reader seed>|<FIT fraction>|<wire hash>|<SEX array hash>')
  where the wire hash covers the view name and the float64 arrays that view reads (income; occupation; income then occupation),
  the SEX hash is scm_bench.common.arr_sha of the MODEL_SELECT SEX array, and the recipe set is implied by the grid version (an
  entry always holds all 16 recipes; checked on read). Lookup order: this campaign's cache (.private/<campaign>/runs/devgrid_cache),
  then the admitted source cache (.private/pcrl_occupation_headroom_overnight_v1/runs/devgrid_cache, read-only). New entries are
  written ONLY to this campaign's cache (with a checksum, re-read and compared after writing). Hits from each cache are counted
  separately from new recipe fits (one hit = one view of one wire in one cell = 16 recipe fits avoided);
- optional third repetition R3 (finalists): the source reader-robustness extra split (RR_SPLIT3_*.json; tag
  'OHN1_devgrid_R3_diag', same 50/20/30 construction, bootstrap index 2), keyed here as 'R3:OHN1_devgrid_R3_diag'. At full FIT the
  source's own R3 readers (RR_PROBE/R3_cache; key SHA-256('OHN1-rr-split3-v1|OHN1_devgrid_R3_diag|<reader seed>|<wire hash>|<SEX
  hash>'), fitted by a verbatim restatement of the same recipes, checked bitwise by the source) are reused read-only;
- extra reader seeds k = 2..9 (never colliding with another release seed's reader seeds) exist only for the screen study
  (identical releases under separately fitted readers); the registered grid uses k0, k1;
- self-pairs (a name that is both candidate and comparator) are not paired; unit counters go to cts.common.unit_stats.

Bank spec (JSON): {"bank_id", "purpose", "members": {name: member}, "candidates": [...], "comparators": [...],
"fractions": [1.0, ...] (default [1.0]), "reps": ["R1", "R2"] (default; add "R3" for finalists)}. A member is either
  a frozen source release (loaded by ohn.devreaders.load_member from the admitted copies):
      {"type": "reference", "name": NAME} | {"type": "occ", "family": F, "config": C, "anchor": "H3" | "TW"} |
      {"type": "complete", "family": F, "config": C}
  or a release given task by task, per release seed:
      {"type": "arrays", "seeds": {"0": {"income": SRC, "occupation": SRC}, "1": {...}}}
  with SRC one of
      {"path": "<repository-relative or absolute .npz>", "key": "<array name>", "sha256": "<SHA-256 of the file>"}
          (the file must lie under this campaign's runs or the admitted source copies; optional "row_id_key" names an array of
          MODEL_SELECT row ids that must equal the export's order),
      {"frozen_anchor": "H3" | "TW"}            (income only: the frozen anchor's income wire),
      {"frozen_reference": NAME}                 (that task's output of a frozen reference release),
      {"frozen_occ": {"family": F, "config": C}} (occupation only: a frozen source occupation release).
Every array must be an (8,001, 2) / (8,001, 5) float64 probability-vector array (finite, non-negative, rows summing to 1 within
1e-9); decisions are its argmax.

    python3 -m cts.devreaders selftest                                  invented data only (no real data, no label)
    python3 -m cts.devreaders manifest                                  label-free grid manifest (household ids only)
    python3 -m cts.devreaders lock                                      write-once DEVGRID_CODE_LOCK.json (commit it before run)
    python3 -m cts.devreaders run --spec BANK.json --seed S [--out DIR] (through cts.jobqueue, with the locked interpreter)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

from scm_bench.common import wire_sha
from scm_bench.readers import SLATE, kinds, lookup_keys, fit_lookup, predict_lookup
from task_conditioned_representation.attacks import MODEL_SEED_BASE, fit_reader, predict_reader, orientation, orient
from task_conditioned_representation.inference import auc
from ffv.readers import NEW8, design_for, fit_new, predict_new, pick, versions
from . import common as C

GRID_VERSION = 'OHN1-devgrid-v1'            # the SOURCE grid version string: part of the cache key (cells, recipes, selection unchanged)
SCHEMA = 'CTS1-devgrid-v1'                  # this module's result schema
SRC8 = tuple(n for n, _ in SLATE) + ('LOOKUP',)
RECIPES = SRC8 + NEW8
REPS = ('R1', 'R2')                         # registered repetitions
ALL_REPS = ('R1', 'R2', 'R3')
REP_TAG = {'R1': 'OHN1_devgrid_R1', 'R2': 'OHN1_devgrid_R2', 'R3': 'OHN1_devgrid_R3_diag'}
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2}
READER_SEEDS = (0, 1)
MAX_K = 9                                   # k <= 9 never reaches the next release seed's reader seeds (10 x release seed)
FRACTIONS = (0.25, 0.5, 1.0)
LC_TAG = 'OHN1_devgrid_LC'
VIEWS = ('income', 'occupation', 'pair')
SINGLE_VIEWS = ('income', 'occupation')
B_BOOT = 400
BOOT_SEED = 2026101806
NEAR_TIE = 0.010
SOURCE_R3_VERSION = 'OHN1-rr-split3-v1'
CACHE = C.RUNS / 'devgrid_cache'                              # this campaign's cache (new entries only here)
SOURCE_CACHE = C.OHN_RUNS / 'devgrid_cache'                   # admitted copy of the source grid cache (read-only)
SOURCE_R3_CACHE = C.OHN_RUNS / 'RR_PROBE' / 'R3_cache'        # admitted copy of the source R3 readers (read-only)
SOURCE_LOCK = C.OHN_RESULTS / 'DEVGRID_CODE_LOCK.json'
LOCK_PATH = C.RESULTS / 'DEVGRID_CODE_LOCK.json'
MANIFEST_PATH = C.RESULTS / 'roles' / 'reader_investigator' / 'DEV_GRID.json'
SELFTEST_PATH = C.RESULTS / 'roles' / 'reader_investigator' / 'checks' / 'DEVGRID_SELFTEST.json'
RECIPE_FILES = ('ffv/readers.py', 'ffv/features.py', 'scm_bench/readers.py', 'task_conditioned_representation/attacks.py',
                'task_conditioned_representation/inference.py')
LOCK_FILES = ('cts/devreaders.py', 'ohn/devreaders.py', 'ohn/devrun.py', 'ohn/common.py') + RECIPE_FILES
RELEASE_ROOTS = (C.RUNS, C.OHN_RUNS, C.ROOT / '.private' / 'pcrl_income_headroom_successor_v1')
N_MODEL_SELECT = 8001


def default_caches():
    return {'own': CACHE, 'source': SOURCE_CACHE, 'source_r3': SOURCE_R3_CACHE}


def new_stats():
    return {'reader_fits': 0, 'cache_hits_own': 0, 'cache_hits_source': 0, 'cache_hits_source_r3': 0, 'reader_cpu_s': 0.0}


# ---------------------------------------------------------------- cells (label-free)
def roles_of(unit, rep):
    """FIT / VALIDATION / SCORE masks of one repetition (R1, R2 = the source grid; R3 = the source extra split)."""
    part = C.household_folds(unit, 10, REP_TAG[rep])
    return {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}


def _frac(frac):
    f = float(frac)
    if f not in FRACTIONS:
        raise ValueError(f'FIT fraction {frac!r} is not one of {FRACTIONS}')
    return f


def lc_fit(unit, fit_mask, frac):
    frac = _frac(frac)
    if frac == 1.0:
        return fit_mask
    part = C.household_folds(unit, 4, LC_TAG)
    keep = part == 0 if frac == 0.25 else part <= 1
    return fit_mask & keep


def reader_seed(release_seed, k):
    if not (0 <= int(k) <= MAX_K):
        raise ValueError(f'reader seed index {k} outside 0..{MAX_K}')
    return MODEL_SEED_BASE + 300 + 10 * int(release_seed) + int(k)


def cell_name(release_seed, rep, k, frac=1.0):
    return f'S{release_seed}{rep}k{k}' + ('' if _frac(frac) == 1.0 else f'@{_frac(frac)}')


# ---------------------------------------------------------------- cache keys (verified against the source definitions)
def wire_hash(view, inc, occ):
    """Identical to ohn.devreaders._wire_hash: the view name, then the float64 arrays the view reads."""
    h = hashlib.sha256(view.encode())
    for a in ((inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def cache_key(rep, rs, frac, wh, sex_sha):
    """R1 / R2: verbatim the source key (ohn.devreaders.cell_readers); R3: the split tag is added so that no key can collide with
    a differently defined 'R3'."""
    tag = rep if rep in REPS else f'{rep}:{REP_TAG[rep]}'
    return hashlib.sha256(f'{GRID_VERSION}|{tag}|{rs}|{_frac(frac)}|{wh}|{sex_sha}'.encode()).hexdigest()


def source_r3_key(rs, wh, sex_sha):
    """The source extra-split reader key (rr_split3.view_readers; full FIT only)."""
    return hashlib.sha256(f'{SOURCE_R3_VERSION}|{REP_TAG["R3"]}|{rs}|{wh}|{sex_sha}'.encode()).hexdigest()


def _checksum(out):
    h = hashlib.sha256()
    for n in RECIPES:
        h.update(n.encode()); h.update(np.ascontiguousarray(np.asarray(out[n][0], dtype=np.float64)).tobytes())
    return h.hexdigest()


def _read_entry(p, n_score):
    with np.load(p, allow_pickle=False) as z:
        info = json.loads(str(z['info']))
        if set(info) != set(RECIPES) or any(f'score__{n}' not in z.files for n in RECIPES):
            raise PermissionError(f'cache entry {p.name}: recipe set differs from the 16 recipes')
        out = {n: (z[f'score__{n}'].copy(), info[n]) for n in RECIPES}
        stored = str(z['checksum']) if 'checksum' in z.files else None
    for n, (s, _) in out.items():
        if s.shape != (n_score,) or s.dtype != np.float64 or not np.isfinite(s).all():
            raise PermissionError(f'cache entry {p.name}: recipe {n} scores malformed (shape {s.shape}, expected {n_score})')
    if stored is not None and stored != _checksum(out):
        raise PermissionError(f'cache entry {p.name}: checksum differs (corrupted entry)')
    return out


def _write_entry(p, out, info):
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = p.parent / f'.{p.stem}.{os.getpid()}.npz'
    with open(tmp, 'wb') as f:
        np.savez(f, info=json.dumps(info), checksum=_checksum(out), **{f'score__{n}': out[n][0] for n in RECIPES})
    os.replace(tmp, p)
    back = _read_entry(p, len(out[RECIPES[0]][0]))                 # re-read and compare (guards against corrupted writes)
    if any(not np.array_equal(back[n][0], out[n][0]) for n in RECIPES):
        p.unlink(missing_ok=True)
        raise RuntimeError(f'cache entry {p.name} did not read back bitwise; removed')


# ---------------------------------------------------------------- reader fitting (verbatim the source recipe dispatch)
def _src_tab(view, inc, occ, kind):
    a, b = kinds(inc, 2)[kind], kinds(occ, 5)[kind]
    return a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))


def _fit_one(name, view, inc, occ, sex, fit, val, rs):
    """Fit one recipe on FIT rows; return oriented scores on all rows and VALIDATION AUC / orientation / convergence."""
    if name == 'LOOKUP':
        keys = lookup_keys(inc) if view == 'income' else lookup_keys(occ) if view == 'occupation' else lookup_keys(inc, occ)
        m = fit_lookup([keys[i] for i in np.flatnonzero(fit)], sex[fit])
        raw = predict_lookup(m, keys)
        conv = {}
    elif name in dict(SLATE):
        X = _src_tab(view, inc, occ, dict(SLATE)[name])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = fit_reader(X[fit], sex[fit], name, rs)
        raw = predict_reader(m, X)
        conv = {}
    else:
        tf, _, _ = design_for(name, view, inc[fit], occ[fit])
        X = tf(inc, occ)
        m, conv = fit_new(name, X[fit], sex[fit], rs)
        raw = predict_new(m, X)
    sign = orientation(sex[val], raw[val])
    s = orient(raw, sign)
    return s, {'val_auc': float(auc(sex[val], s[val])), 'orientation': int(sign), **conv}


def cell_readers(view, inc, occ, sex, unit, rel_seed, rep, k, frac, stats, sex_sha=None, caches=None):
    """All 16 recipes for one view of one wire in one cell. Returns {recipe: (oriented scores on SCORE rows, info)}.
    Own cache, then the source cache (R1 / R2) or the source R3 readers (R3, full FIT), then new fits (own cache only)."""
    caches = caches or default_caches()
    frac = _frac(frac)
    r = roles_of(unit, rep)
    fit = lc_fit(unit, r['FIT'], frac); val = r['VALIDATION']; sc = r['SCORE']
    n_sc = int(sc.sum())
    rs = reader_seed(rel_seed, k)
    wh = wire_hash(view, inc, occ)
    sx = sex_sha or C.arr_sha(np.asarray(sex))
    own = Path(caches['own']) / f'{cache_key(rep, rs, frac, wh, sx)}.npz'
    if own.is_file():
        stats['cache_hits_own'] += 1
        return _read_entry(own, n_sc)
    if rep in REPS:
        src = Path(caches['source']) / f'{cache_key(rep, rs, frac, wh, sx)}.npz'
        if src.is_file():
            stats['cache_hits_source'] += 1
            return _read_entry(src, n_sc)
    elif rep == 'R3' and frac == 1.0:
        src = Path(caches['source_r3']) / f'{source_r3_key(rs, wh, sx)}.npz'
        if src.is_file():
            stats['cache_hits_source_r3'] += 1
            return _read_entry(src, n_sc)
    out, info = {}, {}
    t0 = time.process_time()
    for name in RECIPES:
        s, inf = _fit_one(name, view, inc, occ, sex, fit, val, rs)
        if not np.isfinite(s).all():
            raise RuntimeError(f'{name} {view} {rep} k{k}: non-finite reader scores')
        out[name] = (np.ascontiguousarray(s[sc], dtype=np.float64), inf); info[name] = inf
        stats['reader_fits'] += 1
    stats['reader_cpu_s'] += time.process_time() - t0
    _write_entry(own, out, info)
    return out


# ---------------------------------------------------------------- winners, bootstrap, paired differences (source logic)
def winners(rd_inc, rd_occ, rd_pair=None):
    """Enhanced winners per view from the cached readers (VALIDATION only). The pair winner needs all three views."""
    w = {}
    for view, rd in (('income', rd_inc), ('occupation', rd_occ)):
        if rd is not None:
            w[view] = pick([(n, rd[n][1]['val_auc'], f'{view}__{n}', 'source' if n in SRC8 else 'new') for n in RECIPES])
    if rd_pair is not None:
        if rd_inc is None or rd_occ is None:
            raise ValueError('the pair winner needs the income and occupation readers (ignore-one alternatives)')
        cands = []
        for group in (SRC8, NEW8):
            org = 'source' if group is SRC8 else 'new'
            cands += [(n, rd_pair[n][1]['val_auc'], f'pair__{n}', org) for n in group]
            cands += [(f'IGNORE_occupation__{n}', rd_inc[n][1]['val_auc'], f'income__{n}', org) for n in group]
            cands += [(f'IGNORE_income__{n}', rd_occ[n][1]['val_auc'], f'occupation__{n}', org) for n in group]
        w['pair'] = pick(cands)
    return w


def _score_of(key, rd):
    view, name = key.split('__', 1)
    return rd[view][name][0]


def wauc_draws(y, s, W):
    """AUC of fixed scores s under household-bootstrap weights W (B x n) (verbatim the source)."""
    u, inv = np.unique(s, return_inverse=True)
    out = np.empty(W.shape[0])
    pos = (y == 1).astype(np.float64); neg = 1.0 - pos
    for b in range(W.shape[0]):
        wp = np.bincount(inv, weights=W[b] * pos, minlength=len(u)); wn = np.bincount(inv, weights=W[b] * neg, minlength=len(u))
        cn = np.cumsum(wn) - wn
        out[b] = np.sum(wp * (cn + 0.5 * wn)) / (wp.sum() * wn.sum())
    return out


def boot_weights(unit_sc, rel_seed, rep):
    g, inv = np.unique(unit_sc, return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED + 100 * int(rel_seed) + REP_INDEX[rep])
    W = np.empty((B_BOOT, len(unit_sc)))
    for b in range(B_BOOT):
        cnt = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))
        W[b] = cnt[inv]
    return W


def _check_views(views):
    views = tuple(views)
    if views not in (VIEWS, ('income',), ('occupation',), SINGLE_VIEWS):
        raise ValueError(f'views must be all three {VIEWS} or single views only, got {views}')
    return views


def evaluate_bank(bank, rel_seed, sex, unit, fracs=(1.0,), stats=None, reps=REPS, ks=READER_SEEDS, views=VIEWS, caches=None):
    """bank: {name: {'income': (n, 2), 'occupation': (n, 5)}} on MODEL_SELECT rows. Returns per cell (and FIT fraction): every
    release's winners (VALIDATION) with SCORE AUCs and the household-bootstrap draws of each winner's SCORE AUC."""
    stats = stats if stats is not None else new_stats()
    views = _check_views(views)
    fracs = tuple(_frac(f) for f in fracs)
    for rep in reps:
        if rep not in ALL_REPS:
            raise ValueError(f'unknown repetition {rep}')
    sx = C.arr_sha(np.asarray(sex))
    res = {}
    for rep in reps:
        sc = roles_of(unit, rep)['SCORE']; y = sex[sc]
        W = boot_weights(unit[sc], rel_seed, rep)
        for k in ks:
            for frac in fracs:
                cell = cell_name(rel_seed, rep, k, frac)
                cres = {}
                for name, rel in bank.items():
                    inc, occ = rel['income'], rel['occupation']
                    rd = {v: cell_readers(v, inc, occ, sex, unit, rel_seed, rep, k, frac, stats, sx, caches) for v in views}
                    w = winners(rd.get('income'), rd.get('occupation'), rd.get('pair'))
                    entry = {}
                    for view in views:
                        s = _score_of(w[view]['reader_key'], rd)
                        entry[view] = {'winner': w[view]['candidate'], 'origin': w[view]['origin'],
                                       'val_auc': w[view]['attack_validation_auc'], 'gap_to_runner_up': w[view]['gap_to_runner_up'],
                                       'near_tie': w[view]['AV_NEAR_TIE'], 'score_auc': float(auc(y, s)), 'boot': wauc_draws(y, s, W)}
                    cres[name] = entry
                res[cell] = cres
    return res, stats


def paired(res, cand, comp, views=VIEWS):
    """Per cell: pair benefit (comparator - candidate; positive favours the candidate) and recipient differences (candidate -
    comparator) with household-bootstrap SEs (paired draws: same weights for both releases)."""
    out = {v: {'cells': {}, 'cell_se': {}} for v in views}
    for cell, cres in res.items():
        for view in views:
            a, b = cres[cand][view], cres[comp][view]
            if view == 'pair':
                d, db = b['score_auc'] - a['score_auc'], b['boot'] - a['boot']
            else:
                d, db = a['score_auc'] - b['score_auc'], a['boot'] - b['boot']
            out[view]['cells'][cell] = float(d); out[view]['cell_se'][cell] = float(np.std(db, ddof=1))
    return out


def strip(res):
    return {cell: {n: {v: {k: x for k, x in e.items() if k != 'boot'} for v, e in ent.items()} for n, ent in cres.items()}
            for cell, cres in res.items()}


# ---------------------------------------------------------------- releases (frozen source releases; new arrays by path + sha256)
FROZEN_TYPES = ('reference', 'occ', 'complete')


def _check_prob(p, k, what, n=None):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    if (p.ndim != 2 or p.shape[1] != k or (n is not None and p.shape[0] != n) or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0)):
        raise ValueError(f'{what}: not a valid ({n}, {k}) probability-vector release')
    return p


def _resolve(path, roots):
    p = Path(path)
    p = (p if p.is_absolute() else C.ROOT / p).resolve()
    if not any(p.is_relative_to(Path(r).resolve()) for r in roots):
        raise PermissionError(f'{path}: release files must lie under {[str(r) for r in roots]}')
    return p


def _task_array(src, task, seed, row_id=None, roots=RELEASE_ROOTS):
    from ohn import devrun as D
    if 'frozen_anchor' in src:
        if task != 'income':
            raise ValueError('frozen_anchor supplies the income wire only')
        return D.anchor_income(src['frozen_anchor'], seed, 'MODEL_SELECT'), {'frozen_anchor': src['frozen_anchor']}
    if 'frozen_reference' in src:
        return D.reference_release(src['frozen_reference'], seed, 'MODEL_SELECT')[task], {'frozen_reference': src['frozen_reference']}
    if 'frozen_occ' in src:
        if task != 'occupation':
            raise ValueError('frozen_occ supplies the occupation wire only')
        f = src['frozen_occ']
        return D.load_occupation(f['family'], f['config'], seed, roles=('MODEL_SELECT',))[0]['MODEL_SELECT'], {'frozen_occ': f}
    p = _resolve(src['path'], roots)
    got = C.sha(p)
    if got != src['sha256']:
        raise PermissionError(f'{src["path"]}: SHA-256 {got} differs from the bank spec')
    with np.load(p, allow_pickle=False) as z:
        a = z[src['key']].copy()
        if src.get('row_id_key'):
            if row_id is None or not np.array_equal(z[src['row_id_key']], row_id):
                raise PermissionError(f'{src["path"]}: row ids differ from the MODEL_SELECT export order')
    return a, {'path': str(p.relative_to(C.ROOT)) if p.is_relative_to(C.ROOT) else str(p), 'key': src['key'], 'sha256': got}


def load_member(member, seed, n=N_MODEL_SELECT, row_id=None, roots=RELEASE_ROOTS):
    """One bank member's MODEL_SELECT release {'income': (n, 2), 'occupation': (n, 5)} and its provenance."""
    t = member.get('type')
    if t in FROZEN_TYPES:
        from ohn.devreaders import load_member as source_load          # the admitted copies, exactly as the source loads them
        rel, prov = source_load(member, seed), {'frozen_member': member}
    elif t == 'arrays':
        per = member['seeds'][str(seed)]
        rel, prov = {}, {}
        for task in C.TASKS:
            rel[task], prov[task] = _task_array(per[task], task, seed, row_id, roots)
    else:
        raise ValueError(f'unknown member type {t!r}')
    rel = {task: _check_prob(rel[task], C.K[task], f'{task} release', n) for task in C.TASKS}
    prov['wire_sha256'] = {task: wire_sha(rel[task]) for task in C.TASKS}
    return rel, prov


def validate_spec(spec):
    for key in ('bank_id', 'members', 'candidates', 'comparators'):
        if key not in spec:
            raise ValueError(f'bank spec needs {key!r}')
    names = set(spec['members'])
    missing = [x for x in list(spec['candidates']) + list(spec['comparators']) if x not in names]
    if missing:
        raise ValueError(f'candidates / comparators not among the members: {missing}')
    for rep in spec.get('reps', REPS):
        if rep not in ALL_REPS:
            raise ValueError(f'unknown repetition {rep}')
    if not set(REPS) <= set(spec.get('reps', REPS)):
        raise ValueError('R1 and R2 are always part of the grid (R3 is an addition)')
    for f in spec.get('fractions', [1.0]):
        _frac(f)
    if 1.0 not in [float(f) for f in spec.get('fractions', [1.0])]:
        raise ValueError('the full FIT fraction 1.0 is always part of the grid')
    return spec


def aliases(provs):
    """Members sharing an identical released wire per task (exact aliases: identical readers under the cache)."""
    out = {}
    for task in C.TASKS:
        by = {}
        for name, pv in provs.items():
            by.setdefault(pv['wire_sha256'][task], []).append(name)
        out[task] = [sorted(v) for v in by.values() if len(v) > 1]
    return out


# ---------------------------------------------------------------- lock, manifest, run
def lock_check(lock_path=LOCK_PATH):
    """Refuse to score unless the committed DEVGRID_CODE_LOCK.json matches the code, and the recipe code and library versions equal
    the SOURCE grid lock (otherwise cached source readers and new fits would not be comparable)."""
    if not C.committed(lock_path):
        raise PermissionError('reader grid code not yet locked (DEVGRID_CODE_LOCK.json must be committed first)')
    rec = json.loads(Path(lock_path).read_text())
    cur = {f: C.sha(C.ROOT / f) for f in rec['files_sha256']}
    if cur != rec['files_sha256'] or rec['versions'] != versions():
        raise PermissionError('reader grid code or library versions differ from DEVGRID_CODE_LOCK.json')
    src = json.loads(SOURCE_LOCK.read_text())
    if any(src['files_sha256'][f] != C.sha(C.ROOT / f) for f in RECIPE_FILES) or src['versions'] != versions():
        raise PermissionError('recipe code or library versions differ from the source grid lock (cache reuse would be invalid)')


def make_lock():
    src = json.loads(SOURCE_LOCK.read_text())
    if src['versions'] != versions():
        raise PermissionError(f'library versions {versions()} differ from the source grid lock {src["versions"]}; lock with the '
                              f'interpreter of the source grid ({C.TORCH_PY})')
    bad = [f for f in RECIPE_FILES if src['files_sha256'][f] != C.sha(C.ROOT / f)]
    if bad:
        raise PermissionError(f'recipe files differ from the source grid lock: {bad}')
    rec = {'schema': 'CTS1-devgrid-code-lock-v1', 'utc': C.now(), 'grid_version': GRID_VERSION, 'result_schema': SCHEMA,
           'versions': versions(), 'files_sha256': {f: C.sha(C.ROOT / f) for f in LOCK_FILES},
           'source_lock_sha256': C.sha(SOURCE_LOCK), 'manifest_sha256': C.sha(MANIFEST_PATH) if MANIFEST_PATH.is_file() else None,
           'cache_key': f'SHA-256("{GRID_VERSION}|<rep>|<reader seed>|<FIT fraction>|<wire hash>|<SEX array hash>") (R1/R2 = source key)'}
    C.write_json_x(LOCK_PATH, rec)
    return rec


def manifest():
    """Label-free grid manifest (MODEL_SELECT household ids only)."""
    unit = C.features('MODEL_SELECT')['unit']

    def desc(m):
        hh = np.unique(unit[m])
        return {'people': int(m.sum()), 'households': int(len(hh)),
                'household_set_sha256': hashlib.sha256(','.join(map(str, hh)).encode()).hexdigest()}
    reps = {}
    for rep in ALL_REPS:
        r = roles_of(unit, rep)
        reps[rep] = {k: desc(v) for k, v in r.items()}
        reps[rep]['tag'] = REP_TAG[rep]
        reps[rep]['disjoint'] = bool(not np.intersect1d(unit[r['FIT']], unit[r['VALIDATION']]).size
                                     and not np.intersect1d(unit[r['FIT']], unit[r['SCORE']]).size
                                     and not np.intersect1d(unit[r['VALIDATION']], unit[r['SCORE']]).size)
        reps[rep]['learning_curve_fit_people'] = {str(f): int(lc_fit(unit, r['FIT'], f).sum()) for f in FRACTIONS}
    overlap = {f'{a}_vs_{b}_{role}': int(np.intersect1d(np.unique(unit[roles_of(unit, a)[role]]), np.unique(unit[roles_of(unit, b)[role]])).size)
               for a, b in (('R1', 'R2'), ('R1', 'R3'), ('R2', 'R3')) for role in ('FIT', 'VALIDATION', 'SCORE')}
    src = json.loads((C.OHN_RESULTS / 'DEV_GRID.json').read_text())
    same = {rep: {role: reps[rep][role]['household_set_sha256'] == src['repetitions'][rep][role]['household_set_sha256']
                  for role in ('FIT', 'VALIDATION', 'SCORE')} for rep in REPS}
    return {'schema': SCHEMA, 'pool': 'MODEL_SELECT (8,001 people, 5,306 households)', 'repetitions': reps,
            'registered_repetitions': list(REPS), 'optional_repetitions': ['R3'],
            'R1_R2_identical_to_source_DEV_GRID': same, 'household_overlap_between_repetitions': overlap,
            'reader_seeds': {f'S{s}k{k}': reader_seed(s, k) for s in C.SEEDS for k in READER_SEEDS},
            'cells': [cell_name(s, rep, k) for s in C.SEEDS for rep in REPS for k in READER_SEEDS],
            'optional_cells_R3': [cell_name(s, 'R3', k) for s in C.SEEDS for k in READER_SEEDS],
            'recipes': {'source8': list(SRC8), 'new8': list(NEW8)}, 'versions_at_manifest': versions(),
            'selection': 'max oriented VALIDATION AUC; exact ties to the earlier candidate (source first, pair then ignore-occupation '
                         'then ignore-income); near tie < 0.010 flagged; SCORE never used for selection',
            'bootstrap': {'B': B_BOOT, 'seed': f'{BOOT_SEED} + 100 x release seed + repetition index {REP_INDEX}',
                          'unit': 'SCORE households, readers fixed'},
            'learning_curve_fractions': list(FRACTIONS), 'tags': {'repetitions': REP_TAG, 'learning_curve': LC_TAG},
            'cache': {'key': f'SHA-256("{GRID_VERSION}|<rep>|<reader seed>|<FIT fraction>|<wire hash>|<SEX array hash>") for R1/R2 '
                             '(verbatim ohn.devreaders.cell_readers); R3 keyed "R3:OHN1_devgrid_R3_diag"',
                      'own': str(CACHE.relative_to(C.ROOT)), 'source_read_only': str(SOURCE_CACHE.relative_to(C.ROOT)),
                      'source_r3_read_only': str(SOURCE_R3_CACHE.relative_to(C.ROOT)),
                      'source_r3_key': f'SHA-256("{SOURCE_R3_VERSION}|{REP_TAG["R3"]}|<reader seed>|<wire hash>|<SEX array hash>")'}}


def run(spec_path, seed, out_dir=None):
    """Evaluate one bank (candidates and comparators) on the development grid for one release seed."""
    lock_check()
    spec = validate_spec(json.loads(Path(spec_path).read_text()))
    out = Path(out_dir) if out_dir else C.RUNS / 'GRID' / spec['bank_id'] / f'S{seed}'
    if out.exists():
        raise PermissionError('fresh output directory required')
    S = C.load('MODEL_SELECT', ('features', 'sex'), operation='readers',
               purpose=f"development reader grid {spec['bank_id']} seed {seed} (SEX of MODEL_SELECT; reader FIT / VALIDATION / SCORE)")
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    del S                                                  # readers never see Census inputs
    loaded = {name: load_member(m, seed, n=len(sex), row_id=row_id) for name, m in spec['members'].items()}
    bank = {name: rel for name, (rel, _) in loaded.items()}
    provs = {name: pv for name, (_, pv) in loaded.items()}
    return run_bank(bank, sex, unit, spec, seed, out, provs=provs)


def run_bank(bank, sex, unit, spec, seed, out, provs=None, caches=None):
    out = Path(out)
    out.mkdir(parents=True, mode=0o700)
    fracs = tuple(_frac(f) for f in spec.get('fractions', [1.0]))
    reps = tuple(spec.get('reps', REPS))
    stats = new_stats()
    t0 = time.process_time()
    res, stats = evaluate_bank(bank, seed, sex, unit, fracs, stats, reps=reps, caches=caches)
    pairs = {}
    for cand in spec['candidates']:
        for comp in spec['comparators']:
            if comp != cand:
                pairs.setdefault(cand, {})[comp] = paired(res, cand, comp)
    winner_changes = {}
    for name in bank:
        for view in VIEWS:
            full = [c for c in res if '@' not in c]
            ws = [res[c][name][view]['winner'] for c in full]
            winner_changes[f'{name}|{view}'] = {'distinct_winners': sorted(set(ws)), 'n_cells': len(ws),
                                                'near_ties': sum(bool(res[c][name][view]['near_tie']) for c in full)}
    boot = {f'{cell}__{n}__{v}': e['boot'] for cell, cres in res.items() for n, ent in cres.items() for v, e in ent.items()}
    with open(out / 'BOOT.npz', 'wb') as f:
        np.savez(f, **boot)
    provs = provs or {}
    rec = {'schema': SCHEMA, 'bank_id': spec['bank_id'], 'seed': seed, 'spec': spec, 'grid_version': GRID_VERSION,
           'reps': list(reps), 'fractions': list(fracs), 'cells': strip(res), 'paired': pairs, 'winner_changes': winner_changes,
           'members': provs, 'exact_aliases': aliases(provs) if provs else None, 'stats': stats,
           'cache_hit_note': 'one cache hit = one view of one wire in one cell (16 recipe fits avoided); reader_fits counts recipe fits',
           'cpu_s': round(time.process_time() - t0, 1), 'versions': versions(), 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV),
           'note': 'adaptive development evidence on MODEL_SELECT; reader-relative; lower recovery for these readers is not '
                   'protection against every reader'}
    C.write_json(out / 'GRID.json', rec)
    C.unit_stats(reader_fits=stats['reader_fits'], cache_hits_own=stats['cache_hits_own'],
                 cache_hits_source=stats['cache_hits_source'], cache_hits_source_r3=stats['cache_hits_source_r3'],
                 reader_cpu_s=round(stats['reader_cpu_s'], 3))
    return rec


# ---------------------------------------------------------------- invented-data selftest (no real data, no label)
def _invented(n, seed=4242):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n // 2), 2)[:n]
    sex = (rng.random(n) < 0.47).astype(np.int64)
    z = rng.normal(size=(n, 5)); z[:, 1] += 0.8 * sex
    occ_leaky = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    z2 = z.copy(); z2[:, 1] -= 0.8 * sex * 0.9
    occ_safe = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
    t = 1 / (1 + np.exp(-(rng.normal(size=n) + 0.2 * sex))); inc = np.column_stack((1 - t, t))
    return unit, sex, inc, occ_leaky, occ_safe


def _load_source_r3_module():
    import importlib.util
    import sys
    code = C.OHN_RESULTS / 'roles' / 'reader_robustness' / 'code'
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location('rr_split3_source', code / 'rr_split3.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def selftest(n=1500):
    """Invented data only: cells, source-key compatibility (R1/R2 grid cache and R3 extra-split readers), cache accounting,
    winners, paired differences, bootstrap, learning curves, R3 and extra reader seeds, spec loading and refusals, run_bank."""
    from ohn import devreaders as OD
    tmp = Path(tempfile.mkdtemp(prefix='cts_devgrid_selftest_'))
    caches = {'own': tmp / 'own', 'source': tmp / 'source', 'source_r3': tmp / 'source_r3'}
    unit, sex, inc, occ_leaky, occ_safe = _invented(n)
    chk = {}
    # 1. cells identical to the source definitions; R3 = the source extra split; learning curves nested
    chk['roles_R1_R2_equal_source'] = all(np.array_equal(roles_of(unit, r)[k], OD.roles_of(unit, r)[k])
                                          for r in REPS for k in ('FIT', 'VALIDATION', 'SCORE'))
    p3 = C.household_folds(unit, 10, 'OHN1_devgrid_R3_diag')
    r3 = roles_of(unit, 'R3')
    chk['roles_R3_equal_source_extra_split'] = bool(np.array_equal(r3['FIT'], p3 <= 4) and np.array_equal(r3['SCORE'], p3 >= 7))
    chk['roles_household_disjoint'] = all(not np.intersect1d(unit[roles_of(unit, r)[a]], unit[roles_of(unit, r)[b]]).size
                                          for r in ALL_REPS for a, b in (('FIT', 'VALIDATION'), ('FIT', 'SCORE'), ('VALIDATION', 'SCORE')))
    chk['boot_weights_equal_source'] = all(np.array_equal(boot_weights(unit[roles_of(unit, r)['SCORE']], s, r),
                                                          OD.boot_weights(unit[roles_of(unit, r)['SCORE']], s, r))
                                           for r in REPS for s in C.SEEDS)
    fit1 = roles_of(unit, 'R1')['FIT']
    q, h = lc_fit(unit, fit1, 0.25), lc_fit(unit, fit1, 0.5)
    chk['learning_curve_nested_and_equal_source'] = bool((q <= h).all() and (h <= fit1).all() and q.sum() < h.sum() < fit1.sum()
                                                         and all(np.array_equal(lc_fit(unit, fit1, f), OD.lc_fit(unit, fit1, f)) for f in FRACTIONS))
    chk['reader_seeds_equal_source'] = all(reader_seed(s, k) == OD.reader_seed(s, k) for s in C.SEEDS for k in READER_SEEDS)
    try:
        reader_seed(0, 10); chk['reader_seed_k10_refused'] = False
    except ValueError:
        chk['reader_seed_k10_refused'] = True
    # 2. the source grid code writes an entry; this module must read it as a SOURCE hit (0 fits, bitwise equal, nothing written)
    saved = OD.CACHE
    try:
        OD.CACHE = caches['source']
        st_src = {'reader_fits': 0, 'cache_hits': 0, 'reader_cpu_s': 0.0}
        src_rd = OD.cell_readers('occupation', inc, occ_leaky, sex, unit, 0, 'R1', 0, 1.0, st_src)
    finally:
        OD.CACHE = saved
    st = new_stats()
    mine = cell_readers('occupation', inc, occ_leaky, sex, unit, 0, 'R1', 0, 1.0, st, caches=caches)
    chk['source_key_compatible_hit'] = (st['cache_hits_source'] == 1 and st['reader_fits'] == 0
                                        and all(np.array_equal(mine[r][0], src_rd[r][0]) and mine[r][1] == src_rd[r][1] for r in RECIPES))
    chk['source_hit_writes_nothing_own'] = not caches['own'].exists() or not any(caches['own'].iterdir())
    # 3. the source R3 code writes an R3 reader entry; this module must read it at R3 full FIT as a SOURCE_R3 hit
    try:
        RS = _load_source_r3_module()
        rs0 = reader_seed(0, 0)
        r3_src = RS.view_readers('income', inc, occ_leaky, sex, RS.roles_r3(unit), rs0, RS.SPLIT_TAG, cache=caches['source_r3'])
        st = new_stats()
        r3_mine = cell_readers('income', inc, occ_leaky, sex, unit, 0, 'R3', 0, 1.0, st, caches=caches)
        chk['source_r3_key_compatible_hit'] = (st['cache_hits_source_r3'] == 1 and st['reader_fits'] == 0
                                               and all(np.array_equal(r3_mine[r][0], r3_src[r][0]) for r in RECIPES))
        u3 = unit[r3['SCORE']]
        chk['boot_weights_R3_equal_source_extra_split'] = bool(np.array_equal(boot_weights(u3, 0, 'R3'), RS.boot_weights(u3, 0)))
        chk['roles_R3_equal_source_module'] = all(np.array_equal(r3[k], RS.roles_r3(unit)[k]) for k in ('FIT', 'VALIDATION', 'SCORE'))
    except Exception as e:                                     # noqa: BLE001  (reported, fails the selftest)
        chk['source_r3_key_compatible_hit'] = False
        chk['source_r3_error'] = f'{type(e).__name__}: {e}'
    # 4. new fits go to the own cache only, with a checksum; the second call is an own hit; a corrupted entry is refused
    before_src = sorted(p.name for p in caches['source'].iterdir())
    st = new_stats()
    cell_readers('occupation', inc, occ_safe, sex, unit, 0, 'R1', 0, 1.0, st, caches=caches)
    f1 = st['reader_fits']
    cell_readers('occupation', inc, occ_safe, sex, unit, 0, 'R1', 0, 1.0, st, caches=caches)
    chk['new_fits_16_then_own_hit'] = f1 == len(RECIPES) and st['reader_fits'] == f1 and st['cache_hits_own'] == 1
    chk['source_cache_untouched'] = sorted(p.name for p in caches['source'].iterdir()) == before_src
    own_files = sorted(caches['own'].glob('*.npz'))
    bad = tmp / 'corrupt'; bad.mkdir()
    with np.load(own_files[0], allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    d['score__OUT_LR'] = d['score__OUT_LR'] + 1e-6
    with open(bad / own_files[0].name, 'wb') as f:
        np.savez(f, **d)
    try:
        _read_entry(bad / own_files[0].name, len(d['score__OUT_LR'])); chk['corrupted_entry_refused'] = False
    except PermissionError:
        chk['corrupted_entry_refused'] = True
    # 5. evaluate_bank: R1, R2, R3 x k0, k1 x fractions; paired differences; bootstrap; identical income wire -> exactly zero
    bank = {'CAND': {'income': inc, 'occupation': occ_safe}, 'COMP': {'income': inc, 'occupation': occ_leaky}}
    st = new_stats()
    res, st = evaluate_bank(bank, 0, sex, unit, fracs=(1.0, 0.25), stats=st, reps=ALL_REPS, caches=caches)
    p = paired(res, 'CAND', 'COMP')
    full = [c for c in res if '@' not in c]
    chk['cells'] = sorted(res)
    chk['cells_expected'] = sorted(res) == sorted(cell_name(0, r, k, f) for r in ALL_REPS for k in READER_SEEDS for f in (1.0, 0.25))
    chk['pair_benefit_positive_all_full_cells'] = all(p['pair']['cells'][c] > 0 for c in full)
    chk['income_view_identical_wires_zero_diff_zero_se'] = all(abs(v) < 1e-15 for v in p['income']['cells'].values()) and \
        all(v == 0.0 for v in p['income']['cell_se'].values())
    chk['bootstrap_se_positive_pair'] = all(v > 0 for v in p['pair']['cell_se'].values())
    chk['first_bank_call_counts'] = dict(st)
    # 6. extra reader seeds (screen study) and single views; the same bootstrap weights across k (paired across reader seeds)
    st2 = new_stats()
    res2, st2 = evaluate_bank(bank, 0, sex, unit, stats=st2, reps=('R1',), ks=(2,), views=SINGLE_VIEWS, caches=caches)
    chk['extra_seed_cells'] = sorted(res2) == ['S0R1k2'] and set(res2['S0R1k2']['CAND']) == set(SINGLE_VIEWS)
    # CAND income, CAND occupation, COMP occupation are new wires at k2; COMP income is the same wire as CAND income (own hit)
    chk['extra_seed_fits_new'] = st2['reader_fits'] == 3 * len(RECIPES)
    chk['extra_seed_identical_wire_alias_hit'] = st2['cache_hits_own'] == 1 and st2['cache_hits_source'] == 0
    a0 = res['S0R1k0']['CAND']['income']['score_auc'] - res2['S0R1k2']['CAND']['income']['score_auc']
    chk['extra_seed_zero_shift_difference_finite'] = bool(np.isfinite(a0))
    # 7. second call: everything cached (no new fit)
    st3 = new_stats()
    evaluate_bank(bank, 0, sex, unit, fracs=(1.0, 0.25), stats=st3, reps=ALL_REPS, caches=caches)
    chk['second_call_all_cached'] = st3['reader_fits'] == 0 and st3['cache_hits_own'] > 0
    # 8. spec loading: by path + sha256 under an allowed root; refusals
    root = tmp / 'runs'; (root / 'NEW' / 'S0').mkdir(parents=True)
    fpath = root / 'NEW' / 'S0' / 'OCC.npz'
    with open(fpath, 'wb') as f:
        np.savez(f, MODEL_SELECT__occupation=occ_safe, MODEL_SELECT__income=inc, row_id=np.arange(n))
    good = {'type': 'arrays', 'seeds': {'0': {
        'income': {'path': str(fpath), 'key': 'MODEL_SELECT__income', 'sha256': C.sha(fpath)},
        'occupation': {'path': str(fpath), 'key': 'MODEL_SELECT__occupation', 'sha256': C.sha(fpath), 'row_id_key': 'row_id'}}}}
    rel, prov = load_member(good, 0, n=n, row_id=np.arange(n), roots=(root,))
    chk['spec_load_by_path_sha'] = bool(np.array_equal(rel['occupation'], occ_safe) and prov['occupation']['sha256'] == C.sha(fpath))
    refusals = {}
    for label, mutate, kw in (
            ('wrong_sha256', lambda m: m['seeds']['0']['income'].update(sha256='0' * 64), {'roots': (root,)}),
            ('outside_allowed_roots', lambda m: None, {'roots': (tmp / 'elsewhere',)}),
            ('row_id_order_differs', lambda m: None, {'roots': (root,), 'row_id': np.arange(n)[::-1].copy()}),
            ('anchor_for_occupation', lambda m: m['seeds']['0'].update(occupation={'frozen_anchor': 'H3'}), {'roots': (root,)})):
        m = json.loads(json.dumps(good)); mutate(m)
        try:
            load_member(m, 0, n=n, row_id=kw.get('row_id', np.arange(n)), roots=kw['roots'])
            refusals[label] = False
        except (PermissionError, ValueError):
            refusals[label] = True
    bad_occ = occ_safe.copy(); bad_occ[0] = [0.5, 0.5, 0.5, 0.0, 0.0]
    try:
        _check_prob(bad_occ, 5, 'invented', n); refusals['invalid_probabilities'] = False
    except ValueError:
        refusals['invalid_probabilities'] = True
    chk['spec_refusals'] = refusals
    try:
        validate_spec({'bank_id': 'X', 'members': {'A': good}, 'candidates': ['A'], 'comparators': ['B']})
        chk['spec_unknown_comparator_refused'] = False
    except ValueError:
        chk['spec_unknown_comparator_refused'] = True
    # 9. run_bank writes GRID.json / BOOT.npz in a fresh directory (everything from the cache); an existing directory is refused
    out = tmp / 'run' / 'S0'
    spec = {'bank_id': 'SELFTEST', 'members': {}, 'candidates': ['CAND'], 'comparators': ['COMP', 'CAND'], 'fractions': [1.0, 0.25],
            'reps': ['R1', 'R2', 'R3']}
    rec = run_bank(bank, sex, unit, spec, 0, out, caches=caches)
    g = json.loads((out / 'GRID.json').read_text())
    chk['run_bank_written'] = (out / 'BOOT.npz').is_file() and g['paired']['CAND']['COMP']['pair']['cells'] == rec['paired']['CAND']['COMP']['pair']['cells']
    chk['run_bank_no_self_pair'] = 'CAND' not in rec['paired']['CAND']
    chk['run_bank_all_cached'] = rec['stats']['reader_fits'] == 0
    try:
        run_bank(bank, sex, unit, spec, 0, out, caches=caches); chk['run_bank_existing_dir_refused'] = False
    except FileExistsError:
        chk['run_bank_existing_dir_refused'] = True
    chk['pair_benefits'] = {c: round(v, 4) for c, v in p['pair']['cells'].items()}
    keys = [k for k, v in chk.items() if isinstance(v, bool)]
    chk['all_ok'] = all(chk[k] for k in keys) and all(refusals.values())
    chk['checked'] = keys
    return chk


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('manifest')
    sub.add_parser('selftest')
    sub.add_parser('lock')
    r = sub.add_parser('run'); r.add_argument('--spec', required=True); r.add_argument('--seed', type=int, required=True)
    r.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'manifest':
        m = manifest()
        C.write_json(MANIFEST_PATH, m)
        print(json.dumps({'repetitions': m['repetitions'], 'R1_R2_identical_to_source_DEV_GRID': m['R1_R2_identical_to_source_DEV_GRID']}, indent=1))
    elif a.cmd == 'selftest':
        chk = selftest()
        C.write_json(SELFTEST_PATH, {**chk, 'utc': C.now(), 'versions': versions()})
        print(json.dumps(chk, indent=1, default=str))
        if not chk['all_ok']:
            raise SystemExit(1)
    elif a.cmd == 'lock':
        print(json.dumps(make_lock(), indent=1))
    elif a.cmd == 'run':
        rec = run(a.spec, a.seed, a.out)
        print(json.dumps({'bank_id': rec['bank_id'], 'seed': rec['seed'], 'stats': rec['stats'], 'cpu_s': rec['cpu_s']}))


if __name__ == '__main__':
    main()
