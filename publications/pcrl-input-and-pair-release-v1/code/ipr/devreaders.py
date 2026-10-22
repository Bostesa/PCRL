"""Fresh development reader grid of the PCRL input-and-pair release campaign (`pcrl_input_and_pair_release_v1`; EXPLORATORY
DEVELOPMENT evidence on adaptively used MODEL_SELECT records, never confirmation).

Derived with constant changes only from `ddr.devreaders` (dual-design release, publication e30cf1c6b), itself derived from
`frc.devreaders`, `nrm.devreaders`, `tsz.devreaders`, `cts.devreaders` and the source grid `ohn.devreaders`. Changes: grid version
IPR1-devgrid-v1, split tags IPR1_devgrid_R<r>, reader-seed base offset 900 and bootstrap seed 2026102300 (never used by an earlier
grid), this campaign's cache, member types 'ipr' (this campaign's releases), 'ddr', 'frc' and 'nrm' (earlier campaigns' saved
releases, read-only), no screening split. Readers estimate the recorded SEX column (sex 0 = men, sex 1 = women) of
public-use ACS 2019 1-Year PUMS Florida person records from released predictions only: each recipient's float64 probability vector
and argmax decision (income view, occupation view) and the two outputs aligned to the same person (the linked pair). Readers never
see raw fields, hidden scores, codes, cell ids or labels other than SEX. Pool: MODEL_SELECT (8,001 people, 5,306 households). SEX
is read ONLY through `ipr.common.load('MODEL_SELECT', ('features', 'sex'), operation='readers', purpose=...)` (logged); the Census
inputs are dropped right after loading. ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT labels are never touched.

FRESH grid (splits frozen in the registration before any new fit; READER_SPLITS.json):
- repetitions R1, R2 (registered) and R3, R4 (robustness repetitions): MODEL_SELECT households ordered by
  SHA-256('IPR1_devgrid_<R>|<household>') and dealt round-robin into 10 parts; parts 0-4 = FIT (50% of households), 5-6 = VALIDATION
  (20%), 7-9 = SCORE (30%); household-disjoint. New tags, so the splits differ from every earlier campaign's splits (their overlap
  is recorded); MODEL_SELECT itself was used adaptively before;
- reader seeds k = MODEL_SEED_BASE + 900 + 10 x version + k (never equal to a realized earlier grid's reader seed); k0, k1
  registered, k2, k3 for robustness; a cell = (version, repetition, reader seed); 8 registered cells per release;
- randomized releases: one emitted draw per person everywhere (FIT, VALIDATION and SCORE people each contribute the draw of the
  member's draw index); further draws are evaluated as separate Monte Carlo repetitions of the same member, never pooled per person;
- the source 16 recipes (source 8: OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB, LOOKUP; new 8: ET5, ET25, HGB15,
  HGB31, MLP64, MLP6432, CATI01, CATI1) from the locked recipe code (scm_bench.readers, task_conditioned_representation.attacks /
  .inference, ffv.readers / .features; file hashes and library versions must equal the source DEVGRID_CODE_LOCK.json), fitted on
  FIT, oriented and selected on VALIDATION (max oriented VALIDATION AUC, exact ties to the earlier candidate; pair candidates = pair
  readers, then the ignore-occupation (income-view) and ignore-income (occupation-view) readers, source recipes first; near tie <
  0.010 flagged), scored on SCORE; SCORE is never used for selection; the same recipes, sizes and effort for every release;
- household bootstrap of SCORE (B = 400, seed 2026102300 + 100 x version + repetition index R1 0 / R2 1 / R3 2 / R4 3; readers and
  orientations fixed); every release in a cell shares the same weights, so differences of draws are paired;
- reader cache (this campaign only; no source cache is read): key SHA-256('IPR1-devgrid-v1|<rep>|<reader seed>|<FIT fraction>|
  <wire hash>|<SEX array hash>'); an exact alias (identical released wire) reuses the same readers and is counted as a cache hit.

Bank spec (JSON): {"bank_id", "purpose", "members": {name: member}, "candidates": [...], "comparators": [...], optional "reps",
"reader_seeds", "fractions"}. A member is
  a release of this campaign:  {"type": "ipr", "family": F, "config": C[, "draw": d | "rep"][, "draw_offset": o]}
                               (ipr.release.load, hash-checked; {"type": "nrm", ...} loads a saved source release, read-only; for a randomized release the default "rep" reads instance
                               REP_INSTANCE[repetition] (R1 0, R2 1, R3 6, R4 7) in every cell, plus "draw_offset" for Monte Carlo
                               replicate members (+2, +4; R1/R2 only); a fixed "draw" d reads instance d everywhere),
  the previous witness:        {"type": "witness"}  (A1_PO_16 S<v> occupation + H3 income S<v>),
  a frozen source release (loaded by ohn.devreaders.load_member from the admitted copies):
      {"type": "reference", "name": NAME} | {"type": "occ", "family": F, "config": C, "anchor": "H3" | "TW"} |
      {"type": "complete", "family": F, "config": C}
  or a release given task by task, per version:
      {"type": "arrays", "seeds": {"0": {"income": SRC, "occupation": SRC}, "1": {...}}}
  with SRC one of {"path", "key", "sha256"[, "row_id_key"]} (a file under this campaign's runs or the admitted source copies),
  {"frozen_anchor": "H3" | "TW"}, {"frozen_reference": NAME}, {"frozen_occ": {"family": F, "config": C}}.

    python3 -m ipr.devreaders selftest                                  invented data only (no real data, no label)
    python3 -m ipr.devreaders manifest                                  label-free grid manifest (writes READER_SPLITS.json)
    python3 -m ipr.devreaders lock                                      write-once DEVGRID_CODE_LOCK.json (commit it before run)
    python3 -m ipr.devreaders run --spec BANK.json --seed V [--out DIR] (through ipr.jobqueue, with the locked interpreter)
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
from ohn import common as OC
from . import common as C
from . import readerextra as RX                     # campaign copy of nrm.readerextra (its relative import resolves to ipr.devreaders)

GRID_VERSION = 'IPR1-devgrid-v1'            # this campaign's fresh grid version string: part of the cache key
SCHEMA = 'IPR1-devgrid-v1'                  # this module's result schema
SRC8 = tuple(n for n, _ in SLATE) + ('LOOKUP',)
RECIPES = SRC8 + NEW8
REPS = ('R1', 'R2')                         # registered repetitions
ALL_REPS = ('R1', 'R2', 'R3', 'R4')         # R3, R4: robustness repetitions (repeated household reader splits)
REP_TAG = {r: f'IPR1_devgrid_{r}' for r in ALL_REPS}
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
READER_SEEDS = (0, 1)                       # registered; k2, k3 for robustness (MAX_K bounds every index)
MAX_K = 9                                   # k <= 9 never reaches the next release seed's reader seeds (10 x release seed)
SEED_BASE_OFFSET = 900                      # new base (ohn 300, cts 400, tsz 500, nrm 600, frc 700, ddr 800): never an earlier campaign's reader seed
FRACTIONS = (0.25, 0.5, 1.0)
LC_TAG = 'IPR1_devgrid_LC'
VIEWS = ('income', 'occupation', 'pair')
SINGLE_VIEWS = ('income', 'occupation')
B_BOOT = 400
BOOT_SEED = 2026102300
NEAR_TIE = 0.010
SOURCE_R3_VERSION = 'OHN1-rr-split3-v1'
CACHE = C.RUNS / 'devgrid_cache'                              # this campaign's cache (new entries only here)
SOURCE_CACHE = None                                          # fresh readers: no source cache is read
SOURCE_R3_CACHE = None
OHN_RESULTS = C.ROOT / 'results' / 'pcrl_occupation_headroom_overnight_v1'
SOURCE_LOCK = OHN_RESULTS / 'DEVGRID_CODE_LOCK.json'
LOCK_PATH = C.RESULTS / 'DEVGRID_CODE_LOCK.json'
MANIFEST_PATH = C.RESULTS / 'READER_SPLITS.json'
SELFTEST_PATH = C.RESULTS / 'roles' / 'coordinator' / 'checks' / 'DEVGRID_SELFTEST.json'
RECIPE_FILES = ('ffv/readers.py', 'ffv/features.py', 'scm_bench/readers.py', 'task_conditioned_representation/attacks.py',
                'task_conditioned_representation/inference.py')
EXTRAS = True                                       # score fingerprints, NOLOOKUP winner, TABLE_CV enhanced winner (descriptive)
LOCK_FILES = ('ipr/devreaders.py', 'ipr/release.py', 'ipr/readerextra.py', 'ddr/release.py', 'ddr/common.py', 'nrm/release.py', 'frc/release.py', 'frc/common.py', 'tsz/devreaders.py', 'ohn/devreaders.py',
              'ohn/devrun.py', 'ohn/common.py') + RECIPE_FILES
RELEASE_ROOTS = (C.RUNS, C.DDR_RUNS, C.NRM_RUNS, C.FRC_RUNS, OC.RUNS, C.ROOT / '.private' / 'pcrl_income_headroom_successor_v1')
N_MODEL_SELECT = 8001


def default_caches():
    return {'own': CACHE, 'source': SOURCE_CACHE, 'source_r3': SOURCE_R3_CACHE}


def new_stats():
    return {'reader_fits': 0, 'cache_hits_own': 0, 'cache_hits_source': 0, 'cache_hits_source_r3': 0, 'reader_cpu_s': 0.0}


# ---------------------------------------------------------------- cells (label-free)
def roles_of(unit, rep):
    """FIT / VALIDATION / SCORE masks of one repetition (fresh tags IPR1_devgrid_R1 .. _R4)."""
    part = OC.household_folds(unit, 10, REP_TAG[rep])
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
    return MODEL_SEED_BASE + SEED_BASE_OFFSET + 10 * int(release_seed) + int(k)


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
    """Fresh key of this campaign's grid (no source cache is read)."""
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
    This campaign's cache (exact aliases), then new fits (own cache only); no source cache is read."""
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
    if rep in REPS and caches.get('source'):
        src = Path(caches['source']) / f'{cache_key(rep, rs, frac, wh, sx)}.npz'
        if src.is_file():
            stats['cache_hits_source'] += 1
            return _read_entry(src, n_sc)
    elif rep == 'R3' and frac == 1.0 and caches.get('source_r3'):
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


REP_INSTANCE = {'R1': 0, 'R2': 1, 'R3': 6, 'R4': 7}      # declared (DECLARATIONS_V1; reviewer's ruling): R3/R4 never reuse the
                                                         # R1/R2 Monte Carlo replicate instances 2-5


def draw_for(rule, rep):
    """The emitted draw (release instance) a cell of repetition `rep` reads from a randomized release. Rule {'draw': 'rep'} (the
    registered mapping REP_INSTANCE: R1 -> 0, R2 -> 1, R3 -> 6, R4 -> 7; both reader seeds of a cell read the same instance,
    separating reader-seed from Monte Carlo variation) plus an optional 'draw_offset' (Monte Carlo replicate members, +2 and +4,
    registered repetitions only), or {'draw': d} (one fixed instance). Every person contributes exactly that one draw (FIT,
    VALIDATION and SCORE)."""
    if rule.get('draw', 'rep') == 'rep':
        base = REP_INSTANCE[rep]
        if int(rule.get('draw_offset', 0)) and rep not in REPS:
            raise ValueError('Monte Carlo replicate offsets apply to the registered repetitions R1 and R2 only')
    else:
        base = int(rule['draw'])
    return base + int(rule.get('draw_offset', 0))


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
                    if inc.ndim == 3 or occ.ndim == 3:            # randomized release: this cell's ONE draw per person
                        r_draw = draw_for(rel.get('_draw_rule', {}), rep)
                        inc = inc[r_draw] if inc.ndim == 3 else inc
                        occ = occ[r_draw] if occ.ndim == 3 else occ
                    rd = {v: cell_readers(v, inc, occ, sex, unit, rel_seed, rep, k, frac, stats, sx, caches) for v in views}
                    w = winners(rd.get('income'), rd.get('occupation'), rd.get('pair'))
                    entry = {}
                    for view in views:
                        s = _score_of(w[view]['reader_key'], rd)
                        entry[view] = {'winner': w[view]['candidate'], 'origin': w[view]['origin'],
                                       'val_auc': w[view]['attack_validation_auc'], 'gap_to_runner_up': w[view]['gap_to_runner_up'],
                                       'near_tie': w[view]['AV_NEAR_TIE'], 'score_auc': float(auc(y, s)), 'boot': wauc_draws(y, s, W)}
                    if EXTRAS:                    # reader-robustness extras: descriptive only, never a registered winner or decision
                        ex, eb = RX.cell_extras(rd, inc, occ, sex, unit, rep, frac, W, views)
                        for view in views:
                            entry[view]['extras'] = ex[view]
                            entry[view]['extras_boot'] = eb[view]
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
    return {cell: {n: {v: {k: x for k, x in e.items() if k not in ('boot', 'extras_boot')} for v, e in ent.items()}
                   for n, ent in cres.items()}
            for cell, cres in res.items()}


# ---------------------------------------------------------------- releases (frozen source releases; new arrays by path + sha256)
FROZEN_TYPES = ('reference', 'occ', 'complete')


def _check_prob(p, k, what, n=None):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    if (p.ndim != 2 or p.shape[1] != k or (n is not None and p.shape[0] != n) or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0)):
        raise ValueError(f'{what}: not a valid ({n}, {k}) probability-vector release')
    return p


def _check_stack(p, k, what, n=None):
    """A randomized release: (R, n, k) stack of valid one-draw-per-person probability vectors."""
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    if p.ndim != 3:
        raise ValueError(f'{what}: a randomized release is an (R, n, k) stack')
    for r in range(p.shape[0]):
        _check_prob(p[r], k, f'{what} draw {r}', n)
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
    elif t in ('ipr', 'ddr', 'nrm', 'frc'):
        if t == 'ipr':
            from . import release as RL
        elif t == 'ddr':
            from ddr import release as RL                     # the dual-design release campaign's saved releases (read-only)
        elif t == 'frc':
            from frc import release as RL                     # the finite-routing campaign's saved releases (read-only)
        else:
            from nrm import release as RL                     # the new-release-mechanisms campaign's saved releases (read-only)
        arr, rec = RL.load(member['family'], member['config'], seed, roles=('MODEL_SELECT',))
        ms = arr['MODEL_SELECT']
        rule = {'draw': member.get('draw', 'rep'), 'draw_offset': int(member.get('draw_offset', 0))}
        if rec.get('randomized'):                 # keep the stacks; each cell reads its own instance (draw_for)
            rel = {task: ms[task] for task in C.TASKS}
            for task in C.TASKS:
                R = rel[task].shape[0]
                reps_used = REPS if int(rule.get('draw_offset', 0)) else ALL_REPS
                if max(draw_for(rule, rp) for rp in reps_used) >= R:
                    raise ValueError(f'{member}: the draw rule needs more than the {R} stored instances')
        else:
            rel = {task: ms[task] for task in C.TASKS}
        prov = {f'{t}_member': member, 'release_sha256': rec['release_sha256'], 'kind': rec['kind'],
                'randomized': bool(rec.get('randomized')), 'draw_rule': rule if rec.get('randomized') else None}
        if rec.get('randomized'):
            rel = {**{t: _check_stack(rel[t], C.K[t], f'{t} release', n) for t in C.TASKS}, '_draw_rule': rule}
            prov['wire_sha256'] = {t: [wire_sha(rel[t][r]) for r in range(rel[t].shape[0])] for t in C.TASKS}
            return rel, prov
    elif t == 'witness':
        from tsz.evaluate import source_a1, h3_income
        rel, prov = {'income': h3_income(seed), 'occupation': source_a1(seed)}, {'witness': 'A1_PO_16 S<v> + H3 income S<v>'}
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
        raise ValueError('R1 and R2 are always part of the grid')
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
            w = pv['wire_sha256'][task]
            by.setdefault(tuple(w) if isinstance(w, list) else w, []).append(name)
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
    rec = {'schema': 'IPR1-devgrid-code-lock-v1', 'utc': C.now(), 'grid_version': GRID_VERSION, 'result_schema': SCHEMA,
           'versions': versions(), 'files_sha256': {f: C.sha(C.ROOT / f) for f in LOCK_FILES},
           'source_lock_sha256': C.sha(SOURCE_LOCK), 'manifest_sha256': C.sha(MANIFEST_PATH) if MANIFEST_PATH.is_file() else None,
           'cache_key': f'SHA-256("{GRID_VERSION}|<rep>|<reader seed>|<FIT fraction>|<wire hash>|<SEX array hash>") (fresh: no source cache)'}
    C.write_json_x(LOCK_PATH, rec)
    return rec


def manifest():
    """Label-free grid manifest (MODEL_SELECT household ids only): the frozen reader splits of this campaign, and their overlap with
    the splits earlier campaigns used (disclosure only)."""
    unit = C.features('MODEL_SELECT')['unit']

    def desc(m):
        hh = np.unique(unit[m])
        return {'people': int(m.sum()), 'households': int(len(hh)),
                'household_set_sha256': hashlib.sha256(','.join(map(str, hh)).encode()).hexdigest()}
    roles = ('FIT', 'VALIDATION', 'SCORE')
    reps = {}
    for rep in ALL_REPS:
        r = roles_of(unit, rep)
        reps[rep] = {k: desc(v) for k, v in r.items()}
        reps[rep]['tag'] = REP_TAG[rep]
        reps[rep]['disjoint'] = bool(not np.intersect1d(unit[r['FIT']], unit[r['VALIDATION']]).size
                                     and not np.intersect1d(unit[r['FIT']], unit[r['SCORE']]).size
                                     and not np.intersect1d(unit[r['VALIDATION']], unit[r['SCORE']]).size)
    hh = lambda m: np.unique(unit[m])
    overlap = {f'{a}_vs_{b}_{role}': int(np.intersect1d(hh(roles_of(unit, a)[role]), hh(roles_of(unit, b)[role])).size)
               for i, a in enumerate(ALL_REPS) for b in ALL_REPS[i + 1:] for role in roles}
    earlier = {}
    for rep in ALL_REPS:
        mine = roles_of(unit, rep)
        for camp in ('OHN1', 'TSZ1', 'NRM1', 'FRC1', 'DDR1'):
            for old in ('R1', 'R2'):
                part = OC.household_folds(unit, 10, f'{camp}_devgrid_{old}')
                theirs = {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}
                for role in roles:
                    earlier[f'{rep}_{role}_vs_{camp}_{old}_{role}'] = int(np.intersect1d(hh(mine[role]), hh(theirs[role])).size)
    return {'schema': SCHEMA, 'pool': 'MODEL_SELECT (8,001 people, 5,306 households); used adaptively by earlier campaigns',
            'repetitions': reps, 'registered_repetitions': list(REPS), 'robustness_repetitions': [r for r in ALL_REPS if r not in REPS],
            'household_overlap_between_repetitions': overlap,
            'household_overlap_with_earlier_grid_splits': earlier, 'screening_split': None,
            'reader_seeds': {f'V{s}k{k}': reader_seed(s, k) for s in C.SEEDS for k in range(4)},
            'registered_reader_seed_indices': list(READER_SEEDS), 'robustness_reader_seed_indices': [2, 3],
            'cells': [cell_name(s, rep, k) for s in C.SEEDS for rep in REPS for k in READER_SEEDS],
            'recipes': {'source8': list(SRC8), 'new8': list(NEW8)}, 'versions_at_manifest': versions(),
            'selection': 'max oriented VALIDATION AUC; exact ties to the earlier candidate (source first, pair then ignore-occupation '
                         'then ignore-income); near tie < 0.010 flagged; SCORE never used for selection',
            'bootstrap': {'B': B_BOOT, 'seed': f'{BOOT_SEED} + 100 x version + repetition index {REP_INDEX}',
                          'unit': 'SCORE households, readers fixed'},
            'tags': {'repetitions': REP_TAG},
            'cache': {'key': f'SHA-256("{GRID_VERSION}|<rep>|<reader seed>|<FIT fraction>|<wire hash>|<SEX array hash>")',
                      'own': str(CACHE.relative_to(C.ROOT)), 'source': 'none (fresh readers)'},
            'exclusion': 'MODEL_SELECT households never enter the fitting of release models, calibrators or correction parameters'}


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
    ks = tuple(int(k) for k in spec.get('reader_seeds', READER_SEEDS))
    stats = new_stats()
    t0 = time.process_time()
    res, stats = evaluate_bank(bank, seed, sex, unit, fracs, stats, reps=reps, ks=ks, caches=caches)
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
    bx = {f'{cell}__{n}__{v}__{k}': b for cell, cres in res.items() for n, ent in cres.items() for v, e in ent.items()
          for k, b in e.get('extras_boot', {}).items()}
    if bx:
        with open(out / 'BOOT_EXTRAS.npz', 'wb') as f:
            np.savez(f, **bx)
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


def selftest(n=1500):
    """Invented data only: cells, fresh seeds, cache accounting, winners, paired differences, bootstrap, effective reader seeds,
    spec checks and refusals."""
    import tempfile
    chk = {}
    unit, sex, inc, occ_leaky, occ_safe = _invented(n)
    tmp = Path(tempfile.mkdtemp())
    caches = {'own': tmp / 'own', 'source': None, 'source_r3': None}
    r1, r2 = roles_of(unit, 'R1'), roles_of(unit, 'R2')
    chk['roles_partition'] = all(bool(np.array_equal(r['FIT'] | r['VALIDATION'] | r['SCORE'], np.ones(n, bool)))
                                 and not (r['FIT'] & r['VALIDATION']).any() and not (r['FIT'] & r['SCORE']).any() for r in (r1, r2))
    chk['roles_household_disjoint'] = all(not np.intersect1d(unit[r['FIT']], unit[r['SCORE']]).size for r in (r1, r2))
    chk['reps_differ'] = not np.array_equal(r1['SCORE'], r2['SCORE'])
    seeds = [reader_seed(v, k) for v in C.SEEDS for k in READER_SEEDS]
    old = [MODEL_SEED_BASE + b + 10 * v + k for b in (300, 400, 500, 600) for v in C.SEEDS for k in range(10)]
    chk['reader_seeds_distinct_and_new'] = len(set(seeds)) == len(seeds) and not set(seeds) & set(old)
    st = new_stats()
    bank = {'A': {'income': inc, 'occupation': occ_leaky}, 'B': {'income': inc, 'occupation': occ_safe},
            'A2': {'income': inc.copy(), 'occupation': occ_leaky.copy()}}
    res, st = evaluate_bank(bank, 0, sex, unit, stats=st, caches=caches)
    chk['cells'] = sorted(res) == sorted(cell_name(0, rep, k) for rep in REPS for k in READER_SEEDS)
    pr = paired(res, 'B', 'A')
    chk['less_leaky_occupation_view_benefits'] = all(v > 0 for v in pr['pair']['cells'].values())
    chk['identical_income_wire_zero_difference'] = all(v == 0.0 for v in pr['income']['cells'].values())
    pa = paired(res, 'A2', 'A')
    chk['exact_alias_zero_difference'] = all(v == 0.0 for view in VIEWS for v in pa[view]['cells'].values())
    chk['alias_counted_as_cache_hits'] = st['cache_hits_own'] >= 3 * len(REPS) * len(READER_SEEDS)
    chk['bootstrap_draws'] = all(res[c]['A']['pair']['boot'].shape == (B_BOOT,) for c in res)
    # effective reader seeds: which recipes give different SCORE outputs under k0 and k1 (same wire, same cell otherwise)
    sx = C.arr_sha(np.asarray(sex))
    st2 = new_stats()
    e0 = cell_readers('pair', inc, occ_leaky, sex, unit, 0, 'R1', 0, 1.0, st2, sx, caches)
    e1 = cell_readers('pair', inc, occ_leaky, sex, unit, 0, 'R1', 1, 1.0, st2, sx, caches)
    differs = {name: bool(not np.array_equal(e0[name][0], e1[name][0])) for name in RECIPES}
    chk['seed_sensitive_recipes'] = sorted(n for n, d in differs.items() if d)
    chk['some_recipes_seed_sensitive'] = len(chk['seed_sensitive_recipes']) > 0
    try:
        validate_spec({'bank_id': 'x', 'members': {'A': {}}, 'candidates': ['A'], 'comparators': ['Z']})
        chk['spec_missing_member_refused'] = False
    except ValueError:
        chk['spec_missing_member_refused'] = True
    try:
        _check_prob(np.full((3, 5), 0.3), 5, 'bad')
        chk['invalid_probability_refused'] = False
    except ValueError:
        chk['invalid_probability_refused'] = True
    chk['stats'] = st
    chk['all_ok'] = all(v for k, v in chk.items() if isinstance(v, bool))
    return chk


def selftest_draws():
    """Invented data, no reader fit: the per-cell draw rule of randomized releases and the stack checks."""
    chk = {'rep_rule': [draw_for({'draw': 'rep'}, r) for r in ALL_REPS] == [0, 1, 6, 7],
           'offset_rule': [draw_for({'draw': 'rep', 'draw_offset': 2}, r) for r in REPS] == [2, 3],
           'fixed_rule': draw_for({'draw': 5}, 'R2') == 5}
    try:
        draw_for({'draw': 'rep', 'draw_offset': 2}, 'R3'); chk['offset_refused_on_robustness_reps'] = False
    except ValueError:
        chk['offset_refused_on_robustness_reps'] = True
    rng = np.random.default_rng(3)
    st = rng.dirichlet(np.ones(5), (8, 20))
    chk['stack_ok'] = _check_stack(st, 5, 'x', 20).shape == (8, 20, 5)
    try:
        _check_stack(st[0], 5, 'x', 20); chk['stack_needs_3d'] = False
    except ValueError:
        chk['stack_needs_3d'] = True
    bad = st.copy(); bad[3, 0, 0] += 0.5
    try:
        _check_stack(bad, 5, 'x', 20); chk['stack_rejects_invalid_draw'] = False
    except ValueError:
        chk['stack_rejects_invalid_draw'] = True
    chk['aliases_hash_lists'] = aliases({'a': {'wire_sha256': {'income': ['x', 'y'], 'occupation': 'o'}},
                                        'b': {'wire_sha256': {'income': ['x', 'y'], 'occupation': 'p'}}})['income'] == [['a', 'b']]
    chk['all_ok'] = all(chk.values())
    return chk


def timing(n=8000):
    """Invented data at MODEL_SELECT size (no real data, no label): CPU of the 16 recipes for one release in one cell, per view and
    per recipe (planning the grid's CPU budget; run through the queue so the one-thread limit applies). n is even because _invented
    builds two-person households (n = 8,001 failed in the source campaign: unit had 8,000 rows)."""
    import tempfile
    unit, sex, inc, occ, _ = _invented(n, seed=777)
    r = roles_of(unit, 'R1')
    out = {}
    for view in VIEWS:
        per = {}
        for name in RECIPES:
            t0 = time.process_time()
            _fit_one(name, view, inc, occ, sex, r['FIT'], r['VALIDATION'], reader_seed(0, 0))
            per[name] = round(time.process_time() - t0, 3)
        out[view] = {'per_recipe_cpu_s': per, 'total_cpu_s': round(sum(per.values()), 2)}
    out['one_release_one_cell_cpu_s'] = round(sum(v['total_cpu_s'] for v in out.values() if isinstance(v, dict)), 2)
    out['n'] = n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('manifest')
    sub.add_parser('selftest')
    sub.add_parser('selftest_draws')
    sub.add_parser('timing')
    sub.add_parser('lock')
    r = sub.add_parser('run'); r.add_argument('--spec', required=True); r.add_argument('--seed', type=int, required=True)
    r.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'manifest':
        m = manifest()
        C.write_json(MANIFEST_PATH, m)
        print(json.dumps({'repetitions': m['repetitions'], 'overlap_with_earlier': m['household_overlap_with_earlier_grid_splits']}, indent=1))
    elif a.cmd == 'selftest':
        chk = selftest()
        C.write_json(SELFTEST_PATH, {**chk, 'utc': C.now(), 'versions': versions()})
        print(json.dumps(chk, indent=1, default=str))
        if not chk['all_ok']:
            raise SystemExit(1)
    elif a.cmd == 'selftest_draws':
        chk = selftest_draws()
        print(json.dumps(chk))
        if not chk['all_ok']:
            raise SystemExit(1)
    elif a.cmd == 'timing':
        res = timing()
        C.write_json(C.RESULTS / 'roles' / 'coordinator' / 'checks' / 'DEVGRID_TIMING.json', {**res, 'utc': C.now(), 'versions': versions()})
        print(json.dumps(res, indent=1))
    elif a.cmd == 'lock':
        print(json.dumps(make_lock(), indent=1))
    elif a.cmd == 'run':
        rec = run(a.spec, a.seed, a.out)
        print(json.dumps({'bank_id': rec['bank_id'], 'seed': rec['seed'], 'stats': rec['stats'], 'cpu_s': rec['cpu_s']}))


if __name__ == '__main__':
    main()
