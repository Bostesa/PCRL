"""Development reader grid of the overnight occupation-headroom campaign (DEV_GRID.json; adaptive development evidence only).

Readers estimate the recorded SEX column of public-use ACS 2019 Florida person records from released predictions only (each
recipient's float64 probability vector and argmax decision; the linked pair). Pool: MODEL_SELECT (8,001 people, 5,306 households;
labels read and logged through ohn.common.load_role). Never ATTACK_FIT, ATTACK_VALIDATION or DEV_REPORT.

Grid (frozen in DEV_GRID.json before any candidate is scored):
- repetitions R1, R2: MODEL_SELECT households ordered by SHA-256('OHN1_devgrid_<R>|<household>') and dealt round-robin into 10
  parts; parts 0-4 = FIT (50% of households), 5-6 = VALIDATION (20%), 7-9 = SCORE (30%); household-disjoint, no person in two roles;
- reader seeds k0, k1: model seed MODEL_SEED_BASE + 300 + 10 * release_seed + k (seed-invariant recipes give identical fits);
- a cell = (release seed, repetition, reader seed); 8 cells per configuration over seeds 0 and 1;
- recipes: the validated enhanced slate (source 8: OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB, LOOKUP; new 8: ET5,
  ET25, HGB15, HGB31, MLP64, MLP6432, CATI01, CATI1), fitted on FIT, oriented and selected on VALIDATION (max oriented VALIDATION
  AUC, exact ties to the earlier candidate, near tie < 0.010 flagged), scored on SCORE; pair candidates = pair readers + the
  ignore-occupation (income-view) and ignore-income (occupation-view) readers of the same release;
- every release in a bank gets its own readers in every cell (comparators refitted independently); identical wires are cached;
- paired benefit per cell: comparator winner SCORE AUC minus candidate winner SCORE AUC (pair; positive = candidate reads lower);
  candidate minus comparator for the income and occupation views (recipient conditions); household-bootstrap SE on SCORE
  (B = 400 resamples of SCORE households, readers and orientations fixed, seed 2026101806);
- nested learning curves (diagnostic only): FIT households also ordered by SHA-256('OHN1_devgrid_LC|<household>') into 4 parts;
  fractions 1/4 (part 0), 1/2 (parts 0-1), 1 (all), VALIDATION and SCORE unchanged.
Readers see only released probability vectors and decisions; never raw inputs, hidden scores, labels other than SEX, or codes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from scm_bench.readers import SLATE, kinds, lookup_keys, fit_lookup, predict_lookup
from task_conditioned_representation.attacks import MODEL_SEED_BASE, fit_reader, predict_reader, orientation, orient
from task_conditioned_representation.inference import auc
from ffv.readers import NEW8, design_for, fit_new, predict_new, pick, versions
from . import common as C

GRID_VERSION = 'OHN1-devgrid-v1'
SRC8 = tuple(n for n, _ in SLATE) + ('LOOKUP',)
REPS = ('R1', 'R2')
READER_SEEDS = (0, 1)
FRACTIONS = (0.25, 0.5, 1.0)
VIEWS = ('income', 'occupation', 'pair')
B_BOOT = 400
BOOT_SEED = 2026101806
NEAR_TIE = 0.010
CACHE = C.RUNS / 'devgrid_cache'
GRID_PATH = C.RESULTS / 'DEV_GRID.json'


def roles_of(unit, rep):
    part = C.household_folds(unit, 10, f'OHN1_devgrid_{rep}')
    return {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}


def lc_fit(unit, fit_mask, frac):
    if frac == 1.0:
        return fit_mask
    part = C.household_folds(unit, 4, 'OHN1_devgrid_LC')
    keep = part == 0 if frac == 0.25 else part <= 1
    return fit_mask & keep


def reader_seed(release_seed, k):
    return MODEL_SEED_BASE + 300 + 10 * int(release_seed) + int(k)


def manifest():
    """DEV_GRID.json content (label-free: household ids of MODEL_SELECT only)."""
    F = C.load_role('MODEL_SELECT', ('features',))
    unit = F['unit']
    reps = {}
    for rep in REPS:
        r = roles_of(unit, rep)
        reps[rep] = {k: {'people': int(v.sum()), 'households': int(len(np.unique(unit[v]))),
                         'household_set_sha256': hashlib.sha256(','.join(map(str, np.unique(unit[v]))).encode()).hexdigest()}
                     for k, v in r.items()}
        reps[rep]['disjoint'] = bool(not np.intersect1d(unit[r['FIT']], unit[r['VALIDATION']]).size
                                     and not np.intersect1d(unit[r['FIT']], unit[r['SCORE']]).size
                                     and not np.intersect1d(unit[r['VALIDATION']], unit[r['SCORE']]).size)
        reps[rep]['learning_curve_fit_people'] = {str(f): int(lc_fit(unit, r['FIT'], f).sum()) for f in FRACTIONS}
    overlap = {f'{a}_vs_{b}_{role}': int(np.intersect1d(np.unique(unit[roles_of(unit, a)[role]]), np.unique(unit[roles_of(unit, b)[role]])).size)
               for a, b in (('R1', 'R2'),) for role in ('FIT', 'VALIDATION', 'SCORE')}
    return {'schema': GRID_VERSION, 'pool': 'MODEL_SELECT (8,001 people, 5,306 households)', 'repetitions': reps,
            'household_overlap_between_repetitions': overlap,
            'reader_seeds': {f'S{s}k{k}': reader_seed(s, k) for s in C.SEEDS for k in READER_SEEDS},
            'cells': [f'S{s}{rep}k{k}' for s in C.SEEDS for rep in REPS for k in READER_SEEDS],
            'recipes': {'source8': list(SRC8), 'new8': list(NEW8)}, 'versions_at_manifest': versions(),
            'selection': 'max oriented VALIDATION AUC; exact ties to the earlier candidate (source first, pair then ignore-occupation '
                         'then ignore-income); near tie < 0.010 flagged; SCORE never used for selection',
            'bootstrap': {'B': B_BOOT, 'seed': BOOT_SEED, 'unit': 'SCORE households, readers fixed'},
            'learning_curve_fractions': list(FRACTIONS), 'tags': {'repetition': 'OHN1_devgrid_<R>', 'learning_curve': 'OHN1_devgrid_LC'},
            'crossfit': {'tag': C.CROSSFIT_TAG, 'k': C.CROSSFIT_K, 'pool': 'MODEL_FIT households'}}


# ---------------------------------------------------------------- reader fitting with caching
def _wire_hash(view, inc, occ):
    h = hashlib.sha256(view.encode())
    for a in ((inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


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


def cell_readers(view, inc, occ, sex, unit, rel_seed, rep, k, frac, stats):
    """All 16 recipes for one view of one wire in one cell; cached by wire hash. Returns {recipe: (scores on SCORE rows, info)}."""
    r = roles_of(unit, rep)
    fit = lc_fit(unit, r['FIT'], frac); val = r['VALIDATION']; sc = r['SCORE']
    rs = reader_seed(rel_seed, k)
    key = hashlib.sha256(f"{GRID_VERSION}|{rep}|{rs}|{frac}|{_wire_hash(view, inc, occ)}|{C.arr_sha(np.asarray(sex))}".encode()).hexdigest()
    p = CACHE / f'{key}.npz'
    if p.is_file():
        with np.load(p, allow_pickle=False) as z:
            info = json.loads(str(z['info']))
            stats['cache_hits'] += 1
            return {n: (z[f'score__{n}'], info[n]) for n in SRC8 + NEW8}
    out, info = {}, {}
    t0 = time.process_time()
    for name in SRC8 + NEW8:
        s, inf = _fit_one(name, view, inc, occ, sex, fit, val, rs)
        out[name] = (s[sc], inf); info[name] = inf
        stats['reader_fits'] += 1
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f'.{key}.{os.getpid()}.npz'
    with open(tmp, 'wb') as f:
        np.savez(f, info=json.dumps(info), **{f'score__{n}': out[n][0] for n in out})
    os.replace(tmp, p)
    stats['reader_cpu_s'] += time.process_time() - t0
    return out


def winners(rd_inc, rd_occ, rd_pair):
    """Enhanced winners per view from cached readers (VALIDATION only)."""
    w = {}
    for view, rd in (('income', rd_inc), ('occupation', rd_occ)):
        w[view] = pick([(n, rd[n][1]['val_auc'], f'{view}__{n}', 'source' if n in SRC8 else 'new') for n in SRC8 + NEW8])
    cands = []
    for group in (SRC8, NEW8):
        org = 'source' if group is SRC8 else 'new'
        cands += [(n, rd_pair[n][1]['val_auc'], f'pair__{n}', org) for n in group]
        cands += [(f'IGNORE_occupation__{n}', rd_inc[n][1]['val_auc'], f'income__{n}', org) for n in group]
        cands += [(f'IGNORE_income__{n}', rd_occ[n][1]['val_auc'], f'occupation__{n}', org) for n in group]
    w['pair'] = pick(cands)
    return w


def _score_of(key, rd_inc, rd_occ, rd_pair):
    view, name = key.split('__', 1)
    return {'income': rd_inc, 'occupation': rd_occ, 'pair': rd_pair}[view][name][0]


def wauc_draws(y, s, W):
    """AUC of fixed scores s under household-bootstrap weights W (B x n)."""
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
    rng = np.random.default_rng(BOOT_SEED + 100 * int(rel_seed) + (0 if rep == 'R1' else 1))
    W = np.empty((B_BOOT, len(unit_sc)))
    for b in range(B_BOOT):
        cnt = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))
        W[b] = cnt[inv]
    return W


def evaluate_bank(bank, rel_seed, sex, unit, fracs=(1.0,), stats=None):
    """bank: {name: {'income': (n, 2), 'occupation': (n, 5)}} on MODEL_SELECT rows. Returns per cell and fraction: every release's
    winners (VALIDATION) with SCORE AUCs, plus the bootstrap draws of each winner's SCORE AUC (for paired differences)."""
    stats = stats if stats is not None else {'reader_fits': 0, 'cache_hits': 0, 'reader_cpu_s': 0.0}
    res = {}
    for rep in REPS:
        sc = roles_of(unit, rep)['SCORE']; y = sex[sc]
        W = boot_weights(unit[sc], rel_seed, rep)
        for k in READER_SEEDS:
            for frac in fracs:
                cell = f'S{rel_seed}{rep}k{k}' + ('' if frac == 1.0 else f'@{frac}')
                cres = {}
                for name, rel in bank.items():
                    inc, occ = rel['income'], rel['occupation']
                    rd = {v: cell_readers(v, inc, occ, sex, unit, rel_seed, rep, k, frac, stats) for v in VIEWS}
                    w = winners(rd['income'], rd['occupation'], rd['pair'])
                    entry = {}
                    for view in VIEWS:
                        s = _score_of(w[view]['reader_key'], rd['income'], rd['occupation'], rd['pair'])
                        entry[view] = {'winner': w[view]['candidate'], 'origin': w[view]['origin'], 'val_auc': w[view]['attack_validation_auc'],
                                       'gap_to_runner_up': w[view]['gap_to_runner_up'], 'near_tie': w[view]['AV_NEAR_TIE'],
                                       'score_auc': float(auc(y, s)), 'boot': wauc_draws(y, s, W)}
                    cres[name] = entry
                res[cell] = cres
    return res, stats


def paired(res, cand, comp):
    """Per cell: pair benefit (comp - cand), recipient differences (cand - comp) with bootstrap SEs."""
    out = {'pair': {'cells': {}, 'cell_se': {}}, 'income': {'cells': {}, 'cell_se': {}}, 'occupation': {'cells': {}, 'cell_se': {}}}
    for cell, cres in res.items():
        for view in VIEWS:
            a, b = cres[cand][view], cres[comp][view]
            if view == 'pair':
                d, db = b['score_auc'] - a['score_auc'], b['boot'] - a['boot']
            else:
                d, db = a['score_auc'] - b['score_auc'], a['boot'] - b['boot']
            out[view]['cells'][cell] = float(d); out[view]['cell_se'][cell] = float(np.std(db, ddof=1))
    return out


def strip(res):
    return {cell: {n: {v: {k: x for k, x in e.items() if k != 'boot'} for v, e in ent.items()} for n, ent in cres.items()} for cell, cres in res.items()}


def load_member(member, seed):
    """A bank member's MODEL_SELECT release: {'type': 'reference', 'name': ...} (ohn.devrun.REFERENCE_SET) or
    {'type': 'occ', 'family': F, 'config': ID, 'anchor': 'H3' | 'TW'} (an occupation release beside a frozen income anchor) or
    {'type': 'complete', 'family': F, 'config': ID} (a release directory with RELEASE.npz holding both tasks)."""
    from . import devrun as D
    if member['type'] == 'reference':
        return D.reference_release(member['name'], seed, 'MODEL_SELECT')
    if member['type'] == 'occ':
        occ, _, _ = D.load_occupation(member['family'], member['config'], seed, roles=('MODEL_SELECT',))
        return {'income': D.anchor_income(member['anchor'], seed, 'MODEL_SELECT'), 'occupation': occ['MODEL_SELECT']}
    if member['type'] == 'complete':
        d = D.occ_dir(member['family'], member['config'], seed)
        rec = json.loads((d / 'RELEASE.json').read_text())
        if C.sha(d / 'RELEASE.npz') != rec['release_sha256']:
            raise PermissionError('release file differs from its record')
        with np.load(d / 'RELEASE.npz') as z:
            return {t: z[f'MODEL_SELECT__{t}'].copy() for t in C.TASKS}
    raise ValueError(member['type'])


def lock_check():
    """Refuse to score with readers unless the reader-grid code is frozen by the committed DEVGRID_CODE_LOCK.json."""
    lock = C.RESULTS / 'DEVGRID_CODE_LOCK.json'
    if not C.committed(lock):
        raise PermissionError('reader grid code not yet locked (DEVGRID_CODE_LOCK.json must be committed first)')
    rec = json.loads(lock.read_text())
    cur = {f: C.sha(C.ROOT / f) for f in rec['files_sha256']}
    if cur != rec['files_sha256'] or rec['versions'] != versions():
        raise PermissionError('reader grid code or library versions differ from DEVGRID_CODE_LOCK.json')


LOCK_FILES = ('ohn/devreaders.py', 'ffv/readers.py', 'ffv/features.py', 'scm_bench/readers.py',
              'task_conditioned_representation/attacks.py', 'task_conditioned_representation/inference.py')


def run(spec_path, seed, out_dir):
    """Evaluate one bank (candidates and comparators) on the development grid for one release seed."""
    lock_check()
    spec = json.loads(Path(spec_path).read_text())
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh output directory required')
    S = C.load_role('MODEL_SELECT', ('features', 'sex'), purpose=f"development reader grid {spec['bank_id']} seed {seed}")
    bank = {name: load_member(m, seed) for name, m in spec['members'].items()}
    return run_bank(bank, S['sex'], S['unit'], spec, seed, out)


def run_bank(bank, sex, unit, spec, seed, out, write_ledger=True):
    from . import devrun as D
    from . import ledger as L
    out.mkdir(parents=True, mode=0o700)
    fracs = tuple(spec.get('fractions', [1.0]))
    stats = {'reader_fits': 0, 'cache_hits': 0, 'reader_cpu_s': 0.0}
    t0 = time.process_time()
    res, stats = evaluate_bank(bank, seed, sex, unit, fracs, stats)
    pairs = {}
    for cand in spec['candidates']:
        for comp in spec['comparators']:
            pairs.setdefault(cand, {})[comp] = paired({c: v for c, v in res.items()}, cand, comp)
    winners_changes = {}
    for name in bank:
        for view in VIEWS:
            ws = [res[c][name][view]['winner'] for c in res if '@' not in c]
            winners_changes[f'{name}|{view}'] = {'distinct_winners': sorted(set(ws)), 'n_cells': len(ws),
                                                'near_ties': sum(bool(res[c][name][view]['near_tie']) for c in res if '@' not in c)}
    boot = {f'{cell}__{n}__{v}': e['boot'] for cell, cres in res.items() for n, ent in cres.items() for v, e in ent.items()}
    with open(out / 'BOOT.npz', 'wb') as f:
        np.savez(f, **boot)
    rec = {'schema': 'OHN1-devgrid-result-v1', 'bank_id': spec['bank_id'], 'seed': seed, 'spec': spec, 'grid_version': GRID_VERSION,
           'cells': strip(res), 'paired': pairs, 'winner_changes': winners_changes, 'stats': stats,
           'cpu_s': round(time.process_time() - t0, 1), 'versions': versions(), 'utc': C.now(), 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json(out / 'GRID.json', rec)
    D.unit_stats(reader_fits=stats['reader_fits'], cache_hits=stats['cache_hits'])
    for cand in (spec['candidates'] if write_ledger else ()):
        summ = {comp: {v: {'mean': float(np.mean(list(p[v]['cells'].values()))), 'min': float(np.min(list(p[v]['cells'].values()))),
                           'max': float(np.max(list(p[v]['cells'].values())))} for v in VIEWS} for comp, p in pairs[cand].items()}
        L.append({'family': 'GRID', 'version': spec['bank_id'], 'config_id': cand, 'seed': seed, 'kind': 'reader_grid',
                  'budget_class': 'NONE', 'counts': 0, 'reader_fits': stats['reader_fits'], 'cache_hits': stats['cache_hits'],
                  'unit_id': os.environ.get(C.UNIT_ENV, ''), 'roles_read': 'MODEL_SELECT (SEX: reader grid, logged)',
                  'grid_pair_benefit_summary': json.dumps({c: round(s['pair']['mean'], 4) for c, s in summ.items()}),
                  'grid_recipient_summary': json.dumps({c: [round(s['income']['max'], 4), round(s['occupation']['max'], 4)] for c, s in summ.items()}),
                  'notes': f"cells={len([c for c in res if '@' not in c])}; fractions={list(fracs)}"})
    return rec


def selftest(n=1500):
    """Invented data only (no real data, no label): shapes, caching, winners, paired differences, bootstrap, learning curves."""
    global CACHE
    import tempfile
    CACHE = Path(tempfile.mkdtemp(prefix='ohn_devgrid_selftest_'))
    rng = np.random.default_rng(4242)
    unit = np.repeat(np.arange(n // 2), 2)[:n]
    sex = (rng.random(n) < 0.47).astype(np.int64)
    z = rng.normal(size=(n, 5)); z[:, 1] += 0.8 * sex
    occ_leaky = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    z2 = z.copy(); z2[:, 1] -= 0.8 * sex * 0.9
    occ_safe = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
    inc = 1 / (1 + np.exp(-(rng.normal(size=n) + 0.2 * sex))); inc = np.column_stack((1 - inc, inc))
    bank = {'CAND': {'income': inc, 'occupation': occ_safe}, 'COMP': {'income': inc, 'occupation': occ_leaky}}
    res, stats = evaluate_bank(bank, 0, sex, unit, fracs=(1.0, 0.25))
    p = paired(res, 'CAND', 'COMP')
    res2, stats2 = evaluate_bank(bank, 0, sex, unit, fracs=(1.0,))           # second call: everything from the cache
    checks = {'cells': sorted(res), 'pair_benefit_positive_all_full_cells': all(v > 0 for c, v in p['pair']['cells'].items() if '@' not in c),
              'income_view_identical_wires_zero_diff': all(abs(v) < 1e-12 for v in p['income']['cells'].values()),
              'bootstrap_se_positive': all(v > 0 for v in p['pair']['cell_se'].values()),
              'cache_second_call_all_hits': stats2['reader_fits'] == 0 and stats2['cache_hits'] > 0,
              'reader_fits_first_call': stats['reader_fits'], 'cache_hits_first_call': stats['cache_hits'],
              'pair_benefits': {c: round(v, 4) for c, v in p['pair']['cells'].items()}}
    out = Path(tempfile.mkdtemp(prefix='ohn_devgrid_run_')) / 'run'
    spec = {'bank_id': 'SELFTEST', 'members': {}, 'candidates': ['CAND'], 'comparators': ['COMP'], 'fractions': [1.0, 0.25]}
    rec = run_bank(bank, sex, unit, spec, 0, out, write_ledger=False)
    g = json.loads((out / 'GRID.json').read_text())
    checks['run_bank_written'] = (out / 'BOOT.npz').is_file() and g['paired']['CAND']['COMP']['pair']['cells'] == rec['paired']['CAND']['COMP']['pair']['cells']
    checks['run_bank_all_cached'] = rec['stats']['reader_fits'] == 0
    checks['all_ok'] = (checks['pair_benefit_positive_all_full_cells'] and checks['income_view_identical_wires_zero_diff']
                        and checks['bootstrap_se_positive'] and checks['cache_second_call_all_hits'] and checks['run_bank_written']
                        and checks['run_bank_all_cached'])
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('manifest')
    sub.add_parser('selftest')
    lk = sub.add_parser('lock')
    r = sub.add_parser('run'); r.add_argument('--spec', required=True); r.add_argument('--seed', type=int, required=True)
    r.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'manifest':
        m = manifest()
        C.write_json(GRID_PATH, m)
        print(json.dumps(m['repetitions'], indent=1))
    elif a.cmd == 'selftest':
        chk = selftest()
        C.write_json(C.RESULTS / 'checks' / 'DEVGRID_SELFTEST.json', chk)
        print(json.dumps(chk, indent=1))
        if not chk['all_ok']:
            raise SystemExit(1)
    elif a.cmd == 'lock':
        rec = {'schema': 'OHN1-devgrid-code-lock-v1', 'utc': C.now(), 'grid_version': GRID_VERSION, 'versions': versions(),
               'files_sha256': {f: C.sha(C.ROOT / f) for f in LOCK_FILES}, 'dev_grid_sha256': C.sha(GRID_PATH)}
        C.write_json_x(C.RESULTS / 'DEVGRID_CODE_LOCK.json', rec)
        print(json.dumps(rec, indent=1))
    elif a.cmd == 'run':
        rec = run(a.spec, a.seed, a.out)
        print(json.dumps({'bank_id': rec['bank_id'], 'seed': rec['seed'], 'stats': rec['stats'], 'cpu_s': rec['cpu_s']}))


if __name__ == '__main__':
    main()
