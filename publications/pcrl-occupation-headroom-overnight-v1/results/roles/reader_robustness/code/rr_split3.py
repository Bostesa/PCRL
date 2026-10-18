"""Extra-split diagnostic (coordinator request 07:00Z; reader-robustness investigator). NEVER used for selection.

Question: the registered development grid has only two household splits (R1, R2), shared by both release seeds. How much does a
THIRD, independently drawn household split move the paired benefits?

Public-use ACS 2019 1-Year PUMS Florida person records, MODEL_SELECT development role (adaptively used). Readers estimate the
recorded SEX column (sex0 = men, sex1 = women) from released outputs only (each recipient's float64 probability vector and argmax
decision; the linked pair). Labels: MODEL_SELECT SEX via ohn.common.load_role(..., purpose='reader robustness: extra-split
diagnostic ...') (logged).

Split R3: MODEL_SELECT households ordered by SHA-256('OHN1_devgrid_R3_diag|<household>') and dealt round-robin into 10 parts
(ohn.common.household_folds, the grid's own construction with a new tag); parts 0-4 FIT, 5-6 VALIDATION, 7-9 SCORE.
Readers: the FROZEN development slate (source 8 + new 8, ignore-one alternatives for the pair, max oriented VALIDATION AUC with
exact ties to the earlier candidate, near tie < 0.010), fitted with the LOCKED recipe code (scm_bench.readers,
task_conditioned_representation.attacks / .inference, ffv.readers; file hashes checked against DEVGRID_CODE_LOCK.json); the
per-recipe fitting and winner logic of ohn.devreaders (_fit_one, winners) is re-stated here verbatim, not imported or edited.
Reader seeds = the grid's (MODEL_SEED_BASE + 300 + 10 x release seed + k), k0 / k1. Household bootstrap of R3 SCORE (B = 400,
seed 2026101806 + 100 x release seed + 2, readers fixed). R3 readers live in this module's own cache (never the grid cache).
Equivalence check: the same code reproduces the grid's cached R1 readers bitwise for one release (all views and recipes).

    python rr_split3.py selftest
    python rr_split3.py run --spec <spec.json> --seed 0
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ohn import common as C                                                                       # noqa: E402
from ohn.devreaders import load_member                                                             # noqa: E402  (loading only)
from scm_bench.readers import SLATE, kinds, lookup_keys, fit_lookup, predict_lookup                # noqa: E402  (locked recipes)
from task_conditioned_representation.attacks import MODEL_SEED_BASE, fit_reader, predict_reader, orientation, orient  # noqa: E402
from task_conditioned_representation.inference import auc                                          # noqa: E402
from ffv.readers import NEW8, design_for, fit_new, predict_new, pick, versions                     # noqa: E402
import rr_probe as RP                                                                               # noqa: E402  (grid-cache reader, bootstrap)

SPLIT_TAG = 'OHN1_devgrid_R3_diag'
REP = 'R3'
VERSION = 'OHN1-rr-split3-v1'
SRC8 = tuple(n for n, _ in SLATE) + ('LOOKUP',)
VIEWS = ('income', 'occupation', 'pair')
READER_SEEDS = (0, 1)
B_BOOT = 400
BOOT_SEED = 2026101806
CACHE = RP.RUN_ROOT / 'R3_cache'
LOCK_FILES = ('ffv/readers.py', 'ffv/features.py', 'scm_bench/readers.py', 'task_conditioned_representation/attacks.py',
              'task_conditioned_representation/inference.py')


def lock_check():
    rec = json.loads((C.RESULTS / 'DEVGRID_CODE_LOCK.json').read_text())
    for f in LOCK_FILES:
        if C.sha(C.ROOT / f) != rec['files_sha256'][f]:
            raise PermissionError(f'{f} differs from DEVGRID_CODE_LOCK.json')
    if rec['versions'] != versions():
        raise PermissionError('library versions differ from DEVGRID_CODE_LOCK.json')


def roles_r3(unit):
    part = C.household_folds(unit, 10, SPLIT_TAG)
    return {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}


def reader_seed(rel_seed, k):
    return MODEL_SEED_BASE + 300 + 10 * int(rel_seed) + int(k)


# ---------------------------------------------------------------- verbatim restatement of ohn.devreaders._fit_one / winners
def _src_tab(view, inc, occ, kind):
    a, b = kinds(inc, 2)[kind], kinds(occ, 5)[kind]
    return a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))


def fit_one(name, view, inc, occ, sex, fit, val, rs):
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


def winners(rd_inc, rd_occ, rd_pair):
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


def _score_of(key, rd):
    view, name = key.split('__', 1)
    return rd[view][name][0]


# ---------------------------------------------------------------- one view of one wire on a split (own cache)
def view_readers(view, inc, occ, sex, roles, rs, tag, cache=None, stats=None):
    cache = Path(cache or CACHE)
    key = hashlib.sha256(f'{VERSION}|{tag}|{rs}|{RP.wire_hash(view, inc, occ)}|{C.arr_sha(np.asarray(sex))}'.encode()).hexdigest()
    p = cache / f'{key}.npz'
    names = SRC8 + NEW8
    if p.is_file():
        with np.load(p, allow_pickle=False) as z:
            info = json.loads(str(z['info']))
            if stats is not None:
                stats['cache_hits'] += 1
            return {n: (z[f'score__{n}'].copy(), info[n]) for n in names}
    fit, val, sc = roles['FIT'], roles['VALIDATION'], roles['SCORE']
    out, info = {}, {}
    t0 = time.process_time()
    for n in names:
        s, inf = fit_one(n, view, inc, occ, sex, fit, val, rs)
        out[n] = (s[sc], inf); info[n] = inf
        if stats is not None:
            stats['reader_fits'] += 1
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f'.{key}.{os.getpid()}.npz'
    with open(tmp, 'wb') as f:
        np.savez(f, info=json.dumps(info), **{f'score__{n}': out[n][0] for n in names})
    os.replace(tmp, p)
    if stats is not None:
        stats['reader_cpu_s'] += time.process_time() - t0
    return out


def boot_weights(unit_sc, rel_seed):
    g, inv = np.unique(unit_sc, return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED + 100 * int(rel_seed) + 2)
    W = np.empty((B_BOOT, len(unit_sc)))
    for b in range(B_BOOT):
        W[b] = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))[inv]
    return W


def evaluate(bank, rel_seed, sex, unit, cache=None, stats=None):
    roles = roles_r3(unit)
    sc = roles['SCORE']; y = sex[sc]
    W = boot_weights(unit[sc], rel_seed)
    res = {}
    for k in READER_SEEDS:
        cell = f'S{rel_seed}{REP}k{k}'
        rs = reader_seed(rel_seed, k)
        cres = {}
        for name, rel in bank.items():
            rd = {v: view_readers(v, rel['income'], rel['occupation'], sex, roles, rs, SPLIT_TAG, cache, stats) for v in VIEWS}
            w = winners(rd['income'], rd['occupation'], rd['pair'])
            ent = {}
            for v in VIEWS:
                s = _score_of(w[v]['reader_key'], rd)
                ent[v] = {'winner': w[v]['candidate'], 'origin': w[v]['origin'], 'val_auc': w[v]['attack_validation_auc'],
                          'gap_to_runner_up': w[v]['gap_to_runner_up'], 'near_tie': w[v]['AV_NEAR_TIE'],
                          'score_auc': float(auc(y, s)), 'scores': s,
                          'recipe_score_auc': {n: float(auc(y, rd[v][n][0])) for n in SRC8 + NEW8},
                          'recipe_scores': {n: rd[v][n][0] for n in SRC8 + NEW8}}
            cres[name] = ent
        res[cell] = {'releases': cres, 'W': W, 'y': y}
    return res, roles


_MEMO = {}


def _draws(c, cell, rel_name, v, key, s):
    k = (id(c), cell, rel_name, v, key)
    if k not in _MEMO:
        _MEMO[k] = RP.wauc_draws(c['y'], s, c['W'])
    return _MEMO[k]


def paired(res, cand, comp):
    out = {}
    for cell, c in res.items():
        a, b = c['releases'][cand], c['releases'][comp]
        for v in VIEWS:
            keys = [('WINNER', a[v]['scores'], b[v]['scores'])] + [(n, a[v]['recipe_scores'][n], b[v]['recipe_scores'][n]) for n in SRC8 + NEW8]
            for name, sa, sb in keys:
                da, db = _draws(c, cell, cand, v, name, sa), _draws(c, cell, comp, v, name, sb)
                pa, pb = float(auc(c['y'], sa)), float(auc(c['y'], sb))
                d, dd = (pb - pa, db - da) if v == 'pair' else (pa - pb, da - db)
                e = out.setdefault(v, {}).setdefault(name, {'cells': {}, 'cell_se': {}})
                e['cells'][cell] = d; e['cell_se'][cell] = float(np.std(dd, ddof=1))
    return out


def equivalence_check(rel, sex, unit, rel_seed):
    """The re-stated slate code must reproduce the grid's cached R1 readers bitwise (k0, every view and recipe)."""
    from ohn.devreaders import roles_of
    r1 = roles_of(unit, 'R1')
    rs = reader_seed(rel_seed, 0)
    tmp = Path(tempfile.mkdtemp(prefix='rr_split3_equiv_'))
    checked = mismatched = 0
    for v in VIEWS:
        mine = view_readers(v, rel['income'], rel['occupation'], sex, r1, rs, 'EQUIV_R1', tmp)
        grid = RP.slate_cell(v, rel['income'], rel['occupation'], sex, rel_seed, 'R1', 0, 1.0)
        if grid is None:
            return {'checked': 0, 'note': 'grid cache entry absent'}
        for n in SRC8 + NEW8:
            checked += 1
            same = np.array_equal(mine[n][0], grid[n][0]) and abs(mine[n][1]['val_auc'] - grid[n][1]['val_auc']) < 1e-15
            mismatched += not same
    return {'checked': checked, 'mismatched': mismatched, 'ok': checked > 0 and mismatched == 0}


def run(spec_path, seed):
    lock_check()
    spec = json.loads(Path(spec_path).read_text())
    out = RP.RUN_ROOT / 'R3' / spec['probe_id'] / f'S{seed}'
    if out.exists():
        raise PermissionError('fresh output directory required')
    S = C.load_role('MODEL_SELECT', ('features', 'sex'),
                    purpose=f"reader robustness: extra-split diagnostic {SPLIT_TAG} ({spec['probe_id']}, seed {seed}); never selection")
    sex, unit = S['sex'], S['unit']
    del S['X'], S['feature_names']
    bank = {name: load_member(m, seed) for name, m in spec['members'].items()}
    stats = {'reader_fits': 0, 'cache_hits': 0, 'reader_cpu_s': 0.0}
    t0 = time.process_time()
    equiv = equivalence_check(bank[spec['equivalence_member']], sex, unit, seed)
    res, roles = evaluate(bank, seed, sex, unit, stats=stats)
    pairs = {f'{a}|{b}': paired(res, a, b) for a in spec['candidates'] for b in spec['comparators'] if a != b}
    from ohn.devreaders import roles_of
    ov = {}
    for rep in ('R1', 'R2'):
        r = roles_of(unit, rep)
        for role in ('FIT', 'VALIDATION', 'SCORE'):
            ov[f'R3_vs_{rep}_{role}_households'] = int(np.intersect1d(np.unique(unit[roles[role]]), np.unique(unit[r[role]])).size)
    sizes = {role: {'people': int(m.sum()), 'households': int(len(np.unique(unit[m])))} for role, m in roles.items()}
    cells = {cell: {name: {v: {k: x for k, x in e.items() if k not in ('scores', 'recipe_scores')} for v, e in ent.items()}
                    for name, ent in c['releases'].items()} for cell, c in res.items()}
    out.mkdir(parents=True, mode=0o700)
    rec = {'schema': 'OHN1-rr-split3-v1', 'probe_id': spec['probe_id'], 'seed': seed, 'split_tag': SPLIT_TAG, 'role_sizes': sizes,
           'household_overlap_with_registered_splits': ov, 'equivalence_check_R1': equiv, 'cells': cells, 'paired': pairs,
           'stats': stats, 'cpu_s': round(time.process_time() - t0, 1), 'versions': versions(), 'utc': C.now(),
           'unit_id': os.environ.get(C.UNIT_ENV), 'note': 'diagnostic extra household split; never selection; reader-relative'}
    C.write_json(out / 'R3.json', rec)
    try:
        from ohn import devrun as D
        D.unit_stats(reader_fits=stats['reader_fits'], cache_hits=stats['cache_hits'])
    except Exception:                                   # noqa: BLE001  (bookkeeping only)
        pass
    print(json.dumps({'probe_id': spec['probe_id'], 'seed': seed, 'equivalence': equiv, 'stats': stats, 'cpu_s': rec['cpu_s']}))
    return rec


def selftest():
    """Invented data only: roles are household-disjoint, caching works, winners and paired differences are well formed."""
    rng = np.random.default_rng(31)
    n = 1500
    unit = np.sort(rng.integers(0, 1000, n))
    sex = (rng.random(n) < 0.47).astype(np.int64)
    z = rng.normal(size=(n, 5)); z[:, 1] += 0.8 * sex
    occ = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    z2 = z.copy(); z2[:, 1] -= 0.7 * sex
    occ2 = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
    t = 1 / (1 + np.exp(-(rng.normal(size=n) + 0.2 * sex))); inc = np.column_stack((1 - t, t))
    bank = {'CAND': {'income': inc, 'occupation': occ2}, 'COMP': {'income': inc, 'occupation': occ}}
    tmp = Path(tempfile.mkdtemp(prefix='rr_split3_selftest_'))
    st = {'reader_fits': 0, 'cache_hits': 0, 'reader_cpu_s': 0.0}
    global CACHE
    CACHE = tmp
    res, roles = evaluate(bank, 0, sex, unit, cache=tmp, stats=st)
    f1 = st['reader_fits']
    res2, _ = evaluate(bank, 0, sex, unit, cache=tmp, stats=st)
    p = paired(res, 'CAND', 'COMP')
    disjoint = all(not np.intersect1d(unit[roles[a]], unit[roles[b]]).size for a, b in (('FIT', 'VALIDATION'), ('FIT', 'SCORE'), ('VALIDATION', 'SCORE')))
    chk = {'disjoint_roles': bool(disjoint), 'reader_fits_first': f1, 'second_call_cached': st['reader_fits'] == f1,
           'income_identical_wire_zero': bool(all(abs(x) < 1e-12 for x in p['income']['WINNER']['cells'].values())),
           'pair_benefit_positive': bool(all(x > 0 for x in p['pair']['WINNER']['cells'].values())),
           'winner_scores_reproduce': bool(all(np.array_equal(res[c]['releases'][m][v]['scores'], res2[c]['releases'][m][v]['scores'])
                                               for c in res for m in bank for v in VIEWS)),
           'pair_benefits': {c: round(x, 4) for c, x in p['pair']['WINNER']['cells'].items()}}
    chk['all_ok'] = all(chk[k] for k in ('disjoint_roles', 'second_call_cached', 'income_identical_wire_zero', 'pair_benefit_positive',
                                         'winner_scores_reproduce'))
    return chk


def summarize(probe_id, seed, tag):
    """Label-free: R3 paired benefits next to the registered R1 / R2 cells of the same release seed (grid outputs for the frozen
    winner; probe outputs for fixed recipes where present). How far does a third split move each paired benefit?"""
    r3 = json.loads((RP.RUN_ROOT / 'R3' / probe_id / f'S{seed}' / 'R3.json').read_text())
    grid = {}
    for p in sorted((C.RUNS / 'GRID').glob(f'*/S{seed}/GRID.json')):
        g = json.loads(p.read_text())
        for cell, cres in g['cells'].items():
            if '@' in cell:
                continue
            for name, ent in cres.items():
                grid.setdefault((cell, name), {v: e['score_auc'] for v, e in ent.items()})
    rec_auc = {}
    for p in sorted(RP.RUN_ROOT.glob(f'*/S{seed}/PROBE.json')):
        g = json.loads(p.read_text())
        for cell, cres in g['cells'].items():
            for name, e in cres.items():
                for v in VIEWS:
                    ra = e['views'][v].get('slate_recipe_score_auc')
                    if ra:
                        rec_auc.setdefault((cell, name, v), ra)
    reps = ('R1', 'R2')
    out = {}
    for pk, views in r3['paired'].items():
        cand, comp = pk.split('|')
        for v in ('pair',):
            for scheme, d in views[v].items():
                r3c = d['cells']
                old = {}
                for rep in reps:
                    vals = []
                    for k in READER_SEEDS:
                        cell = f'S{seed}{rep}k{k}'
                        if scheme == 'WINNER':
                            if (cell, cand) in grid and (cell, comp) in grid:
                                vals.append(grid[(cell, comp)][v] - grid[(cell, cand)][v])
                        elif (cell, cand, v) in rec_auc and (cell, comp, v) in rec_auc:
                            vals.append(rec_auc[(cell, comp, v)][scheme] - rec_auc[(cell, cand, v)][scheme])
                    if len(vals) == 2:
                        old[rep] = float(np.mean(vals))
                m3 = float(np.mean(list(r3c.values())))
                row = {'R3_mean': m3, 'R3_cells': r3c, 'R3_cell_se': d['cell_se']}
                if len(old) == 2:
                    m2 = (old['R1'] + old['R2']) / 2
                    row.update({'R1_mean': old['R1'], 'R2_mean': old['R2'], 'two_split_mean': m2, 'three_split_mean': (old['R1'] + old['R2'] + m3) / 3,
                                'R3_minus_two_split_mean': m3 - m2, 'sd_across_three_split_means': float(np.std([old['R1'], old['R2'], m3], ddof=1)),
                                'R3_sign_differs_from_two_split_mean': bool(np.sign(m3) != np.sign(m2)),
                                'R3_outside_R1_R2_range': bool(m3 < min(old.values()) or m3 > max(old.values()))})
                out.setdefault(pk, {})[scheme] = row
    # pooled: how far R3 moves the winner-based benefit relative to the R1-R2 spread
    w = [r['WINNER'] for r in out.values() if 'WINNER' in r and 'two_split_mean' in r['WINNER']]
    pooled = {'n_pairs_winner': len(w),
              'median_abs_R3_minus_two_split_mean': float(np.median([abs(r['R3_minus_two_split_mean']) for r in w])) if w else None,
              'median_abs_R1_minus_R2': float(np.median([abs(r['R1_mean'] - r['R2_mean']) for r in w])) if w else None,
              'pairs_R3_sign_differs': [pk for pk, r in out.items() if r.get('WINNER', {}).get('R3_sign_differs_from_two_split_mean')],
              'pairs_R3_outside_R1_R2_range': [pk for pk, r in out.items() if r.get('WINNER', {}).get('R3_outside_R1_R2_range')]}
    fixed = [r[s] for r in out.values() for s in r if s not in ('WINNER', 'LOOKUP') and 'two_split_mean' in r[s]]
    pooled.update({'n_pair_recipe_rows_fixed': len(fixed),
                   'median_abs_R3_minus_two_split_mean_fixed_recipes': float(np.median([abs(r['R3_minus_two_split_mean']) for r in fixed])) if fixed else None,
                   'median_abs_R1_minus_R2_fixed_recipes': float(np.median([abs(r['R1_mean'] - r['R2_mean']) for r in fixed])) if fixed else None})
    res = {'schema': 'OHN1-rr-split3-summary-v1', 'probe_id': probe_id, 'seed': seed, 'utc': C.now(), 'split_tag': SPLIT_TAG,
           'role_sizes': r3['role_sizes'], 'household_overlap_with_registered_splits': r3['household_overlap_with_registered_splits'],
           'equivalence_check_R1': r3['equivalence_check_R1'], 'pooled': pooled, 'pairs': out,
           'r3_release_winners': {name: {v: [r3['cells'][c][name][v]['winner'] for c in sorted(r3['cells'])] for v in VIEWS}
                                  for name in r3['cells'][sorted(r3['cells'])[0]]},
           'r3_release_pair_auc': {name: float(np.mean([r3['cells'][c][name]['pair']['score_auc'] for c in r3['cells']]))
                                   for name in r3['cells'][sorted(r3['cells'])[0]]},
           'note': 'diagnostic extra split (never selection); R1 / R2 from the registered grid (winner) and the probe outputs (fixed recipes)'}
    C.write_json(RP.OUT_RES / f'{tag}__SPLIT3_SUMMARY.json', res)
    import csv
    with open(RP.OUT_RES / f'{tag}__SPLIT3_PAIRS.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['candidate', 'comparator', 'scheme', 'R1_mean', 'R2_mean', 'R3_mean', 'two_split_mean', 'three_split_mean',
                    'R3_minus_two_split_mean', 'sd_across_three_split_means', 'R3_sign_differs', 'R3_outside_R1_R2_range'])
        for pk, rows in sorted(out.items()):
            cand, comp = pk.split('|')
            for scheme, r in sorted(rows.items()):
                w.writerow([cand, comp, scheme, r.get('R1_mean'), r.get('R2_mean'), r['R3_mean'], r.get('two_split_mean'),
                            r.get('three_split_mean'), r.get('R3_minus_two_split_mean'), r.get('sd_across_three_split_means'),
                            r.get('R3_sign_differs_from_two_split_mean'), r.get('R3_outside_R1_R2_range')])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    r = sub.add_parser('run'); r.add_argument('--spec', required=True); r.add_argument('--seed', type=int, required=True)
    sm = sub.add_parser('summarize'); sm.add_argument('--probe', required=True); sm.add_argument('--seed', type=int, required=True)
    sm.add_argument('--tag', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'summarize':
        res = summarize(a.probe, a.seed, a.tag)
        print(json.dumps(res['pooled'], indent=1))
        return
    if a.cmd == 'selftest':
        chk = selftest()
        C.write_json(RP.OUT_RES / 'checks' / 'RR_SPLIT3_SELFTEST.json', chk)
        print(json.dumps(chk, indent=1))
        if not chk['all_ok']:
            raise SystemExit(1)
    else:
        run(a.spec, a.seed)


if __name__ == '__main__':
    main()
