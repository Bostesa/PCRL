"""Estimator E of EDG1 (PROTOCOL.md section 5): the declared DRF1 reader bank (16 recipes, locked recipe code vendored with
matching SHA-256) fitted on FIT, oriented and selected on VALIDATION (max oriented VALIDATION AUC, exact ties to the earlier
candidate; pair candidates = pair readers, then ignore-occupation and ignore-income readers, source recipes first) and scored ONCE
on SCORE, on the bounded prospectively fixed subset BANK_CELLS, using the SAME synthetic people and split as the main grid's D_bank.

For every fitted recipe the scores of all observable wires of the law are obtained by appending the support wires as extra rows
that are in neither FIT nor VALIDATION (they never enter a fit, an orientation or a selection), so each recipe's exact population
AUC under the known law is available: model-class + finite-training loss = A - pop, selection loss = max_r pop_r - pop_winner,
winner's curse = val_winner - pop_winner, test-sample variation = score_winner - pop_winner. Class limits: the LR-type and LOOKUP
recipes are refitted once per bank view on a large separate synthetic sample (N_BIG; seeds disjoint from every repetition) and
their population AUC is the near-infinite-data value of that model class; tree and MLP recipes are not refitted large (cost).

    bench   timing-only microbenchmark on a separate synthetic law and seed (no AUC is computed or printed)
    bank    the subset; repetitions per cell from BENCH.json by the declared rule"""
from __future__ import annotations

import argparse
import os
import gzip
import json
import math
import resource
import time
import warnings
from datetime import datetime, timezone

import numpy as np

from . import OUT, RESULTS
from . import laws as L
from . import simulate as SIM
from drf import finite_law as FL
from scm_bench.readers import SLATE, kinds, lookup_keys, fit_lookup, predict_lookup
from task_conditioned_representation.attacks import MODEL_SEED_BASE, fit_reader, predict_reader, orientation, orient
from task_conditioned_representation.inference import auc
from ffv.readers import NEW8, design_for, fit_new, predict_new, pick

SRC8 = tuple(n for n, _ in SLATE) + ('LOOKUP',)
RECIPES = SRC8 + NEW8
SEED_OFFSET = 7100                       # synthetic reader seeds MODEL_SEED_BASE + 7100 + rep (no DRF1 reader seed)
BANK_CPU_CAP = 1500.0                    # CPU seconds for the repetitions of the whole subset (declared before the benchmark)
MAX_REPS, MIN_REPS_BANK = 10, 3
N_BIG, N_BIG_VAL = 30000, 10000          # class-limit refit sizes
CLASS_LIMIT_RECIPES = ('OUT_LR', 'LOG_LR', 'DEC_LR', 'CATI01', 'CATI1', 'LOOKUP')
# (grid cell of edg.simulate.scenarios(), {bank view name: (recipe view, law view)}), fixed before any bank or grid outcome
BANK_CELLS = (
    ('IND|0.5|skewed|8000', {'income': ('income', 'income'), 'occupation': ('occupation', 'occupation'), 'pair': ('pair', 'pair')}),
    ('REF|0.5|skewed|8000', {'L16': ('occupation', 'L16'), 'L256': ('occupation', 'L256')}),
    ('RED|0.5|skewed|8000', {'income': ('income', 'income'), 'occupation': ('occupation', 'occupation'), 'pair': ('pair', 'pair')}),
    ('COMP|0.5|uniform|8000', {'income': ('income', 'income'), 'occupation': ('occupation', 'occupation'), 'pair': ('pair', 'pair')}),
    ('RED|0.2|skewed|8000', {'income': ('income', 'income'), 'occupation': ('occupation', 'occupation'), 'pair': ('pair', 'pair')}),
)


def _cell(key):
    fam, pi, occ, n = key.split('|')
    for sc in SIM.scenarios():
        if (sc['family'], sc['pi'], sc['occ'], sc['n'], sc['clustered']) == (fam, float(pi), occ, int(n), False):
            return sc
    raise KeyError(key)


# ---------------------------------------------------------------- verbatim the DRF1 recipe dispatch (drf.devreaders._src_tab/_fit_one)
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


def _winners_view(rd, view, rd_inc=None, rd_occ=None):
    """drf.devreaders.winners for one view (VALIDATION only)."""
    if view != 'pair':
        return pick([(n, rd[n][1]['val_auc'], f'{view}__{n}', 'source' if n in SRC8 else 'new') for n in RECIPES])
    cands = []
    for group in (SRC8, NEW8):
        org = 'source' if group is SRC8 else 'new'
        cands += [(n, rd[n][1]['val_auc'], f'pair__{n}', org) for n in group]
        cands += [(f'IGNORE_occupation__{n}', rd_inc[n][1]['val_auc'], f'income__{n}', org) for n in group]
        cands += [(f'IGNORE_income__{n}', rd_occ[n][1]['val_auc'], f'occupation__{n}', org) for n in group]
    return pick(cands)


# ---------------------------------------------------------------- arrays of a bank view with the support rows appended
def _arrays(law, c, recipe_view, law_view):
    if law['family'] == 'REF':
        occ = law['vectors']['levels'][law_view][c]
        inc = np.tile([0.5, 0.5], (len(c), 1))
    else:
        inc, occ = law['vectors']['income'][c], law['vectors']['occupation'][c]
    V = law['views'][law_view]
    w = V['wires']
    if recipe_view == 'pair':
        sup_inc, sup_occ = w
    elif recipe_view == 'income':
        sup_inc, sup_occ = w, np.tile(occ[0], (len(w), 1))
    else:
        sup_occ, sup_inc = w, np.tile(inc[0], (len(w), 1))
    return np.vstack((inc, sup_inc)), np.vstack((occ, sup_occ)), V


def fit_bank_view(law, c, s, fit, val, recipe_view, law_view, rs, recipes=RECIPES):
    inc, occ, V = _arrays(law, c, recipe_view, law_view)
    n, K = len(c), V['K']
    pad = np.zeros(K, dtype=bool)
    sex = np.concatenate((s, np.zeros(K, dtype=s.dtype)))
    fitm, valm = np.concatenate((fit, pad)), np.concatenate((val, pad))
    rd, pops, cpu = {}, {}, {}
    for name in recipes:
        t0 = time.process_time()
        sc_all, info = _fit_one(name, recipe_view, inc, occ, sex, fitm, valm, rs)
        cpu[name] = time.process_time() - t0
        if not np.isfinite(sc_all).all():
            raise RuntimeError(f'{name}: non-finite scores')
        rd[name] = (sc_all[:n], info)
        pops[name] = float(FL.auc_of_score(sc_all[n:], V['m1'], V['m0']))
    return rd, pops, cpu


# ---------------------------------------------------------------- microbenchmark (timing only)
def bench(out_path):
    law = L.build('COMP', 0.5, 'skewed')
    rng = np.random.default_rng(np.random.SeedSequence(SIM.BASE_SEED + 1, spawn_key=(777,)))
    c, s, _ = L.draw(law, 8000, rng)
    parts = rng.permutation(8000) % 10
    fit, val = parts <= 4, (parts == 5) | (parts == 6)
    t = {}
    for view in ('income', 'occupation', 'pair'):
        _, _, cpu = fit_bank_view(law, c, s, fit, val, view, view, MODEL_SEED_BASE + 99999)   # AUCs discarded
        t[view] = {k: round(v, 3) for k, v in cpu.items()}
    per_view = {v: sum(x.values()) for v, x in t.items()}
    n_views = {'single': 0, 'pair': 0}
    for _, views in BANK_CELLS:
        for rv, _ in views.values():
            n_views['pair' if rv == 'pair' else 'single'] += 1
    t_rep = n_views['single'] * max(per_view['income'], per_view['occupation']) + n_views['pair'] * per_view['pair']
    reps = int(min(MAX_REPS, max(MIN_REPS_BANK, math.floor(BANK_CPU_CAP / t_rep))))
    rec = {'schema': 'EDG1-bank-bench-v1', 'law': 'COMP|0.5|skewed (benchmark seed BASE_SEED + 1, spawn 777; no AUC recorded)',
           'n_fit': int(fit.sum()), 'cpu_s_per_recipe': t, 'cpu_s_per_view': per_view, 'bank_views': n_views,
           'projected_cpu_s_per_rep': round(t_rep, 1), 'rule': f'reps = min({MAX_REPS}, max({MIN_REPS_BANK}, floor({BANK_CPU_CAP} / projected)))',
           'reps_per_cell': reps, 'utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    with open(out_path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ---------------------------------------------------------------- the subset
def run_bank(label, reps):
    t0 = time.process_time()
    started = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    path = OUT / f'bank_{label}.jsonl.gz'
    with gzip.open(path, 'wt') as fh:
        for key, views in BANK_CELLS:
            sc = _cell(key)
            law = L.build(sc['family'], sc['pi'], sc['occ'])
            for rep in range(reps):
                c, s, unit, ct, st, unit_t, folds, parts = SIM.sample_rep(sc, law, rep)
                fit, val, scm = parts <= 4, (parts == 5) | (parts == 6), parts >= 7
                rs = MODEL_SEED_BASE + SEED_OFFSET + rep
                rds, popss, row = {}, {}, {'bank_cell': key, 'cell': sc['cell'], 'rep': rep, 'views': {}}
                for bview, (rview, lview) in views.items():
                    rds[bview], popss[bview], cpu = fit_bank_view(law, c, s, fit, val, rview, lview, rs)
                    row.setdefault('cpu_s', {})[bview] = round(sum(cpu.values()), 2)
                for bview, (rview, lview) in views.items():
                    rd, pops = rds[bview], popss[bview]
                    if rview == 'pair':
                        w = _winners_view(rd, 'pair', rds['income'], rds['occupation'])
                        allpop = {**{n: pops[n] for n in RECIPES},
                                  **{f'IGNORE_occupation__{n}': popss['income'][n] for n in RECIPES},
                                  **{f'IGNORE_income__{n}': popss['occupation'][n] for n in RECIPES}}
                        src = {'pair': rd, 'income': rds['income'], 'occupation': rds['occupation']}
                    else:
                        w = _winners_view(rd, rview)
                        allpop = dict(pops)
                        src = {rview: rd}
                    v_, nm = w['reader_key'].split('__', 1)
                    s_win = src[v_][nm][0]
                    row['views'][bview] = {
                        'A': law['views'][lview]['A'], 'winner': w['candidate'], 'winner_origin': w['origin'],
                        'winner_val_auc': w['attack_validation_auc'], 'near_tie': w['AV_NEAR_TIE'],
                        'winner_score_auc': float(auc(s[scm], s_win[scm])), 'winner_pop_auc': allpop[w['candidate']],
                        'best_pop_auc_in_bank': max(allpop.values()), 'best_pop_candidate': max(allpop, key=allpop.get),
                        'recipes': {n: {'val_auc': rd[n][1]['val_auc'], 'pop_auc': pops[n], 'orientation': rd[n][1]['orientation'],
                                        'converged': rd[n][1].get('converged')} for n in RECIPES}}
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                print(json.dumps({'bank_cell': key, 'rep': rep, 'cpu_s': round(time.process_time() - t0, 1),
                                  'utc': datetime.now(timezone.utc).strftime('%H:%M:%SZ')}), flush=True)
    cl = class_limits()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rec = {'schema': 'EDG1-bank-run-v1', 'reps_per_cell': reps, 'cells': [k for k, _ in BANK_CELLS], 'out': os.path.relpath(path, RESULTS),
           'class_limits': cl, 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           'cpu_s_process': round(time.process_time() - t0, 2), 'rusage_user_s': ru.ru_utime, 'rusage_sys_s': ru.ru_stime}
    with open(OUT / f'bank_{label}.run.json', 'w') as fh:
        json.dump(rec, fh, indent=1)
    return rec


def class_limits():
    """Once per bank view: CLASS_LIMIT_RECIPES fitted on N_BIG people, oriented on N_BIG_VAL others (separate seeds)."""
    out = {}
    for key, views in BANK_CELLS:
        sc = _cell(key)
        law = L.build(sc['family'], sc['pi'], sc['occ'])
        rng = np.random.default_rng(np.random.SeedSequence(SIM.BASE_SEED + 2, spawn_key=(sc['cell'], 9999)))
        c, s, _ = L.draw(law, N_BIG + N_BIG_VAL, rng)
        fit = np.arange(len(c)) < N_BIG
        val = ~fit
        for bview, (rview, lview) in views.items():
            _, pops, cpu = fit_bank_view(law, c, s, fit, val, rview, lview, MODEL_SEED_BASE + SEED_OFFSET + 999,
                                         recipes=CLASS_LIMIT_RECIPES)
            out[f'{key}|{bview}'] = {'A': law['views'][lview]['A'], 'pop_auc_large_fit': pops, 'cpu_s': round(sum(cpu.values()), 1)}
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('bench', 'bank'))
    ap.add_argument('--label', default='all')
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'bench':
        print(json.dumps(bench(RESULTS / 'BENCH.json')))
    else:
        reps = json.load(open(RESULTS / 'BENCH.json'))['reps_per_cell']
        print(json.dumps(run_bank(a.label, reps)))
