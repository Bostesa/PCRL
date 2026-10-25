"""EDG1 self-test (invented tiny laws and the frozen laws' sampler; prints check results, timing and shapes only - never an
estimator summary of the registered grid). `smoke` times one repetition per family on seeds disjoint from the grid
(repetition indices 900000+) and discards its values."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np

from . import RESULTS, WORKTREE
from . import estimators as E
from . import laws as L
from . import simulate as SIM
from . import source_prc as PRC
from drf import finite_law as FL
from scm_bench.readers import fit_lookup, predict_lookup


def _people(m1, m0):
    ids = np.concatenate([np.repeat(np.arange(len(m1)), m1), np.repeat(np.arange(len(m0)), m0)])
    y = np.concatenate([np.ones(int(np.sum(m1)), dtype=np.int64), np.zeros(int(np.sum(m0)), dtype=np.int64)])
    return ids, y


def checks():
    out = {}
    blob = subprocess.run(['git', '-C', str(WORKTREE), 'show', '913eb5c91:nrm/fa/prc.py'], capture_output=True).stdout.decode()
    src = open(RESULTS / 'code' / 'edg' / 'source_prc.py').read()
    ok = True
    for fn in ('def auc(score, s):', 'def lookup_scores(train_cells, train_sex, test_cells, a=1.0):'):
        if not blob:                     # amendment A3: a clone restored from the branch bundle lacks commit 913eb5c91
            break
        a = blob[blob.index(fn):]
        a = a[:a.index('\n\n\n')]
        b = src[src.index(fn):]
        b = b[:b.index('\n\n\n')] if '\n\n\n' in b else b.rstrip('\n')
        ok &= (a == b)
    out['source_prc_text_equals_blob'] = bool(ok) if blob else 'git blob unavailable'
    rng = np.random.default_rng(5)
    good = True
    for _ in range(300):
        k = int(rng.integers(1, 7))
        m1, m0 = rng.integers(0, 5, k), rng.integers(0, 5, k)
        if rng.random() < 0.3 and k > 1:
            m1[1], m0[1] = 2 * m1[0], 2 * m0[0]                          # tied likelihood ratios
        if m1.sum() == 0 or m0.sum() == 0:
            continue
        ids, y = _people(m1, m0)
        v, IF = E.c_same(ids, y, k)
        ex = float(FL.auc_star_exact(m1, m0))
        en = float(FL.best_auc_by_enumeration(m1, m0))
        a, _ = E.auc_if(ids * 0.0 + (m1 / np.maximum(m1 + m0, 1))[ids], y)
        good &= abs(v - ex) < 1e-12 and abs(ex - en) < 1e-12 and abs(a - ex) < 1e-12 and abs(np.sum(IF)) < 1e-9
    out['c_same_equals_exact_and_enumeration_tiny_laws'] = bool(good)
    out['undefined_when_class_missing'] = bool(np.isnan(E.c_same(np.array([0, 1, 1]), np.array([1, 1, 1]), 2)[0])
                                               and np.isnan(E.auc_if(np.array([0.1, 0.2]), np.array([0, 0]))[0]))
    # ties score 1/2: two people, one per class, same score -> 1/2; known-score AUC equals enumeration of all pairs
    out['tie_is_half'] = E.auc_if(np.array([0.3, 0.3]), np.array([1, 0]))[0] == 0.5
    s = rng.integers(0, 4, 40).astype(float)
    y = (rng.random(40) < 0.5).astype(int)
    brute = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in s[y == 1] for b in s[y == 0]])
    out['known_score_auc_equals_pair_enumeration'] = bool(abs(E.auc_if(s, y)[0] - brute) < 1e-12)
    # DeLong: person-unit cluster SE equals the classical DeLong formula
    a, IF = E.auc_if(s, y)
    se = E.cluster_se(IF, np.arange(40))
    V1 = np.array([np.mean([1.0 if a_ > b else 0.5 if a_ == b else 0.0 for b in s[y == 0]]) for a_ in s[y == 1]])
    V0 = np.array([np.mean([1.0 if a_ > b else 0.5 if a_ == b else 0.0 for a_ in s[y == 1]]) for b in s[y == 0]])
    dl = np.sqrt(np.sum((V1 - a) ** 2) / len(V1) ** 2 + np.sum((V0 - a) ** 2) / len(V0) ** 2)
    out['cluster_se_person_units_equals_delong'] = bool(abs(se - dl * np.sqrt(40 / 39)) < 1e-12)
    # bank LOOKUP vectorized == source recipe
    ids = rng.integers(0, 30, 500)
    y = (rng.random(500) < 0.4).astype(np.int64)
    parts = rng.permutation(500) % 10
    m1p, m0p = np.full(30, 0.4 / 30), np.full(30, 0.6 / 30)
    keys = [bytes([i]) * 8 for i in range(30)]
    r = E.d_bank(ids, y, parts, 30, m1p, m0p, wire_keys=keys)
    out['d_bank_vectorized_equals_source'] = bool(r['source_checked'])
    out['independence_any_score_pop_auc_half'] = bool(abs(r['pop'] - 0.5) < 1e-12)
    # frozen laws: analytic relations
    laws = L.all_laws()
    out['laws_analytic_checks'] = bool(all(l['checks']['all_pass'] for l in laws.values()))
    # identical wires: COMP occupation cells 14, 15 are one observable value
    lc = laws[('COMP', 0.5, 'uniform')]
    out['comp_identical_wires_merged'] = bool(lc['views']['occupation']['K'] == 15 and lc['views']['pair']['K'] == 240)
    # household sampler: person-level marginal of (S, L16) equals the law; within-household S correlation negative
    lr = laws[('REF', 0.5, 'skewed')]
    c, s, unit = L.draw_households(lr, 300000, np.random.default_rng(np.random.SeedSequence(SIM.BASE_SEED + 3, spawn_key=(1,))))
    g = lr['views']['L1024']['inv'][c]
    emp = np.bincount(g * 2 + s, minlength=2 * lr['views']['L1024']['K']) / len(c)
    pop = np.column_stack((lr['views']['L1024']['m0'], lr['views']['L1024']['m1'])).reshape(-1)
    out['household_marginal_max_abs_dev'] = float(np.abs(emp - pop).max())
    out['household_marginal_ok'] = bool(np.abs(emp - pop).max() < 0.003)
    first = np.r_[0, np.flatnonzero(np.diff(unit)) + 1]
    second = first[(first + 1 < len(unit))]
    second = second[unit[second] == unit[second + 1]]
    out['household_second_member_opposite_share'] = float(np.mean(s[second] != s[second + 1]))
    out['all_pass'] = bool(all(v is True for k, v in out.items() if isinstance(v, bool)))
    return out


def smoke():
    t = {}
    for sc in SIM.scenarios():
        if sc['n'] != 20000 or sc['pi'] != 0.5 or sc['occ'] != 'skewed':
            continue
        law = L.build(sc['family'], sc['pi'], sc['occ'])
        keys = {k: SIM._wire_keys(v) for k, v in law['views'].items()}
        t0 = time.process_time()
        row = SIM.one_rep(sc, law, 900000, keys=keys)
        t1 = time.process_time()
        SIM.one_rep(sc, law, 900001)
        t[f"{sc['family']}{'*' if sc['clustered'] else ''}"] = {'cpu_s_rep_with_source_check': round(t1 - t0, 3),
                                                              'cpu_s_rep': round(time.process_time() - t1, 3),
                                                              'views': sorted(row['views']), 'cmp': sorted(row['cmp'])}
    return t


def main(argv):
    if argv and argv[0] == 'smoke':
        print(json.dumps(smoke(), indent=1))
        return
    res = checks()
    print(json.dumps(res, indent=1))
    with open(RESULTS / 'roles' / 'coordinator' / 'SELFTEST.json', 'w') as fh:
        json.dump(res, fh, indent=1)
    sys.exit(0 if res['all_pass'] else 1)
