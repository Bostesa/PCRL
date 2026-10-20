"""V0 selftest of the verifier library on INVENTED data only (no real data, no label): every helper against a brute-force or
alternative formulation. Writes one check (V0.VLIB.SELFTEST__<tag>).

    python3 results/pcrl_new_release_mechanisms_v1/roles/verifier/code/v0_selftest.py --tag V1
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402


def brute_auc(y, s, w=None):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    num = den = 0.0
    for i in np.flatnonzero(y == 1):
        for j in np.flatnonzero(y == 0):
            num += w[i] * w[j] * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
            den += w[i] * w[j]
    return num / den


def brute_cluster_se(d, unit):
    d = np.asarray(d, float); n = len(d); m = d.mean()
    gs = sorted(set(unit.tolist()))
    tot = sum(sum(d[i] - m for i in range(n) if unit[i] == g) ** 2 for g in gs)
    G = len(gs)
    return math.sqrt(G / (G - 1) * tot) / n


def brute_subgroup_se(d, mask, unit):
    # delta method for the ratio R = mean(mask * d) / mean(mask): influence z_i = mask_i (d_i - R) / mean(mask)
    d = np.asarray(d, float); n = len(d); p = mask.mean(); R = (mask * d).sum() / mask.sum()
    gs = sorted(set(unit.tolist()))
    tot = 0.0
    for g in gs:
        tot += sum((mask[i] * (d[i] - R) / p) for i in range(n) if unit[i] == g) ** 2
    G = len(gs)
    return math.sqrt(G / (G - 1) * tot) / n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    rng = np.random.default_rng(20261020)
    ok = {}
    # AUC with ties versus brute force; weighted AUC by expansion versus brute force
    y = (rng.random(60) < 0.45).astype(int); s = np.round(rng.random(60), 1)
    ok['auc_ties_brute'] = abs(V.auc(y, s) - brute_auc(y, s)) < 1e-12
    w = rng.integers(0, 3, 60)
    w[np.flatnonzero(y == 1)[0]] = 1; w[np.flatnonzero(y == 0)[0]] = 1
    ok['auc_expanded_brute_weighted'] = abs(V.auc_expanded(y, s, w) - brute_auc(y, s, w)) < 1e-12
    ok['auc_perfect'] = V.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    ok['auc_all_tied_half'] = V.auc(np.array([0, 1, 0, 1]), np.zeros(4)) == 0.5
    # clustered and subgroup SEs versus explicit loops
    unit = np.repeat(np.arange(25), rng.integers(1, 4, 25))
    n = len(unit)
    d = rng.normal(size=n)
    ok['cluster_se_brute'] = abs(V.cluster_se(d, unit) - brute_cluster_se(d, unit)) < 1e-12
    mask = rng.random(n) < 0.5
    ok['subgroup_se_brute'] = abs(V.subgroup_se(d, mask, unit) - brute_subgroup_se(d, mask, unit)) < 1e-12
    ok['subgroup_all_equals_cluster'] = abs(V.subgroup_se(d, np.ones(n, bool), unit) - V.cluster_se(d, unit)) < 1e-12
    ok['cluster_se_singletons_iid'] = abs(V.cluster_se(d, np.arange(n)) - math.sqrt(n / (n - 1) * ((d - d.mean()) ** 2).sum()) / n) < 1e-12
    # household folds: whole households, balanced, deterministic, equal to an argsort formulation
    part = V.household_part(unit, 4, 'TEST_tag')
    ok['folds_whole_households'] = all(len(set(part[unit == g].tolist())) == 1 for g in np.unique(unit))
    per = np.bincount([part[unit == g][0] for g in np.unique(unit)], minlength=4)
    ok['folds_balanced_households'] = per.max() - per.min() <= 1
    groups = np.unique(unit)
    keys = np.array([hashlib.sha256(f'TEST_tag|{g}'.encode()).hexdigest() for g in groups])
    order = np.argsort(keys, kind='stable'); fog = np.empty(len(groups), int); fog[order] = np.arange(len(groups)) % 4
    ok['folds_equal_argsort_formulation'] = bool(np.array_equal(fog[np.searchsorted(groups, unit)], part))
    rr = V.reader_roles(unit, 'R1')
    ok['reader_roles_partition'] = bool(np.array_equal(rr['FIT'] ^ rr['VALIDATION'] ^ rr['SCORE'], np.ones(n, bool))
                                        and not (rr['FIT'] & rr['SCORE']).any())
    # per-person losses on a hand example
    p = np.array([[0.2, 0.8], [0.5, 0.5], [1.0, 0.0]]); yy = np.array([1, 1, 1])
    L = V.person_losses(yy, p, 2)
    ok['losses_err_first_max'] = L['err'].tolist() == [0.0, 1.0, 1.0]             # tie [0.5, 0.5] -> class 0 (first maximum)
    ok['losses_nll_floor'] = abs(L['nll'][2] + math.log(1e-12)) < 1e-9 and abs(L['nll'][0] + math.log(0.8)) < 1e-15
    ok['losses_brier'] = abs(L['brier'][0] - (0.2 ** 2 + 0.2 ** 2)) < 1e-15 and abs(L['brier'][2] - 2.0) < 1e-15
    q = rng.dirichlet(np.ones(5), 300)
    ok['argmax_first_equals_numpy'] = bool(np.array_equal(V.argmax_first(q), np.argmax(q, 1)))
    ok['wire_sha_definition'] = V.wire_sha(q) == hashlib.sha256(np.ascontiguousarray(q).tobytes()
                                                                  + np.argmax(q, 1).astype(np.int64).tobytes()).hexdigest()
    ok['arr_sha_definition'] = V.arr_sha(np.zeros(3, np.int64)) == hashlib.sha256(b'int64(3,)' + bytes(24)).hexdigest()
    # conditions table: identical release -> all values and SEs zero; constant draws equal deterministic
    m = 400; unit2 = np.arange(m) // 2; sex = rng.integers(0, 2, m)
    labels = {'income': rng.integers(0, 2, m), 'occupation': rng.integers(0, 5, m)}
    refs = {'U0': {'income': rng.dirichlet(np.ones(2), m), 'occupation': rng.dirichlet(np.ones(5), m)},
            'Ucal': {'income': rng.dirichlet(np.ones(2), m), 'occupation': rng.dirichlet(np.ones(5), m)}}
    same = V.conditions_table(labels, refs['U0'], refs, sex, unit2)
    ok['identical_to_U0_zero'] = all(same[n]['value'] == 0.0 and same[n]['se'] == 0.0 for n in V.CONDITIONS if '|U0|' in n or n.startswith('guard'))
    rel = {'income': rng.dirichlet(np.ones(2), m), 'occupation': rng.dirichlet(np.ones(5), m)}
    det = V.conditions_table(labels, rel, refs, sex, unit2)
    st = V.conditions_table(labels, {t: np.stack([rel[t]] * 3) for t in V.TASKS}, refs, sex, unit2)
    ok['constant_draws_equal_deterministic'] = all(abs(det[n]['value'] - st[n]['value']) < 1e-15 and st[n]['se_mc'] < 1e-15 for n in V.CONDITIONS)
    st2 = V.conditions_table(labels, {'income': np.stack([rel['income'], refs['U0']['income']]), 'occupation': rel['occupation']},
                             refs, sex, unit2)
    v_a = V.conditions_table(labels, rel, refs, sex, unit2)['income|U0|nll_excess']['value']
    ok['draw_average'] = abs(st2['income|U0|nll_excess']['value'] - 0.5 * v_a) < 1e-15
    ok['accuracy_loss_sign'] = abs(det['income|U0|accuracy_loss']['value']
                                   - (V.person_losses(labels['income'], rel['income'], 2)['err'].mean()
                                      - V.person_losses(labels['income'], refs['U0']['income'], 2)['err'].mean())) < 1e-15
    # reserve: inclusive boundary, floor when own SE is zero, fails just below
    conds = {n: {'value': 0.0, 'se': 0.0, 'cap': V.GUARD_CAP if n.startswith('guard') else V.CAPS[n.split('|')[2]]} for n in V.CONDITIONS}
    floor = {n: 0.001 for n in V.CONDITIONS}
    conds['guard|sex0']['value'] = 0.005 - 2 * 0.001
    r1 = V.reserve_version(conds, floor, 5306)
    ok['reserve_inclusive'] = r1['met'] and r1['binding'] == 'guard|sex0' and abs(r1['min_ratio'] - 2.0) < 1e-12
    conds['guard|sex0']['value'] = 0.005 - 2 * 0.001 + 1e-9
    ok['reserve_fail_just_below'] = not V.reserve_version(conds, floor, 5306)['met']
    ok['reserve_floor_scale'] = abs(V.reserve_version(conds, floor, 5306 * 4)['conditions']['guard|sex0']['floor_se'] - 0.0005) < 1e-15
    try:
        V.reserve_version(conds, {**floor, 'guard|sex1': 0.0}, 5306); ok['reserve_zero_floor_refused'] = False
    except ValueError:
        ok['reserve_zero_floor_refused'] = True
    # winner pick: exact ties to the earlier candidate; near-tie flag
    c = [('A', 0.6, 'a', 'source'), ('B', 0.7, 'b', 'source'), ('C', 0.7, 'c', 'new'), ('D', 0.695, 'd', 'new')]
    w1 = V.pick(c)
    ok['pick_tie_to_earlier'] = w1['candidate'] == 'B' and w1['gap'] == 0.0 and w1['near_tie']
    ok['pick_clear'] = V.pick([('A', 0.6, 'a', 's'), ('B', 0.65, 'b', 's')])['near_tie'] is False
    # bootstrap multiplicities: per draw the household counts sum to G and members of one household share the count
    us = np.repeat(np.arange(30), 2)
    bc = V.boot_counts(us, 7, B=5)
    ok['boot_household_shared'] = bool((bc[:, 0::2] == bc[:, 1::2]).all()) and all(bc[b, 0::2].sum() == 30 for b in range(5))
    ok['boot_reproducible'] = bool(np.array_equal(bc, V.boot_counts(us, 7, B=5)))
    # cache key text and seeds
    ok['cache_key_text'] = V.cache_key('R3', 5, 1.0, 'w', 's') == hashlib.sha256(b'NRM1-devgrid-v1|R3:NRM1_devgrid_R3|5|1.0|w|s').hexdigest()
    ok['reader_seeds'] = [V.reader_seed(v, k) for v in (0, 1) for k in (0, 1)] == [2026101805, 2026101806, 2026101815, 2026101816]
    ok['pair_candidate_order'] = [x[0] for x in V.pair_candidates({v: {n: 0.5 for n in V.RECIPES} for v in V.VIEWS})][:9] == \
        list(V.SRC8) + ['IGNORE_occupation__OUT_LR']
    all_ok = all(ok.values())
    chk = {'id': f'V0.VLIB.SELFTEST__{a.tag}', 'what': 'verifier library self-test on invented data (AUC, weighted AUC, clustered and '
           'subgroup SEs, household folds, losses, decisions, wire / array hashes, clause table and draw averaging, reserve edges, '
           'winner pick, bootstrap multiplicities, cache key, reader seeds)',
           'inputs': V.inputs_sha([V.CODE_DIR / 'vlib.py', Path(__file__)]), 'result': ok,
           'numerical_status': 'PASS' if all_ok else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps(ok, indent=1, default=V._jd))
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
