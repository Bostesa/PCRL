"""V3 nested fixed-routing comparison (T1), own code, label-free.

From the private per-person files .private/<campaign>/fa_cache/nested/NEST_<cap>_V<v>.npz (keys 'unit', 'k', 'l',
'<CONT|A|B0|B1>__<metric>', metrics inc_nll, occ_nll, tot_nll, inc_brier, occ_brier, inc_acc, occ_acc; accuracy stored as
correctness) recompute every derivable number of results/<campaign>/fa/NEST_<cap>_V<v>.json:
- the file hash; the household column equals the admitted MODEL_FIT households in the admitted order;
- the outer folds: households ordered by SHA-256('NRM_FA_ncv_S<v>|<household>') and dealt into 5 parts (own implementation);
  held-out and training counts per fold;
- nested means of CONT, A, B0, B1 (every metric);
- spends over CONT of A, B0, B1 (loss(c) - loss(CONT); accuracy as a loss) with household-clustered SEs (own sandwich);
- the original capacity-gate label of every construction (estimate + 2 SE <= 0.004 NLL, <= 0.003 accuracy loss, per task);
- paired differences B1 - A, B0 - A, B1 - B0 (estimate, household-clustered SE);
- the fold level: the 5 outer-fold means of every paired difference, SD / sqrt(5), SE_used = max(household SE, fold SE), the
  number of negative folds; the per-fold record entries;
- the share of A's spend over CONT that B1 removes (occupation, income, total NLL);
- the GATE reproduction: CONT and A nested means and A's spend over CONT against
  results/pcrl_new_release_mechanisms_v1/roles/designer_a/GATE_V<v>_<KI>x<KO>.json (<= 1e-12) and its gate label;
- partition consistency (label-free): within every (outer fold, cell) each finite construction's NLL takes at most K values,
  brier + 2 exp(-nll) is constant (= sum_c p_c^2 + 1 of the cell's vector), the correctly decided people share one NLL that is
  the smallest in the cell, tot_nll = inc_nll + occ_nll; so the three constructions emit cell-constant vectors on the SAME stored
  held-out cells;
- the Q-PE nested reading and the construction choice from my own numbers (the MODEL_SELECT sign condition uses the recorded
  T3 paired costs, which V4 recomputes).

    python3 .../v3_nested.py --tag V1
    python3 .../v3_nested.py --selftest
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

NAMES = ('CONT', 'A', 'B0', 'B1')
CONS = ('A', 'B0', 'B1')
PAIRS = (('B1', 'A'), ('B0', 'A'), ('B1', 'B0'))
SPEND_KEYS = ('occ_nll', 'inc_nll', 'occ_acc', 'inc_acc', 'tot_nll', 'occ_brier', 'inc_brier')
GATE_KEYS = ('inc_nll', 'inc_brier', 'inc_acc', 'occ_nll', 'occ_brier', 'occ_acc')
TOL = 1e-12
N_OUTER = 5


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def diff_vec(store, a, b, key):
    """loss(a) - loss(b) per person; accuracy (stored as correctness) enters as a loss: acc(b) - acc(a)."""
    if key.endswith('_acc'):
        return store[b][key] - store[a][key]
    return store[a][key] - store[b][key]


def summarize(store, unit, folds):
    """All derivable numbers of a NEST record from per-person arrays (own code)."""
    out = {'nested_means': {nm: {k: float(store[nm][k].mean()) for k in V.NEST_METRICS} for nm in NAMES}}
    out['spend_over_cont'] = {c: {k: [float(diff_vec(store, c, 'CONT', k).mean()), V.cluster_se(diff_vec(store, c, 'CONT', k), unit)]
                                  for k in SPEND_KEYS} for c in CONS}
    out['original_capacity_gate'] = {c: ('ORIGINAL_CAPACITY_GATE_PASS' if all(out['spend_over_cont'][c][k][0] + 2 *
                                                                              out['spend_over_cont'][c][k][1] <= lim
                                                                              for k, lim in V.GATE_LIMITS.items())
                                         else 'ORIGINAL_CAPACITY_GATE_FAIL') for c in CONS}
    out['paired_differences'] = {f'{a}_minus_{b}': {k: [float(diff_vec(store, a, b, k).mean()), V.cluster_se(diff_vec(store, a, b, k), unit)]
                                                    for k in V.NEST_METRICS} for a, b in PAIRS}
    fl, per_fold = {}, []
    for o in range(N_OUTER):
        te = folds == o
        per_fold.append({'fold': o, 'heldout_people': int(te.sum()), 'train_people': int((~te).sum()),
                         'paired_fold_means': {f'{a}_minus_{b}': {k: float(diff_vec(store, a, b, k)[te].mean()) for k in V.NEST_METRICS}
                                               for a, b in PAIRS}})
    for a, b in PAIRS:
        nm = f'{a}_minus_{b}'
        fl[nm] = {}
        for k in V.NEST_METRICS:
            fm = np.array([f['paired_fold_means'][nm][k] for f in per_fold])
            sd = float(fm.std(ddof=1) / math.sqrt(len(fm)))
            fl[nm][k] = {'fold_means': fm.tolist(), 'sd_fold_means_over_sqrt_folds': sd,
                         'se_used': float(max(out['paired_differences'][nm][k][1], sd)), 'negative_folds': int((fm < 0).sum())}
    out['paired_differences_fold_level'] = fl
    out['folds'] = per_fold
    share = {}
    for k in ('occ_nll', 'inc_nll', 'tot_nll'):
        sa = out['spend_over_cont']['A'][k][0]
        share[k] = float((sa - out['spend_over_cont']['B1'][k][0]) / sa) if sa > 0 else None
    out['share_of_A_spend_over_cont_removed_by_B1'] = share
    return out


def compare_nest(mine, rec):
    """Max abs differences and mismatches between my summary and the recorded NEST json."""
    mm, worst = [], 0.0

    def upd(x, y, what):
        nonlocal worst
        if x is None or y is None:
            if x != y:
                mm.append({'what': what, 'mine': x, 'recorded': y})
            return
        d = abs(float(x) - float(y))
        worst = max(worst, d)
        if not d <= TOL:
            mm.append({'what': what, 'mine': x, 'recorded': y, 'abs_diff': d})
    for nm in NAMES:
        for k in V.NEST_METRICS:
            upd(mine['nested_means'][nm][k], rec['nested_means'][nm].get(k), f'nested_means.{nm}.{k}')
    for c in CONS:
        for k in SPEND_KEYS:
            r = rec['spend_over_cont'][c].get(k)
            upd(mine['spend_over_cont'][c][k][0], r[0] if r else None, f'spend_over_cont.{c}.{k}.est')
            upd(mine['spend_over_cont'][c][k][1], r[1] if r else None, f'spend_over_cont.{c}.{k}.se')
        if mine['original_capacity_gate'][c] != rec['original_capacity_gate'].get(c):
            mm.append({'what': f'original_capacity_gate.{c}', 'mine': mine['original_capacity_gate'][c],
                       'recorded': rec['original_capacity_gate'].get(c)})
    for nm, d in mine['paired_differences'].items():
        for k, (e, s) in d.items():
            r = rec['paired_differences'][nm].get(k)
            upd(e, r[0] if r else None, f'paired_differences.{nm}.{k}.est')
            upd(s, r[1] if r else None, f'paired_differences.{nm}.{k}.se')
    for nm, d in mine['paired_differences_fold_level'].items():
        for k, x in d.items():
            r = rec['paired_differences_fold_level'][nm][k]
            for i, (a, b) in enumerate(zip(x['fold_means'], r['fold_means'])):
                upd(a, b, f'fold_level.{nm}.{k}.fold_means[{i}]')
            if len(x['fold_means']) != len(r['fold_means']):
                mm.append({'what': f'fold_level.{nm}.{k}.n_folds'})
            upd(x['sd_fold_means_over_sqrt_folds'], r['sd_fold_means_over_sqrt_folds'], f'fold_level.{nm}.{k}.sd')
            upd(x['se_used'], r['se_used'], f'fold_level.{nm}.{k}.se_used')
            if x['negative_folds'] != r['negative_folds']:
                mm.append({'what': f'fold_level.{nm}.{k}.negative_folds', 'mine': x['negative_folds'], 'recorded': r['negative_folds']})
    for k, x in mine['share_of_A_spend_over_cont_removed_by_B1'].items():
        upd(x, rec['share_of_A_spend_over_cont_removed_by_B1'].get(k), f'share.{k}')
    for f_m, f_r in zip(mine['folds'], rec['folds']):
        for key in ('fold', 'heldout_people', 'train_people'):
            if f_m[key] != f_r.get(key):
                mm.append({'what': f'folds[{f_m["fold"]}].{key}', 'mine': f_m[key], 'recorded': f_r.get(key)})
        for nm, d in f_m['paired_fold_means'].items():
            for k, x in d.items():
                upd(x, f_r['paired_fold_means'][nm].get(k), f'folds[{f_m["fold"]}].paired_fold_means.{nm}.{k}')
        upd(f_m['paired_fold_means']['B1_minus_A']['occ_nll'], f_r.get('paired_fold_mean_B1_minus_A_occ_nll'),
            f'folds[{f_m["fold"]}].paired_fold_mean_B1_minus_A_occ_nll')
    if len(mine['folds']) != len(rec['folds']):
        mm.append({'what': 'number of folds', 'mine': len(mine['folds']), 'recorded': len(rec['folds'])})
    return mm, worst


def gate_reproduction(mine, gate):
    """CONT and A nested means and A's spend over CONT against the published GATE record (<= 1e-12), and its gate label."""
    diffs = []
    for nm_src, nm in (('CONT', 'CONT'), ('PRC_TWIN', 'A')):
        for k, x in gate['nested_means'][nm_src].items():
            diffs.append(abs(float(x) - mine['nested_means'][nm][k]))
    for k, (m, s) in gate['spend_twin_over_cont'].items():
        diffs.append(abs(float(m) - mine['spend_over_cont']['A'][k][0]))
        diffs.append(abs(float(s) - mine['spend_over_cont']['A'][k][1]))
    lab = 'PASS' if all(mine['spend_over_cont']['A'][k][0] + 2 * mine['spend_over_cont']['A'][k][1] <= lim
                        for k, lim in V.GATE_LIMITS.items()) else 'FAIL'
    mx = max(diffs)
    return {'max_abs_difference': mx, 'n_numbers': len(diffs), 'reproduced': bool(mx <= TOL and lab == gate.get('gate')),
            'my_gate_label': lab, 'record_gate_label': gate.get('gate'),
            'gate_keys_cover': sorted(gate['nested_means']['CONT']) == sorted(GATE_KEYS)}


def consistency(store, k, l, folds, ki, ko):
    """Label-free partition consistency of the finite constructions on the stored held-out cells (see the module docstring)."""
    res = {}
    rng_ok = bool((k >= 0).all() and (k < ki).all() and (l >= 0).all() and (l < ko).all())
    res['cells_in_range'] = rng_ok
    for c in CONS + ('CONT',):
        res[f'tot_equals_sum_{c}'] = bool(np.array_equal(store[c]['tot_nll'], store[c]['inc_nll'] + store[c]['occ_nll']))
    for c in CONS:
        for task, cells, Kc in (('inc', k, 2), ('occ', l, 5)):
            nll, br, acc = store[c][f'{task}_nll'], store[c][f'{task}_brier'], store[c][f'{task}_acc']
            key = folds * 1_000_000 + cells
            order = np.argsort(key, kind='stable')
            ks = key[order]
            bounds = np.flatnonzero(np.diff(ks)) + 1
            starts, ends = np.concatenate([[0], bounds]), np.concatenate([bounds, [len(ks)]])
            max_vals, max_inv_range, acc_bad, acc_min_bad = 0, 0.0, 0, 0
            inv = br + 2.0 * np.exp(-nll)
            for s, e in zip(starts, ends):
                idx = order[s:e]
                max_vals = max(max_vals, len(np.unique(nll[idx])))
                max_inv_range = max(max_inv_range, float(inv[idx].max() - inv[idx].min()))
                a1 = idx[acc[idx] == 1.0]
                a0 = idx[acc[idx] == 0.0]
                if len(a1):
                    if len(np.unique(nll[a1])) != 1:
                        acc_bad += 1
                    if len(a0) and nll[a0].min() < nll[a1].max() - 1e-12:
                        acc_min_bad += 1
            res[f'{c}_{task}'] = {'groups': int(len(starts)), 'max_distinct_nll': int(max_vals), 'allowed': Kc,
                                  'max_range_brier_plus_2exp_minus_nll': max_inv_range,
                                  'groups_correct_people_with_two_nll': int(acc_bad),
                                  'groups_correct_nll_not_smallest': int(acc_min_bad)}
    ok = rng_ok and all(v for kk, v in res.items() if kk.startswith('tot_equals_sum_'))
    for c in CONS:
        for task, Kc in (('inc', 2), ('occ', 5)):
            r = res[f'{c}_{task}']
            ok &= (r['max_distinct_nll'] <= Kc and r['max_range_brier_plus_2exp_minus_nll'] <= 1e-12 and
                   r['groups_correct_people_with_two_nll'] == 0 and r['groups_correct_nll_not_smallest'] == 0)
    res['ok'] = bool(ok)
    return res


def qpe_nested(mine_by_v, ms_occ=None):
    """Q-PE from my nested numbers: d_v (B1 - A occ NLL), SE_used_v; label with the recorded MODEL_SELECT signs (if given)."""
    d = {v: (m['paired_differences']['B1_minus_A']['occ_nll'][0], m['paired_differences_fold_level']['B1_minus_A']['occ_nll']['se_used'])
         for v, m in mine_by_v.items()}
    if any(x[0] >= 0 for x in d.values()):
        lab = 'PE_ERROR_NOT_SUPPORTED'
    elif all(x[0] + 2 * x[1] < 0 for x in d.values()):
        if ms_occ is None or any(ms_occ.get(v) is None for v in d):
            lab = 'INCOMPLETE'
        else:
            lab = 'PE_ERROR_SUPPORTED' if all(ms_occ[v] < 0 for v in d) else 'PE_ERROR_POINT_ONLY'
    else:
        lab = 'PE_ERROR_POINT_ONLY'
    choice = 'B1' if all(m['paired_differences']['B1_minus_A']['occ_nll'][0] < 0 and m['paired_differences']['B1_minus_A']['tot_nll'][0] < 0
                         for m in mine_by_v.values()) else 'A'
    return {'d_and_se_used': {str(v): list(x) for v, x in d.items()}, 'upper': {str(v): x[0] + 2 * x[1] for v, x in d.items()},
            'label': lab, 'construction_choice': choice, 'model_select_occ_nll_used': ms_occ}


def run(tag):
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import frc.common as C
    F = C.features('MODEL_FIT')
    unit_adm = np.asarray(F['unit'])
    checks, record = [], {'tag': tag, 'utc': V.utc(), 'per': {}}
    qdec_p = V.RESULTS / 'decisions' / 'QUAL_DECISION.json'
    qdec = json.loads(qdec_p.read_text()) if qdec_p.is_file() else None
    for cid in ('C64', 'C128', 'C256'):
        ki, ko = V.CAPACITIES[cid]
        mine_by_v = {}
        for v in (0, 1):
            npz = V.FA_CACHE / 'nested' / f'NEST_{cid}_V{v}.npz'
            jp = V.RESULTS / 'fa' / f'NEST_{cid}_V{v}.json'
            if not npz.is_file() or not jp.is_file():
                continue
            rec = json.loads(jp.read_text())
            raw = load_npz(npz)
            fsha = V.file_sha(npz)
            unit = np.asarray(raw['unit'])
            store = {nm: {k: np.asarray(raw[f'{nm}__{k}'], dtype=np.float64) for k in V.NEST_METRICS} for nm in NAMES}
            folds = V.household_folds(unit, N_OUTER, V.OUTER_TAG.format(v=v))
            mine = summarize(store, unit, folds)
            mine_by_v[v] = mine
            mm, worst = compare_nest(mine, rec)
            proc = []
            if fsha != rec.get('per_person_private_file_sha256'):
                proc.append('per-person file hash differs from the record')
            if not np.array_equal(unit, unit_adm):
                proc.append('household column differs from the admitted MODEL_FIT households')
            if rec.get('people') != len(unit) or rec.get('households') != len(np.unique(unit)):
                mm.append({'what': 'people / households', 'recorded': [rec.get('people'), rec.get('households')],
                           'mine': [len(unit), int(len(np.unique(unit)))]})
            if rec.get('outer_tag') != V.OUTER_TAG.format(v=v) or rec.get('outer_folds') != N_OUTER:
                proc.append('outer tag / folds differ')
            cons = consistency(store, np.asarray(raw['k']), np.asarray(raw['l']), folds, ki, ko)
            gate = None
            if cid in V.SOURCE_TWIN:
                gp = V.SRC_RESULTS / 'roles' / 'designer_a' / f'GATE_V{v}_{ki}x{ko}.json'
                gate = gate_reproduction(mine, json.loads(gp.read_text()))
                rg = (rec.get('source_reproduction') or {}).get('gate') or {}
                gate['record_says_reproduced'] = rg.get('reproduced')
                gate['record_max_abs_difference'] = rg.get('max_abs_difference')
                gate['gate_file_sha256'] = V.file_sha(gp)
            ok_num = not mm and cons['ok'] and (gate is None or (gate['reproduced'] and gate['record_says_reproduced'] is True))
            res = {'max_abs_difference_all_numbers': worst, 'mismatches': mm[:30], 'n_mismatches': len(mm), 'consistency': cons,
                   'gate_reproduction': gate, 'my_original_capacity_gate': mine['original_capacity_gate'],
                   'my_B1_minus_A': mine['paired_differences']['B1_minus_A'],
                   'my_fold_level_B1_minus_A_occ_nll': mine['paired_differences_fold_level']['B1_minus_A']['occ_nll'],
                   'my_share_removed': mine['share_of_A_spend_over_cont_removed_by_B1'],
                   'my_spend_over_cont_occ_nll': {c: mine['spend_over_cont'][c]['occ_nll'] for c in CONS}}
            record['per'][f'{cid}_V{v}'] = {'file_sha256': fsha, 'summary': mine, **res}
            checks.append({'id': f'V3.NEST.{cid}_V{v}__{tag}',
                           'what': f'NEST_{cid}_V{v}: every derivable number recomputed from the per-person file with own code '
                                   f'(means, spends over CONT with household-clustered SEs, original gate labels, paired '
                                   f'differences, own outer folds and fold-level SEs, shares, per-fold entries), label-free '
                                   f'partition consistency of A / B0 / B1 on the stored cells'
                                   + (', GATE reproduction (<= 1e-12)' if gate else ''),
                           'inputs': {V.rel(npz): fsha, V.rel(jp): V.file_sha(jp)},
                           'result': res, 'numerical_status': 'PASS' if ok_num else 'FAIL',
                           'procedural_status': 'PASS' if not proc else 'FAIL', 'procedural_findings': proc})
        if len(mine_by_v) == 2:
            ms_occ = None
            if qdec is not None:
                pc = (qdec.get('per_capacity') or {}).get(cid) or {}
                ms_occ = {v: ((pc.get('qpe') or {}).get('model_select_occ_nll') or {}).get(str(v),
                              ((pc.get('qpe') or {}).get('model_select_occ_nll') or {}).get(v)) for v in (0, 1)}
            q = qpe_nested(mine_by_v, ms_occ)
            rq = (((qdec or {}).get('per_capacity') or {}).get(cid) or {})
            rec_lab = (rq.get('qpe') or {}).get('label')
            rec_choice = (rq.get('construction_choice') or {}).get('choice')
            agree = (q['label'] == rec_lab) and (q['construction_choice'] == rec_choice)
            # the recorded SE_used and d per version
            rd = (rq.get('qpe') or {}).get('occ_nll') or {}
            dd = 0.0
            for v in (0, 1):
                r = rd.get(str(v)) or rd.get(v) or {}
                if r:
                    dd = max(dd, abs(r.get('estimate', math.nan) - q['d_and_se_used'][str(v)][0]),
                             abs(r.get('se_used', math.nan) - q['d_and_se_used'][str(v)][1]))
            record['per'][f'{cid}_QPE'] = {**q, 'recorded_label': rec_lab, 'recorded_choice': rec_choice, 'max_abs_diff_d_se': dd}
            checks.append({'id': f'V3.QPE.{cid}__{tag}',
                           'what': f'{cid}: Q-PE label (d_v + 2 SE_used < 0 both versions; MODEL_SELECT sign from the recorded T3 '
                                   f'paired costs) and the construction choice (nested B1 - A occupation and total NLL < 0 both '
                                   f'versions) from my own nested numbers, against QUAL_DECISION.json',
                           'inputs': V.inputs_sha([qdec_p]),
                           'result': {**q, 'recorded_label': rec_lab, 'recorded_choice': rec_choice, 'max_abs_diff_d_se': dd},
                           'numerical_status': 'PASS' if (agree and dd <= TOL) else 'FAIL',
                           'procedural_status': 'NOT_APPLICABLE'})
    V.write_record(f'V3_NESTED__{tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['procedural_status'], c['result'].get('n_mismatches'),
              c['result'].get('max_abs_difference_all_numbers'), c['result'].get('label'))
    return 0


# ---------------------------------------------------------------- invented-data selftest
def _fake(rng, n=1500, ki=4, ko=6):
    unit = np.repeat(np.arange(n), 1 + (rng.random(n) < 0.4))[:n].astype(np.int64)
    folds = V.household_folds(unit, N_OUTER, 'TEST_ncv')
    k, l = rng.integers(0, ki, n), rng.integers(0, ko, n)
    yi, yo = rng.integers(0, 2, n), rng.integers(0, 5, n)
    store = {}
    for nm in NAMES:
        if nm == 'CONT':
            pi, po = rng.dirichlet(np.ones(2), n), rng.dirichlet(np.ones(5), n)
        else:
            TI = {o: rng.dirichlet(np.ones(2) * 2, ki) for o in range(N_OUTER)}
            TO = {o: rng.dirichlet(np.ones(5) * 2, ko) for o in range(N_OUTER)}
            pi = np.array([TI[f][c] for f, c in zip(folds, k)])
            po = np.array([TO[f][c] for f, c in zip(folds, l)])
        s = {}
        for t, P, y, Kc in (('inc', pi, yi, 2), ('occ', po, yo, 5)):
            r = np.arange(n)
            s[f'{t}_nll'] = -np.log(P[r, y])
            s[f'{t}_brier'] = ((P - np.eye(Kc)[y]) ** 2).sum(1)
            s[f'{t}_acc'] = (P.argmax(1) == y).astype(np.float64)
        s['tot_nll'] = s['inc_nll'] + s['occ_nll']
        store[nm] = s
    return store, unit, folds, k, l


def selftest():
    rng = np.random.default_rng(7)
    store, unit, folds, k, l = _fake(rng)
    mine = summarize(store, unit, folds)
    rec = json.loads(json.dumps(mine))
    for f in rec['folds']:                    # the coordinator's records also carry this per-fold copy
        f['paired_fold_mean_B1_minus_A_occ_nll'] = f['paired_fold_means']['B1_minus_A']['occ_nll']
    ok = {'self_consistent': compare_nest(mine, rec)[0] == []}
    r0 = json.loads(json.dumps(rec))
    del r0['folds'][1]['paired_fold_mean_B1_minus_A_occ_nll']
    ok['flags_missing_fold_entry'] = bool(compare_nest(mine, r0)[0])
    r2 = json.loads(json.dumps(rec))
    r2['paired_differences']['B1_minus_A']['occ_nll'][1] += 1e-9
    ok['flags_se_change'] = bool(compare_nest(mine, r2)[0])
    r3 = json.loads(json.dumps(rec))
    r3['paired_differences_fold_level']['B1_minus_A']['occ_nll']['fold_means'][2] += 1e-9
    ok['flags_fold_mean_change'] = bool(compare_nest(mine, r3)[0])
    r4 = json.loads(json.dumps(rec))
    r4['original_capacity_gate']['B1'] = 'ORIGINAL_CAPACITY_GATE_PASS' if mine['original_capacity_gate']['B1'].endswith('FAIL') \
        else 'ORIGINAL_CAPACITY_GATE_FAIL'
    ok['flags_gate_label_change'] = bool(compare_nest(mine, r4)[0])
    # cluster SE equals a brute-force double loop
    d = diff_vec(store, 'B1', 'A', 'occ_nll')
    g = {}
    for u, x in zip(unit.tolist(), (d - d.mean()).tolist()):
        g[u] = g.get(u, 0.0) + x
    G = len(g)
    brute = math.sqrt(G / (G - 1) * sum(v * v for v in g.values())) / len(d)
    ok['cluster_se_matches_brute_force'] = abs(brute - mine['paired_differences']['B1_minus_A']['occ_nll'][1]) < 1e-15
    # accuracy enters as a loss
    ok['accuracy_as_loss'] = abs(mine['paired_differences']['B1_minus_A']['occ_acc'][0] -
                                 (store['A']['occ_acc'].mean() - store['B1']['occ_acc'].mean())) < 1e-15
    # consistency: passes on cell-constant vectors; flags a person whose vector differs from its cell
    cons = consistency(store, k, l, folds, 4, 6)
    ok['consistency_passes'] = cons['ok']
    bad = {nm: {kk: vv.copy() for kk, vv in s.items()} for nm, s in store.items()}
    i = int(np.flatnonzero(store['B1']['occ_acc'] == 0)[0])
    bad['B1']['occ_brier'][i] += 0.01
    ok['consistency_flags_inconsistent_person'] = not consistency(bad, k, l, folds, 4, 6)['ok']
    bad2 = {nm: {kk: vv.copy() for kk, vv in s.items()} for nm, s in store.items()}
    bad2['A']['tot_nll'][5] += 1e-9
    ok['consistency_flags_total'] = not consistency(bad2, k, l, folds, 4, 6)['ok']
    # gate reproduction on a self-made gate record; a 1e-9 change is flagged
    gate = {'gate': 'PASS' if all(mine['spend_over_cont']['A'][kk][0] + 2 * mine['spend_over_cont']['A'][kk][1] <= lim
                                  for kk, lim in V.GATE_LIMITS.items()) else 'FAIL',
            'nested_means': {'CONT': {kk: mine['nested_means']['CONT'][kk] for kk in GATE_KEYS},
                             'PRC_TWIN': {kk: mine['nested_means']['A'][kk] for kk in GATE_KEYS}},
            'spend_twin_over_cont': {kk: mine['spend_over_cont']['A'][kk] for kk in ('occ_nll', 'inc_nll', 'occ_acc', 'inc_acc')}}
    ok['gate_reproduced'] = gate_reproduction(mine, gate)['reproduced']
    gate['nested_means']['CONT']['occ_nll'] += 1e-9
    ok['gate_flags_change'] = not gate_reproduction(mine, gate)['reproduced']
    # folds: every household in exactly one fold, 5 non-empty folds
    ok['folds_household_disjoint'] = all(len(set(folds[unit == u])) == 1 for u in np.unique(unit)[:200]) and len(set(folds)) == 5
    # Q-PE ladder
    m0 = json.loads(json.dumps(mine))
    m0['paired_differences']['B1_minus_A']['occ_nll'][0] = -0.01
    m0['paired_differences']['B1_minus_A']['tot_nll'][0] = -0.01
    m0['paired_differences_fold_level']['B1_minus_A']['occ_nll']['se_used'] = 0.001
    q = qpe_nested({0: m0, 1: m0}, {0: -0.001, 1: -0.002})
    ok['qpe_supported'] = q['label'] == 'PE_ERROR_SUPPORTED' and q['construction_choice'] == 'B1'
    ok['qpe_point_only_when_ms_positive'] = qpe_nested({0: m0, 1: m0}, {0: -0.001, 1: 0.001})['label'] == 'PE_ERROR_POINT_ONLY'
    m1 = json.loads(json.dumps(m0))
    m1['paired_differences']['B1_minus_A']['occ_nll'][0] = 0.0
    ok['qpe_tie_not_supported'] = qpe_nested({0: m0, 1: m1}, {0: -1, 1: -1})['label'] == 'PE_ERROR_NOT_SUPPORTED'
    ok['all_ok'] = all(bool(x) for x in ok.values())
    return {kk: bool(x) for kk, x in ok.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())
