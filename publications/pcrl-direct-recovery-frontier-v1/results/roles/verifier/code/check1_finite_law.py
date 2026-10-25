"""Verifier check 1 (DRF1): the finite-law objective of the saved DR fits, recomputed with separate code.

    python3 -m drf.jobqueue run --unit DRF_VER_C1_<rev>_V<v>_V<k> --stage verify --rss-gib 2 -- \
        python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check1_finite_law.py --version V --revision v1|R1

What is recomputed (verifier code only: vlib.py; data access through drf.common.load('MODEL_FIT', ..., 'verify', ...), the pinned
frozen files read by path and SHA-256, saved arrays through drf.release.load / frc.release.load, frozen teachers applied with
nrm.fa.prc._proba on ohn.fam_a.compact_po features):
  A. sources: SHA-256 of every pinned input against drf/dr.py SOURCE_PINS (parsed as text, not imported); row-set hashes; budgets
     recomputed from the pinned FRC1 utility ledger and compared with dr/BUDGETS.json and the fit record.
  B. MODEL_FIT people routed by their OUT-OF-FOLD statistics: nearest twin codeword (cross-entropy), pi quartile bins, fine cells;
     identity B1 tables against the twin's stored tables (a routing check).
  C. per saved configuration (MODEL.pkl): my B1 tables from the saved A_inc / A_occ against the saved T_inc / T_occ; the emitted law
     with identical vectors merged; AUC* (pair, income, occupation; estimated and empirical law), plug-in MI, the 8 spends over the
     identity (v1: realized labels; R1: expected costs under the frozen out-of-fold teachers, and the realized spends reported
     beside), feasibility and the binding constraint; compared with the fit record and the RELEASE.json records (tol 1e-9).
  D. the SEX-blind coarsening path recomputed independently on all MODEL_FIT (v1: realized NLL; R1: expected NLL); its spends and
     in-sample values against path_full; the saved CTRL points' partitions against my path at their recorded steps.
  E. deployment: the release recomputed from MODEL_SELECT (and MODEL_FIT) features alone, bitwise against RELEASE.npz; its rows are
     rows of the saved tables at the atoms of the recomputed fine cells; the twin identity reproduces FRC1's TWB_C64 wires.
  F. the CTRL bracket selection recomputed from the nested records (no label needed).
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import ast
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

CAMPAIGN = 'pcrl_direct_recovery_frontier_v1'
RES = REPO / 'results' / CAMPAIGN
PRIV = REPO / '.private' / CAMPAIGN
OUT = RES / 'roles' / 'verifier' / 'out'
VIEWS = (('pair', 'pair'), ('inc', 'income'), ('occ', 'occupation'))
REVISION = {'v1': {'fit': 'FIT_V{v}.json', 'suffix': '', 'estimator': 'realized'},
            'R1': {'fit': 'FIT_R1_V{v}.json', 'suffix': '_R1', 'estimator': 'expected'},
            'R2': {'fit': 'FIT_R2_V{v}.json', 'suffix': '_R2', 'estimator': 'expected', 'superset_of': 'R1'}}


def parse_pins():
    """SOURCE_PINS and LEDGER_REL of drf/dr.py, read as text (ast), not imported."""
    tree = ast.parse((REPO / 'drf' / 'dr.py').read_text())
    consts, pins = {}, None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if isinstance(node.value, ast.Constant):
                consts[name] = node.value.value
            if name == 'SOURCE_PINS':
                pins = {}
                for k, v in zip(node.value.keys, node.value.values):
                    key = k.value if isinstance(k, ast.Constant) else consts[k.id]
                    pins[key] = v.value
    return pins, consts


def src_path(key):
    R = REPO / '.private'
    if key.startswith('nrm_fa_cache/'):
        return R / 'pcrl_new_release_mechanisms_v1' / 'fa_cache' / key.split('/', 1)[1]
    if key.startswith('frc_fa_cache/'):
        return R / 'pcrl_finite_routing_comparison_v1' / 'fa_cache' / key.split('/', 1)[1]
    if key.startswith('frc_runs/'):
        return R / 'pcrl_finite_routing_comparison_v1' / 'runs' / key.split('/', 1)[1]
    return REPO / key


def pinned_pickle(key, pins, items):
    b = src_path(key).read_bytes()
    items.append(V.cmp_eq(f'A:pin_sha256:{key}', hashlib.sha256(b).hexdigest(), pins.get(key)))
    return pickle.loads(b)


def budgets_from_ledger(path, v):
    rows = [r for r in csv.DictReader(open(path)) if r['config_id'] == 'TWB_C64' and int(r['version']) == v]
    assert len(rows) == 1
    r = {k: float(x) for k, x in rows[0].items() if '|' in k}
    sl = {}
    for t, key in (('inc', 'income'), ('occ', 'occupation')):
        sl[f'{t}_nll'] = 0.010 - max(r[f'{key}|U0|nll_excess'], r[f'{key}|Ucal|nll_excess'])
        sl[f'{t}_brier'] = 0.005 - max(r[f'{key}|U0|brier_excess'], r[f'{key}|Ucal|brier_excess'])
        sl[f'{t}_acc'] = 0.010 - max(r[f'{key}|U0|accuracy_loss'], r[f'{key}|Ucal|accuracy_loss'])
    sl['occ_brier_men'] = 0.005 - r['guard|sex0']
    sl['occ_brier_women'] = 0.005 - r['guard|sex1']
    return sl, {k: 0.6 * x for k, x in sl.items()}


def load_model(cid, v, items):
    d = PRIV / 'runs' / 'DR' / cid / f'V{v}'
    meta = json.loads((d / 'MODEL.json').read_text())
    b = (d / 'MODEL.pkl').read_bytes()
    items.append(V.cmp_eq(f'C:{cid}:model_sha256_vs_MODEL.json', hashlib.sha256(b).hexdigest(), meta['model_sha256']))
    M = pickle.loads(b)
    items.append(V.cmp_eq(f'C:{cid}:payload_identity', [M['config_id'], int(M['version'])], [cid, v]))
    return M, meta


class Ctx:
    """MODEL_FIT inputs shared by every recomputation (out-of-fold routing, identity vectors, label distributions)."""


def insample_values(cx, A_inc, A_occ, T_saved, exact):
    """Everything the fit record reports in-sample, recomputed: atoms, merged emitted wires, both laws, spends (realized and expected)."""
    sex = cx.lab['sex']
    ai, ao = A_inc[cx.ic], A_occ[cx.oc]
    TI, _ = V.b1_tables(ai, cx.qI, V.CAPACITY[0], cx.CI)
    TO, _ = V.b1_tables(ao, cx.qO, V.CAPACITY[1], cx.CO)
    out = {'tables_vs_saved_max_abs': None}
    if T_saved is not None:
        out['tables_vs_saved_max_abs'] = max(float(np.abs(TI - T_saved[0]).max()), float(np.abs(TO - T_saved[1]).max()))
        Ti_emit, To_emit = T_saved
    else:
        Ti_emit, To_emit = TI, TO
    Vi, Vo = Ti_emit[ai], To_emit[ao]
    ids_merged = {'inc': V.row_ids(Vi), 'occ': V.row_ids(Vo)}
    ids_merged['pair'] = V.pair_ids(ids_merged['inc'], ids_merged['occ'])
    ids_atom = {'inc': ai, 'occ': ao, 'pair': V.pair_ids(ai, ao)}
    out['counts'] = {'atoms_used_inc': int(len(np.unique(ai))), 'atoms_used_occ': int(len(np.unique(ao))),
                     'pair_atoms_used': int(len(np.unique(ids_atom['pair']))),
                     'distinct_wires': {'income': int(len(np.unique(ids_merged['inc']))),
                                        'occupation': int(len(np.unique(ids_merged['occ']))),
                                        'pair': int(len(np.unique(ids_merged['pair'])))}}
    for law, w1 in (('estimated', cx.pi), ('empirical', sex.astype(np.float64))):
        rec = {'atom': {}, 'merged': {}}
        for short, _ in VIEWS:
            rec['atom'][short] = V.law_values(ids_atom[short], w1, exact_counts=(exact and law == 'empirical'))
            rec['merged'][short] = V.law_values(ids_merged[short], w1, exact_counts=(exact and law == 'empirical'))
        for kind in ('atom', 'merged'):
            rec[kind]['J_auc'] = sum(rec[kind][s]['auc'] for s, _ in VIEWS) - 1.5
            rec[kind]['J_mi'] = sum(rec[kind][s]['mi'] for s, _ in VIEWS)
        out[law] = rec
    y_i, y_o = cx.lab['income'], cx.lab['occupation']
    out['spend'] = {
        'realized_own_tables': V.spends8(TI[ai], TO[ao], cx.Vi0[0], cx.Vo0[0], y_i, y_o, sex),
        'realized_saved_tables_vs_twin_dec': V.spends8(Vi, Vo, cx.Vi0[1], cx.Vo0[1], y_i, y_o, sex),
        'expected_own_tables': V.spends8_expected(TI[ai], TO[ao], cx.Vi0[0], cx.Vo0[0], cx.qI, cx.qO, cx.Lg, sex),
        'expected_saved_tables_vs_twin_dec': V.spends8_expected(Vi, Vo, cx.Vi0[1], cx.Vo0[1], cx.qI, cx.qO, cx.Lg, sex)}
    return out


def compare_insample(prefix, mine, rec, bud, est, items, tol=V.TOL):
    """mine: insample_values(...); rec: {law: record dict} in the drf.dr._insample layout; est: the fit's constraint estimator."""
    for law in ('estimated', 'empirical'):
        r = (rec or {}).get(law)
        if r is None:
            items.append({'name': f'{prefix}:{law}:record_missing', 'result': 'FAIL'})
            continue
        m = mine[law]
        for short, _ in VIEWS:
            items.append(V.cmp_num(f'{prefix}:{law}:auc_{short}_atoms', m['atom'][short]['auc'], r.get(f'auc_{short}'), tol))
            items.append(V.cmp_num(f'{prefix}:{law}:mi_{short}_atoms', m['atom'][short]['mi'], r.get(f'mi_{short}'), tol))
            obs = r.get('observable_merged_wires', {})
            items.append(V.cmp_num(f'{prefix}:{law}:auc_{short}_merged_wires', m['merged'][short]['auc'], obs.get(f'auc_{short}'), tol))
            items.append(V.cmp_num(f'{prefix}:{law}:auc_{short}_closed_vs_sorted_form', m['merged'][short]['auc'],
                                   m['merged'][short]['auc_sorted'], 1e-12))
            if 'auc_exact_rational' in m['merged'][short]:
                items.append(V.cmp_num(f'{prefix}:{law}:auc_{short}_float_vs_exact_rational', m['merged'][short]['auc'],
                                       m['merged'][short]['auc_exact_rational'], 1e-12))
        items.append(V.cmp_num(f'{prefix}:{law}:J_auc_atoms', m['atom']['J_auc'], r.get('J_auc'), tol))
        items.append(V.cmp_num(f'{prefix}:{law}:J_mi_atoms', m['atom']['J_mi'], r.get('J_mi'), tol))
        items.append(V.cmp_num(f'{prefix}:{law}:J_auc_merged_wires', m['merged']['J_auc'],
                               r.get('observable_merged_wires', {}).get('J_auc'), tol))
        for key in ('atoms_used_inc', 'atoms_used_occ', 'pair_atoms_used'):
            items.append(V.cmp_eq(f'{prefix}:{law}:{key}', mine['counts'][key], r.get(key)))
        items.append(V.cmp_eq(f'{prefix}:{law}:distinct_wires', mine['counts']['distinct_wires'],
                              r.get('observable_merged_wires', {}).get('distinct_wires')))
        items.append(V.cmp_eq(f'{prefix}:{law}:constraint_estimator', r.get('constraint_estimator', 'realized'), est))
        prim = mine['spend'][f'{est}_own_tables']
        sp = r.get('spend', {})
        for k in V.METRICS8:
            items.append(V.cmp_num(f'{prefix}:{law}:spend_{est}_{k}', prim[k], sp.get(k), tol))
            items.append(V.cmp_num(f'{prefix}:{law}:spend_{est}_{k}_saved_tables', mine['spend'][f'{est}_saved_tables_vs_twin_dec'][k],
                                   sp.get(k), tol))
        if est != 'realized':
            sr = r.get('spend_realized', {})
            for k in V.METRICS8:
                items.append(V.cmp_num(f'{prefix}:{law}:spend_realized_beside_{k}', mine['spend']['realized_own_tables'][k],
                                       sr.get(k), tol))
        ratio = {k: prim[k] / bud[k] for k in V.METRICS8}
        feas = all(prim[k] <= bud[k] + 1e-12 for k in V.METRICS8)
        items.append(V.cmp_eq(f'{prefix}:{law}:feasible', feas, r.get('feasible')))
        items.append(V.cmp_eq(f'{prefix}:{law}:binding', max(ratio, key=ratio.get), r.get('binding')))
        for k in V.METRICS8:
            items.append(V.cmp_num(f'{prefix}:{law}:spend_over_budget_{k}', ratio[k], r.get('spend_over_budget', {}).get(k), tol))


def emit_from_features(M, Z, prc):
    qI, qO = prc._proba(M['teachers']['inc'], Z), prc._proba(M['teachers']['occ'], Z)
    pi = np.clip(prc._proba(M['sex_teacher'], Z)[:, 1], *V.PI_CLIP)
    k, gk = V.nearest_codeword(qI, M['book']['CI'])
    l, gl = V.nearest_codeword(qO, M['book']['CO'])
    m = V.pi_bins(pi, M['pi_cuts'])
    ic, oc = V.fine_cells(k, l, m)
    ai, ao = np.asarray(M['A_inc'])[ic], np.asarray(M['A_occ'])[oc]
    return {'income': np.asarray(M['T_inc'])[ai], 'occupation': np.asarray(M['T_occ'])[ao], 'ai': ai, 'ao': ao, 'ic': ic, 'oc': oc,
            'qI': qI, 'qO': qO, 'pi': pi, 'min_gap_inc': float(gk.min()), 'min_gap_occ': float(gl.min())}


def bracket_selection(fit, sfx, items):
    """F: the CTRL bracket points recomputed from the nested records (the brackets grid and the nested path spends)."""
    br, nest = fit.get('brackets', {}), fit.get('nested', {})
    out = {}
    cand = 'DR_AUC' + sfx
    s_c = nest.get('entries', {}).get(cand, {}).get('spend_over_identity', {}).get('total_nll', [None])[0]
    items.append(V.cmp_num('F:candidate_spend_vs_nested_entry', br.get('candidate_spend'), s_c))
    grid = br.get('grid', [])
    path_sp = nest.get('path', {}).get('spend_over_identity', {})
    for g in grid:
        rec = path_sp.get(str(g['step']), {}).get('total_nll', [None])[0]
        items.append(V.cmp_num(f'F:grid_spend_step{g["step"]}_vs_nested_path', g['spend'], rec))
    items.append(V.cmp_eq('F:grid_covers_nested_path_steps', len(grid), nest.get('path', {}).get('steps_common')))
    s = br.get('candidate_spend')
    q_lo = [g for g in grid if g['spend'] <= s]
    q_hi = [g for g in grid if g['spend'] >= s + 2 * g['se_vs_candidate']]
    lo_idx = max([g['step'] for g in q_lo], default=None)
    hi_idx = min([g['step'] for g in q_hi], default=None)
    lo_sp = max(q_lo, key=lambda g: (g['spend'], g['step']))['step'] if q_lo else None
    hi_sp = min(q_hi, key=lambda g: (g['spend'], g['step']))['step'] if q_hi else None
    items.append(V.cmp_eq('F:lower_step_rule_largest_step_index', lo_idx, br.get('lower_step')))
    items.append(V.cmp_eq('F:upper_step_rule_smallest_step_index', hi_idx, br.get('upper_step')))
    out['lower_by_largest_spend'] = lo_sp
    out['upper_by_smallest_spend'] = hi_sp
    out['index_and_spend_readings_agree'] = bool(lo_sp == lo_idx and hi_sp == hi_idx)
    spends = [g['spend'] for g in sorted(grid, key=lambda g: g['step'])]
    out['nested_path_spend_monotone_nondecreasing'] = bool(all(b >= a for a, b in zip(spends, spends[1:])))
    out['nested_path_spend_decreasing_steps'] = [int(i + 1) for i, (a, b) in enumerate(zip(spends, spends[1:])) if b < a]
    out['candidate_nested_total_nll_spend'] = s
    out['lower'] = next(({'step': g['step'], 'spend': g['spend'], 'se_vs_candidate': g['se_vs_candidate']} for g in grid
                         if g['step'] == lo_idx), None)
    out['upper'] = next(({'step': g['step'], 'spend': g['spend'], 'se_vs_candidate': g['se_vs_candidate']} for g in grid
                         if g['step'] == hi_idx), None)
    items.append(V.cmp_eq('F:path_may_not_bracket_flag', hi_idx is None, br.get('path_may_not_bracket')))
    items.append(V.cmp_eq('F:selection_data', br.get('selection_data'), 'MODEL_FIT_NESTED'))
    alias = fit.get('ctrl_alias', {})
    for cid, step in (('DR_CTRL_LO' + sfx, br.get('lower_step')), ('DR_CTRL_HI' + sfx, br.get('upper_step'))):
        expect = 'NOT_AVAILABLE' if step is None else ('TWIN' if step == 0 else 'SAVED')
        got = alias.get(cid, 'SAVED' if cid in fit.get('saved', []) else None)
        items.append(V.cmp_eq(f'F:{cid}_disposition', got, expect))
    # fineness (RULES 4.4, my reading; reported, never gating): s = DR's nested spend, SE = the upper point's paired SE
    if s is not None and lo_idx is not None and hi_idx is not None:
        g_lo = next(g for g in grid if g['step'] == lo_idx)
        g_hi = next(g for g in grid if g['step'] == hi_idx)
        se = g_hi['se_vs_candidate']
        if s <= 0:
            fine = 'NOT_APPLICABLE_CAND_NOT_ABOVE_TWIN'
        else:
            below_ok = 0.75 * s <= g_lo['spend'] <= s
            above_ok = s + 2 * se <= g_hi['spend'] <= s + 2 * se + 0.25 * s
            fine = 'FINE' if below_ok and above_ok else ('COARSE_BOTH' if not below_ok and not above_ok else
                                                         'COARSE_BELOW' if not below_ok else 'COARSE_ABOVE')
        out['fineness_verifier_reading'] = fine
    elif s is not None and hi_idx is None:
        out['fineness_verifier_reading'] = 'CONTROL_PATH_MAY_NOT_BRACKET'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--versions', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--revisions', nargs='+', default=['v1', 'R1'], choices=tuple(REVISION))
    a = ap.parse_args()
    from drf import common as C                      # data access only (label layer, features, row hashes)
    data = C.load('MODEL_FIT', ('income', 'occupation', 'sex'), 'verify',
                  f'independent verifier check 1 (finite-law objective of the DR fits, revisions {a.revisions}, versions '
                  f'{a.versions}): out-of-fold fine cells, AUC*, plug-in MI and the 8 spends recomputed with separate code')
    res = {}
    for rev in a.revisions:
        for v in a.versions:
            if not (RES / 'dr' / REVISION[rev]['fit'].format(v=v)).is_file():
                res[f'{rev}_V{v}'] = {'overall': {'result': 'FAIL', 'n': 1, 'n_fail': 1, 'failed': ['fit_record_missing']}}
                OUT.mkdir(parents=True, exist_ok=True)
                V.dump(OUT / f'{os.environ.get("DRF_UNIT_ID") or "LOCAL"}_check1_{rev}_V{v}.json',
                       {'schema': 'DRF1-verifier-check1-v2', 'info': {'unit_id': os.environ.get('DRF_UNIT_ID'), 'revision': rev,
                                                                      'version': v},
                        'summary': {'A': {'result': 'FAIL', 'n': 1, 'n_fail': 1, 'failed': ['A:fit_record_missing']}},
                        'overall': {'result': 'FAIL', 'n': 1, 'n_fail': 1, 'failed': ['A:fit_record_missing']},
                        'items': [{'name': 'A:fit_record_missing', 'result': 'FAIL'}]})
                continue
            res[f'{rev}_V{v}'] = run_one(rev, v, data)
    print(json.dumps(V.jsonable(res), indent=1))
    return 0                                          # failed checks are recorded in the outputs, never raised


def run_one(revision, v, data):
    rv = REVISION[revision]
    sfx, est = rv['suffix'], rv['estimator']
    t0 = time.process_time()
    unit = os.environ.get('DRF_UNIT_ID')
    from drf import common as C                      # data access only (features, row hashes)
    from drf import release as RL                    # saved arrays only
    from frc import release as FRL                   # the twin's saved arrays only
    from nrm.fa import prc as PRC                    # frozen teachers applied with PRC._proba only
    from ohn import fam_a as FA                      # compact_po only
    items, info = [], {'unit_id': unit, 'version': v, 'revision': revision, 'constraint_estimator': est, 'utc_start': C.now()}
    pins, consts = parse_pins()
    fit_path = RES / 'dr' / rv['fit'].format(v=v)
    fit = json.loads(fit_path.read_text())
    info['fit_record'] = str(fit_path.relative_to(REPO))
    info['fit_record_sha256'] = V.sha256_file(fit_path)
    items.append(V.cmp_eq('A:fit_record_revision', fit.get('revision', 'v1'), revision))
    items.append(V.cmp_eq('A:fit_record_estimator', fit.get('constraint_estimator', 'realized'), est))
    bud_rec = json.loads((RES / 'dr' / 'BUDGETS.json').read_text())
    bud = {k: float(x) for k, x in bud_rec['budgets'][str(v)].items()}
    # ---- A: sources and budgets
    ledger = REPO / consts['LEDGER_REL']
    items.append(V.cmp_eq('A:pin_sha256:ledger', V.sha256_file(ledger), pins[consts['LEDGER_REL']]))
    sl, bud2 = budgets_from_ledger(ledger, v)
    for k in V.METRICS8:
        items.append(V.cmp_num(f'A:budget_{k}_recomputed_from_ledger', bud2[k], bud[k], 1e-15))
        items.append(V.cmp_num(f'A:fit_record_budget_{k}', fit.get('budgets', {}).get(k), bud[k], 0.0))
    task = pinned_pickle(f'nrm_fa_cache/V{v}/TASK.pkl', pins, items)
    ss = pinned_pickle(f'frc_fa_cache/V{v}/SEXSIDE.pkl', pins, items)
    tw = pinned_pickle(f'frc_runs/FA/TWB_C64/V{v}/MODEL.pkl', pins, items)
    cx = Ctx()
    cx.lab = {k: np.asarray(data[k], dtype=np.int64) for k in ('income', 'occupation', 'sex')}
    row_sha = C.arr_sha(np.asarray(data['row_id']))
    items.append(V.cmp_eq('A:task_cache_row_set', task['row_id_sha256'], row_sha))
    items.append(V.cmp_eq('A:sexside_cache_row_set', ss['row_id_sha256'], row_sha))
    items.append(V.cmp_eq('A:twin_identity', [tw['config_id'], int(tw['version']), tw['construction']], ['TWB_C64', v, 'B1']))
    # ---- B: out-of-fold routing and fine cells
    cx.qI = np.asarray(task['oof']['inc'], dtype=np.float64)
    cx.qO = np.asarray(task['oof']['occ'], dtype=np.float64)
    cx.pi = np.clip(np.asarray(ss['oof']['sex'], dtype=np.float64)[:, 1], *V.PI_CLIP)
    qO0 = np.asarray(ss['oof']['occ_s'][0], dtype=np.float64)
    qO1 = np.asarray(ss['oof']['occ_s'][1], dtype=np.float64)
    cx.Lg = np.where(cx.lab['sex'][:, None] == 0, qO0, qO1)        # R1 guard costs: the person's own SEX group's teacher
    cx.CI, cx.CO = np.asarray(tw['book']['CI']), np.asarray(tw['book']['CO'])
    k, gk = V.nearest_codeword(cx.qI, cx.CI)
    l, gl = V.nearest_codeword(cx.qO, cx.CO)
    info['routing_min_gap'] = {'income': float(gk.min()), 'occupation': float(gl.min()),
                               'n_gap_below_1e-12': int((gk < 1e-12).sum() + (gl < 1e-12).sum())}
    TI0, nI0 = V.b1_tables(k, cx.qI, V.CAPACITY[0], cx.CI)
    TO0, nO0 = V.b1_tables(l, cx.qO, V.CAPACITY[1], cx.CO)
    used_i, used_o = nI0 > 0, nO0 > 0
    d_i = float(np.abs(TI0[used_i] - np.asarray(tw['dec']['I'])[used_i]).max())
    d_o = float(np.abs(TO0[used_o] - np.asarray(tw['dec']['O'])[used_o]).max())
    items.append(V.cmp_num('B:identity_B1_tables_vs_twin_dec_income_max_abs', d_i, 0.0, 1e-12))
    items.append(V.cmp_num('B:identity_B1_tables_vs_twin_dec_occupation_max_abs', d_o, 0.0, 1e-12))
    info['identity_empty_atoms'] = {'income': int((~used_i).sum()), 'occupation': int((~used_o).sum())}
    my_cuts = [V.quantile_type7(cx.pi, q) for q in (0.25, 0.5, 0.75)]
    cuts = np.asarray(fit['pi_cuts'], dtype=np.float64)
    for i in range(3):
        items.append(V.cmp_num(f'B:pi_cut_{i}_recomputed', my_cuts[i], float(cuts[i]), 1e-15))
    m = V.pi_bins(cx.pi, cuts)
    items.append(V.cmp_eq('B:pi_bins_same_with_recomputed_cuts', int((V.pi_bins(cx.pi, my_cuts) != m).sum()), 0))
    cx.ic, cx.oc = V.fine_cells(k, l, m)
    info['fine_cells_occupied'] = {'income': int(len(np.unique(cx.ic))), 'occupation': int(len(np.unique(cx.oc)))}
    items.append(V.cmp_eq('B:occupied_fine_cells_vs_fit_record', [info['fine_cells_occupied']['income'],
                                                                  info['fine_cells_occupied']['occupation']],
                          [fit.get('problem', {}).get('occupied_income_fine_cells'),
                           fit.get('problem', {}).get('occupied_occupation_fine_cells')]))
    info['pi_bin_counts'] = np.bincount(m, minlength=4).tolist()
    cx.Vi0 = (TI0[k], np.asarray(tw['dec']['I'])[k])          # identity vectors: own B1 tables, the twin's stored tables
    cx.Vo0 = (TO0[l], np.asarray(tw['dec']['O'])[l])
    # ---- C: saved configurations
    per, models = {}, {}
    saved = list(fit.get('saved', []))
    info['saved_configs'] = saved
    for cid in saved:
        M, meta = load_model(cid, v, items)
        models[cid] = M
        items.append(V.cmp_eq(f'C:{cid}:book_equals_twin_book', bool(np.array_equal(M['book']['CI'], cx.CI) and
                                                                     np.array_equal(M['book']['CO'], cx.CO)), True))
        items.append(V.cmp_eq(f'C:{cid}:pi_cuts_equal_fit_record', bool(np.array_equal(np.asarray(M['pi_cuts']), cuts)), True))
        A_inc, A_occ = np.asarray(M['A_inc'], dtype=np.int64), np.asarray(M['A_occ'], dtype=np.int64)
        items.append(V.cmp_eq(f'C:{cid}:mapping_shapes_and_range', [list(A_inc.shape), list(A_occ.shape), bool(A_inc.min() >= 0),
                                                                    bool(A_inc.max() < 16), bool(A_occ.min() >= 0),
                                                                    bool(A_occ.max() < 64)], [[64], [256], True, True, True, True]))
        mine = insample_values(cx, A_inc, A_occ, (np.asarray(M['T_inc']), np.asarray(M['T_occ'])), exact=True)
        items.append(V.cmp_num(f'C:{cid}:B1_tables_recomputed_vs_saved_max_abs', mine['tables_vs_saved_max_abs'], 0.0, 1e-12))
        rec_fit = fit.get('insample', {}).get(cid)
        if rec_fit is not None:
            compare_insample(f'C:{cid}:fit_record', mine, rec_fit, bud, est, items)
        relrec = json.loads((PRIV / 'runs' / 'DR' / cid / f'V{v}' / 'RELEASE.json').read_text())
        compare_insample(f'C:{cid}:release_record', mine, relrec.get('insample_model_fit'), bud, est, items)
        tr = fit.get('traces', {}).get(cid)
        if tr is not None:
            obj = 'J_auc' if cid.startswith('DR_AUC') else 'J_mi'
            items.append(V.cmp_num(f'C:{cid}:trace_best_J_vs_recomputed_{obj}', mine['estimated']['atom'][obj], tr.get('best_J')))
        prim = mine['spend'][f'{est}_own_tables']
        per[cid] = {'path_step_record': relrec.get('path_step'), 'release_record_sha256': relrec.get('release_sha256'),
                    'values': {law: {kind: {s: {'auc': mine[law][kind][s]['auc'], 'mi': mine[law][kind][s]['mi'],
                                                'mi_unclipped': mine[law][kind][s]['mi_unclipped']} for s, _ in VIEWS}
                                     for kind in ('atom', 'merged')} for law in ('estimated', 'empirical')},
                    'J': {law: {kind: {'J_auc': mine[law][kind]['J_auc'], 'J_mi': mine[law][kind]['J_mi']} for kind in ('atom', 'merged')}
                          for law in ('estimated', 'empirical')},
                    'spend_realized': mine['spend']['realized_own_tables'], 'spend_expected': mine['spend']['expected_own_tables'],
                    'counts': mine['counts'], 'binding': max(V.METRICS8, key=lambda q: prim[q] / bud[q]),
                    'max_spend_over_budget': max(prim[q] / bud[q] for q in V.METRICS8),
                    'realized_max_spend_over_budget': max(mine['spend']['realized_own_tables'][q] / bud[q] for q in V.METRICS8)}
    # ---- D: the SEX-blind coarsening path, recomputed (v1: realized NLL; R1: expected NLL under the out-of-fold teachers)
    if est == 'realized':
        Li, Lo = V.onehot(cx.lab['income'], 2), V.onehot(cx.lab['occupation'], 5)
    else:
        Li, Lo = cx.qI, cx.qO
    path = V.coarsening_path(cx.ic, cx.oc, cx.qI, cx.qO, Li, Lo, max_steps=70, max_spend=0.08)
    pf = fit.get('path_full', [])
    items.append(V.cmp_eq('D:path_length', len(path), len(pf)))
    path_gaps = []
    for (Ai, Ao, step, spend, inc_, gap), rec in zip(path, pf):
        items.append(V.cmp_eq(f'D:path_step_index_{step}', int(rec.get('step', -1)), step))
        items.append(V.cmp_num(f'D:path_step{step}_spend_total_nll', spend, rec.get('spend_total_nll_insample')))
        if gap is not None:
            path_gaps.append(gap)
        mine = insample_values(cx, Ai, Ao, None, exact=False)
        compare_insample(f'D:path_step{step}', mine, rec.get('insample'), bud, est, items)
    info['path_min_gap_between_best_and_runner_up_merge'] = float(min(path_gaps)) if path_gaps else None
    for cid in saved:
        if not cid.startswith('DR_CTRL'):
            continue
        step = per[cid]['path_step_record']
        M = models[cid]
        hit = next((p for p in path if p[2] == step), None)
        ok = hit is not None and V.same_partition(hit[0], M['A_inc']) and V.same_partition(hit[1], M['A_occ'])
        items.append(V.cmp_eq(f'D:{cid}:saved_mapping_is_my_path_step_{step}', ok, True))
        expect = fit.get('brackets', {}).get('lower_step' if cid.startswith('DR_CTRL_LO') else 'upper_step')
        items.append(V.cmp_eq(f'D:{cid}:recorded_step_is_bracket_step', step, expect))
    # ---- E: deployment (label-free features)
    F_ms = C.features('MODEL_SELECT')
    Z_ms = FA.compact_po(F_ms['X'], F_ms['feature_names'])
    Z_mf = FA.compact_po(data['X'], data['feature_names'])
    tw_teach = {'inc': PRC._proba(tw['teachers']['inc'], Z_ms), 'occ': PRC._proba(tw['teachers']['occ'], Z_ms)}
    sex_teach = PRC._proba(ss['models']['sex'], Z_ms)
    deploy = {}
    for cid in saved:
        M = models[cid]
        arr, rec = RL.load('DR', cid, v, roles=('MODEL_SELECT', 'MODEL_FIT'))
        items.append(V.cmp_eq(f'E:{cid}:feature_names', list(M['feature_names']), list(F_ms['feature_names'])))
        e = emit_from_features(M, Z_ms, PRC)
        items.append(V.cmp_eq(f'E:{cid}:task_teachers_equal_pinned_twin_teachers',
                              bool(np.array_equal(e['qI'], tw_teach['inc']) and np.array_equal(e['qO'], tw_teach['occ'])), True))
        items.append(V.cmp_eq(f'E:{cid}:sex_teacher_equals_pinned_propensity_teacher',
                              bool(np.array_equal(e['pi'], np.clip(sex_teach[:, 1], *V.PI_CLIP))), True))
        for t in ('income', 'occupation'):
            items.append(V.cmp_eq(f'E:{cid}:MODEL_SELECT_{t}_bitwise_equal_saved_wire',
                                  bool(np.array_equal(e[t], arr['MODEL_SELECT'][t])), True))
            T = np.asarray(M['T_inc'] if t == 'income' else M['T_occ'])
            atoms = e['ai'] if t == 'income' else e['ao']
            items.append(V.cmp_eq(f'E:{cid}:MODEL_SELECT_{t}_rows_are_saved_table_rows_of_their_atoms',
                                  bool(np.array_equal(arr['MODEL_SELECT'][t], T[atoms])), True))
        dw = {'income': int(len(np.unique(V.row_ids(arr['MODEL_SELECT']['income'])))),
              'occupation': int(len(np.unique(V.row_ids(arr['MODEL_SELECT']['occupation'])))),
              'pair': int(len(np.unique(V.row_ids(np.hstack([arr['MODEL_SELECT']['income'], arr['MODEL_SELECT']['occupation']])))))}
        items.append(V.cmp_eq(f'E:{cid}:distinct_wires_model_select', dw, rec.get('distinct_wires_model_select')))
        f = emit_from_features(M, Z_mf, PRC)
        for t in ('income', 'occupation'):
            items.append(V.cmp_eq(f'E:{cid}:MODEL_FIT_{t}_bitwise_equal_saved_wire', bool(np.array_equal(f[t], arr['MODEL_FIT'][t])), True))
        dwf = {'income': int(len(np.unique(V.row_ids(arr['MODEL_FIT']['income'])))),
               'occupation': int(len(np.unique(V.row_ids(arr['MODEL_FIT']['occupation'])))),
               'pair': int(len(np.unique(V.row_ids(np.hstack([arr['MODEL_FIT']['income'], arr['MODEL_FIT']['occupation']])))))}
        items.append(V.cmp_eq(f'E:{cid}:distinct_wires_model_fit', dwf, rec.get('distinct_wires_model_fit')))
        deploy[cid] = {'model_fit_people_whose_deployed_fine_cells_differ_from_oof': {
            'income': int((f['ic'] != cx.ic).sum()), 'occupation': int((f['oc'] != cx.oc).sum())},
            'model_select_distinct_wires': dw, 'routing_min_gap_model_select': [e['min_gap_inc'], e['min_gap_occ']],
            'release_sha256': rec.get('release_sha256')}
    tarr, _ = FRL.load('FA', 'TWB_C64', v, roles=('MODEL_SELECT',))
    kt, _ = V.nearest_codeword(tw_teach['inc'], cx.CI)
    lt, _ = V.nearest_codeword(tw_teach['occ'], cx.CO)
    items.append(V.cmp_eq('E:twin_identity_reproduces_FRC1_TWB_C64_MODEL_SELECT_wires',
                          bool(np.array_equal(np.asarray(tw['dec']['I'])[kt], tarr['MODEL_SELECT']['income']) and
                               np.array_equal(np.asarray(tw['dec']['O'])[lt], tarr['MODEL_SELECT']['occupation'])), True))
    # ---- G: a revision whose seeds are a superset of an earlier revision's (R2 of R1): the shared seeds' per-seed objectives are
    # reproduced exactly (deterministic optimizer) and the best objective is no worse (card DR_DRF_R2)
    if rv.get('superset_of'):
        base = REVISION[rv['superset_of']]
        bfit = json.loads((RES / 'dr' / base['fit'].format(v=v)).read_text())
        for obj_c in ('DR_AUC', 'DR_MI'):
            tr_new = fit.get('traces', {}).get(obj_c + sfx, {})
            tr_old = bfit.get('traces', {}).get(obj_c + base['suffix'], {})
            old = {int(s['seed']): float(s['J']) for s in tr_old.get('per_seed_best_J', [])}
            new = {int(s['seed']): float(s['J']) for s in tr_new.get('per_seed_best_J', [])}
            items.append(V.cmp_eq(f'G:{obj_c}{sfx}:seeds_superset_of_{rv["superset_of"]}', bool(set(old) <= set(new)), True))
            for s_, j_ in sorted(old.items()):
                items.append(V.cmp_num(f'G:{obj_c}{sfx}:seed{s_}_best_J_equals_{rv["superset_of"]}', new.get(s_), j_, 0.0))
            items.append(V.cmp_eq(f'G:{obj_c}{sfx}:best_J_not_above_{rv["superset_of"]}',
                                  bool(tr_new.get('best_J') is not None and tr_new['best_J'] <= tr_old.get('best_J', float('inf')) + 1e-12),
                                  True))
            info.setdefault('superset_comparison', {})[obj_c] = {'best_J_new': tr_new.get('best_J'), 'best_J_old': tr_old.get('best_J')}
    # ---- F: CTRL bracket selection from the nested records
    brk = bracket_selection(fit, sfx, items)
    groups = {}
    for it in items:
        groups.setdefault(it['name'].split(':', 1)[0], []).append(it)
    out = {'schema': 'DRF1-verifier-check1-v2', 'info': info, 'per_config': per, 'deployment': deploy, 'brackets': brk,
           'summary': {g: V.summarize(x) for g, x in groups.items()}, 'overall': V.summarize(items), 'items': items,
           'cpu_s': round(time.process_time() - t0, 1), 'utc_end': C.now()}
    OUT.mkdir(parents=True, exist_ok=True)
    V.dump(OUT / f'{unit or "LOCAL"}_check1_{revision}_V{v}.json', out)
    return {'overall': out['overall'], 'summary': out['summary'], 'brackets': brk, 'info': info, 'cpu_s': out['cpu_s']}


if __name__ == '__main__':
    sys.exit(main())
