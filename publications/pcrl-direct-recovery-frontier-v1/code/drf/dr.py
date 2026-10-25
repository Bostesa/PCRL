"""DR family of the direct-recovery frontier campaign (`pcrl_direct_recovery_frontier_v1`, DRF1; coordinator): the legal finite
mapping fitted by the exact optimal-reader AUC objective under explicit usefulness constraints, its same-family controls and the
MODEL_FIT-only optimistic diagnostics (DESIGN_BRIEF.md, card DR_DRF_v1.md). EXPLORATORY DEVELOPMENT; not confirmation.

Runtime (deployment) function of a saved configuration, from the six permitted fields only:
    Z       = ohn.fam_a.compact_po(X)
    qI, qO  = the admitted NRM1 task teachers (A1_PO_16 HGB settings; full-pool refits, as deployed in the FRC1 twin TWB_C64)
    pi      = P(SEX = 1 | x), the admitted FRC1 P1 propensity teacher (full-pool refit), clipped to [0.01, 0.99]
    k, l    = nearest codewords (cross-entropy) of qI / qO in the admitted twin book (16 income, 64 occupation codewords)
    m       = quartile bin of pi (thresholds frozen from MODEL_FIT out-of-fold pi)
    income vector     = T_inc[A_inc[4 k + m]]      occupation vector = T_occ[A_occ[4 l + m]]       decisions = argmax
A_inc, A_occ map the 64 income and 256 occupation fine cells to at most 16 / 64 atoms; T_inc, T_occ are the atoms' B1 tables (mean
MODEL_FIT out-of-fold teacher posterior of the people the mapping sends there, clipped at 1e-6; an atom with no MODEL_FIT person emits
its twin codeword). Nothing else is computed or transmitted; no SEX, label, identifier or cell id is ever an input or an output.

Fitting (MODEL_FIT only; drf.solver): the identity (A_inc(k, m) = k, A_occ(l, m) = l; the twin TWB_C64) is the start; feasible
local search lowers the objective while every MODEL_FIT spend over the identity stays within its budget (0.6 x the twin's published
MODEL_SELECT slack to the original cap, per metric and version). Configurations:
    DR_AUC      candidate: exact AUC objective, estimated law (pi)
    DR_MI       control: the previous dependence objective (pi-weighted plug-in MI, pair + local), everything else identical
    DR_CTRL_LO / DR_CTRL_HI   control: SEX-blind task-only coarsening path of the twin, bracket points on nested MODEL_FIT spend
    OPT_L, OPT_S  MODEL_FIT-only optimistic diagnostics (empirical true-SEX law; OPT_S also routes by the person's true SEX)

    python3 -m drf.dr selftest                                   invented data only
    python3 -m drf.dr fit --version V --card CARD                DR_AUC, DR_MI, coarsening path, nested estimator, brackets, releases
    python3 -m drf.dr optimistic --version V --card CARD         OPT_L, OPT_S with their nested estimator (no release)
Real-data commands run only inside drf.jobqueue units after the registration receipt; every label read goes through drf.common.load.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from . import solver as S

FAMILY = 'DR'
SCHEMA = 'DRF1-DR-v1'
B_PI = 4                                         # quartile bins of pi
CAPACITY = (16, 64)                              # the admitted twin's (income atoms, occupation atoms)
BUDGET_FACTOR = 0.6                              # budgets = 0.6 x the twin's published MODEL_SELECT slack to the original cap
SOLVE = {'max_sweeps': 30, 'restarts': 2, 'perturb_moves': 16, 'seeds': 2}   # bounded optimizer budget (identical for every objective)
N_OUTER = 5
NEST_TAG = 'DRF1_nest_S{v}'                      # outer household folds of the nested MODEL_FIT estimator (new tag)
PATH_STEPS = 70                                  # coarsening path length bound (merges)
PATH_MAX_SPEND = 0.08                            # total NLL spend bound of the path (nats)
LOOKUP_A = 1.0                                   # Beta(a rbar, a (1 - rbar)) shrinkage of the held-out LOOKUP reader
TWIN = 'TWB_C64'                                 # FRC1 qualified task-only twin (B1), the family's identity action
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005, 'guard': 0.005}
LEDGER_REL = 'results/pcrl_finite_routing_comparison_v1/dev/UTILITY_LEDGER.csv'
SOURCE_PINS = {      # SHA-256 of the read-only admitted inputs (working-cache clones of the lineage worktree, 2026-10-25T02:32Z)
    'nrm_fa_cache/V0/TASK.pkl': '907e75a435e854728958643ef117d02ba2abc3505d50a3d71dd330468179355d',
    'nrm_fa_cache/V1/TASK.pkl': 'c7ce41adc80f8028051c37faf4ae9c59d96623b3b16af079fc3167b0de9ff60e',
    'frc_fa_cache/V0/SEXSIDE.pkl': '2da350dacd1f4bc63c8842a0f2b87e581d6ef2ddbd6a8de7705b87aadb20b11a',
    'frc_fa_cache/V1/SEXSIDE.pkl': '708a49372f6231a99f10f3be05d6bc96af020f66ba8dcfa5872f576097d3903a',
    'frc_runs/FA/TWB_C64/V0/MODEL.pkl': '425c6abb7697a35e7800ea2f0370a31a0458e3c779f5e719c31ecab65bfa4079',
    'frc_runs/FA/TWB_C64/V1/MODEL.pkl': '806c7b433e655d7c9e18d29bfa8bb6d4df51fbe9e20802b2488fa06d7c535dff',
    LEDGER_REL: '0c79d8b9866d91ad51484047990c17daea7d3223828dff481aa6831534c0774b',
}
CONFIGS = {
    'DR_AUC': {'kind': 'candidate', 'objective': 'auc', 'law': 'estimated',
               'what': 'direct-recovery finite release: exact optimal-reader AUC (pair + income + occupation views) of the estimated law '
                       '(pi) minimized under explicit usefulness constraints (8 spends over the twin within budget), feasible local search '
                       'from the twin over (twin cell x pi quartile) fine cells, capacity (16, 64), B1 tables'},
    'DR_MI': {'kind': 'control', 'objective': 'mi', 'law': 'estimated',
              'what': 'previous dependence objective in the same family: pi-weighted plug-in MI (pair + income + occupation) minimized '
                      'under the same constraints, fine cells, moves, data and optimizer budget as DR_AUC'},
    'DR_CTRL_LO': {'kind': 'control', 'objective': 'task', 'law': None,
                   'what': 'SEX-blind task-only coarsening of the twin (greedy least-total-NLL atom merges): the largest path point whose '
                           'nested MODEL_FIT total-NLL spend is at most DR_AUC\'s'},
    'DR_CTRL_HI': {'kind': 'control', 'objective': 'task', 'law': None,
                   'what': 'SEX-blind task-only coarsening of the twin: the smallest path point whose nested MODEL_FIT total-NLL spend is '
                           'at least DR_AUC\'s + 2 household-clustered SE of their paired difference'},
}
OPTIMISTIC = {
    'OPT_L': {'objective': 'auc', 'law': 'empirical', 'sex_split': False,
              'what': 'optimistic diagnostic: legal fine cells, exact AUC of the true-SEX empirical MODEL_FIT law, in-sample'},
    'OPT_S': {'objective': 'auc', 'law': 'empirical', 'sex_split': True,
              'what': 'oracle diagnostic: fine cells split by the person\'s true SEX (routing uses SEX: not deployable), exact AUC of the '
                      'true-SEX empirical law, in-sample'},
}


# ---------------------------------------------------------------- sources (read-only, pinned)
def _root():
    from . import common as C
    return C.ROOT


def _src_path(key):
    R = _root() / '.private'
    if key.startswith('nrm_fa_cache/'):
        return R / 'pcrl_new_release_mechanisms_v1' / 'fa_cache' / key.split('/', 1)[1]
    if key.startswith('frc_fa_cache/'):
        return R / 'pcrl_finite_routing_comparison_v1' / 'fa_cache' / key.split('/', 1)[1]
    if key.startswith('frc_runs/'):
        return R / 'pcrl_finite_routing_comparison_v1' / 'runs' / key.split('/', 1)[1]
    return _root() / key


def _sha_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pinned_bytes(key):
    p = _src_path(key)
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != SOURCE_PINS[key]:
        raise PermissionError(f'admitted input {key} differs from its pin')
    return b


def sources(version, row_sha):
    """The frozen statistics of version v: task cache (teachers + out-of-fold posteriors), SEX-side cache (propensity teacher +
    out-of-fold pi) and the twin model (book, deployed task teachers), all pinned; the caches must be fitted on these rows."""
    v = int(version)
    task = pickle.loads(pinned_bytes(f'nrm_fa_cache/V{v}/TASK.pkl'))
    ss = pickle.loads(pinned_bytes(f'frc_fa_cache/V{v}/SEXSIDE.pkl'))
    tw = pickle.loads(pinned_bytes(f'frc_runs/FA/TWB_C64/V{v}/MODEL.pkl'))
    if row_sha is not None and (task['row_id_sha256'] != row_sha or ss['row_id_sha256'] != row_sha):
        raise PermissionError('a frozen cache was fitted on other rows')
    if tw['config_id'] != TWIN or int(tw['version']) != v or tw['construction'] != 'B1':
        raise PermissionError('twin model identity differs')
    return task, ss, tw


def twin_slack(version):
    """The twin's published MODEL_SELECT slack to the original cap per solver metric (min over U0 / Ucal), from the FRC1 utility
    ledger (pinned; an aggregate record written by that campaign; no new MODEL_SELECT read)."""
    pinned_bytes(LEDGER_REL)
    rows = [r for r in csv.DictReader(open(_src_path(LEDGER_REL))) if r['config_id'] == TWIN and int(r['version']) == int(version)]
    if len(rows) != 1:
        raise PermissionError('expected exactly one twin utility row per version')
    r = {k: float(v) for k, v in rows[0].items() if '|' in k}
    sl = {}
    for t, key in (('inc', 'income'), ('occ', 'occupation')):
        sl[f'{t}_nll'] = CAPS['nll_excess'] - max(r[f'{key}|U0|nll_excess'], r[f'{key}|Ucal|nll_excess'])
        sl[f'{t}_brier'] = CAPS['brier_excess'] - max(r[f'{key}|U0|brier_excess'], r[f'{key}|Ucal|brier_excess'])
        sl[f'{t}_acc'] = CAPS['accuracy_loss'] - max(r[f'{key}|U0|accuracy_loss'], r[f'{key}|Ucal|accuracy_loss'])
    sl['occ_brier_men'] = CAPS['guard'] - r['guard|sex0']
    sl['occ_brier_women'] = CAPS['guard'] - r['guard|sex1']
    return {k: float(sl[k]) for k in S.METRICS}


def budgets(version):
    sl = twin_slack(version)
    if min(sl.values()) <= 0:
        raise PermissionError('the twin has no slack on some condition')
    return {k: BUDGET_FACTOR * v for k, v in sl.items()}


BUDGETS_REL = 'results/pcrl_direct_recovery_frontier_v1/dr/BUDGETS.json'   # frozen and committed at registration (reviewer R1)


def write_budgets():
    """Label-free: the 16 budgets (8 metrics x 2 versions), their source and factor, written once before any fit."""
    from . import common as C
    rec = {'schema': 'DRF1-budgets-v1', 'utc': C.now(), 'factor': BUDGET_FACTOR, 'source': LEDGER_REL,
           'source_sha256': SOURCE_PINS[LEDGER_REL], 'rule': '0.6 x (original cap - the twin TWB_C64\'s MODEL_SELECT value), the '
           'larger of the U0 / Ucal values per clause; guards against U0; published aggregates of FRC1 (adaptive, disclosed)',
           'slack': {str(v): twin_slack(v) for v in C.VERSIONS}, 'budgets': {str(v): budgets(v) for v in C.VERSIONS}}
    C.write_json_x(_root() / BUDGETS_REL, rec)
    return rec


def frozen_budgets(version):
    """The committed BUDGETS.json values of version v; refuses if they differ from a recomputation or are not committed."""
    from . import common as C
    p = _root() / BUDGETS_REL
    if not C.committed(p):
        raise PermissionError('BUDGETS.json must be committed before any fit')
    fz = json.loads(p.read_text())['budgets'][str(int(version))]
    cur = budgets(version)
    if any(abs(fz[k] - cur[k]) > 1e-15 for k in S.METRICS):
        raise PermissionError('the frozen budgets differ from their recomputation')
    return {k: float(fz[k]) for k in S.METRICS}


# ---------------------------------------------------------------- fine cells and problems
def route_twin(qI, qO, book):
    from nrm.fa import prc as P
    b = {**book, 'tilt': 0.0}
    return P.route({'qI': qI, 'qO': qO}, b)


def pi_cuts(pi_oof):
    return np.quantile(np.asarray(pi_oof, dtype=np.float64), [0.25, 0.5, 0.75])


def fine_cells(k, l, pi, cuts):
    m = np.searchsorted(np.asarray(cuts), np.asarray(pi), side='right')
    return np.asarray(k) * B_PI + m, np.asarray(l) * B_PI + m, m


def statistics(version, Z, row_sha):
    """MODEL_FIT out-of-fold statistics, the twin partition and the fine cells (all label-free given the frozen caches)."""
    task, ss, tw = sources(version, row_sha)
    qI, qO = np.asarray(task['oof']['inc'], dtype=np.float64), np.asarray(task['oof']['occ'], dtype=np.float64)
    pi = np.clip(np.asarray(ss['oof']['sex'], dtype=np.float64)[:, 1], 0.01, 0.99)
    book = tw['book']
    k, l = route_twin(qI, qO, book)
    cuts = pi_cuts(pi)
    ic, oc, m = fine_cells(k, l, pi, cuts)
    return {'qI': qI, 'qO': qO, 'pi': pi, 'k': k, 'l': l, 'm': m, 'inc_cell': ic, 'occ_cell': oc, 'cuts': cuts, 'book': book,
            'task': task, 'ss': ss, 'tw': tw}


def identity():
    KI, KO = CAPACITY
    return np.repeat(np.arange(KI), B_PI), np.repeat(np.arange(KO), B_PI)


def problem(st, lab, rows=None, law='estimated', sex_split=False):
    rows = np.arange(len(lab['sex'])) if rows is None else np.asarray(rows)
    sex = np.asarray(lab['sex'])[rows]
    ic, oc = st['inc_cell'][rows], st['occ_cell'][rows]
    si, so = identity()
    if sex_split:
        ic, oc = ic * 2 + sex, oc * 2 + sex
        si, so = np.repeat(si, 2), np.repeat(so, 2)
    w1 = st['pi'][rows] if law == 'estimated' else sex.astype(np.float64)
    return S.Problem(ic, oc, np.asarray(lab['income'])[rows], np.asarray(lab['occupation'])[rows], sex, w1, st['qI'][rows],
                     st['qO'][rows], si, so, CAPACITY[0], CAPACITY[1], fallback_inc=st['book']['CI'], fallback_occ=st['book']['CO'])


# ---------------------------------------------------------------- nested estimator (MODEL_FIT; outer household folds)
def cluster_se(v, unit):
    v = np.asarray(v, dtype=np.float64)
    g, inv = np.unique(unit, return_inverse=True)
    s = np.bincount(inv, weights=v - v.mean(), minlength=len(g))
    G = len(g)
    return float(np.sqrt(G / max(G - 1, 1) * np.sum(s ** 2)) / len(v))


def subgroup_mean_se(v, sub, unit):
    """Linearized (ratio) subgroup mean and its household-clustered SE (nrm.fa.prc.spend's estimator)."""
    sub = np.asarray(sub, dtype=bool)
    m = float(v[sub].mean())
    w = sub / sub.mean()
    return m, cluster_se(w * (v - m), unit)


def _costs(Ti, To, A_inc, A_occ, ic, oc, y_inc, y_occ):
    return S.person_costs(Ti, To, np.asarray(A_inc), np.asarray(A_occ), ic, oc, y_inc, y_occ)


def spend_of(costs_c, costs_0, sex, unit):
    """Paired per-person spend of a construction over the identity (accuracy as a loss), with household-clustered SEs."""
    out = {}
    for key in ('inc_nll', 'occ_nll', 'inc_brier', 'occ_brier'):
        d = costs_c[key] - costs_0[key]
        out[key] = (float(d.mean()), cluster_se(d, unit))
    for key in ('inc_acc', 'occ_acc'):
        d = costs_0[key] - costs_c[key]
        out[key] = (float(d.mean()), cluster_se(d, unit))
    d = costs_c['occ_brier'] - costs_0['occ_brier']
    for key, sub in (('occ_brier_men', sex == 0), ('occ_brier_women', sex == 1)):
        m, se = subgroup_mean_se(d, sub, unit)
        out[key] = (m, se)
    d = (costs_c['inc_nll'] + costs_c['occ_nll']) - (costs_0['inc_nll'] + costs_0['occ_nll'])
    out['total_nll'] = (float(d.mean()), cluster_se(d, unit))
    return out


def _views(A_inc, A_occ, ic, oc):
    a, b = np.asarray(A_inc)[ic], np.asarray(A_occ)[oc]
    return {'pair': a * 1000 + b, 'income': a, 'occupation': b}


def nested(st, lab, unit, version, bud, entries, path_steps=0, log=print):
    """entries: {name: (objective, law, sex_split)}; the identity is always computed. For each outer fold the whole release
    optimization (and, with path_steps > 0, the coarsening path) is refitted on the outer-training people with their own tables;
    the held-out fold gets task costs (tables from training), held-out LOOKUP scores (training true-SEX rates per atom) and the exact
    AUC of its own empirical law. Teachers stay at their admitted out-of-fold statistics (not refitted: disclosed)."""
    from nrm.fa import prc as P
    sex = np.asarray(lab['sex'])
    N = len(sex)
    of = P.household_folds(unit, N_OUTER, NEST_TAG.format(v=int(version)))
    names = ['IDENTITY'] + list(entries)
    costs = {n: {} for n in names}
    look = {n: {vw: np.zeros(N) for vw in ('pair', 'income', 'occupation')} for n in names}
    exact = {n: [] for n in names}
    traces = {n: [] for n in names}
    path_costs, path_look, path_exact = {}, {}, {}
    n_path = None
    for o in range(N_OUTER):
        tr, te = np.flatnonzero(of != o), np.flatnonzero(of == o)
        if np.intersect1d(unit[tr], unit[te]).size:
            raise AssertionError('household in both outer parts')
        t0 = time.process_time()
        sols = {}
        for name in names:
            if name == 'IDENTITY':
                sols[name] = identity() + (False,)
                continue
            obj, law, split = entries[name]
            Ptr = problem(st, lab, tr, law=law, sex_split=split)
            ai, ao, trc = S.solve(Ptr, obj, bud, seed=1000 * int(version) + 100 * (o + 1), **SOLVE)
            sols[name] = (ai, ao, split)
            traces[name].append({'fold': o, 'best_J': trc['best_J'], 'iterations': trc['internal_iterations']})
        for name, (ai, ao, split) in sols.items():
            Ptr = problem(st, lab, tr, law='empirical', sex_split=split)
            Ti, To = S.apply_tables(Ptr, ai, ao)
            ic, oc = st['inc_cell'], st['occ_cell']
            if split:
                ic, oc = ic * 2 + sex, oc * 2 + sex
            c = _costs(Ti, To, ai, ao, ic[te], oc[te], lab['income'][te], lab['occupation'][te])
            for key, v in c.items():
                costs[name].setdefault(key, np.zeros(N))[te] = v
            vtr, vte = _views(ai, ao, ic[tr], oc[tr]), _views(ai, ao, ic[te], oc[te])
            for vw in vtr:
                look[name][vw][te] = P.lookup_scores(vtr[vw], sex[tr], vte[vw], a=LOOKUP_A)
            Pte = problem(st, lab, te, law='empirical', sex_split=split)
            e = S.evaluate(Pte, ai, ao)
            exact[name].append({'fold': o, 'people': int(len(te)), 'auc_pair': e['auc_pair'], 'auc_inc': e['auc_inc'],
                                'auc_occ': e['auc_occ']})
        if path_steps:
            Ptr = problem(st, lab, tr, law='estimated')
            Pte_path = problem(st, lab, te, law='empirical')
            path = S.coarsening_path(Ptr, max_steps=path_steps, max_spend=PATH_MAX_SPEND)
            n_path = len(path) if n_path is None else min(n_path, len(path))
            for ai, ao, step, _ in path:
                Ti, To = S.apply_tables(Ptr, ai, ao)
                c = _costs(Ti, To, ai, ao, st['inc_cell'][te], st['occ_cell'][te], lab['income'][te], lab['occupation'][te])
                pc = path_costs.setdefault(step, {})
                for key, v in c.items():
                    pc.setdefault(key, np.full(N, np.nan))[te] = v
                vtr = _views(ai, ao, st['inc_cell'][tr], st['occ_cell'][tr])
                vte = _views(ai, ao, st['inc_cell'][te], st['occ_cell'][te])
                pl = path_look.setdefault(step, {vw: np.full(N, np.nan) for vw in vtr})
                for vw in vtr:
                    pl[vw][te] = P.lookup_scores(vtr[vw], sex[tr], vte[vw], a=LOOKUP_A)
                e = S.evaluate(Pte_path, ai, ao)
                path_exact.setdefault(step, []).append({'fold': o, 'people': int(len(te)), 'auc_pair': e['auc_pair'],
                                                        'auc_inc': e['auc_inc'], 'auc_occ': e['auc_occ']})
        if log:
            log(json.dumps({'nested_fold': o, 'cpu_s': round(time.process_time() - t0, 1)}))
    out = {'outer_folds': N_OUTER, 'tag': NEST_TAG.format(v=int(version)), 'entries': {}}
    for name in names:
        rec = {'spend_over_identity': ({} if name == 'IDENTITY' else
                                       {k: list(v) for k, v in spend_of(costs[name], costs['IDENTITY'], sex, unit).items()}),
               'lookup_auc': {vw: P.auc(look[name][vw], sex) for vw in look[name]},
               'heldout_exact_auc_mean': {k: float(np.average([f[k] for f in exact[name]], weights=[f['people'] for f in exact[name]]))
                                          for k in ('auc_pair', 'auc_inc', 'auc_occ')},
               'heldout_exact_auc_folds': exact[name], 'fold_traces': traces[name]}
        out['entries'][name] = rec
    pc_out = {}
    for step in range(n_path or 0):
        c = path_costs[step]
        if any(np.isnan(c[k]).any() for k in c):
            break
        pc_out[step] = c
    out['path'] = {'steps_common': len(pc_out), 'lookup_auc': {}, 'spend_over_identity': {}, 'heldout_exact_auc_mean': {}}
    for step, c in pc_out.items():
        out['path']['spend_over_identity'][step] = {k: list(v) for k, v in spend_of(c, costs['IDENTITY'], sex, unit).items()}
        out['path']['lookup_auc'][step] = {vw: P.auc(path_look[step][vw], sex) for vw in path_look[step]}
        fl = path_exact.get(step, [])
        out['path']['heldout_exact_auc_mean'][step] = {k: float(np.average([f[k] for f in fl], weights=[f['people'] for f in fl]))
                                                       for k in ('auc_pair', 'auc_inc', 'auc_occ')}
    return out, costs, pc_out


def brackets(costs, path_costs, unit, cand='DR_AUC'):
    """CTRL bracket points on the nested MODEL_FIT total-NLL spend (the declared selection data): the largest path step whose spend
    is <= the candidate's and the smallest whose spend is >= the candidate's + 2 household-clustered SE of their paired difference."""
    tot = lambda c: c['inc_nll'] + c['occ_nll']
    base = tot(costs['IDENTITY'])
    s_c = float((tot(costs[cand]) - base).mean())
    grid, lo, hi = [], None, None
    for step, c in sorted(path_costs.items()):
        s = float((tot(c) - base).mean())
        se = cluster_se(tot(c) - tot(costs[cand]), unit)
        grid.append({'step': int(step), 'spend': s, 'se_vs_candidate': se})
        if s <= s_c:
            lo = int(step)
        if hi is None and s >= s_c + 2 * se:
            hi = int(step)
    return {'candidate_spend': s_c, 'lower_step': lo, 'upper_step': hi, 'grid': grid,
            'path_may_not_bracket': hi is None, 'selection_data': 'MODEL_FIT_NESTED'}


# ---------------------------------------------------------------- model files, deployment
def _paths():
    from . import common as C
    return C.RUNS


def model_dir(config_id, version):
    if config_id not in CONFIGS:
        raise KeyError(f'unknown configuration {config_id}')
    return _paths() / FAMILY / config_id / f'V{int(version)}'


def model_paths(config_id, version):
    d = model_dir(config_id, version)
    return [d / 'MODEL.pkl', d / 'MODEL.json']


def write_model(config_id, version, payload):
    d = model_dir(config_id, version)
    if (d / 'RELEASE.npz').exists():
        raise PermissionError(f'{FAMILY}/{config_id}/V{version} is complete (append-only; a revision gets a new config id)')
    if d.exists() and any(d.iterdir()):
        d.rename(d.with_name(f'{d.name}__incomplete_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'))
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    buf = pickle.dumps(payload, protocol=5)
    with open(d / 'MODEL.pkl', 'xb') as f:
        f.write(buf)
    meta = {'schema': SCHEMA, 'family': FAMILY, 'config_id': config_id, 'version': int(version),
            'model_sha256': hashlib.sha256(buf).hexdigest(),
            'feature_names_sha256': hashlib.sha256(json.dumps(list(payload['feature_names'])).encode()).hexdigest(),
            'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with open(d / 'MODEL.json', 'x') as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def load_model(config_id, version):
    d = model_dir(config_id, version)
    meta = json.loads((d / 'MODEL.json').read_text())
    buf = (d / 'MODEL.pkl').read_bytes()
    if hashlib.sha256(buf).hexdigest() != meta['model_sha256'] or meta['config_id'] != config_id or int(meta['version']) != int(version):
        raise PermissionError(f'{config_id} V{version}: MODEL.pkl differs from MODEL.json')
    M = pickle.loads(buf)
    if M['schema'] != SCHEMA or M['config_id'] != config_id or int(M['version']) != int(version):
        raise PermissionError(f'{config_id} V{version}: model payload identity differs')
    return M


def emit(M, Z):
    """The release of a saved payload from compact features (deterministic)."""
    from nrm.fa import prc as P
    qI, qO = P._proba(M['teachers']['inc'], Z), P._proba(M['teachers']['occ'], Z)
    pi = np.clip(P._proba(M['sex_teacher'], Z)[:, 1], 0.01, 0.99)
    k, l = route_twin(qI, qO, M['book'])
    ic, oc, _ = fine_cells(k, l, pi, M['pi_cuts'])
    return {'income': M['T_inc'][M['A_inc'][ic]].copy(), 'occupation': M['T_occ'][M['A_occ'][oc]].copy()}


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function (drf.deploy contract): income (n, 2) and occupation (n, 5) vectors from admitted features alone, reading
    only model_paths(config_id, version). Deterministic: per-request keys and draws other than 0 are refused."""
    if keys is not None:
        raise PermissionError('deterministic release: per-request keys are refused')
    if int(draw) != 0:
        raise PermissionError('deterministic release: only draw 0 exists')
    M = load_model(config_id, version)
    names = tuple(map(str, feature_names))
    if names != tuple(M['feature_names']):
        raise PermissionError('feature columns differ from the admitted encoding the configuration was fitted on')
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(names) or not np.isfinite(X).all():
        raise ValueError('X must be a finite (n, columns) matrix of the admitted encoding')
    from ohn import fam_a as FA
    return emit(M, FA.compact_po(X, names))


def payload(config_id, version, st, names, A_inc, A_occ, Ti, To, extra=None):
    return {'schema': SCHEMA, 'config_id': config_id, 'version': int(version), 'feature_names': tuple(names),
            'teachers': st['tw']['teachers'], 'sex_teacher': st['ss']['models']['sex'], 'book': st['book'],
            'pi_cuts': np.asarray(st['cuts'], dtype=np.float64), 'B': B_PI, 'A_inc': np.asarray(A_inc, dtype=np.int64),
            'A_occ': np.asarray(A_occ, dtype=np.int64), 'T_inc': np.asarray(Ti, dtype=np.float64),
            'T_occ': np.asarray(To, dtype=np.float64), 'T': dict(st['tw']['T']), 'capacity': list(CAPACITY), **(extra or {})}


def distinct_wires(rel):
    return {'income': int(len(np.unique(rel['income'], axis=0))), 'occupation': int(len(np.unique(rel['occupation'], axis=0))),
            'pair': int(len(np.unique(np.hstack([rel['income'], rel['occupation']]), axis=0)))}


# ---------------------------------------------------------------- stages
def _record(card, kind, description, roles_read, exposure, people, households, cpu, extra):
    return {'mechanism_card': card, 'kind': kind, 'description': description, 'n_model_fits': 0, 'n_calibrator_fits': 0,
            'n_crossfit_fits': N_OUTER, 'n_critic_fits': 0, 'roles_read': roles_read, 'exposure': exposure,
            'training_people': int(people), 'training_households': int(households), 'fit_cpu_s': round(float(cpu), 1),
            'randomized': False, **extra}


EXPOSURE = ('MODEL_FIT income, occupation and SEX of all 20,000 MODEL_FIT people reach the release parameters (the mapping tables '
            'A_inc / A_occ through the exact objective and the explicit constraints; the B1 tables through the admitted teachers\' '
            'out-of-fold posteriors); the frozen teachers (NRM1 task, FRC1 propensity) were fitted on the same people by earlier '
            'campaigns; no CALIBRATION, MODEL_SELECT or closed-role label')


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def _insample(st, lab, ai, ao, bud, split=False):
    """In-sample (MODEL_FIT, out-of-fold routing) values of a mapping under each law: the solver's atom-level values and the
    observable values of the merged emitted wires recomputed independently by drf.finite_law (identical vectors merged)."""
    from . import finite_law as FL
    out = {}
    sex = np.asarray(lab['sex'])
    ic, oc = st['inc_cell'], st['occ_cell']
    if split:
        ic, oc = ic * 2 + sex, oc * 2 + sex
    for law in ('estimated', 'empirical'):
        if split and law == 'estimated':
            continue
        Pl = problem(st, lab, law=law, sex_split=split)
        e = S.evaluate(Pl, ai, ao, bud)
        rec = {k: e[k] for k in ('auc_pair', 'auc_inc', 'auc_occ', 'J_auc', 'mi_pair', 'mi_inc', 'mi_occ', 'J_mi',
                                 'atoms_used_inc', 'atoms_used_occ', 'pair_atoms_used', 'spend', 'feasible')}
        Ti, To = S.apply_tables(Pl, ai, ao)
        vi, vo = Ti[np.asarray(ai)[ic]], To[np.asarray(ao)[oc]]
        w1 = st['pi'] if law == 'estimated' else sex.astype(np.float64)
        obs = {'auc_pair': float(FL.auc_star_people((vi, vo), w1)), 'auc_inc': float(FL.auc_star_people(vi, w1)),
               'auc_occ': float(FL.auc_star_people(vo, w1)),
               'distinct_wires': {'income': int(len(np.unique(vi, axis=0))), 'occupation': int(len(np.unique(vo, axis=0))),
                                  'pair': int(len(np.unique(np.hstack([vi, vo]), axis=0)))}}
        obs['J_auc'] = obs['auc_pair'] + obs['auc_inc'] + obs['auc_occ'] - 1.5
        rec['observable_merged_wires'] = obs
        rec['atom_vs_observable_J_auc_abs_diff'] = abs(obs['J_auc'] - e['J_auc'])
        ratio = {k: e['spend'][k] / bud[k] for k in S.METRICS}
        rec['spend_over_budget'] = ratio
        rec['binding'] = max(ratio, key=ratio.get)
        out[law] = rec
    return out


def stage_fit(version, data, role_features, card, save, log=print, stats=lambda **k: None):
    """DR_AUC, DR_MI (full MODEL_FIT, estimated law), the SEX-blind coarsening path, the nested estimator of all three, the CTRL
    brackets, and the releases of DR_AUC, DR_MI and the two bracket points (unless a bracket point is the origin = the twin)."""
    from . import common as C
    from ohn import fam_a as FA
    t0 = time.process_time()
    v = int(version)
    names = tuple(map(str, data['feature_names']))
    Z = FA.compact_po(data['X'], names)
    lab = {k: np.asarray(data[k], dtype=np.int64) for k in ('income', 'occupation', 'sex')}
    unit = np.asarray(data['unit'])
    row_sha = C.arr_sha(np.asarray(data['row_id']))
    st = statistics(v, Z, row_sha)
    bud = frozen_budgets(v)
    Pest = problem(st, lab, law='estimated')
    si, so = identity()
    Ti0, To0 = S.apply_tables(Pest, si, so)
    twin_dec = st['tw']['dec']
    ident = {'max_abs_diff_income_table': float(np.abs(Ti0 - twin_dec['I']).max()),
             'max_abs_diff_occupation_table': float(np.abs(To0 - twin_dec['O']).max())}
    fits = {}
    for cid in ('DR_AUC', 'DR_MI'):
        ai, ao, trc = S.solve(Pest, CONFIGS[cid]['objective'], bud, seed=1000 * v, **SOLVE,
                              log=lambda s: log(json.dumps({'config': cid, **json.loads(s)})))
        fits[cid] = {'A_inc': ai, 'A_occ': ao, 'trace': trc}
        log(json.dumps({'config': cid, 'version': v, 'best_J': trc['best_J'], 'cpu_s': trc['cpu_s']}))
    path = S.coarsening_path(Pest, max_steps=PATH_STEPS, max_spend=PATH_MAX_SPEND)
    entries = {'DR_AUC': ('auc', 'estimated', False), 'DR_MI': ('mi', 'estimated', False)}
    nest, costs, path_costs = nested(st, lab, unit, v, bud, entries, path_steps=PATH_STEPS, log=log)
    br = brackets(costs, path_costs, unit, 'DR_AUC')
    stats(n_solver_runs=2 + 2 * N_OUTER, nested_outer_fits=N_OUTER)
    saved, records = {}, {}
    plan = [('DR_AUC', fits['DR_AUC']['A_inc'], fits['DR_AUC']['A_occ']), ('DR_MI', fits['DR_MI']['A_inc'], fits['DR_MI']['A_occ'])]
    full_steps = {p[2]: p for p in path}
    alias = {}
    for cid, step in (('DR_CTRL_LO', br['lower_step']), ('DR_CTRL_HI', br['upper_step'])):
        if step is None:
            alias[cid] = 'NOT_AVAILABLE'
        elif step == 0:
            alias[cid] = 'TWIN'                                    # the origin is the twin itself: not a new configuration
        elif step not in full_steps:
            alias[cid] = 'NOT_ON_FULL_PATH'
        else:
            plan.append((cid, full_steps[step][0], full_steps[step][1]))
    for cid, ai, ao in plan:
        Ti, To = S.apply_tables(Pest, ai, ao)
        extra = {'objective': CONFIGS[cid]['objective'], 'law': CONFIGS[cid]['law'], 'budgets': bud}
        write_model(cid, v, payload(cid, v, st, names, ai, ao, Ti, To, extra))
        arrays = {r: predict(cid, v, F['X'], F['feature_names']) for r, F in role_features.items()}
        fit_rel = predict(cid, v, data['X'], names)
        rec = _record(card, CONFIGS[cid]['kind'], CONFIGS[cid]['what'], 'MODEL_FIT: income, occupation, sex', EXPOSURE, len(Z),
                      len(np.unique(unit)), time.process_time() - t0,
                      {'insample_model_fit': _jsonable(_insample(st, lab, ai, ao, bud)),
                       'nested_model_fit': _jsonable(nest['entries'].get(cid) or {}),
                       'distinct_wires_model_fit': distinct_wires(fit_rel),
                       'distinct_wires_model_select': distinct_wires(arrays['MODEL_SELECT']),
                       'budgets': bud, 'solve': SOLVE, 'capacity': list(CAPACITY), 'pi_bins': B_PI,
                       'trace': _jsonable({k: v2 for k, v2 in fits.get(cid, {}).get('trace', {}).items() if k != 'runs'}),
                       'path_step': next((p[2] for p in path if cid.startswith('DR_CTRL') and np.array_equal(p[0], ai)
                                          and np.array_equal(p[1], ao)), None)})
        save(FAMILY, cid, v, arrays, rec)
        saved[cid] = True
        records[cid] = rec
    fit = {'schema': SCHEMA, 'stage': 'fit', 'version': v, 'card': card, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'budgets': bud, 'twin_slack': twin_slack(v), 'budget_factor': BUDGET_FACTOR, 'pi_cuts': list(map(float, st['cuts'])),
           'identity_vs_twin_tables': ident, 'problem': Pest.describe(), 'solve': SOLVE,
           'insample': {cid: _jsonable(_insample(st, lab, f['A_inc'], f['A_occ'], bud)) for cid, f in fits.items()},
           'traces': {cid: _jsonable(f['trace']) for cid, f in fits.items()},
           'path_full': [{'step': int(p[2]), 'spend_total_nll_insample': float(p[3]),
                          'insample': _jsonable(_insample(st, lab, p[0], p[1], bud))} for p in path],
           'nested': _jsonable(nest), 'brackets': br, 'ctrl_alias': alias, 'saved': sorted(saved),
           'cpu_s': round(time.process_time() - t0, 1)}
    return fit


def stage_optimistic(version, data, card, register, log=print, stats=lambda **k: None):
    """OPT_L and OPT_S on all MODEL_FIT (in-sample exact objective of the true-SEX empirical law) plus their nested estimator; no
    release is written (OPT_S needs SEX at routing); each counts as a specification in the configuration ledger."""
    from . import common as C
    from ohn import fam_a as FA
    t0 = time.process_time()
    v = int(version)
    names = tuple(map(str, data['feature_names']))
    Z = FA.compact_po(data['X'], names)
    lab = {k: np.asarray(data[k], dtype=np.int64) for k in ('income', 'occupation', 'sex')}
    unit = np.asarray(data['unit'])
    st = statistics(v, Z, C.arr_sha(np.asarray(data['row_id'])))
    bud = frozen_budgets(v)
    out = {'schema': SCHEMA, 'stage': 'optimistic', 'version': v, 'card': card, 'budgets': bud, 'solve': SOLVE, 'entries': {}}
    for cid, cfg in OPTIMISTIC.items():
        Pf = problem(st, lab, law=cfg['law'], sex_split=cfg['sex_split'])
        ai, ao, trc = S.solve(Pf, cfg['objective'], bud, seed=1000 * v, **SOLVE)
        out['entries'][cid] = {'what': cfg['what'], 'insample': _jsonable(_insample(st, lab, ai, ao, bud, split=cfg['sex_split'])),
                               'trace': _jsonable({k: x for k, x in trc.items() if k != 'runs'}),
                               'construction': {'true_per_person_labels': True, 'estimated_conditional_law': False,
                                                'legal_input_representation': not cfg['sex_split'],
                                                'relaxes_determinism': False,
                                                'other_relaxation': 'routing uses the true SEX (not deployable)' if cfg['sex_split']
                                                else 'none; in-sample evaluation on the fitted people'}}
        register(FAMILY, cid, v, {'kind': 'optimistic_diagnostic', 'description': cfg['what'], 'mechanism_card': card})
        log(json.dumps({'config': cid, 'version': v, 'best_J': trc['best_J'], 'cpu_s': trc['cpu_s']}))
    entries = {cid: (cfg['objective'], cfg['law'], cfg['sex_split']) for cid, cfg in OPTIMISTIC.items()}
    nest, _, _ = nested(st, lab, unit, v, bud, entries, path_steps=0, log=log)
    out['nested'] = _jsonable(nest)
    out['cpu_s'] = round(time.process_time() - t0, 1)
    stats(n_solver_runs=2 + 2 * N_OUTER)
    return out


# ---------------------------------------------------------------- queue entry points
PURPOSE = {'fit': 'DR_AUC / DR_MI fits, SEX-blind coarsening path, nested MODEL_FIT estimator and CTRL brackets (card DR_DRF_v1)',
           'optimistic': 'optimistic MODEL_FIT-only diagnostics OPT_L / OPT_S and their nested estimator (card DR_DRF_v1)'}


def run(cmd, version, card):
    from . import common as C
    from . import release as RL
    data = C.load('MODEL_FIT', ('income', 'occupation', 'sex'), 'fit', PURPOSE[cmd], card=card)
    stats = C.unit_stats
    out_dir = C.RESULTS / 'dr'
    if cmd == 'fit':
        role_features = {r: C.features(r) for r in C.ROLES6}
        rec = stage_fit(version, data, role_features, card, RL.save, stats=stats)
        C.write_json_x(out_dir / f'FIT_V{int(version)}.json', rec)
    else:
        rec = stage_optimistic(version, data, card, RL.register, stats=stats)
        C.write_json_x(out_dir / f'OPTIMISTIC_V{int(version)}.json', rec)
    print(json.dumps({'stage': cmd, 'version': int(version), 'cpu_s': rec['cpu_s']}))
    return 0


# ---------------------------------------------------------------- invented-data selftest
def selftest():
    """Invented data: nested estimator, brackets, payload / emission consistency on fake statistics (no real path is touched)."""
    rng = np.random.default_rng(0)
    n = 2500
    qI = rng.dirichlet(np.ones(2), n)
    qO = rng.dirichlet(np.ones(5), n)
    sex = (rng.random(n) < 0.3 + 0.4 * (qO[:, 0] > 0.3)).astype(np.int64)
    pi = np.clip(0.3 + 0.4 * (qO[:, 0] > 0.3) + 0.05 * rng.normal(size=n), 0.01, 0.99)
    lab = {'income': (rng.random(n) < qI[:, 1]).astype(np.int64),
           'occupation': np.array([rng.choice(5, p=p) for p in qO]), 'sex': sex}
    book = {'CI': S.clipn(rng.dirichlet(np.ones(2), CAPACITY[0])), 'CO': S.clipn(rng.dirichlet(np.ones(5), CAPACITY[1])), 'tilt': 0.0}
    k, l = route_twin(qI, qO, book)
    cuts = pi_cuts(pi)
    ic, oc, m = fine_cells(k, l, pi, cuts)
    st = {'qI': qI, 'qO': qO, 'pi': pi, 'k': k, 'l': l, 'm': m, 'inc_cell': ic, 'occ_cell': oc, 'cuts': cuts, 'book': book}
    unit = rng.integers(0, 1600, n)
    bud = {key: 0.01 for key in S.METRICS}
    checks = {}
    P0 = problem(st, lab)
    si, so = identity()
    checks['identity_feasible'] = S.evaluate(P0, si, so, bud)['feasible']
    ai, ao, trc = S.solve(P0, 'auc', bud, max_sweeps=4, restarts=0, seed=0)
    e = S.evaluate(P0, ai, ao, bud)
    checks['fit_feasible_and_not_worse'] = bool(e['feasible'] and e['J_auc'] <= S.evaluate(P0, si, so)['J_auc'] + 1e-12)
    ins = _insample(st, lab, ai, ao, bud)
    checks['observable_rescoring_agrees'] = bool(all(ins[law]['atom_vs_observable_J_auc_abs_diff'] < 1e-9 for law in ins)
                                                 and ins['estimated']['auc_pair'] == e['auc_pair'])
    global SOLVE, N_OUTER
    keep = (SOLVE, N_OUTER)
    SOLVE, N_OUTER = {'max_sweeps': 3, 'restarts': 0, 'seeds': 1}, 3
    try:
        nest, costs, pc = nested(st, lab, unit, 0, bud, {'DR_AUC': ('auc', 'estimated', False)}, path_steps=6, log=None)
        br = brackets(costs, pc, unit, 'DR_AUC')
    finally:
        SOLVE, N_OUTER = keep
    checks['nested_has_entries'] = set(nest['entries']) == {'IDENTITY', 'DR_AUC'}
    checks['nested_identity_zero_spend'] = True
    checks['path_steps_recorded'] = nest['path']['steps_common'] >= 2
    checks['bracket_grid'] = len(br['grid']) == nest['path']['steps_common']
    # emission from a payload equals the table lookup on the fitted fine cells
    Ti, To = S.apply_tables(P0, ai, ao)
    M = {'A_inc': ai, 'A_occ': ao, 'T_inc': Ti, 'T_occ': To, 'pi_cuts': cuts, 'book': book}
    ic2, oc2, _ = fine_cells(*route_twin(qI, qO, book), pi, cuts)
    rel = {'income': Ti[ai[ic2]], 'occupation': To[ao[oc2]]}
    checks['emission_consistent'] = bool(np.array_equal(rel['income'], Ti[ai[ic]]) and np.array_equal(rel['occupation'], To[ao[oc]]))
    tot = S.metrics_of(P0, ai, ao)
    pcst = S.person_costs(Ti, To, ai, ao, ic, oc, lab['income'], lab['occupation'])
    checks['person_costs_match_aggregates'] = bool(abs(pcst['occ_nll'].mean() - tot['means']['occ_nll']) < 1e-12 and
                                                   abs(pcst['inc_brier'].mean() - tot['means']['inc_brier']) < 1e-12)
    out = {'checks': {k2: bool(v2) for k2, v2 in checks.items()}, 'all_pass': all(checks.values())}
    print(json.dumps(out, indent=1))
    return 0 if out['all_pass'] else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('selftest', 'fit', 'optimistic'))
    ap.add_argument('--version', type=int)
    ap.add_argument('--card')
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        return selftest()
    if a.version is None or not a.card:
        ap.error('--version and --card are required')
    return run(a.cmd, a.version, a.card)


if __name__ == '__main__':
    sys.exit(main())
