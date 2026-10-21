"""V3: independent exact references on the registered laws (campaign `pcrl_finite_solver_adequacy_v1`).

From laws.json alone (committed before any experiment), with the verifier's own code (no fsa / frc / nrm function, no import of
the investigator's code), this recomputes and compares with the investigator's outputs (NARROW_V<n>.json, ENUM_V<n>.json and the
compact RESULTS_V<n>.json in roles/exact_reference/):
- posteriors qI, qO, qO0, qO1, pi, P(x) and P(S = 1) (exact rationals);
- population task metrics of a release (income / occupation NLL, accuracy, summed Brier, per-SEX occupation summed Brier) and the
  excess over the full-posterior release, with feasibility against the registered limits;
- the exact Bayes AUC of the pair / income / occupation views (identical released vectors merged; the reader ranks values by
  P(S = 1 | value), exact rational comparisons, ties count 1/2) and the pi-weighted plug-in information on cells and on values;
- the decision-only release, the witness / comparator, the twins and the protected / HQ path states at their reported partitions
  (B1 tables = exact cell means);
- the exact objective (task cross-entropy to the codewords + lam (w_pair I(S; pair) + w_loc (I(S; income) + I(S; occupation)))
  + lam kappa * mean per-SEX occupation guard) at the enumeration optima (codewords = cell means) and along every narrow trace
  (codewords of the refreshed book = cell means of the current partition, chained from the twin), against the reported values.

    python3 .../v3_exact.py --selftest
    python3 .../v3_exact.py --tag V1 --narrow NARROW_V2.json --enum ENUM_V2.json --results RESULTS_V1.json   (queue unit, stage verify)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcore as V  # noqa: E402

SCRIPT = f'results/{V.CAMPAIGN}/roles/verifier/code/v3_exact.py'
EXREF = V.RES / 'roles' / 'exact_reference'
LIMITS = {'inc_acc_loss': 0.01, 'occ_acc_loss': 0.01, 'inc_nll_excess': 0.01, 'occ_nll_excess': 0.01, 'inc_brier_excess': 0.005,
          'occ_brier_excess': 0.005, 'occ_brier_excess_S0': 0.005, 'occ_brier_excess_S1': 0.005}
EPS = 1e-12
CLIP = 1e-6


# ---------------------------------------------------------------- the law
def load_law(L):
    N = int(L['N'])
    st = []
    for s in L['states']:
        P, pi = Fr(int(s['n']), N), Fr(int(s['pi_pct']), 100)
        i0, i1 = [Fr(int(x), 100) for x in s['inc_S0_pct']], [Fr(int(x), 100) for x in s['inc_S1_pct']]
        o0, o1 = [Fr(int(x), 100) for x in s['occ_S0_pct']], [Fr(int(x), 100) for x in s['occ_S1_pct']]
        st.append({'id': s['id'], 'n': int(s['n']), 'P': P, 'pi': pi, 'qI': [(1 - pi) * a + pi * b for a, b in zip(i0, i1)],
                   'qO': [(1 - pi) * a + pi * b for a, b in zip(o0, o1)], 'qO0': o0, 'qO1': o1})
    problems = []
    if sum(s['n'] for s in st) != N:
        problems.append('n does not sum to N')
    for s in st:
        for k in ('qI', 'qO', 'qO0', 'qO1'):
            if sum(s[k]) != 1:
                problems.append(f'{s["id"]} {k} does not sum to 1')
    p1 = sum(s['P'] * s['pi'] for s in st)
    return st, p1, problems


def argmax_first(v):
    m = max(v)
    return next(i for i, x in enumerate(v) if x == m)


def cell_means(st, cells, key, K):
    out = {}
    for c in range(K):
        mem = [s for s, x in zip(st, cells) if x == c]
        if mem:
            W = sum(s['P'] for s in mem)
            out[c] = [sum(s['P'] * s[key][j] for s in mem) / W for j in range(len(mem[0][key]))]
    return out


def release_of_partition(st, k, l, KI, KO):
    TI, TO = cell_means(st, k, 'qI', KI), cell_means(st, l, 'qO', KO)
    return [TI[a] for a in k], [TO[b] for b in l]


# ---------------------------------------------------------------- metrics of a release (per-state vectors rI, rO)
def _brier(r, q):
    return sum(x * x for x in r) - 2 * sum(a * b for a, b in zip(q, r)) + 1


def _nll(r, q):
    return -sum(float(a) * math.log(float(b)) for a, b in zip(q, r) if a != 0)


def bayes_auc(groups, p1):
    """groups: list of (w1, w0) exact masses per released value. Exact AUC of ranking values by w1 / (w1 + w0), ties 1/2."""
    num = Fr(0)
    for a1, a0 in groups:
        for b1, b0 in groups:
            lhs, rhs = a1 * b0, b1 * a0                  # value a vs value b: score_a > score_b iff a1 b0 > b1 a0
            if lhs > rhs:
                num += a1 * b0
            elif lhs == rhs:
                num += a1 * b0 / 2
    return num / (p1 * (1 - p1))


def plugin_mi(groups, p1):
    out = 0.0
    rb = float(p1)
    for w1, w0 in groups:
        Pc = w1 + w0
        if Pc == 0:
            continue
        r = min(max(float(w1 / Pc), 1e-9), 1 - 1e-9)
        out += float(Pc) * (r * math.log(r / rb) + (1 - r) * math.log((1 - r) / (1 - rb)))
    return out


def _groups(st, keys):
    g = {}
    for s, key in zip(st, keys):
        w = g.setdefault(key, [Fr(0), Fr(0)])
        w[0] += s['P'] * s['pi']
        w[1] += s['P'] * (1 - s['pi'])
    return [tuple(x) for x in g.values()]


def metrics(st, p1, rI, rO, k=None, l=None):
    m = {}
    m['inc_nll'] = sum(float(s['P']) * _nll(r, s['qI']) for s, r in zip(st, rI))
    m['occ_nll'] = sum(float(s['P']) * _nll(r, s['qO']) for s, r in zip(st, rO))
    m['inc_acc'] = float(sum(s['P'] * s['qI'][argmax_first(r)] for s, r in zip(st, rI)))
    m['occ_acc'] = float(sum(s['P'] * s['qO'][argmax_first(r)] for s, r in zip(st, rO)))
    m['inc_brier'] = float(sum(s['P'] * _brier(r, s['qI']) for s, r in zip(st, rI)))
    m['occ_brier'] = float(sum(s['P'] * _brier(r, s['qO']) for s, r in zip(st, rO)))
    m['occ_brier_S0'] = float(sum(s['P'] * (1 - s['pi']) * _brier(r, s['qO0']) for s, r in zip(st, rO)) / (1 - p1))
    m['occ_brier_S1'] = float(sum(s['P'] * s['pi'] * _brier(r, s['qO1']) for s, r in zip(st, rO)) / p1)
    vI = [tuple(r) for r in rI]
    vO = [tuple(r) for r in rO]
    vP = list(zip(vI, vO))
    for name, keys in (('pair', vP), ('income', vI), ('occupation', vO)):
        m[f'auc_{name}'] = float(bayes_auc(_groups(st, keys), p1))
    m['mi_value_pair'] = plugin_mi(_groups(st, vP), p1)
    m['values'] = {'pair': len(set(vP)), 'income': len(set(vI)), 'occupation': len(set(vO))}
    if k is not None:
        cP = list(zip(k, l))
        m['mi_cell_pair'] = plugin_mi(_groups(st, cP), p1)
        m['mi_cell_income'] = plugin_mi(_groups(st, list(k)), p1)
        m['mi_cell_occupation'] = plugin_mi(_groups(st, list(l)), p1)
        m['cells'] = {'pair': len(set(cP)), 'income': len(set(k)), 'occupation': len(set(l))}
    return m


def excess(m, full):
    e = {'inc_acc_loss': full['inc_acc'] - m['inc_acc'], 'occ_acc_loss': full['occ_acc'] - m['occ_acc'],
         'inc_nll_excess': m['inc_nll'] - full['inc_nll'], 'occ_nll_excess': m['occ_nll'] - full['occ_nll'],
         'inc_brier_excess': m['inc_brier'] - full['inc_brier'], 'occ_brier_excess': m['occ_brier'] - full['occ_brier'],
         'occ_brier_excess_S0': m['occ_brier_S0'] - full['occ_brier_S0'], 'occ_brier_excess_S1': m['occ_brier_S1'] - full['occ_brier_S1']}
    fails = [k for k, lim in LIMITS.items() if e[k] > lim + EPS]
    return e, fails


def release_record(st, p1, full, k, l, KI, KO):
    rI, rO = release_of_partition(st, k, l, KI, KO)
    m = metrics(st, p1, rI, rO, k, l)
    e, fails = excess(m, full)
    m.update({'excess': e, 'feasible': not fails, 'fails': fails})
    return m


# ---------------------------------------------------------------- the exact objective
def clipn(v):
    v = [max(float(x), CLIP) for x in v]
    s = sum(v)
    return [x / s for x in v]


def objective(st, p1, k, l, CI, CO, lam, rp):
    J = sum(float(s['P']) * (-sum(float(a) * math.log(CI[kk][j]) for j, a in enumerate(s['qI']))
                             - sum(float(a) * math.log(CO[ll][j]) for j, a in enumerate(s['qO'])))
            for s, kk, ll in zip(st, k, l))
    if lam > 0:
        J += lam * (rp['w_pair'] * plugin_mi(_groups(st, list(zip(k, l))), p1)
                    + rp['w_loc'] * (plugin_mi(_groups(st, list(k)), p1) + plugin_mi(_groups(st, list(l)), p1)))
        if rp['kappa'] > 0:
            p0f, p1f = 1.0 - float(p1), float(p1)
            g = 0.0
            for s, ll in zip(st, l):
                c = CO[ll]
                g += float(s['P']) * ((1 - float(s['pi'])) / p0f * sum((float(a) - b) ** 2 for a, b in zip(s['qO0'], c))
                                      + float(s['pi']) / p1f * sum((float(a) - b) ** 2 for a, b in zip(s['qO1'], c)))
            J += lam * rp['kappa'] * g
    return J


def refreshed_codewords(st, k, l, CI, CO, KI, KO):
    """update_book's codeword rule (declared): occupied cells take the mean posterior of their members (clipped, renormalised);
    unoccupied cells keep the current codewords."""
    TI, TO = cell_means(st, k, 'qI', KI), cell_means(st, l, 'qO', KO)
    nCI = [clipn(TI[c]) if c in TI else list(CI[c]) for c in range(KI)]
    nCO = [clipn(TO[c]) if c in TO else list(CO[c]) for c in range(KO)]
    return nCI, nCO


# ---------------------------------------------------------------- comparison helpers
def cmp_metrics(mine, theirs, tol):
    out = []
    for key, v in theirs.items():
        if key in ('excess',):
            for k2, v2 in v.items():
                if abs(mine['excess'][k2] - v2) > tol:
                    out.append(f'excess.{k2}: mine {mine["excess"][k2]} theirs {v2}')
        elif key in ('values', 'cells'):
            if mine.get(key) != v:
                out.append(f'{key}: mine {mine.get(key)} theirs {v}')
        elif key in ('feasible', 'fails'):
            if (mine[key] if key == 'feasible' else sorted(mine[key])) != (v if key == 'feasible' else sorted(v)):
                out.append(f'{key}: mine {mine[key]} theirs {v}')
        elif key in mine and isinstance(v, (int, float)) and not isinstance(v, bool):
            if abs(mine[key] - v) > tol:
                out.append(f'{key}: mine {mine[key]} theirs {v}')
    return out


def verify_law(name, L, common, narrow, enum, results, tol):
    st, p1, prob = load_law(L)
    KI, KO = int(common['KI']), int(common['KO'])
    rp = {k: float(common[k]) for k in ('w_pair', 'w_loc', 'kappa')}
    rows = {'law_problems': prob}
    for sid, pct in (L.get('derived_qO_check') or {}).items():          # the law's own derived-qO check (percent)
        s0 = next(x for x in st if x['id'] == sid)
        if [x * 100 for x in s0['qO']] != [Fr(int(y)) for y in pct]:
            prob.append(f'derived qO of {sid} differs from derived_qO_check')
    mism = []
    n_cmp = 0
    nl = narrow['laws'][name]
    # posteriors
    for s in st:
        q = nl['posteriors'][s['id']]
        n_cmp += 1
        if q['n'] != s['n'] or abs(q['pi'] - float(s['pi'])) > tol or any(
                abs(a - float(b)) > tol for key in ('qI', 'qO', 'qO0', 'qO1') for a, b in zip(q[key], s[key])):
            mism.append(f'posteriors {s["id"]}')
    if abs(nl['p1'] - float(p1)) > tol:
        mism.append(f'p1 mine {float(p1)} theirs {nl["p1"]}')
    # full release (each state its own cell)
    rI, rO = [s['qI'] for s in st], [s['qO'] for s in st]
    full = metrics(st, p1, rI, rO, list(range(len(st))), list(range(len(st))))
    mm = cmp_metrics(full, nl['full_release'], tol)
    n_cmp += 1
    mism += [f'full_release {x}' for x in mm]
    rows['full_release'] = full
    # decision-only release
    kd = [argmax_first(s['qI']) for s in st]
    ld = [argmax_first(s['qO']) for s in st]
    labs = sorted(set(ld))
    ld_idx = [labs.index(x) for x in ld]
    labsI = sorted(set(kd))
    kd_idx = [labsI.index(x) for x in kd]
    dec = release_record(st, p1, full, kd_idx, ld_idx, len(labsI), len(labs))
    mm = cmp_metrics(dec, {k: v for k, v in nl['decision_only'].items() if k not in ('inc_cell', 'occ_cell')}, tol)
    if 'inc_cell' in nl['decision_only']:
        theirs = release_record(st, p1, full, nl['decision_only']['inc_cell'], nl['decision_only']['occ_cell'],
                                max(nl['decision_only']['inc_cell']) + 1, max(nl['decision_only']['occ_cell']) + 1)
        if theirs['auc_pair'] != dec['auc_pair'] or theirs['excess'] != dec['excess']:
            mm.append('their decision cells give a different release than argmax decisions')
    n_cmp += 1
    mism += [f'decision_only {x}' for x in mm]
    rows['decision_only'] = {k: v for k, v in dec.items()}
    # witness / comparator (laws.json cells)
    if isinstance(L.get('witness'), dict) and 'inc_cell' in L['witness'] and results is not None \
            and isinstance(results['laws'][name].get('witness'), dict):
        w = L['witness']
        wr = release_record(st, p1, full, w['inc_cell'], w['occ_cell'], KI, KO)
        mm = cmp_metrics(wr, {k: v for k, v in results['laws'][name]['witness'].items() if k not in ('inc_cell', 'occ_cell')}, tol)
        n_cmp += 1
        mism += [f'witness {x}' for x in mm]
        rows['witness'] = wr
    # seeds: twin, path states, HQ path, traces (objective chain)
    jd = []
    for seed, sd in nl['seeds'].items():
        tw = sd['twin']
        twr = release_record(st, p1, full, tw['inc_cell'], tw['occ_cell'], KI, KO)
        mm = cmp_metrics(twr, {k: v for k, v in tw.items() if k not in ('inc_cell', 'occ_cell')}, tol)
        n_cmp += 1
        mism += [f'seed{seed} twin {x}' for x in mm]
        TI, TO = cell_means(st, tw['inc_cell'], 'qI', KI), cell_means(st, tw['occ_cell'], 'qO', KO)
        if len(TI) == KI and len(TO) == KO and tw.get('max_abs_codeword_minus_B1', 1) < 1e-12:
            CI = [clipn(TI[c]) for c in range(KI)]
            CO = [clipn(TO[c]) for c in range(KO)]
            jt = objective(st, p1, tw['inc_cell'], tw['occ_cell'], CI, CO, 0.0, rp)
            jd.append(abs(jt - tw['J_task_code']))
            if abs(jt - tw['J_task_code']) > 1e-9:
                mism.append(f'seed{seed} twin J_task mine {jt} theirs {tw["J_task_code"]}')
            chainable = True
        else:
            CI = CO = None
            chainable = False
        for lam, ps in sd['path'].items():
            pr = release_record(st, p1, full, ps['inc_cell'], ps['occ_cell'], KI, KO)
            mm = cmp_metrics(pr, {k: v for k, v in ps.items() if k not in ('inc_cell', 'occ_cell')}, tol)
            n_cmp += 1
            mism += [f'seed{seed} path {lam} {x}' for x in mm]
            if chainable:
                jtw = objective(st, p1, tw['inc_cell'], tw['occ_cell'], CI, CO, float(lam), rp)
                jd.append(abs(jtw - ps['J_twin_same_lam_code']))
                if abs(jtw - ps['J_twin_same_lam_code']) > 1e-9:
                    mism.append(f'seed{seed} path {lam} J_twin_same_lam mine {jtw} theirs {ps["J_twin_same_lam_code"]}')
        for lam, ps in sd.get('hq_path', {}).items():
            pr = release_record(st, p1, full, ps['inc_cell'], ps['occ_cell'], KI, KO)
            mm = cmp_metrics(pr, {k: v for k, v in ps.items() if k not in ('inc_cell', 'occ_cell')}, tol)
            n_cmp += 1
            mism += [f'seed{seed} hq_path {lam} {x}' for x in mm]
        # the narrow trace: chain the books from the twin and recompute J at the current state and at every proposal
        if chainable:
            bk = (CI, CO)
            cur = (list(tw['inc_cell']), list(tw['occ_cell']))
            last_lam = None
            for stp in sd['trace']:
                lam = float(stp['lam'])
                if stp.get('reset_to_twin_before_lam'):
                    bk, cur = (CI, CO), (list(tw['inc_cell']), list(tw['occ_cell']))
                jc = objective(st, p1, cur[0], cur[1], bk[0], bk[1], lam, rp)
                jd.append(abs(jc - stp['J_current']))
                if abs(jc - stp['J_current']) > 1e-9:
                    mism.append(f'seed{seed} trace lam {lam} step {stp["step"]} J_current mine {jc} theirs {stp["J_current"]}')
                nCI, nCO = refreshed_codewords(st, cur[0], cur[1], bk[0], bk[1], KI, KO)
                props = stp['narrow']['proposals']
                for j, p in enumerate(props):
                    jp = objective(st, p1, p['k'], p['l'], nCI, nCO, lam, rp)
                    jd.append(abs(jp - p['J_code']))
                    if abs(jp - p['J_code']) > 1e-9:
                        mism.append(f'seed{seed} trace lam {lam} step {stp["step"]} proposal t={p["t"]} J mine {jp} theirs {p["J_code"]}')
                for key in ('card', 'wide'):
                    b = stp['narrow'].get(key)
                    if b and b.get('best_k') is not None:
                        jb = objective(st, p1, b['best_k'], b['best_l'], nCI, nCO, lam, rp)
                        jd.append(abs(jb - b['best_J_code']))
                        if abs(jb - b['best_J_code']) > 1e-9:
                            mism.append(f'seed{seed} trace lam {lam} step {stp["step"]} {key} best J mine {jb} theirs {b["best_J_code"]}')
                a = stp.get('accepted')
                if a is not None and a >= 0:
                    p = props[a]
                    bk, cur = (nCI, nCO), (list(p['k']), list(p['l']))
                last_lam = lam
            rows.setdefault('trace_chained', []).append(seed)
    # enumeration optima: codewords = cell means
    if enum is not None and name in enum['laws']:
        for key, o in enum['laws'][name]['optima'].items():
            if not isinstance(o, dict) or 'inc_cell' not in o:
                continue
            orr = release_record(st, p1, full, o['inc_cell'], o['occ_cell'], KI, KO)
            mm = cmp_metrics(orr, {k: v for k, v in o.items() if k not in ('inc_cell', 'occ_cell', 'value')}, tol)
            n_cmp += 1
            mism += [f'enum {key} {x}' for x in mm]
            if key.startswith('obj_') and 'value' in o:
                lam = float(key[4:])
                TI, TO = cell_means(st, o['inc_cell'], 'qI', KI), cell_means(st, o['occ_cell'], 'qO', KO)
                CIe = [clipn(TI[c]) if c in TI else [1.0 / len(st[0]['qI'])] * len(st[0]['qI']) for c in range(KI)]
                COe = [clipn(TO[c]) if c in TO else [1.0 / len(st[0]['qO'])] * len(st[0]['qO']) for c in range(KO)]
                je = objective(st, p1, o['inc_cell'], o['occ_cell'], CIe, COe, lam, rp)
                jd.append(abs(je - o['value']))
                if abs(je - o['value']) > 1e-9:
                    mism.append(f'enum {key} value mine {je} theirs {o["value"]}')
    rows['objective_max_abs_diff'] = max(jd) if jd else None
    rows['n_objective_values'] = len(jd)
    mism = [f'law: {x}' for x in rows['law_problems']] + mism
    return mism, n_cmp, rows


def enumerate_small(st, p1, full, KI, KO, rp, lams, max_assign=20000):
    """Brute-force enumeration (only small laws): the feasible count and the objective optimum per lam (codewords = cell means)."""
    m = len(st)
    if KI ** m * KO ** m > max_assign:
        return None
    best = {lam: (math.inf, None) for lam in lams}
    nfeas = 0
    for k in itertools.product(range(KI), repeat=m):
        for l in itertools.product(range(KO), repeat=m):
            r = release_record(st, p1, full, list(k), list(l), KI, KO)
            nfeas += r['feasible']
            TI, TO = cell_means(st, k, 'qI', KI), cell_means(st, l, 'qO', KO)
            CI = [clipn(TI[c]) if c in TI else [0.5, 0.5] for c in range(KI)]
            CO = [clipn(TO[c]) if c in TO else [0.2] * 5 for c in range(KO)]
            for lam in lams:
                j = objective(st, p1, list(k), list(l), CI, CO, lam, rp)
                if j < best[lam][0]:
                    best[lam] = (j, (list(k), list(l)))
    return {'n_assignments': KI ** m * KO ** m, 'n_feasible_assignments': nfeas, 'optima': best}


def _resolved_laws(laws):
    out = {}
    for name, L in laws['laws'].items():
        if 'states' not in L and L.get('same_population_law_as'):
            ref, div = laws['laws'][L['same_population_law_as']], int(L['row_divisor'])
            L = dict(L, states=[dict(x, n=x['n'] // div) for x in ref['states']])
        out[name] = L
    return out


def task_view(m, full):
    """The route E 'task' block from own metrics: absolute NLL / Brier, accuracy loss against the full release, per-SEX Brier."""
    return {'inc_nll': m['inc_nll'], 'occ_nll': m['occ_nll'], 'inc_acc_loss': full['inc_acc'] - m['inc_acc'],
            'occ_acc_loss': full['occ_acc'] - m['occ_acc'], 'inc_brier': m['inc_brier'], 'occ_brier': m['occ_brier'],
            'occ_brier_men': m['occ_brier_S0'], 'occ_brier_women': m['occ_brier_S1']}


def route_e(tag, route_e_name, advancement_path):
    laws = _resolved_laws(V.load_json(EXREF / 'laws.json'))
    common = V.load_json(EXREF / 'laws.json')['common']
    KI, KO = int(common['KI']), int(common['KO'])
    rp = {k: float(common[k]) for k in ('w_pair', 'w_loc', 'kappa')}
    re_ = V.load_json(EXREF / route_e_name)
    adv = V.load_json(advancement_path)
    adv_cells = {(tuple(c['cell'])): c for c in adv['result']['routes']['E']['SV1']['cells']}
    cache, mism, n_rel, n_j, jmax, dmax = {}, [], 0, 0, 0.0, 0.0
    for c in re_['cells']:
        law = c['law'].split(':')[0]
        if law not in cache:
            st, p1, _ = load_law(laws[law])
            full = metrics(st, p1, [s['qI'] for s in st], [s['qO'] for s in st])
            cache[law] = (st, p1, full)
        st, p1, full = cache[law]
        mine_task = {}
        for role in ('original', 'revised', 'reference', 'unrestricted'):
            x = c.get(role)
            if not isinstance(x, dict) or 'inc_cell' not in x:
                continue
            k, l = list(x['inc_cell']), list(x['occ_cell'])
            r = release_record(st, p1, full, k, l, KI, KO)
            n_rel += 1
            tv = task_view(r, full)
            mine_task[role] = (r, tv)
            for vw in ('pair', 'income', 'occupation'):
                if abs(r[f'auc_{vw}'] - x['auc'][vw]) > 1e-9:
                    mism.append(f'{c["law"]}@{c["lam"]} {role} auc_{vw}: mine {r[f"auc_{vw}"]} theirs {x["auc"][vw]}')
            for kk, vv in x['task'].items():
                if abs(tv[kk] - vv) > 1e-9:
                    mism.append(f'{c["law"]}@{c["lam"]} {role} {kk}: mine {tv[kk]} theirs {vv}')
            if 'feasible' in x and (r['feasible'] != x['feasible'] or sorted(r['fails']) != sorted(x.get('fails', []))):
                mism.append(f'{c["law"]}@{c["lam"]} {role} feasibility: mine {r["feasible"]} {r["fails"]} theirs {x["feasible"]} {x.get("fails")}')
            if role == 'unrestricted' and 'J' in x:                       # codewords = cell means (the enumeration's rule)
                TI, TO = cell_means(st, k, 'qI', KI), cell_means(st, l, 'qO', KO)
                CI = [clipn(TI[q]) if q in TI else [0.5, 0.5] for q in range(KI)]
                CO = [clipn(TO[q]) if q in TO else [0.2] * 5 for q in range(KO)]
                j = objective(st, p1, k, l, CI, CO, float(c['lam']), rp)
                n_j += 1
                jmax = max(jmax, abs(j - x['J']))
                if abs(j - x['J']) > 1e-9:
                    mism.append(f'{c["law"]}@{c["lam"]} unrestricted J: mine {j} theirs {x["J"]}')
        a = adv_cells.get((c['law'], c['lam']))
        if a is not None and 'original' in mine_task and 'revised' in mine_task:
            (ro, to), (rr, tr) = mine_task['original'], mine_task['revised']
            mine_d = {f'auc_{vw}': rr[f'auc_{vw}'] - ro[f'auc_{vw}'] for vw in ('pair', 'income', 'occupation')}
            mine_d.update({kk: tr[kk] - to[kk] for kk in to})
            mine_d['tot_nll'] = (tr['inc_nll'] + tr['occ_nll']) - (to['inc_nll'] + to['occ_nll'])
            for kk, vv in a['usefulness']['deltas'].items():
                dmax = max(dmax, abs(mine_d[kk] - vv))
                if abs(mine_d[kk] - vv) > 1e-9:
                    mism.append(f'{c["law"]}@{c["lam"]} delta {kk}: mine {mine_d[kk]} recorded {vv}')
        elif a is None:
            mism.append(f'{c["law"]}@{c["lam"]}: no advancement cell')
    chk = {'id': f'V3.ROUTE_E__{tag}', 'block': 'V3', 'kind': 'numerical', 'status': 'PASS' if not mism else 'FAIL',
           'summary': f'{len(re_["cells"])} route E cells of {route_e_name}: exact AUCs, task metrics and feasibility recomputed at '
                      f'{n_rel} reported partitions (original, revised, reference, unrestricted; B1 tables = exact cell means); '
                      f'{n_j} unrestricted objectives (max |diff| {jmax:.2e}); usefulness deltas against ADVANCEMENT.json route E '
                      f'(max |diff| {dmax:.2e}); {len(mism)} mismatches' + (f': {mism[:4]}' if mism else ''),
           'detail': {'n_cells': len(re_['cells']), 'n_partitions': n_rel, 'n_objectives': n_j, 'objective_max_abs_diff': jmax,
                      'delta_max_abs_diff': dmax, 'n_mismatch': len(mism)},
           'record': {'mismatches': mism, 'inputs': {route_e_name: V.sha_file(EXREF / route_e_name),
                                                     'ADVANCEMENT.json': V.sha_file(advancement_path)}},
           'script': SCRIPT}
    V.Checks().append([chk])
    print(json.dumps({chk['id']: (chk['status'], chk['summary'])}, indent=1))
    return 0


def main_real(tag, narrow_name, enum_name, results_name):
    laws_p = EXREF / 'laws.json'
    _, log = V.git('log', '-1', '--format=%H %cI', '--', str(laws_p.relative_to(V.ROOT)))
    laws = V.load_json(laws_p)
    narrow = V.load_json(EXREF / narrow_name)
    enum = V.load_json(EXREF / enum_name) if enum_name else None
    results = V.load_json(EXREF / results_name) if results_name else None
    tol = 1e-9
    checks = []
    for name, L in laws['laws'].items():
        if 'states' not in L and L.get('same_population_law_as'):          # E: the population law of C on N / row_divisor rows
            ref, div = laws['laws'][L['same_population_law_as']], int(L['row_divisor'])
            if any(x['n'] % div for x in ref['states']) or sum(x['n'] for x in ref['states']) // div != int(L['N']):
                raise ValueError(f'{name}: rows of {L["same_population_law_as"]} not divisible by {div}')
            L = dict(L, states=[dict(x, n=x['n'] // div) for x in ref['states']])
        mism, n_cmp, rows = verify_law(name, L, laws['common'], narrow, enum, results, tol)
        st, p1, _ = load_law(L)
        small = None
        if enum is not None and name in enum['laws']:
            full = rows['full_release']
            rp = {k: float(laws['common'][k]) for k in ('w_pair', 'w_loc', 'kappa')}
            small = enumerate_small(st, p1, full, int(laws['common']['KI']), int(laws['common']['KO']), rp,
                                    [float(x) for x in laws['common']['lambda_grid']])
            if small is not None:
                e = enum['laws'][name]
                if small['n_assignments'] != e['n_assignments'] or small['n_feasible_assignments'] != e['n_feasible_assignments']:
                    mism.append(f'enumeration counts mine {small["n_assignments"]}/{small["n_feasible_assignments"]} theirs '
                                f'{e["n_assignments"]}/{e["n_feasible_assignments"]}')
                for lam, (j, part) in small['optima'].items():
                    o = e['optima'].get(f'obj_{lam}')
                    if o is not None and abs(j - o['value']) > 1e-9:
                        mism.append(f'enumeration optimum lam {lam} mine {j} theirs {o["value"]}')
        checks.append({'id': f'V3.LAW_{name}__{tag}', 'block': 'V3', 'kind': 'numerical', 'status': 'PASS' if not mism else 'FAIL',
                       'summary': f'{name}: {n_cmp} release records (posteriors, full, decision-only, witness, twins, paths, HQ paths, '
                                  f'enumeration optima) recomputed from laws.json; {rows["n_objective_values"]} objective values '
                                  f'(max |diff| {rows["objective_max_abs_diff"]}); own brute-force enumeration '
                                  f'{"run" if small else "not run (too many assignments for the pure-Python reference)"}; '
                                  f'{len(mism)} mismatches' + (f': {mism[:6]}' if mism else ''),
                       'detail': {'n_release_records': n_cmp, 'n_objective_values': rows['n_objective_values'],
                                  'objective_max_abs_diff': rows['objective_max_abs_diff'], 'n_mismatch': len(mism),
                                  'own_enumeration': None if small is None else {k: v for k, v in small.items() if k != 'optima'}},
                       'record': {'mismatches': mism, 'law_problems': rows['law_problems'], 'laws_commit': log,
                                  'inputs': {'narrow': narrow_name, 'narrow_sha256': V.sha_file(EXREF / narrow_name),
                                             'enum': enum_name, 'enum_sha256': V.sha_file(EXREF / enum_name) if enum_name else None,
                                             'results': results_name,
                                             'results_sha256': V.sha_file(EXREF / results_name) if results_name else None},
                                  'full_release': rows['full_release'], 'decision_only': rows['decision_only'],
                                  'witness': rows.get('witness')},
                       'script': SCRIPT})
    V.Checks().append(checks)
    print(json.dumps({c['id']: (c['status'], c['summary'][:300]) for c in checks}, indent=1, default=V._json_default))
    return 0                                     # the script ran to completion; statuses (FAIL included) live in CHECKS.json


# ---------------------------------------------------------------- selftest (an invented law, not a registered one)
def selftest():
    res = []
    L = {'N': 100, 'states': [
        {'id': 'x1', 'n': 40, 'pi_pct': 20, 'inc_S0_pct': [70, 30], 'inc_S1_pct': [60, 40], 'occ_S0_pct': [50, 20, 10, 10, 10],
         'occ_S1_pct': [30, 40, 10, 10, 10]},
        {'id': 'x2', 'n': 35, 'pi_pct': 60, 'inc_S0_pct': [40, 60], 'inc_S1_pct': [40, 60], 'occ_S0_pct': [20, 20, 40, 10, 10],
         'occ_S1_pct': [20, 20, 40, 10, 10]},
        {'id': 'x3', 'n': 25, 'pi_pct': 60, 'inc_S0_pct': [40, 60], 'inc_S1_pct': [40, 60], 'occ_S0_pct': [20, 20, 40, 10, 10],
         'occ_S1_pct': [20, 20, 40, 10, 10]}]}
    st, p1, prob = load_law(L)
    res.append(('law_sums', not prob and p1 == Fr(40 * 20 + 35 * 60 + 25 * 60, 10000), prob))
    # replicate the law into rows and compare every metric with a row-level computation
    rows = []
    for s in st:
        rows += [s] * s['n']
    full = metrics(st, p1, [s['qI'] for s in st], [s['qO'] for s in st], [0, 1, 2], [0, 1, 2])
    nll = np.mean([-sum(float(a) * math.log(float(a)) for a in s['qI']) for s in rows])
    res.append(('full_inc_nll_rows', abs(full['inc_nll'] - nll) < 1e-12, (full['inc_nll'], nll)))
    # Bayes AUC by brute force over replicated rows with fractional SEX: pairs of (row, S) weighted
    sc = [float(s['pi']) for s in rows]
    pos = [(float(s['pi']), v) for s, v in zip(rows, sc)]
    neg = [(1 - float(s['pi']), v) for s, v in zip(rows, sc)]
    num = sum(wp * wn * (1.0 if a > b else 0.5 if a == b else 0.0) for wp, a in pos for wn, b in neg)
    den = sum(w for w, _ in pos) * sum(w for w, _ in neg)
    # the full release merges x2 and x3 (identical vectors): AUC on merged values equals the row-level AUC of the posterior score
    res.append(('bayes_auc_vs_rows', abs(full['auc_pair'] - num / den) < 1e-12, (full['auc_pair'], num / den)))
    res.append(('merged_values', full['values']['pair'] == 2 and full['cells']['pair'] == 3, full['values']))
    # null law: S independent of everything -> AUC exactly 0.5 in every view for every release
    Ln = {'N': 10, 'states': [dict(s, pi_pct=40) for s in L['states']]}
    Ln['states'][0]['n'], Ln['states'][1]['n'], Ln['states'][2]['n'] = 4, 3, 3
    stn, p1n, _ = load_law(Ln)
    fn = metrics(stn, p1n, [s['qI'] for s in stn], [s['qO'] for s in stn])
    res.append(('null_auc_half', all(fn[f'auc_{v}'] == 0.5 for v in ('pair', 'income', 'occupation')), fn['auc_pair']))
    # objective: task part equals the row mean cross-entropy; MI part equals a row-level plug-in on pi
    k, l = [0, 1, 1], [0, 1, 1]
    TI, TO = cell_means(st, k, 'qI', 2), cell_means(st, l, 'qO', 2)
    CI, CO = [clipn(TI[c]) for c in range(2)], [clipn(TO[c]) for c in range(2)]
    rp = {'w_pair': 1.0, 'w_loc': 1.0, 'kappa': 1.0}
    j0 = objective(st, p1, k, l, CI, CO, 0.0, rp)
    ce = np.mean([-sum(float(a) * math.log(CI[kk][j]) for j, a in enumerate(s['qI'])) - sum(float(a) * math.log(CO[ll][j])
                                                                                             for j, a in enumerate(s['qO']))
                  for s, kk, ll in zip(rows, sum([[x] * s['n'] for x, s in zip(k, st)], []), sum([[x] * s['n'] for x, s in zip(l, st)], []))])
    res.append(('objective_task_part', abs(j0 - ce) < 1e-12, (j0, ce)))
    j1 = objective(st, p1, k, l, CI, CO, 2.0, rp)
    pis = np.array([float(s['pi']) for s in rows])
    cells = np.array(sum([[x] * s['n'] for x, s in zip(k, st)], []))

    def mi_rows(c, w):
        out = 0.0
        rb = w.mean()
        for v in np.unique(c):
            r = np.clip(w[c == v].mean(), 1e-9, 1 - 1e-9)
            out += (c == v).mean() * (r * np.log(r / rb) + (1 - r) * np.log((1 - r) / (1 - rb)))
        return out
    mi = mi_rows(cells * 2 + cells, pis) + mi_rows(cells, pis) + mi_rows(cells, pis)
    g = np.mean([(1 - float(s['pi'])) / (1 - float(p1)) * sum((float(a) - b) ** 2 for a, b in zip(s['qO0'], CO[ll]))
                 + float(s['pi']) / float(p1) * sum((float(a) - b) ** 2 for a, b in zip(s['qO1'], CO[ll]))
                 for s, ll in zip(rows, sum([[x] * s['n'] for x, s in zip(l, st)], []))])
    res.append(('objective_full', abs(j1 - (ce + 2.0 * mi + 2.0 * g)) < 1e-12, (j1, ce + 2 * mi + 2 * g)))
    # excess and feasibility of a coarse release
    rec = release_record(st, p1, full, [0, 0, 0], [0, 0, 0], 1, 1)
    res.append(('coarse_release_infeasible', not rec['feasible'] and rec['excess']['occ_nll_excess'] > 0.01, rec['fails']))
    rec2 = release_record(st, p1, full, [0, 1, 1], [0, 1, 1], 2, 2)
    res.append(('collision_merge_lossless', rec2['feasible'] and abs(rec2['excess']['occ_nll_excess']) < 1e-12, rec2['excess']))
    # brute-force enumeration on the invented law
    en = enumerate_small(st, p1, full, 2, 2, rp, [0.5])
    res.append(('enumeration_runs', en is not None and en['n_assignments'] == 64 and en['n_feasible_assignments'] >= 1, en and en['n_feasible_assignments']))
    tv = task_view(rec2, full)
    res.append(('task_view', tv['inc_acc_loss'] == full['inc_acc'] - rec2['inc_acc'] and tv['occ_brier_men'] == rec2['occ_brier_S0']
                and abs(tv['occ_nll'] - rec2['occ_nll']) == 0.0, tv))
    rl = _resolved_laws({'laws': {'P': L, 'Q': {'N': 20, 'same_population_law_as': 'P', 'row_divisor': 5}}})
    stq, p1q, _ = load_law(rl['Q'])
    res.append(('resolved_derived_law', [x['n'] for x in rl['Q']['states']] == [8, 7, 5] and p1q == p1
                and [x['P'] for x in stq] == [x['P'] for x in st], [x['n'] for x in rl['Q']['states']]))
    n_ok = sum(r[1] for r in res)
    for name, ok, info in res:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  [{info}]'))
    print(f'v3_exact selftest: {n_ok}/{len(res)}')
    return n_ok == len(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--tag', default='V1')
    ap.add_argument('--narrow', default='NARROW_V2.json')
    ap.add_argument('--enum', default='ENUM_V2.json')
    ap.add_argument('--results', default='RESULTS_V1.json')
    ap.add_argument('--route-e', default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not os.environ.get('FSA_UNIT_ID'):
        sys.exit('real runs go through the queue (fsa.jobqueue run --stage verify)')
    if a.route_e:
        sys.exit(route_e(a.tag, a.route_e, V.RES / 'decisions' / 'ADVANCEMENT.json'))
    sys.exit(main_real(a.tag, a.narrow, a.enum, a.results))
