"""Tiny exact instances of the constrained direct-recovery release problem (campaign `pcrl_direct_recovery_frontier_v1`, DRF1;
investigator A). SYNTHETIC / INVENTED data only: it reads no file, no label and no real record.

For each tiny instance (40-300 invented people, I = 2-3 income fine cells -> 2 income atoms, J = 3-6 occupation fine cells -> 2-3
occupation atoms, identity start A(k, m) = k), EVERY assignment (fine cell -> atom) is enumerated. For every assignment this module
recomputes, independently of drf.solver's internal arithmetic (per person, from the raw synthetic arrays):
  - the B1 emitted tables (mean teacher posterior of the atom's people, clipped at 1e-6, renormalized), the per-person emitted vectors,
    the 8 spends over the identity (income / occupation mean NLL, mean summed Brier, accuracy loss; men's and women's mean summed
    occupation Brier);
  - every AUC* and plug-in MI through drf.finite_law, on two laws: ATOM (wires = atom ids; the solver's own objective) and OBSERVABLE
    (wires = the emitted float64 vectors, identical vectors merged; what a reader actually sees). They differ only when two atoms emit
    identical vectors.
Then, per budget level (unconstrained; per-metric quantiles of the positive spends; zero), it compares the exhaustive constrained
optimum of J_auc and J_mi with drf.solver.solve from the identity (several seeds; with and without restarts): gap, feasibility,
single-move local optimality (all feasible single-cell moves from the solver's answer, from the exhaustive table), whether the optimum
is reachable from the identity by a monotone feasible single-move descent, the AUC* of the MI-optimal assignment versus the
AUC-optimal one (objective misalignment), and whether the identity is already optimal. Classification per objective:
  OBJECTIVE_RESULT       the exhaustive constrained optimum does not improve on the identity (no useful move exists);
  SOLVED_EXACT           the solver reaches the exhaustive optimum of its own objective;
  OPTIMIZER_DEFECT       the solver stays above the exhaustive optimum of its own objective; subtype local_optimum (no feasible
                         improving single move from its answer: a neighbourhood limitation) or not_local_optimum / infeasible /
                         reported_J_mismatch (an implementation defect).
These are properties of invented finite laws only (no population claim).

V2 (this version): the solver now polishes with atom merges and takes `seeds`; the classification uses the campaign call
(SOLVER_RUNS 'campaign'), and the s*r* calls repeat the V1 calls. The solver's internal move / merge candidate values are also
checked against the independent table. V1 output (solver 6eacc900, single moves only) stays in TINY_EXACT.json.

CLI: python3 -m drf.tiny run   (writes results/<campaign>/roles/investigator_a/TINY_EXACT_V2.json and prints a summary)
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from drf import finite_law as FL  # noqa: E402
from drf import solver as SV  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'pcrl_direct_recovery_frontier_v1' / 'roles' / 'investigator_a' / 'TINY_EXACT_V2.json'
V1_OUT = OUT.with_name('TINY_EXACT.json')      # the first run (solver 6eacc900, single moves only); never overwritten here
METRICS = SV.METRICS
TOL = SV.TOL
CLIP = 1e-6
KO = 5
# V2: the updated solver (merge polish + seeds). 's*r*' repeat the V1 calls (now with the polish); 'campaign' is the call the
# campaign will make (coordinator, 2026-10-25: max_sweeps=30, restarts=2, perturb_moves=16, seeds=2) and drives the classification.
SOLVER_RUNS = (('s0r0', dict(max_sweeps=50, restarts=0, seed=0)), ('s1r0', dict(max_sweeps=50, restarts=0, seed=1)),
               ('s2r0', dict(max_sweeps=50, restarts=0, seed=2)), ('s0r3', dict(max_sweeps=50, restarts=3, seed=0)),
               ('campaign', dict(max_sweeps=30, restarts=2, perturb_moves=16, seeds=2, seed=0)))
PRIMARY = 'campaign'
DEFAULT = 's0r0'


# ------------------------------------------------------------------------------------------------ invented instances
def _start(n_cells, n_atoms):
    """identity start: fine cell (k, m) -> its coarse code k, cells listed in contiguous blocks of k."""
    return np.array([c * n_atoms // n_cells for c in range(n_cells)], dtype=np.int64)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def random_instance(seed, N, I, MI, J, MO, law='pi', p_sex=0.38, sex_task=0.6, cell_sex=1.0):
    """Invented people: SEX-dependent fine cells, teacher posteriors driven by the start atom plus a fine-cell shift (and SEX for
    occupation), labels drawn from the teachers, an informative propensity pi; law 'pi' (estimated) or 'sex' (empirical)."""
    rng = np.random.default_rng([int(seed), 20261025, 7])
    sI, sO = _start(I, MI), _start(J, MO)
    sex = (rng.random(N) < p_sex).astype(np.int64)
    sex[0], sex[1] = 0, 1
    pI = rng.dirichlet(np.ones(I) * 2.0, size=2)
    pO = rng.dirichlet(np.ones(J) * 2.0, size=2)
    pI = (1 - cell_sex) * pI.mean(0) + cell_sex * pI
    pO = (1 - cell_sex) * pO.mean(0) + cell_sex * pO
    inc_cell = np.array([rng.choice(I, p=pI[s]) for s in sex], dtype=np.int64)
    occ_cell = np.array([rng.choice(J, p=pO[s]) for s in sex], dtype=np.int64)
    inc_cell[:I] = np.arange(I)                                         # every fine cell occupied
    occ_cell[I:I + J] = np.arange(J)
    gI, hI = rng.normal(0, 1.0, MI), rng.normal(0, 0.5, I)
    q1 = _sig(gI[sI[inc_cell]] + hI[inc_cell] + 0.4 * rng.normal(size=N))
    qI = np.column_stack([1 - q1, q1])
    gO, hO, u = rng.normal(0, 1.0, (MO, KO)), rng.normal(0, 0.5, (J, KO)), rng.normal(0, 1.0, KO)
    z = gO[sO[occ_cell]] + hO[occ_cell] + sex_task * sex[:, None] * u + 0.4 * rng.normal(size=(N, KO))
    qO = np.exp(z - z.max(1, keepdims=True))
    qO /= qO.sum(1, keepdims=True)
    y_inc = (rng.random(N) < q1).astype(np.int64)
    y_occ = np.array([rng.choice(KO, p=p) for p in qO], dtype=np.int64)
    aI, aO = rng.normal(0, 0.4, I), rng.normal(0, 0.4, J)
    pi = np.clip(_sig(np.log(p_sex / (1 - p_sex)) + 1.2 * (2 * sex - 1) + aI[inc_cell] + aO[occ_cell] + 0.6 * rng.normal(size=N)),
                 0.01, 0.99)
    w1 = pi if law == 'pi' else sex.astype(np.float64)
    return dict(inc_cell=inc_cell, occ_cell=occ_cell, y_inc=y_inc, y_occ=y_occ, sex=sex, w1=w1, qI=qI, qO=qO,
                start_inc=sI, start_occ=sO, n_inc_atoms=MI, n_occ_atoms=MO,
                spec=dict(kind='random', seed=int(seed), N=int(N), I=int(I), MI=int(MI), J=int(J), MO=int(MO), law=law,
                          p_sex=p_sex, sex_task=sex_task, cell_sex=cell_sex))


def tied_tables_instance():
    """Deterministic instance whose two identity occupation atoms emit IDENTICAL vectors (dyadic posteriors, proportional counts:
    every float sum is exact), while their SEX rates differ: the atom-level law sees occupation leakage that no reader can see."""
    a = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625])
    b = np.array([0.0625, 0.0625, 0.125, 0.25, 0.5])
    occ_counts = [8, 4, 8, 4]                                           # cells 0, 1 -> atom 0; cells 2, 3 -> atom 1
    occ_vec = [a, b, a, b]
    women = [6, 1, 2, 3]                                                # SEX = 1 counts per occupation fine cell
    rows = []
    for c, (n, nw) in enumerate(zip(occ_counts, women)):
        for t in range(n):
            rows.append((c, int(t < nw)))
    occ_cell = np.array([r[0] for r in rows], dtype=np.int64)
    sex = np.array([r[1] for r in rows], dtype=np.int64)
    N = len(rows)
    inc_cell = (np.arange(N) % 2).astype(np.int64)                       # income: 2 fine cells, SEX-unrelated by design
    qI = np.where(inc_cell[:, None] == 0, np.array([[0.75, 0.25]]), np.array([[0.375, 0.625]]))
    qO = np.array([occ_vec[c] for c in occ_cell])
    y_occ = np.array([int(np.argmax(occ_vec[c])) if t % 3 else (int(np.argmax(occ_vec[c])) + 1) % KO
                      for t, c in enumerate(occ_cell)], dtype=np.int64)
    y_inc = ((np.arange(N) % 3) == 0).astype(np.int64)
    return dict(inc_cell=inc_cell, occ_cell=occ_cell, y_inc=y_inc, y_occ=y_occ, sex=sex, w1=sex.astype(np.float64), qI=qI, qO=qO,
                start_inc=_start(2, 2), start_occ=_start(4, 2), n_inc_atoms=2, n_occ_atoms=2,
                spec=dict(kind='tied_tables', N=int(N), I=2, MI=2, J=4, MO=2, law='sex'))


# ------------------------------------------------------------------------------------------------ independent recomputation
def _side(cells, A, M, q, y, sex, w1, guards):
    """Per-person B1 tables, emitted vectors, task means and the atom / observable laws of one recipient for mapping A."""
    atom = A[cells]
    n = np.bincount(atom, minlength=M).astype(np.float64)
    S = np.stack([np.bincount(atom, weights=q[:, k], minlength=M) for k in range(q.shape[1])], 1)
    T = np.full_like(S, 1.0 / q.shape[1])
    T[n > 0] = S[n > 0] / n[n > 0, None]
    T = np.clip(T, CLIP, 1.0)
    T = T / T.sum(1, keepdims=True)
    V = T[atom]
    idx = np.arange(len(y))
    oh = np.zeros_like(V)
    oh[idx, y] = 1.0
    nll = -np.log(V[idx, y])
    br = ((V - oh) ** 2).sum(1)
    acc = (V.argmax(1) == y).astype(np.float64)
    out = {'nll': nll.mean(), 'brier': br.mean(), 'acc': acc.mean()}
    if guards:
        out['brier_men'] = br[sex == 0].mean()
        out['brier_women'] = br[sex == 1].mean()
    _, _, _, obs = FL.merge_wires(V, w1, return_inverse=True)
    out['atom_ids'] = atom
    out['obs_ids'] = obs
    out['n_obs_wires'] = int(obs.max()) + 1
    out['n_atoms_used'] = int((n > 0).sum())
    out['table'] = T
    return out


def _law(ids, w1):
    _, m1, m0 = FL.merge_wires(ids, w1)
    return FL.auc_star(m1, m0), FL.plugin_mi(m1, m0, clip=1e-9)


def _assignments(n_cells, n_atoms):
    return [np.array(t, dtype=np.int64) for t in itertools.product(range(n_atoms), repeat=n_cells)]


def _index(A, n_atoms):
    v = 0
    for x in A:
        v = v * n_atoms + int(x)
    return v


def exhaustive(inst):
    """Every assignment: independent values (atom and observable laws, spends) and the solver's own evaluate() values."""
    d = inst
    MI, MO = d['n_inc_atoms'], d['n_occ_atoms']
    I, J = len(d['start_inc']), len(d['start_occ'])
    AI, AO = _assignments(I, MI), _assignments(J, MO)
    inc = [_side(d['inc_cell'], a, MI, d['qI'], d['y_inc'], d['sex'], d['w1'], False) for a in AI]
    occ = [_side(d['occ_cell'], a, MO, d['qO'], d['y_occ'], d['sex'], d['w1'], True) for a in AO]
    for s in inc + occ:
        s['law_atom'] = _law(s['atom_ids'], d['w1'])
        s['law_obs'] = _law(s['obs_ids'], d['w1'])
    i0, o0 = _index(d['start_inc'], MI), _index(d['start_occ'], MO)
    base = inc[i0], occ[o0]
    P = SV.Problem(d['inc_cell'], d['occ_cell'], d['y_inc'], d['y_occ'], d['sex'], d['w1'], d['qI'], d['qO'], d['start_inc'],
                   d['start_occ'], MI, MO)
    nA = len(AI) * len(AO)
    cols = ['J_auc', 'J_mi', 'J_auc_obs', 'J_mi_obs', 'auc_pair', 'auc_inc', 'auc_occ', 'auc_pair_obs', 'auc_inc_obs',
            'auc_occ_obs', 'mi_pair', 'mi_inc', 'mi_occ', 'sv_J_auc', 'sv_J_mi']
    V = {c: np.zeros(nA) for c in cols}
    SP = np.zeros((nA, 8))
    SPsv = np.zeros((nA, 8))
    obs_differs = 0
    for ia, si in enumerate(inc):
        for io, so in enumerate(occ):
            k = ia * len(AO) + io
            ap, mp = _law(FL.pair_ids(si['atom_ids'], so['atom_ids']), d['w1'])
            apo, mpo = _law(FL.pair_ids(si['obs_ids'], so['obs_ids']), d['w1'])
            (ai_, mi_), (ao_, mo_) = si['law_atom'], so['law_atom']
            (aio, mio), (aoo, moo) = si['law_obs'], so['law_obs']
            V['auc_pair'][k], V['auc_inc'][k], V['auc_occ'][k] = ap, ai_, ao_
            V['auc_pair_obs'][k], V['auc_inc_obs'][k], V['auc_occ_obs'][k] = apo, aio, aoo
            V['mi_pair'][k], V['mi_inc'][k], V['mi_occ'][k] = mp, mi_, mo_
            V['J_auc'][k] = ap + ai_ + ao_ - 1.5
            V['J_mi'][k] = mp + mi_ + mo_
            V['J_auc_obs'][k] = apo + aio + aoo - 1.5
            V['J_mi_obs'][k] = mpo + mio + moo
            obs_differs += int(abs(V['J_auc_obs'][k] - V['J_auc'][k]) > 1e-12)
            SP[k] = [si['nll'] - base[0]['nll'], so['nll'] - base[1]['nll'], si['brier'] - base[0]['brier'],
                     so['brier'] - base[1]['brier'], base[0]['acc'] - si['acc'], base[1]['acc'] - so['acc'],
                     so['brier_men'] - base[1]['brier_men'], so['brier_women'] - base[1]['brier_women']]
            e = SV.evaluate(P, AI[ia], AO[io])
            V['sv_J_auc'][k], V['sv_J_mi'][k] = e['J_auc'], e['J_mi']
            SPsv[k] = [e['spend'][m] for m in METRICS]
    arith = {'max_abs_diff_J_auc': float(np.abs(V['sv_J_auc'] - V['J_auc']).max()),
             'max_abs_diff_J_mi': float(np.abs(V['sv_J_mi'] - V['J_mi']).max()),
             'max_abs_diff_spend': float(np.abs(SPsv - SP).max()),
             'assignments_where_observable_J_auc_differs_from_atom_J_auc': int(obs_differs)}
    return dict(P=P, AI=AI, AO=AO, V=V, SP=SP, i0=i0, o0=o0, k0=i0 * len(AO) + o0, nAO=len(AO), I=I, J=J, MI=MI, MO=MO,
                arith=arith, inc=inc, occ=occ)


# ------------------------------------------------------------------------------------------------ analysis
def _neighbours(E, k):
    """Assignment indices one fine-cell move away from k (occupied cells only: tiny instances occupy every cell)."""
    ia, io = divmod(k, E['nAO'])
    out = []
    A = E['AI'][ia]
    for c in range(E['I']):
        for t in range(E['MI']):
            if t != A[c]:
                B = A.copy(); B[c] = t
                out.append(_index(B, E['MI']) * E['nAO'] + io)
    A = E['AO'][io]
    for c in range(E['J']):
        for t in range(E['MO']):
            if t != A[c]:
                B = A.copy(); B[c] = t
                out.append(ia * E['nAO'] + _index(B, E['MO']))
    return out


def _merge_neighbours(E, k):
    """Assignment indices one atom MERGE away from k: all cells of one used atom move to another used atom (either recipient)."""
    ia, io = divmod(k, E['nAO'])
    out = []
    for side, A, M in (('inc', E['AI'][ia], E['MI']), ('occ', E['AO'][io], E['MO'])):
        used = sorted(set(A.tolist()))
        for x in range(len(used)):
            for y in range(x + 1, len(used)):
                B = A.copy()
                B[A == used[y]] = used[x]
                out.append(_index(B, M) * E['nAO'] + io if side == 'inc' else ia * E['nAO'] + _index(B, M))
    return out


def _run(r, label):
    return next(s for s in r['solver'] if s['label'] == label)


def candidate_check(E, n_states, seed):
    """The solver's INTERNAL candidate values (drf.solver._move_candidates and _merge_candidates: exact J of both objectives and the
    8 spends) against this module's independent per-person table at the resulting assignments, from random states."""
    rng = np.random.default_rng([int(seed), 99])
    P, V, SP, nAO = E['P'], E['V'], E['SP'], E['nAO']
    w = {'move_J': 0.0, 'move_spend': 0.0, 'merge_J': 0.0, 'merge_spend': 0.0, 'moves_checked': 0, 'merges_checked': 0}
    for _ in range(int(n_states)):
        k = int(rng.integers(len(E['AI']) * nAO))
        ia, io = divmod(k, nAO)
        A_inc, A_occ = E['AI'][ia], E['AO'][io]
        S = SV._State(P, A_inc, A_occ)
        for side, A, M in (('inc', A_inc, E['MI']), ('occ', A_occ, E['MO'])):
            def at(B):
                return _index(B, M) * nAO + io if side == 'inc' else ia * nAO + _index(B, M)
            batches = []
            for pr, new_tot, Mb in SV._merge_candidates(S, side):
                ks = []
                for t, b in pr:
                    B = A.copy()
                    B[A == b] = t
                    ks.append(at(B))
                batches.append(('merge', ks, new_tot, Mb))
            for c in range(len(A)):
                targets, new_tot, Mb = SV._move_candidates(S, side, c)
                ks = []
                for t in targets:
                    B = A.copy()
                    B[c] = t
                    ks.append(at(B))
                batches.append(('move', ks, new_tot, Mb))
            for kind, ks, new_tot, Mb in batches:
                ks = np.array(ks)
                for obj, col in (('auc', 'J_auc'), ('mi', 'J_mi')):
                    w[f'{kind}_J'] = max(w[f'{kind}_J'], float(np.abs(SV._objective(Mb, obj) - V[col][ks]).max()))
                w[f'{kind}_spend'] = max(w[f'{kind}_spend'], float(np.abs(SV._spend(P, new_tot) - SP[ks]).max()))
                w[f'{kind}s_checked'] += len(ks)
    return w


def _monotone_reachable(E, Jv, feas):
    """Assignments reachable from the identity by feasible single moves that each lower J by more than TOL."""
    seen, stack = {E['k0']}, [E['k0']]
    while stack:
        k = stack.pop()
        for q in _neighbours(E, k):
            if q not in seen and feas[q] and Jv[q] < Jv[k] - TOL:
                seen.add(q)
                stack.append(q)
    return seen


def _assignment_dict(E, k):
    ia, io = divmod(int(k), E['nAO'])
    V = E['V']
    return {'A_inc': E['AI'][ia].tolist(), 'A_occ': E['AO'][io].tolist(),
            **{c: float(V[c][k]) for c in ('J_auc', 'J_mi', 'J_auc_obs', 'auc_pair', 'auc_inc', 'auc_occ', 'auc_pair_obs',
                                           'auc_inc_obs', 'auc_occ_obs')},
            'spend': dict(zip(METRICS, [float(x) for x in E['SP'][k]]))}


def budget_levels(E):
    SP = E['SP']
    lv = {'unconstrained': {m: 1e9 for m in METRICS}}
    for name, q in (('q60', 0.60), ('q25', 0.25)):
        b = {}
        for j, m in enumerate(METRICS):
            pos = SP[:, j][SP[:, j] > 1e-15]
            b[m] = float(np.quantile(pos, q)) if pos.size else 0.0
        lv[name] = b
    lv['zero'] = {m: 0.0 for m in METRICS}
    return lv


def analyse_level(E, name, budgets):
    V, SP = E['V'], E['SP']
    lim = np.array([budgets[m] for m in METRICS])
    feas = (SP <= lim + TOL).all(1)
    out = {'level': name, 'budgets': budgets, 'n_feasible': int(feas.sum()), 'n_assignments': int(feas.size),
           'identity_feasible': bool(feas[E['k0']])}
    res = {}
    for obj, col in (('auc', 'J_auc'), ('mi', 'J_mi')):
        Jv = V[col]
        Jf = np.where(feas, Jv, np.inf)
        jmin = float(Jf.min())
        arg = np.flatnonzero(Jf <= jmin + 1e-12)
        unc_arg = np.flatnonzero(Jv <= Jv.min() + 1e-12)
        reach = _monotone_reachable(E, Jv, feas)
        r = {'J_identity': float(Jv[E['k0']]), 'J_min': jmin, 'improvement': float(Jv[E['k0']] - jmin),
             'n_argmin': int(arg.size), 'argmin_example': _assignment_dict(E, arg[0]),
             'identity_optimal': bool(jmin >= Jv[E['k0']] - 1e-12),
             'binding': bool(not feas[unc_arg].any()),
             'optimum_reachable_by_monotone_single_moves': bool(any(int(a) in reach for a in arg)),
             'monotone_reachable_states': len(reach), 'solver': []}
        for label, kw in SOLVER_RUNS:
            A_inc, A_occ, tr = SV.solve(E['P'], obj, budgets, **kw)
            k = _index(A_inc, E['MI']) * E['nAO'] + _index(A_occ, E['MO'])
            nb = _neighbours(E, k)
            imp = [q for q in nb if feas[q] and Jv[q] < Jv[k] - TOL]
            impm = [q for q in _merge_neighbours(E, k) if feas[q] and Jv[q] < Jv[k] - TOL]
            two = {q2 for q in nb for q2 in _neighbours(E, q)} - {k}
            imp2 = [q for q in two if feas[q] and Jv[q] < Jv[k] - TOL]
            converged = all(run.get('converged', False) for run in tr['runs'])
            it = tr['internal_iterations']
            r['solver'].append({'label': label, 'call': kw, 'J': float(Jv[k]), 'gap': float(Jv[k] - jmin),
                                'reported_J': float(tr['best_J']), 'reported_J_consistent': bool(abs(tr['best_J'] - Jv[k]) < 1e-9),
                                'feasible': bool(feas[k]), 'converged': bool(converged),
                                'improving_feasible_single_moves': len(imp), 'improving_feasible_merges': len(impm),
                                'improving_feasible_two_moves': len(imp2),
                                'best_two_move_J': float(min([Jv[q] for q in imp2], default=Jv[k])),
                                'J_auc_obs': float(V['J_auc_obs'][k]), 'assignment': {'A_inc': A_inc.tolist(), 'A_occ': A_occ.tolist()},
                                'moves': int(it['moves']), 'merges': int(it.get('merges', 0)),
                                'local_searches': len(tr['runs'])})
        defects = []
        for s in r['solver']:
            if not s['feasible']:
                defects.append('infeasible')
            if not s['reported_J_consistent']:
                defects.append('reported_J_mismatch')
            if (s['improving_feasible_single_moves'] or s['improving_feasible_merges']) and s['converged']:
                defects.append('not_local_optimum')
        gaps = {s['label']: s['gap'] for s in r['solver']}
        if r['identity_optimal']:
            cls = 'OBJECTIVE_RESULT'
        elif gaps[PRIMARY] <= 1e-12:
            cls = 'SOLVED_EXACT'
        else:
            cls = 'OPTIMIZER_DEFECT'
        r['classification'] = cls
        r['classified_on'] = PRIMARY
        r['gap_by_call'] = gaps
        r['shortfall_runs'] = int(sum(g > 1e-12 for g in gaps.values()))
        r['implementation_defects'] = sorted(set(defects))
        r['shortfall_subtype'] = (None if cls != 'OPTIMIZER_DEFECT' else
                                  ('implementation_defect' if defects else 'local_optimum'))
        res[obj] = r
    # objective misalignment: J_auc at the MI-optimal assignments versus the AUC-optimal value (same feasible set)
    Jmi = np.where(feas, V['J_mi'], np.inf)
    mi_arg = np.flatnonzero(Jmi <= Jmi.min() + 1e-12)
    auc_arg = np.flatnonzero(np.where(feas, V['J_auc'], np.inf) <= res['auc']['J_min'] + 1e-12)
    ja = V['J_auc'][mi_arg]
    s_mi = _run(res['mi'], PRIMARY)['assignment']
    k_mi = _index(s_mi['A_inc'], E['MI']) * E['nAO'] + _index(s_mi['A_occ'], E['MO'])
    out['misalignment'] = {'J_auc_min': res['auc']['J_min'], 'J_auc_at_mi_argmin_min': float(ja.min()),
                           'J_auc_at_mi_argmin_max': float(ja.max()),
                           'excess_J_auc_of_mi_optimum': float(ja.min() - res['auc']['J_min']),
                           'J_mi_min': res['mi']['J_min'], 'J_mi_at_auc_argmin_min': float(V['J_mi'][auc_arg].min()),
                           'witness': bool(ja.min() > res['auc']['J_min'] + 1e-9),
                           'mi_argmin_example': _assignment_dict(E, mi_arg[0]),
                           f'solver_mi_{PRIMARY}_J_auc': float(V['J_auc'][k_mi]),
                           f'solver_auc_{PRIMARY}_J_auc': _run(res['auc'], PRIMARY)['J']}
    # observable objective (identical emitted vectors merged): its own optimum versus the atom-level optimum
    Jo = np.where(feas, V['J_auc_obs'], np.inf)
    ka = int(np.flatnonzero(np.where(feas, V['J_auc'], np.inf) <= res['auc']['J_min'] + 1e-12)[0])
    out['observable'] = {'J_auc_obs_min': float(Jo.min()), 'J_auc_obs_identity': float(V['J_auc_obs'][E['k0']]),
                         'J_auc_obs_at_atom_optimum': float(V['J_auc_obs'][ka]),
                         f'J_auc_obs_at_solver_auc_{PRIMARY}': _run(res['auc'], PRIMARY)['J_auc_obs'],
                         'atom_optimum_suboptimal_observably': bool(V['J_auc_obs'][ka] > Jo.min() + 1e-9)}
    out['objectives'] = res
    return out


def analyse(inst, name):
    t0 = time.process_time()
    E = exhaustive(inst)
    E['arith']['solver_internal_candidates'] = candidate_check(E, 20, inst['spec'].get('seed', 0))
    levels = [analyse_level(E, nm, b) for nm, b in budget_levels(E).items()]
    sp = inst['spec']
    return {'name': name, 'spec': sp, 'describe': E['P'].describe(), 'n_assignments': int(len(E['AI']) * len(E['AO'])),
            'people_women': int(inst['sex'].sum()), 'arithmetic_check': E['arith'],
            'identity': _assignment_dict(E, E['k0']), 'levels': levels,
            '_cpu_s': time.process_time() - t0}


def _instance_payload(inst):
    """the full invented data of a (small) instance, so it can be rebuilt without this module's generator."""
    keys = ('inc_cell', 'occ_cell', 'y_inc', 'y_occ', 'sex', 'w1', 'qI', 'qO', 'start_inc', 'start_occ')
    return {**{k: np.asarray(inst[k]).tolist() for k in keys},
            'n_inc_atoms': int(inst['n_inc_atoms']), 'n_occ_atoms': int(inst['n_occ_atoms'])}


NAMED = (
    ('A_pi_N240_I3J6', dict(seed=11, N=240, I=3, MI=2, J=6, MO=3, law='pi')),
    ('B_label_N120_I3J6', dict(seed=12, N=120, I=3, MI=2, J=6, MO=3, law='sex')),
    ('C_label_N40_I2J4', dict(seed=13, N=40, I=2, MI=2, J=4, MO=2, law='sex')),
    ('D_pi_N300_I3J5_sextask', dict(seed=14, N=300, I=3, MI=2, J=5, MO=3, law='pi', sex_task=1.5)),
    ('F_pi_N60_I2J3', dict(seed=15, N=60, I=2, MI=2, J=3, MO=3, law='pi')),
    ('G_label_N80_I3J4', dict(seed=16, N=80, I=3, MI=2, J=4, MO=2, law='sex', p_sex=0.3)),
)


def run(scan=120, write=True):
    t0 = time.process_time()
    out = {'module': 'drf.tiny', 'campaign': 'pcrl_direct_recovery_frontier_v1', 'synthetic_only': True,
           'solver_sha256': hashlib.sha256((ROOT / 'drf' / 'solver.py').read_bytes()).hexdigest(),
           'finite_law_sha256': hashlib.sha256((ROOT / 'drf' / 'finite_law.py').read_bytes()).hexdigest(),
           'version': 'V2 (updated solver: merge polish + seeds); V1 = TINY_EXACT.json (solver 6eacc900, single moves only)',
           'conventions': {'tol': TOL, 'solver_calls': {lab: kw for lab, kw in SOLVER_RUNS}, 'classified_on': PRIMARY,
                           'default_call': DEFAULT,
                           'budget_levels': 'unconstrained; q60 / q25 = per-metric quantile of the positive spends over all '
                                            'assignments; zero',
                           'atom_law': 'wires = atom ids (the solver objective)',
                           'observable_law': 'wires = emitted float64 vectors, identical vectors merged (drf.finite_law)',
                           'mi': 'plug-in MI with r clipped to [1e-9, 1 - 1e-9] (the solver / PRC definition)'},
           'instances': []}
    cpu = {}
    for name, kw in NAMED:
        inst = random_instance(**kw)
        a = analyse(inst, name)
        cpu[name] = a.pop('_cpu_s')
        out['instances'].append(a)
    tt = tied_tables_instance()
    a = analyse(tt, 'E_tied_tables_N24_I2J4')
    cpu['E_tied_tables'] = a.pop('_cpu_s')
    a['payload'] = _instance_payload(tt)
    out['instances'].append(a)
    # random scan of small instances: frequency of each classification, smallest reproducer of any optimizer shortfall
    rng = np.random.default_rng(2026102502)
    tally, impl, cases, reps = {}, [], [], {'default_call': None, 'persistent': None, 'default_no_move': None,
                                            'primary_call': None}
    mis = []
    arith_scan = {'max_abs_diff_J_auc': 0.0, 'max_abs_diff_J_mi': 0.0, 'max_abs_diff_spend': 0.0, 'move_J': 0.0,
                  'move_spend': 0.0, 'merge_J': 0.0, 'merge_spend': 0.0, 'moves_checked': 0, 'merges_checked': 0}
    for t in range(int(scan)):
        kw = dict(seed=1000 + t, N=int(rng.integers(40, 121)), I=int(rng.integers(2, 4)), MI=2, J=int(rng.integers(3, 5)),
                  MO=int(rng.integers(2, 4)), law=('pi', 'sex')[t % 2])
        inst = random_instance(**kw)
        E = exhaustive(inst)
        cc = candidate_check(E, 4, kw['seed'])
        for key in arith_scan:
            src = E['arith'] if key.startswith('max_abs') else cc
            arith_scan[key] = arith_scan[key] + src[key] if key.endswith('checked') else max(arith_scan[key], src[key])
        for nm, b in budget_levels(E).items():
            if nm == 'zero':
                continue
            L = analyse_level(E, nm, b)
            for obj in ('auc', 'mi'):
                r = L['objectives'][obj]
                key = f"{obj}:{r['classification']}" + (f":{r['shortfall_subtype']}" if r['shortfall_subtype'] else '')
                tally[key] = tally.get(key, 0) + 1
                if r['implementation_defects']:
                    impl.append({'spec': inst['spec'], 'level': nm, 'objective': obj, 'defects': r['implementation_defects']})
                case = {'seed': kw['seed'], 'level': nm, 'objective': obj, 'improvable': not r['identity_optimal'],
                        'gaps': {s['label']: s['gap'] for s in r['solver']},
                        'rel': {s['label']: (s['gap'] / r['improvement']) if r['improvement'] > 1e-12 else 0.0
                                for s in r['solver']},
                        'no_move': {s['label']: bool(s['moves'] == 0 and s['merges'] == 0) for s in r['solver']},
                        'two_move_closes': {s['label']: bool(s['best_two_move_J'] <= r['J_min'] + 1e-12) for s in r['solver']},
                        'reachable': r['optimum_reachable_by_monotone_single_moves'], 'binding': r['binding']}
                cases.append(case)
                size = (kw['N'], len(E['AI']) * len(E['AO']))
                g = case['gaps']
                for which, hit in (('default_call', g[DEFAULT] > 1e-12),
                                   ('persistent', min(g.values()) > 1e-12),
                                   ('default_no_move', g[DEFAULT] > 1e-12 and case['no_move'][DEFAULT]),
                                   ('primary_call', g[PRIMARY] > 1e-12)):
                    if hit and (reps[which] is None or size < reps[which][0]):
                        reps[which] = (size, inst, nm, obj, L)
            tally[f"misalignment_witness:{L['misalignment']['witness']}"] = \
                tally.get(f"misalignment_witness:{L['misalignment']['witness']}", 0) + 1
            mis.append(L['misalignment']['excess_J_auc_of_mi_optimum'])
    out_reps = {}
    for which, rep in reps.items():
        if rep is None:
            out_reps[which] = None
            continue
        _, inst, nm, obj, L = rep
        r = L['objectives'][obj]
        out_reps[which] = {'spec': inst['spec'], 'level': nm, 'objective': obj, 'budgets': L['budgets'],
                           'J_identity': r['J_identity'], 'J_min': r['J_min'], 'argmin_example': r['argmin_example'],
                           'solver_runs': r['solver'],
                           'optimum_reachable_by_monotone_single_moves': r['optimum_reachable_by_monotone_single_moves'],
                           'payload': _instance_payload(inst)}
    imp = [c for c in cases if c['improvable']]
    stats = {}
    for obj in ('auc', 'mi'):
        ci = [c for c in imp if c['objective'] == obj]
        stats[obj] = {'improvable_cases': len(ci)}
        for lab, _ in SOLVER_RUNS:
            sd = [c for c in ci if c['gaps'][lab] > 1e-12]
            rel = [c['rel'][lab] for c in sd]
            stats[obj][lab] = {'shortfalls': len(sd),
                               'median_relative_gap_of_shortfalls': float(np.median(rel)) if rel else 0.0,
                               'max_relative_gap_of_shortfalls': float(max(rel, default=0.0)),
                               'no_move_or_merge_although_improvable': sum(c['no_move'][lab] for c in sd),
                               'shortfalls_closed_by_one_two_cell_move': sum(c['two_move_closes'][lab] for c in sd),
                               'shortfalls_with_optimum_reachable_by_monotone_single_moves': sum(c['reachable'] for c in sd)}
        stats[obj]['shortfall_best_of_all_calls'] = sum(min(c['gaps'].values()) > 1e-12 for c in ci)
    # the V1 minimal reproducers (same generator, seeds and budget levels): solved now?
    v1 = json.loads(V1_OUT.read_text()) if V1_OUT.exists() else None
    repro = []
    if v1 is not None:
        for which in ('smallest_default_call_shortfall', 'smallest_persistent_shortfall', 'smallest_default_no_move_shortfall'):
            rep = v1['random_scan'].get(which)
            if not rep:
                continue
            c = next((c for c in cases if c['seed'] == rep['spec']['seed'] and c['level'] == rep['level']
                      and c['objective'] == rep['objective']), None)
            repro.append({'v1_reproducer': which, 'seed': rep['spec']['seed'], 'level': rep['level'],
                          'objective': rep['objective'], 'v1_J_min': rep['J_min'],
                          'v1_gaps': [s['gap'] for s in rep['solver_runs']],
                          'v2_gaps': None if c is None else c['gaps'],
                          'v2_solved_by_primary': None if c is None else bool(c['gaps'][PRIMARY] <= 1e-12),
                          'v2_solved_by_default': None if c is None else bool(c['gaps'][DEFAULT] <= 1e-12)})
    out['v1_reproducers'] = repro
    out['random_scan'] = {'instances': int(scan), 'levels_per_instance': ['unconstrained', 'q60', 'q25'],
                          'arithmetic_check': arith_scan, 'smallest_primary_call_shortfall': None,
                          'tally': dict(sorted(tally.items())), 'implementation_defects': impl, 'shortfall_stats': stats,
                          'misalignment_excess_J_auc': {'cases': len(mis), 'witnesses': int(sum(m > 1e-9 for m in mis)),
                                                        'max': float(max(mis, default=0.0)),
                                                        'median_of_witnesses': (float(np.median([m for m in mis if m > 1e-9]))
                                                                                if any(m > 1e-9 for m in mis) else 0.0)},
                          'smallest_default_call_shortfall': out_reps['default_call'],
                          'smallest_persistent_shortfall': out_reps['persistent'],
                          'smallest_default_no_move_shortfall': out_reps['default_no_move']}
    out['random_scan']['smallest_primary_call_shortfall'] = out_reps['primary_call']
    out['v1_named_comparison'] = _v1_named(out, v1)
    out['summary'] = _summary(out)
    out['findings'] = _findings(out)
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1, default=float) + '\n')
    return out, cpu, time.process_time() - t0


def _summary(out):
    rows = []
    for inst in out['instances']:
        for L in inst['levels']:
            row = {'instance': inst['name'], 'level': L['level'], 'n_feasible': L['n_feasible']}
            for obj in ('auc', 'mi'):
                r = L['objectives'][obj]
                row[obj] = {'class': r['classification'], 'subtype': r['shortfall_subtype'],
                            'J_identity': round(r['J_identity'], 6), 'J_min': round(r['J_min'], 6),
                            'gap_primary': round(r['gap_by_call'][PRIMARY], 6), 'gap_default': round(r['gap_by_call'][DEFAULT], 6),
                            'max_gap_all_calls': round(max(r['gap_by_call'].values()), 6),
                            'shortfall_runs': r['shortfall_runs'], 'binding': r['binding'],
                            'reachable_monotone': r['optimum_reachable_by_monotone_single_moves'],
                            'impl_defects': r['implementation_defects']}
            row['misalignment_excess_J_auc'] = round(L['misalignment']['excess_J_auc_of_mi_optimum'], 6)
            row['observable_atom_optimum_suboptimal'] = L['observable']['atom_optimum_suboptimal_observably']
            rows.append(row)
    return rows


def _v1_named(out, v1):
    """V1 (TINY_EXACT.json, solver 6eacc900) versus V2 gaps on the named instances, per level and objective."""
    if v1 is None:
        return None
    old = {(i['name'], L['level'], o): L['objectives'][o] for i in v1['instances'] for L in i['levels'] for o in ('auc', 'mi')}
    rows = []
    for i in out['instances']:
        for L in i['levels']:
            for o in ('auc', 'mi'):
                r = L['objectives'][o]
                p = old.get((i['name'], L['level'], o))
                rows.append({'instance': i['name'], 'level': L['level'], 'objective': o, 'J_min': r['J_min'],
                             'v1_J_min': None if p is None else p['J_min'],
                             'v1_gaps_s0r0_s1r0_s2r0_s0r3': None if p is None else [s['gap'] for s in p['solver']],
                             'v1_class_any_of_4': None if p is None else p['classification'],
                             'v2_gaps': r['gap_by_call'], 'v2_class_on_primary': r['classification']})
    return rows


def _findings(out):
    """Machine-readable conclusions drawn only from this file's numbers (invented instances; no population claim)."""
    named = out['instances']
    arith = max(max(i['arithmetic_check']['max_abs_diff_J_auc'], i['arithmetic_check']['max_abs_diff_J_mi'],
                    i['arithmetic_check']['max_abs_diff_spend']) for i in named)
    cand = max(max(i['arithmetic_check']['solver_internal_candidates'][k] for k in ('move_J', 'move_spend', 'merge_J',
                                                                                     'merge_spend')) for i in named)
    sc = out['random_scan']
    sa = sc['arithmetic_check']
    impl = sum(1 for i in named for L in i['levels'] for o in ('auc', 'mi') if L['objectives'][o]['implementation_defects'])
    tied = [i['name'] for i in named if i['arithmetic_check']['assignments_where_observable_J_auc_differs_from_atom_J_auc'] > 0]
    named_short = {o: sum(1 for i in named for L in i['levels'] if L['objectives'][o]['classification'] == 'OPTIMIZER_DEFECT')
                   for o in ('auc', 'mi')}
    return {
        'solver_evaluate_matches_independent_recomputation_max_abs_diff_named': arith,
        'solver_internal_move_and_merge_candidates_max_abs_diff_named': cand,
        'scan_arithmetic_max_abs_diff': max(sa[k] for k in ('max_abs_diff_J_auc', 'max_abs_diff_J_mi', 'max_abs_diff_spend',
                                                            'move_J', 'move_spend', 'merge_J', 'merge_spend')),
        'implementation_defects_named': impl, 'implementation_defects_scan': len(sc['implementation_defects']),
        'every_answer_feasible_consistent_and_a_single_move_and_merge_local_optimum':
            bool(impl == 0 and not sc['implementation_defects']),
        f'named_level_cases_with_{PRIMARY}_shortfall': named_short,
        'scan_shortfall_stats': sc['shortfall_stats'],
        'v1_reproducers': out['v1_reproducers'],
        'scan_misalignment': sc['misalignment_excess_J_auc'],
        'atom_level_objective_differs_from_observable_in': tied,
        'notes': [
            f'Classification uses the {PRIMARY} call {dict(SOLVER_RUNS)[PRIMARY]}; s*r* columns repeat the V1 calls on the '
            'updated solver (which now always includes the merge polish).',
            'A shortfall is a feasible local optimum of the solver neighbourhood (single moves + atom merges), not an arithmetic '
            'error, unless implementation_defects says otherwise.',
            'The solver scores atom ids; when two atoms emit identical vectors the observable AUC* is lower than the atom-level '
            'value (E_tied_tables); the final release must be re-scored on merged emitted wires (drf.finite_law).',
            'MI-optimal assignments can carry a higher J_auc than the AUC-optimal assignment under the same constraints.'],
    }


def _print_summary(out, cpu, total):
    print(f"classification on {PRIMARY} = {dict(SOLVER_RUNS)[PRIMARY]}; default = {DEFAULT}")
    print(f"{'instance':24s} {'level':13s} {'feas':>5s} | {'AUC: class (primary)':26s} {'J_id':>7s} {'J_min':>7s} {'g_def':>7s} "
          f"{'g_prim':>7s} | {'MI: class (primary)':26s} {'g_def':>8s} {'g_prim':>8s} | {'mis_exc':>7s}")
    for r in out['summary']:
        a, m = r['auc'], r['mi']
        ca = a['class'] + ('/' + a['subtype'] if a['subtype'] else '')
        cm = m['class'] + ('/' + m['subtype'] if m['subtype'] else '')
        print(f"{r['instance'][:24]:24s} {r['level']:13s} {r['n_feasible']:5d} | {ca:26s} {a['J_identity']:7.4f} {a['J_min']:7.4f} "
              f"{a['gap_default']:7.4f} {a['gap_primary']:7.4f} | {cm:26s} {m['gap_default']:8.5f} {m['gap_primary']:8.5f} | "
              f"{r['misalignment_excess_J_auc']:7.4f}")
    print('arithmetic checks (named):', json.dumps({i['name']: i['arithmetic_check'] for i in out['instances']}))
    print('arithmetic checks (scan):', json.dumps(out['random_scan']['arithmetic_check']))
    print('random scan tally:', out['random_scan']['tally'])
    print('random scan shortfall stats:', json.dumps(out['random_scan']['shortfall_stats']))
    print('random scan misalignment:', json.dumps(out['random_scan']['misalignment_excess_J_auc']))
    print('implementation defects in scan:', len(out['random_scan']['implementation_defects']))
    print('V1 reproducers:', json.dumps(out['v1_reproducers']))
    for which in ('smallest_primary_call_shortfall', 'smallest_default_call_shortfall', 'smallest_persistent_shortfall',
                  'smallest_default_no_move_shortfall'):
        rep = out['random_scan'][which]
        if rep:
            print(which, {k: rep[k] for k in ('spec', 'level', 'objective', 'J_identity', 'J_min',
                                              'optimum_reachable_by_monotone_single_moves')},
                  'gaps', {s['label']: round(s['gap'], 5) for s in rep['solver_runs']})
        else:
            print(which, None)
    print('cpu_s per named instance:', {k: round(v, 2) for k, v in cpu.items()}, 'total cpu_s:', round(total, 1))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        res, cpu, total = run()
        _print_summary(res, cpu, total)
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
