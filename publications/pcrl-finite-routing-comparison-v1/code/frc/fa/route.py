"""Family FA of the PCRL finite-routing comparison (`pcrl_finite_routing_comparison_v1`; EXPLORATORY DEVELOPMENT on previously used
public-use U.S. Census Bureau ACS 2019 1-Year PUMS Florida person records; not confirmation).

The release maps six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding X) to an income vector (n, 2) and an
occupation vector (n, 5); each recipient receives only its own float64 vector and its argmax decision. SEX (0 = men, 1 = women; the
dataset's recorded sex column) is a training label on MODEL_FIT only in the protected stage, never a runtime input or output.

Source: the propensity-routed finite cells of card FA_PRC_v1 (`nrm.fa.prc`, imported UNCHANGED: teachers, compact coding, routing
rule, KL k-means, damped rule-consistent Lloyd chains, protected / HQ / SHUF / ORD variants, nested household folds, spend and the
budget selection, LOOKUP checks). New here, and nothing else:

1. Probability constructions on a FIXED partition. The routing book (codewords) and the teachers decide the cell of every person;
   the construction only decides the vector each cell emits:
     A   label-refit decoders of the source: (MODEL_FIT label counts of the out-of-fold-routed people + 20 x codeword) / (n + 20);
     B1  the mean of the OUT-OF-FOLD teacher predictions of the people whose out-of-fold route is the cell (the same people A counts),
         clipped at 1e-6 and renormalized (= one Lloyd codeword update of the final partition without re-routing); an unoccupied cell
         emits its stored prototype (counted);
     B0  the stored routing prototype (codeword) itself.
   Decisions = argmax of the emitted vector. Changing the table never changes the partition (asserted wherever both exist).
2. The nested fixed-routing comparison (stage `nest`): the NRM1 capacity-gate estimator (5 outer household folds NRM_FA_ncv_S<v>,
   the whole pipeline refitted on each outer-training part at the full-pool T*) emitting CONT (the continuous teachers), A, B0 and B1
   on identical held-out cells; it reproduces the NRM1 GATE records for A and the saved decoders from labels.
3. Deployable task-only releases (stage `twb`): the saved NRM1 twin's teachers and routing book with B1 tables computed from the
   cached full-pool out-of-fold teacher predictions (label-free); the partition identity with the saved A release is checked on all
   six roles. The prospective capacity (32, 256) builds its own book from the same cache (label-free).
4. The protected stage with the construction selected by the committed qualification decision (stages `paths`, `cfg`, `check`,
   `abl`): FA_PRC_v1 unchanged except that every configuration's tables are the selected construction recomputed on its own
   partition.

Commands:  python3 -m frc.fa.route selftest      (invented arrays only; no real data, no labels)
Real-data entry points run only through the campaign queue after the registration receipt and the committed card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')                    # one numerical thread (campaign rule); set before numpy / sklearn load

import numpy as np

from nrm.fa import prc as P

FAMILY = 'FA'
SCHEMA = 'FRC-FA-v1'
CONSTRUCTIONS = ('A', 'B0', 'B1')
ALPHA = float(P.ROUTE['alpha_dec'])                   # 20 pseudo-counts (source A)
CAPACITIES = {'C64': (16, 64), 'C128': (24, 128), 'C256': (32, 256)}
SOURCE_TWIN = {'C64': 'PRC_TWIN', 'C128': 'PRC_TWIN_C128'}           # the saved NRM1 twins (construction A; read-only)
# ORD ladders (coarser codebooks than the protection capacity; the card's ladder for 16 x 64, the registered capacities prepended)
ORD_LADDERS = {'C64': tuple(P.ORD_LADDER), 'C128': ((16, 64),) + tuple(P.ORD_LADDER),
               'C256': ((24, 128), (16, 64)) + tuple(P.ORD_LADDER)}
OUTER_TAG = 'NRM_FA_ncv_S{v}'                         # the NRM1 gate's outer household folds (unchanged)
GATE_LIMITS = dict(P.CAPACITY_GATE)                   # the NRM1 internal gate (reported only; never this study's eligibility rule)

CONFIGS = {
    'TWB_C64': {'kind': 'twin', 'capacity_id': 'C64', 'construction': 'B1',
                'what': 'task-only twin of the saved NRM1 PRC_TWIN (16 x 64): its teachers and routing book unchanged, B1 tables '
                        '(out-of-fold teacher means of the final out-of-fold cells)'},
    'TWB_C128': {'kind': 'twin', 'capacity_id': 'C128', 'construction': 'B1',
                 'what': 'task-only twin of the saved NRM1 PRC_TWIN_C128 (24 x 128): its teachers and routing book unchanged, B1 tables'},
    'TWB_C256': {'kind': 'twin', 'capacity_id': 'C256', 'construction': 'B1',
                 'what': 'prospective-capacity task-only twin (32 x 256; trigger T5 only): the NRM1 task teachers, a KL k-means book '
                         'from their cached out-of-fold predictions, B1 tables'},
    'TWA_C256': {'kind': 'twin', 'capacity_id': 'C256', 'construction': 'A',
                 'what': 'prospective-capacity task-only twin (32 x 256; trigger T5 only, if A is the construction chosen there): '
                         'label-refit decoders'},
    **{f'PRC_{s}': {'kind': 'candidate', 'budget': s, 'nrm_cfg': f'PRC_{s}',
                    'what': f'propensity-routed cells (FA_PRC_v1 rule), strength chosen by select_path at budget {s}, selected '
                            f'construction recomputed on the protected partition'} for s in P.BUDGETS},
    **{f'ORD_{s}': {'kind': 'control', 'budget': s, 'nrm_cfg': f'ORD_{s}',
                    'what': f'ordinary compression path: the twin at the most coarsened ladder rung whose nested spend meets {s} '
                            f'(SEX-blind mapping, SEX-informed operating point)'} for s in P.BUDGETS},
    'HQ_S2': {'kind': 'control', 'budget': 'S2', 'nrm_cfg': 'HQ_S2',
              'what': 'task-score-only grouping: the same objective, optimizer, grid, tilts and budget rule with pi, qO0, qO1 '
                      'replaced by their MODEL_FIT means within the twin task cells (routing a function of the task posteriors)'},
    'SHUF_S2': {'kind': 'control', 'budget': 'S2', 'nrm_cfg': 'SHUF_S2',
                'what': 'shuffled-signal control: the same optimizer, grid, tilts and budget rule with pi replaced by a keyed-hash '
                        'donor copy (NRM_FA_shuf_S<v>; deployable; no person-level SEX information) and a SEX-blind guard on the '
                        'person\'s own qO (qO0 = qO1 = qO; card FA_FRC_v1 change)'},
    'HQ_BR': {'kind': 'control', 'bracket': 'HQ', 'nrm_cfg': 'HQ_S2',
              'what': 'HQ at its bracketing strength: the first grid lambda whose nested MODEL_FIT total-NLL spend over the twin is '
                      '>= that of the PRC configuration selected at S2'},
    'SHUF_BR': {'kind': 'control', 'bracket': 'SHUF', 'nrm_cfg': 'SHUF_S2',
                'what': 'SHUF at its bracketing strength (as HQ_BR)'},
    'ORD_BR': {'kind': 'control', 'bracket': 'ORD', 'nrm_cfg': None,
               'what': 'ordinary compression at its bracketing rung: the first ladder rung whose nested MODEL_FIT total-NLL spend over '
                       'the twin is >= that of the PRC configuration selected at S2'},
    'ABL_LOC': {'kind': 'ablation', 'nrm_cfg': 'PRC_S2', 'route_override': {'w_pair': 0.0},
                'what': 'ablation of the strongest feasible nonzero release: pair weight 0 (income and occupation dependence terms '
                        'only; separately designed recipients), same lambda and capacity'},
    'ABL_PAIR': {'kind': 'ablation', 'nrm_cfg': 'PRC_S2', 'route_override': {'w_loc': 0.0},
                 'what': 'ablation of the strongest feasible nonzero release: local weights 0 (pair term only), same lambda and capacity'},
}
SELECT_FAMILY = {'PRC': 'PRC', 'ORD': 'ORD', 'HQ': 'HQ', 'SHUF': 'SHUF'}


# ---------------------------------------------------------------- the constructions (emitted tables; the partition is never touched)
def tables(construction, book, k, l, E=None, lab=None):
    """Emitted table of every income cell ('I', KI x 2) and occupation cell ('O', KO x 5) for the out-of-fold partition (k, l).
    Returns (tables, info)."""
    KI, KO = book['CI'].shape[0], book['CO'].shape[0]
    if construction == 'A':
        return ({'I': P.decoders(k, lab['income'], book['CI'], ALPHA), 'O': P.decoders(l, lab['occupation'], book['CO'], ALPHA)},
                {'labels': 'MODEL_FIT income and occupation of the out-of-fold-routed people', 'alpha': ALPHA})
    if construction == 'B0':
        return {'I': book['CI'].copy(), 'O': book['CO'].copy()}, {'labels': 'none (stored routing prototypes)'}
    if construction == 'B1':
        out, info = {}, {'labels': 'none (out-of-fold teacher means)'}
        for key, cells, Q, Cw, Kc in (('I', k, E['qI'], book['CI'], KI), ('O', l, E['qO'], book['CO'], KO)):
            n = np.bincount(cells, minlength=Kc)
            S = np.zeros((Kc, Q.shape[1]))
            np.add.at(S, cells, Q)
            T = Cw.copy()
            occ = n > 0
            T[occ] = S[occ] / n[occ, None]
            out[key] = P.clipn(T)
            info[f'unoccupied_prototype_fallback_{key}'] = int((~occ).sum())
        return out, info
    raise ValueError(f'unknown construction {construction!r}')


SHUF_MODE = 'shuf_pi_own_guard'


def shuf_embedding(E, bank_pi, tag):
    """SHUF of this campaign (card FA_FRC_v1, a change from FA_PRC_v1 before any SHUF fit): pi is replaced by the keyed-hash donor
    copy from the bank (SHA-256(tag | the row's compact features) mod bank size: a deterministic function of the six fields with no
    person-level SEX information), and the guard's SEX-conditional posteriors are replaced by the person's own qO (qO0 = qO1 = qO:
    a SEX-blind task-fidelity guard)."""
    d = P.donor_index(E['Z'], tag, len(bank_pi))
    R = dict(E)
    R['pi'] = np.asarray(bank_pi, dtype=np.float64)[d]
    R['qO0'] = E['qO']
    R['qO1'] = E['qO']
    return R


def entry_embedding(E, book):
    """The statistics an entry's routing rule reads: the shuffled-signal copy for SHUF; E itself otherwise (HQ restricts inside
    the source rule)."""
    if book.get('frc_embed') == SHUF_MODE:
        return shuf_embedding(E, book['shuf_bank_pi'], book['shuf_tag'])
    return E


def emit(E, book, tabs):
    """Release of runtime statistics E through the routing book and the emitted tables (the partition depends on the book and the
    statistics only, never on the tables)."""
    k, l = P.route(entry_embedding(E, book), book)
    return {'income': tabs['I'][k].copy(), 'occupation': tabs['O'][l].copy()}, k, l


def same_partition(a, b):
    """True iff two label vectors induce the same partition of the people (a bijection between their values)."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    na, nb = len(np.unique(a)), len(np.unique(b))
    nab = len(np.unique(np.column_stack([a, b]), axis=0))
    return bool(na == nb == nab)


def partition_identity(rel_a, rel_b):
    """The cell identities read off two releases' vectors (what any reader can recover) induce the same partition per view."""
    ia, ib = P.value_ids(rel_a), P.value_ids(rel_b)
    return {vw: same_partition(ia[vw], ib[vw]) for vw in ('income', 'occupation', 'pair')}


def distinct_equals_cells(rel, k, l):
    """No two internal cells emit the same vector: distinct emitted vectors = occupied cells (per view)."""
    ids = P.value_ids(rel)
    return {'income': same_partition(ids['income'], k), 'occupation': same_partition(ids['occupation'], l),
            'pair': same_partition(ids['pair'], k * 100000 + l)}


# ---------------------------------------------------------------- task-only twin and the nested fixed-routing comparison
def fit_twin(Z, lab, unit, version, capacity, T_fixed=None, hp=None, constructions=CONSTRUCTIONS, log=None):
    """The source twin pipeline (nrm.fa.prc.fit_pipeline with plan {'PRC_TWIN': ('twin',)}) with every construction's tables."""
    tch = P.fit_teachers(Z, lab, unit, version, hp or P.HGB, ('inc', 'occ'), T_fixed=T_fixed, log=log)
    E = P.embed_oof(tch)
    E['Z'] = Z
    tb, k0, l0 = P.twin_book(E, int(capacity[0]), int(capacity[1]), version, P.route_params('PRC_TWIN'))
    tabs, infos = {}, {}
    for c in constructions:
        tabs[c], infos[c] = tables(c, tb, k0, l0, E, lab)
    KI, KO = tb['CI'].shape[0], tb['CO'].shape[0]
    same_people = None
    if 'income' in lab:
        cnt_i = np.bincount(k0 * 2 + lab['income'], minlength=KI * 2).reshape(KI, 2).sum(1)
        cnt_o = np.bincount(l0 * 5 + lab['occupation'], minlength=KO * 5).reshape(KO, 5).sum(1)
        same_people = bool(np.array_equal(cnt_i, np.bincount(k0, minlength=KI)) and np.array_equal(cnt_o, np.bincount(l0, minlength=KO)))
    return {'models': tch['models'], 'T': tch['T'], 'book': tb, 'cells': (k0, l0), 'tables': tabs, 'tables_info': infos, 'E_fit': E,
            'A_and_B1_count_the_same_people': same_people}


METRICS = ('inc_nll', 'occ_nll', 'tot_nll', 'inc_brier', 'occ_brier', 'inc_acc', 'occ_acc')


def _metrics(rel, lab):
    m = P.person_metrics(rel, lab)
    m['tot_nll'] = m['inc_nll'] + m['occ_nll']
    return m


def paired(store, a, b, unit, keys=METRICS):
    """loss(a) - loss(b) per person (accuracy as a loss: acc(b) - acc(a)), household-clustered SEs (nrm.fa.prc.cluster_se)."""
    out = {}
    for key in keys:
        d = (store[b][key] - store[a][key]) if key.endswith('_acc') else (store[a][key] - store[b][key])
        out[key] = [float(d.mean()), P.cluster_se(d, unit)]
    return out


PAIRS = (('B1', 'A'), ('B0', 'A'), ('B1', 'B0'))


def _pdiff(store, a, b, key, mask=slice(None)):
    return (store[b][key][mask] - store[a][key][mask]) if key.endswith('_acc') else (store[a][key][mask] - store[b][key][mask])


def nested_twin(Z, lab, unit, version, capacity, T_fixed, outer=P.N_OUTER, hp=None, log=None, keep_vectors=False):
    """NRM1 gate estimator with all constructions on identical held-out cells. Returns per-person losses for CONT, A, B0, B1, the
    held-out cells and the per-fold checks (and, with keep_vectors, the held-out emitted vectors)."""
    of = P.household_folds(unit, outer, OUTER_TAG.format(v=int(version)))
    names = ('CONT',) + CONSTRUCTIONS
    store = {nm: {} for nm in names}
    cells = {'k': np.full(len(Z), -1), 'l': np.full(len(Z), -1)}
    vectors = {c: {'income': np.zeros((len(Z), 2)), 'occupation': np.zeros((len(Z), 5))} for c in ('CONT',) + CONSTRUCTIONS}
    folds = []
    for o in range(outer):
        tr, te = of != o, of == o
        if np.intersect1d(unit[tr], unit[te]).size:
            raise AssertionError('household in both outer parts')
        st = fit_twin(Z[tr], {k: v[tr] for k, v in lab.items()}, unit[tr], version, capacity, T_fixed=T_fixed, hp=hp, log=log)
        E = P.embed(st['models'], Z[te])
        k, l = P.route(E, st['book'])
        rels = {'CONT': {'income': E['qI'].copy(), 'occupation': E['qO'].copy()}}
        for c in CONSTRUCTIONS:
            rels[c] = {'income': st['tables'][c]['I'][k].copy(), 'occupation': st['tables'][c]['O'][l].copy()}
        ident = {f'{a}_vs_{b}': partition_identity(rels[a], rels[b]) for a, b in (('A', 'B1'), ('A', 'B0'), ('B0', 'B1'))}
        if not all(all(v.values()) for v in ident.values()):
            raise AssertionError(f'fold {o}: the constructions do not induce one partition of the held-out people: {ident}')
        ltr = {kk: v[te] for kk, v in lab.items()}
        for nm, rel in rels.items():
            for key, v in _metrics(rel, ltr).items():
                store[nm].setdefault(key, np.zeros(len(Z)))[te] = v
        cells['k'][te], cells['l'][te] = k, l
        for c, rel in rels.items():
            for task in ('income', 'occupation'):
                vectors[c][task][te] = rel[task]
        folds.append({'fold': o, 'train_people': int(tr.sum()), 'heldout_people': int(te.sum()),
                      'occupied_train': P._occupied(*st['cells']), 'partition_identity': ident,
                      'b1_fallbacks': st['tables_info']['B1'], 'A_and_B1_count_the_same_people': st['A_and_B1_count_the_same_people'],
                      'paired_fold_means': {f'{a}_minus_{b}': {key: float(_pdiff(store, a, b, key, te).mean()) for key in METRICS}
                                            for a, b in PAIRS},
                      'max_abs_B1_minus_A': {t: float(np.abs(st['tables']['B1'][t] - st['tables']['A'][t]).max()) for t in ('I', 'O')},
                      'max_abs_B1_minus_B0': {t: float(np.abs(st['tables']['B1'][t] - st['tables']['B0'][t]).max()) for t in ('I', 'O')},
                      'paired_fold_mean_B1_minus_A_occ_nll': float((store['B1']['occ_nll'][te] - store['A']['occ_nll'][te]).mean())})
    if (cells['k'] < 0).any():
        raise AssertionError('some person was never held out')
    if keep_vectors:
        return store, cells, folds, vectors
    return store, cells, folds


def gate_label(spend_rec, limits=GATE_LIMITS):
    ok = all(spend_rec[k][0] + 2 * spend_rec[k][1] <= v for k, v in limits.items())
    return 'ORIGINAL_CAPACITY_GATE_PASS' if ok else 'ORIGINAL_CAPACITY_GATE_FAIL'


def nest_summary(store, unit, version, capacity, T_fixed, folds, outer=P.N_OUTER):
    names = ('CONT',) + CONSTRUCTIONS
    means = {nm: {key: float(v.mean()) for key, v in store[nm].items()} for nm in names}
    over_cont = {c: paired(store, c, 'CONT', unit, keys=('occ_nll', 'inc_nll', 'occ_acc', 'inc_acc', 'tot_nll', 'occ_brier', 'inc_brier'))
                 for c in CONSTRUCTIONS}
    diffs = {f'{a}_minus_{b}': paired(store, a, b, unit) for a, b in PAIRS}
    fold_se = {}
    for a, b in PAIRS:
        nm = f'{a}_minus_{b}'
        fold_se[nm] = {}
        for key in METRICS:
            fm = np.array([f['paired_fold_means'][nm][key] for f in folds])
            sd = float(fm.std(ddof=1) / np.sqrt(len(fm))) if len(fm) > 1 else float('nan')
            fold_se[nm][key] = {'fold_means': fm.tolist(), 'sd_fold_means_over_sqrt_folds': sd,
                                'se_used': float(max(diffs[nm][key][1], sd)), 'negative_folds': int((fm < 0).sum())}
    share = {}
    for task in ('occ_nll', 'inc_nll', 'tot_nll'):
        sa = over_cont['A'][task][0]
        share[task] = float((sa - over_cont['B1'][task][0]) / sa) if sa > 0 else None
    return {'stage': 'nested_fixed_routing_comparison', 'version': int(version), 'capacity': list(capacity),
            'outer_folds': outer, 'outer_tag': OUTER_TAG.format(v=int(version)), 'T_fixed': T_fixed,
            'nested_T': 'non-nested (full-pool T*), as the NRM1 gate', 'nested_means': means, 'spend_over_cont': over_cont,
            'original_capacity_gate': {c: gate_label(over_cont[c]) for c in CONSTRUCTIONS},
            'original_capacity_gate_limits_estimate_plus_2se': GATE_LIMITS,
            'paired_differences': diffs, 'paired_differences_fold_level': fold_se,
            'se_used_rule': 'max(household-clustered SE, SD of the outer-fold means / sqrt(folds)) (adjudication, challenger 17)',
            'share_of_A_spend_over_cont_removed_by_B1': share, 'folds': folds,
            'people': int(len(unit)), 'households': int(len(np.unique(unit))),
            'note': 'MODEL_FIT nested held-out estimates on identical cells; SEX is never read; accuracy rows are losses; the '
                    'original capacity gate is reported, never this study\'s eligibility rule'}


def gate_reproduction(summary, gate_rec):
    """A and CONT of this estimator against the NRM1 GATE record (nested means and spends of PRC_TWIN over CONT)."""
    diffs = []
    for nm_src, nm in (('CONT', 'CONT'), ('PRC_TWIN', 'A')):
        for key, v in gate_rec['nested_means'][nm_src].items():
            diffs.append(abs(float(v) - summary['nested_means'][nm][key]))
    for key, (m, s) in gate_rec['spend_twin_over_cont'].items():
        diffs.append(abs(float(m) - summary['spend_over_cont']['A'][key][0]))
        diffs.append(abs(float(s) - summary['spend_over_cont']['A'][key][1]))
    mx = float(max(diffs))
    return {'max_abs_difference': mx, 'reproduced': bool(mx <= 1e-12), 'gate_record_utc': gate_rec.get('utc'),
            'gate_record_label': gate_rec.get('gate')}


# ---------------------------------------------------------------- the protected pipeline with a construction (source A = bitwise)
def chain_inputs(cfg, override, E, version):
    """(statistics the chain optimizes on, route parameters, extra book keys) of one strength chain. SHUF runs the source chain
    with no source restriction on the shuffled-signal copy (shuf_embedding; bank = the training rows' out-of-fold pi); PRC, HQ
    and the ablations run the source chain unchanged."""
    rp = P.route_params(cfg, version=version)
    rp.update(override)
    if rp.get('restrict') == 'shuffle':
        tag = rp['shuf_tag']
        rp = P.route_params('PRC_S2', version=version)
        rp.update(override)
        bank = np.array(E['pi'], dtype=np.float64, copy=True)
        return shuf_embedding(E, bank, tag), rp, {'frc_embed': SHUF_MODE, 'shuf_bank_pi': bank, 'shuf_tag': tag}
    return E, rp, {}


def apply_c(st, name, E):
    """Release of entry `name` from runtime statistics E (this module's emit: SHUF entries read their shuffled-signal copy)."""
    book = st['books'][name]
    if book is None:
        return {'income': E['qI'].copy(), 'occupation': E['qO'].copy()}, None, None
    return emit(E, book, st['dec'][name])


def fit_pipeline_c(Z, lab, unit, version, plan, construction, hp=None, T_fixed=None, capacity=(16, 64), log=None):
    """nrm.fa.prc.fit_pipeline with the emitted tables of `construction` recomputed on every entry's own partition."""
    hp = hp or P.HGB
    need_sex = any(p[0] == 'prc' for p in plan.values())
    names = ('inc', 'occ', 'sex', 'occ_s') if need_sex else ('inc', 'occ')
    tch = P.fit_teachers(Z, lab, unit, version, hp, names, T_fixed=T_fixed, log=log)
    E = P.embed_oof(tch)
    E['Z'] = Z
    rp0 = P.route_params('PRC_TWIN')
    KI, KO = capacity
    tb, k0, l0 = P.twin_book(E, KI, KO, version, rp0)
    state = {'version': int(version), 'T': tch['T'], 'models': tch['models'], 'books': {}, 'dec': {}, 'fit_cells': {},
             'capacity': [int(KI), int(KO)], 'E_fit': E, 'twin_fit': (tb, k0, l0), 'construction': construction}
    chains = {}
    for p in plan.values():
        if p[0] == 'prc':
            chains.setdefault((p[1], json.dumps(p[3] if len(p) > 3 else {}, sort_keys=True)), set()).add(float(p[2]))
    paths = {}
    for (c, ov), ls in chains.items():
        Ec, rp, extra = chain_inputs(c, json.loads(ov), E, version)
        paths[(c, ov)] = ({lam: st for lam, st in P.protected_path(Ec, tb, k0, l0, sorted(ls), rp).items()}, extra)
    for name, p in plan.items():
        if p[0] == 'cont':
            state['books'][name] = state['dec'][name] = state['fit_cells'][name] = None
            continue
        if p[0] == 'twin':
            book, k, l = tb, k0, l0
        elif p[0] == 'ord':
            book, k, l = P.twin_book(E, p[1], p[2], version, rp0)
        else:
            chain, extra = paths[(p[1], json.dumps(p[3] if len(p) > 3 else {}, sort_keys=True))]
            book, k, l = chain[float(p[2])]
            book = {**book, **extra}
        state['books'][name] = book
        state['dec'][name] = tables(construction, book, k, l, E, lab)[0]
        state['fit_cells'][name] = (k, l)
    return state


def nested_c(Z, lab, unit, version, plan, construction, hp=None, T_fixed=None, capacity=(16, 64), outer=P.N_OUTER, log=None):
    """nrm.fa.prc.nested with fit_pipeline_c (identical outputs for construction 'A' on every entry except SHUF, which this card
    changed; selftest)."""
    of = P.household_folds(unit, outer, OUTER_TAG.format(v=int(version)))
    res = {name: {'rows': np.zeros(len(Z), dtype=bool)} for name in plan}
    store = {name: {} for name in plan}
    fitted = {name: [] for name in plan}
    moved = {name: [] for name in plan}
    for o in range(outer):
        tr, te = of != o, of == o
        if np.intersect1d(unit[tr], unit[te]).size:
            raise AssertionError('household in both outer parts')
        st = fit_pipeline_c(Z[tr], {k: v[tr] for k, v in lab.items()}, unit[tr], version, plan, construction, hp=hp,
                            T_fixed=T_fixed, capacity=capacity, log=log)
        E = P.embed(st['models'], Z[te])
        _, kt, lt = apply_c(st, 'PRC_TWIN', E) if 'PRC_TWIN' in plan else (None, None, None)
        for name in plan:
            rel, k, l = apply_c(st, name, E)
            m = P.person_metrics(rel, {k2: v[te] for k2, v in lab.items()})
            if k is not None and 'sex' in lab:
                ktr, ltr = st['fit_cells'][name]
                KO = st['books'][name]['CO'].shape[0]
                vtr, vte = P.views(ktr, ltr, KO), P.views(k, l, KO)
                for vw in vtr:
                    m[f'look_{vw}'] = P.lookup_scores(vtr[vw], lab['sex'][tr], vte[vw])
                s_tr = lab['sex'][tr]
                fitted[name].append({'auc_pair_insample': P.auc(P.lookup_scores(vtr['pair'], s_tr, vtr['pair']), s_tr),
                                     'mi_pair_labels_insample': P.plugin_mi(vtr['pair'], s_tr.astype(np.float64)),
                                     'mi_pair_pi_fitted': (P.plugin_mi(vtr['pair'], st['E_fit']['pi'])
                                                           if 'pi' in st['E_fit'] else float('nan'))})
            if k is not None and kt is not None and st['books'][name]['CO'].shape == st['books']['PRC_TWIN']['CO'].shape \
                    and st['books'][name]['CI'].shape == st['books']['PRC_TWIN']['CI'].shape:
                ktr, ltr = st['fit_cells'][name]
                k0, l0 = st['fit_cells']['PRC_TWIN']
                moved[name].append({'heldout_rerouted_vs_twin': float(((k != kt) | (l != lt)).mean()),
                                    'fit_rerouted_vs_twin': float(((ktr != k0) | (ltr != l0)).mean())})
            for key, v in m.items():
                store[name].setdefault(key, np.zeros(len(Z)))[te] = v
            res[name]['rows'] |= te
    out = {}
    for name in plan:
        d = store[name]
        out[name] = {key: float(v.mean()) for key, v in d.items() if not key.startswith('look_')}
        if 'sex' in lab:
            for vw in ('pair', 'income', 'occupation'):
                if f'look_{vw}' in d:
                    out[name][f'auc_{vw}'] = P.auc(d[f'look_{vw}'], lab['sex'])
            sx = lab['sex'].astype(bool)
            out[name]['occ_brier_men'] = float(d['occ_brier'][~sx].mean())
            out[name]['occ_brier_women'] = float(d['occ_brier'][sx].mean())
            if fitted[name]:
                for key in fitted[name][0]:
                    out[name][f'fit_{key}'] = float(np.mean([f[key] for f in fitted[name]]))
                if 'auc_pair' in out[name]:
                    out[name]['divergence_auc_pair_insample_minus_heldout'] = out[name]['fit_auc_pair_insample'] - out[name]['auc_pair']
        if moved[name]:
            for key in moved[name][0]:
                out[name][f'mean_{key}'] = float(np.mean([m[key] for m in moved[name]]))
        out[name]['_per_person'] = d
    return out


# ---------------------------------------------------------------- storage and the deployment interface (frc.deploy contract)
_ROOT = {'runs': None, 'cache': None, 'records': None}      # selftest only: temporary roots (no real path is touched)


def _paths():
    if _ROOT['runs'] is not None:
        return Path(_ROOT['runs']), Path(_ROOT['cache']), Path(_ROOT['records'])
    from frc import common as C
    return C.RUNS, C.PRIV / 'fa_cache', C.RESULTS / 'fa'


def _utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def model_dir(config_id, version):
    if config_id not in CONFIGS:
        raise KeyError(f'unknown configuration {config_id}')
    return _paths()[0] / FAMILY / config_id / f'V{int(version)}'


def model_paths(config_id, version):
    """The files predict() reads for one configuration (hash-pinned in a deployment package)."""
    d = model_dir(config_id, version)
    return [d / 'MODEL.pkl', d / 'MODEL.json']


def write_model(config_id, version, payload):
    """Write-once MODEL.pkl / MODEL.json beside the release (an incomplete earlier attempt is moved aside, kept and named)."""
    d = model_dir(config_id, version)
    if (d / 'RELEASE.npz').exists():
        raise PermissionError(f'{FAMILY}/{config_id}/V{version} is complete (append-only; a revision gets a new config id)')
    if d.exists() and any(d.iterdir()):
        d.rename(d.with_name(f'{d.name}__incomplete_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'))
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    buf = pickle.dumps(payload, protocol=5)
    with open(d / 'MODEL.pkl', 'xb') as f:
        f.write(buf)
    meta = {'schema': SCHEMA, 'family': FAMILY, 'config_id': config_id, 'version': int(version), 'model_sha256': _sha_bytes(buf),
            'feature_names_sha256': _sha_bytes(json.dumps(list(payload['feature_names'])).encode()),
            'teachers': sorted(payload['teachers']), 'T_star': payload['T'], 'construction': payload['construction'],
            'capacity': payload['capacity'], 'utc': _utc()}
    with open(d / 'MODEL.json', 'x') as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def load_model(config_id, version):
    d = model_dir(config_id, version)
    meta = json.loads((d / 'MODEL.json').read_text())
    buf = (d / 'MODEL.pkl').read_bytes()
    if _sha_bytes(buf) != meta['model_sha256'] or meta['config_id'] != config_id or int(meta['version']) != int(version):
        raise PermissionError(f'{config_id} V{version}: MODEL.pkl differs from MODEL.json')
    M = pickle.loads(buf)
    if M['schema'] != SCHEMA or M['config_id'] != config_id or int(M['version']) != int(version):
        raise PermissionError(f'{config_id} V{version}: model payload identity differs')
    return M


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    """Deployment function: the configuration's income (n, 2) and occupation (n, 5) vectors from admitted features alone, reading
    only model_paths(config_id, version). Deterministic: per-request keys and draws other than 0 are refused; the feature columns
    must be exactly the admitted names the configuration was fitted on."""
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
    rel, _, _ = emit(P.embed(M['teachers'], FA.compact_po(X, names)), M['book'], M['dec'])
    return rel


def cells_of(config_id, version, X, feature_names):
    """Internal diagnostic (never released): the routed cells of a saved configuration."""
    M = load_model(config_id, version)
    from ohn import fam_a as FA
    return P.route(P.embed(M['teachers'], FA.compact_po(X, tuple(map(str, feature_names)))), M['book'])


def _payload(config_id, version, names, teachers, book, dec, T, construction, capacity, extra=None):
    return {'schema': SCHEMA, 'config_id': config_id, 'version': int(version), 'feature_names': tuple(names), 'teachers': teachers,
            'book': book, 'dec': dec, 'T': T, 'construction': construction, 'capacity': [int(capacity[0]), int(capacity[1])],
            **(extra or {})}


def _release_all(config_id, version, role_features):
    """Every role's release computed by predict() itself, so the saved wire IS the deployment function's output (bitwise)."""
    return {r: predict(config_id, version, F['X'], F['feature_names']) for r, F in role_features.items()}


def _write_once_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=float)


def _cache_put(version, name, obj):
    d = _paths()[1] / f'V{int(version)}'
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    for suf in ('pkl', 'sha256'):
        p = d / f'{name}.{suf}'
        if p.exists():
            p.rename(d / f'{name}__superseded_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.{suf}')
    buf = pickle.dumps(obj, protocol=5)
    with open(d / f'{name}.pkl', 'xb') as f:
        f.write(buf)
    with open(d / f'{name}.sha256', 'x') as f:
        f.write(_sha_bytes(buf))
    return _sha_bytes(buf)


def _cache_get(version, name, row_sha=None):
    d = _paths()[1] / f'V{int(version)}'
    buf = (d / f'{name}.pkl').read_bytes()
    if _sha_bytes(buf) != (d / f'{name}.sha256').read_text().strip():
        raise PermissionError(f'cache {name} V{version} differs from its hash')
    obj = pickle.loads(buf)
    if row_sha is not None and obj['row_id_sha256'] != row_sha:
        raise PermissionError(f'cache {name} V{version} was fitted on other rows')
    return obj


def _prep(data):
    """Compact coding, labels present in `data` (any subset of income / occupation / sex), households, names, row-set hash."""
    from ohn import fam_a as FA
    names = tuple(map(str, data['feature_names']))
    lab = {k: np.asarray(data[k], dtype=np.int64) for k in ('income', 'occupation', 'sex') if k in data}
    return FA.compact_po(data['X'], names), lab, np.asarray(data['unit']), names, P.array_sha(np.asarray(data['row_id']))


def _record(card, kind, description, fits, roles_read, exposure, people, households, cpu, extra):
    return {'mechanism_card': card, 'kind': kind, 'description': description, 'n_model_fits': int(fits[0]),
            'n_calibrator_fits': 0, 'n_crossfit_fits': int(fits[1]), 'n_critic_fits': 0, 'roles_read': roles_read,
            'exposure': exposure, 'training_people': int(people), 'training_households': int(households),
            'fit_cpu_s': round(float(cpu), 1), 'randomized': False, **extra}


def _no_stats(**kw):
    return None


# ---------------------------------------------------------------- stages (data and storage injected; queue units below)
def source_task(version, row_sha):
    """The NRM1 stage-1 task cache (teachers, out-of-fold predictions) of version v, hash-checked, read-only."""
    return P._cache_get(version, 'TASK', row_sha)


def stage_repro(version, mf, role_features, log=print):
    """T0 (label-free): the saved NRM1 twins recomputed from their saved models on all six roles (bitwise against the saved
    RELEASE.npz) and their routing books refitted from the cached out-of-fold teachers (bitwise against the saved books)."""
    from nrm import release as NRL
    t0 = time.process_time()
    Z, _, unit, names, row_sha = _prep(mf)
    task = source_task(version, row_sha)
    E = P.embed_oof(task)
    out = {}
    for cid, src in SOURCE_TWIN.items():
        M = P._load_model(src, version)
        saved, rec = NRL.load('FA', src, version)
        eq = {}
        for r, F in role_features.items():
            rel = P.predict(src, version, F['X'], F['feature_names'])
            eq[r] = all(np.array_equal(rel[t], saved[r][t]) for t in ('income', 'occupation'))
        KI, KO = M['book']['CI'].shape[0], M['book']['CO'].shape[0]
        tb, k0, l0 = P.twin_book(E, KI, KO, version, P.route_params('PRC_TWIN'))
        book_eq = all(np.array_equal(tb[key], M['book'][key]) for key in ('CI', 'CO')) and \
            all(tb[key] == M['book'][key] for key in tb if not isinstance(tb[key], np.ndarray))
        kr, lr = P.route(E, M['book'])
        out[cid] = {'source_config': src, 'release_sha256': rec['release_sha256'], 'model_sha256': json.loads(
                        (P.model_dir(src, version) / 'MODEL.json').read_text())['model_sha256'],
                    'task_cache_sha256_in_record': rec.get('task_cache_sha256'),
                    'predict_reproduces_saved_release_bitwise': eq, 'book_refit_bitwise': bool(book_eq),
                    'rule_consistent_oof_partition': bool(np.array_equal(kr, k0) and np.array_equal(lr, l0)),
                    'occupied_cells_oof': P._occupied(k0, l0), 'capacity': [int(KI), int(KO)]}
    task_sha = _sha_bytes((P._paths()[1] / f'V{int(version)}' / 'TASK.pkl').read_bytes())
    ok = all(all(v['predict_reproduces_saved_release_bitwise'].values()) and v['book_refit_bitwise'] and
             v['rule_consistent_oof_partition'] and v['task_cache_sha256_in_record'] == task_sha for v in out.values())
    rec = {'stage': 'T0_reproduction', 'version': int(version), 'labels_read': False, 'task_cache_sha256': task_sha, 'twins': out,
           'reproduced': bool(ok), 'cpu_s': round(time.process_time() - t0, 1), 'utc': _utc()}
    _write_once_json(_paths()[2] / f'REPRO_V{int(version)}.json', rec)
    log(json.dumps({'stage': 'repro', 'version': int(version), 'reproduced': ok}))
    if not ok:
        raise AssertionError('T0 reproduction failed: the stage stops for diagnosis')
    return rec


def stage_nest(version, capacity_id, data, log=print, stats=_no_stats, outer=P.N_OUTER):
    """T1 (MODEL_FIT income and occupation; SEX never read): the nested fixed-routing comparison at one capacity, with the source
    reproduction checks (GATE record for A; saved decoders from the full-pool labels)."""
    t0 = time.process_time()
    if 'sex' in data:
        raise PermissionError('the fixed-routing comparison never reads SEX')
    Z, lab, unit, names, row_sha = _prep(data)
    capacity = CAPACITIES[capacity_id]
    task = source_task(version, row_sha)
    store, cells, folds = nested_twin(Z, lab, unit, version, capacity, T_fixed=task['T'], outer=outer, log=log)
    summ = nest_summary(store, unit, version, capacity, task['T'], folds, outer=outer)
    repro = {}
    if capacity_id in SOURCE_TWIN:
        from nrm import common as NC
        g = NC.RESULTS / 'roles' / 'designer_a' / f'GATE_V{int(version)}_{capacity[0]}x{capacity[1]}.json'
        repro['gate'] = gate_reproduction(summ, json.loads(g.read_text())) if outer == P.N_OUTER else {'skipped': 'outer != 5'}
        M = P._load_model(SOURCE_TWIN[capacity_id], version)
        E = P.embed_oof(task)
        k0, l0 = P.route(E, M['book'])
        tA, _ = tables('A', M['book'], k0, l0, E, lab)
        tB1, infB1 = tables('B1', M['book'], k0, l0, E, lab)
        repro['decoders_from_labels_bitwise'] = bool(np.array_equal(tA['I'], M['dec']['I']) and np.array_equal(tA['O'], M['dec']['O']))
        repro['full_pool_max_abs_B1_minus_A'] = {t: float(np.abs(tB1[t] - tA[t]).max()) for t in ('I', 'O')}
        repro['full_pool_max_abs_B1_minus_B0'] = {t: float(np.abs(tB1[t] - M['book']['C' + t]).max()) for t in ('I', 'O')}
        repro['full_pool_b1_fallbacks'] = infB1
    summ['source_reproduction'] = repro
    summ['cpu_s'] = round(time.process_time() - t0, 1)
    summ['utc'] = _utc()
    priv = _paths()[1] / 'nested' / f'NEST_{capacity_id}_V{int(version)}.npz'
    priv.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(priv, 'xb') as f:
        np.savez(f, unit=unit, k=cells['k'], l=cells['l'],
                 **{f'{nm}__{key}': v for nm, d in store.items() for key, v in d.items()})
    summ['per_person_private_file_sha256'] = _sha_bytes(priv.read_bytes())
    _write_once_json(_paths()[2] / f'NEST_{capacity_id}_V{int(version)}.json', summ)
    stats(n_model_fits=2 * outer, n_crossfit_fits=2 * P.N_XF * outer)
    log(json.dumps({'stage': 'nest', 'capacity': capacity_id, 'version': int(version),
                    'B1_minus_A_occ_nll': summ['paired_differences']['B1_minus_A']['occ_nll'], 'repro': repro.get('gate')}))
    if capacity_id in SOURCE_TWIN and not (repro['gate'].get('reproduced', True) and repro['decoders_from_labels_bitwise']):
        raise AssertionError('T1 source reproduction failed: the stage stops for diagnosis')
    return summ


def _source_release(src, version):
    from nrm import release as NRL
    return NRL.load('FA', src, version)[0]


def stage_twb(version, config_id, mf, role_features, save, card, log=print, stats=_no_stats, lab=None,
              source_release=_source_release):
    """T2 / T5 (label-free for B1): a deployable task-only release. C64 / C128: the saved NRM1 twin's teachers and book unchanged,
    the construction's tables on the cached full-pool out-of-fold partition; C256: a book from the cached out-of-fold teachers."""
    t0 = time.process_time()
    cfg = CONFIGS[config_id]
    if cfg['kind'] != 'twin':
        raise ValueError(f'{config_id} is not a task-only twin')
    cid, cons = cfg['capacity_id'], cfg['construction']
    if cons == 'A' and lab is None:
        raise PermissionError('construction A needs the MODEL_FIT labels of its decoders')
    Z, _, unit, names, row_sha = _prep(mf)
    task = source_task(version, row_sha)
    E = P.embed_oof(task)
    if cid in SOURCE_TWIN:
        M = P._load_model(SOURCE_TWIN[cid], version)
        if tuple(M['feature_names']) != names:
            raise PermissionError('source model feature names differ')
        book, teachers, T = M['book'], dict(M['teachers']), dict(M['T'])
        k0, l0 = P.route(E, book)
        src = {'source_model': f'nrm FA/{SOURCE_TWIN[cid]}/V{int(version)}',
               'source_model_sha256': json.loads((P.model_dir(SOURCE_TWIN[cid], version) / 'MODEL.json').read_text())['model_sha256']}
    else:
        KI, KO = CAPACITIES[cid]
        book, k0, l0 = P.twin_book(E, KI, KO, version, P.route_params('PRC_TWIN'))
        teachers, T = dict(task['models']), dict(task['T'])
        src = {'source_model': 'nrm stage-1 task cache (teachers); book refitted here (label-free KL k-means)'}
    tabs, info = tables(cons, book, k0, l0, E, lab)
    write_model(config_id, version, _payload(config_id, version, names, teachers, book, tabs, T, cons, CAPACITIES[cid], src))
    arrays = _release_all(config_id, version, role_features)
    ident = {}
    if cid in SOURCE_TWIN:
        saved = source_release(SOURCE_TWIN[cid], version)
        ident = {r: partition_identity(arrays[r], saved[r]) for r in arrays}
        if not all(all(v.values()) for v in ident.values()):
            raise AssertionError(f'{config_id}: the partition differs from the saved A release: {ident}')
    distinct = distinct_equals_cells(predict(config_id, version, mf['X'], mf['feature_names']), *P.route(P.embed(teachers, Z), book))
    occ_i = np.bincount(k0, minlength=book['CI'].shape[0]) > 0
    occ_o = np.bincount(l0, minlength=book['CO'].shape[0]) > 0
    empty = {}
    from ohn import fam_a as FA
    for r, F in role_features.items():
        kr, lr = P.route(P.embed(teachers, FA.compact_po(F['X'], F['feature_names'])), book)
        empty[r] = {'income': int((~occ_i[kr]).sum()), 'occupation': int((~occ_o[lr]).sum()), 'people': int(len(kr))}
    task_sha = _sha_bytes((P._paths()[1] / f'V{int(version)}' / 'TASK.pkl').read_bytes())
    rec = _record(card, 'twin', cfg['what'], (0, 0),
                  'none (label-free construction)' if cons != 'A' else 'MODEL_FIT: income, occupation (decoders)',
                  'MODEL_FIT income and occupation labels of all 20,000 MODEL_FIT people reached the shared task teachers (NRM1 '
                  'stage 1, inner household CV for T*, full refit, 5-fold household cross-fitting); '
                  + ('the B1 tables use only those teachers\' out-of-fold predictions; ' if cons == 'B1' else
                     'the label-refit decoders count the out-of-fold-routed people\'s labels; ')
                  + 'no SEX, CALIBRATION or MODEL_SELECT label', len(Z), len(np.unique(unit)), time.process_time() - t0,
                  {'construction': cons, 'capacity': list(CAPACITIES[cid]), 'capacity_id': cid, 'T_star': T,
                   'occupied_cells_oof': P._occupied(k0, l0), 'tables_info': info, 'task_cache_sha256': task_sha,
                   'partition_identity_with_saved_A_release': ident, 'distinct_vectors_equal_cells_model_fit': distinct,
                   'people_routed_into_oof_empty_cells': empty,
                   'reader_budget': {r: P.reader_budget(arrays[r]) for r in arrays}, **src})
    save(FAMILY, config_id, version, arrays, rec)
    log(json.dumps({'stage': 'twb', 'config': config_id, 'version': int(version), 'cpu_s': round(time.process_time() - t0, 1)}))
    return rec


def protection_decision():
    """The committed qualification decision (results/<campaign>/decisions/QUAL_DECISION.json): construction, capacity, twin."""
    from frc import common as C
    p = C.RESULTS / 'decisions' / 'QUAL_DECISION.json'
    if not p.is_file() or not C.committed(p):
        raise PermissionError('the protected stage needs the committed qualification decision (decisions/QUAL_DECISION.json)')
    d = json.loads(p.read_text())
    if d.get('outcome') != 'QUALIFIED':
        raise PermissionError(f'qualification outcome {d.get("outcome")}: the protected stage is not fitted')
    if d['construction'] not in ('A', 'B1') or d['capacity_id'] not in CAPACITIES:
        raise ValueError('malformed qualification decision')
    return d


def _plan(capacity_id, ladder=None):
    plan = {'PRC_TWIN': ('twin',)}
    for fam, cfg in (('PRC', 'PRC_S2'), ('HQ', 'HQ_S2'), ('SHUF', 'SHUF_S2')):
        plan.update({f'{fam}@{g}': ('prc', cfg, g) for g in P.LAMBDA_GRID})
    plan.update({f'ORD@{a}x{b}': ('ord', a, b) for a, b in (ladder or ORD_LADDERS[capacity_id])})
    return plan


def total_spend(nest, name, ref='PRC_TWIN'):
    """Nested MODEL_FIT held-out total-NLL (income + occupation) spend of entry `name` over the twin (point estimate)."""
    a, b = nest[name]['_per_person'], nest[ref]['_per_person']
    return float(((a['inc_nll'] + a['occ_nll']) - (b['inc_nll'] + b['occ_nll'])).mean())


def brackets_for(nest, plan, sel):
    """Bracketing points of HQ, SHUF and ORD (adjudication, challenger 5): the first strength in path order (grid lambda, or ladder
    rung) whose nested total-NLL spend over the twin is >= that of the PRC configuration selected at S2 (MODEL_FIT only)."""
    s2 = sel['PRC'].get('S2')
    out = {'prc_s2': s2, 'target_total_nll_spend': None, 'brackets': {}, 'spend': {}}
    if s2 is None:
        out['brackets'] = {f: 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH' for f in ('HQ', 'SHUF', 'ORD')}
        return out
    target = total_spend(nest, s2)
    out['target_total_nll_spend'] = target
    for fam in ('HQ', 'SHUF', 'ORD'):
        cands = [n for n in plan if n.startswith(fam + '@')]
        sp = {c: total_spend(nest, c) for c in cands}
        out['spend'][fam] = sp
        out['brackets'][fam] = next((c for c in cands if sp[c] >= target), 'CONTROL_DID_NOT_REACH_SPEND')
    return out


def _wauc_draws(y, s, W):
    """AUC of fixed scores s under household-bootstrap weights W (B x n) (verbatim frc.devreaders.wauc_draws)."""
    u, inv = np.unique(s, return_inverse=True)
    out = np.empty(W.shape[0])
    pos = (y == 1).astype(np.float64); neg = 1.0 - pos
    for b in range(W.shape[0]):
        wp = np.bincount(inv, weights=W[b] * pos, minlength=len(u)); wn = np.bincount(inv, weights=W[b] * neg, minlength=len(u))
        cn = np.cumsum(wn) - wn
        out[b] = np.sum(wp * (cn + 0.5 * wn)) / (wp.sum() * wn.sum())
    return out


def nested_lookup(nest, lab, unit, version, names, ref='PRC_TWIN', B=400):
    """Nested held-out pair LOOKUP AUC of every entry and its paired difference from the twin with a MODEL_FIT household-bootstrap
    SE (B draws, seed SeedSequence([2026102100, v, 4]); scores fixed)."""
    y = np.asarray(lab['sex'])
    g, inv = np.unique(unit, return_inverse=True)
    rng = np.random.default_rng(np.random.SeedSequence([2026102100, int(version), 4]))
    W = np.empty((B, len(y)))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g))[inv]
    sref = nest[ref]['_per_person']['look_pair']
    bref = _wauc_draws(y, sref, W)
    a_ref = P.auc(sref, y)
    out = {ref: {'auc': a_ref}}
    for nm in names:
        d = nest[nm]['_per_person']
        if 'look_pair' not in d:
            continue
        a = P.auc(d['look_pair'], y)
        out[nm] = {'auc': a, 'diff_twin_minus_entry': a_ref - a,
                   'se_boot': float(np.std(bref - _wauc_draws(y, d['look_pair'], W), ddof=1))}
    return out


def stage_paths(version, data, construction, capacity_id, log=print, stats=_no_stats, outer=P.N_OUTER, budgets=None,
                hp=None, capacity=None, ladder=None):
    """P1 (MODEL_FIT income, occupation, SEX as a training label): full-pool SEX-side teachers (cached), then the nested estimator
    over the PRC / HQ / SHUF strength chains and the ORD ladder with the selected construction, and the predeclared selection."""
    t0 = time.process_time()
    Z, lab, unit, names, row_sha = _prep(data)
    if 'sex' not in lab:
        raise PermissionError('this stage reads SEX as a training label')
    capacity = tuple(capacity or CAPACITIES[capacity_id])
    ladder = tuple(ladder or ORD_LADDERS[capacity_id])
    task = source_task(version, row_sha)
    ss = P.fit_teachers(Z, lab, unit, version, hp or P.HGB, ('sex', 'occ_s'), log=log)
    ss_sha = _cache_put(version, 'SEXSIDE', {'T': ss['T'], 'curve_min': ss['curve_min'], 'models': ss['models'], 'oof': ss['oof'],
                                             'row_id_sha256': row_sha})
    don = P.donor_index(Z, P.route_params('SHUF_S2', version=version)['shuf_tag'], len(Z))
    shuf_overlap = {'self_donor': int((don == np.arange(len(Z))).sum()), 'same_household_donor': int((unit[don] == unit).sum())}
    plan = _plan(capacity_id, ladder)
    nest = nested_c(Z, lab, unit, version, plan, construction, hp=hp, T_fixed={**task['T'], **ss['T']}, capacity=capacity,
                    outer=outer, log=log)
    B = budgets or P.BUDGETS
    sel, audit = {}, {}
    for fam, bud in (('PRC', B), ('HQ', {'S2': B['S2']}), ('SHUF', {'S2': B['S2']}), ('ORD', B)):
        cands = [n for n in plan if n.startswith(fam + '@')]
        chosen, aud = P.select_path(nest, lab, unit, cands, budgets=bud)
        sel[fam] = chosen
        audit[fam] = {c: {k: list(v) for k, v in a.items()} for c, a in aud.items()}
    br = brackets_for(nest, plan, sel)
    look = nested_lookup(nest, lab, unit, version, [n for n in plan if n != 'PRC_TWIN'])
    s2 = sel['PRC'].get('S2')
    f4 = None
    if s2 is not None and s2 in look:
        r = look[s2]
        f4 = {'entry': s2, 'diff_twin_minus_prc': r['diff_twin_minus_entry'], 'se_boot': r['se_boot'],
              'pass_2se': bool(r['diff_twin_minus_entry'] - 2 * r['se_boot'] > 0),
              'rule': 'nested held-out pair LOOKUP AUC of the PRC configuration selected at S2 below the twin by more than 2 '
                      'household-bootstrap SE (MODEL_FIT; FA_PRC_v1 falsifier 4; a D2 conjunct, not a stop)'}
    out = {'stage': 'P1_paths_and_selection', 'version': int(version), 'construction': construction, 'capacity_id': capacity_id,
           'capacity': list(capacity), 'selection': sel, 'audit_spend_over_twin': audit, 'budgets': B,
           'brackets': br, 'nested_lookup_pair': look, 'F4': f4,
           'rule': 'last candidate in path order such that it and every earlier candidate keep nested estimate + 2 household-'
                   'clustered SE <= budget on every metric (nrm.fa.prc.select_path)', 'ladder': [list(x) for x in ladder],
           'lambda_grid': list(P.LAMBDA_GRID),
           'nested_means': {n: {k: v for k, v in d.items() if k != '_per_person'} for n, d in nest.items()},
           'outer_folds': outer, 'T_fixed': {**task['T'], **ss['T']}, 'sexside_cache_sha256': ss_sha,
           'shuf_donor_overlap_full_pool': shuf_overlap,
           'verdict_note': 'selection only; verdicts use MODEL_SELECT spend measured by frc.evaluate and the reader grid, never these '
                           'nested MODEL_FIT estimates', 'cpu_s': round(time.process_time() - t0, 1), 'utc': _utc()}
    _write_once_json(_paths()[2] / f'SELECTION_V{int(version)}.json', out)
    stats(n_model_fits=2 * (P.N_INNER + 1) + 4 * outer, n_crossfit_fits=2 * P.N_XF + 4 * P.N_XF * outer)
    log(json.dumps({'stage': 'paths', 'version': int(version), 'selection': sel, 'cpu_s': out['cpu_s']}))
    return out


def _twin_book_for(decision, version, E):
    tw = decision['twin']
    if tw['source'] == 'nrm':
        return P._load_model(tw['config'], version)['book']
    return load_model(tw['config'], version)['book']


def oof_statistics(version, row_sha, Z, sexside=True):
    """Full-pool out-of-fold statistics of MODEL_FIT: the NRM1 task teachers (qI, qO) and, if cached, the P1 SEX-side teachers."""
    task = source_task(version, row_sha)
    E = {'qI': task['oof']['inc'], 'qO': task['oof']['occ'], 'Z': Z}
    ss = None
    if sexside:
        ss = _cache_get(version, 'SEXSIDE', row_sha)
        E.update(P.embed_oof({'oof': {'inc': E['qI'], 'occ': E['qO'], 'sex': ss['oof']['sex'], 'occ_s': ss['oof']['occ_s']}}))
    return E, task, ss


def _choice(cfg, spec, sel, brackets, ablation_lambda):
    """(family, choice, label) for one configuration id from the P1 selection record (label set when nothing is fitted)."""
    if spec['kind'] == 'ablation':
        if ablation_lambda is None:
            raise ValueError("an ablation needs the strongest release's lambda")
        return 'PRC', f'PRC@{float(ablation_lambda)}', None
    if spec.get('bracket'):
        fam = spec['bracket']
        br = (brackets or {}).get('brackets', {}).get(fam)
        if br is None or '@' not in str(br):
            return fam, None, str(br or 'NOT_ASSESSED')
        same = [b for b in sorted(sel[fam]) if sel[fam][b] == br]
        if same:
            return fam, None, f'ALIAS_OF_{fam}_{same[0]}'
        return fam, br, None
    fam = cfg.split('_')[0]
    choice = sel[fam].get(spec['budget'])
    if choice is None:
        return fam, None, 'NOT_FEASIBLE'
    earlier = [x for x in sorted(sel[fam]) if x < spec['budget'] and sel[fam][x] == choice]
    if earlier:
        return fam, None, f'ALIAS_OF_{fam}_{earlier[0]}'
    return fam, choice, None


EXPOSURE = {
    'PRC': '; MODEL_FIT SEX reaches the SEX propensity and the SEX-conditional occupation teacher (P1) and, through their out-of-fold '
           'predictions, the cell tables and the routing rule',
    'HQ': '; MODEL_FIT SEX reaches the P1 SEX-side teachers, whose out-of-fold predictions enter only as means within the twin task '
          'cells (routing is a function of the task posteriors; no SEX-side teacher is stored)',
    'SHUF': "; the P1 SEX propensity's out-of-fold predictions enter only as a keyed-hash donor copy (no person-level SEX "
            "information); the guard uses the person's own qO; the donor bank is stored in the model file",
    'ORD': '; no SEX reaches the release; the operating point was chosen by the P1 rule, whose subgroup rows used MODEL_FIT SEX '
           'labels as an evaluation of SEX-blind candidates'}


def stage_configs(version, config_ids, mf, role_features, save, card, decision, log=print, stats=_no_stats, lab=None,
                  ablation_lambda=None):
    """P2 / E4: the selected configurations from the cached full-pool statistics, with the decision's construction. Reads MODEL_FIT
    income and occupation only when the construction is A (decoders). SEX reaches PRC and the ablations through the P1 SEX-side
    teachers (runtime pi and qO0 / qO1), HQ only through out-of-fold means within the twin's task cells, SHUF only through the
    keyed-hash donor bank of out-of-fold pi; it never reaches ORD. HQ, SHUF and ORD store the task teachers only. NOT_FEASIBLE
    budgets are not fitted; aliases are never saved twice."""
    t0 = time.process_time()
    cons, cid = decision['construction'], decision['capacity_id']
    if cons == 'A' and lab is None:
        raise PermissionError('construction A needs the MODEL_FIT labels of its decoders')
    Z, _, unit, names, row_sha = _prep(mf)
    selrec = json.loads((_paths()[2] / f'SELECTION_V{int(version)}.json').read_text())
    if selrec['construction'] != cons or selrec['capacity_id'] != cid:
        raise PermissionError('the P1 selection record was made with another construction or capacity')
    sel, brackets = selrec['selection'], selrec.get('brackets')
    E, task, ss = oof_statistics(version, row_sha, Z)
    task_teachers = {'inc': task['models']['inc'], 'occ': task['models']['occ']}
    teachers = {**dict(task['models']), **dict(ss['models'])}
    T = {**dict(task['T']), **dict(ss['T'])}
    tb = _twin_book_for(decision, version, E)
    k0, l0 = P.route(E, tb)
    from ohn import fam_a as FA
    Fms = role_features['MODEL_SELECT']
    Ems = P.embed(teachers, FA.compact_po(Fms['X'], Fms['feature_names']))
    kt, lt = P.route(Ems, tb)
    out = {}
    for cfg in config_ids:
        spec = CONFIGS[cfg]
        if spec['kind'] == 'twin':
            raise ValueError('stage P2 fits selected configurations (twins are stage T2)')
        t1 = time.process_time()
        fam, choice, label = _choice(cfg, spec, sel, brackets, ablation_lambda)
        if label is not None:
            out[cfg] = label
            continue
        if fam == 'ORD':
            KI, KO = (int(x) for x in choice.split('@')[1].split('x'))
            book, k, l = P.twin_book(E, KI, KO, version, P.route_params('PRC_TWIN'))
            used = task_teachers
        else:
            Ec, rp, extra = chain_inputs(spec['nrm_cfg'], spec.get('route_override', {}), E, version)
            book, k, l = P.protected_book(Ec, tb, k0, l0, float(choice.split('@')[1]), rp)
            book = {**book, **extra}
            used = teachers if fam == 'PRC' else task_teachers
        tabs, info = tables(cons, book, k, l, E, lab)
        write_model(cfg, version, _payload(cfg, version, names, used, book, tabs, T, cons, decision['capacity'],
                                           {'selected': choice, 'route_override': spec.get('route_override', {})}))
        arrays = _release_all(cfg, version, role_features)
        obj = None
        if fam != 'ORD':
            Ec = entry_embedding(E, book)
            twinb = dict(book)
            twinb.update({'CI': tb['CI'], 'CO': tb['CO'], 'tilt': 0.0})
            P.set_tables(Ec['pi'], twinb, k0, l0)
            obj = {'fitted_objective': P.objective(Ec, book, k, l), 'twin_objective_same_strength': P.objective(Ec, twinb, k0, l0),
                   'fitted_pi_pair_mi': P.plugin_mi(k * book['CO'].shape[0] + l, E['pi']),
                   'twin_pi_pair_mi': P.plugin_mi(k0 * tb['CO'].shape[0] + l0, E['pi']),
                   'lambda': float(choice.split('@')[1]), 'tilt': float(book.get('tilt', 0.0)),
                   'note': 'objective on the statistics the chain optimized (SHUF: its shuffled-signal copy); pi-weighted plug-in MI '
                           'uses the true out-of-fold pi; estimated propensities and plug-in objectives are approximations, never '
                           'guarantees about other classifiers'}
        moved = None
        if book['CO'].shape == tb['CO'].shape and book['CI'].shape == tb['CI'].shape:
            km, lm = P.route(entry_embedding(Ems, book), book)
            moved = {'model_fit_oof_rerouted_vs_twin': float(((k != k0) | (l != l0)).mean()),
                     'model_select_rerouted_vs_twin': float(((km != kt) | (lm != lt)).mean()),
                     'model_select_income_rerouted': float((km != kt).mean()),
                     'model_select_occupation_rerouted': float((lm != lt).mean())}
        expo = ('MODEL_FIT income and occupation labels reach the shared task teachers (NRM1 stage 1)'
                + (' and the label-refit decoders' if cons == 'A' else '; the B1 tables use only out-of-fold teacher predictions')
                + EXPOSURE[fam])
        mf_cells = P.route(entry_embedding(P.embed(used, Z), book), book)
        rec = _record(card, spec['kind'], spec['what'], (0, 0),
                      ('MODEL_FIT: income, occupation (this unit; decoders)' if cons == 'A' else 'none in this unit')
                      + ('; SEX via the P1 cache' if fam != 'ORD' else ''), expo, len(Z), len(np.unique(unit)),
                      time.process_time() - t1,
                      {'selected': choice, 'family': fam, 'construction': cons, 'capacity_id': cid, 'capacity': decision['capacity'],
                       'occupied_cells_oof': P._occupied(k, l), 'tables_info': info, 'T_star': T, 'objective': obj,
                       'rerouted': moved, 'route_override': spec.get('route_override', {}), 'frc_embed': book.get('frc_embed'),
                       'distinct_vectors_equal_cells_model_fit': distinct_equals_cells(arrays['MODEL_FIT'], *mf_cells),
                       'reader_budget': {r: P.reader_budget(arrays[r]) for r in arrays}, 'twin': decision['twin'],
                       'stored_teachers': sorted(used),
                       'teacher_fits_shared_with': 'NRM1 stage-1 task teachers and the P1 SEX-side teachers'})
        save(FAMILY, cfg, version, arrays, rec)
        out[cfg] = 'SAVED'
    log(json.dumps({'stage': 'configs', 'version': int(version), 'result': out, 'cpu_s': round(time.process_time() - t0, 1)}))
    return out


def stage_check(version, data, config_ids, log=print, loader=None):
    """P3 (optimistic; never a credit input): household 5-fold cross-fitted LOOKUP on MODEL_FIT of the OUT-OF-FOLD-routed fit cells
    of each saved configuration (the partition its book reproduces on the cached out-of-fold statistics; routing was optimized on
    these people). SEX is read as an evaluation label only. The decision-relevant MODEL_FIT reader check is the P1 nested held-out
    LOOKUP (adjudication, challenger 2)."""
    Z, lab, unit, names, row_sha = _prep(data)
    try:
        E, _, _ = oof_statistics(version, row_sha, Z, sexside=True)
    except FileNotFoundError:
        E, _, _ = oof_statistics(version, row_sha, Z, sexside=False)
    out = {}
    for cfg in config_ids:
        M = (loader or load_model)(cfg, version)
        k, l = P.route(entry_embedding(E, M['book']), M['book'])
        v = P.views(k, l, M['book']['CO'].shape[0])
        out[cfg] = {'lookup_crossfit_model_fit_oof_cells': {vw: P.lookup_crossfit(v[vw], lab['sex'], unit, f'NRM_FA_check_S{int(version)}')
                                                            for vw in ('pair', 'income', 'occupation')},
                    'occupied_oof': P._occupied(k, l)}
    rec = {'stage': 'P3_check', 'version': int(version), 'configs': out, 'folds': 5, 'fold_tag': f'NRM_FA_check_S{int(version)}',
           'label': 'routing optimized on these people (optimistic); never a credit input',
           'reader': 'LOOKUP with Beta(1 * base rate) shrinkage on the out-of-fold-routed cells, tables from the other 4 household folds',
           'utc': _utc()}
    _write_once_json(_paths()[2] / f'CHECK1_V{int(version)}_{time.strftime("%H%M%S", time.gmtime())}.json', rec)
    log(json.dumps({'stage': 'check', 'version': int(version),
                    'pair_auc': {c: round(v['lookup_crossfit_model_fit_oof_cells']['pair']['auc'], 4) for c, v in out.items()}}))
    return rec


# ---------------------------------------------------------------- invented-data selftest (no real data, no labels)
def selftest():
    t0 = time.process_time()
    rng = np.random.default_rng(20261021)
    checks, info = {}, {}
    n = 2400
    X, names, Z, lab, unit = P._fake_population(n, rng)
    hp = {'lr': 0.1, 'leaves': 8, 'msl': 20, 'l2': 1.0, 'max_iter': 50}
    cap = (6, 16)
    lab_t = {k: v for k, v in lab.items() if k != 'sex'}
    # 1. fit_twin's A tables equal the source pipeline's decoders; B1 = one Lloyd codeword update of the final partition
    st_src = P.fit_pipeline(Z, lab_t, unit, 0, {'PRC_TWIN': ('twin',)}, hp=hp, capacity=cap)
    tw = fit_twin(Z, lab_t, unit, 0, cap, hp=hp)
    checks['A_equals_source_decoders'] = all(np.array_equal(tw['tables']['A'][t], st_src['dec']['PRC_TWIN'][t]) for t in ('I', 'O'))
    checks['B0_is_stored_prototype'] = all(np.array_equal(tw['tables']['B0'][t], tw['book']['C' + t]) for t in ('I', 'O'))
    upd = P.update_book(tw['E_fit'], dict(tw['book']), *tw['cells'])
    checks['B1_is_one_codeword_update_without_rerouting'] = all(np.array_equal(tw['tables']['B1'][t], upd['C' + t]) for t in ('I', 'O'))
    k0, l0 = tw['cells']
    man = np.zeros_like(tw['tables']['B1']['O'])
    for c in range(cap[1]):
        if (l0 == c).any():
            man[c] = tw['E_fit']['qO'][l0 == c].mean(0)
    checks['B1_occupation_equals_manual_oof_mean'] = bool(np.allclose(P.clipn(man[np.bincount(l0, minlength=cap[1]) > 0]),
                                                                      tw['tables']['B1']['O'][np.bincount(l0, minlength=cap[1]) > 0],
                                                                      rtol=0, atol=1e-15))
    # 2. the constructions never change the partition; distinct vectors = occupied cells
    E = P.embed(tw['models'], Z)
    rels = {c: emit(E, tw['book'], tw['tables'][c])[0] for c in CONSTRUCTIONS}
    kk, ll = P.route(E, tw['book'])
    checks['partition_identical_across_constructions'] = all(all(partition_identity(rels['A'], rels[c]).values()) for c in ('B0', 'B1'))
    checks['distinct_vectors_equal_cells'] = all(all(distinct_equals_cells(rels[c], kk, ll).values()) for c in CONSTRUCTIONS)
    perm = rng.permutation(cap[1])
    bad = {'income': rels['A']['income'], 'occupation': tw['tables']['A']['O'][perm[ll] % 8]}
    checks['partition_check_detects_a_changed_partition'] = not partition_identity(rels['A'], bad)['occupation']
    # 3. the nested comparison: A and CONT equal the source nested estimator bitwise; identical held-out cells
    src = P.nested(Z, lab_t, unit, 0, {'CONT': ('cont',), 'PRC_TWIN': ('twin',)}, hp=hp, T_fixed=tw['T'], capacity=cap, outer=3)
    store, cells, folds = nested_twin(Z, lab_t, unit, 0, cap, T_fixed=tw['T'], hp=hp, outer=3)
    checks['nested_A_and_CONT_equal_source'] = all(np.array_equal(src['PRC_TWIN']['_per_person'][key], store['A'][key]) and
                                                   np.array_equal(src['CONT']['_per_person'][key], store['CONT'][key])
                                                   for key in src['PRC_TWIN']['_per_person'])
    summ = nest_summary(store, unit, 0, cap, tw['T'], folds, outer=3)
    sp_src = P.spend(src, 'PRC_TWIN', 'CONT', lab_t, unit)
    checks['nested_spend_equals_source_spend'] = all(abs(sp_src[k][0] - summ['spend_over_cont']['A'][k][0]) == 0.0 and
                                                     abs(sp_src[k][1] - summ['spend_over_cont']['A'][k][1]) == 0.0 for k in sp_src)
    fake_gate = {'nested_means': {'CONT': {k: v for k, v in src['CONT'].items() if k != '_per_person'},
                                  'PRC_TWIN': {k: v for k, v in src['PRC_TWIN'].items() if k != '_per_person'}},
                 'spend_twin_over_cont': {k: list(v) for k, v in sp_src.items()}, 'gate': 'X', 'utc': 'x'}
    checks['gate_reproduction_detects_equality'] = gate_reproduction(summ, fake_gate)['reproduced']
    fake_gate['nested_means']['CONT']['occ_nll'] += 1e-9
    checks['gate_reproduction_detects_difference'] = not gate_reproduction(summ, fake_gate)['reproduced']
    info['nested_B1_minus_A_invented'] = summ['paired_differences']['B1_minus_A']['occ_nll']
    info['nested_share_removed_invented'] = summ['share_of_A_spend_over_cont_removed_by_B1']
    checks['paired_zero_for_identical'] = all(v[0] == 0.0 for v in paired(store, 'A', 'A', unit).values())
    # 4. the protected pipeline with construction A equals the source pipeline bitwise (all entry kinds)
    plan = {'PRC_TWIN': ('twin',), 'PRC_A': ('prc', 'PRC_S2', 0.5), 'PRC_B': ('prc', 'PRC_S2', 2.0),
            'HQ': ('prc', 'HQ_S2', 2.0), 'SHUF': ('prc', 'SHUF_S2', 2.0), 'ORD': ('ord', 3, 8)}
    s1 = P.fit_pipeline(Z, lab, unit, 0, plan, hp=hp, capacity=cap)
    s2 = fit_pipeline_c(Z, lab, unit, 0, plan, 'A', hp=hp, capacity=cap)
    same_src = [nm for nm in plan if nm != 'SHUF']                      # SHUF is changed by card FA_FRC_v1 (own-qO guard)
    checks['fit_pipeline_c_A_equals_source'] = all(
        np.array_equal(s1['dec'][nm][t], s2['dec'][nm][t]) and np.array_equal(s1['fit_cells'][nm][0], s2['fit_cells'][nm][0])
        and np.array_equal(s1['fit_cells'][nm][1], s2['fit_cells'][nm][1]) for nm in same_src for t in ('I', 'O'))
    # SHUF of this card: routing ignores the person's own pi / qO0 / qO1, its guard reads the person's own qO, it is rule-consistent
    bS = s2['books']['SHUF']
    Ef0 = s2['E_fit']
    checks['shuf_book_marks_mode'] = bS.get('frc_embed') == SHUF_MODE and bS.get('restrict') is None and \
        np.array_equal(bS['shuf_bank_pi'], Ef0['pi'])
    Es = entry_embedding(Ef0, bS)
    checks['shuf_guard_reads_own_qO'] = bool(np.array_equal(Es['qO0'], Ef0['qO']) and np.array_equal(Es['qO1'], Ef0['qO'])
                                             and not np.array_equal(Es['pi'], Ef0['pi']))
    prmS = np.random.default_rng(5).permutation(len(Ef0['pi']))
    Ep = dict(Ef0)
    Ep['pi'], Ep['qO0'], Ep['qO1'] = Ef0['pi'][prmS], Ef0['qO0'][prmS], Ef0['qO1'][prmS]
    checks['shuf_ignores_person_sex_statistics'] = all(np.array_equal(a, b) for a, b in zip(P.route(entry_embedding(Ef0, bS), bS),
                                                                                            P.route(entry_embedding(Ep, bS), bS)))
    kS, lS = P.route(entry_embedding(Ef0, bS), bS)
    checks['shuf_rule_consistent'] = bool(np.array_equal(kS, s2['fit_cells']['SHUF'][0]) and np.array_equal(lS, s2['fit_cells']['SHUF'][1]))
    # (challenger 12b) perturbing the outer-test labels leaves every construction's held-out vectors and cells unchanged; perturbing
    # all labels at fixed teachers leaves B0 and B1 unchanged
    ofo = P.household_folds(unit, 3, OUTER_TAG.format(v=0))
    lab2 = {k2: v.copy() for k2, v in lab_t.items()}
    te0 = np.flatnonzero(ofo == 0)
    lab2['occupation'][te0] = (lab2['occupation'][te0] + 1) % 5
    lab2['income'][te0] = 1 - lab2['income'][te0]
    _, cA, _, vA = nested_twin(Z, lab_t, unit, 0, cap, T_fixed=tw['T'], hp=hp, outer=3, keep_vectors=True)
    _, cB, _, vB = nested_twin(Z, lab2, unit, 0, cap, T_fixed=tw['T'], hp=hp, outer=3, keep_vectors=True)
    checks['outer_test_labels_do_not_reach_tables_or_cells'] = bool(
        np.array_equal(cA['k'][te0], cB['k'][te0]) and np.array_equal(cA['l'][te0], cB['l'][te0]) and
        all(np.array_equal(vA[c][task][te0], vB[c][task][te0]) for c in CONSTRUCTIONS for task in ('income', 'occupation')))
    labp = {k2: np.random.default_rng(11).permutation(v) for k2, v in lab_t.items()}
    checks['labels_never_reach_B0_B1'] = all(np.array_equal(tables(c, tw['book'], k0, l0, tw['E_fit'], labp)[0][tt],
                                                            tw['tables'][c][tt]) for c in ('B0', 'B1') for tt in ('I', 'O'))
    checks['A_and_B1_count_the_same_people'] = bool(tw['A_and_B1_count_the_same_people'])
    # (challenger 12c) at a converged Lloyd fixed point the stored prototype equals the out-of-fold cell mean (B1 = B0)
    bk, kk2, ll2 = P.lloyd(tw['E_fit'], dict(tw['book']), 2000)
    upd2 = P.update_book(tw['E_fit'], dict(bk), kk2, ll2)
    fixed = all(np.array_equal(a, b) for a, b in zip(P.route(tw['E_fit'], upd2), (kk2, ll2)))
    tb1 = tables('B1', bk, kk2, ll2, tw['E_fit'])[0]
    checks['fixed_point_B1_equals_B0'] = bool(fixed and all(np.allclose(tb1[tt], bk['C' + tt], rtol=0, atol=1e-12) for tt in ('I', 'O')))
    info['lloyd_fixed_point_reached'] = bool(fixed)
    s3 = fit_pipeline_c(Z, lab, unit, 0, plan, 'B1', hp=hp, capacity=cap)
    checks['B1_pipeline_same_partitions_as_A'] = all(np.array_equal(s2['fit_cells'][nm][0], s3['fit_cells'][nm][0]) and
                                                     np.array_equal(s2['fit_cells'][nm][1], s3['fit_cells'][nm][1]) for nm in plan)
    checks['B1_recomputed_on_protected_partition'] = all(
        np.array_equal(s3['dec'][nm]['O'], tables('B1', s3['books'][nm], *s3['fit_cells'][nm], s3['E_fit'])[0]['O']) for nm in plan)
    checks['nonzero_strength_changes_partition'] = bool((s3['fit_cells']['PRC_B'][1] != s3['fit_cells']['PRC_TWIN'][1]).any() or
                                                        (s3['fit_cells']['PRC_B'][0] != s3['fit_cells']['PRC_TWIN'][0]).any())
    plan_ab = {'PRC_TWIN': ('twin',), 'L': ('prc', 'PRC_S2', 2.0, {'w_pair': 0.0}), 'Q': ('prc', 'PRC_S2', 2.0, {'w_loc': 0.0})}
    s4 = fit_pipeline_c(Z, lab, unit, 0, plan_ab, 'B1', hp=hp, capacity=cap)
    checks['ablation_overrides_reach_the_chain'] = bool(s4['books']['L']['w_pair'] == 0.0 and s4['books']['Q']['w_loc'] == 0.0)
    n1 = P.nested(Z, lab, unit, 0, {'PRC_TWIN': ('twin',), 'P': ('prc', 'PRC_S2', 2.0)}, hp=hp, T_fixed=s1['T'], capacity=cap, outer=3)
    n2 = nested_c(Z, lab, unit, 0, {'PRC_TWIN': ('twin',), 'P': ('prc', 'PRC_S2', 2.0)}, 'A', hp=hp, T_fixed=s1['T'], capacity=cap,
                  outer=3)
    checks['nested_c_A_equals_source'] = all(np.array_equal(n1[nm]['_per_person'][key], n2[nm]['_per_person'][key])
                                             for nm in n1 for key in n1[nm]['_per_person'])
    checks['nested_c_records_rerouting'] = 'mean_heldout_rerouted_vs_twin' in n2['P']
    # 5. storage and the deployment interface (temporary roots; no real path touched)
    import tempfile
    saved = {}

    def fake_save(family, cfg, v, arrays, rec):
        need = ('mechanism_card', 'kind', 'description', 'n_model_fits', 'n_calibrator_fits', 'n_crossfit_fits', 'n_critic_fits',
                'roles_read', 'exposure', 'training_people', 'training_households', 'fit_cpu_s', 'randomized')
        if [k for k in need if k not in rec] or (family, cfg, v) in saved:
            raise AssertionError('record fields or append-only')
        saved[(family, cfg, v)] = ({r: {t: a.copy() for t, a in d.items()} for r, d in arrays.items()}, rec)

    with tempfile.TemporaryDirectory() as tmp:
        _ROOT.update(runs=f'{tmp}/frc_runs', cache=f'{tmp}/frc_cache', records=f'{tmp}/frc_records')
        P._ROOT.update(runs=f'{tmp}/nrm_runs', cache=f'{tmp}/nrm_cache', records=f'{tmp}/nrm_records')
        bak = (dict(SOURCE_TWIN), dict(CAPACITIES))
        try:
            quiet = lambda *a, **k: None
            X2 = P._fake_population(500, np.random.default_rng(9))[0]
            roles = {'MODEL_FIT': {'X': X, 'feature_names': names}, 'MODEL_SELECT': {'X': X2, 'feature_names': names}}
            data_t = {'X': X, 'feature_names': names, 'unit': unit, 'row_id': np.arange(n), 'income': lab['income'],
                      'occupation': lab['occupation']}
            mf = {k: v for k, v in data_t.items() if k not in ('income', 'occupation')}
            # a fake source twin made by the NRM1 stage-1 code itself, under temporary roots
            P.stage_twin(0, data_t, roles, lambda *a: None, 'card', hp=hp, capacity=cap, log=quiet)
            src_rel = lambda src, v: {r: P.predict(src, v, F['X'], F['feature_names']) for r, F in roles.items()}
            CAPACITIES['C64'] = cap
            SOURCE_TWIN.clear(); SOURCE_TWIN['C64'] = 'PRC_TWIN'
            rec = stage_twb(0, 'TWB_C64', mf, roles, fake_save, 'card', log=quiet, source_release=src_rel)
            arrB = saved[('FA', 'TWB_C64', 0)][0]
            arrA = src_rel('PRC_TWIN', 0)
            checks['twb_partition_identity_recorded'] = all(all(v.values()) for v in rec['partition_identity_with_saved_A_release'].values())
            checks['twb_distinct_vectors_equal_cells'] = all(rec['distinct_vectors_equal_cells_model_fit'].values())
            checks['twb_vectors_differ_from_A'] = not np.array_equal(arrB['MODEL_SELECT']['occupation'], arrA['MODEL_SELECT']['occupation'])
            checks['twb_predict_reproduces_saved'] = all(np.array_equal(predict('TWB_C64', 0, F['X'], F['feature_names'])[t], arrB[r][t])
                                                         for r, F in roles.items() for t in ('income', 'occupation'))
            checks['twb_reads_no_label'] = rec['roles_read'].startswith('none')
            try:
                stage_twb(0, 'TWB_C64', mf, roles, fake_save, 'card', log=quiet, source_release=src_rel)
                checks['twb_append_only'] = False
            except (PermissionError, AssertionError):
                checks['twb_append_only'] = True
            # P1 / P2 / P3 with loose budgets (selftest only) and a short ladder
            data_s = dict(data_t, sex=lab['sex'])
            loose = {b: {k: 1.0 for k in P.BUDGETS[b]} for b in P.BUDGETS}
            sel = stage_paths(0, data_s, 'B1', 'C64', log=quiet, outer=3, budgets=loose, hp=hp, capacity=cap,
                              ladder=((4, 12), (3, 8), (2, 6)))
            info['selection_loose_budgets'] = sel['selection']
            checks['paths_records_rerouting_and_lookup'] = all('auc_pair' in sel['nested_means'][f'PRC@{g}'] and
                                                               'mean_heldout_rerouted_vs_twin' in sel['nested_means'][f'PRC@{g}']
                                                               for g in P.LAMBDA_GRID)
            dec = {'construction': 'B1', 'capacity_id': 'C64', 'capacity': list(cap), 'twin': {'source': 'frc', 'config': 'TWB_C64'}}
            r2 = stage_configs(0, ['PRC_S1', 'PRC_S2', 'PRC_S3', 'ORD_S1', 'ORD_S2', 'ORD_S3', 'HQ_S2', 'SHUF_S2'], mf, roles,
                               fake_save, 'card', dec, log=quiet)
            info['stage_configs'] = r2
            got = [c for c, v in r2.items() if v == 'SAVED']
            checks['configs_saved_and_aliases_not_twice'] = len(got) >= 3 and all(
                v == 'SAVED' or v.startswith('ALIAS_OF_') or v == 'NOT_FEASIBLE' for v in r2.values())
            prc = [c for c in got if c.startswith('PRC_')]
            task_c, ss_c = P._cache_get(0, 'TASK'), _cache_get(0, 'SEXSIDE')
            Eo = {'qI': task_c['oof']['inc'], 'qO': task_c['oof']['occ'], 'Z': Z}
            Eo.update(P.embed_oof({'oof': {'inc': Eo['qI'], 'occ': Eo['qO'], 'sex': ss_c['oof']['sex'], 'occ_s': ss_c['oof']['occ_s']}}))
            ok_tab = True
            for c in got:
                M = load_model(c, 0)
                k, l = P.route(entry_embedding(Eo, M['book']), M['book'])
                ok_tab &= all(np.array_equal(M['dec'][tt], tables('B1', M['book'], k, l, Eo)[0][tt]) for tt in ('I', 'O'))
            checks['configs_B1_tables_recomputed_on_their_partition'] = ok_tab
            checks['configs_record_objective_and_rerouting'] = all(saved[('FA', c, 0)][1]['objective'] is not None and
                                                                   saved[('FA', c, 0)][1]['rerouted'] is not None for c in prc)
            checks['configs_ord_reads_no_sex_teacher'] = all(sorted(load_model(c, 0)['teachers']) == ['inc', 'occ']
                                                             for c in got if c.startswith('ORD_'))
            lam = float(saved[('FA', prc[-1], 0)][1]['selected'].split('@')[1]) if prc else 2.0
            r3 = stage_configs(0, ['ABL_LOC', 'ABL_PAIR'], mf, roles, fake_save, 'card', dec, log=quiet, ablation_lambda=lam)
            checks['ablations_saved_with_overrides'] = r3 == {'ABL_LOC': 'SAVED', 'ABL_PAIR': 'SAVED'} and \
                load_model('ABL_LOC', 0)['book']['w_pair'] == 0.0 and load_model('ABL_PAIR', 0)['book']['w_loc'] == 0.0
            chk = stage_check(0, {'X': X, 'feature_names': names, 'unit': unit, 'row_id': np.arange(n), 'sex': lab['sex']}, got,
                              log=quiet)
            checks['check_runs_on_oof_cells'] = all(np.isfinite(r['lookup_crossfit_model_fit_oof_cells'][vw]['auc'])
                                                    for r in chk['configs'].values() for vw in ('pair', 'income', 'occupation'))
            checks['paths_records_brackets_and_f4'] = ('brackets' in sel and set(sel['brackets']['brackets']) == {'HQ', 'SHUF', 'ORD'}
                                                       and sel['F4'] is not None and np.isfinite(sel['F4']['se_boot']))
            r4 = stage_configs(0, ['HQ_BR', 'SHUF_BR', 'ORD_BR'], mf, roles, fake_save, 'card', dec, log=quiet)
            info['bracket_configs'] = r4
            checks['bracket_configs_saved_or_labelled'] = all(v == 'SAVED' or v.startswith('ALIAS_OF_') or
                                                              v == 'CONTROL_DID_NOT_REACH_SPEND' for v in r4.values())
            checks['shuf_and_hq_store_task_teachers_only'] = all(sorted(load_model(c, 0)['teachers']) == ['inc', 'occ']
                                                                 for c in got if c.startswith(('SHUF_', 'HQ_')))
            try:
                stage_configs(0, ['PRC_S1'], mf, roles, fake_save, 'card', {**dec, 'construction': 'A'}, log=quiet)
                checks['configs_refuse_construction_mismatch'] = False
            except PermissionError:
                checks['configs_refuse_construction_mismatch'] = True
            pr = []
            for kw in ({'keys': {'role': 'x', 'row_id': np.arange(3)}}, {'draw': 1}):
                try:
                    predict('TWB_C64', 0, X[:3], names, **kw); pr.append(False)
                except PermissionError:
                    pr.append(True)
            try:
                predict('TWB_C64', 0, X[:3, ::-1], names[::-1]); pr.append(False)
            except PermissionError:
                pr.append(True)
            checks['deploy_refusals'] = all(pr) and len(pr) == 3
        finally:
            SOURCE_TWIN.clear(); SOURCE_TWIN.update(bak[0])
            CAPACITIES.clear(); CAPACITIES.update(bak[1])
            _ROOT.update(runs=None, cache=None, records=None)
            P._ROOT.update(runs=None, cache=None, records=None)
    info['cpu_s'] = round(time.process_time() - t0, 2)
    passed = all(bool(v) for v in checks.values())
    print(json.dumps({'selftest': 'PASS' if passed else 'FAIL', 'checks': {k: bool(v) for k, v in checks.items()}, 'info': info},
                     indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    return passed


# ---------------------------------------------------------------- real-data entry points (queue units; after the receipt)
PURPOSE = {
    'nest': 'FRC T1: nested fixed-routing comparison of the probability constructions A / B0 / B1 on MODEL_FIT income and '
            'occupation (no SEX), with the NRM1 gate and decoder reproduction',
    'twa': 'FRC T5: label-refit decoders of the prospective-capacity twin on MODEL_FIT income and occupation (no SEX)',
    'paths': 'FRC P1: SEX-side teachers and the nested strength-path selection on MODEL_FIT (SEX as a training label)',
    'cfg': 'FRC P2: label-refit decoders of the selected configurations on MODEL_FIT income and occupation (construction A only)',
    'check': 'FRC P3: household cross-fitted LOOKUP of saved FA releases on MODEL_FIT (SEX as an evaluation label only)',
}


def _real(fields, purpose, card):
    from frc import common as C                                     # refuses before the receipt and without the committed card
    return C.load('MODEL_FIT', fields, 'fit', purpose, card=card)


def _roles():
    from frc import common as C
    return {r: C.features(r) for r in C.ROLES6}


def run(cmd, version, card, capacity_id=None, configs=None, ablation_lambda=None):
    from frc import common as C
    from frc import release as RL
    base = ('features', 'income', 'occupation')
    if cmd == 'repro':
        return stage_repro(version, C.features('MODEL_FIT'), _roles())
    if cmd == 'nest':
        return stage_nest(version, capacity_id, _real(base, PURPOSE['nest'], card), stats=C.unit_stats)
    if cmd == 'twb':
        cfg = configs[0]
        lab = None
        if CONFIGS[cfg]['construction'] == 'A':
            d = _real(base, PURPOSE['twa'], card)
            lab = {k: np.asarray(d[k], dtype=np.int64) for k in ('income', 'occupation')}
        return stage_twb(version, cfg, C.features('MODEL_FIT'), _roles(), RL.save, card, stats=C.unit_stats, lab=lab)
    dec = protection_decision()
    if cmd == 'paths':
        return stage_paths(version, _real(base + ('sex',), PURPOSE['paths'], card), dec['construction'], dec['capacity_id'],
                           stats=C.unit_stats)
    if cmd == 'cfg':
        lab = None
        if dec['construction'] == 'A':
            d = _real(base, PURPOSE['cfg'], card)
            lab = {k: np.asarray(d[k], dtype=np.int64) for k in ('income', 'occupation')}
        return stage_configs(version, configs, C.features('MODEL_FIT'), _roles(), RL.save, card, dec, stats=C.unit_stats, lab=lab,
                             ablation_lambda=ablation_lambda)
    if cmd == 'check':
        return stage_check(version, _real(('features', 'sex'), PURPOSE['check'], card), configs)
    raise ValueError(cmd)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='frc.fa.route')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    for name in ('repro', 'nest', 'twb', 'paths', 'cfg', 'check'):
        p = sub.add_parser(name)
        p.add_argument('--version', type=int, required=True, choices=(0, 1))
        p.add_argument('--card', required=True)
        if name == 'nest':
            p.add_argument('--capacity', required=True, choices=tuple(CAPACITIES))
        if name in ('twb', 'cfg', 'check'):
            p.add_argument('--config', required=True, help='comma list of configuration ids')
        if name == 'cfg':
            p.add_argument('--ablation-lambda', type=float, default=None)
    a = ap.parse_args(argv)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if a.cmd == 'selftest':
            return 0 if selftest() else 1
        cfgs = [c.strip() for c in getattr(a, 'config', '').split(',') if c.strip()] if hasattr(a, 'config') else None
        if cfgs is not None and [c for c in cfgs if c not in CONFIGS]:
            raise SystemExit(f'bad --config {a.config}')
        res = run(a.cmd, a.version, a.card, capacity_id=getattr(a, 'capacity', None), configs=cfgs,
                  ablation_lambda=getattr(a, 'ablation_lambda', None))
        print(json.dumps(res if isinstance(res, dict) else {'result': str(res)}, default=float)[:4000])
        return 0


if __name__ == '__main__':
    sys.exit(main())
