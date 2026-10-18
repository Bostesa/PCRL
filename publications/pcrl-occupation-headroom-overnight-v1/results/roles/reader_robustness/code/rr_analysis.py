"""Reader-robustness analysis of the development reader grid (tasks 1-3 of READER_ROBUSTNESS_BRIEF.md).

LABEL-FREE: reads only the grid outputs written by the locked `ohn.devreaders run` (runs/GRID/<BANK>/S<s>/GRID.json, BOOT.npz),
which hold SCORE AUCs, VALIDATION AUCs, winners and household-bootstrap draws. No release, feature or label is opened here.
Development evidence on MODEL_SELECT (adaptively used); diagnostic only, never a selection input. Public-use ACS 2019 1-Year
PUMS Florida person records; readers estimate the recorded SEX column (sex0 = men, sex1 = women) from released probability vectors.

    python3 rr_analysis.py --bank REF_BANK --bank WITNESS_BANK [--bank ...] --tag <NAME>

Writes results/<campaign>/roles/reader_robustness/<TAG>__{STABILITY,WINNERS,PAIR_FAMILY,INCOME_NOISE,LEARNING_CURVES}.json/.csv
- STABILITY: every registered (candidate, comparator) pair: the frozen rule's label and spread (ohn.rules.stability, unchanged),
  cells at or below 0, per-cell conditional intervals b +- 2 x bootstrap SE (readers fixed), and a variance decomposition of the
  8 cells into a reader-seed part (k0 vs k1 on identical FIT / VALIDATION / SCORE sets) and a split part (R1 vs R2).
- WINNERS: per release and view, the VALIDATION winner of each full cell, its recipe family and access (pair / income-only /
  occupation-only), near ties, distinct winners, and the selection optimism (winner VALIDATION AUC - its SCORE AUC).
- PAIR_FAMILY: per (candidate, comparator, cell), the two pair winners' recipe families and the pair benefit.
- INCOME_NOISE: income-view wires grouped by identical grid entries; every pair of distinct income wires: per-cell differences,
  bootstrap SEs, spread components; pooled noise model and the probability that the frozen per-cell recipient point gate
  (every cell <= 0.010) fails when the true income-view difference is exactly 0 (Monte Carlo, stated assumptions).
- LEARNING_CURVES: per release and view, mean SCORE AUC by FIT fraction; per pair, mean benefit by fraction and the frozen
  rule's BENEFIT_SHRINKS_WITH_READER_DATA flag (ohn.rules.learning_curve_flag, unchanged).
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from ohn import common as C          # noqa: E402
from ohn import rules as RU          # noqa: E402

SEEDS = (0, 1)
CELLS = tuple(f'S{s}{r}k{k}' for s in SEEDS for r in ('R1', 'R2') for k in (0, 1))
VIEWS = ('income', 'occupation', 'pair')
FRACS = ('0.25', '0.5', '1.0')
OUT = C.RESULTS / 'roles' / 'reader_robustness'
GATE = 0.010
Z = 2.0
FAMILY = {'OUT_LR': 'LINEAR', 'LOG_LR': 'LINEAR', 'DEC_LR': 'LINEAR', 'CATI01': 'CATI', 'CATI1': 'CATI',
          'OUT_MLP': 'MLP', 'LOG_MLP': 'MLP', 'MLP64': 'MLP', 'MLP6432': 'MLP',
          'OUT_HGB': 'BOOSTING', 'CONF_HGB': 'BOOSTING', 'HGB15': 'BOOSTING', 'HGB31': 'BOOSTING',
          'ET5': 'EXTRA_TREES', 'ET25': 'EXTRA_TREES', 'LOOKUP': 'LOOKUP'}
SEED_DEPENDENT = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')


def access_recipe(view, winner):
    if winner.startswith('IGNORE_occupation__'):
        return 'income_only', winner.split('__', 1)[1]
    if winner.startswith('IGNORE_income__'):
        return 'occupation_only', winner.split('__', 1)[1]
    return view, winner


def cell_of(key):
    return key.split('@')[0], (key.split('@')[1] if '@' in key else '1.0')


# ---------------------------------------------------------------- loading and merging (identical wires are cached: equal values)
def load(banks):
    cells, boot, paired, problems = {}, {}, {}, []
    for b in banks:
        for s in SEEDS:
            d = C.RUNS / 'GRID' / b / f'S{s}'
            if not (d / 'GRID.json').is_file():
                continue
            g = json.loads((d / 'GRID.json').read_text())
            with np.load(d / 'BOOT.npz') as z:
                bt = {k: z[k].copy() for k in z.files}
            for ck, cres in g['cells'].items():
                for name, ent in cres.items():
                    for v, e in ent.items():
                        key = (ck, name, v)
                        if key in cells:
                            if abs(cells[key]['score_auc'] - e['score_auc']) > 1e-12:
                                problems.append(f'{key}: {b} score {e["score_auc"]} differs from earlier {cells[key]["score_auc"]}')
                            continue
                        cells[key] = dict(e, bank=b)
                        boot[key] = bt[f'{ck}__{name}__{v}']
            for cand, comps in g['paired'].items():
                for comp, views in comps.items():
                    if cand == comp:
                        continue
                    e = paired.setdefault((cand, comp), {v: {'cells': {}, 'cell_se': {}} for v in VIEWS})
                    for v in VIEWS:
                        for ck, val in views[v]['cells'].items():
                            e[v]['cells'].setdefault(ck, val); e[v]['cell_se'].setdefault(ck, views[v]['cell_se'][ck])
    return cells, boot, paired, problems


def members(cells):
    return sorted({k[1] for k in cells})


def full(cells, name, v):
    return {c: cells[(c, name, v)] for c in CELLS if (c, name, v) in cells}


# ---------------------------------------------------------------- task 2: sign stability
def components(vals):
    """vals: {cell: value} over the 8 cells. Reader-seed part: k0 vs k1 on identical FIT / VALIDATION / SCORE sets (4 df);
    split part: R1 vs R2 means within seed (2 df), net of the reader-seed share."""
    dk = [vals[f'S{s}{r}k0'] - vals[f'S{s}{r}k1'] for s in SEEDS for r in ('R1', 'R2')]
    var_k = float(np.mean(np.square(dk)) / 2.0)
    m = {(s, r): (vals[f'S{s}{r}k0'] + vals[f'S{s}{r}k1']) / 2 for s in SEEDS for r in ('R1', 'R2')}
    dr = [m[(s, 'R1')] - m[(s, 'R2')] for s in SEEDS]
    var_rep_means = float(np.mean(np.square(dr)) / 2.0)
    var_split = max(0.0, var_rep_means - var_k / 2.0)
    return {'sd_reader_seed_k0_vs_k1': math.sqrt(var_k), 'sd_split_R1_vs_R2': math.sqrt(var_split),
            'k0_minus_k1': dict(zip([f'S{s}{r}' for s in SEEDS for r in ('R1', 'R2')], dk)),
            'R1_minus_R2_seed_mean': dict(zip([f'S{s}' for s in SEEDS], dr)),
            'cells_identical_k0_k1': int(sum(abs(x) < 1e-12 for x in dk))}


def stability(paired):
    rules = RU.load_rules()
    out = {}
    for (cand, comp), e in sorted(paired.items()):
        if not all(c in e['pair']['cells'] for c in CELLS):
            present = [c for c in CELLS if c in e['pair']['cells']]
            vals = {c: e['pair']['cells'][c] for c in present}; ses = {c: e['pair']['cell_se'][c] for c in present}
            out[f'{cand}|{comp}'] = {'candidate': cand, 'comparator': comp, 'label': 'INCOMPLETE', 'cells_present': present,
                                     'descriptive_present_cells': {
                                         'mean': float(np.mean(list(vals.values()))) if vals else None,
                                         'min': float(min(vals.values())) if vals else None, 'max': float(max(vals.values())) if vals else None,
                                         'cells': vals, 'cell_se': ses, 'cells_at_or_below_0': [c for c in present if vals[c] <= 0],
                                         'cells_positive_conditional_interval': [c for c in present if vals[c] - Z * ses[c] > 0],
                                         'cells_negative_conditional_interval': [c for c in present if vals[c] + Z * ses[c] < 0],
                                         'k0_minus_k1': {c[:4]: vals[c] - vals[c[:4] + 'k1'] for c in present if c.endswith('k0') and c[:4] + 'k1' in vals},
                                         'per_seed_descriptive': {
                                             s: (lambda xs, ss: {'mean': float(np.mean(xs)),
                                                                 'sd_total_one_seed': float(math.sqrt(np.var(xs, ddof=1) + np.mean(np.square(ss)))),
                                                                 'flip_probability': float(0.5 * math.erfc((np.mean(xs) / math.sqrt(np.var(xs, ddof=1) + np.mean(np.square(ss)))) / math.sqrt(2)))
                                                                 if np.var(xs, ddof=1) + np.mean(np.square(ss)) > 0 else None})(
                                                 [vals[c] for c in present if c.startswith(f'S{s}')], [ses[c] for c in present if c.startswith(f'S{s}')])
                                             for s in SEEDS if sum(c.startswith(f'S{s}') for c in present) >= 2},
                                         'occupation_recipient_max': float(max(e['occupation']['cells'][c] for c in present)) if present else None,
                                         'income_recipient_max': float(max(e['income']['cells'][c] for c in present)) if present else None,
                                         'note': 'seed(s) missing: descriptive only; the frozen stability label needs all 8 cells'}}
            continue
        st = RU.stability({v: {'cells': {c: e[v]['cells'][c] for c in CELLS}, 'cell_se': {c: e[v]['cell_se'][c] for c in CELLS}}
                           for v in VIEWS}, rules, comparator=comp)
        b = st['benefit']
        vals = b['cells']; ses = b['cell_se']
        pos_int = [c for c in CELLS if vals[c] - Z * ses[c] > 0]
        neg_int = [c for c in CELLS if vals[c] + Z * ses[c] < 0]
        le0 = [c for c in CELLS if vals[c] <= 0]
        gt0 = [c for c in CELLS if vals[c] > 0]
        comp_ = components(vals)
        rec = {'candidate': cand, 'comparator': comp, 'label': b['label'], 'mean': b['mean'], 'min': b['min'], 'max': b['max'],
               'per_seed_mean': b['per_seed_mean'], 'sd_reader_split': b['sd_reader_split'], 'rms_cell_se': b['rms_cell_se'],
               'sd_total': b['sd_total'], 'lower_per_seed': b['lower_per_seed'],
               'predicted_flip_probability_per_seed': b['predicted_flip_probability_per_seed'],
               'cells': vals, 'cell_se': ses, 'cells_at_or_below_0': le0, 'cells_above_0': gt0,
               'cells_positive_conditional_interval': pos_int, 'cells_negative_conditional_interval': neg_int,
               'flag_positive_interval_with_sign_change': bool(pos_int and le0),
               'flag_opposite_sign_conditional_intervals': bool(pos_int and neg_int),
               'flag_sign_change': bool(le0 and gt0),
               'variance_components': comp_,
               'target_label': st['target']['label'],
               'recipient': {v: {k: st['recipient'][v].get(k) for k in ('label', 'point_pass', 'max', 'mean', 'cells_above_limit', 'per_seed_mean', 'sd_total')}
                             for v in ('income', 'occupation')}}
        out[f'{cand}|{comp}'] = rec
    return out


def pooled_stability(st, subset=None):
    """Pooled descriptive summary over pairs (pairs share releases, so these are descriptive, not independent)."""
    recs = [r for k, r in st.items() if r['label'] != 'INCOMPLETE' and (subset is None or subset(r))]
    if not recs:
        return {'n_pairs': 0}
    miss = tot = 0
    for r in recs:
        for c in CELLS:
            s = int(c[1])
            tot += 1
            miss += abs(r['cells'][c] - r['per_seed_mean'][s]) > Z * r['cell_se'][c]
    # the household splits R1 / R2 are the SAME for both release seeds (roles_of depends on household ids and R only), so a
    # split effect can repeat across seeds; reader seeds differ by release seed (k contrasts should not correlate)
    rc = np.array([[np.mean([r['cells'][f'S{s}R1k0'], r['cells'][f'S{s}R1k1']]) - np.mean([r['cells'][f'S{s}R2k0'], r['cells'][f'S{s}R2k1']])
                    for s in SEEDS] for r in recs])
    kc = np.array([[r['cells'][f'S{s}R1k0'] - r['cells'][f'S{s}R1k1'] for s in SEEDS] for r in recs])
    split_sign = []
    for r in recs:
        c = r['cells']
        r1 = [c[f'S{s}R1k{k}'] for s in SEEDS for k in (0, 1)]; r2 = [c[f'S{s}R2k{k}'] for s in SEEDS for k in (0, 1)]
        if (all(x > 0 for x in r1) and all(x <= 0 for x in r2)) or (all(x <= 0 for x in r1) and all(x > 0 for x in r2)):
            split_sign.append(f"{r['candidate']}|{r['comparator']}")
    shared = {'corr_R1_minus_R2_contrast_seed0_vs_seed1': float(np.corrcoef(rc[:, 0], rc[:, 1])[0, 1]) if len(recs) > 2 else None,
              'corr_k0_minus_k1_contrast_seed0_vs_seed1_R1': float(np.corrcoef(kc[:, 0], kc[:, 1])[0, 1]) if len(recs) > 2 else None,
              'share_same_sign_R1_minus_R2_contrast': float(np.mean(np.sign(rc[:, 0]) == np.sign(rc[:, 1]))),
              'pairs_sign_set_by_split_in_both_seeds': split_sign,
              'note': 'R1 / R2 household splits are shared by both release seeds (DEV_GRID.json: one household set per repetition); '
                      'pairs share releases, so correlations are descriptive'}
    med = lambda key: float(np.median([r[key] for r in recs]))            # noqa: E731
    medc = lambda key: float(np.median([r['variance_components'][key] for r in recs]))   # noqa: E731
    return {'n_pairs': len(recs), 'labels': {lab: sum(r['label'] == lab for r in recs) for lab in sorted({r['label'] for r in recs})},
            'median_sd_reader_split': med('sd_reader_split'), 'median_rms_cell_se': med('rms_cell_se'), 'median_sd_total': med('sd_total'),
            'median_sd_reader_seed_k0_vs_k1': medc('sd_reader_seed_k0_vs_k1'), 'median_sd_split_R1_vs_R2': medc('sd_split_R1_vs_R2'),
            'pairs_with_sign_change': sum(r['flag_sign_change'] for r in recs),
            'pairs_positive_interval_with_sign_change': sum(r['flag_positive_interval_with_sign_change'] for r in recs),
            'pairs_opposite_sign_intervals': sum(r['flag_opposite_sign_conditional_intervals'] for r in recs),
            'conditional_interval_miss_rate': miss / tot, 'split_shared_across_release_seeds': shared,
            'conditional_interval_miss_note': 'share of cells whose readers-fixed interval b +- 2 x bootstrap SE excludes the per-seed mean '
                                              'of the 4 matched cells (about 0.05 or less if the bootstrap SE captured all cell-to-cell variation)'}


# ---------------------------------------------------------------- task 3: winners, near ties, recipe families
def winners(cells):
    out = {}
    for name in members(cells):
        for v in VIEWS:
            f = full(cells, name, v)
            if not f:
                continue
            rows = []
            for c, e in f.items():
                acc, rec = access_recipe(v, e['winner'])
                rows.append({'cell': c, 'winner': e['winner'], 'access': acc, 'recipe': rec, 'family': FAMILY.get(rec, '?'),
                             'seed_dependent': rec in SEED_DEPENDENT, 'origin': e['origin'], 'near_tie': bool(e['near_tie']),
                             'gap_to_runner_up': e['gap_to_runner_up'], 'val_auc': e['val_auc'], 'score_auc': e['score_auc'],
                             'optimism': e['val_auc'] - e['score_auc']})
            n = len(rows)
            fam = [r['family'] for r in rows]
            modal = max(set(fam), key=fam.count)
            out[f'{name}|{v}'] = {'release': name, 'view': v, 'n_cells': n, 'cells': rows,
                                  'distinct_winners': sorted({r['winner'] for r in rows}),
                                  'n_distinct_winners': len({r['winner'] for r in rows}),
                                  'distinct_families': sorted(set(fam)), 'modal_family': modal,
                                  'cells_off_modal_family': sum(x != modal for x in fam),
                                  'access_counts': {a: sum(r['access'] == a for r in rows) for a in sorted({r['access'] for r in rows})},
                                  'near_ties': sum(r['near_tie'] for r in rows),
                                  'winner_changes_within_seed_rep': sum(
                                      1 for s in SEEDS for r in ('R1', 'R2')
                                      if (f'S{s}{r}k0' in f and f'S{s}{r}k1' in f and f[f'S{s}{r}k0']['winner'] != f[f'S{s}{r}k1']['winner'])),
                                  'winner_changes_between_reps_same_k': sum(
                                      1 for s in SEEDS for k in (0, 1)
                                      if (f'S{s}R1k{k}' in f and f'S{s}R2k{k}' in f and f[f'S{s}R1k{k}']['winner'] != f[f'S{s}R2k{k}']['winner'])),
                                  'mean_score_auc': float(np.mean([r['score_auc'] for r in rows])),
                                  'mean_val_auc': float(np.mean([r['val_auc'] for r in rows])),
                                  'mean_optimism': float(np.mean([r['optimism'] for r in rows])),
                                  'sd_score_auc_within_seed': float(math.sqrt(sum((r['score_auc'] - np.mean([q['score_auc'] for q in rows if q['cell'][:2] == r['cell'][:2]])) ** 2
                                                                                  for r in rows) / max(1, n - 2)))}
    return out


def pair_family(cells, paired):
    rows = []
    for (cand, comp) in sorted(paired):
        for c in CELLS:
            if (c, cand, 'pair') not in cells or (c, comp, 'pair') not in cells:
                continue
            a, b = cells[(c, cand, 'pair')], cells[(c, comp, 'pair')]
            acc_a, rec_a = access_recipe('pair', a['winner']); acc_b, rec_b = access_recipe('pair', b['winner'])
            rows.append({'candidate': cand, 'comparator': comp, 'cell': c, 'cand_winner': a['winner'], 'comp_winner': b['winner'],
                         'cand_family': FAMILY.get(rec_a, '?'), 'comp_family': FAMILY.get(rec_b, '?'), 'cand_access': acc_a,
                         'comp_access': acc_b, 'same_family': FAMILY.get(rec_a) == FAMILY.get(rec_b), 'same_recipe': rec_a == rec_b,
                         'same_access': acc_a == acc_b, 'benefit': b['score_auc'] - a['score_auc'],
                         'cand_near_tie': bool(a['near_tie']), 'comp_near_tie': bool(b['near_tie'])})
    summ = {}
    for (cand, comp) in sorted(paired):
        r = [x for x in rows if x['candidate'] == cand and x['comparator'] == comp]
        if not r:
            continue
        same = [x['benefit'] for x in r if x['same_family']]; diff = [x['benefit'] for x in r if not x['same_family']]
        summ[f'{cand}|{comp}'] = {'n_cells': len(r), 'cells_different_family': len(diff), 'cells_different_access': sum(not x['same_access'] for x in r),
                                  'mean_benefit_same_family': float(np.mean(same)) if same else None,
                                  'mean_benefit_different_family': float(np.mean(diff)) if diff else None}
    return rows, summ


# ---------------------------------------------------------------- task 3: near-chance income views
def income_noise(cells, boot, n_mc=200000, seed=20261018):
    groups = {}
    for name in members(cells):
        f = full(cells, name, 'income')
        if set(f) != set(CELLS):
            continue
        sig = tuple(round(f[c]['score_auc'], 12) for c in CELLS) + tuple(round(f[c]['val_auc'], 12) for c in CELLS)
        groups.setdefault(sig, []).append(name)
    reps = {tuple(v)[0]: v for v in groups.values()}
    pairs = {}
    all_dk, all_dr, all_se2, all_resid = [], [], [], []
    for a, b in itertools.combinations(sorted(reps), 2):
        d = {c: cells[(c, a, 'income')]['score_auc'] - cells[(c, b, 'income')]['score_auc'] for c in CELLS}
        se = {c: float(np.std(boot[(c, a, 'income')] - boot[(c, b, 'income')], ddof=1)) for c in CELLS}
        by = {s: [d[c] for c in CELLS if c.startswith(f'S{s}')] for s in SEEDS}
        dm = {s: float(np.mean(v)) for s, v in by.items()}
        v_rep = sum((x - dm[s]) ** 2 for s in SEEDS for x in by[s]) / 6.0
        comp = components(d)
        pairs[f'{a}|{b}'] = {'a': a, 'b': b, 'a_group': reps[a], 'b_group': reps[b], 'cells': d, 'cell_se': se,
                             'mean': float(np.mean(list(d.values()))), 'per_seed_mean': dm, 'sd_within_seed': math.sqrt(v_rep),
                             'rms_cell_se': math.sqrt(float(np.mean(np.square(list(se.values()))))), 'variance_components': comp,
                             'cells_above_gate_a_minus_b': sum(x > GATE for x in d.values()),
                             'cells_above_gate_b_minus_a': sum(-x > GATE for x in d.values()),
                             'range': float(max(d.values()) - min(d.values())),
                             'mean_auc_a': float(np.mean([cells[(c, a, 'income')]['score_auc'] for c in CELLS])),
                             'mean_auc_b': float(np.mean([cells[(c, b, 'income')]['score_auc'] for c in CELLS]))}
        all_dk += list(comp['k0_minus_k1'].values()); all_dr += list(comp['R1_minus_R2_seed_mean'].values())
        all_se2 += [x ** 2 for x in se.values()]
        all_resid += [x - dm[s] for s in SEEDS for x in by[s]]
    if not pairs:
        return {'groups': list(groups.values()), 'pairs': {}}
    # pooled components of a DIFFERENCE between two income wires (pairs share wires, so pooled values are descriptive)
    var_k = float(np.mean(np.square(all_dk)) / 2.0)
    var_repmean = float(np.mean(np.square(all_dr)) / 2.0)
    var_split = max(0.0, var_repmean - var_k / 2.0)
    pooled = {'sd_reader_seed': math.sqrt(var_k), 'sd_split': math.sqrt(var_split), 'rms_cell_se': math.sqrt(float(np.mean(all_se2))),
              'sd_within_seed_pooled': math.sqrt(float(np.sum(np.square(all_resid)) / (6 * len(pairs)))),
              'n_wire_pairs': len(pairs)}
    # Monte Carlo of the frozen per-cell point gate under a TRUE difference of exactly 0. Each release's income-view AUC in a cell =
    # truth + split effect (shared by k0 / k1 of the same seed and repetition) + reader-seed effect; per-release variances are half
    # of the pooled DIFFERENCE components (independent releases). The candidate's noise is shared by its comparisons.
    rng = np.random.default_rng(seed)
    ta, te = math.sqrt(var_split / 2.0), math.sqrt(var_k / 2.0)

    def noise(n):
        a = rng.normal(0, ta, size=(n, 4)); e = rng.normal(0, te, size=(n, 8))
        return np.repeat(a, 2, axis=1) + e                  # cells ordered S0R1k0, S0R1k1, S0R2k0, ... (k0 / k1 share a)
    cand = noise(n_mc)
    one = (cand - noise(n_mc) > GATE).any(axis=1).mean()
    five = np.zeros(n_mc, dtype=bool)
    for _ in range(5):
        five |= (cand - noise(n_mc) > GATE).any(axis=1)
    single_cell = float(1 - 0.5 * math.erfc(-(GATE / math.sqrt(var_split + var_k)) / math.sqrt(2)))
    # failure probability of the per-cell gate against ONE comparator when the true difference is delta (candidate minus comparator)
    curve = {}
    for delta in (-0.03, -0.02, -0.01, 0.0, 0.005):
        curve[f'{delta:+.3f}'] = float((delta + noise(n_mc) - noise(n_mc) > GATE).any(axis=1).mean())
    mc = {'assumptions': 'true income-view difference exactly 0 (curve: delta); normal noise; per-release split SD = sd_split / sqrt 2, '
                         'reader-seed SD = sd_reader_seed / sqrt 2; candidate noise shared across its comparisons; seeds and repetitions '
                         'independent (the R1 / R2 role sets overlap partly, which this ignores)',
          'p_single_cell_exceeds_gate': single_cell,
          'p_gate_fails_vs_one_comparator_8_cells': float(one),
          'p_gate_fails_vs_five_comparators_with_other_income_wires': float(five.mean()),
          'p_gate_fails_vs_one_comparator_by_true_delta': curve, 'n_mc': n_mc}
    # the H3 income wire (shared by SUCC, H3+* and every H3-anchored candidate) against the other income wires
    h3 = [k for k, p in pairs.items() if 'SUCC' in p['a_group'] + p['b_group']]
    h3rec = {}
    for k in h3:
        p = pairs[k]
        sgn = 1.0 if 'SUCC' in p['a_group'] else -1.0                  # orient as H3 minus the other wire
        other = p['b'] if sgn > 0 else p['a']
        h3rec[other] = {'mean_H3_minus_other': sgn * p['mean'], 'per_seed_mean': {s: sgn * v for s, v in p['per_seed_mean'].items()},
                        'cells_H3_minus_other': {c: sgn * v for c, v in p['cells'].items()},
                        'cells_above_gate': p['cells_above_gate_a_minus_b'] if sgn > 0 else p['cells_above_gate_b_minus_a'],
                        'sd_within_seed': p['sd_within_seed'], 'rms_cell_se': p['rms_cell_se'],
                        'sd_split': p['variance_components']['sd_split_R1_vs_R2'],
                        'sd_reader_seed': p['variance_components']['sd_reader_seed_k0_vs_k1']}
    # empirical check: wire pairs whose 8-cell mean income-view difference is below 0.005 in absolute value ("near-identical"
    # recovery); each pair gives two candidate / comparator orientations; does the per-cell gate (some cell > +0.010) fail?
    near = {k: p for k, p in pairs.items() if abs(p['mean']) < 0.005}
    orient = []
    for k, p in near.items():
        orient.append({'candidate': p['a'], 'comparator': p['b'], 'mean_diff': p['mean'], 'cells_above_gate': p['cells_above_gate_a_minus_b'],
                       'gate_fails': p['cells_above_gate_a_minus_b'] > 0})
        orient.append({'candidate': p['b'], 'comparator': p['a'], 'mean_diff': -p['mean'], 'cells_above_gate': p['cells_above_gate_b_minus_a'],
                       'gate_fails': p['cells_above_gate_b_minus_a'] > 0})
    empirical = {'definition': 'income-wire pairs with |8-cell mean difference| < 0.005, both orientations',
                 'n_wire_pairs': len(near), 'n_orientations': len(orient), 'n_gate_fails': sum(o['gate_fails'] for o in orient),
                 'orientations': orient}
    return {'groups': list(groups.values()), 'pairs': pairs, 'pooled_difference_components': pooled, 'gate_false_failure': mc,
            'gate_failures_near_identical_income_views': empirical, 'H3_income_wire_vs_others': h3rec}


# ---------------------------------------------------------------- task 1: learning curves
def learning_curves(cells, paired):
    rules = RU.load_rules()
    rel = {}
    for name in members(cells):
        for v in VIEWS:
            by = {}
            for (ck, n, vv), e in cells.items():
                if n != name or vv != v:
                    continue
                c, fr = cell_of(ck)
                by.setdefault(fr, {})[c] = e['score_auc']
            if not by:
                continue
            rel[f'{name}|{v}'] = {fr: {'mean': float(np.mean(list(x.values()))), 'n_cells': len(x),
                                       'per_seed_mean': {s: float(np.mean([x[c] for c in x if c.startswith(f'S{s}')]))
                                                         for s in SEEDS if any(c.startswith(f'S{s}') for c in x)}}
                                  for fr, x in sorted(by.items())}
    pairs = {}
    for (cand, comp), e in sorted(paired.items()):
        by = {}
        for ck, val in e['pair']['cells'].items():
            c, fr = cell_of(ck)
            by.setdefault(fr, {})[c] = val
        if set(by) != set(FRACS):
            continue
        curve = {fr: float(np.mean(list(x.values()))) for fr, x in by.items()}
        ncell = {fr: len(x) for fr, x in by.items()}
        flag = RU.learning_curve_flag({float(k): v for k, v in curve.items()}, rules)
        pairs[f'{cand}|{comp}'] = {'candidate': cand, 'comparator': comp, 'mean_pair_benefit': curve, 'n_cells': ncell,
                                   'per_seed': {fr: {s: float(np.mean([x[c] for c in x if c.startswith(f'S{s}')])) for s in SEEDS
                                                     if any(c.startswith(f'S{s}') for c in x)} for fr, x in by.items()},
                                   'cells_le0': {fr: sum(v <= 0 for v in x.values()) for fr, x in by.items()},
                                   'flag': flag['flag'], 'monotone_decreasing': flag.get('monotone_decreasing'),
                                   'full_minus_quarter': curve['1.0'] - curve['0.25']}
    return rel, pairs


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', action='append', required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args(argv)
    cells, boot, paired, problems = load(a.bank)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {'banks': a.bank, 'utc': C.now(), 'note': 'label-free analysis of development reader-grid outputs (MODEL_SELECT, adaptively '
            'used); diagnostic only, never a selection input; reader-relative', 'merge_problems': problems,
            'members': members(cells), 'n_pairs': len(paired)}
    st = stability(paired)
    C.write_json(OUT / f'{a.tag}__STABILITY.json', {'meta': meta, 'pooled_all_pairs': pooled_stability(st), 'pairs': st})
    write_csv(OUT / f'{a.tag}__STABILITY.csv', [
        {'candidate': r['candidate'], 'comparator': r['comparator'], 'label': r['label'],
         'n_cells': len(r.get('cells') or (r.get('descriptive_present_cells') or {}).get('cells') or {}),
         'mean': r.get('mean', (r.get('descriptive_present_cells') or {}).get('mean')),
         'min': r.get('min', (r.get('descriptive_present_cells') or {}).get('min')),
         'max': r.get('max', (r.get('descriptive_present_cells') or {}).get('max')),
         'mean_s0': (r.get('per_seed_mean') or {}).get(0), 'mean_s1': (r.get('per_seed_mean') or {}).get(1),
         'sd_reader_split': r.get('sd_reader_split'), 'rms_cell_se': r.get('rms_cell_se'), 'sd_total': r.get('sd_total'),
         'sd_reader_seed_k0k1': (r.get('variance_components') or {}).get('sd_reader_seed_k0_vs_k1'),
         'sd_split_R1R2': (r.get('variance_components') or {}).get('sd_split_R1_vs_R2'),
         'flip_p_s0': (r.get('predicted_flip_probability_per_seed') or {}).get(0), 'flip_p_s1': (r.get('predicted_flip_probability_per_seed') or {}).get(1),
         'cells_le0': ';'.join(r.get('cells_at_or_below_0', [])), 'cells_pos_interval': ';'.join(r.get('cells_positive_conditional_interval', [])),
         'cells_neg_interval': ';'.join(r.get('cells_negative_conditional_interval', [])),
         'pos_interval_with_sign_change': r.get('flag_positive_interval_with_sign_change'),
         'opposite_sign_intervals': r.get('flag_opposite_sign_conditional_intervals'), 'target_label': r.get('target_label'),
         'occ_recipient_max': ((r.get('recipient') or {}).get('occupation') or {}).get('max'),
         'inc_recipient_max': ((r.get('recipient') or {}).get('income') or {}).get('max')} for r in st.values()])
    w = winners(cells)
    C.write_json(OUT / f'{a.tag}__WINNERS.json', {'meta': meta, 'releases': w})
    write_csv(OUT / f'{a.tag}__WINNERS.csv', [{k: v for k, v in r.items() if k != 'cells'} for r in w.values()])
    rows, summ = pair_family(cells, paired)
    C.write_json(OUT / f'{a.tag}__PAIR_FAMILY.json', {'meta': meta, 'summary': summ, 'cells': rows})
    write_csv(OUT / f'{a.tag}__PAIR_FAMILY.csv', rows)
    inc = income_noise(cells, boot)
    C.write_json(OUT / f'{a.tag}__INCOME_NOISE.json', {'meta': meta, **inc})
    rel, lc = learning_curves(cells, paired)
    C.write_json(OUT / f'{a.tag}__LEARNING_CURVES.json', {'meta': meta, 'release_score_auc_by_fraction': rel, 'pair_benefit_by_fraction': lc})
    write_csv(OUT / f'{a.tag}__LEARNING_CURVES.csv', [
        {'candidate': r['candidate'], 'comparator': r['comparator'], 'b_0.25': r['mean_pair_benefit']['0.25'], 'b_0.5': r['mean_pair_benefit']['0.5'],
         'b_1.0': r['mean_pair_benefit']['1.0'], 'full_minus_quarter': r['full_minus_quarter'], 'flag': r['flag'],
         'monotone_decreasing': r['monotone_decreasing'], 'cells_le0_0.25': r['cells_le0']['0.25'], 'cells_le0_0.5': r['cells_le0']['0.5'],
         'cells_le0_1.0': r['cells_le0']['1.0'], 'n_cells_1.0': r['n_cells']['1.0']} for r in lc.values()])
    print(json.dumps({'members': len(meta['members']), 'pairs': len(st), 'merge_problems': len(problems),
                      'income_groups': len(inc['groups']), 'lc_pairs': len(lc)}, indent=1))


if __name__ == '__main__':
    main()
