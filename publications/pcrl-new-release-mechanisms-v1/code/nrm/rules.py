"""Frozen generic claims and selection rules of the PCRL new-release-mechanisms campaign (`pcrl_new_release_mechanisms_v1`;
statistical reviewer; written before any candidate outcome of this campaign).

Human-readable rules: results/pcrl_new_release_mechanisms_v1/roles/statistical_reviewer/RULES.md
Machine-readable constants: .../RULES.json (schema NRM1-rules-v1); `load_rules()` refuses a file that disagrees with this module.
Invented-data tests and known-law screen checks: `python3 -m nrm.test_rules` (writes .../SCREEN_CHECKS.json).

Pure functions on plain dicts / floats. Nothing here reads data or labels; the only files read are this campaign's RULES.json and,
through the imported `ohn.rules` / `tsz.rules` (reused unchanged for the reserve and the old every-cell diagnostics), their rule
files. Every rule is a DEVELOPMENT rule on adaptively used MODEL_SELECT records; no original limit changes.

API (versions are the ints 0 and 1; cells are named S<v>R<r>k<k>; grid entries as in nrm.devreaders `paired()`:
{'cells': {cell: x}, 'cell_se': {cell: s}[, 'cell_se_mc': {cell: m}]}):
  load_rules()                                 constants + '_sha256'
  point_eligibility(evals_by_version)          G1 core: 24 clauses + 4 guards at the point (original limits)
  reserve_status(evals_by_version)             ohn.rules.reserve unchanged (MC SE in quadrature for randomized releases)
  mc_cell(draw_values, draw_ses)               one grid cell of a randomized release: mean, household SE bound, MC SE
  aggregate(entry, cells='registered')         the aggregate estimator and its SE components (no label)
  benefit_screen(entry, threshold=0)           AGG_POSITIVE / ... (+ old STABLE_POSITIVE diagnostic, sign reversals)
  target_screen(entry)                         the 0.020 criterion on the aggregate (+ old DEV_GRID_TARGET diagnostic)
  recipient_screen(entry)                      per-version aggregate non-worsening (+ old every-cell diagnostic)
  comparison(grid_entry)                       pair benefit + both recipient views for one (candidate, comparator)
  alias_groups(wires)                          comparator views with identical wires (counted once)
  reader_stability(cells_by_release, a, b)     sign changes, winner changes, near ties, effective seeds, aliases once
  grid_validity(seeds, checks, minimum_set, distinct_inputs)   A-SC1 criterion, applied prospectively
  frontier_comparison(point, path)             family point vs its SEX-blind control path (bracketing rule)
  reserve_frontier(points, paired)             reserve-qualified frontier, symmetric for every method
  claims(record)                               G1-G7, useful-component label
  select_candidate(ledger)                     at most one development candidate, deterministic ties
"""
from __future__ import annotations
import functools
import hashlib
import json
import math
import re
from pathlib import Path

from ohn import rules as OR                      # the source reserve rule, reused unchanged
from tsz import rules as TR                      # the old every-cell rules, reused unchanged as diagnostics

RULES_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_new_release_mechanisms_v1' / 'roles'
              / 'statistical_reviewer' / 'RULES.json')
SCHEMA = 'NRM1-rules-v1'
OHN_RULE_SHA256 = 'e18f1c68f4646ce9d1395960ab1a364a80c12541b815a55e0ba06b65074cbbb3'
CAPS = {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005}
GUARD_CAP = 0.005
RECIPIENT_LIMIT = 0.010
TARGET = 0.020
K_RESERVE = 2.0
Z = 2.0
SE_FLOOR = 0.001
NEAR_TIE_WIDTH = 0.005
VAL_NEAR_TIE = 0.010
VERSIONS = (0, 1)
REG_REPS = ('R1', 'R2')
ROB_REPS = ('R3', 'R4')
REG_K = (0, 1)
ROB_K = (2, 3)
REGISTERED = tuple(f'S{v}{r}k{k}' for v in VERSIONS for r in REG_REPS for k in REG_K)
VIEWS = ('pair', 'income', 'occupation')
SINGLE_VIEWS = ('income', 'occupation')
INHERITED = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
BENCHMARK = INHERITED + ('W', 'TR100', 'TT100')
TARGET_REFS = ('NEURAL_RAW', 'H_LJ')
SEED_DEPENDENT = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
SEED_INVARIANT = ('OUT_LR', 'OUT_HGB', 'LOG_LR', 'DEC_LR', 'CONF_HGB', 'LOOKUP', 'HGB15', 'HGB31', 'CATI01', 'CATI1')
MODEL_SEED_BASE = 2026101205
FORBIDDEN_OFFSETS = (300, 500)
SECONDARY = ('income|acc_loss', 'income|brier', 'income|nll', 'occupation|acc_loss', 'occupation|brier', 'occupation|nll',
             'occupation|men_brier', 'occupation|women_brier')
TIER_ORDER = ('T2', 'T3', 'T4', 'T5')
NOVELTY_LABELS = ('NOT_CLAIMED', 'NOVELTY_REVIEW_PENDING', 'NOVEL_PER_SEPARATE_REVIEW', 'PRIOR_ART_FOUND')
SCOPE = ('Lower recovery for the declared development readers is not protection against every reader.',
         'MODEL_SELECT is adaptively used: development evidence, never confirmation.',
         'Versions share training rows; repetitions share households; reader recipes and overlapping splits are not independent '
         'datasets.',
         'A reserve-frontier result is not superiority over every historical method.',
         'Novelty is never inferred from performance.')

_cache = {}
_CELL = re.compile(r'S([01])(R[1-4])k([0-9])')


# ---------------------------------------------------------------- the registered rule file
def load_rules(path=RULES_PATH):
    """RULES.json with its SHA-256; refuses a schema or constant that disagrees with this module or the OHN1 source rule."""
    p = Path(path)
    if str(p) not in _cache:
        raw = p.read_bytes()
        r = json.loads(raw)
        if r.get('schema') != SCHEMA:
            raise ValueError(f'unexpected rules schema {r.get("schema")!r}')
        if OR.load_rules().get('_sha256') != OHN_RULE_SHA256:
            raise ValueError('the OHN1 source rule file differs from the registered one')
        ol, ag, sel = r['original_limits'], r['aggregate_screen'], r['selection']
        checks = {
            'caps': (ol['caps'], CAPS), 'guard cap': (ol['within_sex_guard_cap'], GUARD_CAP),
            'recipient allowance': (ol['recipient_allowance_auc'], RECIPIENT_LIMIT), 'target': (ol['material_target_auc'], TARGET),
            'target references': (tuple(ol['target_references']), TARGET_REFS),
            'k': (r['reserve']['multiplier_k'], K_RESERVE), 'k equals OHN1': (float(OR.load_rules()['reserve']['multiplier_k']), K_RESERVE),
            'z': (ag['z'], Z), 'se floor': (ag['se_floor_auc'], SE_FLOOR), 'frontier z': (r['frontier']['z'], Z),
            'registered cells': (tuple(r['cells']['registered']), REGISTERED),
            'benchmark': (tuple(r['references']['original_benchmark']), BENCHMARK),
            'inherited': (tuple(r['references']['inherited_six']), INHERITED),
            'seed dependent': (tuple(r['reader_stability']['seed_dependent_recipes']), SEED_DEPENDENT),
            'near tie width': (sel['near_tie_width'], NEAR_TIE_WIDTH), 'max candidates': (sel['max_candidates'], 1),
            'minimum tier': (sel['minimum_tier'], 'T2'),
            'secondary metrics': (tuple(r['frontier']['secondary_metrics']), SECONDARY),
        }
        bad = [k for k, (a, b) in checks.items() if a != b]
        if bad:
            raise ValueError(f'RULES.json disagrees with nrm.rules on {bad}')
        r['_sha256'] = hashlib.sha256(raw).hexdigest()
        _cache[str(p)] = r
    return _cache[str(p)]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _fin(*xs):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs)


def _vkey(d, v):
    if v in d:
        return d[v]
    if str(v) in d:
        return d[str(v)]
    raise KeyError(v)


def _has(d, v):
    return isinstance(d, dict) and (v in d or str(v) in d)


def parse_cell(name):
    """(version, repetition, reader seed) of a screen cell, or None (learning-curve '@' cells and unknown names)."""
    m = _CELL.fullmatch(str(name))
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None


def is_registered(name):
    return name in REGISTERED


# ---------------------------------------------------------------- G1 core and reserve
def _utility(e):
    return e['utility'] if isinstance(e, dict) and 'utility' in e else e


def point_eligibility(evals_by_version):
    """Original point limits: per version 12 clauses + 2 guards (inclusive), both versions -> 24 + 4. Draw-averaged values for
    randomized releases (as produced by nrm.evaluate); each draw's own point status is reported (WORST_DRAW diagnostic)."""
    per, failed, worst = {}, [], {}
    for v in VERSIONS:
        if not _has(evals_by_version, v):
            return {'label': 'INCOMPLETE', 'error': f'version {v} missing'}
        u = _utility(_vkey(evals_by_version, v))
        try:
            conds = OR.conditions(u, v)
        except (KeyError, ValueError, TypeError) as ex:
            return {'label': 'INCOMPLETE', 'error': f'version {v}: {ex}'}
        if not all(_fin(c['value']) for c in conds.values()):
            return {'label': 'INCOMPLETE', 'error': f'version {v}: non-finite value'}
        fv = [n for n, c in conds.items() if not c['value'] <= c['cap']]
        per[v] = {'point_pass': not fv, 'failed': fv}
        failed += [f'v{v}|{n}' for n in fv]
        draws = [c for c in list(u.get('clauses', [])) + list(u.get('guards', [])) if c.get('draw_values')]
        if draws:
            nd = len(draws[0]['draw_values'])
            worst[v] = [all(float(c['draw_values'][i]) <= float(c['cap']) for c in draws) for i in range(nd)]
    rec = {'label': 'POINT_ELIGIBLE' if not failed else 'POINT_INELIGIBLE', 'per_version': per, 'failed': failed,
           'n_conditions': 28}
    if worst:
        rec['worst_draw'] = {v: {'draws': len(w), 'draws_passing': sum(w), 'every_draw_passes': all(w)} for v, w in worst.items()}
    return rec


def reserve_status(evals_by_version, include_mc=True):
    """Whole-release reserve by ohn.rules.reserve unchanged (k = 2, H_LJ floor), applied identically to every method. For a
    randomized release the own SE of each condition is sqrt(se^2 + se_mc^2) before flooring (include_mc=True); the reserve
    without the MC term is reported beside it."""
    def strip_mc(e, add):
        u = json.loads(json.dumps(_utility(e), default=float))
        for key in ('clauses', 'guards'):
            for c in u[key]:
                if add and _fin(_num(c.get('se_mc'))):
                    c['se'] = math.sqrt(_num(c['se']) ** 2 + _num(c['se_mc']) ** 2)
        return u
    out = {}
    for add in ((True, False) if include_mc else (False,)):
        res = OR.reserve({v: strip_mc(_vkey(evals_by_version, v), add) for v in VERSIONS if _has(evals_by_version, v)})
        per = {v: min((-math.inf if c['ratio'] is None else c['ratio']) for c in res['per_seed'][v]['conditions'].values())
               for v in res['seeds']}
        out['with_mc' if add else 'without_mc'] = {
            'status': res['status'], 'qualified': res['status'] == OR.MET and sorted(per) == list(VERSIONS),
            'min_ratio_by_version': per, 'ratio_vector': res['ratio_vector'], 'binding': res['binding_condition'],
            'failed': res['failed']}
    main = out['with_mc'] if include_mc else out['without_mc']
    return {**main, 'without_mc': out['without_mc'],
            'note': 'development heuristic, not a cap or population guarantee; applied identically to candidates, controls and references'}


# ---------------------------------------------------------------- randomized cells
def mc_cell(draw_values, draw_ses):
    """One grid cell of a randomized release evaluated on D draws (each draw a separate one-draw-per-person release with its own
    readers): value = mean over draws; household SE = RMS of the draws' paired bootstrap SEs (upper bound: draws share SCORE
    households and weights); MC SE = SD over draws / sqrt(D) (None if D = 1: MC_SE_NOT_ESTIMABLE)."""
    xs = [_num(x) for x in draw_values]
    ss = [_num(s) for s in draw_ses]
    if not xs or len(xs) != len(ss) or not all(_fin(x) for x in xs + ss) or any(s < 0 for s in ss):
        raise ValueError('draw values and SEs must be equally long, finite and non-negative')
    d = len(xs)
    m = sum(xs) / d
    mc = math.sqrt(sum((x - m) ** 2 for x in xs) / (d - 1) / d) if d > 1 else None
    return {'value': m, 'se': math.sqrt(sum(s * s for s in ss) / d), 'se_mc': mc, 'draws': d,
            'flags': [] if d > 1 else ['MC_SE_NOT_ESTIMABLE']}


# ---------------------------------------------------------------- aggregate estimator
def _select_cells(entry, which):
    vals = (entry or {}).get('cells')
    ses = (entry or {}).get('cell_se')
    mcs = (entry or {}).get('cell_se_mc') or {}
    if not isinstance(vals, dict) or not isinstance(ses, dict):
        return None, 'cells or cell_se missing'
    names = [c for c in vals if parse_cell(c) is not None]
    if which == 'registered':
        names = [c for c in names if c in REGISTERED]
        if sorted(names) != sorted(REGISTERED):
            return None, f'registered cells missing: {sorted(set(REGISTERED) - set(names))}'
    elif which == 'robustness':
        names = [c for c in names if c not in REGISTERED]
    elif which != 'all':
        raise ValueError(f'unknown cell set {which!r}')
    if not names:
        return None, 'no cells'
    rec = {}
    for c in names:
        x, s, m = _num(vals[c]), _num(ses.get(c)), _num(mcs.get(c, 0.0))
        if mcs.get(c) is None:
            m = 0.0
        if not _fin(x, s, m) or s < 0 or m < 0:
            return None, f'cell {c}: non-finite value or invalid SE'
        rec[c] = (x, s, m)
    return rec, None


def aggregate(entry, cells='registered', z=Z):
    """The aggregate estimator over the chosen cells (no label). Blocks b = (version, repetition); m_b = mean of the block's
    reader-seed cells; x = mean of block means; per-version means. SE components: se_hh = RMS of the cell SEs with the cell
    weights of x (upper bound under any cross-cell correlation of household sampling), se_rs = sqrt(sum_b (m_b - x)^2 / n_b)
    (expectation >= the version / split / interaction / reader part of Var(x) in the crossed 2-version design), se_mc likewise
    from cell MC SEs; se_agg = max(sqrt(sum of squares), SE_FLOOR) unless the comparison is an exact alias (all values and SEs 0).
    Per version, the same quantities from that version's cells only."""
    sel, err = _select_cells(entry, cells)
    if err:
        return {'complete': False, 'error': err}
    blocks = {}
    for c, t in sel.items():
        v, r, k = parse_cell(c)
        blocks.setdefault((v, r), {})[k] = t
    if sorted({v for v, _ in blocks}) != list(VERSIONS):
        return {'complete': False, 'error': 'both versions are required'}

    def summarize(keys):
        nb = len(keys)
        mb = {b: sum(t[0] for t in blocks[b].values()) / len(blocks[b]) for b in keys}
        x = sum(mb.values()) / nb
        w = {(b, k): 1.0 / (nb * len(blocks[b])) for b in keys for k in blocks[b]}
        se_hh = math.sqrt(sum(w[(b, k)] * blocks[b][k][1] ** 2 for b in keys for k in blocks[b]))
        se_mc = math.sqrt(sum(w[(b, k)] * blocks[b][k][2] ** 2 for b in keys for k in blocks[b]))
        se_rs = math.sqrt(sum((m - x) ** 2 for m in mb.values()) / nb)
        raw = math.sqrt(se_hh ** 2 + se_rs ** 2 + se_mc ** 2)
        return {'mean': x, 'block_means': {f'S{b[0]}{b[1]}': m for b, m in sorted(mb.items())}, 'n_blocks': nb,
                'se_household': se_hh, 'se_reader_split': se_rs, 'se_mc': se_mc, 'se_raw': raw}
    allk = sorted(blocks)
    tot = summarize(allk)
    per = {v: summarize([b for b in allk if b[0] == v]) for v in VERSIONS}
    alias = all(t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0 for t in sel.values())
    for s in [tot] + list(per.values()):
        s['se_agg'] = 0.0 if alias else max(s['se_raw'], SE_FLOOR)
        s['floor_binds'] = (not alias) and s['se_raw'] < SE_FLOOR
        s['lower'], s['upper'] = s['mean'] - z * s['se_agg'], s['mean'] + z * s['se_agg']
    # distinct cells (seed-deterministic duplicates once), sign reversals
    kd = {f'S{b[0]}{b[1]}': len({t[0] for t in blocks[b].values()}) > 1 for b in allk if len(blocks[b]) > 1}
    distinct = [x for b in allk for x in sorted({t[0] for t in blocks[b].values()})]
    n_eff = len(distinct)
    sgn = (tot['mean'] > 0) - (tot['mean'] < 0)
    rev = sum(1 for x in distinct if ((x > 0) - (x < 0)) != sgn) if sgn else len(distinct)
    nk = sum(kd.values())
    qual = ('READER_SEEDS_EFFECTIVE_ALL_BLOCKS' if kd and nk == len(kd) else
            'READER_SEED_REPLICATION_ABSENT' if nk == 0 else 'READER_SEEDS_PARTLY_EFFECTIVE')
    return {'complete': True, 'cells_used': cells, 'n_cells': len(sel), 'alias': alias, **tot,
            'per_version': {v: {k: per[v][k] for k in ('mean', 'se_agg', 'se_household', 'se_reader_split', 'se_mc', 'lower', 'upper',
                                                       'block_means', 'floor_binds')} for v in VERSIONS},
            'n_eff_distinct_cells': n_eff, 'k_differs': kd, 'reader_seed_qualifier': qual,
            'sign_reversals': {'cells_opposite_to_mean': rev, 'distinct_cells': n_eff,
                               'blocks_opposite': sum(1 for m in tot['block_means'].values() if ((m > 0) - (m < 0)) != sgn),
                               'versions_opposite': sum(1 for v in VERSIONS if ((per[v]['mean'] > 0) - (per[v]['mean'] < 0)) != sgn)},
            'cells': {c: t[0] for c, t in sel.items()}, 'z': z}


def _old_cells(entry):
    vals = {c: (entry or {}).get('cells', {}).get(c) for c in REGISTERED}
    ses = {c: (entry or {}).get('cell_se', {}).get(c) for c in REGISTERED}
    return vals, ses


def benefit_screen(entry, threshold=0.0, cells='registered', z=Z):
    """Aggregate pair-benefit screen (b = comparator AUC - candidate AUC; positive favours the candidate). Resolved iff
    x - z se_agg > threshold and both version means > threshold; an exact alias is ALIAS_ZERO (never above any threshold >= 0).
    The old 8-cell STABLE_POSITIVE rule (tsz.rules.stability_label) is reported beside as a diagnostic."""
    a = aggregate(entry, cells, z)
    names = (('AGG_POSITIVE', 'AGG_POSITIVE_POINT_ONLY', 'AGG_NOT_POSITIVE') if threshold == 0 else
             ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY', 'TARGET_AGG_NOT_MET'))
    if not a['complete']:
        return {'label': 'INCOMPLETE', 'resolved': False, 'aggregate': a}
    if a['alias']:
        label = 'ALIAS_ZERO'
    elif a['lower'] > threshold and all(a['per_version'][v]['mean'] > threshold for v in VERSIONS):
        label = names[0]
    elif a['mean'] > threshold:
        label = names[1]
    else:
        label = names[2]
    rec = {'label': label, 'resolved': label == names[0], 'threshold': threshold, 'aggregate': a}
    if cells == 'registered':
        vals, ses = _old_cells(entry)
        old = TR.stability_label(vals, ses, threshold=threshold)
        rec['old_every_cell_rule'] = {'label': old['label'], 'stable': old.get('stable'),
                                      'cells_at_or_below': old.get('cells_at_or_below'), 'n_eff': old.get('n_eff')}
        if bool(old.get('stable')) != rec['resolved']:
            rec['flags'] = ['EVERY_CELL_RULE_DIVERGES']
    return rec


def target_screen(entry, z=Z):
    """The original 0.020 material criterion on the aggregate (TARGET_AGG_RESOLVED / _POINT_ONLY / _NOT_MET) with the old
    DEV_GRID_TARGET label beside it."""
    rec = benefit_screen(entry, threshold=TARGET, z=z)
    if rec['label'] in ('INCOMPLETE', 'ALIAS_ZERO'):
        rec['label'] = 'TARGET_AGG_NOT_MET'
    vals, ses = _old_cells(entry)
    rec['old_dev_grid_target'] = TR.target_label(vals, ses)['label'] if all(_fin(_num(x)) for x in vals.values()) \
        else 'DEV_GRID_TARGET_NOT_MET'
    return rec


def recipient_screen(entry, limit=RECIPIENT_LIMIT, cells='registered', z=Z):
    """Aggregate non-worsening for one single-output view (r = candidate AUC - comparator AUC). The original allowance is per
    version: gate = r_v <= limit for both versions at the aggregate point (one noisy cell cannot veto). Labels:
    RECIPIENT_AGG_RESOLVED (r_v + z se_v <= limit both), RECIPIENT_AGG_EDGE, RECIPIENT_AGG_POINT_FAILED,
    RECIPIENT_AGG_RESOLVED_WORSE (some r_v - z se_v > limit), ALIAS_ZERO (identical view: passes). Old every-cell gate beside."""
    a = aggregate(entry, cells, z)
    if not a['complete']:
        return {'label': 'INCOMPLETE', 'gate_pass': False, 'aggregate': a}
    pv = a['per_version']
    if a['alias']:
        label = 'ALIAS_ZERO'
    elif any(pv[v]['lower'] > limit for v in VERSIONS):
        label = 'RECIPIENT_AGG_RESOLVED_WORSE'
    elif not all(pv[v]['mean'] <= limit for v in VERSIONS):
        label = 'RECIPIENT_AGG_POINT_FAILED'
    elif all(pv[v]['upper'] <= limit for v in VERSIONS):
        label = 'RECIPIENT_AGG_RESOLVED'
    else:
        label = 'RECIPIENT_AGG_EDGE'
    gate = label in ('RECIPIENT_AGG_RESOLVED', 'RECIPIENT_AGG_EDGE', 'ALIAS_ZERO')
    rec = {'label': label, 'gate_pass': gate, 'limit': limit, 'aggregate': a,
           'cells_above_limit': sorted(c for c, x in a['cells'].items() if x > limit)}
    if cells == 'registered':
        vals, _ = _old_cells(entry)
        old = TR.recipient_gate(vals)
        rec['old_every_cell_rule'] = {'label': old['label'], 'pass': old['pass'], 'cells_above': old.get('cells_above')}
        if old['pass'] != gate:
            rec['flags'] = ['EVERY_CELL_RULE_DIVERGES']
    return rec


def comparison(grid_entry, cells='registered'):
    """One (candidate, comparator) entry {'pair': ..., 'income': ..., 'occupation': ...}: benefit screen, recipient screens."""
    rec = {'benefit': benefit_screen(grid_entry.get('pair'), cells=cells),
           'recipient': {v: recipient_screen(grid_entry.get(v), cells=cells) for v in SINGLE_VIEWS}}
    rec['benefit_positive'] = rec['benefit']['resolved']
    rec['recipient_gate_pass'] = all(rec['recipient'][v]['gate_pass'] for v in SINGLE_VIEWS)
    rec['flags'] = sorted({f for x in [rec['benefit']] + list(rec['recipient'].values()) for f in x.get('flags', [])})
    return rec


def robustness(grid_entry):
    """Robustness cells (R3 / R4 / k2 / k3) only: ROBUSTNESS_CONFIRMED (AGG_POSITIVE on the robustness cells and on all cells),
    ROBUSTNESS_NOT_CONFIRMED, NOT_RUN. Never changes a decision or a ranking."""
    rob = benefit_screen((grid_entry or {}).get('pair'), cells='robustness')
    if rob['label'] == 'INCOMPLETE':
        return {'label': 'NOT_RUN'}
    allc = benefit_screen((grid_entry or {}).get('pair'), cells='all')
    ok = rob['resolved'] and allc['resolved']
    return {'label': 'ROBUSTNESS_CONFIRMED' if ok else 'ROBUSTNESS_NOT_CONFIRMED', 'robustness_cells': rob, 'all_cells': allc}


# ---------------------------------------------------------------- aliases
def alias_groups(wires):
    """wires = {name: {v: {'income': sha, 'occupation': sha}}}. Per view, groups of names whose wires are identical on both
    versions (pair view: both tasks identical). Each group is one comparison, reported and counted once."""
    out = {}
    for view in VIEWS:
        by = {}
        for name, w in wires.items():
            try:
                key = tuple((_vkey(w, v)['income'] if view != 'occupation' else '', _vkey(w, v)['occupation'] if view != 'income'
                             else '') for v in VERSIONS)
            except (KeyError, TypeError):
                continue
            by.setdefault(key, []).append(name)
        out[view] = sorted(sorted(g) for g in by.values() if len(g) > 1)
    return out


def distinct_comparators(names, groups, view):
    """Comparator names with alias groups of the view collapsed to their first member (counted once)."""
    seen, out = set(), []
    for n in names:
        g = next((tuple(x) for x in groups.get(view, []) if n in x), (n,))
        if g not in seen:
            seen.add(g)
            out.append(g[0] if len(g) == 1 else '='.join(g))
    return out


# ---------------------------------------------------------------- reader stability
def _recipe(winner):
    return str(winner).split('__')[-1]


def reader_stability(cells_by_release, cand, comp, views=VIEWS, wires=None):
    """cells_by_release = {cell: {release: {view: {'winner', 'near_tie', 'score_auc', ...}}}} (GRID.json 'cells', both versions
    merged; '@' cells ignored). Per view: per-release winners, winner changes, near ties, seed-dependent winners, deterministic
    cells (both winners seed-invariant), per-cell differences (pair: comp - cand; single views: cand - comp), sign changes,
    effective reader seeds per block, n_eff; an exact alias (identical wires, `wires` as in alias_groups) is one zero comparison."""
    out = {}
    names = sorted(c for c in cells_by_release if parse_cell(c) is not None)
    for view in views:
        if wires is not None:
            same = cand in wires and comp in wires and bool(alias_groups({cand: wires[cand], comp: wires[comp]})[view])
            if same:
                out[view] = {'label': 'EXACT_ALIAS_COUNTED_ONCE', 'differences': 0.0}
                continue
        rec = {'cells': {}, 'winners': {cand: {}, comp: {}}, 'near_ties': {cand: 0, comp: 0}}
        try:
            for c in names:
                a, b = cells_by_release[c][cand][view], cells_by_release[c][comp][view]
                d = (b['score_auc'] - a['score_auc']) if view == 'pair' else (a['score_auc'] - b['score_auc'])
                det = _recipe(a['winner']) in SEED_INVARIANT and _recipe(b['winner']) in SEED_INVARIANT
                rec['cells'][c] = {'difference': d, 'deterministic': det}
                for n, e in ((cand, a), (comp, b)):
                    rec['winners'][n][c] = e['winner']
                    rec['near_ties'][n] += bool(e.get('near_tie'))
        except (KeyError, TypeError) as ex:
            out[view] = {'label': 'INCOMPLETE', 'error': repr(ex)}
            continue
        blocks = {}
        for c in names:
            v, r, k = parse_cell(c)
            blocks.setdefault(f'S{v}{r}', []).append(rec['cells'][c]['difference'])
        kd = {b: len(set(x)) > 1 for b, x in blocks.items() if len(x) > 1}
        dist = [x for xs in blocks.values() for x in sorted(set(xs))]
        mean = sum(rec['cells'][c]['difference'] for c in names) / len(names) if names else math.nan
        sgn = (mean > 0) - (mean < 0)
        thr = 0.0 if view == 'pair' else RECIPIENT_LIMIT
        rec.update({
            'mean': mean, 'n_eff_distinct_cells': len(dist), 'k_differs': kd,
            'reader_seed_qualifier': ('READER_SEEDS_EFFECTIVE_ALL_BLOCKS' if kd and all(kd.values()) else
                                      'READER_SEED_REPLICATION_ABSENT' if not any(kd.values()) else 'READER_SEEDS_PARTLY_EFFECTIVE'),
            'sign_changes': sum(1 for x in dist if ((x > 0) - (x < 0)) != sgn) if sgn else len(dist),
            'cells_beyond_threshold': (sum(1 for x in dist if x <= thr) if view == 'pair' else sum(1 for x in dist if x > thr)),
            'deterministic_cells': sum(1 for c in names if rec['cells'][c]['deterministic']),
            'winner_changes': {n: {'distinct_winners': sorted(set(w.values())), 'n_distinct': len(set(w.values())),
                                   'seed_dependent_winner_cells': sum(_recipe(x) in SEED_DEPENDENT for x in w.values())}
                               for n, w in rec['winners'].items()},
            'winner_differs_between_releases': sum(1 for c in names if rec['winners'][cand][c] != rec['winners'][comp][c])})
        rec['label'] = 'SUMMARY'
        out[view] = rec
    return out


def forbidden_seeds():
    return {MODEL_SEED_BASE + off + 10 * v + k for off in FORBIDDEN_OFFSETS for v in VERSIONS for k in range(10)}


def grid_validity(seeds, checks, minimum_set, distinct_inputs):
    """A-SC1 criterion, applied prospectively. seeds = {'S<v>k<k>': int}; checks = [{'recipe', 'view', 'wire', 'v', 'R',
    'equal_k0_k1': bool}]; minimum_set = wires that must show no seed-dependent equality (the selected candidate and its matched
    controls); distinct_inputs = {'<wire>|V<v>': {view: int}}. READER_SEEDS_EFFECTIVE iff (i) seeds recorded, distinct, not
    reused; (ii) no seed-dependent equality on the minimum set; (iii) every other seed-dependent equality explained."""
    errors = []
    keys = [f'S{v}k{k}' for v in VERSIONS for k in REG_K]          # the registered reader seeds must all be recorded
    sv = [(seeds or {}).get(k) for k in keys] + [x for k, x in (seeds or {}).items() if k not in keys]
    if any(not isinstance(x, int) or isinstance(x, bool) for x in sv) or len(set(sv)) != len(sv):
        errors.append(f'reader seeds must be recorded distinct integers, at least {keys}')
    elif set(sv) & forbidden_seeds():
        errors.append(f'reader seeds reuse earlier grid seeds {sorted(set(sv) & forbidden_seeds())}')
    seen = {c.get('recipe') for c in checks or []}
    if [r for r in SEED_DEPENDENT if r not in seen]:
        errors.append('seed-dependent recipes not checked')
    eq = [c for c in checks or [] if c.get('recipe') in SEED_DEPENDENT and c.get('equal_k0_k1') is not False]
    # "differs" needs a RECORDED check with equal_k0_k1 False; an unchecked recipe or view is not evidence (review finding 3)
    diff = {(c['wire'], c['view'], c['v'], c['R'], c['recipe']) for c in checks or []
            if c.get('recipe') in SEED_DEPENDENT and c.get('equal_k0_k1') is False}
    on_min = [c for c in eq if c['wire'] in set(minimum_set or ())]
    if on_min:
        errors.append(f'{len(on_min)} seed-dependent equalities on the minimum set')
    unexplained = []
    for c in eq:
        if c['wire'] in set(minimum_set or ()):
            continue
        a = any((c['wire'], c['view'], c['v'], c['R'], r) in diff for r in SEED_DEPENDENT if r != c['recipe'])
        b = any((c['wire'], w, c['v'], c['R'], c['recipe']) in diff for w in VIEWS if w != c['view'])
        n = ((distinct_inputs or {}).get(f"{c['wire']}|V{c['v']}") or {}).get(c['view'])
        if not (a and b and isinstance(n, int)):
            unexplained.append(c)
    if unexplained:
        errors.append(f'{len(unexplained)} unexplained seed-dependent equalities')
    nondet = [c for c in checks or [] if c.get('recipe') in SEED_INVARIANT and c.get('equal_k0_k1') is False]
    return {'label': 'READER_SEEDS_NOT_EFFECTIVE' if errors else 'READER_SEEDS_EFFECTIVE', 'effective': not errors,
            'errors': errors, 'explained_equalities': len(eq) - len(on_min) - len(unexplained),
            'flags': ['NONDETERMINISTIC_RECIPE'] if nondet else []}


# ---------------------------------------------------------------- frontier vs the SEX-blind control path
def _pooled(d):
    """{v: {'value', 'se'[, 'se_mc']}} -> pooled difference over versions; SE = mean of version SEs (correlation bound)."""
    if not all(_has(d, v) for v in VERSIONS):
        return None
    vals = [_num(_vkey(d, v).get('value')) for v in VERSIONS]
    ses = [math.sqrt(_num(_vkey(d, v).get('se')) ** 2 + (_num(_vkey(d, v).get('se_mc')) ** 2
                                                           if _fin(_num(_vkey(d, v).get('se_mc'))) else 0.0)) for v in VERSIONS]
    if not all(_fin(x) for x in vals + ses):
        return None
    return {'value': sum(vals) / 2, 'se': sum(ses) / 2, 'by_version': dict(zip(VERSIONS, vals))}


def _primary(costs):
    """Primary cost difference from paired costs {v: {'<task>|<metric>': {'value', 'se'[, 'se_mc']}}}: 'total|nll' if given,
    else income|nll + occupation|nll with SE bounded by the sum of SEs (flag COST_SE_BOUND)."""
    if not all(_has(costs, v) for v in VERSIONS):
        return None, []
    per, flags = {}, []
    for v in VERSIONS:
        c = _vkey(costs, v)
        if 'total|nll' in c:
            per[v] = c['total|nll']
        elif 'income|nll' in c and 'occupation|nll' in c:
            a, b = c['income|nll'], c['occupation|nll']
            per[v] = {'value': _num(a['value']) + _num(b['value']), 'se': _num(a['se']) + _num(b['se']),
                      'se_mc': (_num(a.get('se_mc', 0) or 0) + _num(b.get('se_mc', 0) or 0))}
            flags = ['COST_SE_BOUND']
        else:
            return None, []
    return _pooled(per), flags


def frontier_comparison(point, path, z=Z):
    """point = {'id', 'costs_vs': {control id: paired costs {v: {'<task>|<metric>': {'value', 'se'}}}} (loss(point) -
    loss(control); 'total|nll' or income|nll + occupation|nll), 'grid_vs': {control id: grid entry (pair view at least)}};
    path = [{'id', 'pair_auc': pooled mean}] the family's SEX-blind control path. Bracket Q* = the cheapest control whose pooled
    point cost is >= the point's (d <= 0). DOMINATES_CONTROL_FRONTIER iff Q* exists, the pair benefit vs Q* is AGG_POSITIVE and
    no secondary metric shows resolved extra spending vs Q*; SEPARATED_ON_PRIMARY_COST_ONLY if only the last fails;
    NOT_SEPARATED_FROM_TASK_SPENDING otherwise (reason given); interpolation reported, never gating."""
    if not path:
        return {'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'CONTROL_PATH_MISSING'}
    prim, flags = {}, set()
    for q in path:
        d, f = _primary((point.get('costs_vs') or {}).get(q['id']))
        if d is None:
            return {'label': 'INCOMPLETE', 'error': f"paired costs vs {q['id']} missing or malformed"}
        prim[q['id']] = d
        flags.update(f)
    above = sorted((q for q in path if prim[q['id']]['value'] <= 0), key=lambda q: -prim[q['id']]['value'])
    rec = {'primary_cost_differences': prim, 'flags': sorted(flags), 'z': z}
    if not above:
        rec.update({'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT'})
        return rec
    qs = above[0]                                         # the cheapest control that spends at least as much as the point
    ben = benefit_screen(((point.get('grid_vs') or {}).get(qs['id']) or {}).get('pair'))
    if ben['label'] == 'INCOMPLETE':
        return {**rec, 'label': 'INCOMPLETE', 'error': f"grid vs bracketing control {qs['id']} missing"}
    sec = {}
    for m in SECONDARY:                                   # every secondary metric is required, finite (review finding 1)
        per = {v: _vkey(point['costs_vs'][qs['id']], v).get(m) for v in VERSIONS}
        p = _pooled(per) if all(isinstance(x, dict) for x in per.values()) else None
        if p is None:
            return {**rec, 'label': 'INCOMPLETE', 'error': f"secondary paired cost {m} vs bracketing control {qs['id']} missing "
                                                           f"or non-finite"}
        sec[m] = {**p, 'resolved_more_spending': p['value'] - z * p['se'] > 0}
    more = sorted(m for m, x in sec.items() if x['resolved_more_spending'])
    rec.update({'bracketing_control': qs['id'], 'benefit_vs_bracket': {k: ben[k] for k in ('label', 'resolved')},
                'benefit_mean': ben['aggregate'].get('mean'), 'benefit_se': ben['aggregate'].get('se_agg'),
                'secondary': sec, 'secondary_more_spending': more})
    below = [q for q in path if prim[q['id']]['value'] > 0]
    if below:
        lo = min(below, key=lambda q: prim[q['id']]['value'])
        dlo, dhi = prim[lo['id']]['value'], prim[qs['id']]['value']
        lam = dlo / (dlo - dhi) if dlo != dhi else 0.0
        blo = benefit_screen(((point.get('grid_vs') or {}).get(lo['id']) or {}).get('pair'))
        if blo['label'] != 'INCOMPLETE' and ben['label'] != 'INCOMPLETE':
            m = (1 - lam) * blo['aggregate']['mean'] + lam * ben['aggregate']['mean']
            s = (1 - lam) * blo['aggregate']['se_agg'] + lam * ben['aggregate']['se_agg']
            rec['interpolation'] = {'lower_control': lo['id'], 'lambda': lam, 'benefit': m, 'se': s, 'note': 'reported, never gating'}
    if not ben['resolved']:
        rec.update({'label': 'NOT_SEPARATED_FROM_TASK_SPENDING', 'reason': 'NO_RESOLVED_BENEFIT_AT_BRACKETING_CONTROL'})
    elif more:
        rec['label'] = 'SEPARATED_ON_PRIMARY_COST_ONLY'
    else:
        rec['label'] = 'DOMINATES_CONTROL_FRONTIER'
    return rec


# ---------------------------------------------------------------- reserve-qualified frontier (symmetric)
def reserve_frontier(points, paired=None):
    """points = [{'id', 'kind' (candidate / control / reference), 'reserve_qualified': bool (reserve_status, identical rule),
    'pair_auc': pooled registered-cell mean pair AUC}]; paired = {(a, b): grid pair entry of a vs b} for paired claims.
    Pareto frontier of reserve-qualified points (lower pair AUC); a candidate's RESERVE_FRONTIER_IMPROVEMENT needs AGG_POSITIVE
    against every reserve-qualified control and reference present in `paired`. Failing references stay listed."""
    q = [p for p in points if p.get('reserve_qualified') is True and _fin(_num(p.get('pair_auc')))]

    def r_of(p):
        x = _num(p.get('min_ratio'))
        return x if _fin(x) else 0.0
    front = {p['id'] for p in q if not any(o['pair_auc'] <= p['pair_auc'] and r_of(o) >= r_of(p)
                                           and (o['pair_auc'] < p['pair_auc'] or r_of(o) > r_of(p)) for o in q)}
    out = []
    for p in points:
        rec = {'id': p['id'], 'kind': p.get('kind'), 'reserve_qualified': p.get('reserve_qualified') is True,
               'pair_auc': p.get('pair_auc'), 'min_ratio': p.get('min_ratio'), 'on_reserve_pareto_frontier': p['id'] in front}
        if p.get('kind') == 'candidate' and rec['reserve_qualified'] and paired is not None:
            others = [o['id'] for o in q if o['id'] != p['id'] and o.get('kind') in ('control', 'reference')]
            labs = {o: benefit_screen((paired.get((p['id'], o)) or {}).get('pair'))['label'] for o in others}
            rec['paired_vs_qualified'] = labs
            rec['label'] = ('RESERVE_FRONTIER_IMPROVEMENT' if others and all(x == 'AGG_POSITIVE' for x in labs.values())
                            else 'NO_RESERVE_FRONTIER_IMPROVEMENT')
        out.append(rec)
    return {'points': out, 'references_failing_reserve': [p['id'] for p in points if p.get('kind') == 'reference'
                                                           and p.get('reserve_qualified') is not True],
            'note': 'a reserve-frontier result is not superiority over every historical method; every point-eligible reference '
                    'stays in the original benchmark with or without the reserve'}


# ---------------------------------------------------------------- claims G1-G7
def _eligible_refs():
    pub = load_rules()['references']['published_labels']
    return tuple(r for r in BENCHMARK if pub[r]['point'] == 'POINT_ELIGIBLE')


def claims(rec):
    """rec = {'evals': {v: utility}, 'grid': {comparator: grid entry}, 'matched': {'twin': name, 'alternative': name},
    'frontier': frontier_comparison record or None, 'joint': None or {'outputs_changed': [...], 'jointly_designed': bool,
    'separate': name, 'income_first': name, 'occupation_first': name, 'costs_vs': {name: paired costs}}, 'novelty': label,
    'grid_valid': grid_validity(...)['effective'] (False -> NOT_COMPUTED_GRID_INVALID for G2-G6)}.
    Returns G1-G7 and the useful-component label; none implies another."""
    out = {'scope': list(SCOPE)}
    gv = rec.get('grid_valid')
    if gv is None or not bool(gv):                        # RULES.md section 6: validity must be recorded and true (finding 4)
        why = 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN' if gv is None else 'NOT_COMPUTED_GRID_INVALID'
        out.update({g: {'label': why} for g in ('G2', 'G3', 'G4', 'G5', 'G6')})
        pl = point_eligibility(rec.get('evals') or {})['label']
        out.update({'G1': {'label': {'POINT_ELIGIBLE': 'TASK_FEASIBLE', 'POINT_INELIGIBLE': 'TASK_INFEASIBLE'}.get(pl, 'INCOMPLETE')},
                    'G7': {'label': 'NOT_CLAIMED'},
                    'useful_component': None, 'tier': None, 'score': None, 'reserve': {'qualified': False}})
        return out
    g1 = point_eligibility(rec.get('evals') or {})
    feas = g1['label'] == 'POINT_ELIGIBLE'
    out['G1'] = {'label': 'TASK_FEASIBLE' if feas else ('INCOMPLETE' if g1['label'] == 'INCOMPLETE' else 'TASK_INFEASIBLE'),
                 'detail': g1}
    try:
        out['reserve'] = reserve_status(rec.get('evals') or {})
    except (KeyError, ValueError, TypeError) as ex:
        out['reserve'] = {'status': 'INCOMPLETE', 'qualified': False, 'error': str(ex)}
    grid = rec.get('grid') or {}
    comps = {name: comparison(e) for name, e in grid.items()}
    out['comparisons'] = comps

    def ok(name):
        c = comps.get(name)
        return c is not None and c['benefit_positive'] and c['recipient_gate_pass']
    # G2
    m = rec.get('matched') or {}
    if not m.get('twin') or not m.get('alternative') or m['twin'] not in comps or m['alternative'] not in comps:
        g2 = 'NO_MATCHED_CONTROLS'
    elif ok(m['twin']) and ok(m['alternative']):
        g2 = 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS' if feas else 'MET_WITHOUT_TASK_FEASIBILITY'
    else:
        g2 = 'NOT_MET'
    out['G2'] = {'label': g2, 'twin': m.get('twin'), 'alternative': m.get('alternative')}
    # G3
    fr = rec.get('frontier') or {}
    if fr.get('label') == 'DOMINATES_CONTROL_FRONTIER':
        g3 = 'STABLE_FRONTIER_IMPROVEMENT' if feas else 'MET_WITHOUT_TASK_FEASIBILITY'
    elif not fr:
        g3 = 'NOT_ASSESSED'
    else:
        g3 = fr.get('label', 'INCOMPLETE')
    out['G3'] = {'label': g3, 'frontier': fr or None}
    # G4
    refs = _eligible_refs()
    missing = [r for r in refs if r not in comps]
    fail = [r for r in refs if r in comps and not ok(r)]
    if missing:
        g4 = 'INCOMPLETE'
    elif fail:
        g4 = 'NOT_SUPERIOR'
    else:
        g4 = 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' if feas else 'MET_WITHOUT_TASK_FEASIBILITY'
    old = {r: {'stable': (comps[r]['benefit'].get('old_every_cell_rule') or {}).get('stable'),
               'recipient_every_cell': all((comps[r]['recipient'][v].get('old_every_cell_rule') or {}).get('pass') is True
                                           for v in SINGLE_VIEWS)} for r in refs if r in comps}
    out['G4'] = {'label': g4, 'references': list(refs), 'failing': fail, 'missing': missing,
                 'old_rule_profile': {'competitive_under_old_rules': bool(old) and not missing and
                                      all(x['stable'] and x['recipient_every_cell'] for x in old.values()), 'per_reference': old}}
    # G5
    tg = {r: target_screen((grid.get(r) or {}).get('pair')) for r in TARGET_REFS}
    labs = [tg[r]['label'] for r in TARGET_REFS]
    rgate = all(r in comps and comps[r]['recipient_gate_pass'] for r in TARGET_REFS)
    if all(x == 'TARGET_AGG_RESOLVED' for x in labs):
        level = 'MATERIAL_0020_RESOLVED' if rgate else 'TARGET_RESOLVED_RECIPIENT_GATE_FAILED'
    elif all(x in ('TARGET_AGG_RESOLVED', 'TARGET_AGG_POINT_ONLY') for x in labs):
        level = 'MATERIAL_0020_POINT_ONLY'
    elif all(r in comps and comps[r]['benefit_positive'] for r in TARGET_REFS):
        level = 'POSITIVE_BELOW_MATERIAL_TARGET'
    else:
        level = 'NOT_MET'
    g5 = level if (feas or level == 'NOT_MET') else 'MET_WITHOUT_TASK_FEASIBILITY'     # no protection label without G1 (finding 7)
    out['G5'] = {'label': g5, 'level': level, 'recipient_gate_vs_targets': rgate,
                 'per_reference': {r: {'label': tg[r]['label'], 'old': tg[r].get('old_dev_grid_target')} for r in TARGET_REFS}}
    # G6
    j = rec.get('joint')
    if not j:
        g6 = {'label': 'NOT_APPLICABLE_NOT_JOINT'}
    else:
        changed = sorted(set(j.get('outputs_changed') or []) & {'income', 'occupation'})     # set semantics (finding 6)
        if len(changed) < 2:
            g6 = {'label': 'NOT_APPLICABLE_ONE_OUTPUT_CHANGED', 'changed_outputs': changed,
                  'statement': f'only the {"/".join(changed) or "no"} output changes; no joint-design claim'}
        elif not j.get('jointly_designed'):
            g6 = {'label': 'NOT_APPLICABLE_NOT_JOINT', 'changed_outputs': changed,
                  'statement': 'both outputs change but they are not declared jointly designed; no joint-design claim'}
        else:
            need = [j.get('separate'), j.get('income_first'), j.get('occupation_first')]
            if not all(need) or any(n not in comps for n in need):
                g6 = {'label': 'NOT_ASSESSED_MISSING_CONTROLS', 'required': ['separate', 'income_first', 'occupation_first']}
            else:
                costs = {n: _primary((j.get('costs_vs') or {}).get(n))[0] for n in need}
                cheap = all(c is not None and not (c['value'] - Z * c['se'] > 0) for c in costs.values())
                good = all(ok(n) for n in need) and cheap
                g6 = {'label': ('JOINT_CONTRIBUTION_SUPPORTED' if feas else 'MET_WITHOUT_TASK_FEASIBILITY') if good
                      else 'NOT_SUPPORTED', 'per_comparison': {n: ok(n) for n in need}, 'primary_cost_not_resolved_higher': cheap}
    out['G6'] = g6
    nov = rec.get('novelty') or 'NOT_CLAIMED'
    out['G7'] = {'label': nov if nov in NOVELTY_LABELS else 'NOT_CLAIMED',
                 'note': 'from a separate prior-art review; never inferred from performance'}
    out['useful_component'] = ('USEFUL_COMPONENT_NOT_COMPETITIVE' if g2 == 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS'
                               and g4 != 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES' else None)
    tiers = []
    if feas and g2 == 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS':
        tiers.append('T2')
        if g3 == 'STABLE_FRONTIER_IMPROVEMENT':
            tiers.append('T3')
            if g4 == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES':
                tiers.append('T4')
                if g5 == 'MATERIAL_0020_RESOLVED':
                    tiers.append('T5')
    out['tier'] = tiers[-1] if tiers else None
    scores = [comps[r]['benefit']['aggregate'].get('mean') for r in refs if r in comps]
    out['score'] = min(scores) if scores and not missing and all(_fin(_num(s)) for s in scores) else None
    return out


# ---------------------------------------------------------------- selection (at most one development candidate)
def _leximin(a, b, tol=1e-9):
    for x, y in zip(a, b):
        if x == y or (math.isfinite(x) and math.isfinite(y) and abs(x - y) <= tol):
            continue
        return -1 if x > y else 1
    return 0


def _wires_equal(a, b):
    return isinstance(a, dict) and isinstance(b, dict) and all(_has(a, v) and _has(b, v) and _vkey(a, v) == _vkey(b, v)
                                                                 for v in VERSIONS)


def _components(x):
    """Fitted components for the tie-break: a number as is (0 stays 0), a per-version dict summed over both versions (a missing
    version gives inf), None or anything else inf (review finding 2)."""
    if isinstance(x, dict):
        vals = [_num(_vkey(x, v)) if _has(x, v) else math.inf for v in VERSIONS]
        return sum(vals) if all(not math.isnan(y) for y in vals) else math.inf
    y = _num(x)
    return y if _fin(y) else math.inf


def _entry(x):
    y = _num(x)
    return y if _fin(y) else math.inf


def select_candidate(ledger, reference_wires):
    """ledger = [{'config_id', 'family', 'kind' ('candidate' / 'control'), 'entry' (ledger order), 'n_fitted_components'
    (number or per-version dict), 'wire_sha256_eval': {v: {'income', 'occupation'}}, 'wire_sha256_grid': {v: {...}}, plus the
    claims() inputs incl. 'grid_valid'}]; reference_wires = {reference: {v: {'income', 'occupation'}}} (REQUIRED).
    Tiers T5 > T4 > T3 > T2 (minimum T2); within the best tier, score = min over the point-eligible references of the
    aggregate pair benefit; near-tie set (score >= best - 0.005); tie-breaks: reserve-qualified first, leximin whole-release
    reserve ratios, fewer fitted components, earliest ledger entry, config_id. Exact aliases (of a reference, by wire hash or by
    an ALIAS_ZERO pair comparison, or of an earlier configuration) and output mismatches are not selectable. A control may be
    chosen (flag CONTROL_DESIGNATED_CANDIDATE)."""
    if not isinstance(reference_wires, dict) or not all(r in reference_wires for r in BENCHMARK):
        raise ValueError(f'reference_wires must give the wires of every benchmark reference {BENCHMARK} (finding 5)')
    recs, firsts = [], {}
    for c in sorted(ledger, key=lambda x: (_entry(x.get('entry')), str(x.get('config_id')))):
        r = {'config_id': c.get('config_id'), 'family': c.get('family'), 'kind': c.get('kind'), 'entry': _entry(c.get('entry')),
             'n_fitted_components': _components(c.get('n_fitted_components')), 'flags': []}
        we, wg = c.get('wire_sha256_eval'), c.get('wire_sha256_grid')
        if not _wires_equal(we, wg):
            r.update({'status': 'NOT_SELECTABLE_OUTPUT_MISMATCH'})
            recs.append(r)
            continue
        key = json.dumps({str(v): _vkey(we, v) for v in VERSIONS}, sort_keys=True)
        ref = next((n for n, w in (reference_wires or {}).items() if _wires_equal(we, w)), None)
        if ref:
            r['status'] = f'ALIAS_OF_{ref}'
            recs.append(r)
            continue
        if key in firsts:
            r['status'] = f'ALIAS_OF_{firsts[key]}'
            recs.append(r)
            continue
        firsts[key] = c.get('config_id')
        cl = claims(c)
        zero = [n for n in BENCHMARK if (cl.get('comparisons') or {}).get(n, {}).get('benefit', {}).get('label') == 'ALIAS_ZERO']
        if zero:                                          # identical pair readout to a reference: an alias, whatever the hashes say
            r['status'] = f'ALIAS_OF_{zero[0]}'
            recs.append(r)
            continue
        res = cl['reserve']
        r.update({'claims': {k: cl[k]['label'] for k in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7')},
                  'useful_component': cl['useful_component'], 'tier': cl['tier'], 'score': cl['score'],
                  'reserve_qualified': bool(res.get('qualified')), 'ratio_vector': res.get('ratio_vector') or [-math.inf],
                  'status': ('BELOW_MINIMUM_TIER' if not cl['tier'] else 'SELECTABLE' if cl['score'] is not None
                             else 'NOT_SELECTABLE_INCOMPLETE_REFERENCE_GRID')})
        recs.append(r)
    sel = [r for r in recs if r['status'] == 'SELECTABLE']
    if not sel:
        return {'outcome': 'NO_DEVELOPMENT_CANDIDATE', 'selected': None, 'records': recs, 'rules_sha256': load_rules()['_sha256']}
    top = max(TIER_ORDER.index(r['tier']) for r in sel)
    tier = [r for r in sel if TIER_ORDER.index(r['tier']) == top]
    best = max(r['score'] for r in tier)
    near = [r for r in tier if r['score'] >= best - NEAR_TIE_WIDTH]

    def cmp(a, b):
        if a['reserve_qualified'] != b['reserve_qualified']:
            return -1 if a['reserve_qualified'] else 1
        x = _leximin(a['ratio_vector'], b['ratio_vector'])
        if x:
            return x
        fa, fb = a['n_fitted_components'], b['n_fitted_components']
        if fa != fb:
            return -1 if fa < fb else 1
        ea, eb = a['entry'], b['entry']
        if ea != eb:
            return -1 if ea < eb else 1
        return -1 if str(a['config_id']) < str(b['config_id']) else (1 if str(a['config_id']) > str(b['config_id']) else 0)
    near.sort(key=functools.cmp_to_key(cmp))
    choice = near[0]
    for r in sel:
        r['status'] = 'SELECTED_DEVELOPMENT_CANDIDATE' if r is choice else 'SELECTABLE_NOT_CHOSEN'
    if choice.get('kind') == 'control':
        choice['flags'].append('CONTROL_DESIGNATED_CANDIDATE')
    return {'outcome': 'DEVELOPMENT_CANDIDATE_SELECTED', 'selected': choice['config_id'], 'tier': choice['tier'],
            'near_tie_set': [r['config_id'] for r in near], 'best_score': best, 'records': recs,
            'rules_sha256': load_rules()['_sha256'],
            'note': 'selected from adaptively used MODEL_SELECT development evidence (winner selection inflates the chosen score); '
                    'labels are those of the actual selected outputs'}
