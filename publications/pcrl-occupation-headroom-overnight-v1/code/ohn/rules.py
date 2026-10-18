"""Selection rules of the PCRL overnight occupation-headroom campaign (statistical and baseline reviewer; written and registered
before any new model outcome of this campaign existed).

Human-readable: results/pcrl_occupation_headroom_overnight_v1/roles/statistical_reviewer/SELECTION_RULE.md
Machine-readable: .../RULES.json (schema OHN1-rules-v1; every threshold is read from it and checked against this module)
Claims: .../CLAIMS_DRAFT.md. Invented-data tests: ohn/test_rules.py.

Pure functions over the evaluator's in-memory structures; no label, no person-level array and no file other than RULES.json
is read:
  reserve(evals_by_seed, floor=None)       whole-release development reserve over 28 conditions per seed
  witness(evals_by_seed)                   stage-1 task-only occupation witness eligibility and ranking vector
  select_witnesses(cands)                  up to two witnesses, from utility evidence alone
  stability(grid_entry)                    one (configuration, comparator): benefit stability screen, recipient point and
                                           stability, development-grid 0.020 target
  income_branch_trigger(...)               the conditional income-branch trigger
  nominate(table)                          labels, separate outcomes and at most one whole-release finalist
Everything here is a DEVELOPMENT SELECTION HEURISTIC on adaptively used MODEL_SELECT data. No cap is changed and no population
guarantee is claimed.
"""
from __future__ import annotations
import functools
import hashlib
import json
import math
from pathlib import Path

RULES_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_occupation_headroom_overnight_v1' / 'roles'
              / 'statistical_reviewer' / 'RULES.json')
SCHEMA = 'OHN1-rules-v1'
CAPS = {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005}
GUARD_CAP = 0.005
RECIPIENT_LIMIT = 0.010
TARGET = 0.020
REFS = ('U0', 'Ucal')
METRICS = ('accuracy_loss', 'nll_excess', 'brier_excess')
INCOME = tuple(f'income|{r}|{m}' for r in REFS for m in METRICS)
OCCUPATION = tuple(f'occupation|{r}|{m}' for r in REFS for m in METRICS)
GUARDS = ('guard|sex0', 'guard|sex1')
OCC_SIDE = OCCUPATION + GUARDS
CONDITIONS = INCOME + OCC_SIDE
INCOME_ACCURACY = ('income|U0|accuracy_loss', 'income|Ucal|accuracy_loss')
SEEDS = (0, 1)
CELLS = tuple(f'S{s}R{r}k{k}' for s in SEEDS for r in (1, 2) for k in (0, 1))
VIEWS = ('income', 'occupation')
REQUIRED = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
TARGET_REFS = ('NEURAL_RAW', 'H_LJ')
TIE_TOL = 1e-9
MET, NOT_MET, ONE_SEED, PENDING = 'RESERVE_MET', 'RESERVE_NOT_MET', 'RESERVE_ONE_SEED_ONLY', 'RESERVE_PENDING_SECOND_SEED'

_cache = {}


# ---------------------------------------------------------------- the registered rule file
def load_rules(path=RULES_PATH):
    """RULES.json with its SHA-256; refuses a schema or constant that disagrees with this module."""
    p = Path(path)
    if str(p) not in _cache:
        raw = p.read_bytes()
        r = json.loads(raw)
        if r.get('schema') != SCHEMA:
            raise ValueError(f'unexpected rules schema {r.get("schema")!r}')
        checks = {
            'caps': (r['original_limits']['caps'], CAPS), 'guard cap': (r['original_limits']['within_sex_guard_cap'], GUARD_CAP),
            'recipient limit': (r['original_limits']['recipient_limit_auc'], RECIPIENT_LIMIT),
            'target': (r['original_limits']['material_target_auc'], TARGET),
            'required comparators': (tuple(r['comparators']['required']), REQUIRED),
            'target references': (tuple(r['comparators']['target_references']), TARGET_REFS),
            'cells': (tuple(r['stability']['cells']), CELLS),
            'income accuracy clauses': (tuple(r['income_branch']['trigger_clauses']), INCOME_ACCURACY),
        }
        bad = [k for k, (a, b) in checks.items() if a != b]
        if bad:
            raise ValueError(f'RULES.json disagrees with ohn.rules on {bad}')
        r['_sha256'] = hashlib.sha256(raw).hexdigest()
        _cache[str(p)] = r
    return _cache[str(p)]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _fin(*xs):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _utility(e):
    return e['utility'] if isinstance(e, dict) and 'utility' in e else e


def _seed_key(d, s):
    if s in d:
        return d[s]
    if str(s) in d:
        return d[str(s)]
    raise KeyError(f'seed {s} missing')


# ---------------------------------------------------------------- whole-release reserve
def conditions(e, seed):
    """Name -> {value, cap, se} for the 12 clauses and 2 guards of one seed; refuses malformed or re-capped input."""
    u = _utility(e)
    out = {}
    for c in u['clauses']:
        if 'seed' in c and int(c['seed']) != int(seed):
            raise ValueError(f'clause of seed {c["seed"]} passed as seed {seed}')
        name = f"{c['task']}|{c['reference']}|{c['metric']}"
        if c['task'] not in ('income', 'occupation') or c['reference'] not in REFS or c['metric'] not in CAPS:
            raise ValueError(f'unknown clause {name}')
        if name in out:
            raise ValueError(f'duplicate clause {name}')
        if _num(c['cap']) != CAPS[c['metric']]:
            raise ValueError(f'{name}: cap {c["cap"]} differs from the original cap (caps are never changed)')
        out[name] = {'value': _num(c['value']), 'cap': _num(c['cap']), 'se': _num(c.get('se'))}
    for g in u['guards']:
        if 'seed' in g and int(g['seed']) != int(seed):
            raise ValueError(f'guard of seed {g["seed"]} passed as seed {seed}')
        name = f"guard|{g['group']}"
        if name not in GUARDS or name in out:
            raise ValueError(f'unknown or duplicate guard {name}')
        if _num(g['cap']) != GUARD_CAP:
            raise ValueError(f'{name}: cap {g["cap"]} differs from the original guard cap')
        out[name] = {'value': _num(g['value']), 'cap': _num(g['cap']), 'se': _num(g.get('se'))}
    missing = [n for n in CONDITIONS if n not in out]
    if missing:
        raise ValueError(f'missing conditions {missing}')
    return out


def _condition(name, c, floor_se, k):
    value, cap, own = c['value'], c['cap'], c['se']
    point = bool(_fin(value) and value <= cap)
    if not _fin(floor_se) or floor_se <= 0:
        raise ValueError(f'{name}: floor SE must be finite and > 0 (got {floor_se!r}); a zero floor would void the rule')
    if _fin(own) and own < 0:
        raise ValueError(f'{name}: negative SE {own}')
    rec = {'condition': name, 'value': value, 'cap': cap, 'own_se': own, 'floor_se': floor_se, 'point_pass': point}
    if not _fin(value, own):
        rec.update({'margin': None, 'se_used': None, 'se_source': None, 'ratio': None, 'met': False, 'nonfinite': True,
                    'own_se_zero': False, 'met_with_own_se_only': False})
        return rec
    margin = cap - value
    se_used = max(own, floor_se)                       # floor_se > 0, so se_used > 0 even when own SE is 0 or tiny
    rec.update({'margin': margin, 'se_used': se_used, 'se_source': 'own' if own >= floor_se else 'floor',
                'required_margin': k * se_used, 'ratio': margin / se_used, 'met': bool(margin >= k * se_used),
                'nonfinite': False, 'own_se_zero': own == 0.0,
                'met_with_own_se_only': bool(own > 0 and margin >= k * own and not margin >= k * se_used)})
    return rec


def _ratio(r):
    return -math.inf if r['ratio'] is None else r['ratio']


def reserve(evals_by_seed, floor=None, rules=None):
    """Whole-release development reserve. evals_by_seed: {seed: utility dict (or a dict with 'utility')} for seeds 0 and/or 1.
    Per seed, all 28 conditions (12 income + 12 occupation clauses + 4 guards over 2 seeds) need margin = cap - value >=
    k * max(own SE, floor SE); k = 2; floor = H_LJ's MODEL_SELECT SE of the same condition and seed, rescaled by
    sqrt(floor households / households). Status: RESERVE_MET only if both seeds meet every condition."""
    rules = rules or load_rules()
    rr = rules['reserve']
    k = float(rr['multiplier_k'])
    floor = rr['floor']['se'] if floor is None else floor
    G0 = int(rr['floor']['households'])
    seeds = sorted({int(s) for s in evals_by_seed})
    if not seeds or any(s not in SEEDS for s in seeds):
        raise ValueError(f'seeds must be a non-empty subset of {SEEDS}, got {seeds}')
    per = {}
    for s in seeds:
        e = _seed_key(evals_by_seed, s)
        conds = conditions(e, s)
        G = int(_utility(e)['households'])
        if G <= 1:
            raise ValueError(f'seed {s}: invalid household count {G}')
        scale = math.sqrt(G0 / G)
        fl = _seed_key(floor, s)
        recs = {n: _condition(n, conds[n], _num(fl.get(n)) * scale, k) for n in CONDITIONS}
        failed = [n for n in CONDITIONS if not recs[n]['met']]
        per[s] = {'seed': s, 'status': NOT_MET if failed else MET, 'failed': failed, 'conditions': recs, 'households': G,
                  'floor_scale': scale, 'point_pass_all': all(r['point_pass'] for r in recs.values())}
    met = [s for s in seeds if per[s]['status'] == MET]
    if len(seeds) == 2:
        status = MET if len(met) == 2 else (ONE_SEED if met else NOT_MET)
    else:
        status = PENDING if met else NOT_MET
    allr = {f's{s}|{n}': per[s]['conditions'][n] for s in seeds for n in CONDITIONS}
    occ = {k_: v for k_, v in allr.items() if k_.split('|', 1)[1] in OCC_SIDE}
    inc = {k_: v for k_, v in allr.items() if k_.split('|', 1)[1] in INCOME}
    failed = [f's{s}|{n}' for s in seeds for n in per[s]['failed']]
    mn = min(allr, key=lambda x: _ratio(allr[x]))
    return {'rule': SCHEMA, 'rule_sha256': rules.get('_sha256'), 'status': status, 'meets_reserve': status == MET,
            'seeds': seeds, 'per_seed': per, 'failed': failed, 'multiplier_k': k,
            'min_ratio': _ratio(allr[mn]), 'binding_condition': mn,
            'min_occupation_side_ratio': min(_ratio(v) for v in occ.values()),
            'min_income_ratio': min(_ratio(v) for v in inc.values()),
            'ratio_vector': sorted(_ratio(v) for v in allr.values()),
            'occupation_ratio_vector': sorted(_ratio(v) for v in occ.values()),
            'point_pass_all': all(per[s]['point_pass_all'] for s in seeds),
            'income_accuracy_only_failure': bool(failed) and all(f.split('|', 1)[1] in INCOME_ACCURACY for f in failed),
            'own_se_zero': [k_ for k_, v in allr.items() if v.get('own_se_zero')],
            'met_with_own_se_only': [k_ for k_, v in allr.items() if v.get('met_with_own_se_only')],
            'note': 'development selection heuristic on adaptively used MODEL_SELECT data; caps unchanged; no population guarantee'}


def leximin_cmp(a, b, tol=TIE_TOL):
    """Compare two ascending ratio vectors: the larger first differing element (beyond tol) wins; returns -1 if a is better."""
    for x, y in zip(a, b):
        if x == y or (math.isfinite(x) and math.isfinite(y) and abs(x - y) <= tol):
            continue
        return -1 if x > y else 1
    return 0


# ---------------------------------------------------------------- stage 1: task-only occupation witnesses
def witness(evals_by_seed, rules=None, floor=None):
    """Witness eligibility from utility evidence alone. PROMISING_WITNESS needs: both seeds; all 28 conditions of the complete
    release pass at the point (original caps); every one of the 16 occupation-side conditions (12 occupation clauses + 4
    within-SEX guards) meets the reserve. No aggregate quantity can compensate for a failed condition."""
    res = reserve(evals_by_seed, floor, rules)
    per = res['per_seed']
    occ_point_fail = [f's{s}|{n}' for s in per for n in OCC_SIDE if not per[s]['conditions'][n]['point_pass']]
    inc_point_fail = [f's{s}|{n}' for s in per for n in INCOME if not per[s]['conditions'][n]['point_pass']]
    occ_res_fail = [f's{s}|{n}' for s in per for n in OCC_SIDE if not per[s]['conditions'][n]['met']]
    if res['seeds'] != [0, 1]:
        label = 'NOT_CONSIDERED_ONE_SEED_ONLY'
    elif occ_point_fail:
        label = 'OCCUPATION_POINT_INELIGIBLE'
    elif inc_point_fail:
        label = 'RELEASE_POINT_INELIGIBLE'
    elif occ_res_fail:
        label = 'OCCUPATION_RESERVE_NOT_MET'
    else:
        label = 'PROMISING_WITNESS'
    occ = {f's{s}|{n}': per[s]['conditions'][n] for s in per for n in OCC_SIDE}
    binding = sorted(occ, key=lambda x: _ratio(occ[x]))[:3]
    return {'label': label, 'promising': label == 'PROMISING_WITNESS', 'occupation_point_failed': occ_point_fail,
            'release_point_failed': inc_point_fail, 'occupation_reserve_failed': occ_res_fail,
            'min_occupation_side_ratio': res['min_occupation_side_ratio'], 'occupation_ratio_vector': res['occupation_ratio_vector'],
            'binding_occupation_conditions': binding, 'whole_release_reserve': res['status'],
            'whole_release_failed': res['failed'], 'full_constraints': {k: {f: v.get(f) for f in ('value', 'cap', 'own_se', 'se_used', 'margin', 'ratio', 'met', 'point_pass')}
                                                                       for k, v in {f's{s}|{n}': per[s]['conditions'][n] for s in per for n in CONDITIONS}.items()},
            'reserve': res}


def crossfit_gap_flag(deployed, fold_average, se_used):
    """Diagnostic only (never a gate): MODEL_SELECT occupation NLL or Brier of the deployed model minus the fold-average of the
    cross-fitted submodels; CROSSFIT_DEPLOYED_GAP when its absolute value exceeds the clause's se_used."""
    d, f, s = _num(deployed), _num(fold_average), _num(se_used)
    if not _fin(d, f, s) or s <= 0:
        return {'flag': 'CROSSFIT_CHECK_INCOMPLETE'}
    gap = d - f
    return {'flag': 'CROSSFIT_DEPLOYED_GAP' if abs(gap) > s else 'NO_CROSSFIT_GAP_FLAG', 'gap': gap, 'gap_se_units': gap / s}


def _simplicity(c):
    return int(c.get('n_model_fits') or 0) + int(c.get('n_calibrator_fits') or 0)


def select_witnesses(cands, rules=None, floor=None):
    """cands: {config_id: {'evals': {0: u, 1: u}, 'task_only': bool, 'n_model_fits', 'n_calibrator_fits', 'entry'}}.
    Only task-only configurations (no SEX in training) can be witnesses. Among PROMISING_WITNESS, rank by the occupation-side
    ratio vector (leximin: larger minimum first), then fewer fitted components, then the earliest ledger entry; keep the top
    max_witnesses (2)."""
    rules = rules or load_rules()
    n_max = int(rules['witness']['max_witnesses'])
    recs = {}
    for cid, c in cands.items():
        if not c.get('task_only'):
            recs[cid] = {'label': 'NOT_TASK_ONLY', 'promising': False}
            continue
        recs[cid] = witness(c['evals'], rules, floor)
    prom = [cid for cid in recs if recs[cid]['promising']]

    def cmp(a, b):
        r = leximin_cmp(recs[a]['occupation_ratio_vector'], recs[b]['occupation_ratio_vector'])
        if r:
            return r
        sa, sb = _simplicity(cands[a]), _simplicity(cands[b])
        if sa != sb:
            return -1 if sa < sb else 1
        ea, eb = cands[a].get('entry', math.inf), cands[b].get('entry', math.inf)
        return -1 if ea < eb else (1 if ea > eb else 0)
    order = sorted(prom, key=functools.cmp_to_key(cmp))
    chosen = order[:n_max]
    for cid in recs:
        if cid in chosen:
            recs[cid]['selection'] = 'SELECTED_WITNESS'
        elif cid in prom:
            recs[cid]['selection'] = 'PROMISING_NOT_SELECTED'
        else:
            recs[cid]['selection'] = 'NOT_A_WITNESS'
    return {'selected': chosen, 'ranking': order, 'records': recs, 'promising_exists': bool(prom)}


# ---------------------------------------------------------------- reader stability on the development grid
def _cell_values(view_entry, field):
    d = (view_entry or {}).get(field)
    if not isinstance(d, dict):
        return None, 'missing'
    if set(d) != set(CELLS):
        return None, f'cells differ from the registered 8 (missing {sorted(set(CELLS) - set(d))}, extra {sorted(set(d) - set(CELLS))})'
    vals = {c: _num(d[c]) for c in CELLS}
    if not all(_fin(v) for v in vals.values()):
        return None, 'non-finite cell value'
    if field == 'cell_se' and any(v < 0 for v in vals.values()):
        return None, 'negative cell SE'
    return vals, None


def _spread(vals, ses):
    """Per-seed means; pooled within-seed variance over the 8 cells (df 6); mean squared cell SE; total SD."""
    by = {s: [vals[c] for c in CELLS if c.startswith(f'S{s}')] for s in SEEDS}
    d = {s: sum(v) / len(v) for s, v in by.items()}
    v_rep = sum((x - d[s]) ** 2 for s in SEEDS for x in by[s]) / (len(CELLS) - len(SEEDS))
    v_samp = sum(ses[c] ** 2 for c in CELLS) / len(CELLS)
    return d, v_rep, v_samp, math.sqrt(v_rep + v_samp)


def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def benefit_summary(view_entry, z, threshold=0.0):
    """Pair benefit b = comparator AUC - candidate AUC per cell (positive favours the candidate)."""
    vals, err = _cell_values(view_entry, 'cells')
    ses, err2 = _cell_values(view_entry, 'cell_se')
    if err or err2:
        return {'label': 'INCOMPLETE', 'stable': False, 'all_cells_above': False, 'error': err or err2}
    d, v_rep, v_samp, sd = _spread(vals, ses)
    b = list(vals.values())
    mean = sum(b) / len(b)
    above = sum(x > threshold for x in b)
    stable_each = {s: d[s] - z * sd > threshold for s in SEEDS}
    all_above = above == len(CELLS)
    stable = all_above and all(stable_each.values())
    if stable:
        label = 'STABLE_POSITIVE' if threshold == 0 else 'STABLE_ABOVE_THRESHOLD'
    elif all_above:
        label = 'POSITIVE_NOT_STABLE' if threshold == 0 else 'ABOVE_THRESHOLD_NOT_STABLE'
    elif mean > threshold:
        label = 'SIGN_CHANGES' if threshold == 0 else 'THRESHOLD_CROSSINGS'
    else:
        label = 'NOT_POSITIVE' if threshold == 0 else 'BELOW_THRESHOLD'
    return {'label': label, 'stable': stable, 'all_cells_above': all_above, 'threshold': threshold, 'cells_above': above,
            'cells_at_or_below': len(CELLS) - above, 'mean': mean, 'min': min(b), 'max': max(b), 'per_seed_mean': d,
            'sd_reader_split': math.sqrt(v_rep), 'rms_cell_se': math.sqrt(v_samp), 'sd_total': sd,
            'lower_per_seed': {s: d[s] - z * sd for s in SEEDS},
            'predicted_flip_probability_per_seed': {s: (_phi(-(d[s] - threshold) / sd) if sd > 0 else (0.0 if d[s] > threshold else 1.0)) for s in SEEDS},
            'cells': vals, 'cell_se': ses}


def recipient_summary(view_entry, z, limit=RECIPIENT_LIMIT):
    """Recipient difference r = candidate AUC - comparator AUC per cell; point pass iff every cell <= limit."""
    vals, err = _cell_values(view_entry, 'cells')
    if err:
        return {'label': 'INCOMPLETE', 'point_pass': False, 'stable': False, 'error': err}
    r = list(vals.values())
    point = all(x <= limit for x in r)
    rec = {'point_pass': point, 'max': max(r), 'mean': sum(r) / len(r), 'cells_above_limit': sum(x > limit for x in r), 'cells': vals}
    ses, err2 = _cell_values(view_entry, 'cell_se')
    if err2:
        rec.update({'stable': None, 'label': 'RECIPIENT_POINT_PASS_STABILITY_NOT_COMPUTED' if point else 'RECIPIENT_POINT_FAILED',
                    'stability_error': err2})
        return rec
    d, v_rep, v_samp, sd = _spread(vals, ses)
    stable = point and all(d[s] + z * sd <= limit for s in SEEDS)
    rec.update({'stable': stable, 'per_seed_mean': d, 'sd_total': sd, 'upper_per_seed': {s: d[s] + z * sd for s in SEEDS},
                'label': 'RECIPIENT_STABLE' if stable else ('RECIPIENT_EDGE' if point else 'RECIPIENT_POINT_FAILED')})
    return rec


def stability(grid_entry, rules=None, comparator=None):
    """One (configuration, comparator) grid entry: {'pair': {'cells', 'cell_se'}, 'income': {...}, 'occupation': {...}}."""
    rules = rules or load_rules()
    z = float(rules['stability']['z'])
    ben = benefit_summary(grid_entry.get('pair'), z, 0.0)
    tgt = benefit_summary(grid_entry.get('pair'), z, TARGET)
    if tgt.get('stable'):
        tlab = 'DEV_GRID_TARGET_STABLE'
    elif tgt.get('label') != 'INCOMPLETE' and tgt['mean'] > TARGET:
        tlab = 'DEV_GRID_TARGET_POINT_ONLY'
    else:
        tlab = 'DEV_GRID_TARGET_NOT_MET'
    rec = {v: recipient_summary(grid_entry.get(v), z) for v in VIEWS}
    diag = {k: v for k, v in grid_entry.items() if k not in ('pair',) + VIEWS}
    return {'comparator': comparator, 'benefit': ben, 'benefit_stable': ben['stable'], 'target': {'label': tlab, 'detail': tgt},
            'recipient': rec, 'recipient_point_pass': all(rec[v]['point_pass'] for v in VIEWS),
            'recipient_stable': all(rec[v].get('stable') is True for v in VIEWS), 'diagnostics': diag}


def learning_curve_flag(mean_benefit_by_fraction, rules=None):
    """Diagnostic only: {0.25: b, 0.5: b, 1.0: b} mean pair benefits at nested FIT sizes. Flag when the full-size benefit is
    below the quarter-size benefit by more than the near-tie width (benefit shrinking as readers get more data)."""
    rules = rules or load_rules()
    w = float(rules['nomination']['near_tie_width'])
    b = {float(k): _num(v) for k, v in mean_benefit_by_fraction.items()}
    if not all(f in b and _fin(b[f]) for f in (0.25, 0.5, 1.0)):
        return {'flag': 'LEARNING_CURVE_INCOMPLETE'}
    shrink = b[1.0] < b[0.25] - w
    return {'flag': 'BENEFIT_SHRINKS_WITH_READER_DATA' if shrink else 'NO_SHRINKAGE_FLAG', 'values': b,
            'monotone_decreasing': b[0.25] > b[0.5] > b[1.0], 'note': 'diagnostic; cannot establish full-budget protection'}


# ---------------------------------------------------------------- conditional income branch
def income_branch_trigger(witness_selection, complete_reserve, anchor_dev=None, rules=None):
    """Fires iff at stage-1 close a PROMISING_WITNESS exists AND income accuracy prevents the complete release from meeting
    the registered whole-release reserve, through
      leg A: the MODEL_SELECT whole-release reserve of (anchor income + top witness occupation) fails, and every failing
             condition is an income accuracy-loss clause (income|U0|accuracy_loss or income|Ucal|accuracy_loss, either seed), or
      leg B: the anchor's already published DEV_REPORT income accuracy clauses fail the registered final-stage replication
             rule (margin >= R2 factor x max(SE_DEV(anchor), SE_DEV(H_LJ))) on either seed. No new DEV_REPORT read."""
    rules = rules or load_rules()
    ib = rules['income_branch']
    f = float(rules['final_stage']['r2_factor'])
    leg_a = bool(complete_reserve is not None and complete_reserve['status'] != MET and complete_reserve['income_accuracy_only_failure'])
    anchor_dev = ib['anchor_dev_record'] if anchor_dev is None else anchor_dev
    b_detail = {}
    for key, v in anchor_dev['clauses'].items():
        se = max(_num(v['se_anchor']), _num(v['se_h_lj']))
        margin = _num(v['cap']) - _num(v['point'])
        b_detail[key] = {'margin': margin, 'se_used': se, 'ratio': margin / se if se > 0 else None, 'meets_r2': bool(margin >= f * se)}
    leg_b = any(not d['meets_r2'] for d in b_detail.values())
    promising = bool(witness_selection.get('promising_exists'))
    fires = promising and (leg_a or leg_b)
    return {'fires': fires, 'promising_witness_exists': promising, 'leg_A_model_select': leg_a, 'leg_B_published_dev': leg_b,
            'leg_B_detail': b_detail, 'max_configurations': int(ib['max_seed_specific_configurations']),
            'evaluated_at': ib['evaluated_at'],
            'note': 'leg B uses published DEV_REPORT values of the frozen anchor (no new read); a branch opened through it is DEV-guided and disclosed'}


# ---------------------------------------------------------------- nomination
def _bonferroni(evals_by_seed, z):
    out = []
    for s in SEEDS:
        conds = conditions(_seed_key(evals_by_seed, s), s)
        for n in CONDITIONS:
            c = conds[n]
            if not _fin(c['value'], c['se']) or c['value'] + z * c['se'] > c['cap']:
                out.append(f's{s}|{n}')
    return out


def evaluate_config(c, rules=None):
    """One configuration record of the nomination table (see nominate)."""
    rules = rules or load_rules()
    nr = rules['nomination']
    rec = {'config_id': c['config_id'], 'family': c.get('family'), 'kind': c.get('kind'), 'entry': c.get('entry'),
           'simplicity': _simplicity(c), 'flags': []}
    seeds = sorted(int(s) for s in (c.get('evals') or {}))
    if seeds != [0, 1]:
        rec.update({'label': 'NOT_CONSIDERED_ONE_SEED_ONLY', 'qualifying': False})
        return rec
    adm = []
    if not c.get('runnable_package'):
        adm.append('no runnable package')
    if not c.get('six_raw_fields'):
        adm.append('release not recorded as computed from the six permitted raw fields')
    grids = c.get('grid') or {}
    missing = [x for x in REQUIRED if x not in grids]
    if missing:
        adm.append(f'grid missing required comparators {missing}')
    if adm:
        rec.update({'label': 'NOT_ADMISSIBLE', 'qualifying': False, 'admissibility_failures': adm})
        return rec
    if c.get('alias_of'):
        rec.update({'label': f"ALIAS_OF_{c['alias_of']}", 'qualifying': False})
        return rec
    res = reserve(c['evals'], rules=rules)
    util_fail = [f's{s}|{n}' for s in SEEDS for n in CONDITIONS if not res['per_seed'][s]['conditions'][n]['point_pass']]
    st = {x: stability(grids[x], rules, x) for x in REQUIRED}
    rec_fail = [f'{x}|{v}' for x in REQUIRED for v in VIEWS if not st[x]['recipient'][v]['point_pass']]
    ben_unstable = {x: st[x]['benefit']['label'] for x in REQUIRED if not st[x]['benefit_stable']}
    mc = c.get('matched_component') or {}
    mc_st = stability(mc['grid'], rules, mc.get('config_id')) if mc.get('grid') else None
    mc_ok = bool(mc_st and mc_st['benefit']['label'] != 'INCOMPLETE')
    unresolved = _bonferroni(c['evals'], float(nr['bonferroni_z_28']))
    score = min(st[x]['benefit'].get('mean', -math.inf) if st[x]['benefit']['label'] != 'INCOMPLETE' else -math.inf for x in REQUIRED)
    tgt_labels = {x: st[x]['target']['label'] for x in TARGET_REFS}
    if all(v == 'DEV_GRID_TARGET_STABLE' for v in tgt_labels.values()):
        tlab = 'DEV_GRID_TARGET_STABLE'
    elif all(v in ('DEV_GRID_TARGET_STABLE', 'DEV_GRID_TARGET_POINT_ONLY') for v in tgt_labels.values()):
        tlab = 'DEV_GRID_TARGET_POINT_ONLY'
    else:
        tlab = 'DEV_GRID_TARGET_NOT_MET'
    point_ok = not util_fail and not rec_fail
    rec['outcomes'] = {
        'point': 'POINT_ELIGIBLE' if point_ok else 'POINT_INELIGIBLE',
        'reserve': res['status'],
        'conditional_confidence': 'UTILITY_CONFIDENCE_RESOLVED' if not unresolved else f'UTILITY_CONFIDENCE_UNRESOLVED ({len(unresolved)} of 28)',
        'recipient_protection': ('RECIPIENT_POINT_FAILED' if rec_fail else
                                 'RECIPIENT_STABLE' if all(st[x]['recipient_stable'] for x in REQUIRED) else 'RECIPIENT_EDGE_OR_UNCOMPUTED'),
        'reader_stability': 'BENEFIT_STABLE_ALL_REQUIRED' if not ben_unstable else 'BENEFIT_NOT_STABLE',
        'material_target': tlab,
        'matched_component_comparison': ('PRESENT' if mc_ok else 'MISSING'),
        'incremental_protection_vs_matched_component': (None if not mc_ok else mc_st['benefit']['label']),
    }
    rec.update({'utility_point_failed': util_fail, 'recipient_point_failed': rec_fail, 'reserve': {k: res[k] for k in (
        'status', 'failed', 'min_ratio', 'binding_condition', 'min_income_ratio', 'min_occupation_side_ratio', 'ratio_vector',
        'own_se_zero', 'met_with_own_se_only')}, 'benefit_not_stable': ben_unstable, 'utility_confidence_unresolved': unresolved,
        'score_min_required_mean_benefit': score, 'per_comparator': st, 'matched_component': None if not mc_st else {
            'config_id': mc.get('config_id'), 'stability': mc_st}})
    if util_fail:
        label = 'UTILITY_POINT_INELIGIBLE'
    elif rec_fail:
        label = 'RECIPIENT_POINT_FAILED'
    elif res['status'] != MET:
        label = 'RESERVE_NOT_MET'
        if not ben_unstable:
            rec['flags'].append('EDGE_STABLE_BENEFIT_WITHOUT_RESERVE')
    elif ben_unstable:
        label = 'BENEFIT_NOT_STABLE'
    elif not mc_ok:
        label = 'NO_MATCHED_COMPONENT_COMPARISON'
    else:
        label = 'QUALIFYING'
    if any(st[x]['recipient']['income'].get('label') == 'RECIPIENT_EDGE' or st[x]['recipient']['occupation'].get('label') == 'RECIPIENT_EDGE' for x in REQUIRED):
        rec['flags'].append('RECIPIENT_EDGE')
    if any(st[x]['benefit'].get('mean', 0) < float(nr['near_tie_width']) for x in REQUIRED if st[x]['benefit']['label'] != 'INCOMPLETE'):
        rec['flags'].append('PAIR_NEAR_TIE')
    rec.update({'label': label, 'qualifying': label == 'QUALIFYING'})
    return rec


def nominate(table, rules=None):
    """table: list of configuration dicts {'config_id', 'family', 'kind', 'entry', 'evals': {0: u, 1: u},
    'grid': {comparator: grid entry} for every REQUIRED comparator, 'matched_component': {'config_id', 'grid'} or None,
    'runnable_package', 'six_raw_fields', 'alias_of', 'n_model_fits', 'n_calibrator_fits'}.
    Score = minimum over the required comparators of the 8-cell mean pair benefit (higher is better). Near-tie set: score >=
    best - 0.005. Within it: leximin whole-release reserve ratio vector (larger minimum first), then fewer fitted components,
    then the earliest ledger entry. At most one NOMINATED_FINALIST; none -> DEV_REPORT stays closed."""
    rules = rules or load_rules()
    w = float(rules['nomination']['near_tie_width'])
    recs = [evaluate_config(c, rules) for c in table]
    q = [r for r in recs if r['qualifying']]
    choice = None
    if q:
        best = max(r['score_min_required_mean_benefit'] for r in q)
        tie = [r for r in q if r['score_min_required_mean_benefit'] >= best - w]

        def cmp(a, b):
            x = leximin_cmp(a['reserve']['ratio_vector'], b['reserve']['ratio_vector'])
            if x:
                return x
            if a['simplicity'] != b['simplicity']:
                return -1 if a['simplicity'] < b['simplicity'] else 1
            ea, eb = a.get('entry', math.inf), b.get('entry', math.inf)
            return -1 if ea < eb else (1 if ea > eb else 0)
        tie.sort(key=functools.cmp_to_key(cmp))
        choice = tie[0]
        for r in q:
            r['label'] = 'NOMINATED_FINALIST' if r is choice else 'QUALIFYING_NOT_CHOSEN'
        if choice.get('kind') == 'control':
            choice['flags'].append('CONTROL_DESIGNATED_FINALIST')
        choice['near_tie_set'] = [r['config_id'] for r in tie]
        choice['near_tie_detail'] = [{'config_id': r['config_id'], 'score': r['score_min_required_mean_benefit'],
                                      'best_minus_score': best - r['score_min_required_mean_benefit'],
                                      'min_whole_release_ratio': r['reserve']['min_ratio'], 'simplicity': r['simplicity'],
                                      'entry': r.get('entry')} for r in tie]
    return {'schema': SCHEMA + '/nomination', 'rules_sha256': rules.get('_sha256'), 'configurations': recs,
            'nominated': None if choice is None else choice['config_id'],
            'outcome': 'FINALIST_NOMINATED' if choice else 'NO_QUALIFYING_CONFIGURATION',
            'dev_report': 'OPEN_FOR_ONE_FINALIST' if choice else 'STAYS_CLOSED'}
