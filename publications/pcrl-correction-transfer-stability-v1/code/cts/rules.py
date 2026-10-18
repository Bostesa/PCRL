"""Strength rules, transfer diagnostics, screens and claim helpers of the PCRL correction-transfer campaign
(`pcrl_correction_transfer_stability_v1`; statistical reviewer; written before any fit or outcome of this campaign).

Human-readable rules: results/pcrl_correction_transfer_stability_v1/roles/statistical_reviewer/RULES.md
Machine-readable constants: .../RULES.json (schema CTS1-rules-v1); `load_rules()` refuses a file that disagrees with this module.
Invented-data tests: `python3 -m cts.test_rules`.

Pure functions on in-memory arrays and dicts. Nothing here reads data or any file other than RULES.json.

Units of every cost: an INCREMENTAL cost = corrected release minus uncorrected release on the same people (summed Brier, NLL in
nats with the source's 1e-12 probability floor, accuracy loss = uncorrected accuracy minus corrected accuracy). Five statistics
(STATS) map onto the 8 occupation-side conditions exactly as ohn.protect.SLACK_OF.

Sections
  metrics            task_vectors, probs, logp, cluster_se, subgroup_se, cost_stats, cost_stats_se, incremental_vectors,
                     cost_table (identical to scm_bench / ihs.evaluate / ohn.protect definitions)
  reserve            reserve_slack, witness_slacks, stat_slack
  OLD                budget_at, project (identical re-implementation of ohn.protect.project), old_rule
  SIMPLE             simple_rule
  UNCERTAINTY        uncertainty_rule
  diagnostics        transfer_errors, rule_outcome, accepted_crossings, correction_magnitude
  question A         household_bootstrap_means, prediction_comparison
  screens            per_cell_gate, aggregate_paired_screen, synthetic_cells, oc_rates
  selection, claims  select_recipe, compare_rules, stability_summary, claim_ladder, reserve_qualified_frontier
Every rule is a DEVELOPMENT rule on previously used records; uncertainty statements are asymptotic and model-based.
"""
from __future__ import annotations
import functools
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

RULES_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_correction_transfer_stability_v1' / 'roles'
              / 'statistical_reviewer' / 'RULES.json')
SCHEMA = 'CTS1-rules-v1'
K_OCC = 5
CAPS = {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005}
GUARD_CAP = 0.005
RECIPIENT_ALLOWANCE = 0.010
TARGET = 0.020
NLL_FLOOR = 1e-12
STATS = ('men_brier', 'women_brier', 'brier', 'nll', 'acc_loss')
SLACK_OF = {'men_brier': ('guard|sex0',), 'women_brier': ('guard|sex1',),
            'brier': ('occupation|U0|brier_excess', 'occupation|Ucal|brier_excess'),
            'nll': ('occupation|U0|nll_excess', 'occupation|Ucal|nll_excess'),
            'acc_loss': ('occupation|U0|accuracy_loss', 'occupation|Ucal|accuracy_loss')}
OCC_CONDITIONS = ('occupation|U0|accuracy_loss', 'occupation|U0|nll_excess', 'occupation|U0|brier_excess',
                  'occupation|Ucal|accuracy_loss', 'occupation|Ucal|nll_excess', 'occupation|Ucal|brier_excess',
                  'guard|sex0', 'guard|sex1')
STAT_OF = {c: k for k, cs in SLACK_OF.items() for c in cs}
POP_OF = {'men_brier': 'men', 'women_brier': 'women', 'brier': 'all', 'nll': 'all', 'acc_loss': 'all'}
LADDER = tuple(j / 10 for j in range(11))
K_RESERVE = 2.0
FLOOR_HOUSEHOLDS = 5306
OLD_LEVELS = {'B80': 0.8, 'B50': 0.5}
GRID, ITERS = 101, 30
GAMMA = 0.5
ALPHA = 0.05
Z = 1.644853626951472                       # scipy.stats.norm.ppf(0.95)
# AGG1 dispersion priors, GROSS construction (variance of a repetition mean INCLUDING its reader-seed share), computed label-free
# by the reader investigator from the frozen source grid (13 banks): income 9 wire groups / 36 pairs, occupation 19 / 171,
# pair 26 / 325. Income check: (0.02123682861190588^2 + 0.0018486366498115536^2 / 2) / 0.008962013990683278^2.
PHI_PRIOR = {'income': 5.636511599640258, 'occupation': 0.7781545862915747, 'pair': 1.0867113014695036}
PHI_PRIOR_INCOME = PHI_PRIOR['income']
NEAR_TIE = 0.005
STABILITY_Z = 2.0
BOOT_B = 2000
BOOT_TAG = 'CTS1_transfer_boot'
REQUIRED = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
TARGET_REFS = ('NEURAL_RAW', 'H_LJ')
RULE_ORDER = ('SIMPLE', 'UNCERTAINTY', 'OLD-B50', 'OLD-B80')
_CELL = re.compile(r'^S(\d+)R(\d+)k(\d+)$')
_cache = {}


# ================================================================ the registered rule file
def load_rules(path=RULES_PATH):
    """RULES.json with its SHA-256 under '_sha256'; refuses a schema or constant that disagrees with this module."""
    p = Path(path)
    if str(p) not in _cache:
        raw = p.read_bytes()
        r = json.loads(raw)
        if r.get('schema') != SCHEMA:
            raise ValueError(f'unexpected rules schema {r.get("schema")!r}')
        ol, ru = r['original_limits'], r['rules']
        agg = r['screens']['aggregate_paired_rule']
        checks = {
            'caps': (ol['caps'], CAPS), 'guard cap': (ol['within_sex_guard_cap'], GUARD_CAP),
            'recipient allowance': (ol['recipient_allowance_auc'], RECIPIENT_ALLOWANCE),
            'target': (ol['material_target_auc'], TARGET), 'nll floor': (r['metrics']['nll_probability_floor'], NLL_FLOOR),
            'stats': (tuple(r['metrics']['stats']), STATS),
            'slack_of': ({k: tuple(v) for k, v in r['metrics']['slack_of'].items()}, SLACK_OF),
            'conditions': (tuple(r['conditions']['occupation_side']), OCC_CONDITIONS),
            'k': (r['reserve']['multiplier_k'], K_RESERVE), 'floor households': (r['reserve']['floor_households'], FLOOR_HOUSEHOLDS),
            'ladder': (tuple(r['ladder']['strengths']), LADDER),
            'old levels': (ru['OLD']['levels'], OLD_LEVELS), 'grid': (ru['OLD']['grid_points'], GRID),
            'bisection': (ru['OLD']['bisection_steps'], ITERS), 'gamma': (ru['SIMPLE']['gamma'], GAMMA),
            'alpha': (ru['UNCERTAINTY']['alpha_one_sided'], ALPHA), 'z': (ru['UNCERTAINTY']['z'], Z),
            'agg z': (agg['z'], Z), 'agg alpha': (agg['alpha_one_sided'], ALPHA),
            'phi prior': (agg['phi_prior'], PHI_PRIOR),
            'near tie': (r['recipe_selection']['near_tie_width'], NEAR_TIE),
            'bootstrap B': (r['rule_evaluation']['prediction_comparison']['bootstrap_B'], BOOT_B),
            'required': (tuple(r['comparators']['required']), REQUIRED),
            'target refs': (tuple(r['comparators']['target_references']), TARGET_REFS),
        }
        bad = [k for k, (a, b) in checks.items() if a != b]
        if bad:
            raise ValueError(f'RULES.json disagrees with cts.rules on {bad}')
        fl = r['reserve']['floor_se']
        if any(set(fl[s]) != set(OCC_CONDITIONS) for s in ('0', '1')):
            raise ValueError('floor SEs must cover exactly the 8 occupation-side conditions per seed')
        r['_sha256'] = hashlib.sha256(raw).hexdigest()
        _cache[str(p)] = r
    return _cache[str(p)]


def _fin(*xs):
    return all(isinstance(x, (int, float, np.floating)) and math.isfinite(float(x)) for x in xs)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


# ================================================================ metrics (identical to the source definitions)
def task_vectors(y, p, k=K_OCC):
    """Per-person accuracy, NLL (1e-12 probability floor) and summed Brier; identical to scm_bench.common.task_vectors."""
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), k) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
        raise ValueError('probability convention differs')
    target = y[:, None] == np.arange(k)[None, :]
    return {'accuracy': (p.argmax(1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(p[np.arange(len(y)), y], NLL_FLOOR)),
            'brier': ((p - target) ** 2).sum(1)}


def probs(z):
    """Occupation probabilities from logits: identical operations to acs_replication.trees.probabilities('occupation', z)."""
    raw = np.asarray(z, dtype=np.float64)
    logits = raw - raw.mean(1, keepdims=True)
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


def logp(p):
    """Centred log-probabilities (ohn.protect.logp)."""
    z = np.log(np.maximum(np.asarray(p, dtype=np.float64), 1e-300))
    return z - z.mean(1, keepdims=True)


def cluster_se(d, unit):
    """Household-clustered SE of the mean of d (identical to ihs.evaluate.cluster_se)."""
    d = np.asarray(d, dtype=np.float64); n = len(d)
    _, inv = np.unique(unit, return_inverse=True); G = inv.max() + 1
    s = np.bincount(inv, weights=d - d.mean(), minlength=G)
    return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


def subgroup_se(d, mask, unit):
    """Linearized household-clustered SE of the subgroup mean of d over mask (identical to ihs.evaluate.subgroup_se)."""
    d = np.asarray(d, dtype=np.float64); mask = np.asarray(mask, dtype=bool); n = len(d)
    m = d[mask].mean(); p = mask.mean()
    z = np.where(mask, (d - m) / p, 0.0)
    _, inv = np.unique(unit, return_inverse=True); G = inv.max() + 1
    s = np.bincount(inv, weights=z, minlength=G)
    return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


def incremental_vectors(p, base, y):
    """Per-person increments (corrected p minus uncorrected base on the same people): summed Brier, NLL, accuracy loss."""
    v = task_vectors(y, p, K_OCC)
    return v['brier'] - base['brier'], v['nll'] - base['nll'], base['accuracy'] - v['accuracy']


def cost_stats(p, base, y, sex):
    """The 5 incremental statistics (identical to ohn.protect.cost_stats); base = task_vectors of the uncorrected release."""
    db, dn, da = incremental_vectors(p, base, y)
    men, women = sex == 0, sex == 1
    return {'men_brier': float(db[men].mean()), 'women_brier': float(db[women].mean()), 'brier': float(db.mean()),
            'nll': float(dn.mean()), 'acc_loss': float(da.mean())}


def cost_stats_se(p, base, y, sex, unit):
    """Household-clustered SEs of the 5 statistics (identical to ohn.protect.cost_stats_se): SEX-group statistics use the
    linearized clustered SE within the group, computed on the whole set."""
    db, dn, da = incremental_vectors(p, base, y)
    return {'men_brier': subgroup_se(db, sex == 0, unit), 'women_brier': subgroup_se(db, sex == 1, unit),
            'brier': cluster_se(db, unit), 'nll': cluster_se(dn, unit), 'acc_loss': cluster_se(da, unit)}


def cost_table(p_by_s, p0, y, sex, unit):
    """GUARD / CHECK / MODEL_SELECT incremental-cost table on one people set. p_by_s: {ladder strength: (n, 5) corrected
    probabilities}; p0: the uncorrected release. Returns {'costs': {s: stats}, 'se': {s: stats}, 'n': {'all','men','women'},
    'households'}."""
    base = task_vectors(y, p0, K_OCC)
    sex = np.asarray(sex)
    tab = _ladder_map(p_by_s, allow_missing=True)
    costs = {s: cost_stats(p, base, y, sex) for s, p in tab.items()}
    ses = {s: cost_stats_se(p, base, y, sex, unit) for s, p in tab.items()}
    return {'costs': costs, 'se': ses, 'n': {'all': int(len(y)), 'men': int((sex == 0).sum()), 'women': int((sex == 1).sum())},
            'households': int(len(np.unique(unit)))}


# ================================================================ ladder helpers
def _ladder_key(s):
    v = round(float(s), 6)
    for x in LADDER:
        if abs(v - x) <= 1e-9:
            return x
    raise ValueError(f'strength {s!r} is not on the ladder {LADDER}')


def _ladder_map(m, allow_missing=False):
    """Normalise a {strength: value} mapping (float or string keys) onto the ladder; every nonzero strength required."""
    out = {}
    for k, v in m.items():
        key = _ladder_key(k)
        if key in out:
            raise ValueError(f'duplicate strength {k!r}')
        out[key] = v
    missing = [s for s in LADDER[1:] if s not in out]
    if missing and not allow_missing:
        raise ValueError(f'ladder strengths missing: {missing}')
    return dict(sorted(out.items()))


def _fixed_sequence(passes):
    """passes: list of (strength, bool) in ascending order over the nonzero ladder. Returns (selected, first_failure):
    the last strength before the first failure (0.0 = identity when the first nonzero strength fails)."""
    selected, first_fail = 0.0, None
    for s, ok in passes:
        if not ok:
            first_fail = s
            break
        selected = s
    return selected, first_fail


# ================================================================ reserve slack (baseline)
def reserve_slack(margin, own_se, floor_se, k=K_RESERVE):
    """rho = margin - k * se_used, se_used = max(own SE, floor SE) (ohn.rules._condition conventions).
    Refuses (ValueError) a floor that is missing, non-finite or <= 0 and a negative own SE. A non-finite margin or own SE gives
    rho = -inf (the condition admits no correction). Records whether the floor was used and whether own SE was exactly 0."""
    fl = _num(floor_se)
    if not _fin(fl) or fl <= 0:
        raise ValueError(f'floor SE must be finite and > 0 (got {floor_se!r}); a zero floor would void the reserve')
    own = _num(own_se); m = _num(margin)
    if _fin(own) and own < 0:
        raise ValueError(f'negative own SE {own}')
    if not _fin(m, own):
        return {'slack': -math.inf, 'margin': m, 'own_se': own, 'floor_se': fl, 'se_used': None, 'se_source': None,
                'nonfinite': True, 'own_se_zero': False, 'met': False}
    se_used = max(own, fl)
    return {'slack': m - k * se_used, 'margin': m, 'own_se': own, 'floor_se': fl, 'se_used': se_used,
            'se_source': 'own' if own >= fl else 'floor', 'nonfinite': False, 'own_se_zero': own == 0.0,
            'met': bool(m >= k * se_used)}


def witness_slacks(conds, seed, households, floor=None, rules=None):
    """Baseline record of the 8 occupation-side conditions of one partition's witness on MODEL_SELECT.
    conds: {condition: {'value', 'cap', 'se'}} (the evaluator's clauses / guards, own household-clustered SEs);
    seed: the associated source seed (0 or 1); households: MODEL_SELECT household count of the evaluation.
    The floor (H_LJ's MODEL_SELECT SE of the same condition and seed) is rescaled by sqrt(5306 / households).
    Caps are checked against the original caps (never changed)."""
    if floor is None:
        floor = (rules or load_rules())['reserve']['floor_se'][str(int(seed))]
    G = int(households)
    if G <= 1:
        raise ValueError(f'invalid household count {G}')
    scale = math.sqrt(FLOOR_HOUSEHOLDS / G)
    out = {}
    for c in OCC_CONDITIONS:
        if c not in conds:
            raise ValueError(f'missing condition {c}')
        cap = _num(conds[c]['cap'])
        want = GUARD_CAP if c.startswith('guard|') else CAPS[c.split('|')[2]]
        if cap != want:
            raise ValueError(f'{c}: cap {cap} differs from the original cap {want}')
        rec = reserve_slack(cap - _num(conds[c]['value']), conds[c].get('se'), _num(floor.get(c)) * scale)
        rec.update({'condition': c, 'value': _num(conds[c]['value']), 'cap': cap, 'floor_scale': scale})
        out[c] = rec
    return out


def _slack_value(x):
    return float(x['slack']) if isinstance(x, dict) else float(x)


def stat_slack(slack):
    """Per statistic: the smallest reserve slack among its mapped conditions."""
    return {k: min(_slack_value(slack[c]) for c in cs) for k, cs in SLACK_OF.items()}


# ================================================================ OLD: the source projection (identical re-implementation)
def budget_at(slack, f):
    """f x the smallest reserve slack of each statistic's mapped conditions (ohn.protect.budget_at)."""
    return {stat: float(f) * min(_slack_value(slack[c]) for c in conds) for stat, conds in SLACK_OF.items()}


def feasible(st, budget):
    return all(st[k] <= budget[k] for k in STATS)


def utilization(st, budget):
    u = 0.0
    for k in STATS:
        b = budget[k]
        u = max(u, st[k] / b if b > 0 else (math.inf if st[k] > b else 0.0))
    return u


def project(z0, delta, y, sex, budget, grid=GRID, iters=ITERS):
    """Identical re-implementation of ohn.protect.project: the largest s in [0, 1] such that every grid point 0, 1/(grid-1), ...
    up to s is feasible, refined by bisection toward the first infeasible grid point. Costs against the base at s = 0."""
    base = task_vectors(y, probs(z0), K_OCC)

    def at(s):
        return cost_stats(probs(z0 + s * delta), base, y, sex)
    curve, lo, hi = [], 0.0, None
    for i in range(grid):
        s = i / (grid - 1)
        st = at(s)
        curve.append({'s': s, **st, 'utilization': utilization(st, budget), 'feasible': feasible(st, budget)})
    for c in curve:
        if not c['feasible']:
            hi = c['s']
            break
        lo = c['s']
    if hi is not None:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if feasible(at(mid), budget):
                lo = mid
            else:
                hi = mid
    st = at(lo)
    return lo, st, curve, base


def old_rule(z0, delta_oof, y, sex, slack, level='B80'):
    """OLD rule = the source projection on out-of-fold FIT costs: z0 = centred logits of the cross-fitted witness output on
    FIT, delta_oof = fold pipeline k's correction on fold k, budget = f x reserve slack (B80: f = 0.8 primary; B50: 0.5).
    Returns the selected (continuous) scale, the OOF costs there and at the ladder strengths (for transfer diagnostics)."""
    f = OLD_LEVELS[level]
    budget = budget_at(slack, f)
    s, st, curve, _ = project(np.asarray(z0, dtype=np.float64), np.asarray(delta_oof, dtype=np.float64), np.asarray(y),
                              np.asarray(sex), budget)
    ladder_oof = {LADDER[j]: {k: curve[10 * j][k] for k in STATS} for j in range(len(LADDER))}
    flags = ['OLD_NONPOSITIVE_SLACK'] if any(_slack_value(v) <= 0 for v in slack.values()) else []
    return {'rule': f'OLD-{level}', 'selected': s, 'identity': s <= 0.0, 'budget': budget, 'oof_stats': st,
            'oof_utilization': utilization(st, budget), 'ladder_oof': ladder_oof, 'flags': flags,
            'note': 'continuous scale on the 101-point grid + 30 bisection steps (not restricted to the ladder)'}


# ================================================================ SIMPLE
def simple_rule(guard_costs, slack, gamma=GAMMA):
    """Conservative scalar shrinkage on GUARD. guard_costs: {strength: {stat: incremental cost on GUARD}} over the ladder;
    slack: {condition: reserve slack rho_c (float or witness_slacks record)}.
    Accept s iff for every condition c: cost_{stat(c)}(s) <= b_c, b_c = gamma * rho_c if rho_c > 0 else rho_c (never looser
    than the slack). Ascending fixed sequence; the last accepted strength is selected; identity if s = 0.1 fails.
    A non-finite cost fails. No uncertainty is used (a point rule)."""
    gc = _ladder_map(guard_costs)
    thr = {c: (gamma * r if r > 0 else r) for c in OCC_CONDITIONS for r in (_slack_value(slack[c]),)}
    per, passes = {}, []
    for s in LADDER[1:]:
        rec = {}
        for c in OCC_CONDITIONS:
            x = _num(gc[s][STAT_OF[c]])
            rec[c] = {'cost': x, 'threshold': thr[c], 'pass': bool(_fin(x) and x <= thr[c])}
        ok = all(v['pass'] for v in rec.values())
        per[s] = {'accept': ok, 'conditions': rec, 'failing': [c for c in OCC_CONDITIONS if not rec[c]['pass']]}
        passes.append((s, ok))
    sel, ff = _fixed_sequence(passes)
    flags = ['NONPOSITIVE_SLACK'] if any(_slack_value(slack[c]) <= 0 for c in OCC_CONDITIONS) else []
    later = [s for s, ok in passes if ff is not None and s > ff and ok]
    return {'rule': 'SIMPLE', 'gamma': gamma, 'selected': sel, 'identity': sel == 0.0, 'first_failure': ff,
            'accepted_after_first_failure_not_used': later, 'thresholds': thr, 'per_strength': per, 'flags': flags}


# ================================================================ UNCERTAINTY
def uncertainty_rule(guard_costs, guard_se, baseline, n_guard, n_ms, alpha=ALPHA, z=None):
    """Household-aware rule with an explicit baseline-slack term, finite ladder, fixed sequence, intersection-union.
    guard_costs / guard_se: {strength: {stat: GUARD incremental mean / its household-clustered SE (SEX-group stats: within-group
    linearized clustered SE)}}; baseline: witness_slacks(...) record per condition ({'slack', 'margin', 'se_used'});
    n_guard / n_ms: people counts {'all', 'men', 'women'} on GUARD and MODEL_SELECT.
    For each condition c with statistic k and population g = POP_OF[k]:
      se_k(s)  = max(SE_k(s), phi_k), phi_acc_loss = 1 / n_guard['all'] (one changed decision), phi = 0 otherwise;
      tau_k(s) = se_k(s) * sqrt(1 + n_guard[g] / n_ms[g])          (predictive SE for a fresh MODEL_SELECT-sized sample)
      U1: Delta_k(s) + z tau_k(s) <= rho_c                           (keeps the development reserve on the benchmark role)
      U2: Delta_k(s) + z sqrt(se_k(s)^2 + se_used_c^2) <= margin_c    (population value of the corrected release <= cap)
    s is accepted iff U1 and U2 hold for all 8 conditions (intersection-union: no Bonferroni); ascending fixed sequence; the
    last accepted strength is selected (identity if s = 0.1 fails). z = Phi^-1(1 - alpha) unless given.
    Asymptotic and model-based (RULES.md section 5); not a finite-sample certificate."""
    if z is None:
        z = Z if alpha == ALPHA else _norm_ppf(1 - alpha)
    gc, gs = _ladder_map(guard_costs), _ladder_map(guard_se)
    for g in ('all', 'men', 'women'):
        if not (_fin(_num(n_guard.get(g)), _num(n_ms.get(g))) and n_guard[g] > 0 and n_ms[g] > 0):
            raise ValueError(f'people counts for {g} must be positive on GUARD and MODEL_SELECT')
    phi = {k: 0.0 for k in STATS}
    phi['acc_loss'] = 1.0 / float(n_guard['all'])
    infl = {k: math.sqrt(1.0 + float(n_guard[POP_OF[k]]) / float(n_ms[POP_OF[k]])) for k in STATS}
    per, passes = {}, []
    for s in LADDER[1:]:
        rec = {}
        for c in OCC_CONDITIONS:
            k = STAT_OF[c]
            d, se = _num(gc[s][k]), _num(gs[s][k])
            if _fin(se) and se < 0:
                raise ValueError(f'negative SE at s={s}, {k}')
            b = baseline[c]
            rho, m, sb = _num(b['slack']), _num(b.get('margin')), _num(b.get('se_used'))
            if not _fin(d, se, rho, m, sb):
                rec[c] = {'cost': d, 'se': se, 'pass': False, 'nonfinite': True}
                continue
            se_u = max(se, phi[k])
            tau = se_u * infl[k]
            b1 = d + z * tau
            b2 = d + z * math.sqrt(se_u ** 2 + sb ** 2)
            rec[c] = {'cost': d, 'se': se, 'se_used_increment': se_u, 'se_floor_used': se < phi[k], 'tau': tau,
                      'bound_U1': b1, 'rho': rho, 'pass_U1': bool(b1 <= rho), 'bound_U2': b2, 'margin': m,
                      'pass_U2': bool(b2 <= m), 'pass': bool(b1 <= rho and b2 <= m)}
        ok = all(v['pass'] for v in rec.values())
        per[s] = {'accept': ok, 'conditions': rec, 'failing': [c for c in OCC_CONDITIONS if not rec[c]['pass']]}
        passes.append((s, ok))
    sel, ff = _fixed_sequence(passes)
    later = [s for s, ok in passes if ff is not None and s > ff and ok]
    return {'rule': 'UNCERTAINTY', 'alpha': alpha, 'z': z, 'selected': sel, 'identity': sel == 0.0, 'first_failure': ff,
            'accepted_after_first_failure_not_used': later, 'inflation': infl, 'increment_se_floor': phi,
            'per_strength': per}


def _norm_ppf(q):
    from scipy.stats import norm
    return float(norm.ppf(q))


# ================================================================ recorded diagnostics (never gating)
def transfer_errors(predicted, realized, slack):
    """Signed error = realized - predicted (positive = cost under-predicted), absolute error and error / stat slack, per
    ladder strength present in both and per statistic. predicted / realized: {strength: {stat: cost}}; slack: per condition."""
    P, R = _ladder_map(predicted, True), _ladder_map(realized, True)
    ss = stat_slack(slack)
    out = {}
    for s in sorted(set(P) & set(R)):
        out[s] = {}
        for k in STATS:
            e = _num(R[s][k]) - _num(P[s][k])
            out[s][k] = {'predicted': _num(P[s][k]), 'realized': _num(R[s][k]), 'signed': e, 'abs': abs(e),
                         'rel_slack': (e / ss[k]) if ss[k] > 0 else None, 'abs_rel_slack': (abs(e) / ss[k]) if ss[k] > 0 else None}
    return out


def rule_outcome(selected, realized, baseline):
    """Realized consequences of a selected strength on one people set (CHECK or MODEL_SELECT): per condition the realized
    incremental cost, its use of the reserve slack (cost / rho_c), a reserve crossing (cost > rho_c) and a point-cap crossing
    (cost > margin_c). realized: {stat: incremental cost at the selected strength} (all zeros for identity)."""
    rec, use = {}, []
    for c in OCC_CONDITIONS:
        x = _num(realized[STAT_OF[c]])
        rho, m = _num(baseline[c]['slack']), _num(baseline[c]['margin'])
        u = x / rho if rho > 0 else (math.inf if x > rho else 0.0)
        use.append(u)
        rec[c] = {'cost': x, 'rho': rho, 'margin': m, 'slack_use': u, 'reserve_crossing': bool(x > rho),
                  'point_cap_crossing': bool(x > m)}
    return {'selected': selected, 'identity': selected == 0.0, 'max_slack_use': max(use), 'conditions': rec,
            'reserve_crossing': any(v['reserve_crossing'] for v in rec.values()),
            'point_cap_crossing': any(v['point_cap_crossing'] for v in rec.values())}


def accepted_crossings(rule_result, realized, baseline):
    """Every strength the rule ACCEPTED on its fixed-sequence path (0 < s <= selected): realized reserve and point-cap
    crossings on one people set. realized: {strength: {stat: realized incremental cost}}. Recorded, never gating."""
    R_ = _ladder_map(realized, True)
    acc = [s for s in LADDER[1:] if s <= rule_result['selected']]
    out = {s: rule_outcome(s, R_[s], baseline) for s in acc if s in R_}
    return {'accepted': acc, 'per_strength': out,
            'n_reserve_crossings': sum(v['reserve_crossing'] for v in out.values()),
            'n_point_cap_crossings': sum(v['point_cap_crossing'] for v in out.values())}


def correction_magnitude(z0, delta, s, p0=None):
    """Label-free size of a correction at strength s: mean over people and classes of |s * centred Delta| (logit change;
    softmax is invariant to row constants) and the decision flip rate against the uncorrected release."""
    d = float(s) * np.asarray(delta, dtype=np.float64)
    dc = d - d.mean(1, keepdims=True)
    p0 = probs(z0) if p0 is None else np.asarray(p0, dtype=np.float64)
    p1 = probs(np.asarray(z0, dtype=np.float64) + d)
    return {'mean_abs_logit_change': float(np.abs(dc).mean()), 'flip_rate': float((p0.argmax(1) != p1.argmax(1)).mean())}


# ================================================================ question A: cost prediction on excluded households
def household_bootstrap_means(x, unit, B, rng, mask=None):
    """B household-bootstrap replicates of column means of x (n, m) over the rows in mask (all rows if None): households
    resampled with replacement (multinomial counts), each person weighted by the count of its household. Returns (B, m)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    _, inv = np.unique(unit, return_inverse=True)
    G = inv.max() + 1
    msk = np.ones(len(x), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    S = np.zeros((G, x.shape[1])); np.add.at(S, inv[msk], x[msk])
    cnt = np.bincount(inv[msk], minlength=G).astype(np.float64)
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)
    return (W @ S) / (W @ cnt)[:, None]


def _set_stats(db, dn, da, sex):
    """(10-column) statistic arrays from per-person increments of one set at the 10 nonzero ladder strengths."""
    sex = np.asarray(sex)
    return {'men_brier': (db, sex == 0), 'women_brier': (db, sex == 1), 'brier': (db, None), 'nll': (dn, None),
            'acc_loss': (da, None)}


def prediction_comparison(sets, slack, B=BOOT_B, seed_tag=BOOT_TAG):
    """Question A on one partition: does the GUARD estimate predict the realized CHECK increment better than OOF?
    sets: {'OOF': {...}, 'GUARD': {...}, 'CHECK': {...}}, each {'db', 'dn', 'da': (n, 10) per-person increments at the 10
    nonzero ladder strengths (summed Brier, NLL, accuracy loss), 'unit': household ids, 'sex': 0/1}. slack: per condition.
    e_X(s,k) = Delta_CHECK - Delta_X; MAE_rel(X) = mean over (s, k with stat slack > 0) of |e_X| / stat slack;
    D = MAE_rel(OOF) - MAE_rel(GUARD) (positive favours GUARD). Paired household bootstrap: households resampled independently
    within each set (conditional on the fitted pipelines); returns D, its bootstrap SE and 5% / 95% quantiles, the label and
    the sign count over the (s, k) cells."""
    ss = stat_slack(slack)
    ks = [k for k in STATS if ss[k] > 0]
    if not ks:
        raise ValueError('no statistic has a positive slack; relative errors are undefined')
    rng = np.random.default_rng(int(hashlib.sha256(seed_tag.encode()).hexdigest()[:15], 16))
    point, boot = {}, {}
    for name in ('OOF', 'GUARD', 'CHECK'):
        st = sets[name]
        for k, (arr, mask) in _set_stats(st['db'], st['dn'], st['da'], st['sex']).items():
            a = np.asarray(arr, dtype=np.float64)
            if a.shape[1] != 10:
                raise ValueError('increments must have 10 columns (the nonzero ladder strengths)')
            sel = np.ones(len(a), dtype=bool) if mask is None else mask
            point[name, k] = a[sel].mean(0)
            boot[name, k] = household_bootstrap_means(a, st['unit'], B, rng, mask)

    def mae(get, X):
        return np.mean([np.abs(get('CHECK', k) - get(X, k)) / ss[k] for k in ks], axis=0)
    p_g = mae(lambda n, k: point[n, k], 'GUARD').mean()
    p_o = mae(lambda n, k: point[n, k], 'OOF').mean()
    b_g = np.mean([np.abs(boot['CHECK', k] - boot['GUARD', k]) / ss[k] for k in ks], axis=0).mean(1)
    b_o = np.mean([np.abs(boot['CHECK', k] - boot['OOF', k]) / ss[k] for k in ks], axis=0).mean(1)
    Db = b_o - b_g
    D = float(p_o - p_g)
    q05, q95 = float(np.quantile(Db, 0.05)), float(np.quantile(Db, 0.95))
    wins = int(sum((np.abs(point['CHECK', k] - point['GUARD', k]) < np.abs(point['CHECK', k] - point['OOF', k])).sum() for k in ks))
    bias = {X: {k: float((point['CHECK', k] - point[X, k]).mean()) for k in STATS} for X in ('OOF', 'GUARD')}
    return {'MAE_rel_GUARD': float(p_g), 'MAE_rel_OOF': float(p_o), 'D': D, 'boot_se': float(Db.std(ddof=1)),
            'q05': q05, 'q95': q95, 'label': 'PREDICTION_IMPROVED' if (D > 0 and q05 > 0) else 'NOT_IMPROVED',
            'guard_closer_cells': wins, 'cells': 10 * len(ks), 'stats_used': ks, 'mean_signed_error': bias, 'B': B}


def prediction_comparison_registered(sets, slack, B=BOOT_B, seed_tag=BOOT_TAG):
    """Recorded correction (20:36Z; roles/statistical_reviewer/ADJUDICATION_QA_BOOTSTRAP.md): the scheme RULES.md section 6 defines.
    ONE household-resample weight matrix per set (drawn in the fixed order OOF, GUARD, CHECK from the generator seeded by the
    registered tag) is shared by every statistic (with its SEX mask) and every strength of that set, so the correlation between the
    statistics on the same households enters Var(D). The point estimates, D, B, the tag and the q05 > 0 rule are unchanged.
    `prediction_comparison` above drew a fresh resample per (set, statistic); its labels are kept as superseded."""
    ss = stat_slack(slack)
    ks = [k for k in STATS if ss[k] > 0]
    if not ks:
        raise ValueError('no statistic has a positive slack; relative errors are undefined')
    rng = np.random.default_rng(int(hashlib.sha256(seed_tag.encode()).hexdigest()[:15], 16))
    point, boot = {}, {}
    for name in ('OOF', 'GUARD', 'CHECK'):
        st = sets[name]
        _, inv = np.unique(st['unit'], return_inverse=True)
        G = inv.max() + 1
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)          # one resample per set and replicate
        for k, (arr, mask) in _set_stats(st['db'], st['dn'], st['da'], st['sex']).items():
            a = np.asarray(arr, dtype=np.float64)
            if a.shape[1] != 10:
                raise ValueError('increments must have 10 columns (the nonzero ladder strengths)')
            msk = np.ones(len(a), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
            point[name, k] = a[msk].mean(0)
            S = np.zeros((G, a.shape[1])); np.add.at(S, inv[msk], a[msk])
            cnt = np.bincount(inv[msk], minlength=G).astype(np.float64)
            boot[name, k] = (W @ S) / (W @ cnt)[:, None]

    def mae(get, X):
        return np.mean([np.abs(get('CHECK', k) - get(X, k)) / ss[k] for k in ks], axis=0)
    p_g = mae(lambda n, k: point[n, k], 'GUARD').mean()
    p_o = mae(lambda n, k: point[n, k], 'OOF').mean()
    b_g = np.mean([np.abs(boot['CHECK', k] - boot['GUARD', k]) / ss[k] for k in ks], axis=0).mean(1)
    b_o = np.mean([np.abs(boot['CHECK', k] - boot['OOF', k]) / ss[k] for k in ks], axis=0).mean(1)
    Db = b_o - b_g
    D = float(p_o - p_g)
    q05, q95 = float(np.quantile(Db, 0.05)), float(np.quantile(Db, 0.95))
    wins = int(sum((np.abs(point['CHECK', k] - point['GUARD', k]) < np.abs(point['CHECK', k] - point['OOF', k])).sum() for k in ks))
    bias = {X: {k: float((point['CHECK', k] - point[X, k]).mean()) for k in STATS} for X in ('OOF', 'GUARD')}
    return {'MAE_rel_GUARD': float(p_g), 'MAE_rel_OOF': float(p_o), 'D': D, 'boot_se': float(Db.std(ddof=1)),
            'q05': q05, 'q95': q95, 'label': 'PREDICTION_IMPROVED' if (D > 0 and q05 > 0) else 'NOT_IMPROVED',
            'guard_closer_cells': wins, 'cells': 10 * len(ks), 'stats_used': ks, 'mean_signed_error': bias, 'B': B,
            'scheme': 'registered: one household resample per set and replicate, shared by statistics and strengths'}


# ================================================================ screens (question C)
def _parse_cells(cells):
    out = {}
    for c, v in cells.items():
        m = _CELL.match(c)
        if not m:
            raise ValueError(f'cell id {c!r} is not S<seed>R<rep>k<reader seed>')
        out[c] = (int(m.group(1)), int(m.group(2)), int(m.group(3)), _num(v))
    return out


def per_cell_gate(cells, limit=RECIPIENT_ALLOWANCE):
    """OLD recipient gate for one (candidate, comparator, view): pass iff every cell difference (candidate AUC - comparator
    AUC) <= limit (inclusive). A non-finite cell fails."""
    vals = {c: _num(v) for c, v in cells.items()}
    if not vals or not all(_fin(v) for v in vals.values()):
        return {'pass': False, 'label': 'INCOMPLETE'}
    above = [c for c, v in vals.items() if v > limit]
    return {'pass': not above, 'label': 'GATE_PASS' if not above else 'GATE_FAIL', 'cells_above': above,
            'max': max(vals.values()), 'mean': float(np.mean(list(vals.values())))}


def aggregate_paired_screen(cells, cell_se, phi_prior, allowance=RECIPIENT_ALLOWANCE, alpha=ALPHA, alias=False):
    """AGG1 aggregate paired rule for one (candidate, comparator, view). cells / cell_se: {'S<s>R<r>k<k>': difference, SE}
    (difference = candidate AUC - comparator AUC; SE = household bootstrap of SCORE households, readers fixed).
      y_R = mean over seeds and reader seeds of repetition R; se_R = mean of the same cells' SEs (Minkowski upper bound: the
            cells of a repetition share SCORE households); dbar = mean of all cells; n_R repetitions (>= 2)
      v_samp = sum_R se_R^2 / n_R^2;  phi_emp = [sum_R (y_R - dbar)^2 / (n_R - 1)] / mean_R se_R^2
      phi = max(1, phi_prior, phi_emp);  U = dbar + z sqrt(phi v_samp);  pass iff U <= allowance.
    phi_prior = PHI_PRIOR[view] (gross construction, the same scale as phi_emp: both use repetition means that include the
    reader-seed share).
    alias=True (candidate view wire bitwise equal to the comparator's on both seeds): exact zero, label EXACT_ALIAS; nonzero
    cached cells for an alias are an error. Recorded (not gating): per-seed bounds and AGG_SEED_EDGE."""
    if alias:
        if any(_num(v) != 0.0 for v in cells.values()):
            raise ValueError('an exact alias must give identical cached reader results (all cells 0)')
        return {'pass': True, 'label': 'EXACT_ALIAS', 'U': 0.0, 'dbar': 0.0}
    if not _fin(_num(phi_prior)):
        raise ValueError(f'phi_prior must be a finite number computed from the frozen source grid (got {phi_prior!r})')
    pc = _parse_cells(cells)
    if set(cell_se) != set(cells):
        return {'pass': False, 'label': 'INCOMPLETE', 'error': 'cells and SEs differ'}
    se = {c: _num(v) for c, v in cell_se.items()}
    if not all(_fin(v[3]) for v in pc.values()) or not all(_fin(v) for v in se.values()):
        return {'pass': False, 'label': 'INCOMPLETE', 'error': 'non-finite cell or SE'}
    if any(v < 0 for v in se.values()):
        raise ValueError('negative cell SE')
    reps = sorted({v[1] for v in pc.values()})
    seeds = sorted({v[0] for v in pc.values()})
    full = {(s, r, k) for s, r, k, _ in pc.values()}
    ks = sorted({v[2] for v in pc.values()})
    if len(reps) < 2 or len(full) != len(seeds) * len(reps) * len(ks):
        return {'pass': False, 'label': 'INCOMPLETE', 'error': 'need a complete seeds x repetitions x reader-seeds design, >= 2 repetitions'}
    z = Z if alpha == ALPHA else _norm_ppf(1 - alpha)
    y = {r: np.mean([v[3] for v in pc.values() if v[1] == r]) for r in reps}
    ser = {r: np.mean([se[c] for c, v in pc.items() if v[1] == r]) for r in reps}
    nR = len(reps)
    dbar = float(np.mean([v[3] for v in pc.values()]))
    v_samp = sum(ser[r] ** 2 for r in reps) / nR ** 2
    v_btw = sum((y[r] - dbar) ** 2 for r in reps) / (nR - 1)
    mse = float(np.mean([ser[r] ** 2 for r in reps]))
    flags = []
    if mse > 0:
        phi_emp = v_btw / mse
        phi = max(1.0, float(phi_prior), phi_emp)
        V = phi * v_samp
    else:                                           # all bootstrap SEs zero for distinct wires: no scale to inflate
        phi_emp, phi = math.inf if v_btw > 0 else 1.0, math.nan
        V = v_btw / nR
        flags.append('ZERO_BOOTSTRAP_SE')
    U = dbar + z * math.sqrt(V)
    per_seed = {}
    for s in seeds:
        ys = [np.mean([v[3] for v in pc.values() if v[0] == s and v[1] == r]) for r in reps]
        ss_ = [np.mean([se[c] for c, v in pc.items() if v[0] == s and v[1] == r]) for r in reps]
        vs = (phi if math.isfinite(phi) else 1.0) * sum(x ** 2 for x in ss_) / nR ** 2
        per_seed[s] = {'dbar': float(np.mean(ys)), 'U': float(np.mean(ys) + z * math.sqrt(vs))}
    ok = bool(U <= allowance)
    if ok and any(v['U'] > allowance for v in per_seed.values()):
        flags.append('AGG_SEED_EDGE')
    return {'pass': ok, 'label': 'AGG_PASS' if ok else 'AGG_FAIL', 'U': float(U), 'dbar': dbar, 'se_total': math.sqrt(V),
            'phi': phi, 'phi_prior': float(phi_prior), 'phi_emp': phi_emp, 'v_samp': v_samp, 'n_repetitions': nR,
            'z': z, 'per_seed': per_seed, 'flags': flags}


def synthetic_cells(mu, rng, sd_split, sd_reader, se_cell, rho_seed=0.0, n_rep=2, n_k=2, seeds=(0, 1)):
    """Cell-level synthetic law L1 for one comparison: d_{s,R,k} = mu_s + A_{s,R} + B_{s,R,k};
    A_{.,R} ~ N(0, sd_split^2) jointly normal across seeds with correlation rho_seed (shared household splits), independent
    across R; B ~ N(0, sd_reader^2) independent; every reported cell SE = se_cell. mu: number or {seed: number}."""
    mu = {s: (mu[s] if isinstance(mu, dict) else mu) for s in seeds}
    cov = sd_split ** 2 * (rho_seed * np.ones((len(seeds), len(seeds))) + (1 - rho_seed) * np.eye(len(seeds)))
    cells, ses = {}, {}
    for r in range(1, n_rep + 1):
        A = rng.multivariate_normal(np.zeros(len(seeds)), cov)
        for i, s in enumerate(seeds):
            for k in range(n_k):
                cells[f'S{s}R{r}k{k}'] = mu[s] + A[i] + rng.normal(0.0, sd_reader)
                ses[f'S{s}R{r}k{k}'] = se_cell
    return cells, ses


def oc_rates(mu, n_sim, seed, sd_split, sd_reader, se_cell, rho_seed=0.0, phi_prior=PHI_PRIOR_INCOME, n_rep=2):
    """Simulated pass rates of the per-cell gate and AGG1 under law L1 (one comparison). Simulated rates under that law only."""
    rng = np.random.default_rng(seed)
    g = a = 0
    for _ in range(n_sim):
        c, s = synthetic_cells(mu, rng, sd_split, sd_reader, se_cell, rho_seed, n_rep=n_rep)
        g += per_cell_gate(c)['pass']
        a += aggregate_paired_screen(c, s, phi_prior)['pass']
    return {'per_cell_gate_pass': g / n_sim, 'aggregate_pass': a / n_sim, 'n_sim': n_sim}


# ================================================================ stability (source convention), selection, comparisons, claims
def stability_summary(cells, cell_se, z=STABILITY_Z, threshold=0.0):
    """The source stability screen for a paired benefit (positive favours the candidate) over the 8 cells of one comparison:
    per-seed (partition) mean d_s; v_rep = pooled within-seed variance (df 8 - 2); v_samp = mean squared cell SE;
    sd_total = sqrt(v_rep + v_samp); STABLE_POSITIVE iff every cell > threshold and every d_s - z sd_total > threshold."""
    pc = _parse_cells(cells)
    seeds = sorted({v[0] for v in pc.values()})
    by = {s: [v[3] for v in pc.values() if v[0] == s] for s in seeds}
    d = {s: float(np.mean(v)) for s, v in by.items()}
    v_rep = sum((x - d[s]) ** 2 for s in seeds for x in by[s]) / (len(pc) - len(seeds))
    v_samp = float(np.mean([_num(cell_se[c]) ** 2 for c in pc]))
    sd = math.sqrt(v_rep + v_samp)
    allpos = all(v[3] > threshold for v in pc.values())
    stable = allpos and all(d[s] - z * sd > threshold for s in seeds)
    mean = float(np.mean([v[3] for v in pc.values()]))
    label = ('STABLE_POSITIVE' if stable else 'POSITIVE_NOT_STABLE' if allpos else 'SIGN_CHANGES' if mean > threshold
             else 'NOT_POSITIVE')
    return {'label': label, 'stable': stable, 'per_seed_mean': d, 'sd_total': sd, 'mean': mean,
            'lower_per_seed': {s: d[s] - z * sd for s in seeds}, 'min_lower': min(d[s] - z * sd for s in seeds)}


def _leximin_cmp(a, b, tol=1e-9):
    for x, y in zip(sorted(a), sorted(b)):
        if abs(x - y) <= tol:
            continue
        return -1 if x > y else 1
    return 0


def select_recipe(cands, near_tie=NEAR_TIE):
    """At most ONE development recipe, before any repair CHECK panel is opened. cands: list of dicts with
    'recipe' (id), 'routing' (REFIT/FROZEN), 'arm' (PAIR/LOC), 'deployment' (full/ensemble), 'rule' (RULE_ORDER), 'entry',
    and per D partition ('D0', 'D1') in 'D': {'selected', 'reserve_crossing_check', 'reserve_crossing_ms', 'point_pass_ms',
    'cont_grid': {'cells', 'cell_se'} (pair benefit vs the matched CONT control, the partition's 4 cells) or None,
    'ms_reserve_ratios': list of (rho_c - realized cost_c) / se_used_c over the 8 conditions}.
    Gates (both D partitions): nonzero strength, no realized reserve crossing on D CHECK or MODEL_SELECT, point pass, CONT grid.
    Score: min over D partitions of (d_P - 2 sd_total) (source spread over the partition's 4 cells). Near-tie set: score >=
    best - 0.005; tie-breaks: leximin realized reserve ratios, REFIT, full, rule order, LOC before PAIR, earliest entry."""
    recs = []
    for c in cands:
        if c['arm'] not in ('PAIR', 'LOC'):
            recs.append({'recipe': c['recipe'], 'label': 'NOT_A_RECIPE_ARM'}); continue
        fails, lows, ratios = [], [], []
        for P in ('D0', 'D1'):
            e = (c.get('D') or {}).get(P)
            if not e:
                fails.append(f'{P}: missing'); continue
            if not e['selected'] > 0:
                fails.append(f'{P}: identity')
            if e['reserve_crossing_check'] or e['reserve_crossing_ms']:
                fails.append(f'{P}: realized reserve crossing')
            if not e['point_pass_ms']:
                fails.append(f'{P}: point failure on MODEL_SELECT')
            if not e.get('cont_grid'):
                fails.append(f'{P}: no CONT grid'); continue
            cells = e['cont_grid']['cells']; ses = e['cont_grid']['cell_se']
            vals = np.array([_num(cells[k]) for k in sorted(cells)])
            sd = math.sqrt(vals.var(ddof=1) + float(np.mean([_num(ses[k]) ** 2 for k in cells])))
            lows.append(float(vals.mean() - STABILITY_Z * sd))
            ratios += list(e['ms_reserve_ratios'])
        rec = {'recipe': c['recipe'], 'gate_failures': fails, 'eligible': not fails,
               'score': min(lows) if len(lows) == 2 else -math.inf, 'ratios': ratios, 'c': c}
        rec['label'] = 'ELIGIBLE' if not fails else 'GATE_FAILED'
        recs.append(rec)
    el = [r for r in recs if r.get('eligible')]
    if not el:
        return {'selected': None, 'outcome': 'NO_RECIPE', 'records': [{k: v for k, v in r.items() if k != 'c'} for r in recs]}
    best = max(r['score'] for r in el)
    tie = [r for r in el if r['score'] >= best - near_tie]
    rank = {'REFIT': 0, 'FROZEN': 1, 'full': 0, 'ensemble': 1, 'LOC': 0, 'PAIR': 1}

    def cmp(a, b):
        x = _leximin_cmp(a['ratios'], b['ratios'])
        if x:
            return x
        for key, order in (('routing', rank), ('deployment', rank), ('rule', {r: i for i, r in enumerate(RULE_ORDER)}),
                           ('arm', rank)):
            ra, rb = order[a['c'][key]], order[b['c'][key]]
            if ra != rb:
                return -1 if ra < rb else 1
        ea, eb = a['c'].get('entry', math.inf), b['c'].get('entry', math.inf)
        return -1 if ea < eb else (1 if ea > eb else 0)
    tie.sort(key=functools.cmp_to_key(cmp))
    for r in el:
        r['label'] = 'SELECTED_RECIPE' if r is tie[0] else ('NEAR_TIE_NOT_CHOSEN' if r in tie else 'ELIGIBLE_NOT_CHOSEN')
    return {'selected': tie[0]['recipe'], 'outcome': 'RECIPE_SELECTED', 'near_tie_set': [r['recipe'] for r in tie],
            'best_score': best, 'records': [{k: v for k, v in r.items() if k != 'c'} for r in recs]}


def compare_rules(a, b):
    """Matched comparison of two rules on the same fitted directions and partition. a / b: {'rule', 'selected',
    'reserve_crossing' (realized on MODEL_SELECT or CHECK), 'pair_auc' (reader grid mean at the selected strength, or None)}.
    A stricter rule never 'improves the method' merely by shrinking: identity earns nothing, and a smaller strength counts
    only if it avoids a realized crossing that the other rule incurs (its protection cost is reported)."""
    sa, sb = float(a['selected']), float(b['selected'])
    pa, pb = a.get('pair_auc'), b.get('pair_auc')
    prot = (pb - pa) if (pa is not None and pb is not None) else None
    if sa == sb:
        cat = 'SAME_STRENGTH'
    elif a['reserve_crossing']:
        cat = 'A_CROSSES'
    elif sa == 0.0:
        cat = 'A_IDENTITY_NO_IMPROVEMENT'
    elif sa < sb and b['reserve_crossing']:
        cat = 'A_SAFER_AVOIDS_CROSSING'
    elif sa > sb:
        cat = 'A_STRONGER_WITHOUT_CROSSING'
    else:
        cat = 'A_WEAKER_NO_GAIN'
    improves = cat in ('A_SAFER_AVOIDS_CROSSING', 'A_STRONGER_WITHOUT_CROSSING')
    return {'category': cat, 'a_improves': improves, 'protection_change_b_minus_a': prot}


def claim_ladder(ev):
    """Generic claim ladder. ev keys (booleans or labels computed by the registered procedures):
      'arm': the selected recipe's arm (PAIR or LOC; the LOC control is required only for a PAIR recipe)
      'prediction_labels': {partition: PREDICTION_IMPROVED / NOT_IMPROVED} for D0, D1, R0 and R1 (all four required)
      'utility': {'R0': {'nonzero', 'ms_reserve_kept', 'check_no_crossing', 'point_pass'}, 'R1': {...}}
      'matched_controls': {control: STABLE_POSITIVE / ... } over the R cells for CONT, LOC, COMP, W
      'frontier': {'R0': bool, 'R1': bool} (stable frontier improvement, RULES.md section 9)
      'benchmark': {'point_and_gate': bool, 'superiority_all_required': bool, 'target_label': DEV_GRID_TARGET_*}
    Rungs 1-5 are claimable only in order; rung 6 is fixed: joint-design advantage NOT_CLAIMABLE, novelty NOT_CLAIMED."""
    pl = ev.get('prediction_labels') or {}
    r1 = all(pl.get(P) == 'PREDICTION_IMPROVED' for P in ('D0', 'D1', 'R0', 'R1'))
    ut = ev.get('utility') or {}
    r2 = all(all((ut.get(P) or {}).get(k) for k in ('nonzero', 'ms_reserve_kept', 'check_no_crossing', 'point_pass'))
             for P in ('R0', 'R1'))
    mc = ev.get('matched_controls') or {}
    controls = ('CONT', 'LOC', 'COMP', 'W') if ev.get('arm', 'PAIR') == 'PAIR' else ('CONT', 'COMP', 'W')
    r3 = all(mc.get(x) == 'STABLE_POSITIVE' for x in controls)
    fr = ev.get('frontier') or {}
    r4 = bool(fr.get('R0')) and bool(fr.get('R1'))
    bm = ev.get('benchmark') or {}
    r5 = bool(bm.get('point_and_gate')) and bool(bm.get('superiority_all_required'))
    met = [r1, r2, r3, r4, r5]
    names = ['IMPROVED_COST_PREDICTION', 'UTILITY_RETAINED_AT_NONZERO_CORRECTION',
             'INCREMENTAL_PROTECTION_BEYOND_MATCHED_CONTROLS', 'STABLE_FRONTIER_IMPROVEMENT', 'ORIGINAL_BENCHMARK_COMPETITIVENESS']
    out, ok, highest = {}, True, 0
    for i, (n, m) in enumerate(zip(names, met), start=1):
        out[n] = 'CLAIMABLE' if (ok and m) else ('NOT_MET' if ok else ('MET_BUT_NOT_CLAIMABLE_LOWER_RUNG_UNMET' if m
                                                                        else 'NOT_MET_LOWER_RUNG_UNMET'))
        ok = ok and m
        highest = i if ok else highest
    out['MATERIAL_TARGET_0020'] = bm.get('target_label', 'DEV_GRID_TARGET_NOT_MET')
    out['JOINT_DESIGN_ADVANTAGE'] = 'NOT_CLAIMABLE'
    out['NOVELTY'] = 'NOT_CLAIMED'
    out['highest_claimable_rung'] = highest
    out['note'] = ('the 0.020 target is labelled separately from positive superiority; Track F is a coalition-aware occupation '
                   'correction, never a joint two-output design')
    return out


def reserve_qualified_frontier(points, k_ratio=K_RESERVE):
    """points: list of {'id', 'kind' ('candidate' / 'reference' / 'control'), 'min_ratio': {0: r, 1: r} (whole-release reserve
    ratios, same rule for every point), 'pair_auc': {0: a, 1: a}}. A point is reserve-qualified iff its minimum ratio >= 2 on
    both seeds. Returns qualification per point and the Pareto frontier of qualified points (lower mean pair AUC, higher
    minimum ratio). Historical references that fail are listed, not dropped."""
    recs = []
    for p in points:
        mr = min(_num(p['min_ratio'][s]) for s in (0, 1))
        pa = float(np.mean([_num(p['pair_auc'][s]) for s in (0, 1)]))
        recs.append({'id': p['id'], 'kind': p['kind'], 'min_ratio': mr, 'pair_auc_mean': pa, 'qualified': bool(mr >= k_ratio)})
    q = [r for r in recs if r['qualified']]
    front = [r['id'] for r in q if not any((o['pair_auc_mean'] <= r['pair_auc_mean'] and o['min_ratio'] >= r['min_ratio'])
                                             and (o['pair_auc_mean'] < r['pair_auc_mean'] or o['min_ratio'] > r['min_ratio'])
                                             for o in q)]
    return {'points': recs, 'frontier': front, 'references_failing_reserve': [r['id'] for r in recs
                                                                            if r['kind'] == 'reference' and not r['qualified']],
            'note': 'a positive reserve-qualified result is not superiority over every original reference'}
