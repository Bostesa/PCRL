"""Frozen statistical rules of the PCRL controlled training-size campaign (`pcrl_controlled_training_size_v1`; statistical
reviewer; written before any new real-label computation, fit or outcome of this campaign).

Human-readable rules: results/pcrl_controlled_training_size_v1/roles/statistical_reviewer/RULES.md
Machine-readable constants: .../RULES.json (schema TSZ1-rules-v1); `load_rules()` refuses a file that disagrees with this module.
Invented-data tests: `python3 -m tsz.test_rules`.

Pure functions on plain dicts / floats (numpy arrays only in `reproduction_record`). Nothing here reads data or labels; the only
files read are this campaign's RULES.json and, through the imported `ohn.rules` (reused unchanged for the reserve), the source rule
file OHN1 RULES.json (SHA-256 e18f1c68...). Every rule is a DEVELOPMENT rule on adaptively used records; no original limit changes.

API (inputs are plain dicts; versions are the ints 0 and 1; orderings 'O0'..'O3'):
  load_rules()                                   constants + '_sha256'
  occupation_reserve(evals_by_version)           ohn.rules.reserve unchanged; per-version occupation-side status, ratios, slack rho;
                                                 income conditions separately. evals_by_version = {v: evaluator utility dict}
  summarize_orderings(values, ses)               mean, SD_ord, se_eval, se_comb, RESOLVED_POSITIVE / _NEGATIVE / UNRESOLVED
                                                 (values in the orientation to be labelled: positive = the effect of interest)
  across_versions({v: per-version label})        POSITIVE_BOTH_VERSIONS / NEGATIVE_BOTH_VERSIONS / VERSION_DEPENDENT / NOT_RESOLVED
  size_summary(changes)                          changes[contrast][metric][v] = {'values': {o: loss(larger) - loss(smaller)},
                                                 'se': {o: paired SE}} -> labels per contrast / metric + registered 'size_effect'
  margin_pattern(status)                         status = {'W100': {v: bool}, 'S50': {o: {v: bool}}, 'S70': {...}} (occupation-side
                                                 reserve met) -> FULL_ONLY / ALL_SIZES / FULL_NOT_REPRODUCED / INCOMPLETE
  size_explanation(pattern, size_effect)         SUPPORTED_ON_THIS_POOL / NOT_SUPPORTED / NOT_APPLICABLE_FULL_NOT_REPRODUCED /
                                                 UNRESOLVED; interpretation(...) gives the IR1-IR4 readings
  control_level(diffs) / control_did(diffs)      loss(A1) - loss(control) / change_A1 - change_control (loss orientation)
  control_explains(recipe, control, did)         CONTROL_EXPLAINS_GAIN / RECIPE_SPECIFIC_SIZE_GAIN / NO_RECIPE_SIZE_GAIN / ...
  control_also_meets_reserve(recipe, control)    equal-reserve flag at one size
  reproduction_record(...) / reproduction(...)   REPRODUCED_BITWISE / REPRODUCED_WITHIN_TOLERANCE / NOT_REPRODUCED / INCOMPLETE
  encoding_summary(E)                            E[metric] = {'values': {o: loss(FIT-only) - loss(admitted)}, 'se': {o: SE}}
  trigger(evals_by_version, deployment, h3_ok)   the registered trigger decision record (T0-T3)
  allowances()                                   {v: {statistic: allowance}} (lambda x min mapped OHN1 floor SE)
  select_strength(cost_table, version)           selected strength + audit trail; cost_table = {s: {stat: {'estimate', 'se'}}}
  stability_label(cells, cell_se, threshold)     the source 8-cell screen; target_label(...) for the 0.020 criterion
  recipient_gate(cells)                          old per-cell recipient gate (<= 0.010 every cell)
  reader_seed_check(seeds, recipe_checks)        READER_SEEDS_EFFECTIVE / READER_SEEDS_NOT_EFFECTIVE (+ NONDETERMINISTIC_RECIPE)
  seed_distinct_pairs(cells)                     replication actually supplied by the reader seeds (qualifier)
  equal_reserve_table(points)                    one reserve for every method; Pareto frontier of qualified points
  claim_labels(ev)                               C1-C8, arm qualification, closure outcome
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

from ohn import rules as OR                      # the source reserve rule, reused unchanged

RULES_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_controlled_training_size_v1' / 'roles'
              / 'statistical_reviewer' / 'RULES.json')
SCHEMA = 'TSZ1-rules-v1'
OHN_RULE_SHA256 = 'e18f1c68f4646ce9d1395960ab1a364a80c12541b815a55e0ba06b65074cbbb3'
CAPS = {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005}
GUARD_CAP = 0.005
RECIPIENT_LIMIT = 0.010
TARGET = 0.020
K_RESERVE = 2.0
FLOOR_HOUSEHOLDS = 5306
VERSIONS = (0, 1)
ORDERINGS = ('O0', 'O1', 'O2', 'O3')
SIZES = (50, 70, 100)
METRICS = ('acc_loss', 'nll', 'brier', 'men_brier', 'women_brier')
CONTRASTS = {'50->70': (50, 70), '70->100': (70, 100), '50->100': (50, 100)}
PRIMARY_ENDPOINT = ('50->100', 'men_brier')
Z = 2.0                                          # contrasts, selection and stability all use the two-SE convention
OCC_SIDE = OR.OCC_SIDE
STRENGTHS = (0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
ALLOWANCE_LAMBDA = 1.0
STATS = ('men_brier', 'women_brier', 'brier', 'nll', 'acc_loss')
SLACK_OF = {'men_brier': ('guard|sex0',), 'women_brier': ('guard|sex1',),
            'brier': ('occupation|U0|brier_excess', 'occupation|Ucal|brier_excess'),
            'nll': ('occupation|U0|nll_excess', 'occupation|Ucal|nll_excess'),
            'acc_loss': ('occupation|U0|accuracy_loss', 'occupation|Ucal|accuracy_loss')}
PRIMARY_ARM, SECONDARY_ARM = 'LOC', 'PAIR'
GUARD_NONE_TARGETS = {'men': 0.005, 'women': 0.005}  # fixed train_correction `targets` under guard 'none' (no MODEL_SELECT input)
DEPLOYMENT_RECORD = 'results/pcrl_controlled_training_size_v1/roles/verifier/DEPLOY_W100.json'
DEPLOYMENT_FIELDS = ('bitwise', 'refusals_all_pass')
CELLS = tuple(f'S{v}R{r}k{k}' for v in VERSIONS for r in (1, 2) for k in (0, 1))
SEED_DEPENDENT = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
SEED_INVARIANT = ('OUT_LR', 'OUT_HGB', 'LOG_LR', 'DEC_LR', 'CONF_HGB', 'LOOKUP', 'HGB15', 'HGB31', 'CATI01', 'CATI1')
MODEL_SEED_BASE = 2026101205                     # task_conditioned_representation.attacks (OHN1 / CTS1 grid seeds derive from it)
REPRO_TOL = 1e-12
REQUIRED = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')
TARGET_REFS = ('NEURAL_RAW', 'H_LJ')
SCOPE = ('Occupation-only optimization is not a new two-output joint method.',
         'Lower recovery for declared readers is not arbitrary-reader protection.',
         'Do not claim a population learning curve, independent replication or proof that size caused every historical difference.',
         'reader recipes and overlapping splits are not independent datasets')

_cache = {}


# ---------------------------------------------------------------- the registered rule file
def load_rules(path=RULES_PATH):
    """RULES.json with its SHA-256; refuses a schema or a constant that disagrees with this module or with the OHN1 source rule."""
    p = Path(path)
    if str(p) not in _cache:
        raw = p.read_bytes()
        r = json.loads(raw)
        if r.get('schema') != SCHEMA:
            raise ValueError(f'unexpected rules schema {r.get("schema")!r}')
        ohn = OR.load_rules()
        if ohn.get('_sha256') != OHN_RULE_SHA256 or r['reserve']['rule_sha256'] != OHN_RULE_SHA256:
            raise ValueError('the OHN1 source rule file differs from the registered one')
        fl = ohn['reserve']['floor']['se']
        sel = r['selection']
        derived = {str(v): {j: ALLOWANCE_LAMBDA * min(float(fl[str(v)][c]) for c in cs) for j, cs in SLACK_OF.items()}
                   for v in VERSIONS}
        checks = {
            'caps': (r['original_limits']['caps'], CAPS),
            'guard cap': (r['original_limits']['within_sex_guard_cap'], GUARD_CAP),
            'recipient limit': (r['original_limits']['recipient_limit_auc'], RECIPIENT_LIMIT),
            'target': (r['original_limits']['material_target_auc'], TARGET),
            'k': (r['reserve']['multiplier_k'], K_RESERVE),
            'k equals OHN1': (float(ohn['reserve']['multiplier_k']), K_RESERVE),
            'floor households': (r['reserve']['floor_households'], FLOOR_HOUSEHOLDS),
            'occupation side': (tuple(r['reserve']['occupation_side_conditions']), tuple(OCC_SIDE)),
            'floor copy': ({v: d for v, d in r['reserve']['floor_se_occupation_side'].items()},
                           {str(v): {c: fl[str(v)][c] for c in OCC_SIDE} for v in VERSIONS}),
            'strengths': (tuple(float(s) for s in r['correction']['strengths']), STRENGTHS),
            'primary arm': (r['correction']['primary_arm'], PRIMARY_ARM),
            'lambda': (sel['allowance_lambda'], ALLOWANCE_LAMBDA),
            'allowance': (sel['allowance'], derived),
            'selection z': (sel['z'], Z),
            'size z': (r['size_design']['z'], Z),
            'stability z': (r['readers']['z'], Z),
            'cells': (tuple(r['readers']['cells']), CELLS),
            'seed dependent': (tuple(r['readers']['seed_dependent_recipes']), SEED_DEPENDENT),
            'required': (tuple(r['comparators']['required']), REQUIRED),
            'target references': (tuple(r['comparators']['target_references']), TARGET_REFS),
            'tolerance': (r['reproduction']['max_abs_tolerance'], REPRO_TOL),
            'primary endpoint': ((r['size_design']['primary_endpoint']['contrast'], r['size_design']['primary_endpoint']['metric']),
                                 PRIMARY_ENDPOINT),
            'guard none targets': (r['correction']['guard_none_targets_constant'], GUARD_NONE_TARGETS),
            'deployment record': ((r['trigger']['deployment_record'], tuple(r['trigger']['deployment_record_fields'])),
                                  (DEPLOYMENT_RECORD, DEPLOYMENT_FIELDS)),
            'closure uses': (r['claims']['closure_uses'], 'C2'),
        }
        bad = [k for k, (a, b) in checks.items() if a != b]
        if bad:
            raise ValueError(f'RULES.json disagrees with tsz.rules on {bad}')
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


# ---------------------------------------------------------------- 1. reserve (ohn.rules.reserve unchanged)
def occupation_reserve(evals_by_version):
    """Occupation-side reserve per version from the unchanged source rule. evals_by_version = {v: evaluator utility dict (or
    {'utility': ...})} with all 28 conditions of version v. Returns per-version status OCC_RESERVE_MET / OCC_RESERVE_NOT_MET, the 8
    records (with usable slack rho = margin - 2 se_used), minimum ratio and binding condition; income conditions separately;
    across versions OCC_RESERVE_MET_BOTH / OCC_RESERVE_ONE_VERSION / OCC_RESERVE_NOT_MET (INCOMPLETE if a version is missing)."""
    res = OR.reserve(evals_by_version)
    per = {}
    for v in res['seeds']:
        conds = res['per_seed'][v]['conditions']
        occ = {}
        for c in OCC_SIDE:
            rec = dict(conds[c])
            rec['rho'] = (rec['margin'] - K_RESERVE * rec['se_used']) if rec.get('margin') is not None else None
            rec['ratio_own_se'] = (rec['margin'] / rec['own_se']) if (rec.get('margin') is not None and _fin(rec['own_se'])
                                                                       and rec['own_se'] > 0) else None
            occ[c] = rec
        failed = [c for c in OCC_SIDE if not occ[c]['met']]
        ratios = {c: (-math.inf if occ[c]['ratio'] is None else occ[c]['ratio']) for c in OCC_SIDE}
        binding = min(ratios, key=ratios.get)
        inc = {c: conds[c] for c in OR.INCOME}
        inc_r = [(-math.inf if r['ratio'] is None else r['ratio']) for r in inc.values()]
        per[v] = {'status': 'OCC_RESERVE_NOT_MET' if failed else 'OCC_RESERVE_MET', 'met': not failed, 'failed': failed,
                  'point_pass': all(occ[c]['point_pass'] for c in OCC_SIDE),
                  'point_failed': [c for c in OCC_SIDE if not occ[c]['point_pass']],
                  'min_ratio': ratios[binding], 'binding': binding, 'conditions': occ,
                  'income': {'min_income_ratio': min(inc_r), 'income_reserve_met': all(r['met'] for r in inc.values()),
                             'income_point_pass': all(r['point_pass'] for r in inc.values())}}
    if sorted(per) != list(VERSIONS):
        status = 'INCOMPLETE'
    else:
        n = sum(per[v]['met'] for v in VERSIONS)
        status = 'OCC_RESERVE_MET_BOTH' if n == 2 else ('OCC_RESERVE_ONE_VERSION' if n == 1 else 'OCC_RESERVE_NOT_MET')
    return {'status': status, 'per_version': per, 'whole_release_status': res['status'], 'ohn_rule_sha256': res['rule_sha256'],
            'note': 'development heuristic on adaptively used MODEL_SELECT data; not a new cap or population guarantee'}


# ---------------------------------------------------------------- 3. contrasts summarized over orderings
def summarize_orderings(values, ses, z=Z):
    """values / ses: {ordering: float} (or a single-entry dict for one fit). Positive values = the effect of interest.
    mean, SD_ord (df n-1), se_eval = sqrt(mean se^2), se_comb = sqrt(se_eval^2 + SD_ord^2 / n); RESOLVED_POSITIVE iff
    mean - z se_comb > 0 and every value > 0; RESOLVED_NEGATIVE iff mean + z se_comb < 0 and every value < 0; else UNRESOLVED."""
    if not isinstance(values, dict) or not values or set(values) != set(ses or {}):
        return {'label': 'INCOMPLETE', 'error': 'values and SEs must be non-empty dicts over the same orderings'}
    xs = {o: _num(values[o]) for o in values}
    ss = {o: _num(ses[o]) for o in values}
    if not all(_fin(x) for x in xs.values()) or not all(_fin(s) and s >= 0 for s in ss.values()):
        return {'label': 'INCOMPLETE', 'error': 'non-finite value or invalid SE'}
    n = len(xs)
    mean = sum(xs.values()) / n
    sd_ord = math.sqrt(sum((x - mean) ** 2 for x in xs.values()) / (n - 1)) if n > 1 else 0.0
    se_eval = math.sqrt(sum(s ** 2 for s in ss.values()) / n)
    se_comb = math.sqrt(se_eval ** 2 + sd_ord ** 2 / n)
    npos, nneg = sum(x > 0 for x in xs.values()), sum(x < 0 for x in xs.values())
    lower, upper = mean - z * se_comb, mean + z * se_comb
    if lower > 0 and npos == n:
        label = 'RESOLVED_POSITIVE'
    elif upper < 0 and nneg == n:
        label = 'RESOLVED_NEGATIVE'
    else:
        label = 'UNRESOLVED'
    reversal = 0 < npos < n or (nneg and npos)
    return {'label': label, 'mean': mean, 'sd_ord': sd_ord, 'se_eval': se_eval, 'se_comb': se_comb, 'lower': lower, 'upper': upper,
            'n': n, 'n_positive': npos, 'n_negative': nneg, 'sign_reversal': bool(reversal),
            'sign_reversals': (n - npos) if mean > 0 else (n - nneg) if mean < 0 else n,
            'variance_share_ordering': (sd_ord ** 2 / (sd_ord ** 2 + n * se_eval ** 2)) if (sd_ord > 0 or se_eval > 0) else None,
            'min': min(xs.values()), 'max': max(xs.values()), 'values': xs, 'ses': ss}


def across_versions(labels):
    """{v: per-version label} -> POSITIVE_BOTH_VERSIONS / NEGATIVE_BOTH_VERSIONS / VERSION_DEPENDENT / NOT_RESOLVED / INCOMPLETE."""
    if not all(_has(labels, v) for v in VERSIONS):
        return 'INCOMPLETE'
    ls = [_vkey(labels, v) for v in VERSIONS]
    if 'INCOMPLETE' in ls:
        return 'INCOMPLETE'
    if all(x == 'RESOLVED_POSITIVE' for x in ls):
        return 'POSITIVE_BOTH_VERSIONS'
    if all(x == 'RESOLVED_NEGATIVE' for x in ls):
        return 'NEGATIVE_BOTH_VERSIONS'
    if any(x in ('RESOLVED_POSITIVE', 'RESOLVED_NEGATIVE') for x in ls):
        return 'VERSION_DEPENDENT'
    return 'NOT_RESOLVED'


def _negate(d):
    return {k: -_num(x) for k, x in d.items()}


def _summarize_block(block, flip):
    """block[v] = {'values': {o: x}, 'se': {o: s}}; flip=True converts loss orientation to improvement orientation."""
    per = {}
    for v in VERSIONS:
        if not _has(block, v):
            per[v] = {'label': 'INCOMPLETE', 'error': f'version {v} missing'}
            continue
        e = _vkey(block, v)
        vals = _negate(e.get('values') or {}) if flip else dict(e.get('values') or {})
        per[v] = summarize_orderings(vals, e.get('se') or {})
    return {'per_version': per, 'across': across_versions({v: per[v]['label'] for v in VERSIONS})}


def size_summary(changes):
    """changes[contrast][metric][v] = {'values': {o: loss(larger fit) - loss(smaller fit)}, 'se': {o: paired SE}} (loss
    orientation: negative = the larger training set is better). Labels use the improvement g = -change. Returns per contrast and
    metric the per-version summaries and the across-version label, plus the registered primary 'size_effect' label
    (50->100 men's Brier): SIZE_GAIN_RESOLVED / SIZE_LOSS_RESOLVED / SIZE_EFFECT_UNRESOLVED / INCOMPLETE."""
    out = {}
    for c in CONTRASTS:
        out[c] = {}
        for m in METRICS:
            blk = ((changes or {}).get(c) or {}).get(m)
            out[c][m] = _summarize_block(blk, flip=True) if blk is not None else {'per_version': {}, 'across': 'INCOMPLETE'}
    prim = out[PRIMARY_ENDPOINT[0]][PRIMARY_ENDPOINT[1]]['across']
    eff = {'POSITIVE_BOTH_VERSIONS': 'SIZE_GAIN_RESOLVED', 'NEGATIVE_BOTH_VERSIONS': 'SIZE_LOSS_RESOLVED',
           'INCOMPLETE': 'INCOMPLETE'}.get(prim, 'SIZE_EFFECT_UNRESOLVED')
    return {'contrasts': out, 'primary_endpoint': {'contrast': PRIMARY_ENDPOINT[0], 'metric': PRIMARY_ENDPOINT[1], 'across': prim},
            'size_effect': eff,
            'secondary_resolved': sorted(f'{c}|{m}|{out[c][m]["across"]}' for c in CONTRASTS for m in METRICS
                                         if out[c][m]['across'] in ('POSITIVE_BOTH_VERSIONS', 'NEGATIVE_BOTH_VERSIONS')
                                         and (c, m) != PRIMARY_ENDPOINT),
            'note': 'within-recipe effect on this finite pool, evaluated on adaptively used MODEL_SELECT; not a population learning curve'}


def margin_pattern(status):
    """status = {'W100': {v: bool}, 'S50': {o: {v: bool}}, 'S70': {o: {v: bool}}}: occupation-side reserve met (k = 2).
    FULL_ONLY (W100 met on both versions and some 50% cell fails; sub-flag FULL_ONLY_70_RETAINS when all 70% cells meet it),
    ALL_SIZES (every 50% and 70% cell meets it too), FULL_NOT_REPRODUCED (W100 fails on some version), INCOMPLETE."""
    try:
        w = {v: _vkey(status['W100'], v) for v in VERSIONS}
        s50 = {(o, v): _vkey(status['S50'][o], v) for o in ORDERINGS for v in VERSIONS}
        s70 = {(o, v): _vkey(status['S70'][o], v) for o in ORDERINGS for v in VERSIONS}
    except (KeyError, TypeError):
        return {'pattern': 'INCOMPLETE', 'error': 'a W100, S50 or S70 cell is missing'}
    if not all(isinstance(x, bool) for x in list(w.values()) + list(s50.values()) + list(s70.values())):
        return {'pattern': 'INCOMPLETE', 'error': 'statuses must be booleans (a missing fit is INCOMPLETE, never imputed)'}
    met100 = all(w.values())
    n50, n70 = sum(s50.values()), sum(s70.values())
    flags = []
    if not met100:
        pat = 'FULL_NOT_REPRODUCED'
    elif n50 < 8:
        pat = 'FULL_ONLY'
        if n70 == 8:
            flags.append('FULL_ONLY_70_RETAINS')
    elif n70 == 8:
        pat = 'ALL_SIZES'
    else:
        pat = 'FULL_ONLY'
        flags.append('NON_NESTED_PATTERN_50_RETAINS_70_LOSES')
    return {'pattern': pat, 'met100': met100, 'n50_met_of_8': n50, 'n70_met_of_8': n70, 'flags': flags,
            'w100_by_version': w}


def size_explanation(pattern, size_effect):
    """SUPPORTED_ON_THIS_POOL (FULL_ONLY and SIZE_GAIN_RESOLVED), NOT_SUPPORTED (ALL_SIZES, or SIZE_LOSS_RESOLVED),
    NOT_APPLICABLE_FULL_NOT_REPRODUCED, INCOMPLETE, UNRESOLVED otherwise."""
    if pattern == 'INCOMPLETE' or size_effect == 'INCOMPLETE':
        return 'INCOMPLETE'
    if pattern == 'FULL_NOT_REPRODUCED':
        return 'NOT_APPLICABLE_FULL_NOT_REPRODUCED'
    if pattern == 'ALL_SIZES' or size_effect == 'SIZE_LOSS_RESOLVED':
        return 'NOT_SUPPORTED'
    if pattern == 'FULL_ONLY' and size_effect == 'SIZE_GAIN_RESOLVED':
        return 'SUPPORTED_ON_THIS_POOL'
    return 'UNRESOLVED'


def interpretation(pattern, size_effect, control_explains_by_metric=None, control_reserve_by_size=None):
    """The four registered readings (all computed, all reported). control_explains_by_metric = {metric: control_explains label};
    control_reserve_by_size = {size: bool CONTROL_ALSO_MEETS_RESERVE}."""
    expl = size_explanation(pattern, size_effect)
    out = []
    if pattern == 'FULL_ONLY':
        out.append(('IR1', 'Full-data refitting reproduces the occupation-side margin while smaller fits lose it: the previous '
                           'reduced-training experiment did not settle full-data feasibility.'))
    if pattern == 'FULL_NOT_REPRODUCED':
        out.append(('IR2', 'The source full-data margin does not reproduce under the fixed recipe: inspect exact implementation and '
                           'exposure differences (T*, dictionary, fold tags, training rows, library versions, encoding); no retuning.'))
    ce = sorted(m for m, lab in (control_explains_by_metric or {}).items() if lab == 'CONTROL_EXPLAINS_GAIN')
    cr = sorted(str(f) for f, ok in (control_reserve_by_size or {}).items() if ok)
    if ce or cr:
        out.append(('IR3', 'An ordinary equally exposed control explains the gain: same training-size gain on '
                           f'{ce or "no metric"}; control also meets the occupation-side reserve at sizes {cr or "none"}.'))
    if expl == 'UNRESOLVED' or size_effect == 'SIZE_EFFECT_UNRESOLVED':      # RULES 3.6: IR4 on SIZE_EFFECT_UNRESOLVED (any pattern)
        out.append(('IR4', 'Uncertainty leaves the size explanation unresolved.'))
    if pattern == 'ALL_SIZES':
        out.append(('ALL_SIZES', 'The margin is retained at 50%, 70% and 100% of this pool; the reduced-training loss of the CTS1 '
                                 'partitions is not explained by training size between 50% and 100% under this design.'))
    return {'size_explanation': expl, 'readings': [{'rule': r, 'text': t} for r, t in out], 'scope': SCOPE[2]}


# ---------------------------------------------------------------- 4. control comparison
def control_level(diffs):
    """diffs[metric][v] = {'values': {o: loss(A1_PO_16) - loss(control)}, 'se': {o: SE}} at one size (one entry per version at
    100%). Improvement = -diff (positive = recipe better). Labels RECIPE_BETTER_RESOLVED / RECIPE_WORSE_RESOLVED /
    VERSION_DEPENDENT / NOT_RESOLVED / INCOMPLETE."""
    tr = {'POSITIVE_BOTH_VERSIONS': 'RECIPE_BETTER_RESOLVED', 'NEGATIVE_BOTH_VERSIONS': 'RECIPE_WORSE_RESOLVED'}
    out = {}
    for m in METRICS:
        blk = (diffs or {}).get(m)
        s = _summarize_block(blk, flip=True) if blk is not None else {'per_version': {}, 'across': 'INCOMPLETE'}
        s['label'] = tr.get(s['across'], s['across'])
        out[m] = s
    return out


def control_did(diffs):
    """diffs[metric][v] = {'values': {o: change_A1 - change_control}, 'se': {o: SE of the per-person double difference}} for one
    contrast (loss orientation). Returns the improvement-oriented summaries (positive = the recipe gains more from size)."""
    return {m: (_summarize_block((diffs or {})[m], flip=True) if (diffs or {}).get(m) is not None
                else {'per_version': {}, 'across': 'INCOMPLETE'}) for m in METRICS}


def control_explains(recipe_gain, control_gain, did):
    """Across-version labels of the 50->100 improvement for one metric: recipe, control and DiD."""
    if 'INCOMPLETE' in (recipe_gain, control_gain, did):
        return 'INCOMPLETE'
    if recipe_gain != 'POSITIVE_BOTH_VERSIONS':
        return 'NO_RECIPE_SIZE_GAIN'
    if did == 'POSITIVE_BOTH_VERSIONS':
        return 'RECIPE_SPECIFIC_SIZE_GAIN'
    if control_gain == 'POSITIVE_BOTH_VERSIONS':
        return 'CONTROL_EXPLAINS_GAIN'
    return 'RECIPE_GAIN_CONTROL_UNRESOLVED'


def control_also_meets_reserve(recipe_status, control_status):
    """{cell: bool} occupation-side reserve met for the recipe and the control at the same (ordering, version) cells of one size.
    True iff the control meets it at every cell where the recipe meets it (and the recipe meets it somewhere)."""
    cells = [c for c, ok in recipe_status.items() if ok]
    return bool(cells) and all(control_status.get(c) is True for c in cells)


# ---------------------------------------------------------------- 5. reproduction
def reproduction_record(p_new, p_src, tstar_new=None, tstar_src=None):
    """Arrays in (float64 probability arrays of one role), record out: sha_equal, max_abs_diff, decisions_equal, tstar_equal."""
    import numpy as np
    a = np.ascontiguousarray(np.asarray(p_new, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(p_src, dtype=np.float64))
    if a.shape != b.shape:
        return {'sha_equal': False, 'max_abs_diff': math.inf, 'decisions_equal': False, 'tstar_equal': tstar_new == tstar_src,
                'shape_mismatch': [list(a.shape), list(b.shape)]}
    return {'sha_equal': hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest(),
            'max_abs_diff': float(np.max(np.abs(a - b))) if a.size else 0.0,
            'decisions_equal': bool(np.array_equal(a.argmax(1), b.argmax(1))), 'tstar_equal': tstar_new == tstar_src}


def reproduction(records_by_version, tol=REPRO_TOL):
    """records_by_version = {v: record or list of records (one per role)}. REPRODUCED_BITWISE iff every record is SHA-equal with
    equal T*; REPRODUCED_WITHIN_TOLERANCE iff every record has max |diff| <= tol, identical decisions and equal T*;
    NOT_REPRODUCED otherwise; INCOMPLETE if a version is missing."""
    per = {}
    for v in VERSIONS:
        if not _has(records_by_version, v):
            per[v] = 'INCOMPLETE'
            continue
        recs = _vkey(records_by_version, v)
        recs = recs if isinstance(recs, list) else [recs]
        if not recs:
            per[v] = 'INCOMPLETE'
        elif all(r.get('sha_equal') is True and r.get('tstar_equal') is True for r in recs):
            per[v] = 'REPRODUCED_BITWISE'
        elif all(_fin(_num(r.get('max_abs_diff'))) and _num(r.get('max_abs_diff')) <= tol and r.get('decisions_equal') is True
                 and r.get('tstar_equal') is True for r in recs):
            per[v] = 'REPRODUCED_WITHIN_TOLERANCE'
        else:
            per[v] = 'NOT_REPRODUCED'
    order = ('INCOMPLETE', 'NOT_REPRODUCED', 'REPRODUCED_WITHIN_TOLERANCE', 'REPRODUCED_BITWISE')
    label = min(per.values(), key=order.index)
    return {'label': label, 'per_version': per}


def margin_reproduced(occ_reserve_record):
    """MARGIN_REPRODUCED iff the newly refitted W100 meets the occupation-side reserve on both versions (separate from bitwise)."""
    return 'MARGIN_REPRODUCED' if occ_reserve_record.get('status') == 'OCC_RESERVE_MET_BOTH' else 'MARGIN_NOT_REPRODUCED'


# ---------------------------------------------------------------- 6. encoding sensitivity (version 0 only)
def encoding_summary(E):
    """E[metric] = {'values': {o: loss(FIT-only dictionary) - loss(admitted dictionary)}, 'se': {o: SE}} at S50(o), version 0.
    Positive = the admitted dictionary is better. ENCODING_EFFECT_RESOLVED / _RESOLVED_NEGATIVE / _UNRESOLVED / INCOMPLETE."""
    tr = {'RESOLVED_POSITIVE': 'ENCODING_EFFECT_RESOLVED', 'RESOLVED_NEGATIVE': 'ENCODING_EFFECT_RESOLVED_NEGATIVE',
          'UNRESOLVED': 'ENCODING_EFFECT_UNRESOLVED', 'INCOMPLETE': 'INCOMPLETE'}
    out = {}
    for m in METRICS:
        e = (E or {}).get(m)
        s = summarize_orderings((e or {}).get('values'), (e or {}).get('se')) if e else {'label': 'INCOMPLETE'}
        s['encoding_label'] = tr[s['label']]
        out[m] = s
    return out


def encoding_share(mean_E, mean_size_gain):
    """Descriptive share of the 50->100 improvement (version 0) matched by the encoding effect; None unless both are positive."""
    if _fin(_num(mean_E), _num(mean_size_gain)) and mean_E > 0 and mean_size_gain > 0:
        return mean_E / mean_size_gain
    return None


# ---------------------------------------------------------------- 7. trigger
def trigger(evals_by_version, deployment, h3_income_verified):
    """The registered trigger (RULES.md section 7), evaluated once at task-stage close.
    evals_by_version = {v: evaluator utility dict of the complete release (H3 income S<v> + newly refitted W100_v occupation)};
    deployment = {v: {'bitwise': bool, 'refusals_all_pass': bool}}: the per-version entries ("0", "1") of the independent verifier's
    record results/pcrl_controlled_training_size_v1/roles/verifier/DEPLOY_W100.json (loaded by the caller; the coordinator's
    deploy/DEPLOYMENT_CHECKS_W100.json is a second, non-gating record);
    h3_income_verified = {v: bool} (H3 income hash equals the admitted wire).
    TRIGGER_FIRES iff T0 (complete), T1 (8 occupation-side points), T2 (8 occupation-side reserve conditions, k = 2, OHN1 floor)
    and T3 (bitwise deployment with every refusal) hold on BOTH versions. Income, smaller subsets and controls play no role."""
    rules = load_rules()
    parts = {k: {'pass': False, 'detail': 'not evaluated (T0 failed)'} for k in ('T0', 'T1', 'T2', 'T3')}
    occ = None
    t0 = []
    for v in VERSIONS:
        if not _has(evals_by_version, v):
            t0.append(f'v{v}: evaluation missing')
            continue
        e = _vkey(evals_by_version, v)
        u = e['utility'] if isinstance(e, dict) and 'utility' in e else e
        try:
            conds = OR.conditions(u, v)
            bad = [c for c, r in conds.items() if not _fin(r['value']) or not _fin(r['se'])]
            if bad:
                t0.append(f'v{v}: non-finite value or SE in {bad}')
        except (KeyError, ValueError, TypeError) as ex:
            t0.append(f'v{v}: malformed evaluation ({ex})')
        if not (_has(h3_income_verified, v) and _vkey(h3_income_verified, v) is True):
            t0.append(f'v{v}: H3 income not verified against the admitted wire')
    parts['T0'] = {'pass': not t0, 'detail': t0}
    if not t0:
        occ = occupation_reserve(evals_by_version)
        pv = occ['per_version']
        parts['T1'] = {'pass': all(pv[v]['point_pass'] for v in VERSIONS),
                       'detail': {v: pv[v]['point_failed'] for v in VERSIONS}}
        parts['T2'] = {'pass': all(pv[v]['met'] for v in VERSIONS),
                       'detail': {v: {'failed': pv[v]['failed'], 'min_ratio': pv[v]['min_ratio'], 'binding': pv[v]['binding']}
                                  for v in VERSIONS}}
    t3 = []
    for v in VERSIONS:
        d = _vkey(deployment, v) if _has(deployment, v) else None
        if not isinstance(d, dict) or d.get('bitwise') is not True or d.get('refusals_all_pass') is not True:
            t3.append(f'v{v}: deployed implementation not verified (bitwise and every refusal)')
    parts['T3'] = {'pass': not t3, 'detail': t3}
    fires = all(p['pass'] for p in parts.values())
    return {'decision': 'TRIGGER_FIRES' if fires else 'TRIGGER_DOES_NOT_FIRE', 'fires': fires, 'parts': parts,
            'failing_parts': [k for k, p in parts.items() if not p['pass']], 'occupation_reserve': occ,
            'income_reported_separately': None if occ is None else {v: occ['per_version'][v]['income'] for v in VERSIONS},
            'rules_sha256': rules['_sha256'], 'ohn_rule_sha256': OHN_RULE_SHA256,
            'note': 'occupation-component trigger only; smaller subsets never substituted; not a declaration that the '
                    'H3-anchored service qualifies'}


# ---------------------------------------------------------------- 8. strength selection (fitting information only)
def allowances():
    """{v: {statistic: lambda x min over mapped conditions of the OHN1 H_LJ floor SE of version v}} (as registered)."""
    r = load_rules()
    return {v: dict(r['selection']['allowance'][str(v)]) for v in VERSIONS}


def _strength_of(key):
    x = _num(key)
    for s in STRENGTHS:
        if _fin(x) and abs(x - s) <= 1e-12:
            return s
    raise ValueError(f'strength {key!r} is not in the registered set {STRENGTHS}')


def select_strength(cost_table, version, z=Z):
    """cost_table = {s: {stat: {'estimate': pooled nested cross-fitted increment vs s = 0, 'se': household-clustered SE}}} for
    every nonzero registered strength (an s = 0 entry, if given, is ignored after a zero check). Pass at s iff for all 5
    statistics estimate + z x SE <= allowance (inclusive; non-finite fails). Selected = largest s with every nonzero strength
    <= s passing (fixed sequence); identity if the smallest fails. A pass after a failure is never selected (PASS_AFTER_FAIL)."""
    rules = load_rules()
    v = int(version)
    if v not in VERSIONS:
        raise ValueError(f'version {version!r}')
    allow = allowances()[v]
    table = {}
    for k, row in (cost_table or {}).items():
        s = _strength_of(k)
        if s in table:
            raise ValueError(f'duplicate strength {s}')
        table[s] = row
    nonzero = [s for s in STRENGTHS if s > 0]
    missing = [s for s in nonzero if s not in table]
    if missing:
        raise ValueError(f'cost table misses registered strengths {missing}')
    if 0.0 in table:
        for j in STATS:
            est = _num(((table[0.0] or {}).get(j) or {}).get('estimate'))
            if _fin(est) and est != 0.0:
                raise ValueError(f'nonzero increment at identity for {j}: {est}')
    audit, passes = [], {}
    for s in nonzero:
        row = table[s] or {}
        st, ok = {}, True
        for j in STATS:
            if j not in row:
                raise ValueError(f'strength {s}: statistic {j} missing')
            est, se = _num(row[j].get('estimate')), _num(row[j].get('se'))
            if _fin(est, se) and se >= 0:
                up = est + z * se
                pj = bool(up <= allow[j])
            else:
                up, pj = math.nan, False
            st[j] = {'estimate': est, 'se': se, 'upper': up, 'allowance': allow[j],
                     'upper_over_allowance': (up / allow[j]) if _fin(up) else None, 'pass': pj}
            ok = ok and pj
        passes[s] = ok
        fails = [j for j in STATS if not st[j]['pass']]
        binding = max(STATS, key=lambda j: (st[j]['upper_over_allowance'] if st[j]['upper_over_allowance'] is not None
                                            else math.inf))
        audit.append({'strength': s, 'pass': ok, 'failing_statistics': fails, 'binding_statistic': binding, 'stats': st})
    selected, first_fail = 0.0, None
    for s in nonzero:
        if passes[s]:
            selected = s
        else:
            first_fail = s
            break
    after = [s for s in nonzero if first_fail is not None and s > first_fail and passes[s]]
    flags = ['PASS_AFTER_FAIL'] if after else []
    if selected == 0.0:
        flags.append('IDENTITY_SELECTED')
    ff = next((a for a in audit if a['strength'] == first_fail), None)
    return {'version': v, 'selected': selected, 'identity': selected == 0.0, 'first_failing_strength': first_fail,
            'binding_at_first_failure': None if ff is None else ff['failing_statistics'], 'pass_after_fail_strengths': after,
            'flags': flags, 'audit': audit, 'allowance': allow, 'z': z, 'strengths': list(STRENGTHS),
            'rules_sha256': rules['_sha256'],
            'note': 'fitting information only (nested cross-fitted MODEL_FIT costs); MODEL_SELECT not used; cross-fitted costs '
                    'target reduced-size submodels and do not guarantee the deployed 100% model costs; identity always admissible'}


# ---------------------------------------------------------------- 9. readers
def _cells(d):
    if not isinstance(d, dict) or set(d) != set(CELLS):
        return None
    vals = {c: _num(d[c]) for c in CELLS}
    return vals if all(_fin(x) for x in vals.values()) else None


def k_differs(cells):
    """{(v, r): b(k0) != b(k1)} (exact float comparison) for the 4 (version, repetition) pairs; None if a cell is missing."""
    if not isinstance(cells, dict) or not all(c in cells for c in CELLS):
        return None
    return {(v, r): _num(cells[f'S{v}R{r}k0']) != _num(cells[f'S{v}R{r}k1']) for v in VERSIONS for r in (1, 2)}


def seed_qualifier(n_eff):
    return ('READER_SEEDS_EFFECTIVE_ALL_PAIRS' if n_eff == 8 else 'READER_SEED_REPLICATION_ABSENT' if n_eff == 4
            else 'READER_SEEDS_PARTLY_EFFECTIVE')


def stability_label(cells, cell_se, threshold=0.0, z=Z):
    """The source 8-cell screen for a paired benefit b = comparator AUC - candidate AUC (positive favours the candidate), with
    the registered correction for seed-deterministic winners: d_v = mean of version v's 4 cells; v_rep = SS / (n_eff - 2), SS =
    sum over DISTINCT cells of (b - d_v)^2 (a (v, R) pair with b(k0) == b(k1) enters once), n_eff = 4 + pairs that differ;
    v_samp = mean of the 8 squared cell SEs; sd = sqrt(v_rep + v_samp). Identical to the source screen when n_eff = 8.
    STABLE_POSITIVE iff all 8 cells > threshold and d_v - z sd > threshold for both versions; POSITIVE_NOT_STABLE;
    SIGN_CHANGES (mean > threshold, some cell <= threshold); NOT_POSITIVE; INCOMPLETE. The reader-seed qualifier is attached
    (READER_SEEDS_EFFECTIVE_ALL_PAIRS / READER_SEEDS_PARTLY_EFFECTIVE / READER_SEED_REPLICATION_ABSENT) and never changes the label."""
    vals, ses = _cells(cells), _cells(cell_se)
    if vals is None or ses is None or any(s < 0 for s in ses.values()):
        return {'label': 'INCOMPLETE', 'stable': False, 'error': 'the 8 registered cells, their SEs and finite values are required'}
    by = {v: [vals[c] for c in CELLS if c.startswith(f'S{v}')] for v in VERSIONS}
    d = {v: sum(x) / len(x) for v, x in by.items()}
    kd = k_differs(vals)
    distinct = {v: [x for r in (1, 2) for x in ((vals[f'S{v}R{r}k0'], vals[f'S{v}R{r}k1']) if kd[(v, r)]
                                                else (vals[f'S{v}R{r}k0'],))] for v in VERSIONS}
    n_eff = sum(len(x) for x in distinct.values())
    v_rep = sum((x - d[v]) ** 2 for v in VERSIONS for x in distinct[v]) / (n_eff - len(VERSIONS))
    v_samp = sum(s ** 2 for s in ses.values()) / len(CELLS)
    sd = math.sqrt(v_rep + v_samp)
    above = sum(x > threshold for x in vals.values())
    mean = sum(vals.values()) / len(CELLS)
    allabove = above == len(CELLS)
    stable = allabove and all(d[v] - z * sd > threshold for v in VERSIONS)
    if stable:
        label = 'STABLE_POSITIVE'
    elif allabove:
        label = 'POSITIVE_NOT_STABLE'
    elif mean > threshold:
        label = 'SIGN_CHANGES'
    else:
        label = 'NOT_POSITIVE'
    return {'label': label, 'stable': stable, 'threshold': threshold, 'mean': mean, 'min': min(vals.values()),
            'max': max(vals.values()), 'per_version_mean': d, 'lower_per_version': {v: d[v] - z * sd for v in VERSIONS},
            'cells_at_or_below': len(CELLS) - above, 'sd_reader_split': math.sqrt(v_rep), 'rms_cell_se': math.sqrt(v_samp),
            'sd_total': sd, 'n_eff': n_eff, 'spread_df': n_eff - len(VERSIONS),
            'k_differs': {f'S{v}R{r}': kd[(v, r)] for v in VERSIONS for r in (1, 2)},
            'seed_distinct_pairs': n_eff - 4, 'reader_seed_qualifier': seed_qualifier(n_eff), 'cells': vals}


def target_label(cells, cell_se):
    """Original 0.020 material criterion against one target reference: DEV_GRID_TARGET_STABLE / _POINT_ONLY / _NOT_MET."""
    t = stability_label(cells, cell_se, threshold=TARGET)
    if t['label'] == 'INCOMPLETE':
        return {'label': 'DEV_GRID_TARGET_NOT_MET', 'detail': t}
    if t['stable']:
        return {'label': 'DEV_GRID_TARGET_STABLE', 'detail': t}
    return {'label': 'DEV_GRID_TARGET_POINT_ONLY' if t['mean'] > TARGET else 'DEV_GRID_TARGET_NOT_MET', 'detail': t}


def recipient_gate(cells, limit=RECIPIENT_LIMIT):
    """Old per-cell recipient gate: candidate minus comparator SCORE AUC <= 0.010 in every registered cell (one view)."""
    vals = _cells(cells)
    if vals is None:
        return {'pass': False, 'label': 'INCOMPLETE'}
    over = [c for c in CELLS if vals[c] > limit]
    return {'pass': not over, 'label': 'RECIPIENT_POINT_PASS' if not over else 'RECIPIENT_POINT_FAILED', 'cells_above': over,
            'max': max(vals.values())}


def seed_distinct_pairs(cells):
    """Number of (version, repetition) pairs whose k0 and k1 cell values differ (0-4); 0 = READER_SEED_REPLICATION_ABSENT."""
    kd = k_differs(cells)
    return None if kd is None else sum(kd.values())


def forbidden_seeds():
    """OHN1 / CTS1 grid reader seeds (MODEL_SEED_BASE + 300 + 10 v + k, k = 0..9) that the new grid must not reuse."""
    return {MODEL_SEED_BASE + 300 + 10 * v + k for v in VERSIONS for k in range(10)}


def reader_seed_check(seeds, recipe_checks):
    """seeds = {'S<v>k<k>': int} for v in {0, 1}, k in {0, 1}; recipe_checks = list of {'recipe', 'view', 'wire', 'v', 'R',
    'equal_k0_k1': bool (SCORE scores bitwise equal under k0 and k1)}.
    READER_SEEDS_EFFECTIVE iff the 4 seeds are distinct integers outside the OHN1 / CTS1 grid seeds, every seed-dependent recipe
    of the registered list appears in the checks, and none of its checks has equal_k0_k1. Seed-invariant recipes that differ are
    flagged NONDETERMINISTIC_RECIPE (recorded, not fatal)."""
    errors, flags = [], []
    keys = [f'S{v}k{k}' for v in VERSIONS for k in (0, 1)]
    vals = [seeds.get(k) for k in keys] if isinstance(seeds, dict) else [None] * 4
    if any(not isinstance(x, int) or isinstance(x, bool) for x in vals):
        errors.append('the 4 reader seeds S<v>k<k> must be recorded integers')
    else:
        if len(set(vals)) != 4:
            errors.append('reader seeds are not distinct')
        clash = sorted(set(vals) & forbidden_seeds())
        if clash:
            errors.append(f'reader seeds reuse OHN1 / CTS1 grid seeds {clash}')
    checks = list(recipe_checks or [])
    seen = {c.get('recipe') for c in checks}
    absent = [r for r in SEED_DEPENDENT if r not in seen]
    if absent:
        errors.append(f'seed-dependent recipes not checked: {absent}')
    ineffective = [c for c in checks if c.get('recipe') in SEED_DEPENDENT and c.get('equal_k0_k1') is not False]
    if ineffective:
        errors.append(f'{len(ineffective)} seed-dependent checks show identical (or unrecorded) k0 / k1 fits')
    nondet = [c for c in checks if c.get('recipe') in SEED_INVARIANT and c.get('equal_k0_k1') is False]
    if nondet:
        flags.append('NONDETERMINISTIC_RECIPE')
    return {'label': 'READER_SEEDS_NOT_EFFECTIVE' if errors else 'READER_SEEDS_EFFECTIVE', 'effective': not errors,
            'errors': errors, 'flags': flags, 'ineffective_checks': ineffective, 'nondeterministic_checks': nondet,
            'note': 'a NOT_EFFECTIVE grid is invalid: no stability label is computed until it is repaired'}


# ---------------------------------------------------------------- 10. equal reserve for every method
def equal_reserve_table(points, k_ratio=K_RESERVE):
    """points: [{'id', 'kind' (candidate / control / reference), 'min_ratio': {v: r}, 'pair_auc': {v: a}}] with the same reserve
    (28 or 16 conditions, k = 2, H_LJ floor) for every point. Qualified iff min ratio >= 2 on both versions. Returns the
    qualification of every point and the Pareto frontier (lower mean pair AUC, higher minimum ratio); failing references kept."""
    recs = []
    for p in points:
        mr = min(_num(_vkey(p['min_ratio'], v)) for v in VERSIONS)
        pa = sum(_num(_vkey(p['pair_auc'], v)) for v in VERSIONS) / len(VERSIONS)
        recs.append({'id': p['id'], 'kind': p.get('kind'), 'min_ratio': mr, 'pair_auc_mean': pa,
                     'qualified': bool(_fin(mr) and mr >= k_ratio)})
    q = [r for r in recs if r['qualified']]
    front = [r['id'] for r in q if not any(o['pair_auc_mean'] <= r['pair_auc_mean'] and o['min_ratio'] >= r['min_ratio']
                                           and (o['pair_auc_mean'] < r['pair_auc_mean'] or o['min_ratio'] > r['min_ratio'])
                                           for o in q)]
    return {'points': recs, 'frontier': front,
            'references_failing_reserve': [r['id'] for r in recs if r['kind'] == 'reference' and not r['qualified']],
            'note': 'a positive equal-reserve result is not superiority over every original reference; both tables are reported'}


# ---------------------------------------------------------------- 11. claim labels
def _utility_claim(arm, with_reserve=True):
    """C2 (with_reserve: point limits AND occupation-side reserve; used by the closure rule) or C2p (original point limits only)."""
    sel = arm.get('selected') or {}
    ut = arm.get('utility') or {}
    pre = 'UTILITY_' if with_reserve else 'UTILITY_POINT_'
    if not all(_has(sel, v) for v in VERSIONS):
        return 'INCOMPLETE'
    nonzero = [v for v in VERSIONS if _num(_vkey(sel, v)) > 0]
    if not nonzero:
        return 'NO_NONZERO_SELECTION'
    for v in nonzero:
        u = _vkey(ut, v) if _has(ut, v) else {}
        ok = u.get('point_pass') is True and (u.get('occ_reserve_met') is True or not with_reserve)
        if not ok:
            return pre + 'NOT_PRESERVED'
    return pre + ('PRESERVED_BOTH_VERSIONS' if len(nonzero) == 2 else 'PRESERVED_ONE_VERSION')


def _selected_status(arm):
    """Point status and occupation-side reserve status of every selected release (identity included), both versions."""
    sel, ut = arm.get('selected') or {}, arm.get('utility') or {}
    return {v: {'selected': _vkey(sel, v) if _has(sel, v) else None,
                'point_pass_12_plus_2': (_vkey(ut, v) if _has(ut, v) else {}).get('point_pass'),
                'occ_reserve_met': (_vkey(ut, v) if _has(ut, v) else {}).get('occ_reserve_met')} for v in VERSIONS}


def _protection(stab, name, utility_both):
    if stab != 'STABLE_POSITIVE':
        return {'label': 'NOT_MET', 'stability': stab}
    return {'label': name if utility_both else 'MET_WITHOUT_UTILITY', 'stability': stab}


def claim_labels(ev):
    """ev = {'trigger_fired': bool, 'size_explanation': label,
             'arms': {'LOC': arm, 'PAIR': arm}} with arm = {
               'selected': {v: s*}, 'utility': {v: {'point_pass': complete release 12 + 2 at the point, 'occ_reserve_met': bool}},
               'vs_CONT': stability label vs CONT at the same strength, 'vs_W': stability label vs W100,
               'vs_LOC': (PAIR only) stability label vs LOC at LOC's own selected strength,
               'competitive': {'point_all': bool, 'recipient_gate_all': bool, 'stable_vs_all_required': bool},
               'target': DEV_GRID_TARGET_* label}.
    Returns C1-C8, per-arm qualification and the closure outcome. Claims are separate; none implies another."""
    out = {'C1': {'label': 'MARGIN_RESTORED_BY_TRAINING_SIZE_ON_THIS_POOL'
                  if ev.get('size_explanation') == 'SUPPORTED_ON_THIS_POOL' else 'NOT_ESTABLISHED',
                  'size_explanation': ev.get('size_explanation'), 'scope': SCOPE[2]},
           'C7': 'NOT_CLAIMABLE', 'C8': 'NOT_CLAIMED', 'scope': list(SCOPE), 'arms': {}}
    if not ev.get('trigger_fired'):
        out['outcome'] = 'NOT_RUN_TRIGGER_NOT_FIRED'
        return out
    qual = {}
    for name in (PRIMARY_ARM, SECONDARY_ARM):
        arm = (ev.get('arms') or {}).get(name) or {}
        c2 = _utility_claim(arm, with_reserve=True)
        both = c2 == 'UTILITY_PRESERVED_BOTH_VERSIONS'
        rec = {'C2': c2, 'C2p': _utility_claim(arm, with_reserve=False), 'closure_uses': 'C2',
               'selected_release_status': _selected_status(arm),
               'C3': _protection(arm.get('vs_CONT'), 'RECOVERY_BELOW_CONTINUATION', both),
               'C3w': _protection(arm.get('vs_W'), 'RECOVERY_BELOW_WITNESS', both)}
        if name == SECONDARY_ARM:
            rec['C4'] = _protection(arm.get('vs_LOC'), 'RECOVERY_BELOW_LOCAL_CORRECTION', both)
        comp = arm.get('competitive') or {}
        met5 = all(comp.get(k) is True for k in ('point_all', 'recipient_gate_all', 'stable_vs_all_required'))
        rec['C5'] = {'label': ('COMPETITIVE_WITH_ALL_REQUIRED' if both else 'MET_WITHOUT_UTILITY') if met5 else 'NOT_MET'}
        rec['C6'] = {'label': arm.get('target') or 'DEV_GRID_TARGET_NOT_MET', 'claimable': both,
                     'note': 'distinct from positive superiority; never implied by C5'}
        keys = ('C3', 'C3w') + (('C4',) if name == SECONDARY_ARM else ())
        rec['qualifies'] = both and all(rec[k]['label'] not in ('NOT_MET', 'MET_WITHOUT_UTILITY') for k in keys)
        rec['role'] = 'primary' if name == PRIMARY_ARM else 'secondary'
        out['arms'][name] = rec
        qual[name] = rec['qualifies']
    if qual[PRIMARY_ARM]:
        out['outcome'] = 'PRIMARY_ARM_QUALIFIES'
    elif qual[SECONDARY_ARM]:
        out['outcome'] = 'SECONDARY_ARM_ONLY'
    else:
        out['outcome'] = 'CORRECTION_RECIPE_CLOSED'
    return out
