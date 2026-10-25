"""Frozen decision rules of the PCRL direct-recovery frontier campaign (`pcrl_direct_recovery_frontier_v1`; statistical reviewer;
written before any outcome of this campaign existed; no label read).

Human-readable rules: results/pcrl_direct_recovery_frontier_v1/roles/statistical_reviewer/RULES.md
Machine-readable constants: .../RULES.json (schema DRF1-rules-v1); `load_rules()` refuses a RULES.json that disagrees with this
module, a changed NRM1 rule file or a changed nrm/rules.py.
Invented-data tests: `python3 -m drf.test_rules` (prints JSON; exit 0 iff every test passes).

Pure functions on plain dicts / floats. Nothing here reads data or labels (the only files read are this campaign's RULES.json, the
NRM1 RULES.json and, through nrm.rules, the OHN1 reserve rule file). The inherited machinery is `nrm.rules`, imported UNCHANGED as
NR and never monkeypatched: point_eligibility (G1, 28 original conditions), reserve_status (two-SE reserve heuristic), the
aggregate paired screen, benefit_screen, target_screen, recipient_screen, comparison, frontier_comparison (the spend-bracketing
rule), robustness and claims (G4 / G5 / G6). Adapted from sar.rules (SAR1; copied, not imported): grid validity with the
extra-trees and few-valued explanations, refit effectiveness, the reader-budget credit rule, the control-path alias collapse,
the non-table winner's version means, D5's collapse rules and the registered / adaptive labelling.

DRF1 structure (RULES.md 0.1): the candidate is DR, the constrained exact-AUC finite mapping (one fitted release per version; no
strength path). D2 (component gain = "the mapping beats its controls") gates on two family controls: the CTRL frontier (origin
TWIN = the family origin, followed by the two CTRL points saved on MODEL_FIT nested held-out total-NLL spend) and OBJ (the previous
dependence objective, one point per version, gated by the inherited frontier conjuncts with a spend-resolution rule for a single
point). TWIN alone is never sufficient (reported beside); the family change against START = SUCC is reported separately (FC);
D3 / D4 against the nine frozen references; the failure-stage classification (stages 1-5, with the optimistic MODEL_FIT diagnostic
reported, never gating); D2R on R3 / R4 with sign changes and robustness recipient failures carried into the headline.

API (RULES.md section in brackets):
  load_rules()                                     constants + '_sha256'                                                   [0]
  forbidden_seeds(), registered_seed(v, k), registered_boot_seed(v, rep)                                                   [9]
  grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=None)       validity of the registered DRF1 grid   [9]
  refit_effectiveness(splits, seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, rob_reps=None)               [8]
  confidence_conjunction(evals)                    PASS / UNRESOLVED / VIOLATION per condition (reported)                  [2]
  brackets(costs_vs, paths, alias_of=None, origin='TWIN')   the CTRL bracket Q* (NR's rule)                                [4]
  selection_data_error(sd)                         allowed bracket data (MODEL_FIT_NESTED only)                            [4]
  grid_fineness(spends)                            fineness of the two saved CTRL points (reported only)                 [4.4]
  obj_spend_state(costs)                           OBJ_MI vs DR spend state on MODEL_SELECT (single-point rule)            [4.2]
  comparator_ids(rec)                              (twin id, OBJ_MI id) used as grid / cost keys (defaults TWIN, OBJ_MI)    [3]
  useful_optimistic(opt)                           the MODEL_FIT optimistic diagnostic (reported, never gating)            [5]
  d1(rec), d2(rec), family_change(rec), twin_beside(rec)                                                                 [2-4]
  failure_stage(dec, opt_record)                   stages 1-5 (reported)                                                   [5]
  decisions(rec)                                   D1, D2, FC, TWIN, D3-D6, D2R, stage, the three outcomes, the class      [2-7]
  adaptive_tag(rec), campaign_summary(decision_records)                                                                    [7]
  d2r(rec)                                                                                                                 [8]

Record of one DR reading (`d1`, `d2`, `decisions`):
  {'family': 'DR', 'card', 'revision': 0 | 1 | 2, 'card_committed_before_any_model_select_outcome': bool,
   'revision_note': {'observed_failure': str, 'changed_operation': str, 'trigger': 'MODEL_SELECT' | 'MODEL_FIT' | 'SYNTHETIC'}
                    (required for revision >= 1),
   'grid_valid': grid_validity(...)['effective'] (None = not recorded),
   'cand': {'status': 'FITTED', 'config_id': 'DR', 'evals': {v: utility}, 'alias_of_twin': bool (exact wire equality with TWIN in
            both versions)},
   'reader_stage': 'RUN' (default) | 'STOPPED_USEFULNESS_FAILURE' (allowed only when DR fails G1),
   'start': {'id': 'SUCC', 'evals'}, 'twin': {'id': 'TWIN' (or e.g. 'TWB_C64'), 'evals'},
   'obj_mi': {'config_id': 'OBJ_MI' (or e.g. 'DR_MI'), 'evals'} (comparator_ids(rec): the keys used below),
   'paths': {'CTRL': [the saved CTRL ids] | None}, 'selection_data': {'CTRL': 'MODEL_FIT_NESTED'},
   'controls_not_run': {'CTRL': bool, 'OBJ': bool} (budget only; never inferred), 'alias_of': {id: earlier id | 'TWIN'},
   'grid': {comparator id: NR grid entry of DR vs it} for SUCC, TWIN, OBJ_MI, every CTRL point and the nine references,
   'costs_vs': {id: paired costs of DR vs id ('total|nll' + the 8 secondary metrics)} for TWIN, OBJ_MI, every CTRL point, SUCC,
   'nolookup': {id: {'cells': {cell: non-table winner pair benefit}}}, 'enhanced' (reported only), 'reader_budget',
   'fineness': grid_fineness(...) record, 'optimistic': useful_optimistic(...) input, 'joint' (D5), 'novelty' (D6),
   'mc_flags', 'd2r': d2r(...) record (optional)}

Sign conventions are NR's: pair benefit b = comparator AUC - candidate AUC (positive favours DR); recipient difference
r = candidate AUC - comparator AUC; paired costs loss(DR) - loss(comparator) (negative favours DR; accuracy enters as a loss).
Versions are the ints 0 and 1 (string keys '0' / '1' accepted).
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

from nrm import rules as NR                      # inherited claims machinery, imported unchanged (never monkeypatched)

RULES_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_direct_recovery_frontier_v1' / 'roles'
              / 'statistical_reviewer' / 'RULES.json')
SCHEMA = 'DRF1-rules-v1'
NRM1_RULES_SHA256 = '84f187768817e79ef9f2bb36151bea81f00b8f62d9179c8cd523069550674652'     # NRM1 RULES.json (registered)
NRM_RULES_PY_SHA256 = 'd5df868678d483a3452e251d8f24575932d23f0e5d22e3ef1ca3627a35030df1'   # nrm/rules.py (registered)
VERSIONS = NR.VERSIONS
Z = 2.0
TIE_TOL = 1e-12
RECIPIENT_LIMIT = NR.RECIPIENT_LIMIT               # 0.010 AUC, per version at the aggregate point (original)
# ---- the registered DRF1 reader grid
SEED_OFFSET = 1100                                 # MODEL_SEED_BASE + 1100 + 10 v + k (never used by an earlier grid)
FORBIDDEN_OFFSETS = (300, 400, 500, 600, 700, 800, 900, 1000)   # every realized earlier grid (OHN1 ... IPR1, SAR1)
BOOT_SEED_BASE = 2026102500                        # + 100 v + repetition index
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
ALL_REPS = NR.REG_REPS + NR.ROB_REPS
ALL_K = NR.REG_K + NR.ROB_K
SPLIT_TAG = 'DRF1_devgrid_R{r}'
# ---- seed-equality explanations (prospective)
ER_MIN_LEAF = {'ET5': 5, 'ET25': 25}               # min_samples_leaf of the extra-trees recipes
ER_POOL_PEOPLE = 8001                              # MODEL_SELECT people: support bound = 8001 // min leaf (ET5 1600, ET25 320)
FEW_VALUED_MAX = ER_POOL_PEOPLE // ER_MIN_LEAF['ET25']    # 320: at least 25 MODEL_SELECT people per distinct value on average
CAPACITY = {'income': 16, 'occupation': 64}        # the finite family's atoms (the twin's capacity)
PAIR_FINITE_MAX = CAPACITY['income'] * CAPACITY['occupation']   # 1024: a finite release's pair view, by construction
# ---- comparators
START_ID = 'SUCC'                                  # the unchanged strong start (not in the finite family)
TWIN_ID = 'TWIN'                                   # the family origin TWB_C64 (frozen; weak task-only twin)
OBJ_ID = 'OBJ_MI'                                  # the previous dependence objective in the same family
FAMILY = 'DR'
MAX_REVISIONS = 2
REVISION_TRIGGERS = ('MODEL_SELECT', 'MODEL_FIT', 'SYNTHETIC')
# ---- decisions
D1_LABELS = ('TASK_FEASIBLE', 'CAND_INFEASIBLE', 'INCOMPLETE')
CONDITION_STATES = ('PASS', 'UNRESOLVED', 'VIOLATION')
CONFIDENCE_LABELS = ('CONFIDENCE_ALL_PASS', 'CONFIDENCE_UNRESOLVED', 'CONFIDENCE_VIOLATION', 'INCOMPLETE')
D2_COMPARISONS = ('CTRL_FRONTIER', 'OBJ')
D2_MET = 'COMPONENT_GAIN_BEYOND_FAMILY_CONTROLS'
D2_ASSESSED = (D2_MET, 'READER_TARGETED', 'NOT_SEPARATED')     # plus PARTIAL_<passed>_ONLY
D2_GATES = ('NOT_ASSESSED_EXACT_ALIAS', 'NOT_ASSESSED_TASK_INFEASIBLE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN',
            'NOT_COMPUTED_GRID_INVALID', 'NOT_ASSESSED_CONTROLS_NOT_RUN', 'INCOMPLETE')
OBJ_SPEND_STATES = ('OBJ_MI_SPENDS_AT_LEAST_AS_MUCH', 'OBJ_MI_SPENDS_LESS_UNRESOLVED', 'OBJ_MI_SPENDS_LESS_RESOLVED')
SELECTION_DATA = ('MODEL_FIT_NESTED',)             # the CTRL bracket data: MODEL_FIT nested held-out paired costs; never MODEL_SELECT
FC_LABELS = ('FAMILY_CHANGE_BELOW_START_RESOLVED', 'FAMILY_CHANGE_NOT_RESOLVED')
TWIN_LABELS = ('BELOW_TWIN_RESOLVED', 'NOT_RESOLVED_VS_TWIN')
D2_SCOPE = ('the component gain is a resolved linked-pair benefit (with both recipient gates) beyond the SEX-blind task-only '
            'coarsening path of the same family at matched MODEL_SELECT total-NLL spend and beyond the previous dependence '
            'objective (OBJ_MI) in the same family with identical constraints, moves, data exposure and optimizer budget, at no '
            'resolved lower OBJ_MI spend, for the declared development readers on reused MODEL_SELECT people; it is not superiority '
            'over SUCC or the strong references (D3), not the family change (FC) and not the material target (D4)')
NO_BRACKET_STATEMENT = ('no {key} point spent as much MODEL_SELECT total NLL as DR: lower recovery cannot be separated from task '
                        'spending; this is a limit of the control path, never evidence that DR equals the control or that the '
                        'mechanism has no effect, and the mechanism\'s contribution is not called isolated')
FINENESS_FRACTION = 0.25                           # saved CTRL points: lower >= 0.75 s, upper <= s + 2 SE + 0.25 s
# ---- the optimistic MODEL_FIT diagnostic (reported, never gating)
OPT_KINDS = ('OPT_S', 'OPT_L')
OPT_MODES = ('in_sample', 'nested')
USEFUL_MARGIN = NR.TARGET                          # 0.020 linked-pair AUC below the matched-spend CTRL path point (comparator-based)
OPT_LABELS = ('USEFUL_OPTIMISTIC_POINT', 'NO_USEFUL_OPTIMISTIC_POINT', 'UNDETERMINED')
STAGES = ('STAGE_1_NO_USEFUL_OPTIMISTIC_POINT', 'STAGE_2_OPTIMISTIC_SUCCEEDS_LEGAL_MAPPING_FAILS',
          'STAGE_3_KEEPS_UTILITY_TIES_CONTROLS', 'STAGE_4_BEATS_CONTROLS_MISSES_STRONG_REFERENCES',
          'STAGE_5_COMPETITIVE_CONDITIONS_PASS', 'UNDETERMINED_OPTIMISTIC_DIAGNOSTIC', 'INCOMPLETE')
STAGE_STATEMENTS = {
    STAGES[0]: 'no useful optimistic point: a failure within the chosen finite model, estimated law and optimizer; not a general '
               'impossibility',
    STAGES[1]: 'an optimistic point is useful but the legal mapping fails (usefulness lost on MODEL_SELECT or no output change): a '
               'possible estimation, representation, optimization or generalization limitation',
    STAGES[2]: 'the legal mapping keeps utility but ties its controls: no demonstrated mechanism advantage',
    STAGES[3]: 'the mapping beats its controls but misses strong references: at most a scoped component result, and only if '
               'robustness supports it',
    STAGES[4]: 'the original competitive conditions pass: exploratory development evidence requiring fresh confirmation',
    STAGES[5]: 'the legal mapping failed or tied its controls, but the optimistic diagnostic is incomplete or unbracketed: stage 1 '
               'cannot be told from stage 2 / 3',
    STAGES[6]: 'inputs missing'}
OUTCOMES = ('METHOD_CANDIDATE', 'COMPONENT_GAIN_NOT_COMPETITIVE', 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED',
            'FEASIBLE_NO_RELIABLE_GAIN', 'NO_OUTPUT_CHANGE', 'TASK_INFEASIBLE', 'PROTECTION_NOT_ASSESSED', 'INCOMPLETE')
D2R_LABELS = ('ROBUSTNESS_CONFIRMED', 'ROBUSTNESS_NOT_CONFIRMED', 'NOT_RUN', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN',
              'NOT_COMPUTED_GRID_INVALID', 'INCOMPLETE')
ADAPTIVE_KINDS = ('ADAPTIVE',)
PER_LOOK_FALSE_PASS = ('one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under six moderate NRM1 '
                       'known laws and about 8% when one version or split dominates (NRM1 RULES.md 5.5); a D2 look is an '
                       'intersection-union conjunction of two such comparisons (the CTRL frontier at its bracket and OBJ_MI), so '
                       'its false-pass rate is at most that of its weaker member; with K D2 looks the chance that at least one '
                       'passes by chance is at most K times the per-look rate; no multiplicity correction is applied or claimed')
MDE_NOTE = ('per comparison, 2 se_agg is the resolution threshold (50% power) and about 2.84 se_agg the benefit resolved with '
            'roughly 80% power (normal approximation, version-mean condition ignored); with se_agg 0.008 (CTRL) and 0.007 (OBJ), '
            'correlation 0.5, D2 resolves a true benefit of 0.010 / 0.015 / 0.020 / 0.025 / 0.030 against both with probability '
            'about 0.12 / 0.33 / 0.61 / 0.84 / 0.95, and only 0.25 / 0.40 / 0.55 at 0.020 / 0.025 / 0.030 when OBJ_MI reproduces '
            'half of the gain; "not separated" is the default reading, never "equal"')

_cache = {}
_SEEDKEY = re.compile(r'S([01])k([0-9])')
_BOOTKEY = re.compile(r'S([01])(R[1-4])')


# ---------------------------------------------------------------- rule file
def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rules(path=RULES_PATH):
    """RULES.json with its SHA-256; refuses a schema or constant that disagrees with this module, a changed NRM1 rule file or a
    changed nrm/rules.py."""
    p = Path(path)
    if str(p) not in _cache:
        raw = p.read_bytes()
        r = json.loads(raw)
        if r.get('schema') != SCHEMA:
            raise ValueError(f'unexpected rules schema {r.get("schema")!r}')
        if NR.load_rules().get('_sha256') != NRM1_RULES_SHA256:
            raise ValueError('the NRM1 rule file differs from the registered one')
        if _file_sha(NR.__file__) != NRM_RULES_PY_SHA256:
            raise ValueError('nrm/rules.py differs from the registered one')
        g, er, dec, rs, cm = r['reader_grid'], r['er_explanation'], r['decisions'], r['registration_status'], r['comparators']
        checks = {
            'z': (r['z'], Z), 'z equals NRM1': (Z, NR.Z), 'tie tolerance': (r['tie_tolerance'], TIE_TOL),
            'recipient limit': (r['recipient_limit_auc'], RECIPIENT_LIMIT),
            'model seed base': (g['model_seed_base'], NR.MODEL_SEED_BASE), 'seed offset': (g['seed_offset'], SEED_OFFSET),
            'forbidden offsets': (tuple(g['forbidden_offsets']), FORBIDDEN_OFFSETS),
            'bootstrap seed base': (g['bootstrap_seed_base'], BOOT_SEED_BASE), 'repetition index': (g['repetition_index'], REP_INDEX),
            'split tag': (g['split_tag'], SPLIT_TAG), 'er recipes': (er['min_leaf'], ER_MIN_LEAF),
            'er pool people': (er['pool_people'], ER_POOL_PEOPLE), 'few-valued max': (er['few_valued_max'], FEW_VALUED_MAX),
            'capacity': (er['capacity'], CAPACITY), 'pair finite max': (er['pair_finite_max'], PAIR_FINITE_MAX),
            'start': (cm['START'], START_ID), 'twin': (cm['TWIN'], TWIN_ID), 'obj': (cm['OBJ'], OBJ_ID),
            'family': (rs['family'], FAMILY), 'max revisions': (rs['max_revisions'], MAX_REVISIONS),
            'revision triggers': (tuple(rs['revision_triggers']), REVISION_TRIGGERS),
            'd1 labels': (tuple(dec['D1']['labels']), D1_LABELS),
            'confidence labels': (tuple(dec['D1']['confidence_labels']), CONFIDENCE_LABELS),
            'condition states': (tuple(dec['D1']['condition_states']), CONDITION_STATES),
            'd2 comparisons': (tuple(dec['D2']['comparisons']), D2_COMPARISONS), 'd2 met': (dec['D2']['met_label'], D2_MET),
            'obj spend states': (tuple(dec['D2']['obj_spend_states']), OBJ_SPEND_STATES),
            'selection data': (tuple(dec['D2']['selection_data']['allowed']), SELECTION_DATA),
            'fineness fraction': (dec['D2']['grid_fineness']['fraction'], FINENESS_FRACTION),
            'fc labels': (tuple(dec['FC']['labels']), FC_LABELS), 'twin labels': (tuple(dec['TWIN']['labels']), TWIN_LABELS),
            'opt kinds': (tuple(r['optimistic']['kinds']), OPT_KINDS), 'opt modes': (tuple(r['optimistic']['modes']), OPT_MODES),
            'useful margin': (r['optimistic']['useful_margin_auc'], USEFUL_MARGIN),
            'opt labels': (tuple(r['optimistic']['labels']), OPT_LABELS),
            'stages': (tuple(r['failure_stage']['labels']), STAGES),
            'd2r labels': (tuple(dec['D2R']['labels']), D2R_LABELS),
            'outcomes': (tuple(r['outcome_class']['labels']), OUTCOMES),
            'adaptive kinds': (tuple(rs['kinds']), ADAPTIVE_KINDS),
            'target references': (tuple(dec['D4']['target_references']), NR.TARGET_REFS),
            'benchmark': (tuple(dec['D3']['references']), NR.BENCHMARK),
        }
        bad = [k for k, (a, b) in checks.items() if a != b]
        if bad:
            raise ValueError(f'RULES.json disagrees with drf.rules on {bad}')
        r['_sha256'] = hashlib.sha256(raw).hexdigest()
        _cache[str(p)] = r
    return _cache[str(p)]


# ---------------------------------------------------------------- small helpers
def _fin(*xs):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs)


def _v(d, v):
    """Version entry of a dict keyed by 0 / 1 or '0' / '1'; None if absent."""
    if not isinstance(d, dict):
        return None
    return d[v] if v in d else d.get(str(v))


def _gate_label(gv):
    if gv is None:
        return 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN'
    return None if gv is True else 'NOT_COMPUTED_GRID_INVALID'


def _version_means(entry):
    """Per-version means of the 8 registered cells of a grid-format entry ({'cells': {cell: x}} or {cell: x}): block means over
    reader seeds, then the mean over repetitions. None unless all 8 are finite."""
    cells = entry.get('cells') if isinstance(entry, dict) and 'cells' in entry else entry
    if not isinstance(cells, dict) or not all(_fin(cells.get(c)) for c in NR.REGISTERED):
        return None
    out = {}
    for v in VERSIONS:
        blocks = {}
        for c in NR.REGISTERED:
            vv, r, _ = NR.parse_cell(c)
            if vv == v:
                blocks.setdefault(r, []).append(float(cells[c]))
        out[v] = sum(sum(b) / len(b) for b in blocks.values()) / len(blocks)
    return out


def _point_positive(entry):
    """{'pass': both version means > 0, 'version_means'} of a per-cell benefit entry, or None if incomplete."""
    vm = _version_means(entry)
    return None if vm is None else {'pass': all(vm[v] > 0 for v in VERSIONS), 'version_means': vm}


def _path(ids, alias_of):
    """Control-path ids that exist, aliases mapped to the saved id, duplicates removed (order kept)."""
    out = []
    for i in ids or []:
        j = (alias_of or {}).get(i, i) if i else None
        if j and j not in out:
            out.append(j)
    return out


def control_path(ids, alias_of=None, origin=TWIN_ID):
    """The CTRL path of D2: its origin (TWIN, the family origin: the SEX-blind coarsening starts there) followed by the distinct
    saved CTRL points (aliases once; an alias of the origin collapses into it). None if the path was not given."""
    if ids is None:
        return None
    return _path([origin] + list(ids), alias_of)


def _resolved_negative(entry):
    """True iff the pair benefit of an entry is resolved NEGATIVE (aggregate upper bound < 0 and both version means < 0): the
    comparator is resolved less recoverable than DR. None if the entry is incomplete."""
    a = NR.aggregate(entry)
    if not a.get('complete'):
        return None
    if a['alias']:
        return False
    return a['upper'] < 0 and all(a['per_version'][v]['mean'] < 0 for v in VERSIONS)


def _sign(x):
    return (x > 0) - (x < 0)


# ---------------------------------------------------------------- (9) reader-grid validity of the registered DRF1 grid
def forbidden_seeds():
    """Every realized earlier grid's reader seeds: MODEL_SEED_BASE + {300, ..., 1000} + 10 v + k, k = 0..9 (160 seeds; a local
    equivalent of nrm.rules.forbidden_seeds, which reads its own constant (300, 500); nrm is not monkeypatched)."""
    return {NR.MODEL_SEED_BASE + off + 10 * v + k for off in FORBIDDEN_OFFSETS for v in VERSIONS for k in range(10)}


def registered_seed(v, k):
    return NR.MODEL_SEED_BASE + SEED_OFFSET + 10 * int(v) + int(k)


def registered_boot_seed(v, rep):
    return BOOT_SEED_BASE + 100 * int(v) + REP_INDEX[rep]


def _seed_errors(seeds, required_k):
    """Recorded distinct integer seeds for the required keys S<v>k<k>; every key S<v>k<k> equal to the registered MODEL_SEED_BASE
    + 1100 + 10 v + k; none an earlier grid's seed."""
    keys = [f'S{v}k{k}' for v in VERSIONS for k in required_k]
    s = seeds if isinstance(seeds, dict) else {}
    sv = [s.get(k) for k in keys] + [x for k, x in s.items() if k not in keys]
    if any(not isinstance(x, int) or isinstance(x, bool) for x in sv) or len(set(sv)) != len(sv):
        return [f'reader seeds must be recorded distinct integers, at least {keys}']
    errs = []
    bad = sorted(str(k) for k in s if not _SEEDKEY.fullmatch(str(k)))
    if bad:
        errs.append(f'reader seed keys not of the form S<v>k<k>: {bad}')
    reused = sorted(set(sv) & forbidden_seeds())
    if reused:
        errs.append(f'reader seeds reuse earlier grid seeds {reused}')
    wrong = sorted(k for k, x in s.items() if _SEEDKEY.fullmatch(str(k)) and x != registered_seed(*_SEEDKEY.fullmatch(str(k)).groups()))
    if wrong:
        errs.append(f'reader seeds differ from the registered MODEL_SEED_BASE + {SEED_OFFSET} + 10 v + k: {wrong}')
    return errs


def _boot_errors(boot_seeds, reps):
    """Household-bootstrap seeds {'S<v><R>': int}: every required (version, repetition) recorded and equal to the registered
    2026102500 + 100 v + repetition index; no other key form. None -> not recorded (an error)."""
    if boot_seeds is None:
        return [f'bootstrap seeds not recorded (registered: {BOOT_SEED_BASE} + 100 v + repetition index)']
    s = boot_seeds if isinstance(boot_seeds, dict) else {}
    errs = []
    for v in VERSIONS:
        for rep in reps:
            x = s.get(f'S{v}{rep}')
            if not isinstance(x, int) or isinstance(x, bool) or x != registered_boot_seed(v, rep):
                errs.append(f'bootstrap seed S{v}{rep} is not the registered {registered_boot_seed(v, rep)}')
    for k, x in s.items():
        m = _BOOTKEY.fullmatch(str(k))
        if not m:
            errs.append(f'bootstrap seed key {k!r} not of the form S<v>R<r>')
        elif x != registered_boot_seed(int(m.group(1)), m.group(2)):
            errs.append(f'bootstrap seed {k} is not the registered {registered_boot_seed(int(m.group(1)), m.group(2))}')
    return sorted(set(errs))


def _diff_set(checks):
    return {(c['wire'], c['view'], c['v'], c['R'], c['recipe']) for c in checks
            if c.get('recipe') in NR.SEED_DEPENDENT and c.get('equal_k0_k1') is False}


def _distinct_of(wire, v, view, distinct_inputs):
    n = ((distinct_inputs or {}).get(f'{wire}|V{v}') or {}).get(view)
    return n if isinstance(n, int) and not isinstance(n, bool) else None


def _distinct(c, distinct_inputs):
    return _distinct_of(c['wire'], c['v'], c['view'], distinct_inputs)


def _explained_nrm(c, diff, distinct_inputs):
    """NRM1 A-SC1 (iii), exactly as nrm.rules.grid_validity states it: (a) another seed-dependent recipe differs in the same wire,
    view, version and repetition; (b) the same recipe differs on another view of the same wire; (c) the view's distinct-input count
    is recorded. 'Differs' = a RECORDED check with equal_k0_k1 False."""
    a = any((c['wire'], c['view'], c['v'], c['R'], r) in diff for r in NR.SEED_DEPENDENT if r != c['recipe'])
    b = any((c['wire'], w, c['v'], c['R'], c['recipe']) in diff for w in NR.VIEWS if w != c['view'])
    n = ((distinct_inputs or {}).get(f"{c['wire']}|V{c['v']}") or {}).get(c['view'])
    return bool(a and b and isinstance(n, int))


def support_bound(recipe):
    """Largest distinct-input count of a view on which the extra-trees recipe can reduce to a seed-invariant table (8001 //
    min_leaf); None for other recipes."""
    return ER_POOL_PEOPLE // ER_MIN_LEAF[recipe] if recipe in ER_MIN_LEAF else None


def _explained_er(c, diff, distinct_inputs):
    """The inherited minimum-set explanation (FRC1 ER1, restricted to its mechanism): an extra-trees recipe (ET5 / ET25), NRM1
    (iii) (a) + (b) + (c), and the recorded distinct-input count of the view at most support_bound(recipe)."""
    if c.get('recipe') not in ER_MIN_LEAF or not _explained_nrm(c, diff, distinct_inputs):
        return False
    n = _distinct(c, distinct_inputs)
    return n is not None and 0 < n <= support_bound(c['recipe'])


def _explained_fv(c, diff, distinct_inputs):
    """The few-valued explanation (SAR1, unchanged; RULES.md 9): the view's recorded distinct-input count is at most
    FEW_VALUED_MAX (320) AND the same recipe demonstrably uses its seed in the same version, repetition and seed pair (a recorded
    check with equal_k0_k1 False on some other wire or view). Such an equality is counted once (n_eff); it never invalidates."""
    n = _distinct(c, distinct_inputs)
    if n is None or not 0 < n <= FEW_VALUED_MAX:
        return False
    return any(d[4] == c['recipe'] and d[2] == c['v'] and d[3] == c['R'] and (d[0], d[1]) != (c['wire'], c['view']) for d in diff)


def _explained_fv_pair(c, diff, distinct_inputs):
    """The DRF1 finite-pair extension (prospective; RULES.md 9): a PAIR-view equality whose recorded distinct-input count n
    exceeds 320 is explained iff (i) the same wire's recorded income and occupation distinct counts in that version are both at
    most 320 and n <= their product <= 1024 (the pair input set of a finite release, bounded by construction); (ii) the same recipe
    differs (a recorded check with equal_k0_k1 False) on the PAIR view of another wire in the same version, repetition and seed
    pair (the pair-view code path demonstrably receives the seed). Flagged FEW_VALUED_PAIR_ABOVE_320; counted once (n_eff). A
    necessary condition of the mechanism only: with more than 320 values there are fewer than 25 people per value on average, so
    the count alone does not make seed invariance likely."""
    if c.get('view') != 'pair':
        return False
    n = _distinct(c, distinct_inputs)
    ni = _distinct_of(c['wire'], c['v'], 'income', distinct_inputs)
    no = _distinct_of(c['wire'], c['v'], 'occupation', distinct_inputs)
    if None in (n, ni, no) or not (FEW_VALUED_MAX < n <= min(ni * no, PAIR_FINITE_MAX)):
        return False
    if not (0 < ni <= FEW_VALUED_MAX and 0 < no <= FEW_VALUED_MAX):
        return False
    return any(d[4] == c['recipe'] and d[2] == c['v'] and d[3] == c['R'] and d[1] == 'pair' and d[0] != c['wire'] for d in diff)


def _group_checks(checks):
    """Checks grouped by their reader-seed pair ('k_pair', default (0, 1)); each pair is judged on its own."""
    groups, errs = {}, []
    for c in checks or []:
        kp = c.get('k_pair', (0, 1)) if isinstance(c, dict) else None
        if not isinstance(kp, (list, tuple)) or len(kp) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in kp):
            errs.append(f'malformed seed check {c!r}')
            continue
        groups.setdefault(tuple(kp), []).append(c)
    return (groups or {(0, 1): []}), errs


def _recipe_part(seeds, checks, minimum_set, distinct_inputs):
    """One seed-pair group: NRM1 A-SC1 (ii)-(iii) with the DRF1 explanations; nrm.rules.grid_validity is called unchanged (with
    the real minimum set for the strict reading, with an empty one for (iii) on every wire) and the local classification is
    asserted against it."""
    sd = seeds if isinstance(seeds, dict) else {}
    try:
        base = NR.grid_validity(sd, checks, [], distinct_inputs)
        strict = NR.grid_validity(sd, checks, minimum_set, distinct_inputs)
        eq = [c for c in checks if c.get('recipe') in NR.SEED_DEPENDENT and c.get('equal_k0_k1') is not False]
        diff = _diff_set(checks)
        unexp_nrm = [c for c in eq if not _explained_nrm(c, diff, distinct_inputs)]
    except (KeyError, TypeError, AttributeError) as ex:
        return {'errors': [f'malformed seed checks: {ex!r}'], 'flags': [], 'explained': [], 'unexplained_min': [],
                'strict_errors': ['malformed seed checks'], 'sar1_errors': ['malformed seed checks']}
    if len(eq) - base['explained_equalities'] != len(unexp_nrm):
        raise AssertionError('DRF1 classification disagrees with nrm.rules.grid_validity')
    mins = set(minimum_set or ())
    explained, unexp_min, unexp_other, sar1_unexp = [], [], [], 0
    for c in eq:
        on = c['wire'] in mins
        if on and _explained_er(c, diff, distinct_inputs):
            explained.append({**c, 'explanation': 'ER_EXTRA_TREES_SUPPORT', 'minimum_set': True})
        elif (not on) and _explained_nrm(c, diff, distinct_inputs):
            explained.append({**c, 'explanation': 'NRM1_III', 'minimum_set': False})
        elif _explained_fv(c, diff, distinct_inputs):
            explained.append({**c, 'explanation': 'FEW_VALUED', 'minimum_set': on})
        elif _explained_fv_pair(c, diff, distinct_inputs):
            explained.append({**c, 'explanation': 'FEW_VALUED_PAIR_ABOVE_320', 'minimum_set': on})
            sar1_unexp += 1
        elif on:
            unexp_min.append(c)
        else:
            unexp_other.append(c)
    errors = [e for e in base['errors'] if not e.startswith('reader seeds') and 'unexplained seed-dependent' not in e]
    if unexp_min:
        errors.append(f'{len(unexp_min)} unexplained seed-dependent equalities on the minimum set')
    if unexp_other:
        errors.append(f'{len(unexp_other)} unexplained seed-dependent equalities')
    sar1 = list(errors) + ([f'{sar1_unexp} pair-view equalities above 320 (not explained under SAR1)'] if sar1_unexp else [])
    return {'errors': errors, 'flags': base.get('flags', []), 'explained': explained, 'unexplained_min': unexp_min,
            'unexplained_other': len(unexp_other), 'checks': len(checks),
            'strict_errors': [e for e in strict['errors'] if not e.startswith('reader seeds')], 'sar1_errors': sar1}


def _n_eff_report(explained, checks, distinct_inputs):
    out = {}
    for c in explained:
        kp = tuple(c.get('k_pair', (0, 1)))
        key = f"{c['wire']}|{c['view']}|{c['recipe']}|k{kp[0]}k{kp[1]}"
        if key in out:
            continue
        same = (lambda x, c=c, kp=kp: x.get('wire') == c['wire'] and x.get('view') == c['view'] and x.get('recipe') == c['recipe']
                and tuple(x.get('k_pair', (0, 1))) == kp)
        blocks = {(x['v'], x['R']) for x in checks if same(x)}
        eqb = {(x['v'], x['R']) for x in explained if same(x)}
        out[key] = {'blocks_checked': len(blocks), 'blocks_seed_pair_equal': len(eqb),
                    'n_eff_seed_cells_if_this_recipe_wins': 2 * len(blocks) - len(eqb), 'explanation': c['explanation'],
                    'minimum_set': c['minimum_set'],
                    'distinct_inputs': {f'V{v}': ((distinct_inputs or {}).get(f"{c['wire']}|V{v}") or {}).get(c['view'])
                                        for v in VERSIONS}, 'support_bound': support_bound(c['recipe']),
                    'few_valued_max': FEW_VALUED_MAX, 'pair_finite_max': PAIR_FINITE_MAX}
    return out


def grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, required_k=NR.REG_K, reps=NR.REG_REPS):
    """Reader-seed validity of the registered DRF1 grid, applied prospectively (RULES.md 9). READER_SEEDS_EFFECTIVE iff
    (1) seeds: the required S<v>k<k> recorded, distinct integers, every one equal to MODEL_SEED_BASE + 1100 + 10 v + k, none an
        earlier realized grid's seed (offsets 300, 400, ..., 1000);
    (2) bootstrap: S<v><R> recorded for every version and required repetition, equal to 2026102500 + 100 v + repetition index;
    (3) per seed pair (k0 / k1; for the refit also k2 / k3): every seed-dependent recipe checked; every seed-dependent equality
        explained: off the minimum set by NRM1 A-SC1 (iii), the few-valued or the finite-pair explanation; on the minimum set (DR,
        TWIN, the saved CTRL points, OBJ_MI and SUCC; for the refit the D2R release and its decisive comparators) by the
        extra-trees support, the few-valued or the finite-pair explanation. Explained equalities count once (n_eff), reported.
    The strict NRM1 reading and the SAR1 reading (few-valued bound 320 on every view) are reported beside."""
    errors = _seed_errors(seeds, required_k) + _boot_errors(boot_seeds, reps)
    groups, gerrs = _group_checks(checks)
    errors += gerrs
    per, exp_all, strict_errors, sar1_errors, flags = {}, [], [], [], set()
    for kp, ch in sorted(groups.items()):
        part = _recipe_part(seeds, ch, minimum_set, distinct_inputs)
        tag = f'k{kp[0]}k{kp[1]}'
        errors += [f'{tag}: {e}' for e in part['errors']]
        strict_errors += [f'{tag}: {e}' for e in part['strict_errors']]
        sar1_errors += [f'{tag}: {e}' for e in part['sar1_errors']]
        flags.update(part['flags'])
        exp_all += [{**c, 'k_pair': list(kp)} for c in part['explained']]
        per[tag] = {'checks': part.get('checks', 0), 'errors': part['errors'],
                    'explained': len(part['explained']),
                    'explained_few_valued': sum(c['explanation'].startswith('FEW_VALUED') for c in part['explained']),
                    'unexplained_minimum_set': len(part['unexplained_min']), 'unexplained_other': part.get('unexplained_other')}
    seed_errs = _seed_errors(seeds, required_k) + _boot_errors(boot_seeds, reps)
    ok = not errors
    if any(c['explanation'] == 'FEW_VALUED' for c in exp_all):
        flags.add('FEW_VALUED_SEED_EQUALITIES_COUNTED_ONCE')
    if any(c['explanation'] == 'FEW_VALUED_PAIR_ABOVE_320' for c in exp_all):
        flags.add('FEW_VALUED_PAIR_ABOVE_320')
    return {'label': 'READER_SEEDS_EFFECTIVE' if ok else 'READER_SEEDS_NOT_EFFECTIVE', 'effective': ok, 'errors': errors,
            'per_seed_pair': per, 'flags': sorted(flags),
            'explained_equalities': {
                'count': len(exp_all), 'n_eff_implied': _n_eff_report(exp_all, [c for c in checks or [] if isinstance(c, dict)],
                                                                      distinct_inputs),
                'items': [{k: c.get(k) for k in ('wire', 'view', 'v', 'R', 'recipe', 'k_pair', 'explanation', 'minimum_set')}
                          for c in exp_all],
                'note': 'a reader on a few-valued view can reduce to a seed-invariant ranking of the few values; its seed-pair '
                        'cells count once (n_eff) and the comparison is not declared invalid for it; the seed is shown to reach '
                        'the recipe elsewhere; se_agg is unaffected (se_rs uses block means over reader seeds)'},
            'strict_nrm1': {'label': 'READER_SEEDS_NOT_EFFECTIVE' if (seed_errs or strict_errors) else 'READER_SEEDS_EFFECTIVE',
                            'errors': seed_errs + strict_errors},
            'sar1_reading': {'label': 'READER_SEEDS_NOT_EFFECTIVE' if (errors or sar1_errors) else 'READER_SEEDS_EFFECTIVE',
                             'errors': seed_errs + sar1_errors},
            'seed_offset': SEED_OFFSET, 'forbidden_offsets': list(FORBIDDEN_OFFSETS), 'required_k': list(required_k),
            'reps': list(reps), 'few_valued_max': FEW_VALUED_MAX, 'pair_finite_max': PAIR_FINITE_MAX,
            'rule': 'DRF1 prospective: NRM1 (iii) off the minimum set, the extra-trees support explanation on it, the few-valued '
                    'explanation (<= 320) everywhere, and the finite-pair extension (pair view, 320 < n <= n_inc x n_occ <= 1024, '
                    'seed shown on the pair view of another wire)'}


# ---------------------------------------------------------------- (8) genuinely effective classifier refit
def refit_effectiveness(splits, seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, rob_reps=None):
    """Refit-effectiveness check (inherited logic, DRF1 explanations). splits = {'R1'..'R4': {'assignment_sha256': str,
    'score_overlap': {other repetition: share of SCORE households shared}}}; rob_reps = the robustness repetitions actually run
    (default: those of R3 / R4 present in splits; at least one is required); seeds = {'S<v>k<k>': int} for k = 0..3; checks with
    'k_pair' ([0, 1] or [2, 3]); minimum_set = the D2R release and its decisive comparators; boot_seeds = {'S<v><R>': int} for the
    repetitions run. REFIT_GENUINELY_EFFECTIVE iff (a) every robustness repetition run uses a different household split (assignment
    hashes recorded and distinct from each other and from R1 / R2; every SCORE overlap involving it recorded, finite and in [0, 1))
    and (b) grid_validity holds on all four reader seeds and every repetition run. Otherwise REFIT_NOT_GENUINELY_EFFECTIVE and the
    failed part's cells are counted once (no replication credit). Flag ONE_ROBUSTNESS_SPLIT_ONLY when only one of R3 / R4 ran."""
    errs, flags = [], []
    rob = tuple(r for r in NR.ROB_REPS if r in (splits or {})) if rob_reps is None else tuple(rob_reps)
    if not rob or any(r not in NR.ROB_REPS for r in rob):
        errs.append('NO_ROBUSTNESS_SPLIT')
        rob = tuple(r for r in rob if r in NR.ROB_REPS)
    if len(rob) == 1:
        flags.append('ONE_ROBUSTNESS_SPLIT_ONLY')
    reps = NR.REG_REPS + rob
    hs = [((splits or {}).get(r) or {}).get('assignment_sha256') for r in reps]
    if not all(isinstance(h, str) and h for h in hs) or len(set(hs)) != len(hs):
        errs.append('SPLITS_NOT_DIFFERENT_OR_NOT_RECORDED')
    else:
        for r in rob:
            for q in reps:
                if q == r:
                    continue
                o = (((splits or {}).get(r) or {}).get('score_overlap') or {}).get(q)
                if o is None:
                    o = (((splits or {}).get(q) or {}).get('score_overlap') or {}).get(r)
                if not _fin(o) or not 0 <= o < 1:
                    errs.append(f'SCORE_OVERLAP_{r}_{q}_NOT_RECORDED_OR_IDENTICAL')
    gv = grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=boot_seeds, required_k=ALL_K, reps=reps)
    if not gv['effective']:
        errs.append('READER_SEEDS_NOT_EFFECTIVE')
    count_once = (['robustness repetitions'] if any(e.startswith(('SPLITS', 'SCORE', 'NO_ROB')) for e in errs) else []) + \
                 (['robustness reader seeds'] if not gv['effective'] else [])
    return {'label': 'REFIT_GENUINELY_EFFECTIVE' if not errs else 'REFIT_NOT_GENUINELY_EFFECTIVE', 'reasons': errs,
            'flags': flags, 'robustness_repetitions': list(rob), 'grid_validity': gv, 'counted_once': count_once,
            'note': 'R3 / R4 are new household splits of the same MODEL_SELECT people and k2 / k3 new reader seeds: robustness to '
                    'classifier refitting, never replication'}


# ---------------------------------------------------------------- (2) D1: point eligibility, confidence conjunction, reserve
def _conditions(evals_by_version):
    """{v: {name: {'value', 'cap', 'se' (with se_mc in quadrature when recorded)}}} for the 14 conditions per version, or raises."""
    out = {}
    for v in VERSIONS:
        e = _v(evals_by_version, v)
        if e is None:
            raise ValueError(f'version {v} missing')
        u = e['utility'] if isinstance(e, dict) and 'utility' in e else e
        conds = NR.OR.conditions(u, v)
        mc = {}
        for c in u.get('clauses', []):
            mc[f"{c['task']}|{c['reference']}|{c['metric']}"] = c.get('se_mc')
        for g in u.get('guards', []):
            mc[f"guard|{g['group']}"] = g.get('se_mc')
        rec = {}
        for n, c in conds.items():
            se = c['se']
            m = mc.get(n)
            if _fin(se) and _fin(m) and m >= 0:
                se = math.sqrt(se * se + m * m)
            rec[n] = {'value': c['value'], 'cap': c['cap'], 'se': se}
        out[v] = rec
    return out


def confidence_conjunction(evals_by_version, z=Z):
    """Each of the 28 conditions from its own household-clustered SE (sqrt(se^2 + se_mc^2) when an MC SE is recorded; no floor):
    PASS iff value + z se <= cap; VIOLATION iff value - z se > cap; UNRESOLVED otherwise (neither a pass nor a violation).
    CONFIDENCE_ALL_PASS iff all 28 pass; CONFIDENCE_VIOLATION if any violation; else CONFIDENCE_UNRESOLVED; INCOMPLETE if a
    value or SE is missing. Reported beside G1: never gating, never redefining eligibility; 28 one-sided per-condition bounds, not
    a simultaneous statement (no multiplicity correction)."""
    try:
        conds = _conditions(evals_by_version or {})
    except (KeyError, ValueError, TypeError) as ex:
        return {'label': 'INCOMPLETE', 'error': str(ex)}
    per, lists = {}, {s: [] for s in CONDITION_STATES}
    for v in VERSIONS:
        per[v] = {}
        for n, c in conds[v].items():
            if not _fin(c['value'], c['se']) or c['se'] < 0:
                return {'label': 'INCOMPLETE', 'error': f'v{v}|{n}: value or SE missing'}
            lo, hi = c['value'] - z * c['se'], c['value'] + z * c['se']
            s = 'PASS' if hi <= c['cap'] else ('VIOLATION' if lo > c['cap'] else 'UNRESOLVED')
            per[v][n] = {'state': s, 'value': c['value'], 'se': c['se'], 'upper': hi, 'lower': lo, 'cap': c['cap']}
            lists[s].append(f'v{v}|{n}')
    lab = ('CONFIDENCE_VIOLATION' if lists['VIOLATION'] else 'CONFIDENCE_UNRESOLVED' if lists['UNRESOLVED']
           else 'CONFIDENCE_ALL_PASS')
    return {'label': lab, 'per_version': per, 'unresolved': lists['UNRESOLVED'], 'violations': lists['VIOLATION'],
            'n_pass': len(lists['PASS']), 'z': z,
            'note': 'one-sided z = 2 bound per condition from its own household-clustered SE; 28 bounds, not simultaneous; '
                    'reported, never gating; an unresolved condition is neither a pass nor a violation'}


def _reserve(evals):
    try:
        r = NR.reserve_status(evals or {})
        return {'status': r.get('status'), 'qualified': bool(r.get('qualified')), 'min_ratio_by_version': r.get('min_ratio_by_version'),
                'binding': r.get('binding')}
    except (KeyError, ValueError, TypeError) as ex:
        return {'status': 'INCOMPLETE', 'qualified': False, 'error': str(ex)}


_BESIDE = (('start', 'START'), ('twin', 'TWIN'), ('obj_mi', 'OBJ_MI'))


def d1(rec):
    """D1 task feasibility at DR (RULES.md 2). Gating label from G1 (NR.point_eligibility: all 28 original conditions at the
    point, original caps): TASK_FEASIBLE, CAND_INFEASIBLE, INCOMPLETE. Reported separately and never gating: the confidence
    conjunction and the two-SE reserve of DR, SUCC, TWIN and OBJ_MI; flags START_POINT_INELIGIBLE, TWIN_POINT_INELIGIBLE,
    OBJ_MI_POINT_INELIGIBLE (a point-ineligible control stays a control: its spend is read by the spend rules)."""
    rec = rec or {}
    cand = rec.get('cand') or {}
    flags, beside = [], {}
    for key, name in _BESIDE:
        ev = (rec.get(key) or {}).get('evals')
        lab = NR.point_eligibility(ev)['label'] if ev else None
        beside[name] = lab
        if lab == 'POINT_INELIGIBLE':
            flags.append(f'{name}_POINT_INELIGIBLE')
    out = {'beside': beside, 'cand': None, 'flags': flags}
    if cand.get('status') != 'FITTED':
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'error': 'DR status missing (FITTED required)'}
    cg = NR.point_eligibility(cand.get('evals') or {})
    out['cand'] = cg['label']
    if cg['label'] == 'INCOMPLETE':
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'error': cg.get('error')}
    lab = 'CAND_INFEASIBLE' if cg['label'] == 'POINT_INELIGIBLE' else 'TASK_FEASIBLE'
    conf = {'cand': confidence_conjunction(cand.get('evals'))}
    res = {'cand': _reserve(cand.get('evals'))}
    for key, name in _BESIDE:
        ev = (rec.get(key) or {}).get('evals')
        conf[name] = confidence_conjunction(ev) if ev else None
        res[name] = _reserve(ev) if ev else None
    out.update({'label': lab, 'met': lab == 'TASK_FEASIBLE', 'cand_failed': cg.get('failed'),
                'confidence': {k: (None if x is None else {kk: x.get(kk) for kk in ('label', 'unresolved', 'violations', 'n_pass')})
                               for k, x in conf.items()},
                'confidence_detail_cand': conf['cand'], 'reserve': res,
                'note': 'G1 at the point (gating) is not confidence preservation; the confidence conjunction and the two-SE reserve '
                        'are reported separately, never gating, never redefining eligibility'})
    return out


# ---------------------------------------------------------------- (4) brackets, fineness, OBJ spend, D2
def brackets(costs_vs, paths, alias_of=None, origin=TWIN_ID):
    """The bracketing control Q* of the CTRL path by NR's rule (the cheapest path point whose pooled MODEL_SELECT paired
    total-NLL cost is >= DR's, i.e. d = cost(DR) - cost(point) <= 0; ties keep path order). CTRL path = TWIN + its distinct saved
    points. {'CTRL': {'bracket': id | None, 'status': BRACKETED | NO_BRACKET | PATH_NOT_GIVEN | INCOMPLETE, 'path', 'primary'}}."""
    ids = control_path((paths or {}).get('CTRL'), alias_of, origin)
    if ids is None:
        return {'CTRL': {'bracket': None, 'status': 'PATH_NOT_GIVEN'}}
    prim = {}
    for i in ids:
        d, _ = NR._primary((costs_vs or {}).get(i))
        if d is None:
            return {'CTRL': {'bracket': None, 'status': 'INCOMPLETE', 'error': f'paired costs vs {i} missing or malformed',
                             'path': ids}}
        prim[i] = d['value']
    above = sorted((i for i in ids if prim[i] <= 0), key=lambda i: -prim[i])
    return {'CTRL': {'bracket': above[0] if above else None, 'status': 'BRACKETED' if above else 'NO_BRACKET', 'path': ids,
                     'primary': prim}}


def selection_data_error(sd):
    """None if the CTRL bracket data are allowed: MODEL_FIT_NESTED (MODEL_FIT nested held-out paired costs); never MODEL_SELECT,
    never CALIBRATION (not read by this campaign). Else 'NOT_RECORDED' or 'NOT_ALLOWED'."""
    if sd is None:
        return 'NOT_RECORDED'
    return None if sd in SELECTION_DATA else 'NOT_ALLOWED'


def grid_fineness(spends):
    """Fineness of the two saved CTRL points on MODEL_FIT nested held-out paired total-NLL spend (RULES.md 4.4; reported beside D2,
    never gating; computed before any MODEL_SELECT read). spends = {'CTRL': {'selection_data': 'MODEL_FIT_NESTED', v: {'cand': s =
    DR's nested spend over TWIN, 'lower': spend of the saved lower point (TWIN = 0.0), 'upper': spend of the saved upper point,
    'upper_se': household-clustered SE (same data) of the paired difference upper - DR, 'max': the largest path spend, optionally
    'grid': [{'spend', 'se'}] every path point with its paired-difference SE vs DR}}}. Per version: SELECTION_DATA_NOT_ALLOWED;
    INCOMPLETE (selection data not recorded, or an input missing); NOT_APPLICABLE_CAND_NOT_ABOVE_TWIN (s <= 0: the origin is the
    bracket); CONTROL_PATH_MAY_NOT_BRACKET (max < s + 2 SE); SAVED_POINT_RULE_NOT_FOLLOWED (lower > s, upper < s + 2 SE although
    max reaches it, or, with 'grid', lower not the largest point with spend <= s or upper not the smallest with spend >= s + 2 SE);
    FINE iff lower >= 0.75 s and upper <= s + 2 SE + 0.25 s; else COARSE_BELOW / COARSE_ABOVE / COARSE_BOTH. NOT_GIVEN if absent."""
    out, flags, data = {}, [], {}
    key = 'CTRL'
    src = (spends or {}).get(key)
    if src is None:
        return {'per_path': {key: 'NOT_GIVEN'}, 'selection_data': {}, 'flags': [], 'fraction': FINENESS_FRACTION}
    sd = src.get('selection_data') if isinstance(src, dict) else None
    data[key] = sd
    sde = selection_data_error(sd)
    per = {}
    for v in VERSIONS:
        x = _v(src, v)
        if sde == 'NOT_ALLOWED':
            per[v] = 'SELECTION_DATA_NOT_ALLOWED'
            continue
        if sde == 'NOT_RECORDED' or not isinstance(x, dict) or not _fin(x.get('cand')):
            per[v] = 'INCOMPLETE'
            continue
        s = float(x['cand'])
        if s <= 0:
            per[v] = 'NOT_APPLICABLE_CAND_NOT_ABOVE_TWIN'
            continue
        if not _fin(x.get('lower'), x.get('upper'), x.get('upper_se'), x.get('max')) or x['upper_se'] < 0:
            per[v] = 'INCOMPLETE'
            continue
        need = s + Z * float(x['upper_se'])
        if x['max'] < need:
            per[v] = 'CONTROL_PATH_MAY_NOT_BRACKET'
            continue
        exact = True
        g = x.get('grid')
        if g is not None:
            if not isinstance(g, list) or not all(isinstance(p, dict) and _fin(p.get('spend'), p.get('se')) for p in g):
                per[v] = 'INCOMPLETE'
                continue
            lows = [0.0] + [float(p['spend']) for p in g if p['spend'] <= s]
            ups = [float(p['spend']) for p in g if p['spend'] >= s + Z * float(p['se'])]
            exact = abs(max(lows) - x['lower']) <= TIE_TOL and (not ups or abs(min(ups) - x['upper']) <= TIE_TOL)
        if x['lower'] > s or x['upper'] < need or not exact:
            per[v] = 'SAVED_POINT_RULE_NOT_FOLLOWED'
            continue
        lo_ok = x['lower'] >= (1 - FINENESS_FRACTION) * s
        hi_ok = x['upper'] <= need + FINENESS_FRACTION * s
        per[v] = 'FINE' if lo_ok and hi_ok else ('COARSE_BELOW' if hi_ok else ('COARSE_ABOVE' if lo_ok else 'COARSE_BOTH'))
    vals = list(per.values())
    if any(p.startswith('COARSE') for p in vals):
        flags.append(f'CONTROL_GRID_COARSE_{key}')
    if 'CONTROL_PATH_MAY_NOT_BRACKET' in vals:
        flags.append(f'CONTROL_PATH_MAY_NOT_BRACKET_{key}')
    if 'SAVED_POINT_RULE_NOT_FOLLOWED' in vals:
        flags.append(f'CONTROL_POINT_RULE_NOT_FOLLOWED_{key}')
    if sde == 'NOT_ALLOWED':
        flags.append(f'CONTROL_SELECTION_DATA_NOT_ALLOWED_{key}')
    elif sde == 'NOT_RECORDED':
        flags.append(f'CONTROL_SELECTION_DATA_NOT_RECORDED_{key}')
    elif 'INCOMPLETE' in vals:
        flags.append(f'CONTROL_GRID_FINENESS_INCOMPLETE_{key}')
    out[key] = per
    return {'per_path': out, 'selection_data': data, 'flags': sorted(flags), 'fraction': FINENESS_FRACTION,
            'rule': 'on MODEL_FIT nested held-out paired costs (never MODEL_SELECT), with the SE from the same data: lower saved '
                    'point >= 0.75 s and upper saved point <= s + 2 SE + 0.25 s (s = DR spend over TWIN > 0); reported, never '
                    'gating, never a reason to set a reading aside'}


def obj_spend_state(costs):
    """The MODEL_SELECT spend state of OBJ_MI against DR (RULES.md 4.2; a single control point cannot be chosen to match spend, so
    the inherited point-bracket rule becomes a resolution rule). d = pooled paired total-NLL cost(DR) - cost(OBJ_MI) (NR._primary:
    mean of versions, SE = mean of version SEs): OBJ_MI_SPENDS_AT_LEAST_AS_MUCH iff d <= 0; OBJ_MI_SPENDS_LESS_UNRESOLVED iff
    0 < d and d - 2 SE <= 0 (credited, disclosed by a flag); OBJ_MI_SPENDS_LESS_RESOLVED iff d - 2 SE > 0 (the objective is not
    credited for a spend change). (record, error)."""
    d, f = NR._primary(costs)
    if d is None:
        return None, 'paired costs vs OBJ_MI missing or malformed'
    if d['value'] <= 0:
        st = OBJ_SPEND_STATES[0]
    elif d['value'] - Z * d['se'] <= 0:
        st = OBJ_SPEND_STATES[1]
    else:
        st = OBJ_SPEND_STATES[2]
    return {'state': st, 'd': d['value'], 'se': d['se'], 'upper': d['value'] + Z * d['se'], 'lower': d['value'] - Z * d['se'],
            'by_version': d['by_version'], 'flags': f,
            'note': 'd = cost(DR) - cost(OBJ_MI) on MODEL_SELECT total NLL: positive means DR spends more'}, None


def _targeting_flags(tg, enhanced):
    rep, flags = {}, []
    for q in tg:
        x = _point_positive((enhanced or {}).get(q))
        rep[q] = x
        if x is not None and not x['pass']:
            flags.append(f'READER_SENSITIVE_ENHANCED_{q}')
    return rep, flags


def _gate_of(grid, q):
    c = NR.comparison(grid.get(q) or {})
    if c['benefit']['label'] == 'INCOMPLETE' or any(c['recipient'][w]['label'] == 'INCOMPLETE' for w in NR.SINGLE_VIEWS):
        return None, c
    return c['recipient_gate_pass'], c


def _frontier_part(ids, point, grid, origin_id):
    """The CTRL frontier comparison of D2 (c): (record, error, flags)."""
    key = 'CTRL'
    flags = [f'{key}_AT_ORIGIN'] if ids == [origin_id] else []
    fr = NR.frontier_comparison(point, [{'id': i} for i in ids])
    if fr['label'] == 'INCOMPLETE':
        return None, f'{key}_FRONTIER: {fr.get("error")}', flags
    q = fr.get('bracketing_control')
    gate, rcp = None, None
    if q is not None:
        gate, cq = _gate_of(grid, q)
        if gate is None:
            return None, f'{key}_FRONTIER: recipient views vs bracketing control {q} missing or incomplete', flags
        rcp = {w: cq['recipient'][w]['label'] for w in NR.SINGLE_VIEWS}
    prim = fr.get('primary_cost_differences') or {}
    dominated = []
    for i in ids:
        pv = (prim.get(i) or {}).get('value')
        if _fin(pv) and pv > 0:                       # the point is cheaper than DR
            neg = _resolved_negative((grid.get(i) or {}).get('pair'))
            if neg is None:
                return None, f'{key}_FRONTIER: grid vs cheaper control point {i} missing', flags
            if neg:
                dominated.append(i)
    ok = fr['label'] == 'DOMINATES_CONTROL_FRONTIER' and gate is True and not dominated
    reason, label = fr.get('reason'), fr['label']
    no_bracket = fr.get('reason') == 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT'
    if fr['label'] == 'DOMINATES_CONTROL_FRONTIER' and gate is False:
        reason = 'RECIPIENT_GATE_FAILED_AT_BRACKETING_CONTROL'
        flags.append(f'RECIPIENT_GATE_FAILED_VS_{q}')
    if dominated:
        reason = 'DOMINATED_BY_CHEAPER_CONTROL' if fr['label'] == 'DOMINATES_CONTROL_FRONTIER' else reason
        flags += [f'DOMINATED_BY_CHEAPER_{i}' for i in dominated]
    if no_bracket:
        flags.append(f'NO_BRACKET_{key}')
        label, reason = 'NOT_SEPARATED', 'NO_BRACKET'
    if q is not None and q == origin_id:
        flags.append(f'{key}_BRACKET_IS_ORIGIN')
    over = {i: prim[i]['value'] for i in ids if _fin((prim.get(i) or {}).get('value')) and prim[i]['value'] > 0}
    mismatch = {'at_bracket': ({'value': prim[q]['value'], 'se': prim[q]['se'], 'by_version': prim[q].get('by_version'),
                                'note': 'cost(DR) - cost(Q*) <= 0: Q* spends at least this much more total NLL than DR'}
                               if q is not None else None),
                'smallest_overspend_vs_cheaper_point': (min(over.values()) if over else None),
                'interpolation': fr.get('interpolation')}
    return ({'label': label, 'frontier_label': fr['label'], 'reason': reason, 'path': ids, 'point': q,
             'recipient_gate_vs_bracket': gate, 'recipient_vs_bracket': rcp, 'dominated_by_cheaper': dominated, 'pass': ok,
             'gating': True, 'benefit_mean': fr.get('benefit_mean'), 'benefit_se': fr.get('benefit_se'),
             'residual_spend_mismatch': mismatch, 'statement': NO_BRACKET_STATEMENT.format(key=key) if no_bracket else None,
             'frontier': fr}, None, flags)


def _obj_part(point, grid, obj_id=OBJ_ID):
    """The OBJ comparison of D2 (o) against the single OBJ_MI point: the pair benefit and recipient gates by NR.comparison, the
    secondary-metric guard computed exactly as NR.frontier_comparison computes it at its bracket (pooled paired cost, resolved
    higher iff value - 2 SE > 0), and the single-point spend state (obj_spend_state) in place of the bracket choice. Pass iff the
    pair benefit is AGG_POSITIVE, no secondary metric is resolved higher, the recipient gate holds and OBJ_MI is not resolved
    cheaper than DR. NR.frontier_comparison with the one-point path [OBJ_MI] (the strict point rule) is reported beside.
    (record, error, flags)."""
    flags = []
    if obj_id not in grid:
        return None, f'grid entry vs OBJ_MI ({obj_id}) missing', flags
    gate, c = _gate_of(grid, obj_id)
    if gate is None:
        return None, 'grid entry vs OBJ_MI incomplete', flags
    sp, err = obj_spend_state((point.get('costs_vs') or {}).get(obj_id))
    if err:
        return None, err, flags
    sec = {}
    for m in NR.SECONDARY:
        per = {v: (_v((point.get('costs_vs') or {}).get(obj_id), v) or {}).get(m) for v in VERSIONS}
        p = NR._pooled(per) if all(isinstance(x, dict) for x in per.values()) else None
        if p is None:
            return None, f'secondary paired cost {m} vs OBJ_MI missing or non-finite', flags
        sec[m] = {**p, 'resolved_more_spending': p['value'] - Z * p['se'] > 0}
    more = sorted(m for m, x in sec.items() if x['resolved_more_spending'])
    ben = c['benefit']
    alias = ben['label'] == 'ALIAS_ZERO'
    if alias:
        flags.append('CAND_ALIAS_OF_OBJ_MI')
    if sp['state'] == OBJ_SPEND_STATES[1]:
        flags.append('OBJ_MI_SPENDS_LESS_UNRESOLVED')
    if sp['state'] == OBJ_SPEND_STATES[2]:
        flags.append('OBJ_MI_SPENDS_LESS_RESOLVED')
    if gate is False:
        flags.append(f'RECIPIENT_GATE_FAILED_VS_{obj_id}')
    benefit_ok = bool(c['benefit_positive'])
    spend_ok = sp['state'] != OBJ_SPEND_STATES[2]
    ok = benefit_ok and gate is True and spend_ok and not more
    if ok:
        reason = None
    elif alias:
        reason = 'DR_EQUALS_OBJ_MI'
    elif benefit_ok and not spend_ok:
        reason = 'NOT_SEPARATED_FROM_SPEND'
    elif benefit_ok and more:
        reason = 'SECONDARY_METRIC_RESOLVED_HIGHER'
    elif benefit_ok:
        reason = 'RECIPIENT_GATE_FAILED'
    else:
        reason = 'NO_RESOLVED_BENEFIT_VS_OBJ_MI'
    stmt = None
    if reason == 'NOT_SEPARATED_FROM_SPEND':
        stmt = ('DR is resolved less recoverable than OBJ_MI but spends resolved more MODEL_SELECT total NLL: at identical '
                'constraints DR used more of the budget; the exact-AUC objective is not credited for a spend change (reported as '
                'a budget-use difference, never as objective alignment)')
    elif ok and sp['state'] == OBJ_SPEND_STATES[1]:
        stmt = ('OBJ_MI spends less than DR at the point but not resolved less; the objective comparison is credited with this '
                'unresolved spend mismatch disclosed')
    strict = NR.frontier_comparison(point, [{'id': obj_id}])
    return ({'label': 'OBJ_SEPARATED' if ok else 'NOT_SEPARATED', 'reason': reason, 'point': obj_id, 'pass': ok, 'gating': True,
             'inherited_point_rule_beside': {'label': strict.get('label'), 'reason': strict.get('reason')},
             'benefit': ben['label'], 'benefit_mean': ben['aggregate'].get('mean'), 'benefit_se': ben['aggregate'].get('se_agg'),
             'recipient_gate_pass': gate, 'recipient': {w: c['recipient'][w]['label'] for w in NR.SINGLE_VIEWS},
             'spend': sp, 'secondary_more_spending': more, 'secondary': sec,
             'passed_if_spend_matched': bool(benefit_ok and gate is True and not more), 'statement': stmt}, None, flags)


def _cand_gate(rec):
    """Shared gate of D2, FC and TWIN: (label, extra) or (None, g1 record)."""
    cand = rec.get('cand') or {}
    if cand.get('status') != 'FITTED':
        return 'INCOMPLETE', {'error': 'DR status missing (FITTED required)'}
    if cand.get('alias_of_twin') is True:
        return 'NOT_ASSESSED_EXACT_ALIAS', {'statement': 'DR transmits exactly the twin\'s vectors in both versions: the optimizer '
                                                         'made no move; no output change, no protection experiment'}
    g1 = NR.point_eligibility(cand.get('evals') or {})
    if g1['label'] == 'INCOMPLETE':
        return 'INCOMPLETE', {'error': f'DR utility incomplete: {g1.get("error")}'}
    stage = rec.get('reader_stage', 'RUN')
    if g1['label'] != 'POINT_ELIGIBLE':
        return 'NOT_ASSESSED_TASK_INFEASIBLE', {'g1_failed': g1.get('failed'), 'reader_stage': stage,
                                                'statement': 'DR fails an original usefulness condition: the competitive reader '
                                                             'stage may stop; the fitted release is preserved; reported as not run '
                                                             '(never negative, never protection)'}
    if stage not in ('RUN', None):
        return 'INCOMPLETE', {'error': f'reader stage recorded {stage!r} although DR passes G1 (the reader stage must run)'}
    return None, g1


def comparator_ids(rec):
    """(twin id, OBJ_MI id) of a reading: rec['twin']['id'] (default 'TWIN', e.g. 'TWB_C64') and rec['obj_mi']['config_id']
    (default 'OBJ_MI', e.g. 'DR_MI'); the keys of 'grid', 'costs_vs' and 'nolookup' for these two comparators."""
    return ((((rec or {}).get('twin') or {}).get('id') or TWIN_ID), (((rec or {}).get('obj_mi') or {}).get('config_id') or OBJ_ID))


def _alias_map(rec):
    tid = comparator_ids(rec)[0]
    return {k: (tid if x == 'TWIN' else x) for k, x in (rec.get('alias_of') or {}).items()}


def d2(rec):
    """D2 component gain at DR (RULES.md 4). Labels in order of precedence: INCOMPLETE (DR status or utility);
    NOT_ASSESSED_EXACT_ALIAS (DR = TWIN); NOT_ASSESSED_TASK_INFEASIBLE (DR fails G1); NOT_COMPUTED_GRID_VALIDITY_UNKNOWN /
    NOT_COMPUTED_GRID_INVALID; NOT_ASSESSED_CONTROLS_NOT_RUN (CTRL or OBJ recorded not run for budget); INCOMPLETE (CTRL selection
    data not recorded or not MODEL_FIT_NESTED; any needed input missing); then the conjuncts (c) CTRL_FRONTIER (origin TWIN + the
    saved CTRL points: NR.frontier_comparison = DOMINATES_CONTROL_FRONTIER, the recipient gate vs Q*, no cheaper path point
    resolved less recoverable; a missing bracket reads NOT_SEPARATED, reason NO_BRACKET); (o) OBJ (AGG_POSITIVE vs OBJ_MI, the
    recipient gate, no secondary metric resolved higher, OBJ_MI not resolved cheaper); (d) reader targeting: the non-table winner's
    point benefit > 0 in both version means vs Q* and OBJ_MI. Result: COMPONENT_GAIN_BEYOND_FAMILY_CONTROLS (c, o and d);
    READER_TARGETED (c and o, d fails); NOT_SEPARATED (neither); else PARTIAL_<passed>_ONLY. TWIN (beside, never sufficient) and
    the family change vs SUCC (FC) are separate records. No partial outcome is a claim."""
    rec = rec or {}
    lab, extra = _cand_gate(rec)
    if lab:
        return {'label': lab, 'met': False, **extra}
    cand = rec.get('cand') or {}
    why = _gate_label(rec.get('grid_valid'))
    if why:
        return {'label': why, 'met': False}
    paths = rec.get('paths') or {}
    grid, nr = rec.get('grid') or {}, rec.get('controls_not_run') or {}
    not_run = ([k for k in ('CTRL',) if paths.get('CTRL') is None and nr.get('CTRL') is True] +
               [k for k in ('OBJ',) if comparator_ids(rec)[1] not in grid and nr.get('OBJ') is True])
    if not_run:
        return {'label': 'NOT_ASSESSED_CONTROLS_NOT_RUN', 'met': False, 'not_run': not_run,
                'statement': 'a family control was not run (budget): the comparison is reported as not run'}
    flags, comps = [], {}

    def incomplete(msg):
        return {'label': 'INCOMPLETE', 'met': False, 'error': msg, 'comparisons': comps, 'flags': sorted(set(flags))}
    e = selection_data_error((rec.get('selection_data') or {}).get('CTRL'))
    if e == 'NOT_RECORDED':
        return incomplete('the CTRL bracket\'s selection data are not recorded (plan: MODEL_FIT_NESTED)')
    if e == 'NOT_ALLOWED':
        flags.append('CONTROL_SELECTION_DATA_NOT_ALLOWED_CTRL')
        return incomplete(f'the CTRL bracket was selected on {(rec.get("selection_data") or {}).get("CTRL")!r}: only MODEL_FIT '
                          'nested held-out paired costs are allowed (never MODEL_SELECT); the control must be re-selected (a POST '
                          'HOC repair)')
    flags += list(((rec.get('fineness') or {}).get('flags')) or [])
    if paths.get('CTRL') is None:
        return incomplete('CTRL path missing (not recorded as not run)')
    point = {'id': cand.get('config_id', 'DR'), 'costs_vs': rec.get('costs_vs') or {}, 'grid_vs': grid}
    tid, oid = comparator_ids(rec)
    ids = control_path(paths.get('CTRL'), _alias_map(rec), tid)
    c, err, f = _frontier_part(ids, point, grid, tid)
    flags += f
    if err:
        return incomplete(err)
    comps['CTRL_FRONTIER'] = c
    c, err, f = _obj_part(point, grid, oid)
    flags += f
    if err:
        return incomplete(err)
    comps['OBJ'] = c
    tg = {}
    for q in dict.fromkeys(x['point'] for x in comps.values() if x.get('point')):
        x = _point_positive((rec.get('nolookup') or {}).get(q))
        if x is None:
            return incomplete(f'reader-targeting input (non-table winner) vs {q} missing')
        tg[q] = x
    passed = []
    for n in D2_COMPARISONS:
        x = comps[n]
        x['targeting_pass'] = tg[x['point']]['pass'] if x.get('point') in tg else None
        if x['pass']:
            passed.append(n)
            if x['targeting_pass'] is False:
                flags.append(f'READER_TARGETED_{n}')
    enh, eflags = _targeting_flags(tg, rec.get('enhanced'))
    flags += eflags
    allc = len(passed) == len(D2_COMPARISONS)
    targeted = not all(x['pass'] for x in tg.values())
    if allc and targeted:
        label = 'READER_TARGETED'
    elif allc:
        label = D2_MET
    elif not passed:
        label = 'NOT_SEPARATED'
    else:
        label = 'PARTIAL_' + '+'.join(passed) + '_ONLY'
    failed = [n for n in D2_COMPARISONS if n not in passed]
    statement = (D2_SCOPE if label == D2_MET else
                 'both family controls passed but the non-table winner does not show a positive point benefit in both versions '
                 'against every comparison point: the gain may be specific to the selected reader; not a component gain'
                 if label == 'READER_TARGETED' else
                 f'passed: {", ".join(passed) or "none"}; not passed: {", ".join(failed)}; a partial outcome is not a component '
                 'gain' + ('; without separation from the CTRL frontier the gain may be explained by task spending (SEX-blind '
                           'coarsening at matched spend)' if 'CTRL_FRONTIER' in failed else '') +
                 ('; without separation from OBJ_MI the gain may come from SEX-aware routing in the finite family rather than the '
                  'changed objective' if 'OBJ' in failed else '') +
                 ('; the objective is not credited for a spend change' if comps['OBJ'].get('reason') == 'NOT_SEPARATED_FROM_SPEND'
                  else ''))
    return {'label': label, 'met': label == D2_MET, 'passed': passed, 'failed': failed, 'gating': list(D2_COMPARISONS),
            'comparisons': comps, 'targeting': tg, 'enhanced_reported_only': enh, 'flags': sorted(set(flags)),
            'minimum_detectable': MDE_NOTE, 'scope': D2_SCOPE, 'statement': statement,
            'note': 'no partial outcome is a claim; "not separated" is the default reading; every sentence quoting this label '
                    'carries its scope'}


def _simple_comparison(rec, comparator, labels, name):
    """FC / TWIN: DR vs one frozen comparator, NR.comparison (AGG_POSITIVE + recipient gates); shares D2's gates."""
    rec = rec or {}
    lab, extra = _cand_gate(rec)
    if lab:
        return {'label': lab, 'met': False, **extra}
    why = _gate_label(rec.get('grid_valid'))
    if why:
        return {'label': why, 'met': False}
    gate, c = _gate_of(rec.get('grid') or {}, comparator)
    if gate is None:
        return {'label': 'INCOMPLETE', 'met': False, 'error': f'grid entry vs {comparator} missing or incomplete'}
    ok = bool(c['benefit_positive'] and gate)
    sp, _ = NR._primary((rec.get('costs_vs') or {}).get(comparator))
    return {'label': labels[0] if ok else labels[1], 'met': ok, 'comparator': comparator, 'benefit': c['benefit']['label'],
            'benefit_mean': c['benefit']['aggregate'].get('mean'), 'benefit_se': c['benefit']['aggregate'].get('se_agg'),
            'recipient_gate_pass': gate, 'recipient': {w: c['recipient'][w]['label'] for w in NR.SINGLE_VIEWS},
            'spend_vs_comparator': sp, 'name': name}


def family_change(rec):
    """FC: the family change, DR (finite family) vs START = SUCC (the unchanged strong start, outside the family), reported
    separately from the objective comparison (RULES.md 4.5): FAMILY_CHANGE_BELOW_START_RESOLVED iff AGG_POSITIVE pair benefit and
    both recipient gates; otherwise FAMILY_CHANGE_NOT_RESOLVED; D2's gate labels otherwise. Not attributable to the objective."""
    out = _simple_comparison(rec, START_ID, FC_LABELS, 'FC')
    out['statement'] = ('family change (finite family vs the unchanged strong start SUCC): reported separately from the objective '
                        'comparison; it attributes nothing to the exact-AUC objective (that is D2\'s OBJ conjunct) and is not a '
                        'component gain; SUCC is also one of the nine D3 references')
    return out


def twin_beside(rec):
    """TWIN: DR vs the family origin (weak task-only twin), reported beside D2, never sufficient alone (RULES.md 4.5)."""
    out = _simple_comparison(rec, comparator_ids(rec)[0], TWIN_LABELS, 'TWIN')
    out['statement'] = ('DR vs the family origin TWB_C64 (a weak task-only twin): reported beside, never sufficient alone; an '
                        'attribution against a weak twin is insufficient')
    return out


# ---------------------------------------------------------------- (5) D3, D4 with the reader-budget credit rule
def reference_credit(rec, refs):
    """A reference flagged READER_BUDGET_ASYMMETRIC (rec['reader_budget'][ref], pair view) is credited iff the non-table winner's
    point benefit against it ('nolookup') is > 0 in both version means; missing input -> INCOMPLETE (never 'not beaten')."""
    out = {}
    for ref in refs:
        if ((rec or {}).get('reader_budget') or {}).get(ref) == 'READER_BUDGET_ASYMMETRIC':
            x = _point_positive(((rec or {}).get('nolookup') or {}).get(ref))
            out[ref] = 'INCOMPLETE' if x is None else ('CREDITED' if x['pass'] else 'NOT_CREDITED')
    return out


def _d3(rec, cl):
    """D3 = NRM1 G4 (every point-eligible reference of the nine, intersection-union, recipient gates) with the reader-budget rule;
    the references passed and failed are listed."""
    lab = cl['G4']['label']
    cr = reference_credit(rec, NR.BENCHMARK)
    nc = sorted(r for r, x in cr.items() if x == 'NOT_CREDITED')
    inc = sorted(r for r, x in cr.items() if x == 'INCOMPLETE')
    if lab in ('SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES', 'MET_WITHOUT_TASK_FEASIBILITY'):
        lab = 'NOT_SUPERIOR' if nc else ('INCOMPLETE' if inc else lab)
    comps = cl.get('comparisons') or {}
    refs = cl['G4'].get('references') or []
    passed = [r for r in refs if r in comps and comps[r]['benefit_positive'] and comps[r]['recipient_gate_pass'] and r not in nc]
    failed = sorted(set(cl['G4'].get('failing') or []) | set(nc))
    return {'label': lab, 'met': lab == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES', 'passed': passed, 'failing': failed,
            'reader_budget_credit': cr, 'not_credited': nc, 'incomplete_references': inc, 'missing': cl['G4'].get('missing'),
            'per_reference': {r: {'benefit': comps[r]['benefit']['label'], 'mean': comps[r]['benefit']['aggregate'].get('mean'),
                                  'se_agg': comps[r]['benefit']['aggregate'].get('se_agg'),
                                  'recipient_gate_pass': comps[r]['recipient_gate_pass']} for r in refs if r in comps}}


def _d4(rec, cl):
    """D4 = NRM1 G5 (a resolved 0.020 pair benefit vs NEURAL_RAW and H_LJ with the recipient gates; levels kept) with the same
    reader-budget rule for the two targets; met iff MATERIAL_0020_RESOLVED; never weakened."""
    level = cl['G5'].get('level')
    cr = reference_credit(rec, NR.TARGET_REFS)
    if any(x == 'NOT_CREDITED' for x in cr.values()):
        level = 'NOT_MET'
    elif any(x == 'INCOMPLETE' for x in cr.values()) and level != 'NOT_MET':
        level = 'INCOMPLETE'
    feas = cl['G1']['label'] == 'TASK_FEASIBLE'
    lab = level if (feas or level in ('NOT_MET', 'INCOMPLETE')) else 'MET_WITHOUT_TASK_FEASIBILITY'
    return {'label': lab, 'level': level, 'met': lab == 'MATERIAL_0020_RESOLVED', 'reader_budget_credit': cr,
            'nrm_g5_label': cl['G5']['label'], 'per_reference': cl['G5'].get('per_reference')}


def _d5(rec, cand):
    """D5 = NRM1 G6 unchanged (rec['joint'] passed to NR.claims), except INCOMPLETE when a declared joint design names all three
    comparators but a grid entry or paired cost of one is missing; sequential orders that collapse BY CONSTRUCTION
    (joint['collapsed']) give the reduced test (separate) or NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL (joint). No joint design
    declared -> NOT_APPLICABLE_NOT_JOINT (not claimed)."""
    j = rec.get('joint')
    grid = rec.get('grid') or {}
    keys = ('separate', 'income_first', 'occupation_first')
    flags = []
    joint_declared = isinstance(j, dict) and j.get('jointly_designed') is True and \
        {'income', 'occupation'} <= set(j.get('outputs_changed') or [])
    if joint_declared and j.get('collapsed') is not None:
        col = j.get('collapsed')
        if not isinstance(col, dict) or not col or any(k not in keys[1:] or x not in ('SEPARATE', 'JOINT') for k, x in col.items()):
            return {'label': 'INCOMPLETE', 'error': 'joint.collapsed must map income_first / occupation_first to SEPARATE or JOINT'}
        if any(x == 'JOINT' for x in col.values()):
            return {'label': 'NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL', 'collapsed': dict(col),
                    'statement': 'the joint design equals a sequential order by construction (card): no joint-design claim'}
        j = dict(j)
        for k in col:
            if j.get(k) not in (None, j.get('separate')):
                return {'label': 'INCOMPLETE', 'error': f'{k} collapses to the separate design but names another comparator'}
            j[k] = j.get('separate')
            flags.append(f'D5_REDUCED_{k.upper()}_EQUALS_SEPARATE')
    if joint_declared and all(j.get(k) for k in keys):
        missing = []
        for k in keys:
            n = j[k]
            c = NR.comparison(grid.get(n) or {})
            if c['benefit']['label'] == 'INCOMPLETE' or any(c['recipient'][w]['label'] == 'INCOMPLETE' for w in NR.SINGLE_VIEWS):
                missing.append(f'grid entry vs {k} comparator {n}')
            if NR._primary((j.get('costs_vs') or {}).get(n))[0] is None:
                missing.append(f'paired costs vs {k} comparator {n}')
        if missing:
            return {'label': 'INCOMPLETE', 'missing': missing, 'flags': flags}
    cl = NR.claims({'evals': cand.get('evals') or {}, 'grid': grid, 'grid_valid': rec.get('grid_valid'), 'frontier': None,
                    'matched': {}, 'joint': j if j else None})
    lab = cl['G6']['label']
    return {'label': lab, 'detail': cl['G6'], 'flags': flags, 'reduced_test': bool(flags),
            'statement': ('joint-design contribution supported against the separately designed recipients and both sequential '
                          'orders (NRM1 G6)' if lab == 'JOINT_CONTRIBUTION_SUPPORTED' else 'no joint-design advantage is claimed')}


# ---------------------------------------------------------------- (5) the optimistic MODEL_FIT diagnostic and the failure stage
def _useful_one(opt, kind, mode):
    """One construction in one mode, both versions: USEFUL / NOT_USEFUL / NOT_EVALUABLE_NO_BRACKET / INCOMPLETE."""
    per, labs = {}, []
    for v in VERSIONS:
        x = (_v((opt or {}).get(kind), v) or {}).get(mode)
        xin = (_v((opt or {}).get(kind), v) or {}).get('in_sample') or {}
        path = (_v((opt or {}).get('CTRL'), v) or {}).get(mode)
        if not isinstance(x, dict) or not isinstance(path, list) or not path:
            per[v] = {'label': 'INCOMPLETE', 'error': f'{kind} {mode} or the CTRL path on the same law missing'}
            labs.append('INCOMPLETE')
            continue
        auc = x.get('auc') or {}
        pts = [p for p in path if isinstance(p, dict)]
        if (not _fin(x.get('spend'), *(auc.get(w) for w in NR.VIEWS)) or len(pts) != len(path) or
                any(not _fin(p.get('spend'), *((p.get('auc') or {}).get(w) for w in NR.VIEWS)) for p in pts) or
                not isinstance(xin.get('within_budget'), bool)):
            per[v] = {'label': 'INCOMPLETE', 'error': f'{kind} {mode}: spend, AUCs or within_budget missing'}
            labs.append('INCOMPLETE')
            continue
        if any(p.get('law') != x.get('law') or x.get('law') is None for p in pts):
            per[v] = {'label': 'INCOMPLETE', 'error': f'{kind} {mode}: the CTRL path is not on the same law ({x.get("law")!r})'}
            labs.append('INCOMPLETE')
            continue
        above = [p for p in pts if p['spend'] >= x['spend']]
        if not above:
            per[v] = {'label': 'NOT_EVALUABLE_NO_BRACKET', 'spend': x['spend'], 'max_path_spend': max(p['spend'] for p in pts)}
            labs.append('NOT_EVALUABLE_NO_BRACKET')
            continue
        b = min(above, key=lambda p: p['spend'])            # min keeps path order among ties
        gain = b['auc']['pair'] - x['auc']['pair']
        r = {w: x['auc'][w] - b['auc'][w] for w in NR.SINGLE_VIEWS}
        ok = xin['within_budget'] and gain >= USEFUL_MARGIN - TIE_TOL and all(r[w] <= RECIPIENT_LIMIT + TIE_TOL for w in r)
        per[v] = {'label': 'USEFUL' if ok else 'NOT_USEFUL', 'bracket': b.get('id'), 'gain_pair': gain, 'single_view_excess': r,
                  'within_budget': xin['within_budget'], 'spend': x['spend'], 'bracket_spend': b['spend'], 'law': x.get('law')}
        labs.append(per[v]['label'])
    if 'INCOMPLETE' in labs:
        lab = 'INCOMPLETE'
    elif 'NOT_EVALUABLE_NO_BRACKET' in labs:
        lab = 'NOT_EVALUABLE_NO_BRACKET'
    else:
        lab = 'USEFUL' if all(x == 'USEFUL' for x in labs) else 'NOT_USEFUL'
    return {'label': lab, 'by_version': per}


def useful_optimistic(opt):
    """The MODEL_FIT optimistic diagnostic (RULES.md 5; prospective, comparator-based, reported, NEVER a gate). opt = {'OPT_S' |
    'OPT_L': {v: {'in_sample': X, 'nested': X}}, 'CTRL': {v: {'in_sample': [P, ...], 'nested': [P, ...]}}}, X = {'law', 'spend':
    MODEL_FIT total-NLL spend over the twin in that mode, 'auc': {'pair', 'income', 'occupation'} exact optimal-reader AUC in that
    mode, 'within_budget': bool (in-sample record: all 8 MODEL_FIT spends within the DR budgets)}, P = {'id', 'law', 'spend', 'auc'}
    the SEX-blind CTRL path on the same law and mode (TWIN first, spend 0). A construction is USEFUL in a mode iff in both versions
    it is within budget and, against the bracket B (the cheapest path point whose spend is >= its own), AUC_pair(B) - AUC_pair(X)
    >= 0.020 and neither single view exceeds B's by more than 0.010. USEFUL_OPTIMISTIC_POINT iff OPT_S or OPT_L is USEFUL in the
    in-sample mode; NO_USEFUL_OPTIMISTIC_POINT iff both are evaluable there and neither is; else UNDETERMINED. The nested mode is
    reported beside (held_out_survives). No absolute AUC threshold; no SE (exact plug-in values; a diagnostic, not inference)."""
    per = {k: {m: _useful_one(opt, k, m) for m in OPT_MODES} for k in OPT_KINDS}
    ins = [per[k]['in_sample']['label'] for k in OPT_KINDS]
    if 'USEFUL' in ins:
        lab = OPT_LABELS[0]
    elif all(x == 'NOT_USEFUL' for x in ins):
        lab = OPT_LABELS[1]
    else:
        lab = OPT_LABELS[2]
    held = [k for k in OPT_KINDS if per[k]['in_sample']['label'] == 'USEFUL' and per[k]['nested']['label'] == 'USEFUL']
    return {'label': lab, 'per_construction': per, 'useful_in_sample': [k for k in OPT_KINDS if per[k]['in_sample']['label'] ==
                                                                          'USEFUL'],
            'held_out_survives': held, 'useful_margin': USEFUL_MARGIN,
            'note': 'MODEL_FIT-only diagnostic of optimistic constructions (OPT_S is illegal at release by construction; OPT_L '
                    'memorizes the true-SEX law); reported, never a gate; neither success nor failure proves population '
                    'feasibility or impossibility'}


def failure_stage(dec, opt_record):
    """The failure-stage classification (RULES.md 5; reported, never gating). Legal stage from DR's decisions (raw labels):
    L2 (DR fails G1 or DR = TWIN), L3 (G1 kept, neither D2 nor D3 met), L4 (D2 met, D3 not), L5 (D3 met). Stage: L5 -> 5; L4 -> 4
    (scoped component result only with D2R ROBUSTNESS_CONFIRMED and no sign change or robustness recipient failure); L2 / L3 with
    NO_USEFUL_OPTIMISTIC_POINT -> 1; L2 with a useful point -> 2; L3 with a useful point -> 3; L2 / L3 with an UNDETERMINED
    diagnostic -> UNDETERMINED_OPTIMISTIC_DIAGNOSTIC; INCOMPLETE when D1, D2 or D3 is undecided."""
    d1l, d2l, d3l = (_raw(dec.get(k)) for k in ('D1', 'D2', 'D3'))
    olab = (opt_record or {}).get('label', 'UNDETERMINED')
    flags = []
    undecided = ('INCOMPLETE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID', 'NOT_ASSESSED_CONTROLS_NOT_RUN')
    if d1l == 'INCOMPLETE' or d1l is None:
        return {'label': 'INCOMPLETE', 'legal_stage': None, 'optimistic': olab, 'flags': flags,
                'statement': STAGE_STATEMENTS['INCOMPLETE']}
    if d2l == 'NOT_ASSESSED_EXACT_ALIAS' or d1l == 'CAND_INFEASIBLE':
        legal = 'L2_LEGAL_MAPPING_FAILS'
    elif d2l in undecided or d3l in undecided or d2l is None or d3l is None:
        return {'label': 'INCOMPLETE', 'legal_stage': None, 'optimistic': olab, 'flags': flags,
                'statement': STAGE_STATEMENTS['INCOMPLETE']}
    elif (dec.get('D3') or {}).get('met'):
        legal = 'L5_COMPETITIVE_CONDITIONS_PASS'
    elif (dec.get('D2') or {}).get('met'):
        legal = 'L4_BEATS_CONTROLS_MISSES_REFERENCES'
    else:
        legal = 'L3_KEEPS_UTILITY_TIES_CONTROLS'
    scoped = None
    if legal.startswith('L5'):
        lab = STAGES[4]
        if not (dec.get('D2') or {}).get('met'):
            flags.append('D3_WITHOUT_D2')
    elif legal.startswith('L4'):
        lab = STAGES[3]
        r2 = dec.get('D2R') or {}
        scoped = (r2.get('label') == 'ROBUSTNESS_CONFIRMED' and not r2.get('sign_changes') and
                  not r2.get('recipient_failures'))
        flags.append('SCOPED_COMPONENT_RESULT_ROBUSTNESS_SUPPORTED' if scoped else 'COMPONENT_RESULT_NOT_ROBUSTNESS_SUPPORTED')
    elif olab == OPT_LABELS[1]:
        lab = STAGES[0]
    elif olab == OPT_LABELS[0]:
        lab = STAGES[1] if legal.startswith('L2') else STAGES[2]
    else:
        lab = STAGES[5]
    if olab == OPT_LABELS[1] and legal[:2] in ('L4', 'L5'):
        flags.append('NO_USEFUL_OPTIMISTIC_POINT_BUT_LEGAL_MAPPING_PROGRESSED')
    return {'label': lab, 'legal_stage': legal, 'optimistic': olab, 'scoped_component_result': scoped, 'flags': flags,
            'statement': STAGE_STATEMENTS[lab], 'note': 'reported, never gating; the optimistic diagnostic only separates stage 1 '
                                                        'from stages 2 and 3'}


# ---------------------------------------------------------------- (7) registered vs adaptive readings
def _family(rec):
    """The reading's family: DR (the only candidate family of DRF1); revisions 0, 1, 2; anything else is refused."""
    fam = (rec or {}).get('family')
    if fam != FAMILY:
        raise ValueError(f'unknown family {fam!r}: a DRF1 reading is family DR')
    n = (rec or {}).get('revision', 0)
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= MAX_REVISIONS:
        raise ValueError(f'revision {n!r}: at most {MAX_REVISIONS} evidence-driven revisions (0, 1, 2)')
    return fam


def _note_ok(rec):
    note = (rec or {}).get('revision_note')
    return (isinstance(note, dict) and isinstance(note.get('observed_failure'), str) and note['observed_failure'].strip() and
            isinstance(note.get('changed_operation'), str) and note['changed_operation'].strip() and
            note.get('trigger') in REVISION_TRIGGERS)


def adaptive_tag(rec):
    """None for a registered reading: its card was committed before any MODEL_SELECT outcome of this campaign (revision 0, or a
    revision >= 1 with a complete prospective note whose trigger is MODEL_FIT or SYNTHETIC evidence). Otherwise 'ADAPTIVE_<parts>':
    R0_LATE_CARD (revision 0 committed after an outcome or not recorded), R<n> (a revision committed after an outcome, or triggered
    by a MODEL_SELECT outcome), R<n>_NO_NOTE (a revision without a complete prospective note naming the observed failure, the
    changed operation and the trigger)."""
    _family(rec)
    n = int((rec or {}).get('revision') or 0)
    early = (rec or {}).get('card_committed_before_any_model_select_outcome') is True
    if n == 0:
        return None if early else 'ADAPTIVE_R0_LATE_CARD'
    if not _note_ok(rec):
        return f'ADAPTIVE_R{n}_NO_NOTE'
    if early and rec['revision_note']['trigger'] != 'MODEL_SELECT':
        return None
    return f'ADAPTIVE_R{n}'


def _mark(out, tag):
    for k in ('D1', 'D2', 'FC', 'TWIN', 'D3', 'D4', 'D5', 'D6', 'failure_stage', 'outcome_class'):
        x = out.get(k)
        if isinstance(x, dict) and 'label' in x:
            x['label_unprefixed'] = x['label']
            if tag:
                x['label'] = f'{tag}__{x["label"]}'


def _raw(x):
    return (x or {}).get('label_unprefixed', (x or {}).get('label'))


# ---------------------------------------------------------------- (6) decisions, the three outcomes and the outcome class
def decisions(rec):
    """D1, D2, FC, TWIN, D3-D6, D2R, the failure stage, the three separately reported outcomes and the outcome class of one DR
    reading (RULES.md 2-7). D3-D5 are read at DR and carry D2's gate label when D2 is not assessed for aliasing, feasibility or
    grid validity. Outcome class, first applicable: INCOMPLETE (D1 incomplete); NO_OUTPUT_CHANGE (DR = TWIN); TASK_INFEASIBLE (DR
    fails G1); INCOMPLETE (D2 incomplete or not computed); PROTECTION_NOT_ASSESSED (a family control not run); INCOMPLETE (D3
    undecided); METHOD_CANDIDATE (D2 and D3 met; D4 reported separately, never implied); COMPONENT_GAIN_NOT_COMPETITIVE (D2 met,
    D3 not); REFERENCE_ADVANTAGE_NOT_ATTRIBUTED (D3 met, D2 not); FEASIBLE_NO_RELIABLE_GAIN. D4, D5, FC, TWIN, D2R and the stage
    never change the class. A non-registered reading has every label prefixed '<tag>__'."""
    rec = rec or {}
    _family(rec)
    cand = rec.get('cand') or {}
    out = {'family': rec.get('family'), 'card': rec.get('card'), 'revision': int(rec.get('revision') or 0),
           'D1': d1(rec), 'D2': d2(rec), 'FC': family_change(rec), 'TWIN': twin_beside(rec)}
    d2lab = out['D2']['label']
    cg = NR.point_eligibility(cand.get('evals') or {})['label'] if cand.get('status') == 'FITTED' else None
    gates = ('NOT_ASSESSED_EXACT_ALIAS', 'NOT_ASSESSED_TASK_INFEASIBLE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN',
             'NOT_COMPUTED_GRID_INVALID')
    if d2lab in gates:
        for k in ('D3', 'D4', 'D5'):
            out[k] = {'label': d2lab, 'met': False}
    elif cand.get('status') != 'FITTED' or cg != 'POINT_ELIGIBLE' or rec.get('reader_stage', 'RUN') not in ('RUN', None):
        for k in ('D3', 'D4', 'D5'):
            out[k] = {'label': 'INCOMPLETE', 'met': False}
    else:
        cl = NR.claims({'evals': cand.get('evals') or {}, 'grid': rec.get('grid') or {}, 'grid_valid': rec.get('grid_valid'),
                        'frontier': None, 'matched': {}, 'joint': None})
        out['D3'] = _d3(rec, cl)
        out['D4'] = _d4(rec, cl)
        out['D5'] = _d5(rec, cand)
    nov = rec.get('novelty') or 'NOT_CLAIMED'
    out['D6'] = {'label': nov if nov in NR.NOVELTY_LABELS else 'NOT_CLAIMED',
                 'note': 'from a separate prior-art review; never inferred from performance'}
    r2 = rec.get('d2r') if isinstance(rec.get('d2r'), dict) else None
    out['D2R'] = {'label': (r2 or {}).get('label', 'NOT_RUN'), 'sign_changes': (r2 or {}).get('sign_changes', []),
                  'recipient_failures': (r2 or {}).get('recipient_failures', []),
                  'headline_caveat': (r2 or {}).get('headline_caveat')}
    out['optimistic'] = useful_optimistic(rec.get('optimistic'))
    out['failure_stage'] = failure_stage(out, out['optimistic'])
    flags = list(out['D2'].get('flags') or []) + list(rec.get('mc_flags') or []) + list(out['D5'].get('flags') or []) + \
        list(out['D1'].get('flags') or []) + list(out['failure_stage'].get('flags') or [])
    d1lab, d3lab = out['D1']['label'], out['D3']['label']
    why = None
    d2met, d3met = bool(out['D2'].get('met')), bool(out['D3'].get('met'))
    undecided = ('INCOMPLETE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID')
    if d1lab == 'INCOMPLETE':
        oc, why = 'INCOMPLETE', 'D1 INCOMPLETE'
    elif d2lab == 'NOT_ASSESSED_EXACT_ALIAS':
        oc = 'NO_OUTPUT_CHANGE'
        flags.append('FAMILY_STOPS_EXACT_ALIAS_OF_TWIN')
    elif d1lab == 'CAND_INFEASIBLE':
        oc = 'TASK_INFEASIBLE'
        if rec.get('reader_stage') == 'STOPPED_USEFULNESS_FAILURE':
            flags.append('READER_STAGE_STOPPED_RELEASE_PRESERVED')
    elif d2lab in undecided:
        oc, why = 'INCOMPLETE', f'D2 {d2lab}'
    elif d2lab == 'NOT_ASSESSED_CONTROLS_NOT_RUN':
        oc = 'PROTECTION_NOT_ASSESSED'
    elif d3lab in undecided:
        oc, why = 'INCOMPLETE', f'D3 {d3lab}'
    elif d2met and d3met:
        oc = 'METHOD_CANDIDATE'
    elif d2met:
        oc = 'COMPONENT_GAIN_NOT_COMPETITIVE'
    elif d3met:
        oc = 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED'
    else:
        oc = 'FEASIBLE_NO_RELIABLE_GAIN'
    if d3met and not d2met:
        flags.append('D3_WITHOUT_D2')
    if out['D4'].get('met') and not d2met:
        flags.append('D4_WITHOUT_D2')
    if out['TWIN'].get('met') and not d2met:
        flags.append('TWIN_ONLY_ATTRIBUTION_INSUFFICIENT')
    if cg == 'POINT_ELIGIBLE':
        try:
            if not NR.reserve_status(cand.get('evals') or {}).get('qualified'):
                flags.append('POINT_FEASIBLE_NOT_RESERVE_QUALIFIED')
        except (KeyError, ValueError, TypeError):
            flags.append('RESERVE_INCOMPLETE')
        conf = (out['D1'].get('confidence') or {}).get('cand') or {}
        if conf.get('label') != 'CONFIDENCE_ALL_PASS':
            flags.append('POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED')
    if out['D2R']['sign_changes']:
        flags.append('D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE')
    if out['D2R']['recipient_failures']:
        flags.append('D2R_RECIPIENT_FAILURE_CARRIED_INTO_HEADLINE')
    tag = adaptive_tag(rec)
    if int(rec.get('revision') or 0) and not _note_ok(rec):
        flags.append('REVISION_NOTE_MISSING')
    out['outcome_class'] = {'label': oc, 'flags': sorted(set(flags)), 'reason': why,
                            'note': 'development evidence on adaptively used MODEL_SELECT records, never confirmation; no class '
                                    'renames a partial result as the desired method; a near-limit point estimate is never called '
                                    'robust'}
    out['registered'] = tag is None
    out['adaptive_tag'] = tag
    out['registration_note'] = ('registered reading (card committed before any MODEL_SELECT outcome of this campaign)'
                                if tag is None else 'NOT a registered result: ' + tag + ' (labels prefixed; report beside the '
                                                    'registered reading with the number of looks)')
    _mark(out, tag)
    o = {'1_component_gain': {'label': out['D2']['label'], 'met': d2met, 'robustness': out['D2R']['label'],
                              'scope': D2_SCOPE if d2met else out['D2'].get('statement')},
         '2_strong_references': {'label': out['D3']['label'], 'met': d3met, 'passed': out['D3'].get('passed', []),
                                 'failed': out['D3'].get('failing', [])},
         '3_material_target': {'label': out['D4']['label'], 'met': bool(out['D4'].get('met'))},
         'family_change_vs_start': {'label': out['FC']['label'], 'met': bool(out['FC'].get('met'))},
         'twin_beside': {'label': out['TWIN']['label'], 'met': bool(out['TWIN'].get('met'))},
         'failure_stage': out['failure_stage']['label']}
    o['statement'] = (f"(1) component gain: {o['1_component_gain']['label']} (D2R {o['1_component_gain']['robustness']}); "
                      f"(2) strong references: {o['2_strong_references']['label']} (passed {o['2_strong_references']['passed']}, "
                      f"failed {o['2_strong_references']['failed']}); (3) material target: {o['3_material_target']['label']}; "
                      f"family change vs SUCC (separate): {o['family_change_vs_start']['label']}; TWIN (beside, never sufficient): "
                      f"{o['twin_beside']['label']}; failure stage: {o['failure_stage']}"
                      + (f"; {out['D2R']['headline_caveat']}" if out['D2R']['headline_caveat'] else ''))
    out['outcomes'] = o
    out['scope'] = list(NR.SCOPE) + ['Point feasibility and the reserve are not confidence preservation.',
                                     'A late-card or MODEL_SELECT-triggered revision is never the registered result.',
                                     'The three outcomes (component gain, strong references, material target) are reported '
                                     'separately; none implies another; the family change and TWIN are reported apart.',
                                     'The failure stage and the optimistic diagnostic never gate.']
    return out


def campaign_summary(decision_records):
    """The registered headline (the registered DR reading with the highest revision: the latest card committed before any
    MODEL_SELECT outcome; earlier registered readings are listed as superseded), every non-registered reading with its tag, the
    number K of D2 looks, the readings with D2, D3, D4 met, and the multiplicity statement every campaign-level sentence must
    carry. Flags D2_MET_ONLY_IN_NON_REGISTERED_LOOKS, MULTIPLE_REGISTERED_READINGS."""
    reg, adaptive, looks, met, d3m, d4m = [], [], 0, [], [], []
    for d in decision_records or []:
        lab = _raw(d.get('D2')) or ''
        assessed = lab in D2_ASSESSED or lab.startswith('PARTIAL_')
        looks += assessed
        row = {'family': d.get('family'), 'card': d.get('card'), 'revision': d.get('revision'), 'tag': d.get('adaptive_tag'),
               'D2': lab, 'D3': _raw(d.get('D3')), 'D4': _raw(d.get('D4')), 'FC': _raw(d.get('FC')),
               'stage': _raw(d.get('failure_stage')), 'outcome_class': _raw(d.get('outcome_class')), 'd2_assessed': assessed,
               'outcomes_statement': (d.get('outcomes') or {}).get('statement')}
        (reg if d.get('registered') else adaptive).append(row)
        if (d.get('D2') or {}).get('met'):
            met.append(row)
        if (d.get('D3') or {}).get('met'):
            d3m.append(row)
        if (d.get('D4') or {}).get('met'):
            d4m.append(row)
    revs = [r['revision'] for r in reg]
    if len(set(revs)) != len(revs):
        raise ValueError('two registered readings with the same revision')
    head = max(reg, key=lambda r: r['revision']) if reg else None
    superseded = [r for r in reg if r is not head]
    k0 = int(bool(head and head['d2_assessed']))
    m0 = int(bool(head and head in met))
    flags = (['D2_MET_ONLY_IN_NON_REGISTERED_LOOKS'] if met and not m0 else []) + \
            (['MULTIPLE_REGISTERED_READINGS'] if superseded else [])
    n = len(reg) + len(adaptive)
    return {'registered': head, 'registered_superseded': superseded, 'non_registered': adaptive, 'd2_looks': looks,
            'd2_met': met, 'd3_met': d3m, 'd4_met': d4m, 'flags': flags,
            'statement': (f'D2 was met in {m0} of {k0} registered headline readings and in {len(met) - m0} of {looks - k0} other '
                          f'readings ({looks} D2 looks in total); D3 was met in {len(d3m)} and D4 in {len(d4m)} of {n} readings; '
                          f'{PER_LOOK_FALSE_PASS}.')}


# ---------------------------------------------------------------- (8) D2R
def d2r(rec):
    """D2R robustness to classifier refitting (RULES.md 8; descriptive; never changes D2). rec = {'grid_valid', 'release': 'DR',
    'refit': refit_effectiveness(...) record, 'rob': {comparator: NR grid entry (pair, income, occupation) of DR vs it, registered
    and robustness cells MERGED}, 'twin_id' (optional, default 'TWIN'), 'comparators': {'CTRL': id | None (no bracket) | a
    brackets() record, 'OBJ': id | None}} (the
    registered D2 comparators; never re-chosen on robustness cells). Per decisive comparator: ROBUSTNESS_CONFIRMED iff
    NR.robustness confirms (AGG_POSITIVE on the robustness cells and on all cells), the recipient gates hold on the robustness cells
    (both single views), the registered cells are present and the refit is REFIT_GENUINELY_EFFECTIVE; NOT_RUN if the entry is
    missing; else ROBUSTNESS_NOT_CONFIRMED. Family label over the decisive set. A sign change (robustness-cell against
    registered-cell aggregate mean) or a failed robustness recipient gate on any comparator, decisive or beside (SUCC, TWIN, the
    references present, at least NEURAL_RAW and H_LJ), is flagged and carried into the headline (headline_caveat)."""
    rec = rec or {}
    why = _gate_label(rec.get('grid_valid'))
    if why:
        return {'label': why}
    comp = rec.get('comparators')
    if not isinstance(comp, dict) or any(k not in comp for k in ('CTRL', 'OBJ')):
        return {'label': 'INCOMPLETE', 'reason': 'decisive comparators not recorded (CTRL bracket id, None or a brackets() record; '
                                                 'OBJ id or None)'}
    x = comp['CTRL']
    if isinstance(x, dict):
        if (x.get('CTRL') or x).get('status') == 'INCOMPLETE':
            return {'label': 'INCOMPLETE', 'reason': 'CTRL bracket record INCOMPLETE'}
        x = (x.get('CTRL') or x).get('bracket')
    refit_ok = (rec.get('refit') or {}).get('label') == 'REFIT_GENUINELY_EFFECTIVE'
    decisive = list(dict.fromkeys([y for y in (x, comp.get('OBJ')) if y]))
    rob_all = rec.get('rob') or {}
    beside = [r for r in (START_ID, rec.get('twin_id') or TWIN_ID) + tuple(NR.BENCHMARK) if r not in decisive and r in rob_all]
    beside += [r for r in NR.TARGET_REFS if r not in decisive and r not in beside]
    per, signs, rfail = {}, [], []
    for c in decisive + beside:
        e = rob_all.get(c)
        if not e:
            per[c] = {'label': 'NOT_RUN'}
            continue
        r = NR.robustness(e)
        if r['label'] == 'NOT_RUN':
            per[c] = {'label': 'NOT_RUN'}
            continue
        cells = ((e.get('pair') or {}).get('cells') or {})
        has_reg = all(y in cells for y in NR.REGISTERED)
        regm = NR.aggregate(e.get('pair'), 'registered').get('mean') if has_reg else None
        rob = r['robustness_cells']['aggregate']
        sc = _fin(regm, rob.get('mean')) and _sign(regm) != 0 and _sign(rob['mean']) != 0 and _sign(regm) != _sign(rob['mean'])
        if sc:
            signs.append(c)
        rg = {w: NR.recipient_screen(e.get(w), cells='robustness') for w in NR.SINGLE_VIEWS}
        rg_rec = {w: rg[w]['label'] for w in NR.SINGLE_VIEWS}
        rg_ok = all(rg[w]['gate_pass'] for w in NR.SINGLE_VIEWS)
        rg_missing = any(rg[w]['label'] == 'INCOMPLETE' for w in NR.SINGLE_VIEWS)
        if not rg_ok and not rg_missing:
            rfail.append(c)
        reasons = ([] if r['label'] == 'ROBUSTNESS_CONFIRMED' else ['NOT_AGG_POSITIVE_ON_ROBUSTNESS_OR_ALL_CELLS']) + \
                  ([] if refit_ok else ['REFIT_NOT_GENUINELY_EFFECTIVE']) + ([] if has_reg else ['REGISTERED_CELLS_ABSENT']) + \
                  (['RECIPIENT_ROBUSTNESS_NOT_RECORDED'] if rg_missing else ([] if rg_ok else ['RECIPIENT_GATE_FAILED_ON_ROBUSTNESS']))
        per[c] = {'label': 'ROBUSTNESS_CONFIRMED' if not reasons else 'ROBUSTNESS_NOT_CONFIRMED', 'reasons': reasons,
                  'decisive': c in decisive, 'nrm_label': r['label'], 'registered_mean': regm,
                  'robustness_mean': rob.get('mean'), 'robustness_se': rob.get('se_agg'),
                  'n_eff_robustness_cells': rob.get('n_eff_distinct_cells'), 'reader_seed_qualifier': rob.get('reader_seed_qualifier'),
                  'all_cells_label': r['all_cells']['label'], 'sign_change': sc, 'recipient_robustness': rg_rec}
    labs = [per[c]['label'] for c in decisive]
    fam = ('INCOMPLETE' if not decisive else 'NOT_RUN' if any(y == 'NOT_RUN' for y in labs) else
           'ROBUSTNESS_CONFIRMED' if all(y == 'ROBUSTNESS_CONFIRMED' for y in labs) else 'ROBUSTNESS_NOT_CONFIRMED')
    parts = ([f'the pair benefit changes sign on the robustness cells against {", ".join(signs)}'] if signs else []) + \
            ([f'a single-view recipient condition fails on the robustness cells against {", ".join(rfail)}'] if rfail else [])
    cav = ('; '.join(parts) + ': carried into the headline') if parts else None
    return {'label': fam, 'release': rec.get('release', 'DR'), 'decisive': decisive, 'beside': beside, 'per_comparator': per,
            'sign_changes': signs, 'recipient_failures': rfail, 'headline_caveat': cav,
            'flags': [f'SIGN_CHANGE_{c}' for c in signs] + [f'ROBUSTNESS_RECIPIENT_FAILED_{c}' for c in rfail],
            'refit': (rec.get('refit') or {}).get('label'), 'counted_once': (rec.get('refit') or {}).get('counted_once'),
            'note': 'descriptive; never changes D2; no retuning on the robustness cells (DR, the CTRL bracket, OBJ_MI and every '
                    'selection are the registered ones); repeated reader splits and reader seeds on the same people are not '
                    'replication'}
