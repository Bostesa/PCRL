"""Frozen decision rules of the PCRL reader-fit-aware release campaign (`pcrl_reader_fit_aware_release_v1`, RFA1; statistical
reviewer; written before any outcome of this campaign existed; no label read, no real-data computation).

Human-readable rules: results/pcrl_reader_fit_aware_release_v1/roles/statistical_reviewer/RULES.md
Machine-readable constants: .../RULES.json (schema RFA1-rules-v1); `load_rules()` refuses a RULES.json that disagrees with this
module, a changed NRM1 RULES.json or a changed nrm/rules.py.
Invented-data tests: `python3 -m rfa.test_rules` (prints JSON; exit 0 iff every test passes).

Pure functions on plain dicts / floats. Nothing here reads data or labels (the only files read are this campaign's RULES.json, the
NRM1 RULES.json and, through nrm.rules, the OHN1 reserve rule file). The inherited machinery is `nrm.rules`, imported UNCHANGED as
NR and never monkeypatched: point_eligibility (G1: the 28 original usefulness conditions at the point), reserve_status (the two-SE
reserve heuristic), aggregate / benefit_screen / target_screen (the aggregate paired screen; se_agg = max(sqrt(se_hh^2 + se_rs^2 +
se_mc^2), 0.001)), recipient_screen (the 0.010 recipient gate), comparison, frontier_comparison (the spend-bracketing rule),
robustness and claims (G4 / G5 / G6). Adapted from sar.rules (SAR1) and drf.rules (DRF1), copied, never imported: grid validity
with the extra-trees support and few-valued explanations, refit effectiveness, the reader-budget credit rule, the control-path alias
collapse, the non-table winner's version means (reader targeting), the confidence conjunction, the single-point spend rule (DRF1
4.2), the robustness recipient gate (DRF1 D2R), D5's collapse rules and the registered / adaptive labelling.

RFA1 changes (RULES.md 0.1): one family RFA (START = NEURAL_RAW, frozen) whose CAND is one MODEL_FIT-selected operating point per
version of the reader-fit-aware continuation; D2 gates on START, CONT (zero-privacy continuation) and three matched mechanism
controls in a fixed order: STOPFIT (identical episodes, readers, loss, schedule and lambda, fitted reader parameters detached from
their fitting path; a frontier path from START), ALTCRIT (an ordinary alternating persistent critic of the same family and loss; a
path, a single point, or an exact alias of STOPFIT counted once) and RAWJ (NEURAL_RAW's own online-critic recipe continued to
matched spend; a frontier path from START); a bracket's selection data are MODEL_FIT out-of-fold paired costs only; UM and the
strength machinery are removed; the learning-curve reading (readers fitted on the full FIT and on its fixed half) is new and a
benefit that shrinks at the larger budget is carried into the headline as a limitation; D2R also needs the robustness recipient
gates; five separately reported outcomes (usefulness, gain beyond the mechanism controls, strong references, material target,
joint design and novelty); outcome classes COMPETITIVE_CANDIDATE, MECHANISM_COMPONENT_NOT_COMPETITIVE,
REFERENCE_ADVANTAGE_NOT_ATTRIBUTED and BOUNDED_NEGATIVE_EXPERIMENT (with D2's partial name and the development bounds); reader
grid offset +1200 with offsets 300-1100 forbidden, bootstrap base 2026102520, B = 400 recorded.

Card RF_RFIT_v1 (registration adaptation, RULES.md 0.2): the code family FAM is the rules' family RFA (alias, one family); the F0
mechanism gate (MODEL_FIT, SEX-based, decides only whether the family is fitted) stops a reading as MECHANISM_INERT; ALTCRIT and
RAWJ save only their HI point (LO recorded, not saved: LO_NOT_SAVED_SPEC_BUDGET_<key>); ALTCRIT is a NEAR_ALIAS of STOPFIT when its
median relative critic lag on the deployed fit is below 1% (still gating, never counted as an independent control); the card's
registered falsifiers F0-F6 are reported (F0, F2, F4 from these rules), never changing a label.

API (RULES.md section in brackets):
  load_rules()                                     constants + '_sha256'                                                   [0]
  forbidden_seeds(), registered_seed(v, k), registered_boot_seed(v, rep)                                                   [9]
  grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, boot_B=None)   validity of the RFA1 grid     [9]
  refit_effectiveness(splits, seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, rob_reps=None, boot_B=None)   [8]
  confidence_conjunction(evals)                    PASS / UNRESOLVED / VIOLATION per condition (reported)                  [2]
  control_spec(rec), brackets(costs_vs, paths, kinds=None, alias_of=None, origin='NEURAL_RAW')   the control Q*            [4]
  selection_data_error(sd, kind='PATH'), grid_fineness(spends)                                                             [4]
  d1(rec), d2(rec), learning_curve(rec, comparators)                                                                     [2-5]
  decisions(rec), adaptive_tag(rec), campaign_summary(decision_records), d2r(rec)                                        [6-8]

Record of one reading (`d1`, `d2`, `decisions`):
  {'family': 'RFA' (or its alias 'FAM') | 'FS_<name>', 'card', 'revision': 0 (card RF_RFIT_v1) | 1 (R1),
   'card_committed_before_any_model_select_outcome': bool,
   'grid_valid': grid_validity(...)['effective'] (None = not recorded),
   'start': {'id': 'NEURAL_RAW' (RFA) | a benchmark reference (FS), 'evals': {v: utility}},
   'cand': {'status': 'FITTED' | 'NO_FEASIBLE_POINT' | 'MECHANISM_INERT', 'config_id' (RF_FULL), 'evals',
            'operating_point': {v: ...} (k*), 'selection_data': 'MODEL_FIT_OOF' | 'MODEL_FIT_EPISODES',
            'selection_uses_sex_recovery_diagnostic': bool, 'alias_of_start': bool},
   'f0': {v: 'MECHANISM_ACTIVE' | 'MECHANISM_INERT'} (the F0 gate per version, MODEL_FIT only),
   'points': [{'config_id', 'evals'}] (every fitted operating point saved, CAND included; optional),
   'cont': {'config_id' (RF_TASK), 'evals'} (always gating),
   'paths': {'STOPFIT': [saved ids] | None, 'ALTCRIT': [saved ids] | [one point id] | None, 'RAWJ': [saved ids] | None},
   'control_kind': {'ALTCRIT': 'PATH' | 'POINT' | 'ALIAS_OF_STOPFIT'} (card; STOPFIT and RAWJ are always PATH),
   'altcrit_alias_wire_check': True (only with ALIAS_OF_STOPFIT: identical wires shown for every saved point and version),
   'altcrit_median_relative_lag': float (deployed fit; < 0.01 -> NEAR_ALIAS of STOPFIT), 'altcrit_near_alias': bool (optional;
           must agree with the lag rule), 'lo_not_saved': ['ALTCRIT', 'RAWJ'] (card: HI saved only),
   'falsifier_records': {'F1' | 'F3' | 'F5' | 'F6': 'TRIGGERED' | 'NOT_TRIGGERED'} (coordinator, from the logged diagnostics),
   'selection_data': {'STOPFIT': 'MODEL_FIT_OOF', 'ALTCRIT': 'MODEL_FIT_OOF' (PATH) | CAND's (POINT), 'RAWJ': 'MODEL_FIT_OOF'},
   'controls_not_run': {'CONT' | 'STOPFIT' | 'ALTCRIT' | 'RAWJ': bool} (budget only; never inferred),
   'alias_of': {id: earlier id | 'START' | 'CONT'},
   'grid': {comparator id: NR grid entry of CAND vs it} for START (keyed by its reference name), CONT, every control point, the
           nine references and the D5 comparators,
   'costs_vs': {id: paired costs of CAND vs id ('total|nll' + the 8 secondary metrics)} for START, CONT and every control point,
   'nolookup': {id: {'cells': {cell: non-table winner pair benefit}}}, 'enhanced' (reported only), 'reader_budget',
   'fineness': grid_fineness(...) record, 'control_evals': {id: evals} (reported only), 'joint' (D5), 'novelty' (D6),
   'mc_flags', 'd2r': d2r(...) record (optional),
   'learning_curve': {'grid_valid': bool, 'budgets': {'FULL': int, 'HALF': int}, 'inner_episode_fit_size': int (card),
                      'half': {comparator id: NR grid entry (pair at least) with readers fitted on the fixed half of FIT},
                      'diff_cell_se': {comparator id: {cell: paired-bootstrap SE of FULL - HALF}} (optional)}}

Sign conventions are NR's: pair benefit b = comparator AUC - candidate AUC (positive favours the candidate); recipient difference
r = candidate AUC - comparator AUC (positive = the candidate's single view is more recoverable); paired costs loss(candidate) -
loss(comparator) (negative favours the candidate; accuracy enters as a loss); learning-curve change d = b(FULL FIT) - b(HALF FIT)
(negative = the benefit shrinks when the readers get the larger auxiliary sample). Versions are the ints 0 and 1 ('0' / '1'
accepted).
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

from nrm import rules as NR                      # inherited claims machinery, imported unchanged (never monkeypatched)

RULES_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_reader_fit_aware_release_v1' / 'roles'
              / 'statistical_reviewer' / 'RULES.json')
SCHEMA = 'RFA1-rules-v1'
NRM1_RULES_SHA256 = '84f187768817e79ef9f2bb36151bea81f00b8f62d9179c8cd523069550674652'     # NRM1 RULES.json (registered)
NRM_RULES_PY_SHA256 = 'd5df868678d483a3452e251d8f24575932d23f0e5d22e3ef1ca3627a35030df1'   # nrm/rules.py (registered)
VERSIONS = NR.VERSIONS
Z = 2.0
TIE_TOL = 1e-12
RECIPIENT_LIMIT = NR.RECIPIENT_LIMIT               # 0.010 AUC, per version at the aggregate point (original)
# ---- the registered RFA1 reader grid (READER_SPLITS.json; rfa.devreaders constants)
SEED_OFFSET = 1200                                 # MODEL_SEED_BASE + 1200 + 10 v + k (never used by an earlier grid)
FORBIDDEN_OFFSETS = (300, 400, 500, 600, 700, 800, 900, 1000, 1100)   # OHN1, CTS, TSZ1, NRM1, FRC1, DDR1, IPR1, SAR1, DRF1
BOOT_SEED_BASE = 2026102520                        # + 100 v + repetition index (no seed shared with SAR1's or DRF1's sets)
BOOT_B = 400                                       # household bootstrap replicates of SCORE per cell
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
ALL_REPS = NR.REG_REPS + NR.ROB_REPS
ALL_K = NR.REG_K + NR.ROB_K
SPLIT_TAG = 'RFA1_devgrid_R{r}'
# ---- seed-equality explanations (prospective; inherited from SAR1)
ER_MIN_LEAF = {'ET5': 5, 'ET25': 25}               # min_samples_leaf of the extra-trees recipes
ER_POOL_PEOPLE = 8001                              # MODEL_SELECT people: support bound = 8001 // min leaf (ET5 1600, ET25 320)
FEW_VALUED_MAX = ER_POOL_PEOPLE // ER_MIN_LEAF['ET25']    # 320
# ---- the family and its comparators
START_ID = 'NEURAL_RAW'
STARTS = {'RFA': START_ID}
FAMILY_RE = re.compile(r'(RFA|FAM|FS_[A-Za-z0-9]+)')
FAMILY_ALIASES = {'FAM': 'RFA'}                    # card RF_RFIT_v1's code family FAM is the rules' family RFA (one family)
ALTCRIT_NEAR_ALIAS_LAG = 0.01                      # card section 4: median relative critic lag below 1% -> NEAR_ALIAS of STOPFIT
F0_LABELS = ('MECHANISM_ACTIVE', 'MECHANISM_INERT')
FALSIFIERS = ('F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6')
REGISTERED_FAMILIES = ('RFA',)
PLANNED_REVISIONS = (0, 1)
CONTROL_KEYS = ('STOPFIT', 'ALTCRIT', 'RAWJ')      # the matched mechanism controls, in D2 order
CONTROL_KINDS = {'STOPFIT': ('PATH',), 'ALTCRIT': ('PATH', 'POINT', 'ALIAS_OF_STOPFIT'), 'RAWJ': ('PATH',)}
D2_COMPARISONS = ('START', 'CONT') + CONTROL_KEYS
BRACKET_DATA = ('MODEL_FIT_OOF',)                  # a path's saved points: MODEL_FIT out-of-fold paired costs only
CAND_SELECTION_DATA = ('MODEL_FIT_OOF', 'MODEL_FIT_EPISODES')   # CAND's (and a POINT control's) operating-point data
# ---- decisions
D1_LABELS = ('TASK_FEASIBLE', 'CAND_INFEASIBLE', 'NO_FEASIBLE_POINT', 'NOT_FITTED_MECHANISM_INERT', 'INCOMPLETE')
CONDITION_STATES = ('PASS', 'UNRESOLVED', 'VIOLATION')
CONFIDENCE_LABELS = ('CONFIDENCE_ALL_PASS', 'CONFIDENCE_UNRESOLVED', 'CONFIDENCE_VIOLATION', 'INCOMPLETE')
D2_MET = 'COMPONENT_GAIN_BEYOND_START_AND_MECHANISM_CONTROLS'
D2_ASSESSED = (D2_MET, 'READER_TARGETED', 'NOT_SEPARATED')     # plus PARTIAL_<passed>_ONLY
D2_GATES = ('NOT_ASSESSED_MECHANISM_INERT', 'NOT_ASSESSED_NO_FEASIBLE_POINT', 'NOT_ASSESSED_EXACT_ALIAS', 'NOT_ASSESSED_TASK_INFEASIBLE',
            'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID', 'NOT_ASSESSED_CONTROLS_NOT_RUN', 'INCOMPLETE')
POINT_STATES = ('SPENDS_AT_LEAST_AS_MUCH', 'SPENDS_LESS_UNRESOLVED', 'SPENDS_LESS_RESOLVED')
LC_LABELS = ('BENEFIT_SHRINKS_RESOLVED', 'BENEFIT_SHRINKS_POINT', 'NO_SHRINKAGE_AT_POINT', 'NOT_RUN', 'INCOMPLETE')
LC_FAMILY_LABELS = ('LEARNABILITY_LIMITATION', 'NO_SHRINKAGE_SEEN', 'NOT_RUN', 'INCOMPLETE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN',
                    'NOT_COMPUTED_GRID_INVALID')
OUTCOMES = ('COMPETITIVE_CANDIDATE', 'MECHANISM_COMPONENT_NOT_COMPETITIVE', 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED',
            'BOUNDED_NEGATIVE_EXPERIMENT', 'MECHANISM_INERT', 'NO_OUTPUT_CHANGE', 'TASK_INFEASIBLE', 'SELECTED_POINT_INFEASIBLE',
            'MECHANISM_NOT_ASSESSED', 'INCOMPLETE')
D2R_LABELS = ('ROBUSTNESS_CONFIRMED', 'ROBUSTNESS_NOT_CONFIRMED', 'NOT_RUN', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN',
              'NOT_COMPUTED_GRID_INVALID', 'INCOMPLETE')
ADAPTIVE_KINDS = ('ADAPTIVE', 'REVISED')
D2_SCOPE = ('the component gain is a resolved linked-pair benefit (with both recipient gates) beyond the unchanged start '
            'NEURAL_RAW, its zero-privacy continuation, the stop-fit control (identical episodes, readers, loss, schedule and lambda '
            'with the fitted reader parameters detached from their fitting path) and the alternating persistent critic, each at '
            'matched MODEL_SELECT total-NLL spend where bracketed, and beyond NEURAL_RAW\'s own online-critic recipe continued to '
            'matched spend, for the declared finite-data reader procedure on reused MODEL_SELECT people; it is evidence about this '
            'procedure and auxiliary budget, not an arbitrary-attacker guarantee and not information removal; it is not '
            'superiority over the strong references (D3) and not the material target (D4)')
D2_SCOPE_NEAR_ALIAS = (D2_SCOPE + '; ALTCRIT was a near alias of STOPFIT by the card\'s rule (median relative critic lag '
                       'below 1%): it still gates but is not counted as an independent control')
NO_BRACKET_STATEMENT = ('no {key} point spent as much MODEL_SELECT total NLL as CAND: lower recovery cannot be separated from task '
                        'spending; this is a limit of the control path, never evidence that CAND equals the control or that the '
                        'mechanism has no effect, and the mechanism\'s contribution is not called isolated')
FAILED_EXPLANATION = {
    'STOPFIT': 'without separation from STOPFIT the gain may come from the held-out reader penalty itself or from task spending, '
               'not from differentiating through the reader\'s fit: the fit-aware derivative is not isolated',
    'ALTCRIT': 'without separation from ALTCRIT an ordinary alternating persistent critic of the same family and loss may '
               'reproduce the gain',
    'RAWJ': 'without separation from RAWJ the gain may be explained by running NEURAL_RAW\'s own online-critic recipe further '
            '(spending more of its margin)',
    'CONT': 'without separation from CONT the gain may be explained by continued task training alone',
    'START': 'not lower than the start'}
PER_LOOK_FALSE_PASS = ('one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under six moderate NRM1 '
                       'known laws and about 8% when one version or split dominates (NRM1 RULES.md 5.5); a D2 look is an '
                       'intersection-union conjunction of up to five such comparisons (START, CONT, STOPFIT at its bracket, '
                       'ALTCRIT at its bracket or point unless an exact alias of STOPFIT, RAWJ at its bracket) plus reader '
                       'targeting, so its false-pass rate is at most that of its weakest member; with K D2 looks the chance that '
                       'at least one passes by chance is at most K times the per-look rate; no multiplicity correction is '
                       'applied or claimed')
MDE_NOTE = ('per comparison, 2 se_agg is the resolution threshold (50% power) and about 2.84 se_agg the benefit resolved with '
            'roughly 80% power (normal approximation, version-mean condition ignored); D2 needs every gating comparison plus '
            'targeting, so its power is lower (RULES.md 4.5); "not separated" is the default reading, never "equal"')
BOUND_NOTE = ('development bounds: per gating comparison the aggregate interval mean +- 2 se_agg on the registered cells for the '
              'declared reader procedure on reused MODEL_SELECT people; conditional on the releases fitted; not a population '
              'bound, not an arbitrary-attacker bound and not corrected for the accumulated adaptive search')
LC_NOTE = ('learning curve: readers refitted on the fixed half of each FIT part (same recipes, seeds, VALIDATION selection and SCORE '
           'households); d = b(FULL) - b(HALF); a benefit that shrinks at the larger budget is a possible learnability effect and is '
           'reported as a limitation, never as information removal; no shrinkage between two budgets is not evidence that '
           'information was removed, and budgets above the declared FIT were not tested')

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
        g, er, dec, rs, u = r['reader_grid'], r['er_explanation'], r['decisions'], r['registration_status'], r['usefulness']
        checks = {
            'z': (r['z'], Z), 'z equals NRM1': (Z, NR.Z), 'tie tolerance': (r['tie_tolerance'], TIE_TOL),
            'recipient limit': (r['recipient_limit_auc'], RECIPIENT_LIMIT),
            'recipient limit equals NRM1': (r['recipient_limit_auc'], NR.RECIPIENT_LIMIT),
            'caps': (u['caps'], NR.CAPS), 'guard cap': (u['within_sex_guard_cap'], NR.GUARD_CAP),
            'n conditions': (u['n_conditions'], 28), 'conditions per version': (tuple(u['conditions_per_version']),
                                                                               NR.OR.CONDITIONS),
            'material target': (dec['D4']['threshold_auc'], NR.TARGET), 'se floor': (r['se_floor_auc'], NR.SE_FLOOR),
            'model seed base': (g['model_seed_base'], NR.MODEL_SEED_BASE), 'seed offset': (g['seed_offset'], SEED_OFFSET),
            'forbidden offsets': (tuple(g['forbidden_offsets']), FORBIDDEN_OFFSETS),
            'bootstrap seed base': (g['bootstrap_seed_base'], BOOT_SEED_BASE), 'bootstrap B': (g['bootstrap_B'], BOOT_B),
            'repetition index': (g['repetition_index'], REP_INDEX), 'split tag': (g['split_tag'], SPLIT_TAG),
            'registered cells': (tuple(g['registered_cells']), NR.REGISTERED),
            'er recipes': (er['min_leaf'], ER_MIN_LEAF), 'er pool people': (er['pool_people'], ER_POOL_PEOPLE),
            'few-valued max': (er['few_valued_max'], FEW_VALUED_MAX),
            'starts': (rs['starts'], STARTS), 'families': (rs['family_pattern'], FAMILY_RE.pattern),
            'family aliases': (rs['family_aliases'], FAMILY_ALIASES),
            'near alias lag': (dec['D2']['altcrit_near_alias_lag'], ALTCRIT_NEAR_ALIAS_LAG),
            'f0 labels': (tuple(dec['F0']['labels']), F0_LABELS), 'falsifiers': (tuple(dec['falsifiers']['keys']), FALSIFIERS),
            'registered families': (tuple(rs['registered_families']), REGISTERED_FAMILIES),
            'planned revisions': (tuple(rs['planned_revisions']), PLANNED_REVISIONS),
            'adaptive kinds': (tuple(rs['kinds']), ADAPTIVE_KINDS),
            'd1 labels': (tuple(dec['D1']['labels']), D1_LABELS),
            'confidence labels': (tuple(dec['D1']['confidence_labels']), CONFIDENCE_LABELS),
            'condition states': (tuple(dec['D1']['condition_states']), CONDITION_STATES),
            'cand selection data': (tuple(dec['D1']['cand_selection_data']), CAND_SELECTION_DATA),
            'd2 comparisons': (tuple(dec['D2']['comparisons']), D2_COMPARISONS), 'd2 met': (dec['D2']['met_label'], D2_MET),
            'control kinds': ({k: tuple(x) for k, x in dec['D2']['control_kinds'].items()}, CONTROL_KINDS),
            'bracket data': (tuple(dec['D2']['bracket_selection_data']), BRACKET_DATA),
            'point states': (tuple(dec['D2']['single_point_rule']['states']), POINT_STATES),
            'fineness fraction': (dec['D2']['grid_fineness']['fraction'], FINENESS_FRACTION),
            'lc labels': (tuple(dec['LC']['labels']), LC_LABELS),
            'lc family labels': (tuple(dec['LC']['family_labels']), LC_FAMILY_LABELS),
            'd2r labels': (tuple(dec['D2R']['labels']), D2R_LABELS),
            'outcomes': (tuple(r['outcome_class']['labels']), OUTCOMES),
            'target references': (tuple(dec['D4']['target_references']), NR.TARGET_REFS),
            'benchmark': (tuple(dec['D3']['references']), NR.BENCHMARK),
        }
        bad = [k for k, (a, b) in checks.items() if a != b]
        if bad:
            raise ValueError(f'RULES.json disagrees with rfa.rules on {bad}')
        r['_sha256'] = hashlib.sha256(raw).hexdigest()
        _cache[str(p)] = r
    return _cache[str(p)]


FINENESS_FRACTION = 0.25                           # saved path points: lower >= 0.75 s, upper <= s + 2 SE + 0.25 s (MODEL_FIT OOF)


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


def control_path(ids, alias_of=None, origin=START_ID):
    """A frontier path of D2: its origin (START's id) followed by the distinct saved points (aliases once; an alias of the origin
    collapses into it). None if the path was not given."""
    if ids is None:
        return None
    return _path([origin] + list(ids), alias_of)


def _resolved_negative(entry):
    """True iff the pair benefit of an entry is resolved NEGATIVE (aggregate upper bound < 0 and both version means < 0): the
    comparator is resolved less recoverable than the candidate. None if the entry is incomplete."""
    a = NR.aggregate(entry)
    if not a.get('complete'):
        return None
    if a['alias']:
        return False
    return a['upper'] < 0 and all(a['per_version'][v]['mean'] < 0 for v in VERSIONS)


def _sign(x):
    return (x > 0) - (x < 0)


def _differs_by_version(x):
    if not isinstance(x, dict):
        return False
    return _v(x, 0) != _v(x, 1)


# ---------------------------------------------------------------- (9) reader-grid validity of the registered RFA1 grid
def forbidden_seeds():
    """Every realized earlier grid's reader seeds: MODEL_SEED_BASE + {300, ..., 1100} + 10 v + k, k = 0..9 (180 seeds; a local
    equivalent of nrm.rules.forbidden_seeds, which reads its own constant (300, 500); nrm is not monkeypatched)."""
    return {NR.MODEL_SEED_BASE + off + 10 * v + k for off in FORBIDDEN_OFFSETS for v in VERSIONS for k in range(10)}


def registered_seed(v, k):
    return NR.MODEL_SEED_BASE + SEED_OFFSET + 10 * int(v) + int(k)


def registered_boot_seed(v, rep):
    return BOOT_SEED_BASE + 100 * int(v) + REP_INDEX[rep]


def _seed_errors(seeds, required_k):
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


def _boot_errors(boot_seeds, reps, boot_B):
    """Household-bootstrap seeds {'S<v><R>': int} equal to the registered 2026102520 + 100 v + repetition index for every required
    (version, repetition), no other key form, and the replicate count B recorded and equal to 400."""
    errs = []
    if boot_B is None:
        errs.append(f'bootstrap replicate count not recorded (registered: B = {BOOT_B})')
    elif not isinstance(boot_B, int) or isinstance(boot_B, bool) or boot_B != BOOT_B:
        errs.append(f'bootstrap replicate count {boot_B!r} is not the registered B = {BOOT_B}')
    if boot_seeds is None:
        return errs + [f'bootstrap seeds not recorded (registered: {BOOT_SEED_BASE} + 100 v + repetition index)']
    s = boot_seeds if isinstance(boot_seeds, dict) else {}
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


def _distinct(c, distinct_inputs):
    n = ((distinct_inputs or {}).get(f"{c['wire']}|V{c['v']}") or {}).get(c['view'])
    return n if isinstance(n, int) and not isinstance(n, bool) else None


def _explained_nrm(c, diff, distinct_inputs):
    """NRM1 A-SC1 (iii), as nrm.rules.grid_validity states it: (a) another seed-dependent recipe differs in the same wire, view,
    version and repetition; (b) the same recipe differs on another view of the same wire; (c) the distinct-input count is recorded."""
    a = any((c['wire'], c['view'], c['v'], c['R'], r) in diff for r in NR.SEED_DEPENDENT if r != c['recipe'])
    b = any((c['wire'], w, c['v'], c['R'], c['recipe']) in diff for w in NR.VIEWS if w != c['view'])
    n = ((distinct_inputs or {}).get(f"{c['wire']}|V{c['v']}") or {}).get(c['view'])
    return bool(a and b and isinstance(n, int))


def support_bound(recipe):
    return ER_POOL_PEOPLE // ER_MIN_LEAF[recipe] if recipe in ER_MIN_LEAF else None


def _explained_er(c, diff, distinct_inputs):
    if c.get('recipe') not in ER_MIN_LEAF or not _explained_nrm(c, diff, distinct_inputs):
        return False
    n = _distinct(c, distinct_inputs)
    return n is not None and 0 < n <= support_bound(c['recipe'])


def _explained_fv(c, diff, distinct_inputs):
    """Few-valued explanation (SAR1, prospective): the view's recorded distinct-input count is at most 320 AND the same recipe
    differs (a recorded check with equal_k0_k1 False) on another wire or view in the same version, repetition and seed pair."""
    n = _distinct(c, distinct_inputs)
    if n is None or not 0 < n <= FEW_VALUED_MAX:
        return False
    return any(d[4] == c['recipe'] and d[2] == c['v'] and d[3] == c['R'] and (d[0], d[1]) != (c['wire'], c['view']) for d in diff)


def _group_checks(checks):
    groups, errs = {}, []
    for c in checks or []:
        kp = c.get('k_pair', (0, 1)) if isinstance(c, dict) else None
        if not isinstance(kp, (list, tuple)) or len(kp) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in kp):
            errs.append(f'malformed seed check {c!r}')
            continue
        groups.setdefault(tuple(kp), []).append(c)
    return (groups or {(0, 1): []}), errs


def _recipe_part(seeds, checks, minimum_set, distinct_inputs):
    sd = seeds if isinstance(seeds, dict) else {}
    try:
        base = NR.grid_validity(sd, checks, [], distinct_inputs)
        strict = NR.grid_validity(sd, checks, minimum_set, distinct_inputs)
        eq = [c for c in checks if c.get('recipe') in NR.SEED_DEPENDENT and c.get('equal_k0_k1') is not False]
        diff = _diff_set(checks)
        unexp_nrm = [c for c in eq if not _explained_nrm(c, diff, distinct_inputs)]
    except (KeyError, TypeError, AttributeError) as ex:
        return {'errors': [f'malformed seed checks: {ex!r}'], 'flags': [], 'explained': [], 'unexplained_min': [],
                'strict_errors': ['malformed seed checks']}
    if len(eq) - base['explained_equalities'] != len(unexp_nrm):
        raise AssertionError('RFA1 classification disagrees with nrm.rules.grid_validity')
    mins = set(minimum_set or ())
    explained, unexp_min, unexp_other = [], [], []
    for c in eq:
        on = c['wire'] in mins
        if on and _explained_er(c, diff, distinct_inputs):
            explained.append({**c, 'explanation': 'ER_EXTRA_TREES_SUPPORT', 'minimum_set': True})
        elif (not on) and _explained_nrm(c, diff, distinct_inputs):
            explained.append({**c, 'explanation': 'NRM1_III', 'minimum_set': False})
        elif _explained_fv(c, diff, distinct_inputs):
            explained.append({**c, 'explanation': 'FEW_VALUED', 'minimum_set': on})
        elif on:
            unexp_min.append(c)
        else:
            unexp_other.append(c)
    errors = [e for e in base['errors'] if not e.startswith('reader seeds') and 'unexplained seed-dependent' not in e]
    if unexp_min:
        errors.append(f'{len(unexp_min)} unexplained seed-dependent equalities on the minimum set')
    if unexp_other:
        errors.append(f'{len(unexp_other)} unexplained seed-dependent equalities')
    return {'errors': errors, 'flags': base.get('flags', []), 'explained': explained, 'unexplained_min': unexp_min,
            'unexplained_other': len(unexp_other), 'checks': len(checks),
            'strict_errors': [e for e in strict['errors'] if not e.startswith('reader seeds')]}


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
                    'minimum_set': c['minimum_set'], 'support_bound': support_bound(c['recipe']), 'few_valued_max': FEW_VALUED_MAX}
    return out


def grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, required_k=NR.REG_K, reps=NR.REG_REPS,
                  boot_B=None):
    """Reader-seed validity of the registered RFA1 grid, applied prospectively (RULES.md 9). READER_SEEDS_EFFECTIVE iff
    (1) seeds: the required S<v>k<k> recorded, distinct integers, every one equal to MODEL_SEED_BASE + 1200 + 10 v + k, none an
        earlier realized grid's seed (offsets 300, 400, ..., 1100);
    (2) bootstrap: B recorded and equal to 400; S<v><R> recorded for every version and required repetition and equal to
        2026102520 + 100 v + repetition index;
    (3) per seed pair (k0 / k1; for the refit also k2 / k3): every seed-dependent recipe checked; every seed-dependent equality
        explained: off the minimum set by NRM1 A-SC1 (iii) or the few-valued explanation; on the minimum set (a D2 reading's CAND,
        START, CONT and the saved STOPFIT, ALTCRIT and RAWJ points; for the refit the D2R release and its decisive comparators) by
        the extra-trees support explanation or the few-valued explanation. Explained equalities count once (n_eff) and are
        reported. The strict NRM1 reading (any minimum-set equality invalidates) is reported beside as 'strict_nrm1'."""
    seed_errs = _seed_errors(seeds, required_k) + _boot_errors(boot_seeds, reps, boot_B)
    errors = list(seed_errs)
    groups, gerrs = _group_checks(checks)
    errors += gerrs
    per, exp_all, strict_errors, flags = {}, [], [], set()
    for kp, ch in sorted(groups.items()):
        part = _recipe_part(seeds, ch, minimum_set, distinct_inputs)
        tag = f'k{kp[0]}k{kp[1]}'
        errors += [f'{tag}: {e}' for e in part['errors']]
        strict_errors += [f'{tag}: {e}' for e in part['strict_errors']]
        flags.update(part['flags'])
        exp_all += [{**c, 'k_pair': list(kp)} for c in part['explained']]
        per[tag] = {'checks': part.get('checks', 0), 'errors': part['errors'], 'explained': len(part['explained']),
                    'explained_few_valued': sum(c['explanation'] == 'FEW_VALUED' for c in part['explained']),
                    'unexplained_minimum_set': len(part['unexplained_min']), 'unexplained_other': part.get('unexplained_other')}
    ok = not errors
    if any(c['explanation'] == 'FEW_VALUED' for c in exp_all):
        flags.add('FEW_VALUED_SEED_EQUALITIES_COUNTED_ONCE')
    return {'label': 'READER_SEEDS_EFFECTIVE' if ok else 'READER_SEEDS_NOT_EFFECTIVE', 'effective': ok, 'errors': errors,
            'per_seed_pair': per, 'flags': sorted(flags),
            'explained_equalities': {
                'count': len(exp_all), 'n_eff_implied': _n_eff_report(exp_all, [c for c in checks or [] if isinstance(c, dict)],
                                                                      distinct_inputs),
                'items': [{k: c.get(k) for k in ('wire', 'view', 'v', 'R', 'recipe', 'k_pair', 'explanation', 'minimum_set')}
                          for c in exp_all]},
            'strict_nrm1': {'label': 'READER_SEEDS_NOT_EFFECTIVE' if (seed_errs or strict_errors) else 'READER_SEEDS_EFFECTIVE',
                            'errors': seed_errs + strict_errors},
            'seed_offset': SEED_OFFSET, 'forbidden_offsets': list(FORBIDDEN_OFFSETS), 'required_k': list(required_k),
            'reps': list(reps), 'bootstrap_B': BOOT_B, 'few_valued_max': FEW_VALUED_MAX,
            'rule': 'RFA1 prospective: NRM1 (iii) off the minimum set, the extra-trees support explanation on it, and the '
                    'few-valued explanation everywhere'}


# ---------------------------------------------------------------- (8) genuinely effective classifier refit
def refit_effectiveness(splits, seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, rob_reps=None, boot_B=None):
    """REFIT_GENUINELY_EFFECTIVE iff (a) every robustness repetition run (at least one of R3 / R4) uses a different household split
    (assignment hashes recorded and distinct from each other and from R1 / R2; every SCORE overlap involving it recorded, finite and
    in [0, 1)) and (b) grid_validity holds on all four reader seeds and every repetition run. Otherwise
    REFIT_NOT_GENUINELY_EFFECTIVE and the failed part's cells are counted once. Flag ONE_ROBUSTNESS_SPLIT_ONLY."""
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
    gv = grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=boot_seeds, required_k=ALL_K, reps=reps,
                       boot_B=boot_B)
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
    """Each of the 28 conditions from its own household-clustered SE (sqrt(se^2 + se_mc^2) for a randomized release; no floor):
    PASS iff value + z se <= cap; VIOLATION iff value - z se > cap; UNRESOLVED otherwise. CONFIDENCE_ALL_PASS iff all 28 pass;
    CONFIDENCE_VIOLATION if any violation; else CONFIDENCE_UNRESOLVED; INCOMPLETE if a value or SE is missing. Reported beside
    G1: never gating; 28 one-sided per-condition bounds, not a simultaneous statement (no multiplicity correction)."""
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


def _cand_selection_error(cand):
    """None if CAND's operating point was selected on a declared MODEL_FIT source (MODEL_FIT_OOF or MODEL_FIT_EPISODES); else a
    message (not recorded, or any other source: never MODEL_SELECT)."""
    sd = (cand or {}).get('selection_data')
    if sd is None:
        return 'CAND_SELECTION_DATA_NOT_RECORDED'
    return None if sd in CAND_SELECTION_DATA else 'CAND_SELECTION_DATA_NOT_ALLOWED'


def _f0_state(rec):
    """(record, flags, error) of the F0 mechanism gate (card RF_RFIT_v1 5: MODEL_FIT only, SEX-based; it decides whether the family
    is fitted and never selects a point). An inert version saves nothing, so a reading with any inert version is MECHANISM_INERT;
    a FITTED reading with an inert version, or a MECHANISM_INERT status without an inert version, is a record error."""
    f0 = (rec or {}).get('f0')
    st = ((rec or {}).get('cand') or {}).get('status')
    per = {v: _v(f0, v) for v in VERSIONS} if isinstance(f0, dict) else {v: None for v in VERSIONS}
    inert = [v for v in VERSIONS if per[v] == 'MECHANISM_INERT']
    flags, err = [], None
    if f0 is None:
        flags.append('F0_NOT_RECORDED')
    else:
        flags.append('F0_SEX_BASED_FIT_GATE_DISCLOSED')
        if any(per[v] not in F0_LABELS for v in VERSIONS) and st != 'MECHANISM_INERT':
            flags.append('F0_LABEL_MISSING_OR_UNKNOWN')
    if len(inert) == 1:
        flags.append('MECHANISM_INERT_IN_ONE_VERSION_ONLY')
    if st == 'FITTED' and inert:
        err = 'F0 read MECHANISM_INERT in a version, which saves nothing: the reading cannot be FITTED'
    if st == 'MECHANISM_INERT' and not inert:
        err = 'status MECHANISM_INERT needs the F0 record of the inert version'
    return {'per_version': per, 'inert_versions': inert}, flags, err


def d1(rec):
    """D1 task feasibility at CAND (RULES.md 2). Gating label from G1 (NR.point_eligibility: all 28 at the point, original caps):
    TASK_FEASIBLE, CAND_INFEASIBLE, NO_FEASIBLE_POINT (no operating point met the card's MODEL_FIT rule: no CAND), INCOMPLETE (also
    when CAND's selection data are not recorded or not a MODEL_FIT source). Reported separately and never gating: the confidence
    conjunction and the inherited two-SE reserve of CAND, START and CONT; each recorded operating point's point label; flags
    START_POINT_INELIGIBLE, CONT_POINT_INELIGIBLE, CAND_SELECTED_ON_MODEL_FIT_SEX_RECOVERY_DIAGNOSTIC (disclosure),
    F0_SEX_BASED_FIT_GATE_DISCLOSED / F0_NOT_RECORDED; NOT_FITTED_MECHANISM_INERT when the F0 gate stopped the family."""
    rec = rec or {}
    cand, start, cont = rec.get('cand') or {}, rec.get('start') or {}, rec.get('cont') or {}
    flags = []
    sl = NR.point_eligibility(start.get('evals') or {})['label'] if start.get('evals') else None
    cl = NR.point_eligibility(cont.get('evals') or {})['label'] if cont.get('evals') else None
    if sl == 'POINT_INELIGIBLE':
        flags.append('START_POINT_INELIGIBLE')
    if cl == 'POINT_INELIGIBLE':
        flags.append('CONT_POINT_INELIGIBLE')
    f0, f0flags, f0err = _f0_state(rec)
    flags += f0flags
    out = {'start': sl, 'cont': cl, 'cand': None, 'flags': flags, 'f0': f0}
    if f0err:
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'error': f0err}
    if cand.get('status') == 'MECHANISM_INERT':
        return {**out, 'label': 'NOT_FITTED_MECHANISM_INERT', 'met': False,
                'statement': 'the F0 gate read MECHANISM_INERT (MODEL_FIT only): the family was not fitted in that version and nothing '
                             'was saved; a design failure on these data, not a usefulness or protection result'}
    if cand.get('status') == 'NO_FEASIBLE_POINT':
        return {**out, 'label': 'NO_FEASIBLE_POINT', 'met': False,
                'statement': 'no operating point met the card\'s MODEL_FIT selection rule: no candidate exists (no fallback)'}
    if cand.get('status') != 'FITTED':
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'error': 'CAND status missing'}
    se = _cand_selection_error(cand)
    if se:
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'flags': flags + [se],
                'error': 'CAND\'s operating point must be selected on MODEL_FIT (MODEL_FIT_OOF or MODEL_FIT_EPISODES), never on '
                         'MODEL_SELECT, and the source recorded'}
    if cand.get('selection_uses_sex_recovery_diagnostic') is True:
        flags.append('CAND_SELECTED_ON_MODEL_FIT_SEX_RECOVERY_DIAGNOSTIC')
    cg = NR.point_eligibility(cand.get('evals') or {})
    out['cand'] = cg['label']
    if cg['label'] == 'INCOMPLETE':
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'error': cg.get('error')}
    lab = 'CAND_INFEASIBLE' if cg['label'] == 'POINT_INELIGIBLE' else 'TASK_FEASIBLE'
    pts = None
    if rec.get('points') is not None:
        pts = {p.get('config_id'): NR.point_eligibility(p.get('evals') or {})['label'] for p in rec.get('points') or []}
    conf = {'cand': confidence_conjunction(cand.get('evals')),
            'start': confidence_conjunction(start.get('evals')) if start.get('evals') else None,
            'cont': confidence_conjunction(cont.get('evals')) if cont.get('evals') else None}
    res = {'cand': _reserve(cand.get('evals')), 'start': _reserve(start.get('evals')) if start.get('evals') else None,
           'cont': _reserve(cont.get('evals')) if cont.get('evals') else None}
    if lab == 'TASK_FEASIBLE':
        if conf['cand'].get('label') != 'CONFIDENCE_ALL_PASS':
            flags.append('POINT_FEASIBLE_NOT_CONFIDENCE_RESOLVED')
        if not res['cand'].get('qualified'):
            flags.append('POINT_FEASIBLE_NOT_RESERVE_QUALIFIED')
    out.update({'label': lab, 'met': lab == 'TASK_FEASIBLE', 'cand_failed': cg.get('failed'),
                'confidence': {k: (None if x is None else {kk: x.get(kk) for kk in ('label', 'unresolved', 'violations', 'n_pass')})
                               for k, x in conf.items()},
                'confidence_detail_cand': conf['cand'], 'reserve': res, 'points': pts, 'flags': flags,
                'note': 'three separate usefulness results: G1 at the point (gating), the confidence conjunction and the two-SE '
                        'reserve (both reported, never gating, never redefining eligibility, never removing a reference)'})
    return out


# ---------------------------------------------------------------- (4) controls, brackets, fineness
def control_spec(rec):
    """({key: {'kind', 'ids'}}, error) for STOPFIT, ALTCRIT, RAWJ from rec['paths'] and rec['control_kind'] (card). STOPFIT and
    RAWJ are PATH; ALTCRIT is PATH, POINT (exactly one saved id) or ALIAS_OF_STOPFIT (requires altcrit_alias_wire_check True: the
    exact alias must be shown by identical wires, never asserted)."""
    paths = (rec or {}).get('paths') or {}
    kinds = (rec or {}).get('control_kind') or {}
    out = {}
    for k in CONTROL_KEYS:
        kind = kinds.get(k, CONTROL_KINDS[k][0])
        if kind not in CONTROL_KINDS[k]:
            return None, f'{k}: kind {kind!r} not allowed (allowed: {list(CONTROL_KINDS[k])})'
        ids = paths.get(k)
        if ids is not None and (not isinstance(ids, (list, tuple)) or not all(isinstance(i, str) and i for i in ids)):
            return None, f'{k}: saved ids must be a list of names'
        if kind == 'POINT' and ids is not None and len(ids) != 1:
            return None, f'{k}: a POINT control names exactly one saved point'
        if kind == 'ALIAS_OF_STOPFIT':
            if (rec or {}).get('altcrit_alias_wire_check') is not True:
                return None, 'ALTCRIT declared an exact alias of STOPFIT without a recorded wire check (identical wires per saved point and version)'
            ids = None
        out[k] = {'kind': kind, 'ids': None if ids is None else list(ids)}
    return out, None


def brackets(costs_vs, paths, kinds=None, alias_of=None, origin=START_ID):
    """Q* of each PATH control by NR's rule (the cheapest path point whose pooled MODEL_SELECT paired total-NLL cost is >= the
    candidate's, i.e. d = cost(CAND) - cost(point) <= 0; ties keep path order). Path = origin (START) + the distinct saved points.
    A POINT control's comparator is its point (status POINT); an ALIAS_OF_STOPFIT control has none. Per key: {'bracket': id | None,
    'status': BRACKETED | NO_BRACKET | POINT | ALIAS_OF_STOPFIT | PATH_NOT_GIVEN | INCOMPLETE, 'path', 'primary'}."""
    out = {}
    for key in CONTROL_KEYS:
        kind = (kinds or {}).get(key, CONTROL_KINDS[key][0])
        if kind == 'ALIAS_OF_STOPFIT':
            out[key] = {'bracket': None, 'status': 'ALIAS_OF_STOPFIT'}
            continue
        ids = (paths or {}).get(key)
        if kind == 'POINT':
            out[key] = ({'bracket': None, 'status': 'PATH_NOT_GIVEN'} if not ids else
                        {'bracket': (alias_of or {}).get(ids[0], ids[0]), 'status': 'POINT', 'path': list(ids)})
            continue
        pth = control_path(ids, alias_of, origin)
        if pth is None:
            out[key] = {'bracket': None, 'status': 'PATH_NOT_GIVEN'}
            continue
        prim, err = {}, None
        for i in pth:
            d, _ = NR._primary((costs_vs or {}).get(i))
            if d is None:
                err = f'paired costs vs {i} missing or malformed'
                break
            prim[i] = d['value']
        if err:
            out[key] = {'bracket': None, 'status': 'INCOMPLETE', 'error': err, 'path': pth}
            continue
        above = sorted((i for i in pth if prim[i] <= 0), key=lambda i: -prim[i])
        out[key] = {'bracket': above[0] if above else None, 'status': 'BRACKETED' if above else 'NO_BRACKET', 'path': pth,
                    'primary': prim}
    return out


def selection_data_error(sd, kind='PATH'):
    """None if a control's declared selection data are allowed: a PATH's saved points on MODEL_FIT out-of-fold paired costs only;
    a POINT control's operating point on CAND's sources (MODEL_FIT_OOF or MODEL_FIT_EPISODES); never MODEL_SELECT or CALIBRATION."""
    if sd is None:
        return 'NOT_RECORDED'
    allowed = BRACKET_DATA if kind == 'PATH' else CAND_SELECTION_DATA
    return None if sd in allowed else 'NOT_ALLOWED'


def grid_fineness(spends):
    """Fineness of the two saved points of each PATH control on MODEL_FIT out-of-fold paired costs (RULES.md 4.4; reported beside
    D2, never gating; computed before any MODEL_SELECT read). spends = {key: {'selection_data': 'MODEL_FIT_OOF', v: {'cand': s =
    CAND's paired total-NLL spend over START, 'lower', 'upper', 'upper_se': household-clustered SE (same data) of the paired
    difference upper - CAND, 'max': the largest path spend, optionally 'grid': [{'spend', 'se'}] every path point}}}. Per path and
    version: SELECTION_DATA_NOT_ALLOWED; INCOMPLETE; NOT_APPLICABLE_CAND_NOT_ABOVE_START (s <= 0); CONTROL_PATH_MAY_NOT_BRACKET
    (max < s + 2 SE); SAVED_POINT_RULE_NOT_FOLLOWED; FINE iff lower >= 0.75 s and upper <= s + 2 SE + 0.25 s; else COARSE_BELOW /
    COARSE_ABOVE / COARSE_BOTH; NOT_GIVEN."""
    out, flags, data = {}, [], {}
    for key in CONTROL_KEYS:
        src = (spends or {}).get(key)
        if src is None:
            out[key] = 'NOT_GIVEN'
            continue
        sd = src.get('selection_data') if isinstance(src, dict) else None
        data[key] = sd
        sde = selection_data_error(sd, 'PATH')
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
                per[v] = 'NOT_APPLICABLE_CAND_NOT_ABOVE_START'
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
            'rule': 'on MODEL_FIT out-of-fold paired costs with the SE from the same data: lower saved point >= 0.75 s and upper '
                    'saved point <= s + 2 SE + 0.25 s (s = CAND spend over START > 0); reported, never gating'}


# ---------------------------------------------------------------- (4) D2
def _secondary(costs):
    """Every NR.SECONDARY paired metric pooled (NR._pooled) with resolved_more_spending = pooled - 2 SE > 0; None if any missing."""
    sec = {}
    for m in NR.SECONDARY:
        per = {v: (_v(costs, v) or {}).get(m) for v in VERSIONS}
        p = NR._pooled(per) if all(isinstance(x, dict) for x in per.values()) else None
        if p is None:
            return None
        sec[m] = {**p, 'resolved_more_spending': p['value'] - Z * p['se'] > 0}
    return sec


def _frontier_part(key, ids, point, grid, gate_of, origin_id, alias_of):
    """One PATH control comparison of D2 (key STOPFIT, ALTCRIT or RAWJ): (record, error, flags). Conjuncts: (c1)
    NR.frontier_comparison = DOMINATES_CONTROL_FRONTIER; (c2) the recipient gate vs Q*; (c3) no cheaper path point resolved less
    recoverable than CAND. No bracket -> NOT_SEPARATED, reason NO_BRACKET."""
    pth = control_path(ids, alias_of, origin_id)
    flags = [f'{key}_AT_ORIGIN'] if pth == [origin_id] else []
    fr = NR.frontier_comparison(point, [{'id': i} for i in pth])
    if fr['label'] == 'INCOMPLETE':
        return None, f'{key}: {fr.get("error")}', flags
    q = fr.get('bracketing_control')
    gate, rcp = None, None
    if q is not None:
        gate, cq = gate_of(q)
        if gate is None:
            return None, f'{key}: recipient views vs bracketing point {q} missing or incomplete', flags
        rcp = {w: cq['recipient'][w]['label'] for w in NR.SINGLE_VIEWS}
        if cq['benefit']['label'] == 'ALIAS_ZERO':
            flags.append(f'CAND_ALIAS_OF_{key}_{q}')
    prim = fr.get('primary_cost_differences') or {}
    dominated = []
    for i in pth:
        pv = (prim.get(i) or {}).get('value')
        if _fin(pv) and pv > 0:                       # the point is cheaper than CAND
            neg = _resolved_negative((grid.get(i) or {}).get('pair'))
            if neg is None:
                return None, f'{key}: grid vs cheaper control point {i} missing', flags
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
    over = {i: prim[i]['value'] for i in pth if _fin((prim.get(i) or {}).get('value')) and prim[i]['value'] > 0}
    mismatch = {'at_bracket': ({'value': prim[q]['value'], 'se': prim[q]['se'], 'by_version': prim[q].get('by_version'),
                                'note': 'cost(CAND) - cost(Q*) <= 0: Q* spends at least this much more total NLL than CAND'}
                               if q is not None else None),
                'smallest_overspend_vs_cheaper_point': (min(over.values()) if over else None),
                'interpolation': fr.get('interpolation')}
    return ({'label': label, 'kind': 'PATH', 'frontier_label': fr['label'], 'reason': reason, 'path': pth, 'point': q,
             'recipient_gate_vs_bracket': gate, 'recipient_vs_bracket': rcp, 'dominated_by_cheaper': dominated, 'pass': ok,
             'gating': True, 'benefit_mean': fr.get('benefit_mean'), 'benefit_se': fr.get('benefit_se'),
             'secondary_more_spending': fr.get('secondary_more_spending'), 'residual_spend_mismatch': mismatch,
             'statement': NO_BRACKET_STATEMENT.format(key=key) if no_bracket else None}, None, flags)


def _point_part(key, pid, point, gate_of):
    """A single-point control (DRF1 4.2): d = pooled MODEL_SELECT paired total-NLL cost(CAND) - cost(point) (NR._primary).
    SPENDS_AT_LEAST_AS_MUCH (d <= 0); SPENDS_LESS_UNRESOLVED (0 < d, d - 2 SE <= 0: credited, flagged); SPENDS_LESS_RESOLVED
    (d - 2 SE > 0: NOT_SEPARATED, reason NOT_SEPARATED_FROM_SPEND; the benefit is reported as passed_if_spend_matched). Otherwise pass
    iff AGG_POSITIVE pair benefit, both recipient gates and no secondary metric resolved higher than the point."""
    d, _ = NR._primary((point.get('costs_vs') or {}).get(pid))
    if d is None:
        return None, f'{key}: paired costs vs {pid} missing or malformed', []
    sec = _secondary((point.get('costs_vs') or {}).get(pid))
    if sec is None:
        return None, f'{key}: a secondary paired cost vs {pid} is missing or non-finite', []
    gate, cq = gate_of(pid)
    if gate is None:
        return None, f'{key}: grid vs {pid} missing or incomplete', []
    flags = []
    state = ('SPENDS_AT_LEAST_AS_MUCH' if d['value'] <= 0 else
             'SPENDS_LESS_UNRESOLVED' if d['value'] - Z * d['se'] <= 0 else 'SPENDS_LESS_RESOLVED')
    if state == 'SPENDS_LESS_UNRESOLVED':
        flags.append(f'{key}_SPENDS_LESS_UNRESOLVED')
    if cq['benefit']['label'] == 'ALIAS_ZERO':
        flags.append(f'CAND_ALIAS_OF_{key}_{pid}')
    more = sorted(m for m, x in sec.items() if x['resolved_more_spending'])
    matched_pass = bool(cq['benefit_positive'] and gate and not more)
    if state == 'SPENDS_LESS_RESOLVED':
        label, reason = 'NOT_SEPARATED', 'NOT_SEPARATED_FROM_SPEND'
    elif not cq['benefit_positive']:
        label, reason = 'NOT_SEPARATED', 'NO_RESOLVED_BENEFIT_AT_POINT'
    elif not gate:
        label, reason = 'NOT_SEPARATED', 'RECIPIENT_GATE_FAILED_AT_POINT'
        flags.append(f'RECIPIENT_GATE_FAILED_VS_{pid}')
    elif more:
        label, reason = 'SEPARATED_ON_PRIMARY_COST_ONLY', 'SECONDARY_METRIC_RESOLVED_HIGHER'
    else:
        label, reason = 'DOMINATES_CONTROL_POINT', None
    ok = label == 'DOMINATES_CONTROL_POINT'
    return ({'label': label, 'kind': 'POINT', 'reason': reason, 'point': pid, 'spend_state': state,
             'spend_difference': {'value': d['value'], 'se': d['se'], 'by_version': d.get('by_version')},
             'recipient_gate_vs_point': gate, 'recipient_vs_point': {w: cq['recipient'][w]['label'] for w in NR.SINGLE_VIEWS},
             'secondary_more_spending': more, 'pass': ok, 'passed_if_spend_matched': matched_pass, 'gating': True,
             'benefit_mean': cq['benefit']['aggregate'].get('mean'), 'benefit_se': cq['benefit']['aggregate'].get('se_agg')},
            None, flags)


def _start_of(rec):
    """(START id, error): RFA -> NEURAL_RAW (frozen); FS_<name> -> a declared benchmark reference."""
    fam = _family(rec)
    sid = ((rec or {}).get('start') or {}).get('id')
    want = STARTS.get(fam)
    if want is not None and sid != want:
        return None, f'START of {fam} must be {want} (frozen), got {sid!r}'
    if want is None and sid not in NR.BENCHMARK:
        return None, f'START of {fam} must be a frozen benchmark reference, got {sid!r}'
    return sid, None


def _cand_gate(rec):
    """Shared gate of D2: (label, extra) or (None, g1 record)."""
    cand = rec.get('cand') or {}
    st = cand.get('status')
    _, _, f0err = _f0_state(rec)
    if f0err:
        return 'INCOMPLETE', {'error': f0err}
    if st == 'MECHANISM_INERT':
        return 'NOT_ASSESSED_MECHANISM_INERT', {'statement': 'the F0 gate stopped the family (MECHANISM_INERT): no protection '
                                                             'experiment was run'}
    if st == 'NO_FEASIBLE_POINT':
        return 'NOT_ASSESSED_NO_FEASIBLE_POINT', {}
    if cand.get('alias_of_start') is True:
        return 'NOT_ASSESSED_EXACT_ALIAS', {'statement': 'CAND transmits exactly START\'s vectors: the operation never changed the '
                                                         'output; no protection experiment'}
    if st != 'FITTED':
        return 'INCOMPLETE', {'error': 'CAND status missing'}
    se = _cand_selection_error(cand)
    if se:
        return 'INCOMPLETE', {'error': se, 'flags': [se]}
    g1 = NR.point_eligibility(cand.get('evals') or {})
    if g1['label'] == 'INCOMPLETE':
        return 'INCOMPLETE', {'error': f'CAND utility incomplete: {g1.get("error")}'}
    if g1['label'] != 'POINT_ELIGIBLE':
        return 'NOT_ASSESSED_TASK_INFEASIBLE', {'g1_failed': g1.get('failed'),
                                                'statement': 'the candidate fails G1: the comparison is reported as not run (never '
                                                             'negative, never protection)'}
    return None, g1


def _alias_map(rec, sid, cid):
    return {k: (sid if x == 'START' else (cid if x == 'CONT' else x)) for k, x in (rec.get('alias_of') or {}).items()}


def _bounds(comps, gating):
    out = {}
    for n in gating:
        x = comps.get(n) or {}
        m, s = x.get('benefit_mean'), x.get('benefit_se')
        out[n] = ({'at': x.get('point'), 'mean': m, 'se_agg': s, 'lower': m - Z * s, 'upper': m + Z * s}
                  if _fin(m, s) else {'at': x.get('point'), 'mean': None, 'note': x.get('reason') or 'not bounded'})
    return {'per_comparison': out, 'note': BOUND_NOTE}


def d2(rec):
    """D2 component gain at CAND (RULES.md 4). Labels in order of precedence: NOT_ASSESSED_NO_FEASIBLE_POINT;
    NOT_ASSESSED_EXACT_ALIAS (CAND = START); INCOMPLETE (CAND status, selection data or utility); NOT_ASSESSED_TASK_INFEASIBLE;
    NOT_COMPUTED_GRID_VALIDITY_UNKNOWN / NOT_COMPUTED_GRID_INVALID; INCOMPLETE (START not NEURAL_RAW for RFA; control kinds or saved
    ids malformed; an exact-alias ALTCRIT without a wire check); NOT_ASSESSED_CONTROLS_NOT_RUN (a gating control recorded as not run
    for budget); INCOMPLETE (a gating control's selection data not recorded or not allowed; any needed input missing); then the
    conjuncts: (a) G1 (above); (b) START and (b') CONT: NR.comparison passes (AGG_POSITIVE pair benefit + both recipient gates);
    (c) STOPFIT, ALTCRIT, RAWJ: a PATH by the frontier conjuncts (c1)-(c3) at its Q*, a POINT by the single-point spend rule, an
    exact alias of STOPFIT counted once (not a second comparison); (d) reader targeting: the non-table winner's point pair benefit
    > 0 in both version means vs START, CONT and every gating Q* / point. Result: COMPONENT_GAIN_BEYOND_START_AND_MECHANISM_CONTROLS
    (all gating comparisons and d); READER_TARGETED (all gating comparisons, d fails); NOT_SEPARATED (none passes); else
    PARTIAL_<passed>_ONLY (order START, CONT, STOPFIT, ALTCRIT, RAWJ). No partial outcome is a claim."""
    rec = rec or {}
    lab, extra = _cand_gate(rec)
    if lab:
        return {'label': lab, 'met': False, **extra}
    cand = rec.get('cand') or {}
    why = _gate_label(rec.get('grid_valid'))
    if why:
        return {'label': why, 'met': False}
    sid, err = _start_of(rec)
    if err:
        return {'label': 'INCOMPLETE', 'met': False, 'error': err}
    spec, err = control_spec(rec)
    if err:
        return {'label': 'INCOMPLETE', 'met': False, 'error': err}
    lns = rec.get('lo_not_saved') or []
    if not isinstance(lns, (list, tuple)) or any(k not in CONTROL_KEYS or spec[k]['kind'] != 'PATH' for k in lns) or \
            any(spec[k]['ids'] is not None and len(spec[k]['ids']) > 1 for k in lns):
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'lo_not_saved must list PATH controls that save at most one point (HI)'}
    lag = rec.get('altcrit_median_relative_lag')
    near = spec['ALTCRIT']['kind'] != 'ALIAS_OF_STOPFIT' and _fin(lag) and lag < ALTCRIT_NEAR_ALIAS_LAG
    decl = rec.get('altcrit_near_alias')
    if decl is not None and (not isinstance(decl, bool) or (_fin(lag) and decl != near)):
        return {'label': 'INCOMPLETE', 'met': False,
                'error': 'altcrit_near_alias disagrees with the card\'s lag rule (median relative critic lag below 1%)'}
    cid = (rec.get('cont') or {}).get('config_id') or 'CONT'
    grid, nr = rec.get('grid') or {}, rec.get('controls_not_run') or {}
    keys = [k for k in CONTROL_KEYS if spec[k]['kind'] != 'ALIAS_OF_STOPFIT']
    gating = ['START', 'CONT'] + keys
    not_run = ((['CONT'] if cid not in grid and nr.get('CONT') is True else []) +
               [k for k in keys if spec[k]['ids'] is None and nr.get(k) is True])
    if not_run:
        return {'label': 'NOT_ASSESSED_CONTROLS_NOT_RUN', 'met': False, 'not_run': not_run,
                'statement': 'a matched control was not run (budget): the mechanism comparison is reported as not run'}
    flags, comps = [], {}

    def incomplete(msg):
        return {'label': 'INCOMPLETE', 'met': False, 'error': msg, 'comparisons': comps, 'flags': sorted(set(flags))}
    sel = rec.get('selection_data') or {}
    for k in keys:
        e = selection_data_error(sel.get(k), spec[k]['kind'])
        if e == 'NOT_RECORDED':
            return incomplete(f'the {k} control\'s selection data are not recorded (plan)')
        if e == 'NOT_ALLOWED':
            flags.append(f'CONTROL_SELECTION_DATA_NOT_ALLOWED_{k}')
            return incomplete(f'the {k} control was selected on {sel.get(k)!r}: a path\'s saved points on MODEL_FIT out-of-fold '
                              'paired costs only, a single point on CAND\'s MODEL_FIT sources; never MODEL_SELECT (a re-selection '
                              'is a POST HOC repair)')

    def gate_of(q):
        c = NR.comparison(grid.get(q) or {})
        if c['benefit']['label'] == 'INCOMPLETE' or any(c['recipient'][w]['label'] == 'INCOMPLETE' for w in NR.SINGLE_VIEWS):
            return None, c
        return c['recipient_gate_pass'], c

    def simple(name, q):
        gt, ct = gate_of(q)
        if gt is None:
            return None
        if ct['benefit']['label'] == 'ALIAS_ZERO':
            flags.append(f'CAND_ALIAS_OF_{name}')
        return {'label': ct['benefit']['label'], 'kind': 'SIMPLE', 'benefit': ct['benefit']['label'],
                'benefit_mean': ct['benefit']['aggregate'].get('mean'), 'benefit_se': ct['benefit']['aggregate'].get('se_agg'),
                'recipient_gate_pass': gt, 'recipient': {w: ct['recipient'][w]['label'] for w in NR.SINGLE_VIEWS}, 'point': q,
                'pass': bool(ct['benefit_positive'] and gt), 'gating': True}
    flags += list(((rec.get('fineness') or {}).get('flags')) or [])
    flags += [f'LO_NOT_SAVED_SPEC_BUDGET_{k}' for k in lns]
    if spec['ALTCRIT']['kind'] != 'ALIAS_OF_STOPFIT':
        if near:
            flags.append('ALTCRIT_NEAR_ALIAS_OF_STOPFIT_NOT_INDEPENDENT')
        elif not _fin(lag):
            flags.append('ALTCRIT_LAG_NOT_RECORDED')
    for name, q in (('START', sid), ('CONT', cid)):
        if q not in grid:
            return incomplete(f'grid entry vs {name} ({q}) missing (not recorded as not run)')
        c = simple(name, q)
        if c is None:
            return incomplete(f'grid entry vs {name} ({q}) incomplete')
        comps[name] = c
    point = {'id': cand.get('config_id', 'CAND'), 'costs_vs': rec.get('costs_vs') or {}, 'grid_vs': grid}
    amap = _alias_map(rec, sid, cid)
    for k in CONTROL_KEYS:
        s = spec[k]
        if s['kind'] == 'ALIAS_OF_STOPFIT':
            comps[k] = {'label': 'EXACT_ALIAS_OF_STOPFIT', 'kind': 'ALIAS_OF_STOPFIT', 'alias_of': 'STOPFIT', 'gating': False,
                        'pass': None, 'point': None,
                        'statement': 'ALTCRIT is an exact alias of STOPFIT (identical wires, verified): its artifact is reused and '
                                     'the comparison is counted once'}
            flags.append('ALTCRIT_EXACT_ALIAS_OF_STOPFIT_COUNTED_ONCE')
            continue
        if s['ids'] is None:
            return incomplete(f'{k} saved points missing (not recorded as not run)')
        if s['kind'] == 'PATH':
            c, err, f = _frontier_part(k, s['ids'], point, grid, gate_of, sid, amap)
        else:
            c, err, f = _point_part(k, amap.get(s['ids'][0], s['ids'][0]), point, gate_of)
        flags += f
        if err:
            return incomplete(err)
        c['lo_saved'] = k not in lns
        if k == 'ALTCRIT':
            c['independent'] = not near
        comps[k] = c
    tg = {}
    for q in dict.fromkeys(x['point'] for x in comps.values() if x.get('point') and x.get('gating')):
        x = _point_positive((rec.get('nolookup') or {}).get(q))
        if x is None:
            return incomplete(f'reader-targeting input (non-table winner) vs {q} missing')
        tg[q] = x
    passed = []
    for n in gating:
        x = comps[n]
        x['targeting_pass'] = tg[x['point']]['pass'] if x.get('point') in tg else None
        if x['pass']:
            passed.append(n)
            if x['targeting_pass'] is False:
                flags.append(f'READER_TARGETED_{n}')
    for q in tg:
        y = _point_positive((rec.get('enhanced') or {}).get(q))
        if y is not None and not y['pass']:
            flags.append(f'READER_SENSITIVE_ENHANCED_{q}')
    allc = len(passed) == len(gating)
    targeted = not all(x['pass'] for x in tg.values())
    if allc and targeted:
        label = 'READER_TARGETED'
    elif allc:
        label = D2_MET
    elif not passed:
        label = 'NOT_SEPARATED'
    else:
        label = 'PARTIAL_' + '+'.join(passed) + '_ONLY'
    failed = [n for n in gating if n not in passed]
    scope = D2_SCOPE_NEAR_ALIAS if near else D2_SCOPE
    if label == D2_MET:
        statement = scope
    elif label == 'READER_TARGETED':
        statement = ('every gating comparison passed but the non-table winner does not show a positive point benefit in both '
                     'versions against every comparison point: the gain may be specific to the selected reader; not a component gain')
    else:
        statement = (f'passed: {", ".join(passed) or "none"}; not passed: {", ".join(failed)}; a partial outcome is not a '
                     'component gain' + ''.join(f'; {FAILED_EXPLANATION[n]}' for n in D2_COMPARISONS if n in failed))
    return {'label': label, 'met': label == D2_MET, 'passed': passed, 'failed': failed, 'gating': list(gating),
            'comparisons': comps, 'targeting': tg, 'flags': sorted(set(flags)), 'bounds': _bounds(comps, gating),
            'minimum_detectable': MDE_NOTE, 'scope': scope, 'statement': statement, 'start': sid, 'cont': cid,
            'altcrit_near_alias': near, 'lo_not_saved': list(lns),
            'control_kinds': {k: spec[k]['kind'] for k in CONTROL_KEYS},
            'selection_data': {k: sel.get(k) for k in keys},
            'note': 'no partial outcome is a claim; "not separated" is the default reading; every sentence quoting this label '
                    'carries its scope'}


# ---------------------------------------------------------------- (5) learning curve (budget sensitivity; descriptive)
def _lc_one(full, half, diff_se=None):
    """One comparator: d = b(FULL) - b(HALF) per registered cell; SE = the paired-bootstrap SE when recorded for every cell, else
    sqrt(se_full^2 + se_half^2) (an upper bound when the two budgets' SCORE errors are non-negatively correlated; flag
    LC_SE_UNPAIRED_BOUND); aggregated by NR.aggregate. None if an entry is incomplete."""
    fa, ha = NR.aggregate(full), NR.aggregate(half)
    if not fa.get('complete') or not ha.get('complete'):
        return None
    fc, hc = full['cells'], half['cells']
    fs, hs = full['cell_se'], half['cell_se']
    cells = {c: float(fc[c]) - float(hc[c]) for c in NR.REGISTERED}
    paired = isinstance(diff_se, dict) and all(_fin(diff_se.get(c)) and diff_se[c] >= 0 for c in NR.REGISTERED)
    ses = ({c: float(diff_se[c]) for c in NR.REGISTERED} if paired else
           {c: math.sqrt(float(fs[c]) ** 2 + float(hs[c]) ** 2) for c in NR.REGISTERED})
    fm, hm = full.get('cell_se_mc') or {}, half.get('cell_se_mc') or {}
    e = {'cells': cells, 'cell_se': ses}
    if fm or hm:
        e['cell_se_mc'] = {c: math.sqrt(float(fm.get(c) or 0.0) ** 2 + float(hm.get(c) or 0.0) ** 2) for c in NR.REGISTERED}
    a = NR.aggregate(e)
    pv = {v: a['per_version'][v]['mean'] for v in VERSIONS}
    if (not a['alias']) and a['upper'] < 0 and all(pv[v] < 0 for v in VERSIONS):
        lab = 'BENEFIT_SHRINKS_RESOLVED'
    elif a['mean'] < 0:
        lab = 'BENEFIT_SHRINKS_POINT'
    else:
        lab = 'NO_SHRINKAGE_AT_POINT'
    return {'label': lab, 'change_mean': a['mean'], 'change_se': a['se_agg'], 'change_by_version': pv,
            'full_mean': fa['mean'], 'half_mean': ha['mean'], 'full_label': NR.benefit_screen(full)['label'],
            'half_label': NR.benefit_screen(half)['label'], 'limitation': bool(fa['mean'] > 0 and a['mean'] < 0),
            'se_source': 'PAIRED_BOOTSTRAP' if paired else 'UNPAIRED_BOUND'}


def learning_curve(rec, comparators):
    """Budget-sensitivity reading at CAND (RULES.md 5; descriptive; never changes D1-D6 or the class). comparators = the ids to read
    (decisions(): START, CONT, every gating Q* / point, and H_LJ). Per comparator: BENEFIT_SHRINKS_RESOLVED (aggregate upper bound of
    d < 0 and both version means < 0), BENEFIT_SHRINKS_POINT (mean d < 0), NO_SHRINKAGE_AT_POINT, NOT_RUN (no half-budget entry),
    INCOMPLETE; 'limitation' iff the full-budget point benefit is > 0 and mean d < 0. Family: LEARNABILITY_LIMITATION if any
    comparator has a limitation (carried into the headline); NOT_RUN if no half-budget entry exists; INCOMPLETE if a comparator is
    not run or incomplete; else NO_SHRINKAGE_SEEN (never evidence of information removal)."""
    lc = (rec or {}).get('learning_curve')
    if not isinstance(lc, dict) or not lc.get('half'):
        return {'label': 'NOT_RUN', 'note': LC_NOTE}
    why = _gate_label(rec.get('grid_valid')) or _gate_label(lc.get('grid_valid'))
    if why:
        return {'label': why, 'note': LC_NOTE}
    grid, half, dse = rec.get('grid') or {}, lc.get('half') or {}, lc.get('diff_cell_se') or {}
    per, flags = {}, []
    for c in dict.fromkeys(x for x in comparators if x):
        h, f = (half.get(c) or {}).get('pair'), (grid.get(c) or {}).get('pair')
        if h is None:
            per[c] = {'label': 'NOT_RUN'}
            continue
        one = _lc_one(f, h, dse.get(c)) if f is not None else None
        per[c] = one if one is not None else {'label': 'INCOMPLETE'}
        if one is not None and one['se_source'] == 'UNPAIRED_BOUND':
            flags.append('LC_SE_UNPAIRED_BOUND')
    lim = [c for c, x in per.items() if x.get('limitation')]
    if lim:
        fam = 'LEARNABILITY_LIMITATION'
    elif any(x['label'] in ('NOT_RUN', 'INCOMPLETE') for x in per.values()):
        fam = 'INCOMPLETE'
    else:
        fam = 'NO_SHRINKAGE_SEEN'
    b = lc.get('budgets') or {}
    inner = lc.get('inner_episode_fit_size')
    if not (_fin(b.get('FULL')) and _fin(b.get('HALF'))):
        flags.append('LC_BUDGETS_NOT_RECORDED')
    if not _fin(inner):
        flags.append('LC_INNER_EPISODE_SIZE_NOT_RECORDED')
    elif _fin(b.get('FULL')) and inner >= b['FULL']:
        flags.append('INNER_EPISODE_BUDGET_AT_LEAST_EVALUATION_FIT')
    cav = (f'the pair benefit against {", ".join(lim)} is smaller with readers fitted on the full FIT than on its fixed half: a '
           'learnability effect is possible; reported as a limitation, never as information removal' if lim else None)
    return {'label': fam, 'per_comparator': per, 'limitations': lim, 'headline_caveat': cav, 'flags': sorted(set(flags)),
            'budgets': b, 'inner_episode_fit_size': inner, 'note': LC_NOTE}


# ---------------------------------------------------------------- (6) D3, D4 with the reader-budget credit rule
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
    """D3 = NRM1 G4 (every point-eligible reference of the nine by the published point labels, intersection-union, recipient gates;
    no reserve rule removes a reference) with the reader-budget rule; the references passed and failed are listed."""
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
            'references': list(refs), 'reader_budget_credit': cr, 'not_credited': nc, 'incomplete_references': inc,
            'missing': cl['G4'].get('missing'),
            'per_reference': {r: {'benefit': comps[r]['benefit']['label'], 'mean': comps[r]['benefit']['aggregate'].get('mean'),
                                  'se_agg': comps[r]['benefit']['aggregate'].get('se_agg'),
                                  'recipient_gate_pass': comps[r]['recipient_gate_pass']} for r in refs if r in comps}}


def _d4(rec, cl):
    """D4 = NRM1 G5 (TARGET_AGG_RESOLVED: aggregate mean - 2 se_agg > 0.020 and both version means > 0.020, vs NEURAL_RAW and H_LJ,
    with the recipient gates; levels kept) with the reader-budget rule for the two targets; never weakened."""
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
    """D5 = NRM1 G6 unchanged, only when a card claims joint design (a local twin and both sequential orders built); otherwise
    NOT_APPLICABLE_NOT_JOINT (not claimed: RFA1 builds no local twin). Collapsed sequential orders as SAR1."""
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
                    'statement': 'the joint design equals a sequential order by construction: no joint-design claim'}
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
    return {'label': lab, 'detail': cl['G6'], 'flags': flags,
            'statement': ('joint-design contribution supported against the separately designed recipients and both sequential '
                          'orders (NRM1 G6)' if lab == 'JOINT_CONTRIBUTION_SUPPORTED' else 'no joint-design advantage is claimed')}


# ---------------------------------------------------------------- (7) registered vs adaptive readings
def _family(rec):
    """The canonical family: RFA (its code alias FAM maps to RFA) or FS_<name>; anything else is refused."""
    fam = (rec or {}).get('family')
    if not isinstance(fam, str) or not FAMILY_RE.fullmatch(fam):
        raise ValueError(f'unknown family {fam!r}: a reading is RFA (code alias FAM) or FS_<name>')
    return FAMILY_ALIASES.get(fam, fam)


def adaptive_tag(rec):
    """None for the registered reading: revision 0 of RFA whose card was committed before any MODEL_SELECT outcome of the campaign.
    Otherwise '<KIND>_<parts>': KIND = REVISED if the card was committed before any MODEL_SELECT outcome, else ADAPTIVE; parts = FS
    (replacement family), R<n> (revision n >= 1), R0_LATE_CARD (a revision-0 card committed after an outcome or not recorded)."""
    fam = _family(rec)
    n = int((rec or {}).get('revision') or 0)
    fs = fam.startswith('FS_')
    early = (rec or {}).get('card_committed_before_any_model_select_outcome') is True
    if n == 0 and fam in REGISTERED_FAMILIES and early:
        return None
    parts = (['FS'] if fs else []) + ([f'R{n}'] if n else ([] if fs else ['R0_LATE_CARD']))
    return f"{'REVISED' if early else 'ADAPTIVE'}_{'_'.join(parts)}"


def _mark(out, tag):
    for k in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'LC', 'outcome_class'):
        x = out.get(k)
        if isinstance(x, dict) and 'label' in x:
            x['label_unprefixed'] = x['label']
            if tag:
                x['label'] = f'{tag}__{x["label"]}'


def _raw(x):
    return (x or {}).get('label_unprefixed', (x or {}).get('label'))


# ---------------------------------------------------------------- (7) decisions, the five outcomes and the outcome class
def decisions(rec):
    """D1, D2, D3-D6, the learning curve, D2R, the five separately reported outcomes and the outcome class of one reading
    (RULES.md 2-7). Outcome class, first applicable: TASK_INFEASIBLE; SELECTED_POINT_INFEASIBLE; INCOMPLETE (D1 incomplete; D2
    incomplete or not computed); NO_OUTPUT_CHANGE (CAND = START); MECHANISM_NOT_ASSESSED (a gating control not run); INCOMPLETE (D3
    undecided); COMPETITIVE_CANDIDATE (D2 met and D3 met; D4 reported separately, never implied); MECHANISM_COMPONENT_NOT_COMPETITIVE
    (D2 met, D3 not); REFERENCE_ADVANTAGE_NOT_ATTRIBUTED (D3 met, D2 not); BOUNDED_NEGATIVE_EXPERIMENT (otherwise; 'partial' = D2's
    label and the development bounds). D4, D5, D6, D2R and the learning curve never change the class. A non-registered reading has
    every label prefixed '<tag>__'."""
    rec = rec or {}
    fam = _family(rec)
    cand = rec.get('cand') or {}
    rev = int(rec.get('revision') or 0)
    out = {'family': fam, 'card': rec.get('card'), 'revision': rev, 'start': ((rec.get('start') or {}).get('id')),
           'D1': d1(rec), 'D2': d2(rec)}
    d2lab = out['D2']['label']
    cg = NR.point_eligibility(cand.get('evals') or {})['label'] if cand.get('status') == 'FITTED' else None
    gates = ('NOT_ASSESSED_MECHANISM_INERT', 'NOT_ASSESSED_NO_FEASIBLE_POINT', 'NOT_ASSESSED_EXACT_ALIAS',
             'NOT_ASSESSED_TASK_INFEASIBLE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID')
    if d2lab in gates:
        for k in ('D3', 'D4', 'D5'):
            out[k] = {'label': d2lab, 'met': False}
    elif cand.get('status') != 'FITTED' or cg != 'POINT_ELIGIBLE' or _cand_selection_error(cand):
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
    assessed = d2lab in D2_ASSESSED or d2lab.startswith('PARTIAL_')
    if assessed:
        pts = [x.get('point') for x in (out['D2'].get('comparisons') or {}).values() if x.get('gating')]
        out['LC'] = learning_curve(rec, pts + ['H_LJ'])
    else:
        out['LC'] = {'label': d2lab if d2lab in gates or d2lab == 'NOT_ASSESSED_CONTROLS_NOT_RUN' else 'INCOMPLETE',
                     'note': 'read only when D2 reached its conjuncts'}
    r2 = rec.get('d2r') if isinstance(rec.get('d2r'), dict) else None
    out['D2R'] = {'label': (r2 or {}).get('label', 'NOT_RUN'), 'sign_changes': (r2 or {}).get('sign_changes', []),
                  'recipient_failures': (r2 or {}).get('recipient_failures', []),
                  'headline_caveat': (r2 or {}).get('headline_caveat')}
    flags = list(out['D2'].get('flags') or []) + list(rec.get('mc_flags') or []) + list(out['D5'].get('flags') or []) + \
        list(out['D1'].get('flags') or []) + list(out['LC'].get('flags') or [])
    d1lab, d3lab = out['D1']['label'], out['D3']['label']
    if _differs_by_version(cand.get('operating_point')):
        flags.append('CAND_OPERATING_POINT_DIFFERS_BY_VERSION')
    if rev not in PLANNED_REVISIONS:
        flags.append('REVISION_BEYOND_PLANNED')
    why, partial = None, None
    d2met, d3met = bool(out['D2'].get('met')), bool(out['D3'].get('met'))
    undecided = ('INCOMPLETE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID')
    if d1lab == 'NOT_FITTED_MECHANISM_INERT':
        oc = 'MECHANISM_INERT'
    elif d1lab == 'NO_FEASIBLE_POINT':
        oc = 'TASK_INFEASIBLE'
        flags.append('NO_POINT_MEETS_MODEL_FIT_RULE')
    elif d1lab == 'CAND_INFEASIBLE':
        pts = out['D1'].get('points')
        if pts is None:
            oc, why = 'INCOMPLETE', 'CAND fails G1 and the fitted operating points are not recorded'
        else:
            others = {k: x for k, x in pts.items() if k != cand.get('config_id')}
            if any(x == 'POINT_ELIGIBLE' for x in others.values()):
                oc = 'SELECTED_POINT_INFEASIBLE'
            elif all(x == 'POINT_INELIGIBLE' for x in others.values()):
                oc = 'TASK_INFEASIBLE'
            else:
                oc, why = 'INCOMPLETE', 'a fitted operating point has incomplete utility'
    elif d1lab == 'INCOMPLETE' or d2lab in undecided:
        oc, why = 'INCOMPLETE', f'D1 {d1lab}, D2 {d2lab}'
    elif d2lab == 'NOT_ASSESSED_EXACT_ALIAS':
        oc = 'NO_OUTPUT_CHANGE'
    elif d2lab == 'NOT_ASSESSED_CONTROLS_NOT_RUN':
        oc = 'MECHANISM_NOT_ASSESSED'
    elif d3lab in undecided:
        oc, why = 'INCOMPLETE', f'D3 {d3lab}'
    elif d2met and d3met:
        oc = 'COMPETITIVE_CANDIDATE'
    elif d2met:
        oc = 'MECHANISM_COMPONENT_NOT_COMPETITIVE'
    elif d3met:
        oc = 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED'
        partial = d2lab
    else:
        oc = 'BOUNDED_NEGATIVE_EXPERIMENT'
        partial = d2lab
    if d3met and not d2met:
        flags.append('D3_WITHOUT_D2')
    if out['D4'].get('met') and not d2met:
        flags.append('D4_WITHOUT_D2')
    if out['D2R']['sign_changes']:
        flags.append('D2R_SIGN_CHANGE_CARRIED_INTO_HEADLINE')
    if out['D2R']['recipient_failures']:
        flags.append('D2R_RECIPIENT_FAILURE_CARRIED_INTO_HEADLINE')
    if out['LC'].get('headline_caveat'):
        flags.append('LEARNABILITY_LIMITATION_CARRIED_INTO_HEADLINE')
    out['falsifiers'] = _falsifiers(rec, out, assessed)
    flags += [f'FALSIFIER_{k}_TRIGGERED' for k, x in out['falsifiers']['per_falsifier'].items() if x == 'TRIGGERED']
    tag = adaptive_tag(rec)
    out['outcome_class'] = {'label': oc, 'partial': partial, 'flags': sorted(set(flags)), 'reason': why,
                            'bounds': (out['D2'].get('bounds') if oc == 'BOUNDED_NEGATIVE_EXPERIMENT' else None),
                            'note': 'development evidence on adaptively used MODEL_SELECT records, never confirmation; no class '
                                    'renames a partial result as the desired method; a near-limit point estimate is never called '
                                    'robust'}
    out['registered'] = tag is None
    out['adaptive_tag'] = tag
    out['registration_note'] = ('registered reading (revision 0 of RFA, card committed before any MODEL_SELECT outcome)'
                                if tag is None else 'NOT a registered result: ' + tag + ' (labels prefixed; report beside the '
                                                    'registered reading with the number of looks)')
    _mark(out, tag)
    conf = (out['D1'].get('confidence') or {}).get('cand') or {}
    o = {'1_usefulness': {'label': out['D1']['label'], 'met': bool(out['D1'].get('met')),
                          'confidence_conjunction': conf.get('label'),
                          'reserve_qualified': ((out['D1'].get('reserve') or {}).get('cand') or {}).get('qualified')},
         '2_gain_beyond_mechanism_controls': {'label': out['D2']['label'], 'met': d2met, 'passed': out['D2'].get('passed', []),
                                              'failed': out['D2'].get('failed', []), 'robustness': out['D2R']['label'],
                                              'learning_curve': out['LC']['label'],
                                              'scope': out['D2'].get('scope') if d2met else out['D2'].get('statement')},
         '3_strong_references': {'label': out['D3']['label'], 'met': d3met, 'passed': out['D3'].get('passed', []),
                                 'failed': out['D3'].get('failing', [])},
         '4_material_target': {'label': out['D4']['label'], 'met': bool(out['D4'].get('met'))},
         '5_joint_design_and_novelty': {'joint_design': out['D5']['label'], 'novelty': out['D6']['label'],
                                        'note': 'no joint-design claim is planned (no local twin is built); novelty only from a '
                                                'separate prior-art review'}}
    cav = [x for x in (out['D2R']['headline_caveat'], out['LC'].get('headline_caveat')) if x]
    o['statement'] = (f"(1) usefulness: {o['1_usefulness']['label']} (confidence {o['1_usefulness']['confidence_conjunction']}, "
                      f"reserve qualified {o['1_usefulness']['reserve_qualified']}); (2) gain beyond the mechanism-specific "
                      f"controls: {o['2_gain_beyond_mechanism_controls']['label']} (D2R "
                      f"{o['2_gain_beyond_mechanism_controls']['robustness']}; learning curve "
                      f"{o['2_gain_beyond_mechanism_controls']['learning_curve']}); (3) strong references: "
                      f"{o['3_strong_references']['label']} (passed {o['3_strong_references']['passed']}, failed "
                      f"{o['3_strong_references']['failed']}); (4) material target: {o['4_material_target']['label']}; (5) joint "
                      f"design: {o['5_joint_design_and_novelty']['joint_design']}, novelty: "
                      f"{o['5_joint_design_and_novelty']['novelty']}" + ''.join(f'; {c}' for c in cav))
    out['outcomes'] = o
    out['scope'] = list(NR.SCOPE) + ['A lower reader AUC is evidence about the declared finite-data reader procedure and its '
                                     'auxiliary budget, not an arbitrary-attacker guarantee and not information removal.',
                                     'Point feasibility, the confidence conjunction and the reserve heuristic are separate results.',
                                     'A revision or replacement-family reading is never the registered result.',
                                     'The five outcomes (usefulness, gain beyond the mechanism-specific controls, strong '
                                     'references, material target, joint design and novelty) are reported separately; none '
                                     'implies another.']
    return out


def _falsifiers(rec, out, assessed):
    """The card's registered falsifiers (RF_RFIT_v1 6), descriptive, never changing a label: F0 from the F0 record; F2 (no resolved
    FULL benefit over STOPFIT at matched spend) from D2's STOPFIT comparison; F4 (learnability limitation) from the learning curve;
    F1, F3, F5, F6 as recorded by the coordinator from the logged diagnostics (NOT_RECORDED otherwise)."""
    f0 = (out['D1'].get('f0') or {})
    per0 = f0.get('per_version') or {}
    f = {'F0': ('TRIGGERED' if f0.get('inert_versions') else
                'NOT_TRIGGERED' if all(per0.get(v) == 'MECHANISM_ACTIVE' for v in VERSIONS) else 'NOT_RECORDED')}
    st = (out['D2'].get('comparisons') or {}).get('STOPFIT') or {}
    if not assessed or not st:
        f['F2'] = 'NOT_ASSESSED'
    elif st.get('reason') == 'NO_BRACKET':
        f['F2'] = 'NOT_ASSESSABLE_NO_BRACKET'
    else:
        f['F2'] = ('NOT_TRIGGERED' if st.get('frontier_label') in ('DOMINATES_CONTROL_FRONTIER', 'SEPARATED_ON_PRIMARY_COST_ONLY')
                   else 'TRIGGERED')
    lc = _raw(out.get('LC'))
    f['F4'] = 'TRIGGERED' if lc == 'LEARNABILITY_LIMITATION' else ('NOT_TRIGGERED' if lc == 'NO_SHRINKAGE_SEEN' else lc)
    recd = (rec or {}).get('falsifier_records') or {}
    for k in ('F1', 'F3', 'F5', 'F6'):
        x = recd.get(k)
        f[k] = x if x in ('TRIGGERED', 'NOT_TRIGGERED') else 'NOT_RECORDED'
    return {'per_falsifier': {k: f[k] for k in FALSIFIERS},
            'note': 'registered falsifiers of card RF_RFIT_v1 (section 6); descriptive, never changing a label or the class; F1, F3, '
                    'F5 and F6 come from the coordinator\'s logged MODEL_FIT and per-recipe diagnostics'}


def campaign_summary(decision_records):
    """The registered headline (one registered reading; a second one is refused), every non-registered reading with its tag, the
    number K of D2 looks, the readings with D2, D3, D4 met and a learnability limitation, and the multiplicity statement every
    campaign-level sentence must carry. Flag D2_MET_ONLY_IN_NON_REGISTERED_LOOKS."""
    reg, adaptive, looks, met, d3m, d4m, lcl = {}, [], 0, [], [], [], []
    for d in decision_records or []:
        fam = d.get('family')
        lab = _raw(d.get('D2')) or ''
        assessed = lab in D2_ASSESSED or lab.startswith('PARTIAL_')
        looks += assessed
        row = {'family': fam, 'card': d.get('card'), 'revision': d.get('revision'), 'tag': d.get('adaptive_tag'),
               'D2': lab, 'D3': _raw(d.get('D3')), 'D4': _raw(d.get('D4')), 'LC': _raw(d.get('LC')),
               'outcome_class': _raw(d.get('outcome_class')), 'd2_assessed': assessed,
               'outcomes_statement': (d.get('outcomes') or {}).get('statement')}
        if d.get('registered'):
            if fam in reg:
                raise ValueError(f'two registered readings for family {fam!r}')
            reg[fam] = row
        else:
            adaptive.append(row)
        if (d.get('D2') or {}).get('met'):
            met.append(row)
        if (d.get('D3') or {}).get('met'):
            d3m.append(row)
        if (d.get('D4') or {}).get('met'):
            d4m.append(row)
        if row['LC'] == 'LEARNABILITY_LIMITATION':
            lcl.append(row)
    k0 = sum(r['d2_assessed'] for r in reg.values())
    m0 = sum(1 for r in met if r['tag'] is None)
    flags = ['D2_MET_ONLY_IN_NON_REGISTERED_LOOKS'] if met and not m0 else []
    n = len(reg) + len(adaptive)
    return {'registered': reg, 'non_registered': adaptive, 'd2_looks': looks, 'd2_met': met, 'd3_met': d3m, 'd4_met': d4m,
            'learnability_limitations': lcl, 'flags': flags,
            'statement': (f'D2 was met in {m0} of {k0} registered readings and in {len(met) - m0} of {looks - k0} '
                          f'non-registered readings ({looks} D2 looks in total); D3 was met in {len(d3m)} and D4 in {len(d4m)} of '
                          f'{n} readings; a learnability limitation was carried in {len(lcl)}; {PER_LOOK_FALSE_PASS}.')}


# ---------------------------------------------------------------- (8) D2R
def d2r(rec):
    """D2R robustness to classifier refitting (RULES.md 8; descriptive; never changes D2). rec = {'grid_valid', 'release': the D2R
    release (CAND), 'refit': refit_effectiveness(...) record, 'rob': {comparator: NR grid entry (pair, income, occupation) of the
    release vs it, registered and robustness cells MERGED}, 'comparators': {'START': id, 'CONT': id, 'STOPFIT': id | None (no
    bracket) | a brackets() entry, 'ALTCRIT': the same | 'ALIAS_OF_STOPFIT', 'RAWJ': the same}} (the registered D2 comparators;
    never re-chosen on robustness cells). Per decisive comparator: ROBUSTNESS_CONFIRMED iff NR.robustness confirms (AGG_POSITIVE on
    the robustness cells and on all cells), both recipient gates hold on the robustness cells, the registered cells are present and
    the refit is REFIT_GENUINELY_EFFECTIVE; NOT_RUN if the entry is missing; else ROBUSTNESS_NOT_CONFIRMED. Family label over the
    decisive set. A sign change (robustness-cell against registered-cell aggregate mean) or a failed robustness recipient gate on
    any comparator, decisive or beside (the references present, at least NEURAL_RAW and H_LJ), is carried into the headline."""
    rec = rec or {}
    why = _gate_label(rec.get('grid_valid'))
    if why:
        return {'label': why}
    comp = rec.get('comparators')
    if not isinstance(comp, dict) or any(k not in comp for k in D2_COMPARISONS) or not comp.get('START') or not comp.get('CONT'):
        return {'label': 'INCOMPLETE', 'reason': 'decisive comparators not recorded (START id, CONT id, STOPFIT / ALTCRIT / RAWJ '
                                                 'bracket id, None, a brackets() entry or ALIAS_OF_STOPFIT for ALTCRIT)'}
    br = {}
    for k in CONTROL_KEYS:
        x = comp[k]
        if x == 'ALIAS_OF_STOPFIT':
            if k != 'ALTCRIT':
                return {'label': 'INCOMPLETE', 'reason': f'{k} cannot be an alias of STOPFIT'}
            x = None
        elif isinstance(x, dict):
            if x.get('status') in ('INCOMPLETE', 'PATH_NOT_GIVEN'):
                return {'label': 'INCOMPLETE', 'reason': f'{k} bracket record {x.get("status")}'}
            x = x.get('bracket')
        br[k] = x
    refit_ok = (rec.get('refit') or {}).get('label') == 'REFIT_GENUINELY_EFFECTIVE'
    decisive = list(dict.fromkeys([x for x in (comp['START'], comp['CONT']) + tuple(br[k] for k in CONTROL_KEYS) if x]))
    rob_all = rec.get('rob') or {}
    beside = list(dict.fromkeys([r for r in NR.BENCHMARK if r not in decisive and r in rob_all] +
                                [r for r in NR.TARGET_REFS if r not in decisive]))
    per, signs, rfail = {}, [], []
    for c in decisive + beside:
        e = rob_all.get(c)
        if not e:
            per[c] = {'label': 'NOT_RUN', 'decisive': c in decisive}
            continue
        r = NR.robustness(e)
        if r['label'] == 'NOT_RUN':
            per[c] = {'label': 'NOT_RUN', 'decisive': c in decisive}
            continue
        cells = ((e.get('pair') or {}).get('cells') or {})
        has_reg = all(y in cells for y in NR.REGISTERED)
        regm = NR.aggregate(e.get('pair'), 'registered').get('mean') if has_reg else None
        rob = r['robustness_cells']['aggregate']
        sc = _fin(regm, rob.get('mean')) and _sign(regm) != 0 and _sign(rob['mean']) != 0 and _sign(regm) != _sign(rob['mean'])
        if sc:
            signs.append(c)
        rg = {w: NR.recipient_screen(e.get(w), cells='robustness') for w in NR.SINGLE_VIEWS}
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
                  'all_cells_label': r['all_cells']['label'], 'sign_change': sc,
                  'recipient_robustness': {w: rg[w]['label'] for w in NR.SINGLE_VIEWS}}
    labs = [per[c]['label'] for c in decisive]
    fam = ('NOT_RUN' if any(y == 'NOT_RUN' for y in labs) else
           'ROBUSTNESS_CONFIRMED' if all(y == 'ROBUSTNESS_CONFIRMED' for y in labs) else 'ROBUSTNESS_NOT_CONFIRMED')
    parts = ([f'the pair benefit changes sign on the robustness cells against {", ".join(signs)}'] if signs else []) + \
            ([f'a single-view recipient condition fails on the robustness cells against {", ".join(rfail)}'] if rfail else [])
    cav = ('; '.join(parts) + ': carried into the headline') if parts else None
    return {'label': fam, 'release': rec.get('release', 'CAND'), 'decisive': decisive, 'beside': beside, 'per_comparator': per,
            'sign_changes': signs, 'recipient_failures': rfail, 'headline_caveat': cav,
            'flags': [f'SIGN_CHANGE_{c}' for c in signs] + [f'ROBUSTNESS_RECIPIENT_FAILED_{c}' for c in rfail],
            'refit': (rec.get('refit') or {}).get('label'), 'counted_once': (rec.get('refit') or {}).get('counted_once'),
            'note': 'descriptive; never changes D2; no retuning on the robustness cells (CAND, its brackets and every selection are '
                    'the registered ones); repeated reader splits and reader seeds on the same people are not replication'}
