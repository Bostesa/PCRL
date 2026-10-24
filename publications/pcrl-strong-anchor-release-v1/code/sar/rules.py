"""Frozen decision rules of the PCRL strong-anchor release campaign (`pcrl_strong_anchor_release_v1`; statistical reviewer; written
before any outcome of this campaign existed; no label read).

Human-readable rules: results/pcrl_strong_anchor_release_v1/roles/statistical_reviewer/RULES.md
Machine-readable constants: .../RULES.json (schema SAR1-rules-v1); `load_rules()` refuses a RULES.json that disagrees with this
module, a changed NRM1 rule file or a changed nrm/rules.py.
Invented-data tests: `python3 -m sar.test_rules` (prints JSON; exit 0 iff every test passes).

Pure functions on plain dicts / floats. Nothing here reads data or labels (the only files read are this campaign's RULES.json, the
NRM1 RULES.json and, through nrm.rules, the OHN1 reserve rule file). The inherited machinery is `nrm.rules`, imported UNCHANGED as
NR and never monkeypatched: point_eligibility (G1), reserve_status (the two-SE reserve heuristic), the aggregate screen (with NR's
mc_cell / se_mc handling), benefit_screen, target_screen, recipient_screen, comparison (benefit + recipient gate),
frontier_comparison (the spend-bracketing rule), robustness and claims (G4 / G5 / G6). Copied from ipr.rules (itself ddr / frc
logic; none is imported): grid validity with the prospective extra-trees explanation, refit effectiveness, the reader-budget credit
rule, the control-path alias collapse, the non-table winner's version means, D5's collapse rules and the registered / adaptive
labelling.

SAR1 changes (RULES.md 0.1): the comparison structure is START (the family's unchanged starting release: FA SUCC, FB NEURAL_RAW),
CONT (an equally exposed task-only or zero-protection continuation, gating iff the card declares it needed), the CTRL frontier
(the path origin, START or CONT as the card declares, followed by the two saved CTRL points that bracket CAND's spend over START on
the card's declared selection data: MODEL_FIT out-of-fold, or CALIBRATION when START is in-sample on MODEL_FIT, never
MODEL_SELECT) and, when the card declares it, the ALT frontier (START followed by two saved points of the starting release's own
protection operation continued to matched spend; same bracket rule and conjuncts), with the nine strong references for D3 / D4;
D1 reports point eligibility (gating), the confidence conjunction
(PASS / UNRESOLVED / VIOLATION per condition) and the reserve separately; the usefulness margin UM against START's binding
condition (preparatory evidence only); D2R is read at CAND with sign changes carried into the headline; families FA, FB, FS_<name>;
an outcome class that keeps the three required outcomes separate; the few-valued explanation of seed equalities (counted once,
n_eff) and the +1000 reader-seed offset with offsets 300-900 forbidden.

API (RULES.md section in brackets):
  load_rules()                                     constants + '_sha256'                                                   [0]
  forbidden_seeds(), registered_seed(v, k), registered_boot_seed(v, rep)                                                   [9]
  grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=None)       validity of the registered SAR1 grid   [9]
  refit_effectiveness(splits, seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, rob_reps=None)               [8]
  confidence_conjunction(evals)                    PASS / UNRESOLVED / VIOLATION per condition (reported)                  [2]
  binding_condition(evals), point_margin(evals), condition_metric(name)                                                    [3]
  recovery_profile(entry)                          CAND vs START recovery profile retained (UM)                            [3]
  brackets(costs_vs, paths, alias_of=None, origin='START', alt_origin=None)   the CTRL / ALT brackets Q* (NR's rule)       [4]
  selection_data_error(sd, start_in_sample_on_model_fit)   allowed bracket data (MODEL_FIT_OOF; CALIBRATION if START in-sample) [4]
  grid_fineness(spends, start_in_sample_on_model_fit=None)   fineness of the saved CTRL / ALT points (reported only)     [4.4]
  d1(rec), um(rec), d2(rec)                        D1 task feasibility, UM usefulness margin, D2 component gain           [2-4]
  decisions(rec)                                   D1, UM, D2-D6, the three outcomes and the outcome class                 [2-7]
  adaptive_tag(rec), campaign_summary(decision_records)                                                                    [7]
  strongest_nonzero(configs), d2r(rec)                                                                                     [8]

Record of one family reading (`d1`, `um`, `d2`, `decisions`):
  {'family': 'FA' | 'FB' | 'FS_<name>', 'card', 'revision': 0 | 1 | 2, 'card_committed_before_any_model_select_outcome': bool,
   'grid_valid': grid_validity(...)['effective'] (None = not recorded),
   'start': {'id': 'SUCC' (FA) | 'NEURAL_RAW' (FB) | a benchmark reference (FS), 'evals': {v: utility},
             'binding': {v: condition name} (declared from the frozen anchor table)},
   'cand': {'status': 'FITTED' | 'NO_FEASIBLE_STRENGTH', 'config_id', 'evals', 'strength', 'every_nonzero_strength_alias_of_start'},
   'strengths': [{'config_id', 'strength', 'evals'}],
   'cont_declared': bool (card), 'cont': {'config_id', 'evals'} (when declared),
   'ctrl_origin': 'START' | 'CONT' (card; default START), 'alt_declared': bool (card; absent = False),
   'paths': {'CTRL': [saved CTRL ids] | None, 'ALT': [saved ALT ids] | None (origin START)},
   'selection_data': {'CTRL': 'MODEL_FIT_OOF' | 'CALIBRATION', 'ALT': ...} (plan; the bracket data of each gating path),
   'start_in_sample_on_model_fit': bool (plan; required for CALIBRATION bracket data),
   'controls_not_run': {'CONT': bool, 'CTRL': bool, 'ALT': bool} (budget only; never inferred),
   'alias_of': {id: earlier id | 'START' | 'CONT'},
   'grid': {comparator id: NR grid entry of CAND vs it} for START (keyed by its reference name), CONT, every CTRL point, the nine
           references and the D5 comparators,
   'costs_vs': {id: paired costs of CAND vs id ('total|nll' + the 8 secondary metrics)} for START, CONT and every CTRL point,
   'nolookup': {id: {'cells': {cell: non-table winner pair benefit}}}, 'enhanced' (reported only), 'reader_budget',
   'fineness': grid_fineness(...) record, 'joint' (D5), 'novelty' (D6), 'mc_flags', 'd2r': d2r(...) record (optional)}

Sign conventions are NR's: pair benefit b = comparator AUC - candidate AUC (positive favours the candidate); recipient difference
r = candidate AUC - comparator AUC; paired costs loss(candidate) - loss(comparator) (negative favours the candidate; accuracy enters
as a loss). Versions are the ints 0 and 1 (string keys '0' / '1' accepted).
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

from nrm import rules as NR                      # inherited claims machinery, imported unchanged (never monkeypatched)

RULES_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_strong_anchor_release_v1' / 'roles'
              / 'statistical_reviewer' / 'RULES.json')
SCHEMA = 'SAR1-rules-v1'
NRM1_RULES_SHA256 = '84f187768817e79ef9f2bb36151bea81f00b8f62d9179c8cd523069550674652'     # NRM1 RULES.json (registered)
NRM_RULES_PY_SHA256 = 'd5df868678d483a3452e251d8f24575932d23f0e5d22e3ef1ca3627a35030df1'   # nrm/rules.py (registered)
VERSIONS = NR.VERSIONS
Z = 2.0
TIE_TOL = 1e-12
RECIPIENT_LIMIT = NR.RECIPIENT_LIMIT               # 0.010 AUC, per version at the aggregate point (original)
# ---- the registered SAR1 reader grid (READER_SPLITS.json; sar.devreaders constants)
SEED_OFFSET = 1000                                 # MODEL_SEED_BASE + 1000 + 10 v + k (never used by an earlier grid)
FORBIDDEN_OFFSETS = (300, 400, 500, 600, 700, 800, 900)   # every realized earlier grid: OHN1, CTS, TSZ1, NRM1, FRC1, DDR1, IPR1
BOOT_SEED_BASE = 2026102400                        # + 100 v + repetition index
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
ALL_REPS = NR.REG_REPS + NR.ROB_REPS
ALL_K = NR.REG_K + NR.ROB_K
SPLIT_TAG = 'SAR1_devgrid_R{r}'
# ---- seed-equality explanations (prospective)
ER_MIN_LEAF = {'ET5': 5, 'ET25': 25}               # min_samples_leaf of the extra-trees recipes
ER_POOL_PEOPLE = 8001                              # MODEL_SELECT people: support bound = 8001 // min leaf (ET5 1600, ET25 320)
FEW_VALUED_MAX = ER_POOL_PEOPLE // ER_MIN_LEAF['ET25']    # 320: a view with at most 320 distinct released inputs is few-valued
# ---- anchors (frozen: anchors/ANCHOR_SELECTION.md, ANCHOR_TABLE.md)
STARTS = {'FA': 'SUCC', 'FB': 'NEURAL_RAW'}
START_BINDING = {'SUCC': {0: 'guard|sex0', 1: 'guard|sex0'},
                 'NEURAL_RAW': {0: 'guard|sex0', 1: 'income|U0|brier_excess'}}
GUARD_METRIC = {'guard|sex0': 'occupation|men_brier', 'guard|sex1': 'occupation|women_brier'}   # sex0 = men, sex1 = women
CLAUSE_METRIC = {'accuracy_loss': 'acc_loss', 'nll_excess': 'nll', 'brier_excess': 'brier'}
# ---- decisions
D1_LABELS = ('TASK_FEASIBLE', 'CAND_INFEASIBLE', 'NO_FEASIBLE_STRENGTH', 'INCOMPLETE')
CONDITION_STATES = ('PASS', 'UNRESOLVED', 'VIOLATION')
CONFIDENCE_LABELS = ('CONFIDENCE_ALL_PASS', 'CONFIDENCE_UNRESOLVED', 'CONFIDENCE_VIOLATION', 'INCOMPLETE')
UM_LABELS = ('MARGIN_RECOVERED', 'MARGIN_GAIN_SHIFTED_ELSEWHERE', 'MARGIN_GAIN_PROFILE_NOT_RETAINED', 'MARGIN_NOT_RESOLVED',
             'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'NOT_ASSESSED_EXACT_ALIAS', 'NOT_ASSESSED_TASK_INFEASIBLE',
             'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID', 'INCOMPLETE')
UM_STATEMENT = ('a usefulness-margin gain against START is preparatory evidence only: never a protection, component-gain or '
                'competitive claim; it is read on adaptively reused MODEL_SELECT people, and START\'s binding condition was '
                'identified on the same people (selection on its level, not on the paired difference)')
D2_COMPARISONS = ('START', 'CONT', 'CTRL_FRONTIER', 'ALT_FRONTIER')
FRONTIER_PATHS = (('CTRL_FRONTIER', 'CTRL'), ('ALT_FRONTIER', 'ALT'))  # CTRL always gating; ALT gating iff declared
D2_MET = 'COMPONENT_GAIN_BEYOND_START_AND_CONTROLS'
D2_ASSESSED = (D2_MET, 'READER_TARGETED', 'NOT_SEPARATED')     # plus PARTIAL_<passed>_ONLY
D2_GATES = ('NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'NOT_ASSESSED_EXACT_ALIAS', 'NOT_ASSESSED_TASK_INFEASIBLE',
            'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID', 'NOT_ASSESSED_CONTROLS_NOT_RUN', 'INCOMPLETE')
CTRL_ORIGINS = ('START', 'CONT')
SELECTION_DATA = ('MODEL_FIT_OOF', 'CALIBRATION')   # the card's declared bracket data; CALIBRATION only when START is in-sample
                                                    # on MODEL_FIT; never MODEL_SELECT
D2_SCOPE = ('the component gain is a resolved linked-pair benefit (with both recipient gates) beyond the unchanged start, the '
            'task-only continuation when the card declares it needed, and the untargeted control at matched MODEL_SELECT total-NLL '
            'spend, for the declared development readers on reused MODEL_SELECT people; it is not superiority over the strong '
            'references (D3) and not the material target (D4)')
D2_SCOPE_ALT = (D2_SCOPE + '; with the ALT path declared, it is also beyond the starting release\'s own protection operation '
                'continued to matched MODEL_SELECT total-NLL spend')
NO_BRACKET_STATEMENT = ('no {key} point spent as much MODEL_SELECT total NLL as CAND: lower recovery cannot be separated from task '
                        'spending; this is a limit of the control path, never evidence that CAND equals the control or that the '
                        'mechanism has no effect, and the mechanism\'s contribution is not called isolated')
FAMILY_RE = re.compile(r'(FA|FB|FS_[A-Za-z0-9]+)')
REGISTERED_FAMILIES = ('FA', 'FB')
FINENESS_FRACTION = 0.25                           # saved CTRL points: lower >= 0.75 s, upper <= s + 2 SE + 0.25 s (MODEL_FIT)
OUTCOMES = ('METHOD_CANDIDATE', 'COMPONENT_GAIN_NOT_COMPETITIVE', 'REFERENCE_ADVANTAGE_NOT_ATTRIBUTED', 'USEFULNESS_MARGIN_ONLY',
            'FEASIBLE_NO_RELIABLE_GAIN', 'NO_OUTPUT_CHANGE', 'TASK_INFEASIBLE', 'SELECTED_STRENGTH_INFEASIBLE',
            'PROTECTION_NOT_ASSESSED', 'INCOMPLETE')
D2R_LABELS = ('ROBUSTNESS_CONFIRMED', 'ROBUSTNESS_NOT_CONFIRMED', 'NOT_RUN', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN',
              'NOT_COMPUTED_GRID_INVALID', 'INCOMPLETE')
ADAPTIVE_KINDS = ('ADAPTIVE', 'REVISED')
PER_LOOK_FALSE_PASS = ('one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under six moderate NRM1 '
                       'known laws and about 8% when one version or split dominates (NRM1 RULES.md 5.5); a D2 look is an '
                       'intersection-union conjunction of two to four such comparisons (START, CONT when declared, the CTRL '
                       'frontier at its bracket, the ALT frontier at its bracket when declared), so its false-pass rate is at '
                       'most that of its weakest member; with K D2 looks '
                       'the chance that at least one passes by chance is at most K times the per-look rate; no multiplicity '
                       'correction is applied or claimed')
MDE_NOTE = ('per comparison, 2 se_agg is the resolution threshold (50% power) and about 2.84 se_agg the benefit resolved with '
            'roughly 80% power (normal approximation, version-mean condition ignored): 0.011-0.016 / 0.015-0.023 at the se_agg '
            '0.0053-0.0081 realized against a close comparator on the input-and-pair grid, 0.015-0.028 / 0.021-0.039 at the '
            '0.0074-0.0139 realized between strong references; D2 needs every gating comparison plus targeting, so its power is '
            'lower; "not separated" is the default reading, never "equal"')

_cache = {}
_SEEDKEY = re.compile(r'S([01])k([0-9])')
_BOOTKEY = re.compile(r'S([01])(R[1-4])')


# ---------------------------------------------------------------- rule file
def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _int_keys(d):
    return {int(k): x for k, x in d.items()} if isinstance(d, dict) else d


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
        g, er, dec, rs = r['reader_grid'], r['er_explanation'], r['decisions'], r['registration_status']
        checks = {
            'z': (r['z'], Z), 'z equals NRM1': (Z, NR.Z), 'tie tolerance': (r['tie_tolerance'], TIE_TOL),
            'recipient limit': (r['recipient_limit_auc'], RECIPIENT_LIMIT),
            'model seed base': (g['model_seed_base'], NR.MODEL_SEED_BASE), 'seed offset': (g['seed_offset'], SEED_OFFSET),
            'forbidden offsets': (tuple(g['forbidden_offsets']), FORBIDDEN_OFFSETS),
            'bootstrap seed base': (g['bootstrap_seed_base'], BOOT_SEED_BASE), 'repetition index': (g['repetition_index'], REP_INDEX),
            'split tag': (g['split_tag'], SPLIT_TAG), 'er recipes': (er['min_leaf'], ER_MIN_LEAF),
            'er pool people': (er['pool_people'], ER_POOL_PEOPLE), 'few-valued max': (er['few_valued_max'], FEW_VALUED_MAX),
            'starts': (rs['starts'], STARTS),
            'start binding': ({k: _int_keys(x) for k, x in dec['UM']['start_binding'].items()}, START_BINDING),
            'guard metrics': (dec['UM']['guard_metrics'], GUARD_METRIC), 'clause metrics': (dec['UM']['clause_metrics'], CLAUSE_METRIC),
            'd1 labels': (tuple(dec['D1']['labels']), D1_LABELS),
            'confidence labels': (tuple(dec['D1']['confidence_labels']), CONFIDENCE_LABELS),
            'condition states': (tuple(dec['D1']['condition_states']), CONDITION_STATES),
            'um labels': (tuple(dec['UM']['labels']), UM_LABELS),
            'd2 comparisons': (tuple(dec['D2']['comparisons']), D2_COMPARISONS), 'd2 met': (dec['D2']['met_label'], D2_MET),
            'ctrl origins': (tuple(dec['D2']['ctrl_origins']), CTRL_ORIGINS),
            'frontier paths': (tuple(tuple(x) for x in dec['D2']['frontier_paths']), FRONTIER_PATHS),
            'selection data': (tuple(dec['D2']['selection_data']['allowed']), SELECTION_DATA),
            'fineness fraction': (dec['D2']['grid_fineness']['fraction'], FINENESS_FRACTION),
            'families': (rs['family_pattern'], FAMILY_RE.pattern),
            'registered families': (tuple(rs['registered_families']), REGISTERED_FAMILIES),
            'd2r labels': (tuple(dec['D2R']['labels']), D2R_LABELS),
            'outcomes': (tuple(r['outcome_class']['labels']), OUTCOMES),
            'adaptive kinds': (tuple(rs['kinds']), ADAPTIVE_KINDS),
            'target references': (tuple(dec['D4']['target_references']), NR.TARGET_REFS),
            'benchmark': (tuple(dec['D3']['references']), NR.BENCHMARK),
        }
        bad = [k for k, (a, b) in checks.items() if a != b]
        if bad:
            raise ValueError(f'RULES.json disagrees with sar.rules on {bad}')
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
    reader seeds, then the mean over repetitions (frc.rules logic). None unless all 8 are finite."""
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
    """Control-path ids that exist, aliases mapped to the saved id, duplicates removed (order kept) (frc.rules logic)."""
    out = []
    for i in ids or []:
        j = (alias_of or {}).get(i, i) if i else None
        if j and j not in out:
            out.append(j)
    return out


def control_path(ids, alias_of=None, origin='START'):
    """The CTRL path of D2: its origin (START's id, or CONT's when the card declares the untargeted operation's zero strength
    there) followed by the distinct saved CTRL points (aliases once; an alias of the origin collapses into it). None if the path
    was not given."""
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


def _components(x):
    """One version's strength as a list of components: a number -> [x]; a list / tuple of numbers -> its entries; else None."""
    if isinstance(x, (list, tuple)):
        return [float(y) for y in x] if x and all(_fin(y) for y in x) else None
    return [float(x)] if _fin(x) else None


def _per_version_components(x):
    if isinstance(x, dict):
        out = {v: _components(_v(x, v)) for v in VERSIONS}
    else:
        out = {v: _components(x) for v in VERSIONS}
    return out if all(c is not None for c in out.values()) else None


def _strength_value(x):
    """The tie-break scalar of a strength: the sum of every component over both versions; missing, malformed or negative -> None."""
    pv = _per_version_components(x)
    if pv is None or any(c < 0 for cs in pv.values() for c in cs):
        return None
    return sum(sum(cs) for cs in pv.values())


def _nonzero_strength(x):
    """Nonzero in every version: every component finite and >= 0, at least one component > 0 in each version."""
    pv = _per_version_components(x)
    return pv is not None and all(all(c >= 0 for c in cs) and any(c > 0 for c in cs) for cs in pv.values())


def _differs_by_version(x):
    if not isinstance(x, dict):
        return False
    a, b = _v(x, 0), _v(x, 1)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) != list(b)
    return a != b


def _sign(x):
    return (x > 0) - (x < 0)


# ---------------------------------------------------------------- (9) reader-grid validity of the registered SAR1 grid
def forbidden_seeds():
    """Every realized earlier grid's reader seeds: MODEL_SEED_BASE + {300, ..., 900} + 10 v + k, k = 0..9 (140 seeds; a local
    equivalent of nrm.rules.forbidden_seeds, which reads its own constant (300, 500); nrm is not monkeypatched)."""
    return {NR.MODEL_SEED_BASE + off + 10 * v + k for off in FORBIDDEN_OFFSETS for v in VERSIONS for k in range(10)}


def registered_seed(v, k):
    return NR.MODEL_SEED_BASE + SEED_OFFSET + 10 * int(v) + int(k)


def registered_boot_seed(v, rep):
    return BOOT_SEED_BASE + 100 * int(v) + REP_INDEX[rep]


def _seed_errors(seeds, required_k):
    """Recorded distinct integer seeds for the required keys S<v>k<k>; every key S<v>k<k> equal to the registered MODEL_SEED_BASE
    + 1000 + 10 v + k; none an earlier grid's seed."""
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
    2026102400 + 100 v + repetition index; no other key form. None -> not recorded (an error)."""
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


def _distinct(c, distinct_inputs):
    n = ((distinct_inputs or {}).get(f"{c['wire']}|V{c['v']}") or {}).get(c['view'])
    return n if isinstance(n, int) and not isinstance(n, bool) else None


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
    """The SAR1 few-valued explanation (prospective; RULES.md 9): the view's recorded distinct-input count is at most
    FEW_VALUED_MAX (320) AND the same recipe demonstrably uses its seed in the same version, repetition and seed pair (a recorded
    check with equal_k0_k1 False on some other wire or view). Such an equality is counted once (n_eff); it never invalidates."""
    n = _distinct(c, distinct_inputs)
    if n is None or not 0 < n <= FEW_VALUED_MAX:
        return False
    return any(d[4] == c['recipe'] and d[2] == c['v'] and d[3] == c['R'] and (d[0], d[1]) != (c['wire'], c['view']) for d in diff)


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
    """One seed-pair group: NRM1 A-SC1 (ii)-(iii) with the SAR1 explanations; nrm.rules.grid_validity is called unchanged (with
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
                'strict_errors': ['malformed seed checks']}
    if len(eq) - base['explained_equalities'] != len(unexp_nrm):
        raise AssertionError('SAR1 classification disagrees with nrm.rules.grid_validity')
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
                    'minimum_set': c['minimum_set'],
                    'distinct_inputs': {f'V{v}': ((distinct_inputs or {}).get(f"{c['wire']}|V{v}") or {}).get(c['view'])
                                        for v in VERSIONS}, 'support_bound': support_bound(c['recipe']),
                    'few_valued_max': FEW_VALUED_MAX}
    return out


def grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, required_k=NR.REG_K, reps=NR.REG_REPS):
    """Reader-seed validity of the registered SAR1 grid, applied prospectively (RULES.md 9). READER_SEEDS_EFFECTIVE iff
    (1) seeds: the required S<v>k<k> recorded, distinct integers, every one equal to MODEL_SEED_BASE + 1000 + 10 v + k, none an
        earlier realized grid's seed (offsets 300, 400, 500, 600, 700, 800, 900);
    (2) bootstrap: S<v><R> recorded for every version and required repetition, equal to 2026102400 + 100 v + repetition index;
    (3) per seed pair (k0 / k1; for the refit also k2 / k3): every seed-dependent recipe checked; every seed-dependent equality
        explained: off the minimum set by NRM1 A-SC1 (iii) or the few-valued explanation; on the minimum set (a D2 reading's CAND,
        START, CONT when declared and the saved CTRL points; for the refit the D2R release and its decisive comparators) by the
        extra-trees support explanation or the few-valued explanation. Explained equalities count once (n_eff) and are reported.
    The strict NRM1 reading (any minimum-set equality invalidates) is reported beside as 'strict_nrm1'."""
    errors = _seed_errors(seeds, required_k) + _boot_errors(boot_seeds, reps)
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
        per[tag] = {'checks': part.get('checks', 0), 'errors': part['errors'],
                    'explained': len(part['explained']), 'explained_few_valued': sum(c['explanation'] == 'FEW_VALUED'
                                                                                     for c in part['explained']),
                    'unexplained_minimum_set': len(part['unexplained_min']), 'unexplained_other': part.get('unexplained_other')}
    seed_errs = _seed_errors(seeds, required_k) + _boot_errors(boot_seeds, reps)
    ok = not errors
    if any(c['explanation'] == 'FEW_VALUED' for c in exp_all):
        flags.add('FEW_VALUED_SEED_EQUALITIES_COUNTED_ONCE')
    return {'label': 'READER_SEEDS_EFFECTIVE' if ok else 'READER_SEEDS_NOT_EFFECTIVE', 'effective': ok, 'errors': errors,
            'per_seed_pair': per, 'flags': sorted(flags),
            'explained_equalities': {
                'count': len(exp_all), 'n_eff_implied': _n_eff_report(exp_all, [c for c in checks or [] if isinstance(c, dict)],
                                                                      distinct_inputs),
                'items': [{k: c.get(k) for k in ('wire', 'view', 'v', 'R', 'recipe', 'k_pair', 'explanation', 'minimum_set')}
                          for c in exp_all],
                'note': 'a reader on a few-valued view can reduce to a seed-invariant ranking; its seed-pair cells count once '
                        '(n_eff) and the comparison is not declared invalid for it; the seed is shown effective elsewhere'},
            'strict_nrm1': {'label': 'READER_SEEDS_NOT_EFFECTIVE' if (seed_errs or strict_errors) else 'READER_SEEDS_EFFECTIVE',
                            'errors': seed_errs + strict_errors},
            'seed_offset': SEED_OFFSET, 'forbidden_offsets': list(FORBIDDEN_OFFSETS), 'required_k': list(required_k),
            'reps': list(reps), 'few_valued_max': FEW_VALUED_MAX,
            'rule': 'SAR1 prospective: NRM1 (iii) off the minimum set, the extra-trees support explanation on it, and the '
                    'few-valued explanation everywhere'}


# ---------------------------------------------------------------- (8) genuinely effective classifier refit
def refit_effectiveness(splits, seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, rob_reps=None):
    """Refit-effectiveness check (frc.rules logic, SAR1 explanations). splits = {'R1'..'R4': {'assignment_sha256': str,
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
    """Each of the 28 conditions from its own household-clustered SE (sqrt(se^2 + se_mc^2) for a randomized release; no floor):
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


def d1(rec):
    """D1 task feasibility at CAND (RULES.md 2). Gating label from G1 (NR.point_eligibility: all 28 at the point, original caps):
    TASK_FEASIBLE, CAND_INFEASIBLE, NO_FEASIBLE_STRENGTH (no nonzero grid strength met the MODEL_FIT budget: no CAND), INCOMPLETE.
    Reported separately and never gating: the confidence conjunction of CAND, START and CONT, the inherited two-SE reserve of each,
    each fitted nonzero strength's point label; flags START_POINT_INELIGIBLE (disagrees with the frozen anchor record) and
    CONT_POINT_INELIGIBLE."""
    rec = rec or {}
    cand = rec.get('cand') or {}
    start = rec.get('start') or {}
    cont = rec.get('cont') or {}
    flags = []
    sl = NR.point_eligibility(start.get('evals') or {})['label'] if start.get('evals') else None
    cl = NR.point_eligibility(cont.get('evals') or {})['label'] if cont.get('evals') else None
    if sl == 'POINT_INELIGIBLE':
        flags.append('START_POINT_INELIGIBLE')
    if cl == 'POINT_INELIGIBLE':
        flags.append('CONT_POINT_INELIGIBLE')
    out = {'start': sl, 'cont': cl, 'cand': None, 'flags': flags}
    if cand.get('status') == 'NO_FEASIBLE_STRENGTH':
        return {**out, 'label': 'NO_FEASIBLE_STRENGTH', 'met': False,
                'statement': 'no nonzero grid strength met the MODEL_FIT selection budget: no candidate exists (no fallback)'}
    if cand.get('status') != 'FITTED':
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'error': 'CAND status missing'}
    cg = NR.point_eligibility(cand.get('evals') or {})
    out['cand'] = cg['label']
    if cg['label'] == 'INCOMPLETE':
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'error': cg.get('error')}
    lab = 'CAND_INFEASIBLE' if cg['label'] == 'POINT_INELIGIBLE' else 'TASK_FEASIBLE'
    strengths, ignored = {}, []
    for s in rec.get('strengths') or []:
        if not _nonzero_strength(s.get('strength')):
            ignored.append(s.get('config_id'))
            continue
        strengths[s.get('config_id')] = NR.point_eligibility(s.get('evals') or {})['label']
    conf = {'cand': confidence_conjunction(cand.get('evals')),
            'start': confidence_conjunction(start.get('evals')) if start.get('evals') else None,
            'cont': confidence_conjunction(cont.get('evals')) if cont.get('evals') else None}
    out.update({'label': lab, 'met': lab == 'TASK_FEASIBLE', 'cand_failed': cg.get('failed'),
                'confidence': {k: (None if x is None else {kk: x.get(kk) for kk in ('label', 'unresolved', 'violations', 'n_pass')})
                               for k, x in conf.items()},
                'confidence_detail_cand': conf['cand'],
                'reserve': {'cand': _reserve(cand.get('evals')), 'start': _reserve(start.get('evals')) if start.get('evals') else None,
                            'cont': _reserve(cont.get('evals')) if cont.get('evals') else None},
                'fitted_strengths': strengths if rec.get('strengths') is not None else None,
                'ignored_zero_strength_entries': ignored,
                'note': 'G1 at the point (gating) is not confidence preservation; the confidence conjunction and the two-SE reserve '
                        'are reported separately, never gating, never redefining eligibility'})
    return out


# ---------------------------------------------------------------- (3) UM usefulness margin against START
def condition_metric(name):
    """The paired-cost metric whose CAND - START difference equals the difference of a condition: the reference (U0 / Ucal) cancels
    in a paired difference on the same people; guard|sex0 (men) -> occupation|men_brier, guard|sex1 (women) ->
    occupation|women_brier; <task>|<ref>|accuracy_loss / nll_excess / brier_excess -> <task>|acc_loss / nll / brier."""
    if name in GUARD_METRIC:
        return GUARD_METRIC[name]
    task, ref, m = name.split('|')
    if task not in ('income', 'occupation') or ref not in ('U0', 'Ucal') or m not in CLAUSE_METRIC:
        raise ValueError(f'unknown condition {name!r}')
    return f'{task}|{CLAUSE_METRIC[m]}'


def binding_condition(evals_by_version):
    """START's binding condition per version: the condition with the smallest two-SE reserve ratio (NR.reserve_status, read one
    version at a time: margin / max(own SE, H_LJ floor SE)), as in the frozen anchor table. ({v: name}, error)."""
    out = {}
    for v in VERSIONS:
        e = _v(evals_by_version, v)
        if e is None:
            return None, f'version {v} missing'
        try:
            r = NR.reserve_status({v: e})
        except (KeyError, ValueError, TypeError) as ex:
            return None, f'version {v}: {ex}'
        b = r.get('binding')
        if not isinstance(b, str) or not b.startswith(f's{v}|'):
            return None, f'version {v}: no binding condition'
        out[v] = b.split('|', 1)[1]
    return out, None


def point_margin(evals_by_version):
    """Whole-release point margin per version: the smallest fraction of its cap that a condition leaves unspent, min over the 14
    conditions of (cap - value) / cap, and the condition attaining it. ({v: {'margin', 'condition'}}, error)."""
    try:
        conds = _conditions(evals_by_version or {})
    except (KeyError, ValueError, TypeError) as ex:
        return None, str(ex)
    out = {}
    for v in VERSIONS:
        fr = {n: (c['cap'] - c['value']) / c['cap'] for n, c in conds[v].items() if _fin(c['value'])}
        if len(fr) != len(conds[v]):
            return None, f'version {v}: non-finite condition value'
        n = min(fr, key=lambda k: fr[k])
        out[v] = {'margin': fr[n], 'condition': n}
    return out, None


def _cost_diffs(costs):
    """Every one of the 8 distinct paired condition differences CAND - START (NR.SECONDARY metrics), per version and pooled (NR's
    pooling: mean of versions, SE = mean of version SEs incl. MC), with resolved_better (pooled upper < 0 and both versions < 0)
    and resolved_worse (pooled lower > 0 and both versions > 0). (record, error)."""
    if not all(isinstance(_v(costs, v), dict) for v in VERSIONS):
        return None, 'paired costs of CAND vs START missing'
    out = {}
    for m in NR.SECONDARY:
        per = {v: _v(costs, v).get(m) for v in VERSIONS}
        if not all(isinstance(x, dict) and _fin(x.get('value'), x.get('se')) for x in per.values()):
            return None, f'paired cost {m} vs START missing or non-finite'
        p = NR._pooled(per)
        if p is None:
            return None, f'paired cost {m} vs START malformed'
        bv = p['by_version']
        out[m] = {'pooled': p['value'], 'se': p['se'], 'by_version': bv,
                  'per_version_se': {v: float(per[v]['se']) for v in VERSIONS},
                  'resolved_better': p['value'] + Z * p['se'] < 0 and all(bv[v] < 0 for v in VERSIONS),
                  'resolved_worse': p['value'] - Z * p['se'] > 0 and all(bv[v] > 0 for v in VERSIONS)}
    return out, None


def recovery_profile(entry, limit=RECIPIENT_LIMIT):
    """CAND's recovery profile retained against START (RULES.md 3): the pair and both single views not more recoverable than
    START's by more than 0.010 AUC per version at the aggregate point (pair: -b_v <= 0.010; single views: NR.recipient_screen
    gate), and no resolved pair worsening (aggregate upper bound of the pair benefit START - CAND below 0). (record, error)."""
    if not isinstance(entry, dict):
        return None, 'grid entry vs START missing'
    a = NR.aggregate(entry.get('pair'))
    if not a.get('complete'):
        return None, f'pair view vs START: {a.get("error")}'
    worse_point = {v: -a['per_version'][v]['mean'] for v in VERSIONS}          # candidate AUC - START AUC
    pair_point_ok = a['alias'] or all(worse_point[v] <= limit for v in VERSIONS)
    pair_resolved_worse = (not a['alias']) and a['upper'] < 0
    views = {}
    for w in NR.SINGLE_VIEWS:
        s = NR.recipient_screen(entry.get(w))
        if s['label'] == 'INCOMPLETE':
            return None, f'{w} view vs START incomplete'
        views[w] = {'label': s['label'], 'gate_pass': s['gate_pass'],
                    'point_by_version': {v: s['aggregate']['per_version'][v]['mean'] for v in VERSIONS}}
    retained = pair_point_ok and not pair_resolved_worse and all(x['gate_pass'] for x in views.values())
    return {'retained': retained, 'pair': {'cand_minus_start_by_version': worse_point, 'point_within_allowance': pair_point_ok,
                                           'resolved_worsening': pair_resolved_worse, 'benefit_mean': a['mean'],
                                           'benefit_se': a['se_agg'], 'upper': a['upper']},
            'single_views': views, 'limit': limit}, None


def _start_of(rec):
    """(START id, error): FA -> SUCC, FB -> NEURAL_RAW (frozen anchor selection); FS_<name> -> a benchmark reference."""
    fam = _family(rec)
    sid = ((rec or {}).get('start') or {}).get('id')
    want = STARTS.get(fam)
    if want is not None and sid != want:
        return None, f'START of {fam} must be {want} (frozen anchor selection), got {sid!r}'
    if want is None and sid not in NR.BENCHMARK:
        return None, f'START of {fam} must be a frozen benchmark reference, got {sid!r}'
    return sid, None


def _cand_gate(rec):
    """Shared gate of UM and D2: (label, extra) or (None, g1 record)."""
    cand = rec.get('cand') or {}
    st = cand.get('status')
    if st == 'NO_FEASIBLE_STRENGTH':
        return 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', {}
    if cand.get('every_nonzero_strength_alias_of_start') is True:
        return 'NOT_ASSESSED_EXACT_ALIAS', {'family_stops': True,
                                            'statement': "every allowed nonzero strength transmits exactly START's vectors: the "
                                                         'operation never changes the output; the family stops'}
    if st != 'FITTED':
        return 'INCOMPLETE', {'error': 'CAND status missing'}
    g1 = NR.point_eligibility(cand.get('evals') or {})
    if g1['label'] == 'INCOMPLETE':
        return 'INCOMPLETE', {'error': f'CAND utility incomplete: {g1.get("error")}'}
    if g1['label'] != 'POINT_ELIGIBLE':
        return 'NOT_ASSESSED_TASK_INFEASIBLE', {'g1_failed': g1.get('failed'),
                                                'statement': 'the candidate fails G1: the comparison is reported as not run (never '
                                                             'negative, never protection)'}
    return None, g1


def um(rec):
    """UM usefulness margin against START (RULES.md 3; defined for every reading, meaningful for FA). Labels in order:
    NOT_ASSESSED_NO_FEASIBLE_STRENGTH; NOT_ASSESSED_EXACT_ALIAS; INCOMPLETE (CAND status / utility); NOT_ASSESSED_TASK_INFEASIBLE
    (CAND fails G1); INCOMPLETE (START record, its binding condition disagreeing with the declared / frozen one, paired costs vs
    START); MARGIN_NOT_RESOLVED (START's binding condition not resolved improved); NOT_COMPUTED_GRID_* (the recovery profile needs
    a valid grid); INCOMPLETE (grid entry vs START); MARGIN_GAIN_PROFILE_NOT_RETAINED; MARGIN_GAIN_SHIFTED_ELSEWHERE (CAND's
    whole-release point margin below START's in a version); MARGIN_RECOVERED. Resolved improvement: the pooled paired difference
    CAND - START of START's binding condition (per version its own binding condition; NR pooling) has upper bound < 0 and both
    version differences < 0. Reported: every paired condition difference, CAND's and START's reserve and confidence states."""
    rec = rec or {}
    lab, extra = _cand_gate(rec)
    if lab:
        return {'label': lab, 'met': False, **extra, 'statement': UM_STATEMENT}
    cand = rec.get('cand') or {}

    def inc(msg):
        return {'label': 'INCOMPLETE', 'met': False, 'error': msg, 'statement': UM_STATEMENT}
    sid, err = _start_of(rec)
    if err:
        return inc(err)
    start = rec.get('start') or {}
    sev = start.get('evals') or {}
    if NR.point_eligibility(sev)['label'] != 'POINT_ELIGIBLE':
        return inc('START utility missing or point-ineligible (the frozen anchors pass all 28 at the point)')
    bind, err = binding_condition(sev)
    if err:
        return inc(f'START binding condition: {err}')
    declared = start.get('binding')
    if not isinstance(declared, dict) or any(_v(declared, v) != bind[v] for v in VERSIONS):
        return inc(f'declared binding condition {declared!r} disagrees with START\'s evaluated record {bind!r} (or is missing)')
    if sid in START_BINDING and _family(rec) in STARTS and any(bind[v] != START_BINDING[sid][v] for v in VERSIONS):
        return inc(f'START\'s binding condition {bind!r} disagrees with the frozen anchor table {START_BINDING[sid]!r}')
    costs = (rec.get('costs_vs') or {}).get(sid)
    diffs, err = _cost_diffs(costs)
    if err:
        return inc(err)
    per = {v: _v(costs, v)[condition_metric(bind[v])] for v in VERSIONS}
    p = NR._pooled(per)
    resolved = p['value'] + Z * p['se'] < 0 and all(p['by_version'][v] < 0 for v in VERSIONS)
    cm, e1 = point_margin(cand.get('evals'))
    sm, e2 = point_margin(sev)
    if e1 or e2:
        return inc(f'point margin: {e1 or e2}')
    kept = all(cm[v]['margin'] >= sm[v]['margin'] - TIE_TOL for v in VERSIONS)
    cc, sc = confidence_conjunction(cand.get('evals')), confidence_conjunction(sev)
    flags = []
    if bind[0] != bind[1]:
        flags.append('BINDING_CONDITION_DIFFERS_BY_VERSION')
    worse = sorted(m for m, x in diffs.items() if x['resolved_worse'])
    flags += [f'CONDITION_RESOLVED_WORSE_{m}' for m in worse]
    if not kept:
        flags.append('WHOLE_RELEASE_POINT_MARGIN_BELOW_START')
    rc, rs = _reserve(cand.get('evals')), _reserve(sev)
    body = {'binding': bind, 'binding_metric': {v: condition_metric(bind[v]) for v in VERSIONS},
            'binding_difference': {'pooled': p['value'], 'se': p['se'], 'upper': p['value'] + Z * p['se'],
                                   'by_version': p['by_version'], 'per_version_se': {v: per[v].get('se') for v in VERSIONS},
                                   'resolved_improvement': resolved},
            'binding_confidence': {'cand': {v: (cc.get('per_version') or {}).get(v, {}).get(bind[v], {}).get('state')
                                            for v in VERSIONS},
                                   'start': {v: (sc.get('per_version') or {}).get(v, {}).get(bind[v], {}).get('state')
                                             for v in VERSIONS}},
            'whole_release': {'point_margin': {'cand': cm, 'start': sm, 'not_below_start': kept},
                              'paired_condition_differences': diffs, 'resolved_worse': worse,
                              'reserve': {'cand': rc, 'start': rs,
                                          'min_ratio_change_by_version': {v: ((rc.get('min_ratio_by_version') or {}).get(v, math.nan)
                                                                              - (rs.get('min_ratio_by_version') or {}).get(v, math.nan))
                                                                          for v in VERSIONS}}},
            'flags': flags, 'statement': UM_STATEMENT}
    prof, perr = (None, None)
    gv = _gate_label(rec.get('grid_valid'))
    if gv is None:
        prof, perr = recovery_profile((rec.get('grid') or {}).get(sid))
    body['recovery_profile'] = prof
    if not resolved:
        return {'label': 'MARGIN_NOT_RESOLVED', 'met': False, **body}
    if gv:
        return {'label': gv, 'met': False, **body}
    if perr:
        return {'label': 'INCOMPLETE', 'met': False, 'error': perr, **body}
    if not prof['retained']:
        lab = 'MARGIN_GAIN_PROFILE_NOT_RETAINED'
    elif not kept:
        lab = 'MARGIN_GAIN_SHIFTED_ELSEWHERE'
    else:
        lab = 'MARGIN_RECOVERED'
    return {'label': lab, 'met': lab == 'MARGIN_RECOVERED', **body}


# ---------------------------------------------------------------- (4) brackets, fineness, D2
def brackets(costs_vs, paths, alias_of=None, origin='START', alt_origin=None):
    """The bracketing control Q* of each frontier path by NR's rule (the cheapest path point whose pooled MODEL_SELECT paired
    total-NLL cost is >= the candidate's, i.e. d = cost(CAND) - cost(point) <= 0; ties keep path order). CTRL path = origin (START's
    id, or CONT's) + its distinct saved points; ALT path = alt_origin (START's id; default: origin) + its saved points. Per key
    ('CTRL', 'ALT'): {'bracket': id | None, 'status': BRACKETED | NO_BRACKET | PATH_NOT_GIVEN | INCOMPLETE, 'path', 'primary'}."""
    out = {}
    for _, key in FRONTIER_PATHS:
        o = origin if key == 'CTRL' else (alt_origin if alt_origin is not None else origin)
        ids = control_path((paths or {}).get(key), alias_of, o)
        if ids is None:
            out[key] = {'bracket': None, 'status': 'PATH_NOT_GIVEN'}
            continue
        prim, err = {}, None
        for i in ids:
            d, _ = NR._primary((costs_vs or {}).get(i))
            if d is None:
                err = f'paired costs vs {i} missing or malformed'
                break
            prim[i] = d['value']
        if err:
            out[key] = {'bracket': None, 'status': 'INCOMPLETE', 'error': err, 'path': ids}
            continue
        above = sorted((i for i in ids if prim[i] <= 0), key=lambda i: -prim[i])
        out[key] = {'bracket': above[0] if above else None, 'status': 'BRACKETED' if above else 'NO_BRACKET', 'path': ids,
                    'primary': prim}
    return out


def selection_data_error(sd, start_in_sample_on_model_fit):
    """None if a path's declared bracket data are allowed: MODEL_FIT_OOF (out-of-fold paired costs), or CALIBRATION (task-label
    paired costs; never SEX) only when START is in-sample on MODEL_FIT (recorded True); never MODEL_SELECT. Else a message."""
    if sd is None:
        return 'NOT_RECORDED'
    if sd not in SELECTION_DATA:
        return 'NOT_ALLOWED'
    if sd == 'CALIBRATION' and start_in_sample_on_model_fit is not True:
        return 'NOT_ALLOWED'
    return None


def grid_fineness(spends, start_in_sample_on_model_fit=None):
    """Fineness of the two saved points of each frontier path on the card's declared selection data (RULES.md 4.4; reported beside
    D2, never gating; computed before any MODEL_SELECT read). spends = {'CTRL' | 'ALT': {'selection_data': 'MODEL_FIT_OOF' |
    'CALIBRATION', v: {'cand': s = CAND's paired total-NLL spend over START on those data, 'lower': spend of the saved lower point
    (the origin = 0.0), 'upper': spend of the saved upper point, 'upper_se': household-clustered SE (same data) of the paired
    difference upper - CAND, 'max': the largest grid spend, optionally 'grid': [{'spend', 'se'}] every grid point with its
    paired-difference SE vs CAND}}}. Per path and version: SELECTION_DATA_NOT_ALLOWED (MODEL_SELECT or another source, or
    CALIBRATION while START is not recorded in-sample on MODEL_FIT); INCOMPLETE (selection data not recorded, or an input missing);
    NOT_APPLICABLE_CAND_NOT_ABOVE_START (s <= 0: the origin is the bracket); CONTROL_PATH_MAY_NOT_BRACKET (max < s + 2 SE);
    SAVED_POINT_RULE_NOT_FOLLOWED (lower > s, upper < s + 2 SE although max reaches it, or, with 'grid', lower not the largest point
    with spend <= s or upper not the smallest with spend >= s + 2 SE); FINE iff lower >= 0.75 s and upper <= s + 2 SE + 0.25 s; else
    COARSE_BELOW / COARSE_ABOVE / COARSE_BOTH. A path given as None reads NOT_GIVEN."""
    out, flags, data = {}, [], {}
    for _, key in FRONTIER_PATHS:
        src = (spends or {}).get(key)
        if src is None:
            out[key] = 'NOT_GIVEN'
            continue
        sd = src.get('selection_data') if isinstance(src, dict) else None
        data[key] = sd
        sde = selection_data_error(sd, start_in_sample_on_model_fit)
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
            'rule': 'on the card\'s declared selection data (MODEL_FIT out-of-fold paired costs, or CALIBRATION paired costs when '
                    'START is in-sample on MODEL_FIT; never MODEL_SELECT), with the SE from the same data: lower saved point >= 0.75 '
                    's and upper saved point <= s + 2 SE + 0.25 s (s = CAND spend over START > 0); reported, never gating, never a '
                    'reason to set a reading aside'}


def _targeting_flags(tg, enhanced):
    rep, flags = {}, []
    for q in tg:
        x = _point_positive((enhanced or {}).get(q))
        rep[q] = x
        if x is not None and not x['pass']:
            flags.append(f'READER_SENSITIVE_ENHANCED_{q}')
    return rep, flags


def _frontier_part(key, rec, point, grid, gate_of, origin_id, alias_of):
    """One frontier comparison of D2 (c) (key CTRL or ALT): (record, error, flags)."""
    ids = control_path((rec.get('paths') or {}).get(key), alias_of, origin_id)
    flags = [f'{key}_AT_ORIGIN'] if ids == [origin_id] else []
    fr = NR.frontier_comparison(point, [{'id': i} for i in ids])
    if fr['label'] == 'INCOMPLETE':
        return None, f'{key}_FRONTIER: {fr.get("error")}', flags
    q = fr.get('bracketing_control')
    gate, rcp = None, None
    if q is not None:
        gate, cq = gate_of(q)
        if gate is None:
            return None, f'{key}_FRONTIER: recipient views vs bracketing control {q} missing or incomplete', flags
        rcp = {w: cq['recipient'][w]['label'] for w in NR.SINGLE_VIEWS}
    prim = fr.get('primary_cost_differences') or {}
    dominated = []
    for i in ids:
        pv = (prim.get(i) or {}).get('value')
        if _fin(pv) and pv > 0:                       # the point is cheaper than CAND
            neg = _resolved_negative((grid.get(i) or {}).get('pair'))
            if neg is None:
                return None, f'{key}_FRONTIER: grid vs cheaper control point {i} missing', flags
            if neg:
                dominated.append(i)
    ok = fr['label'] == 'DOMINATES_CONTROL_FRONTIER' and gate is True and not dominated
    reason = fr.get('reason')
    label = fr['label']
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
                                'note': 'cost(CAND) - cost(Q*) <= 0: Q* spends at least this much more total NLL than CAND'}
                               if q is not None else None),
                'smallest_overspend_vs_cheaper_point': (min(over.values()) if over else None),
                'interpolation': fr.get('interpolation')}
    return ({'label': label, 'frontier_label': fr['label'], 'reason': reason, 'path': ids, 'point': q,
             'recipient_gate_vs_bracket': gate, 'recipient_vs_bracket': rcp, 'dominated_by_cheaper': dominated, 'pass': ok,
             'gating': True, 'benefit_mean': fr.get('benefit_mean'), 'benefit_se': fr.get('benefit_se'),
             'residual_spend_mismatch': mismatch, 'statement': NO_BRACKET_STATEMENT.format(key=key) if no_bracket else None,
             'frontier': fr}, None, flags)


def _alias_map(rec, sid, cid):
    return {k: (sid if x == 'START' else (cid if x == 'CONT' else x)) for k, x in (rec.get('alias_of') or {}).items()}


def d2(rec):
    """D2 component gain at CAND (RULES.md 4). Labels in order of precedence: NOT_ASSESSED_NO_FEASIBLE_STRENGTH;
    NOT_ASSESSED_EXACT_ALIAS (every nonzero strength transmits START's vectors: the family stops); INCOMPLETE (CAND status or
    utility); NOT_ASSESSED_TASK_INFEASIBLE (CAND fails G1); NOT_COMPUTED_GRID_VALIDITY_UNKNOWN / NOT_COMPUTED_GRID_INVALID;
    INCOMPLETE (START not the frozen anchor of the family; cont_declared not a boolean; alt_declared recorded but not a boolean;
    ctrl_origin malformed or CONT without a declared CONT); NOT_ASSESSED_CONTROLS_NOT_RUN (a gating control recorded as not run for
    budget); INCOMPLETE (a gating frontier path's selection data not recorded or not allowed: MODEL_FIT_OOF, or CALIBRATION when
    START is in-sample on MODEL_FIT, never MODEL_SELECT; any needed input missing); then the conjuncts (a) G1 (passed above); (b)
    START: NR.comparison passes (AGG_POSITIVE pair benefit + recipient gate); (b') CONT, when declared: the same; (c) each gating
    frontier path, CTRL_FRONTIER (origin + saved CTRL points) and, when declared, ALT_FRONTIER (START + saved ALT points: the
    starting release's own protection operation continued to matched spend): NR.frontier_comparison = DOMINATES_CONTROL_FRONTIER,
    the recipient gate vs Q*, no cheaper path point resolved less recoverable than CAND (a missing bracket reads NOT_SEPARATED,
    reason NO_BRACKET); (d) reader targeting: the non-table winner's point benefit > 0 in both version means vs START, CONT (when
    gating) and every gating Q*. Result: COMPONENT_GAIN_BEYOND_START_AND_CONTROLS (all gating comparisons and d); READER_TARGETED
    (every gating comparison, d fails); NOT_SEPARATED (none passes); else PARTIAL_<passed>_ONLY (order START, CONT, CTRL_FRONTIER,
    ALT_FRONTIER). A CONT or ALT path given without a declaration is reported beside, never gating. No partial outcome is a claim."""
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
    cd = rec.get('cont_declared')
    if not isinstance(cd, bool):
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'cont_declared must be recorded as a boolean (card)'}
    ad = rec.get('alt_declared', False)
    if not isinstance(ad, bool):
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'alt_declared, when recorded, must be a boolean (card)'}
    origin = rec.get('ctrl_origin', 'START')
    if origin not in CTRL_ORIGINS or (origin == 'CONT' and not cd):
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'ctrl_origin must be START, or CONT for a declared CONT (card)'}
    cid = (rec.get('cont') or {}).get('config_id') or 'CONT'
    paths = rec.get('paths') or {}
    grid, nr = rec.get('grid') or {}, rec.get('controls_not_run') or {}
    gating = ('START',) + (('CONT',) if cd else ()) + ('CTRL_FRONTIER',) + (('ALT_FRONTIER',) if ad else ())
    gating_paths = ['CTRL'] + (['ALT'] if ad else [])
    not_run = ([k for k in (['CONT'] if cd else []) if cid not in grid and nr.get('CONT') is True] +
               [k for k in gating_paths if paths.get(k) is None and nr.get(k) is True])
    if not_run:
        return {'label': 'NOT_ASSESSED_CONTROLS_NOT_RUN', 'met': False, 'not_run': not_run,
                'statement': 'a matched control was not run (budget): the comparison is reported as not run'}
    flags, comps = [], {}

    def incomplete(msg):
        return {'label': 'INCOMPLETE', 'met': False, 'error': msg, 'comparisons': comps, 'flags': sorted(set(flags))}
    sel = rec.get('selection_data') or {}
    for k in gating_paths:
        e = selection_data_error(sel.get(k), rec.get('start_in_sample_on_model_fit'))
        if e == 'NOT_RECORDED':
            return incomplete(f'the {k} bracket\'s selection data are not recorded (plan: MODEL_FIT_OOF or CALIBRATION)')
        if e == 'NOT_ALLOWED':
            flags.append(f'CONTROL_SELECTION_DATA_NOT_ALLOWED_{k}')
            return incomplete(f'the {k} bracket was selected on {sel.get(k)!r}: only MODEL_FIT out-of-fold paired costs, or '
                              'CALIBRATION paired costs when START is in-sample on MODEL_FIT, are allowed (never MODEL_SELECT); '
                              'the control must be re-selected (a POST HOC repair)')
        if sel.get(k) == 'CALIBRATION':
            flags.append(f'SELECTION_DATA_CALIBRATION_{k}')

    def gate_of(q):
        c = NR.comparison(grid.get(q) or {})
        if c['benefit']['label'] == 'INCOMPLETE' or any(c['recipient'][w]['label'] == 'INCOMPLETE' for w in NR.SINGLE_VIEWS):
            return None, c
        return c['recipient_gate_pass'], c

    def simple(name, q, gating_):
        gt, ct = gate_of(q)
        if gt is None:
            return None
        rec_ = {'benefit': ct['benefit']['label'], 'benefit_mean': ct['benefit']['aggregate'].get('mean'),
                'benefit_se': ct['benefit']['aggregate'].get('se_agg'), 'recipient_gate_pass': gt,
                'recipient': {w: ct['recipient'][w]['label'] for w in NR.SINGLE_VIEWS}, 'point': q,
                'pass': bool(ct['benefit_positive'] and gt), 'gating': gating_}
        if ct['benefit']['label'] == 'ALIAS_ZERO':
            flags.append(f'CAND_ALIAS_OF_{name}')
        return rec_
    flags += list(((rec.get('fineness') or {}).get('flags')) or [])
    if sid not in grid:
        return incomplete(f'grid entry vs START ({sid}) missing')
    c = simple('START', sid, True)
    if c is None:
        return incomplete(f'grid entry vs START ({sid}) incomplete')
    comps['START'] = c
    if cid in grid:
        c = simple('CONT', cid, cd)
        if c is None:
            return incomplete(f'grid entry vs CONT ({cid}) incomplete')
        if not cd:
            c = {**c, 'passed_if_it_were_gating': c['pass'], 'pass': None}
            flags.append('CONT_REPORTED_BESIDE_NOT_DECLARED')
            if not c['passed_if_it_were_gating']:
                flags.append('CONT_NOT_SEPARATED_REPORTED_BESIDE')
        comps['CONT'] = c
    elif cd:
        return incomplete(f'grid entry vs the declared CONT ({cid}) missing (not recorded as not run)')
    point = {'id': cand.get('config_id', 'CAND'), 'costs_vs': rec.get('costs_vs') or {}, 'grid_vs': grid}
    amap = _alias_map(rec, sid, cid)
    for name, key in FRONTIER_PATHS:
        o = (sid if origin == 'START' else cid) if key == 'CTRL' else sid
        if key in gating_paths:
            if paths.get(key) is None:
                return incomplete(f'{key} path missing (not recorded as not run)')
            c, err, f = _frontier_part(key, rec, point, grid, gate_of, o, amap)
            flags += f
            if err:
                return incomplete(err)
            comps[name] = c
        elif paths.get(key) is not None:              # an ALT path given without a declaration: reported beside, never gating
            c, err, _ = _frontier_part(key, rec, point, grid, gate_of, o, amap)
            comps[name] = ({**c, 'gating': False, 'pass': None, 'passed_if_it_were_gating': c['pass']} if c else
                           {'label': 'INCOMPLETE', 'error': err, 'gating': False, 'pass': None})
            flags.append(f'{key}_REPORTED_BESIDE_NOT_DECLARED')
            if c and not c['pass']:
                flags.append(f'{key}_NOT_SEPARATED_REPORTED_BESIDE')
    tg = {}
    for q in dict.fromkeys(x['point'] for x in comps.values() if x.get('point') and x.get('gating')):
        x = _point_positive((rec.get('nolookup') or {}).get(q))
        if x is None:
            return incomplete(f'reader-targeting input (non-table winner) vs {q} missing')
        tg[q] = x
    passed = []
    for n in D2_COMPARISONS:
        x = comps.get(n)
        if not x or not x.get('gating'):
            continue
        x['targeting_pass'] = tg[x['point']]['pass'] if x.get('point') in tg else None
        if x['pass']:
            passed.append(n)
            if x['targeting_pass'] is False:
                flags.append(f'READER_TARGETED_{n}')
    enh, eflags = _targeting_flags(tg, rec.get('enhanced'))
    flags += eflags
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
    scope = D2_SCOPE_ALT if ad else D2_SCOPE
    statement = (scope if label == D2_MET else
                 'every gating comparison passed but the non-table winner does not show a positive point benefit in both versions '
                 'against every comparison point: the gain may be specific to the selected reader; not a component gain'
                 if label == 'READER_TARGETED' else
                 f'passed: {", ".join(passed) or "none"}; not passed: {", ".join(failed)}; a partial outcome is not a component '
                 'gain' + ('; without separation from the CTRL frontier the gain may be explained by task spending' if
                           'CTRL_FRONTIER' in failed else '') +
                 ('; without separation from the ALT frontier the gain may be explained by running the starting release\'s own '
                  'protection operation further (spending more of its margin)' if 'ALT_FRONTIER' in failed else '') +
                 ('; without separation from CONT the gain may be explained by retraining alone' if 'CONT' in failed else '') +
                 ('; not lower than START' if 'START' in failed else ''))
    return {'label': label, 'met': label == D2_MET, 'passed': passed, 'failed': failed, 'gating': list(gating),
            'comparisons': comps, 'targeting': tg, 'enhanced_reported_only': enh, 'flags': sorted(set(flags)),
            'minimum_detectable': MDE_NOTE, 'scope': scope, 'statement': statement, 'start': sid,
            'cont': cid if cd else None, 'ctrl_origin': origin, 'alt_declared': ad,
            'selection_data': {k: sel.get(k) for k in gating_paths},
            'note': 'no partial outcome is a claim; "not separated" is the default reading; every sentence quoting this label '
                    'carries its scope'}


# ---------------------------------------------------------------- (5) D3, D4 with the reader-budget credit rule (frc logic)
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


# ---------------------------------------------------------------- (7) registered vs adaptive readings
def _family(rec):
    """The reading's family: FA, FB or FS_<name> (letters and digits); anything else (CTL included: shared controls are not a
    family reading) is refused."""
    fam = (rec or {}).get('family')
    if not isinstance(fam, str) or not FAMILY_RE.fullmatch(fam):
        raise ValueError(f'unknown family {fam!r}: a family reading is FA, FB or FS_<name>')
    return fam


def adaptive_tag(rec):
    """None for a registered reading: revision 0 of FA / FB whose card was committed before any MODEL_SELECT outcome of the
    campaign. Otherwise '<KIND>_<parts>': KIND = REVISED if the card was committed before any MODEL_SELECT outcome, else ADAPTIVE;
    parts = FS (replacement family), R<n> (revision n >= 1), R0_LATE_CARD (a revision-0 card committed after an outcome or not
    recorded)."""
    fam = _family(rec)
    n = int((rec or {}).get('revision') or 0)
    fs = fam.startswith('FS_')
    early = (rec or {}).get('card_committed_before_any_model_select_outcome') is True
    if n == 0 and fam in REGISTERED_FAMILIES and early:
        return None
    parts = (['FS'] if fs else []) + ([f'R{n}'] if n else ([] if fs else ['R0_LATE_CARD']))
    return f"{'REVISED' if early else 'ADAPTIVE'}_{'_'.join(parts)}"


def _mark(out, tag):
    for k in ('D1', 'UM', 'D2', 'D3', 'D4', 'D5', 'D6', 'outcome_class'):
        x = out.get(k)
        if isinstance(x, dict) and 'label' in x:
            x['label_unprefixed'] = x['label']
            if tag:
                x['label'] = f'{tag}__{x["label"]}'


def _d5(rec, cand):
    """D5 = NRM1 G6 unchanged (rec['joint'] passed to NR.claims), except INCOMPLETE when a declared joint design names all three
    comparators but a grid entry (pair and both recipient views) or paired cost of one is missing; sequential orders that collapse
    BY CONSTRUCTION (joint['collapsed'] = {order: 'SEPARATE' | 'JOINT'}, card) give the reduced test (separate) or
    NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL (joint); malformed declarations are INCOMPLETE. No joint design declared ->
    NOT_APPLICABLE_NOT_JOINT (not claimed)."""
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
                    'statement': 'the joint design equals a sequential order by construction (card): no joint-design advantage '
                                 'over sequential design can exist; no joint-design claim'}
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


# ---------------------------------------------------------------- (6) decisions, the three outcomes and the outcome class
def decisions(rec):
    """D1, UM, D2-D6, the three separately reported outcomes and the outcome class of one family reading (CAND of one card)
    (RULES.md 2-7). D3-D5 are read at CAND and carry D2's gate label when D2 is not assessed for feasibility, aliasing or grid
    validity. Outcome class, first applicable: TASK_INFEASIBLE; SELECTED_STRENGTH_INFEASIBLE; INCOMPLETE (D1 incomplete; D2
    incomplete or not computed); NO_OUTPUT_CHANGE (exact alias of START); PROTECTION_NOT_ASSESSED (a gating control not run);
    INCOMPLETE (D3 undecided); METHOD_CANDIDATE (D2 met and D3 met; D4 reported separately, never implied);
    COMPONENT_GAIN_NOT_COMPETITIVE (D2 met, D3 not); REFERENCE_ADVANTAGE_NOT_ATTRIBUTED (D3 met, D2 not: lower recovery than every
    reference, not separated from START's controls); USEFULNESS_MARGIN_ONLY (UM MARGIN_RECOVERED only: preparatory);
    INCOMPLETE (UM undecided); FEASIBLE_NO_RELIABLE_GAIN. D4, D5 and D2R never change the class. A non-registered reading has every
    label prefixed '<tag>__'."""
    rec = rec or {}
    _family(rec)
    cand = rec.get('cand') or {}
    out = {'family': rec.get('family'), 'card': rec.get('card'), 'revision': int(rec.get('revision') or 0),
           'start': ((rec.get('start') or {}).get('id')), 'D1': d1(rec), 'UM': um(rec), 'D2': d2(rec)}
    d2lab = out['D2']['label']
    cg = NR.point_eligibility(cand.get('evals') or {})['label'] if cand.get('status') == 'FITTED' else None
    gates = ('NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'NOT_ASSESSED_EXACT_ALIAS', 'NOT_ASSESSED_TASK_INFEASIBLE',
             'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID')
    if d2lab in gates:
        for k in ('D3', 'D4', 'D5'):
            out[k] = {'label': d2lab, 'met': False}
    elif cand.get('status') != 'FITTED' or cg != 'POINT_ELIGIBLE':
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
                  'headline_caveat': (r2 or {}).get('headline_caveat')}
    flags = list(out['D2'].get('flags') or []) + list(rec.get('mc_flags') or []) + list(out['D5'].get('flags') or []) + \
        list(out['D1'].get('flags') or []) + list(out['UM'].get('flags') or [])
    d1lab, d3lab, umlab = out['D1']['label'], out['D3']['label'], out['UM']['label']
    if d2lab == 'NOT_ASSESSED_EXACT_ALIAS':
        flags.append('FAMILY_STOPS_EXACT_ALIAS')
    if _differs_by_version(cand.get('strength')):
        flags.append('CAND_STRENGTH_DIFFERS_BY_VERSION')
    why = None
    d2met, d3met = bool(out['D2'].get('met')), bool(out['D3'].get('met'))
    undecided = ('INCOMPLETE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID')
    if d1lab == 'NO_FEASIBLE_STRENGTH':
        oc = 'TASK_INFEASIBLE'
        flags.append('NO_STRENGTH_MEETS_MODEL_FIT_BUDGET')
    elif d1lab == 'CAND_INFEASIBLE':
        fs = out['D1'].get('fitted_strengths')
        if fs is None:
            oc, why = 'INCOMPLETE', 'CAND fails G1 and the fitted strengths are not recorded'
        else:
            others = {k: x for k, x in fs.items() if k != cand.get('config_id')}
            if any(x == 'POINT_ELIGIBLE' for x in others.values()):
                oc = 'SELECTED_STRENGTH_INFEASIBLE'
            elif all(x == 'POINT_INELIGIBLE' for x in others.values()):
                oc = 'TASK_INFEASIBLE'
            else:
                oc, why = 'INCOMPLETE', 'a fitted strength has incomplete utility'
    elif d1lab == 'INCOMPLETE' or d2lab in undecided:
        oc, why = 'INCOMPLETE', f'D1 {d1lab}, D2 {d2lab}'
    elif d2lab == 'NOT_ASSESSED_EXACT_ALIAS':
        oc = 'NO_OUTPUT_CHANGE'
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
    elif umlab == 'MARGIN_RECOVERED':
        oc = 'USEFULNESS_MARGIN_ONLY'
    elif umlab in undecided:
        oc, why = 'INCOMPLETE', f'UM {umlab}'
    else:
        oc = 'FEASIBLE_NO_RELIABLE_GAIN'
    if d3met and not d2met:
        flags.append('D3_WITHOUT_D2')
    if out['D4'].get('met') and not d2met:
        flags.append('D4_WITHOUT_D2')
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
    tag = adaptive_tag(rec)
    out['outcome_class'] = {'label': oc, 'flags': sorted(set(flags)), 'reason': why,
                            'note': 'development evidence on adaptively used MODEL_SELECT records, never confirmation; no class '
                                    'renames a partial result as the desired method; a near-limit point estimate is never called '
                                    'robust'}
    out['registered'] = tag is None
    out['adaptive_tag'] = tag
    out['registration_note'] = ('registered reading (revision 0 of FA or FB, card committed before any MODEL_SELECT outcome)'
                                if tag is None else 'NOT a registered result: ' + tag + ' (labels prefixed; report beside the '
                                                    'registered reading with the number of looks)')
    _mark(out, tag)
    o = {'1_component_gain': {'label': out['D2']['label'], 'met': d2met, 'robustness': out['D2R']['label'],
                              'scope': D2_SCOPE if d2met else out['D2'].get('statement')},
         '2_strong_references': {'label': out['D3']['label'], 'met': d3met, 'passed': out['D3'].get('passed', []),
                                 'failed': out['D3'].get('failing', [])},
         '3_material_target': {'label': out['D4']['label'], 'met': bool(out['D4'].get('met'))},
         'usefulness_margin_preparatory': {'label': out['UM']['label'], 'met': bool(out['UM'].get('met'))}}
    o['statement'] = (f"(1) component gain: {o['1_component_gain']['label']} (D2R {o['1_component_gain']['robustness']}); "
                      f"(2) strong references: {o['2_strong_references']['label']} (passed {o['2_strong_references']['passed']}, "
                      f"failed {o['2_strong_references']['failed']}); (3) material target: {o['3_material_target']['label']}; "
                      f"usefulness margin (preparatory only): {o['usefulness_margin_preparatory']['label']}"
                      + (f"; {out['D2R']['headline_caveat']}" if out['D2R']['headline_caveat'] else ''))
    out['outcomes'] = o
    out['scope'] = list(NR.SCOPE) + ['Point feasibility and the reserve are not confidence preservation.',
                                     'A revision or replacement-family reading is never the registered result.',
                                     'A usefulness-margin gain alone is preparatory evidence, never a protection or competitive '
                                     'claim.',
                                     'The three outcomes (component gain, strong references, material target) are reported '
                                     'separately; none implies another.']
    return out


def _raw(x):
    return (x or {}).get('label_unprefixed', (x or {}).get('label'))


def campaign_summary(decision_records):
    """The registered headline (one registered reading per family; a second one is refused), every non-registered reading with
    its tag, the number K of D2 looks, the readings with D2, D3, D4 met and UM MARGIN_RECOVERED, and the multiplicity statement
    every campaign-level sentence must carry. Flag D2_MET_ONLY_IN_NON_REGISTERED_LOOKS."""
    reg, adaptive, looks, met, d3m, d4m, umm = {}, [], 0, [], [], [], []
    for d in decision_records or []:
        fam = d.get('family')
        lab = _raw(d.get('D2')) or ''
        assessed = lab in D2_ASSESSED or lab.startswith('PARTIAL_')
        looks += assessed
        row = {'family': fam, 'card': d.get('card'), 'revision': d.get('revision'), 'tag': d.get('adaptive_tag'),
               'D2': lab, 'D3': _raw(d.get('D3')), 'D4': _raw(d.get('D4')), 'UM': _raw(d.get('UM')),
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
        if (d.get('UM') or {}).get('met'):
            umm.append(row)
    k0 = sum(r['d2_assessed'] for r in reg.values())
    m0 = sum(1 for r in met if r['tag'] is None)
    flags = ['D2_MET_ONLY_IN_NON_REGISTERED_LOOKS'] if met and not m0 else []
    n = len(reg) + len(adaptive)
    return {'registered': reg, 'non_registered': adaptive, 'd2_looks': looks, 'd2_met': met, 'd3_met': d3m, 'd4_met': d4m,
            'um_margin_recovered': umm, 'flags': flags,
            'statement': (f'D2 was met in {m0} of {k0} registered family readings and in {len(met) - m0} of {looks - k0} '
                          f'non-registered readings ({looks} D2 looks in total); D3 was met in {len(d3m)} and D4 in {len(d4m)} of '
                          f'{n} readings; UM MARGIN_RECOVERED (preparatory only) in {len(umm)}; {PER_LOOK_FALSE_PASS}.')}


# ---------------------------------------------------------------- (8) strongest feasible nonzero release; D2R
def strongest_nonzero(configs):
    """The strongest feasible nonzero release of a family (frc.rules logic; descriptive). configs = [{'config_id', 'strength',
    'revision', 'start_id', 'evals', 'pair_vs_start': NR grid pair entry vs START, 'alias_of'}] of the family's OPERATION
    (controls excluded). Eligible: nonzero strength in every version, G1, not an alias, pair benefit vs START neither ALIAS_ZERO nor
    INCOMPLETE. Choice: the largest registered-cell aggregate pair benefit vs START; |dx| <= 1e-12 ties to the smaller strength,
    then the lower revision, then config_id. Chosen on registered MODEL_SELECT cells (winner's curse); never changes D1-D6."""
    elig, rows = [], []
    for c in configs or []:
        r = {'config_id': c.get('config_id'), 'strength': c.get('strength'), 'revision': int(c.get('revision') or 0)}
        if not _nonzero_strength(c.get('strength')) or _strength_value(c.get('strength')) is None:
            r['status'] = 'NOT_NONZERO'
        elif c.get('alias_of'):
            r['status'] = f'ALIAS_OF_{c["alias_of"]}'
        elif NR.point_eligibility(c.get('evals') or {})['label'] != 'POINT_ELIGIBLE':
            r['status'] = 'NOT_TASK_FEASIBLE'
        else:
            b = NR.benefit_screen(c.get('pair_vs_start'))
            r['benefit_label'] = b['label']
            if b['label'] in ('ALIAS_ZERO', 'INCOMPLETE'):
                r['status'] = 'ALIAS_OF_START' if b['label'] == 'ALIAS_ZERO' else 'INCOMPLETE_GRID'
            else:
                r.update({'status': 'ELIGIBLE', 'x': b['aggregate']['mean'], 'start_id': c.get('start_id')})
                elig.append(r)
        rows.append(r)
    if not elig:
        return {'label': 'NO_FEASIBLE_NONZERO_RELEASE', 'selected': None, 'records': rows}
    best = max(r['x'] for r in elig)
    near = sorted((r for r in elig if r['x'] >= best - TIE_TOL),
                  key=lambda r: (_strength_value(r['strength']), r['revision'], str(r['config_id'])))
    starts = {r.get('start_id') for r in elig}
    return {'label': 'SELECTED', 'selected': near[0]['config_id'], 'config_id': near[0]['config_id'], 'x': near[0]['x'],
            'revision': near[0]['revision'], 'tied': [r['config_id'] for r in near], 'records': rows,
            'flags': ['MIXED_STARTS'] if len(starts) > 1 else [],
            'note': 'chosen on registered MODEL_SELECT cells (winner\'s curse); descriptive; never changes D1-D6'}


def d2r(rec):
    """D2R robustness to classifier refitting (RULES.md 8; descriptive; never changes D2). rec = {'grid_valid', 'release': the
    D2R release (CAND by default), 'refit': refit_effectiveness(...) record, 'rob': {comparator: NR grid entry of the release vs
    it, registered and robustness cells MERGED}, 'comparators': {'START': id, 'CONT': id | None (not gating), 'CTRL': id | None
    (no bracket) | a brackets() record, 'ALT': the same (required iff rec['alt_declared'] is True)}} (the registered D2 brackets;
    never re-chosen on robustness cells). Per decisive
    comparator: ROBUSTNESS_CONFIRMED iff NR.robustness is ROBUSTNESS_CONFIRMED (AGG_POSITIVE on the robustness cells and on all
    cells), the registered cells are present and the refit is REFIT_GENUINELY_EFFECTIVE; NOT_RUN if the entry is missing; else
    ROBUSTNESS_NOT_CONFIRMED. Family label over the decisive set: CONFIRMED iff all; NOT_RUN if any NOT_RUN; else NOT_CONFIRMED.
    A sign change (robustness-cell aggregate mean against the registered aggregate mean) on any comparator, decisive or beside, is
    flagged and must be carried into the headline (headline_caveat). The other references present are reported beside."""
    rec = rec or {}
    why = _gate_label(rec.get('grid_valid'))
    if why:
        return {'label': why}
    comp = rec.get('comparators')
    if not isinstance(comp, dict) or any(k not in comp for k in ('START', 'CONT', 'CTRL')) or not comp.get('START'):
        return {'label': 'INCOMPLETE', 'reason': 'decisive comparators not recorded (START id, CONT id or None, CTRL bracket id, '
                                                 'None or a brackets() record)'}
    ad = rec.get('alt_declared', False)
    if not isinstance(ad, bool) or (ad and 'ALT' not in comp):
        return {'label': 'INCOMPLETE', 'reason': 'alt_declared must be a boolean, and a declared ALT needs its bracket recorded '
                                                 '(id, None or a brackets() record)'}
    br = {}
    for k in ('CTRL',) + (('ALT',) if ad else ()):
        x = comp[k]
        if isinstance(x, dict):
            if x.get('status') == 'INCOMPLETE':
                return {'label': 'INCOMPLETE', 'reason': f'{k} bracket record INCOMPLETE'}
            x = x.get('bracket')
        br[k] = x
    refit_ok = (rec.get('refit') or {}).get('label') == 'REFIT_GENUINELY_EFFECTIVE'
    decisive = list(dict.fromkeys([x for x in (comp['START'], comp.get('CONT'), br['CTRL'], br.get('ALT')) if x]))
    beside = [r for r in NR.BENCHMARK if r not in decisive and r in (rec.get('rob') or {})]
    beside += [r for r in NR.TARGET_REFS if r not in decisive and r not in beside]
    per, signs = {}, []
    for c in decisive + beside:
        e = (rec.get('rob') or {}).get(c)
        if not e:
            per[c] = {'label': 'NOT_RUN'}
            continue
        r = NR.robustness(e)
        if r['label'] == 'NOT_RUN':
            per[c] = {'label': 'NOT_RUN'}
            continue
        cells = ((e.get('pair') or {}).get('cells') or {})
        has_reg = all(x in cells for x in NR.REGISTERED)
        regm = NR.aggregate(e.get('pair'), 'registered').get('mean') if has_reg else None
        rob = r['robustness_cells']['aggregate']
        sc = _fin(regm, rob.get('mean')) and _sign(regm) != 0 and _sign(rob['mean']) != 0 and _sign(regm) != _sign(rob['mean'])
        if sc:
            signs.append(c)
        reasons = ([] if r['label'] == 'ROBUSTNESS_CONFIRMED' else ['NOT_AGG_POSITIVE_ON_ROBUSTNESS_OR_ALL_CELLS']) + \
                  ([] if refit_ok else ['REFIT_NOT_GENUINELY_EFFECTIVE']) + ([] if has_reg else ['REGISTERED_CELLS_ABSENT'])
        per[c] = {'label': 'ROBUSTNESS_CONFIRMED' if not reasons else 'ROBUSTNESS_NOT_CONFIRMED', 'reasons': reasons,
                  'decisive': c in decisive, 'nrm_label': r['label'], 'registered_mean': regm,
                  'robustness_mean': rob.get('mean'), 'robustness_se': rob.get('se_agg'),
                  'n_eff_robustness_cells': rob.get('n_eff_distinct_cells'), 'reader_seed_qualifier': rob.get('reader_seed_qualifier'),
                  'all_cells_label': r['all_cells']['label'], 'sign_change': sc}
    labs = [per[c]['label'] for c in decisive]
    fam = ('NOT_RUN' if any(x == 'NOT_RUN' for x in labs) else
           'ROBUSTNESS_CONFIRMED' if all(x == 'ROBUSTNESS_CONFIRMED' for x in labs) else 'ROBUSTNESS_NOT_CONFIRMED')
    cav = (f'the pair benefit changes sign on the robustness cells against {", ".join(signs)}: carried into the headline'
           if signs else None)
    return {'label': fam, 'release': rec.get('release', 'CAND'), 'decisive': decisive, 'beside': beside, 'per_comparator': per,
            'sign_changes': signs, 'headline_caveat': cav, 'flags': [f'SIGN_CHANGE_{c}' for c in signs],
            'refit': (rec.get('refit') or {}).get('label'), 'counted_once': (rec.get('refit') or {}).get('counted_once'),
            'note': 'descriptive; never changes D2; no retuning on the robustness cells (CAND, its brackets and every selection are '
                    'the registered ones); repeated reader splits and reader seeds on the same people are not replication'}
