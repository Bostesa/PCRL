"""Frozen decision rules of the PCRL dual-design release campaign (`pcrl_dual_design_release_v1`; statistical reviewer; written
before any outcome of this campaign existed; no label read).

Human-readable rules: results/pcrl_dual_design_release_v1/roles/statistical_reviewer/RULES.md
Machine-readable constants: .../RULES.json (schema DDR1-rules-v1); `load_rules()` refuses a RULES.json that disagrees with this
module, a changed NRM1 rule file or a changed nrm/rules.py.
Invented-data tests: `python3 -m ddr.test_rules` (prints JSON; exit 0 iff every test passes).

Pure functions on plain dicts / floats. Nothing here reads data or labels. The inherited machinery is `nrm.rules`, imported
UNCHANGED as NR and never monkeypatched: point_eligibility (G1), reserve_status, the aggregate screen (with NR's mc_cell / se_mc
handling for randomized releases), benefit_screen, comparison (benefit + recipient gate), frontier_comparison (the spend-bracketing
rule), robustness, claims (G4 / G5 / G6) and alias handling. Logic copied from frc.rules (frc is not imported): the reader-budget
credit rule for references (D3 / D4), the control-path alias collapse, the non-table winner's version means, strongest_nonzero,
refit_effectiveness and the seed part of grid validity with FRC1's ER1 explanation, here applied PROSPECTIVELY and restricted to
extra-trees recipes on bounded-support views (RULES.md section 9).

API (RULES.md section in brackets):
  load_rules()                                     constants + '_sha256'                                                   [0]
  forbidden_seeds(), registered_seed(v, k), registered_boot_seed(v, rep)                                                   [9]
  grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=None)       validity of the registered DDR1 grid   [9]
  refit_effectiveness(splits, seeds, checks, minimum_set, distinct_inputs, boot_seeds=None)   genuinely effective refit    [8]
  brackets(costs_vs, paths, alias_of=None)         bracketing control Q* of each control path (NR's bracket rule)          [3]
  d1(rec), d2(rec)                                 D1 task feasibility, D2 improvement beyond matched controls             [2, 3]
  decisions(rec)                                   D1-D6, the outcome class and the registered / adaptive labelling        [2-7]
  adaptive_tag(rec)                                None for a registered reading, else ADAPTIVE_* / REVISED_*              [7]
  campaign_summary(decision_records)               registered headline per family, adaptive looks, multiplicity           [7]
  strongest_nonzero(configs)                       strongest feasible nonzero release (D2R and the ablation)               [8]
  d2r(rec)                                         D2R robustness to classifier refitting (descriptive)                    [8]

Record of one family reading (`d1`, `d2`, `decisions`):
  {'family': 'DA' | 'DB' | 'FS_<name>', 'card': card id, 'revision': 0 (registered card) | 1 | 2,
   'card_committed_before_any_model_select_outcome': bool,
   'grid_valid': grid_validity(...)['effective'] (None = not recorded),
   'twin': {'config_id', 'evals': {v: utility}},
   'cand': {'status': 'FITTED' | 'NO_FEASIBLE_STRENGTH', 'config_id', 'evals': {v: utility},
            'every_nonzero_strength_alias_of_twin': bool},
   'strengths': [{'config_id', 'strength', 'evals'}]  every saved nonzero strength of the operation under this card (CAND included;
                                                       needed only when CAND fails G1),
   'grid': {comparator id: NR grid entry of CAND vs it ('pair', 'income', 'occupation')} for TWIN, every control-path point and the
           nine references,
   'paths': {'SBC': [saved control ids] | None, 'SPEND': [...] | None}   (TWIN, strength 0, is prepended by these rules),
   'controls_not_run': {'SBC': bool, 'SPEND': bool}   (True only when the budget forbade the path; recorded, never inferred),
   'alias_of': {id: earlier saved id or 'TWIN'},  'costs_vs': {id: nrm.evaluate paired costs of CAND vs id ('total|nll' + the 8
   secondary metrics)},  'nolookup': {id: {'cells': {registered cell: non-table winner pair benefit}}},  'enhanced': same format
   (reported only),  'reader_budget': {reference: 'READER_BUDGET_ASYMMETRIC' | ...},  'joint': None | NR G6 record,
   'novelty': label of a separate review,  'mc_flags': [...] (e.g. MC_SE_NOT_ESTIMABLE from NR.mc_cell)}

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

RULES_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_dual_design_release_v1' / 'roles'
              / 'statistical_reviewer' / 'RULES.json')
SCHEMA = 'DDR1-rules-v1'
NRM1_RULES_SHA256 = '84f187768817e79ef9f2bb36151bea81f00b8f62d9179c8cd523069550674652'     # NRM1 RULES.json (registered)
NRM_RULES_PY_SHA256 = 'd5df868678d483a3452e251d8f24575932d23f0e5d22e3ef1ca3627a35030df1'   # nrm/rules.py (registered)
VERSIONS = NR.VERSIONS
Z = 2.0
TIE_TOL = 1e-12
# ---- the registered DDR1 reader grid (READER_SPLITS.json; ddr.devreaders constants)
SEED_OFFSET = 800                                  # MODEL_SEED_BASE + 800 + 10 v + k (registered by FSA1, never used: FSA1 built no grid)
FORBIDDEN_OFFSETS = (300, 400, 500, 600, 700)      # every realized earlier grid: OHN1/CTS1, CTS, TSZ1, NRM1, FRC1
BOOT_SEED_BASE = 2026102200                        # + 100 v + repetition index
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
ALL_REPS = NR.REG_REPS + NR.ROB_REPS
ALL_K = NR.REG_K + NR.ROB_K
SPLIT_TAG = 'DDR1_devgrid_R{r}'
# ---- the prospective ER explanation (FRC1 ER1 restricted to its mechanism)
ER_MIN_LEAF = {'ET5': 5, 'ET25': 25}               # min_samples_leaf of the extra-trees recipes (ffv.readers)
ER_POOL_PEOPLE = 8001                              # MODEL_SELECT people: support bound = 8001 // min leaf (ET5 1600, ET25 320)
# ---- decisions
D1_LABELS = ('TASK_FEASIBLE', 'CAND_FEASIBLE_TWIN_INFEASIBLE', 'CAND_INFEASIBLE', 'NO_FEASIBLE_STRENGTH', 'INCOMPLETE')
D2_COMPARISONS = ('TWIN', 'SBC_FRONTIER', 'SPEND_FRONTIER')
D2_PATHS = (('SBC_FRONTIER', 'SBC'), ('SPEND_FRONTIER', 'SPEND'))
D2_MET = 'IMPROVEMENT_BEYOND_MATCHED_CONTROLS'
D2_ASSESSED = (D2_MET, 'READER_TARGETED', 'NOT_SEPARATED')             # plus PARTIAL_<passed>_ONLY
D2_GATES = ('NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'NOT_ASSESSED_EXACT_ALIAS', 'NOT_ASSESSED_TASK_INFEASIBLE',
            'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID', 'NOT_ASSESSED_CONTROLS_NOT_RUN', 'INCOMPLETE')
OUTCOMES = ('CANDIDATE', 'COMPONENT', 'TASK_ONLY_FEASIBLE_NO_PROTECTION', 'TASK_INFEASIBLE', 'SELECTED_STRENGTH_INFEASIBLE',
            'PROTECTION_NOT_ASSESSED', 'INCOMPLETE')
D2R_LABELS = ('ROBUSTNESS_CONFIRMED', 'ROBUSTNESS_NOT_CONFIRMED', 'NOT_RUN', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN',
              'NOT_COMPUTED_GRID_INVALID')
ADAPTIVE_KINDS = ('ADAPTIVE', 'REVISED')
PER_LOOK_FALSE_PASS = ('one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under six moderate NRM1 '
                       'known laws and about 8% when one version or split dominates (NRM1 RULES.md 5.5); with K D2 looks the '
                       'chance that at least one passes by chance is at most K times the per-look rate')
MDE_NOTE = ('per comparison, 2 se_agg (about 0.010 at se_agg 0.005) is the resolution threshold and about 0.015 is the benefit '
            'resolved with roughly 80% power; the conjunction of three comparisons (and targeting) has lower power; '
            '"not separated" is the default reading, never "equal"')

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
        g, er, dec = r['reader_grid'], r['er_explanation'], r['decisions']
        checks = {
            'z': (r['z'], Z), 'z equals NRM1': (Z, NR.Z), 'tie tolerance': (r['tie_tolerance'], TIE_TOL),
            'model seed base': (g['model_seed_base'], NR.MODEL_SEED_BASE), 'seed offset': (g['seed_offset'], SEED_OFFSET),
            'forbidden offsets': (tuple(g['forbidden_offsets']), FORBIDDEN_OFFSETS),
            'bootstrap seed base': (g['bootstrap_seed_base'], BOOT_SEED_BASE), 'repetition index': (g['repetition_index'], REP_INDEX),
            'split tag': (g['split_tag'], SPLIT_TAG), 'er recipes': (er['min_leaf'], ER_MIN_LEAF),
            'er pool people': (er['pool_people'], ER_POOL_PEOPLE), 'd1 labels': (tuple(dec['D1']['labels']), D1_LABELS),
            'd2 comparisons': (tuple(dec['D2']['comparisons']), D2_COMPARISONS), 'd2 met': (dec['D2']['met_label'], D2_MET),
            'd2r labels': (tuple(dec['D2R']['labels']), D2R_LABELS),
            'outcomes': (tuple(r['outcome_class']['labels']), OUTCOMES),
            'adaptive kinds': (tuple(r['registration_status']['kinds']), ADAPTIVE_KINDS),
            'target references': (tuple(dec['D4']['target_references']), NR.TARGET_REFS),
            'benchmark': (tuple(dec['D3']['references']), NR.BENCHMARK),
        }
        bad = [k for k, (a, b) in checks.items() if a != b]
        if bad:
            raise ValueError(f'RULES.json disagrees with ddr.rules on {bad}')
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


def control_path(ids, alias_of=None):
    """The control path of D2 (c): TWIN (strength 0 of every SEX-blind path) followed by the distinct saved control points
    (aliases once; an alias of TWIN collapses into TWIN). None if the path was not given."""
    if ids is None:
        return None
    return _path(['TWIN'] + list(ids), alias_of)


def _resolved_negative(entry):
    """True iff the pair benefit of an entry is resolved NEGATIVE (aggregate upper bound < 0 and both version means < 0): the
    comparator is resolved less recoverable than the candidate. None if the entry is incomplete."""
    a = NR.aggregate(entry)
    if not a.get('complete'):
        return None
    if a['alias']:
        return False
    return a['upper'] < 0 and all(a['per_version'][v]['mean'] < 0 for v in VERSIONS)


def _strength_value(x):
    """A strength as a number (a per-version dict is summed over both versions; missing or non-finite -> None)."""
    if isinstance(x, dict):
        vals = [_v(x, v) for v in VERSIONS]
        return sum(float(y) for y in vals) if all(_fin(y) for y in vals) else None
    return float(x) if _fin(x) else None


def _nonzero_strength(x):
    if isinstance(x, dict):
        vals = [_v(x, v) for v in VERSIONS]
        return all(_fin(y) and y > 0 for y in vals)
    return _fin(x) and x > 0


# ---------------------------------------------------------------- (9) reader-grid validity of the registered DDR1 grid
def forbidden_seeds():
    """Every realized earlier grid's reader seeds: MODEL_SEED_BASE + {300, 400, 500, 600, 700} + 10 v + k, k = 0..9 (100 seeds;
    a local equivalent of nrm.rules.forbidden_seeds, which reads its own constant (300, 500); nrm is not monkeypatched)."""
    return {NR.MODEL_SEED_BASE + off + 10 * v + k for off in FORBIDDEN_OFFSETS for v in VERSIONS for k in range(10)}


def registered_seed(v, k):
    return NR.MODEL_SEED_BASE + SEED_OFFSET + 10 * int(v) + int(k)


def registered_boot_seed(v, rep):
    return BOOT_SEED_BASE + 100 * int(v) + REP_INDEX[rep]


def _seed_errors(seeds, required_k):
    """Recorded distinct integer seeds for the required keys S<v>k<k>; every key S<v>k<k> equal to the registered MODEL_SEED_BASE
    + 800 + 10 v + k; none an earlier grid's seed (frc.rules logic with this grid's offsets)."""
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
    2026102200 + 100 v + repetition index; no other key form. None -> not recorded (an error)."""
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


def _explained_nrm(c, diff, distinct_inputs):
    """NRM1 A-SC1 (iii), exactly as nrm.rules.grid_validity states it: (a) another seed-dependent recipe differs in the same wire,
    view, version and repetition; (b) the same recipe differs on another view of the same wire; (c) the view's distinct-input count
    is recorded. 'Differs' = a RECORDED check with equal_k0_k1 False."""
    a = any((c['wire'], c['view'], c['v'], c['R'], r) in diff for r in NR.SEED_DEPENDENT if r != c['recipe'])
    b = any((c['wire'], w, c['v'], c['R'], c['recipe']) in diff for w in NR.VIEWS if w != c['view'])
    n = ((distinct_inputs or {}).get(f"{c['wire']}|V{c['v']}") or {}).get(c['view'])
    return bool(a and b and isinstance(n, int))


def support_bound(recipe):
    """Largest distinct-input count of a view on which the extra-trees recipe can reduce to a seed-invariant table: every distinct
    value needs at least min_leaf reader-FIT people, so more than 8001 // min_leaf distinct values (MODEL_SELECT) rules it out."""
    return ER_POOL_PEOPLE // ER_MIN_LEAF[recipe] if recipe in ER_MIN_LEAF else None


def _explained_er(c, diff, distinct_inputs):
    """The DDR1 minimum-set explanation (FRC1 ER1, prospective, restricted to its mechanism): an extra-trees recipe (ET5 / ET25),
    NRM1 (iii) (a) + (b) + (c), and the recorded distinct-input count of the view at most support_bound(recipe)."""
    if c.get('recipe') not in ER_MIN_LEAF or not _explained_nrm(c, diff, distinct_inputs):
        return False
    n = ((distinct_inputs or {}).get(f"{c['wire']}|V{c['v']}") or {}).get(c['view'])
    return isinstance(n, int) and not isinstance(n, bool) and 0 < n <= support_bound(c['recipe'])


def _group_checks(checks):
    """Checks grouped by their reader-seed pair ('k_pair', default (0, 1)); each pair is judged on its own (the NRM1 'differs'
    key has no seed pair, so pooling pairs would let a difference in one pair explain an equality in another)."""
    groups, errs = {}, []
    for c in checks or []:
        kp = c.get('k_pair', (0, 1)) if isinstance(c, dict) else None
        if not isinstance(kp, (list, tuple)) or len(kp) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in kp):
            errs.append(f'malformed seed check {c!r}')
            continue
        groups.setdefault(tuple(kp), []).append(c)
    return (groups or {(0, 1): []}), errs


def _recipe_part(seeds, checks, minimum_set, distinct_inputs):
    """One seed-pair group: NRM1 A-SC1 (ii)-(iii) with the DDR1 minimum-set explanation; nrm.rules.grid_validity is called
    unchanged (with the real minimum set for the strict reading, with an empty one for (iii) on every wire) and the local
    classification is asserted against it."""
    sd = seeds if isinstance(seeds, dict) else {}
    try:
        base = NR.grid_validity(sd, checks, [], distinct_inputs)
        strict = NR.grid_validity(sd, checks, minimum_set, distinct_inputs)
        eq = [c for c in checks if c.get('recipe') in NR.SEED_DEPENDENT and c.get('equal_k0_k1') is not False]
        diff = _diff_set(checks)
        unexp_nrm = [c for c in eq if not _explained_nrm(c, diff, distinct_inputs)]
    except (KeyError, TypeError, AttributeError) as ex:
        return {'errors': [f'malformed seed checks: {ex!r}'], 'flags': [], 'explained_min': [], 'unexplained_min': [],
                'strict_errors': ['malformed seed checks']}
    if len(eq) - base['explained_equalities'] != len(unexp_nrm):
        raise AssertionError('DDR1 classification disagrees with nrm.rules.grid_validity')
    mins = set(minimum_set or ())
    on_min = [c for c in eq if c['wire'] in mins]
    exp_min = [c for c in on_min if _explained_er(c, diff, distinct_inputs)]
    unexp_min = [c for c in on_min if not _explained_er(c, diff, distinct_inputs)]
    unexp_other = [c for c in eq if c['wire'] not in mins and not _explained_nrm(c, diff, distinct_inputs)]
    errors = [e for e in base['errors'] if not e.startswith('reader seeds') and 'unexplained seed-dependent' not in e]
    if unexp_min:
        errors.append(f'{len(unexp_min)} unexplained seed-dependent equalities on the minimum set')
    if unexp_other:
        errors.append(f'{len(unexp_other)} unexplained seed-dependent equalities')
    return {'errors': errors, 'flags': base.get('flags', []), 'explained_min': exp_min, 'unexplained_min': unexp_min,
            'unexplained_other': len(unexp_other), 'checks': len(checks),
            'strict_errors': [e for e in strict['errors'] if not e.startswith('reader seeds')]}


def _n_eff_report(exp_min, checks, distinct_inputs):
    out = {}
    for c in exp_min:
        key = f"{c['wire']}|{c['view']}|{c['recipe']}|k{c.get('k_pair', (0, 1))[0]}k{c.get('k_pair', (0, 1))[1]}"
        if key in out:
            continue
        same = (lambda x, c=c: x.get('wire') == c['wire'] and x.get('view') == c['view'] and x.get('recipe') == c['recipe']
                and tuple(x.get('k_pair', (0, 1))) == tuple(c.get('k_pair', (0, 1))))
        blocks = {(x['v'], x['R']) for x in checks if same(x)}
        eqb = {(x['v'], x['R']) for x in exp_min if same(x)}
        out[key] = {'blocks_checked': len(blocks), 'blocks_seed_pair_equal': len(eqb),
                    'n_eff_seed_cells_if_this_recipe_wins': 2 * len(blocks) - len(eqb),
                    'distinct_inputs': {f'V{v}': ((distinct_inputs or {}).get(f"{c['wire']}|V{v}") or {}).get(c['view'])
                                        for v in VERSIONS}, 'support_bound': support_bound(c['recipe'])}
    return out


def grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=None, required_k=NR.REG_K, reps=NR.REG_REPS):
    """Reader-seed validity of the registered DDR1 grid, applied prospectively (RULES.md 9). READER_SEEDS_EFFECTIVE iff
    (1) seeds: the required S<v>k<k> recorded, distinct integers, every one equal to MODEL_SEED_BASE + 800 + 10 v + k, none an
        earlier realized grid's seed (offsets 300, 400, 500, 600, 700);
    (2) bootstrap: S<v><R> recorded for every version and required repetition, equal to 2026102200 + 100 v + repetition index;
    (3) per seed pair (k0 / k1; for the refit also k2 / k3): every seed-dependent recipe checked; NRM1 A-SC1 (iii) explains every
        seed-dependent equality off the minimum set; on the minimum set (a D2 reading's CAND, TWIN and saved control points; for the
        refit the strongest release, its TWIN and its bracketing controls) an equality is explained only for an extra-trees recipe
        (ET5, ET25) with (a) + (b) + (c) AND a recorded distinct-input count of the view of at most 8001 // min_leaf (1600 / 320).
        Explained minimum-set equalities count once (n_eff) and are reported.
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
        exp_all += [{**c, 'k_pair': list(kp)} for c in part['explained_min']]
        per[tag] = {'checks': part.get('checks', 0), 'errors': part['errors'], 'explained_minimum_set': len(part['explained_min']),
                    'unexplained_minimum_set': len(part['unexplained_min']), 'unexplained_other': part.get('unexplained_other')}
    seed_errs = _seed_errors(seeds, required_k) + _boot_errors(boot_seeds, reps)
    ok = not errors
    return {'label': 'READER_SEEDS_EFFECTIVE' if ok else 'READER_SEEDS_NOT_EFFECTIVE', 'effective': ok, 'errors': errors,
            'per_seed_pair': per, 'flags': sorted(flags),
            'explained_minimum_set_equalities': {
                'count': len(exp_all), 'n_eff_implied': _n_eff_report(exp_all, [c for c in checks or [] if isinstance(c, dict)],
                                                                      distinct_inputs),
                'items': [{k: c.get(k) for k in ('wire', 'view', 'v', 'R', 'recipe', 'k_pair')} for c in exp_all],
                'note': 'an extra-trees recipe reduces to a seed-invariant table on a few-valued view; its seed-pair cells count once '
                        '(n_eff); the reader seed is effective for the other recipes (explanation a) and other views (b)'},
            'strict_nrm1': {'label': 'READER_SEEDS_NOT_EFFECTIVE' if (seed_errs or strict_errors) else 'READER_SEEDS_EFFECTIVE',
                            'errors': seed_errs + strict_errors},
            'seed_offset': SEED_OFFSET, 'forbidden_offsets': list(FORBIDDEN_OFFSETS), 'required_k': list(required_k),
            'reps': list(reps), 'rule': 'DDR1 prospective: FRC1 ER1 explanation restricted to extra-trees recipes on views of '
                                        'bounded support'}


# ---------------------------------------------------------------- (8) genuinely effective classifier refit
def refit_effectiveness(splits, seeds, checks, minimum_set, distinct_inputs, boot_seeds=None):
    """PROTOCOL 6.3 'refit-effectiveness check' (frc.rules logic, with the DDR1 explanation applied prospectively). splits = {'R1'..
    'R4': {'assignment_sha256': str, 'score_overlap': {other repetition: share of SCORE households shared}}}; seeds = {'S<v>k<k>':
    int} for k = 0..3; checks = recorded seed checks with 'k_pair' ([0, 1] or [2, 3]); minimum_set = the strongest release, its TWIN
    and its bracketing controls; boot_seeds = {'S<v><R>': int} for R1-R4. REFIT_GENUINELY_EFFECTIVE iff (a) R3 and R4 use different
    household splits (assignment hashes recorded and distinct from each other and from R1 / R2; every SCORE overlap involving R3 or
    R4 recorded, finite and in [0, 1)) and (b) grid_validity holds on all eight seeds and four repetitions. Otherwise
    REFIT_NOT_GENUINELY_EFFECTIVE and the failed part's cells are counted once (no replication credit)."""
    errs = []
    hs = [((splits or {}).get(r) or {}).get('assignment_sha256') for r in ALL_REPS]
    if not all(isinstance(h, str) and h for h in hs) or len(set(hs)) != len(hs):
        errs.append('SPLITS_NOT_DIFFERENT_OR_NOT_RECORDED')
    else:
        for r in NR.ROB_REPS:
            for q in ALL_REPS:
                if q == r:
                    continue
                o = (((splits or {}).get(r) or {}).get('score_overlap') or {}).get(q)
                if o is None:
                    o = (((splits or {}).get(q) or {}).get('score_overlap') or {}).get(r)
                if not _fin(o) or not 0 <= o < 1:
                    errs.append(f'SCORE_OVERLAP_{r}_{q}_NOT_RECORDED_OR_IDENTICAL')
    gv = grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=boot_seeds, required_k=ALL_K, reps=ALL_REPS)
    if not gv['effective']:
        errs.append('READER_SEEDS_NOT_EFFECTIVE')
    count_once = (['robustness repetitions'] if any(e.startswith(('SPLITS', 'SCORE')) for e in errs) else []) + \
                 (['robustness reader seeds'] if not gv['effective'] else [])
    return {'label': 'REFIT_GENUINELY_EFFECTIVE' if not errs else 'REFIT_NOT_GENUINELY_EFFECTIVE', 'reasons': errs,
            'grid_validity': gv, 'counted_once': count_once,
            'note': 'R3 / R4 are new household splits of the same MODEL_SELECT people and k2 / k3 new reader seeds: robustness to '
                    'classifier refitting, never replication'}


# ---------------------------------------------------------------- (3) brackets, D1, D2
def brackets(costs_vs, paths, alias_of=None):
    """The bracketing control Q* of each control path, by NR's rule (the cheapest path point whose pooled MODEL_SELECT paired
    total-NLL cost is >= the candidate's, i.e. d = cost(point) - cost(control) <= 0; ties keep path order). Path = TWIN + the
    distinct saved points (control_path). Per key: {'bracket': id | None, 'status': BRACKETED | NO_BRACKET | PATH_NOT_GIVEN |
    INCOMPLETE, 'path', 'primary'}."""
    out = {}
    for _, key in D2_PATHS:
        ids = control_path((paths or {}).get(key), alias_of)
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


def d1(rec):
    """D1 task feasibility: G1 (NR.point_eligibility, 28 conditions at the point) for TWIN and CAND, the reserve beside. Labels:
    TASK_FEASIBLE (both), CAND_FEASIBLE_TWIN_INFEASIBLE, CAND_INFEASIBLE (whatever TWIN), NO_FEASIBLE_STRENGTH (no nonzero grid
    strength met the MODEL_FIT budget: no CAND), INCOMPLETE. Also each fitted nonzero strength's point label."""
    tw = NR.point_eligibility(((rec or {}).get('twin') or {}).get('evals') or {})
    cand = (rec or {}).get('cand') or {}
    out = {'twin': tw['label'], 'cand': None}

    def reserve(evals):
        try:
            r = NR.reserve_status(evals)
            return {'status': r.get('status'), 'qualified': bool(r.get('qualified'))}
        except (KeyError, ValueError, TypeError) as ex:
            return {'status': 'INCOMPLETE', 'qualified': False, 'error': str(ex)}
    if cand.get('status') == 'NO_FEASIBLE_STRENGTH':
        return {**out, 'label': 'NO_FEASIBLE_STRENGTH', 'met': False,
                'statement': 'no nonzero grid strength met the MODEL_FIT selection budget: no candidate exists (no fallback)'}
    if cand.get('status') != 'FITTED':
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'error': 'CAND status missing'}
    cg = NR.point_eligibility(cand.get('evals') or {})
    out['cand'] = cg['label']
    if cg['label'] == 'INCOMPLETE':
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'error': cg.get('error')}
    if cg['label'] == 'POINT_INELIGIBLE':
        lab = 'CAND_INFEASIBLE'
    elif tw['label'] == 'INCOMPLETE':
        return {**out, 'label': 'INCOMPLETE', 'met': False, 'error': f'TWIN utility: {tw.get("error")}'}
    else:
        lab = 'TASK_FEASIBLE' if tw['label'] == 'POINT_ELIGIBLE' else 'CAND_FEASIBLE_TWIN_INFEASIBLE'
    strengths, ignored = {}, []
    for s in (rec or {}).get('strengths') or []:
        if not _nonzero_strength(s.get('strength')):
            ignored.append(s.get('config_id'))
            continue
        strengths[s.get('config_id')] = NR.point_eligibility(s.get('evals') or {})['label']
    out.update({'label': lab, 'met': lab == 'TASK_FEASIBLE', 'cand_failed': cg.get('failed'), 'twin_failed': tw.get('failed'),
                'reserve': {'cand': reserve(cand.get('evals') or {}), 'twin': reserve(((rec or {}).get('twin') or {}).get('evals') or {})},
                'fitted_strengths': strengths if (rec or {}).get('strengths') is not None else None,
                'ignored_zero_strength_entries': ignored,
                'note': 'G1 at the point on adaptively used MODEL_SELECT people; point feasibility is not confidence preservation'})
    return out


def _targeting_flags(tg, enhanced):
    """Reported only: the enhanced (TABLE_CV) winner's point benefit against each targeting point; READER_SENSITIVE_ENHANCED_<q>
    if some version mean <= 0."""
    rep, flags = {}, []
    for q in tg:
        x = _point_positive((enhanced or {}).get(q))
        rep[q] = x
        if x is not None and not x['pass']:
            flags.append(f'READER_SENSITIVE_ENHANCED_{q}')
    return rep, flags


def d2(rec):
    """D2 improvement beyond matched controls at CAND (RULES.md 3). Labels in order of precedence:
    NOT_ASSESSED_NO_FEASIBLE_STRENGTH; NOT_ASSESSED_EXACT_ALIAS (every nonzero strength transmits the twin's vectors: the family
    stops); INCOMPLETE (CAND status or utility missing); NOT_ASSESSED_TASK_INFEASIBLE (CAND fails G1: the protection comparison is
    not run, never negative, never protection); NOT_COMPUTED_GRID_VALIDITY_UNKNOWN / NOT_COMPUTED_GRID_INVALID;
    NOT_ASSESSED_CONTROLS_NOT_RUN (a control path recorded as not run for budget); INCOMPLETE (any needed input missing); then the
    conjuncts (a) G1 (passed above); (b) TWIN: NR.comparison passes (AGG_POSITIVE pair benefit + recipient gate); (c) for the SBC
    and the SPEND paths (each TWIN + saved points): NR.frontier_comparison = DOMINATES_CONTROL_FRONTIER, AND the recipient gate vs
    the bracketing point Q*, AND no cheaper path point resolved less recoverable than CAND; (d) reader targeting: the non-table
    winner's point benefit > 0 in both version means vs TWIN and every bracketing point. Result: IMPROVEMENT_BEYOND_MATCHED_CONTROLS
    (the only 'met': b, c, d all hold), READER_TARGETED (b and c hold, d fails), NOT_SEPARATED (none of b, c), else
    PARTIAL_<passed>_ONLY (order TWIN, SBC_FRONTIER, SPEND_FRONTIER). No partial outcome is a claim."""
    rec = rec or {}
    cand = rec.get('cand') or {}
    st = cand.get('status')
    if st == 'NO_FEASIBLE_STRENGTH':
        return {'label': 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'met': False}
    if cand.get('every_nonzero_strength_alias_of_twin') is True:
        return {'label': 'NOT_ASSESSED_EXACT_ALIAS', 'met': False, 'family_stops': True,
                'statement': "every allowed nonzero strength transmits exactly the twin's vectors: the family stops"}
    if st != 'FITTED':
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'CAND status missing'}
    g1 = NR.point_eligibility(cand.get('evals') or {})
    if g1['label'] == 'INCOMPLETE':
        return {'label': 'INCOMPLETE', 'met': False, 'error': f'CAND utility incomplete: {g1.get("error")}'}
    if g1['label'] != 'POINT_ELIGIBLE':
        return {'label': 'NOT_ASSESSED_TASK_INFEASIBLE', 'met': False, 'g1_failed': g1.get('failed'),
                'statement': 'the candidate fails G1: the protection comparison is reported as not run (never negative, never '
                             'protection)'}
    why = _gate_label(rec.get('grid_valid'))
    if why:
        return {'label': why, 'met': False}
    paths = rec.get('paths') or {}
    not_run = [key for _, key in D2_PATHS if paths.get(key) is None and (rec.get('controls_not_run') or {}).get(key) is True]
    if not_run:
        return {'label': 'NOT_ASSESSED_CONTROLS_NOT_RUN', 'met': False, 'not_run': not_run,
                'statement': 'a matched control path was not run (budget): the comparison is reported as not run'}
    grid, flags, comps = rec.get('grid') or {}, [], {}

    def incomplete(msg):
        return {'label': 'INCOMPLETE', 'met': False, 'error': msg, 'comparisons': comps, 'flags': sorted(set(flags))}

    def gate_of(q):
        c = NR.comparison(grid.get(q) or {})
        if c['benefit']['label'] == 'INCOMPLETE' or any(c['recipient'][w]['label'] == 'INCOMPLETE' for w in NR.SINGLE_VIEWS):
            return None, c
        return c['recipient_gate_pass'], c
    if 'TWIN' not in grid:
        return incomplete('grid entry vs TWIN missing')
    gt, ct = gate_of('TWIN')
    if gt is None:
        return incomplete('grid entry vs TWIN incomplete')
    comps['TWIN'] = {'benefit': ct['benefit']['label'], 'benefit_mean': ct['benefit']['aggregate'].get('mean'),
                     'benefit_se': ct['benefit']['aggregate'].get('se_agg'), 'recipient_gate_pass': gt,
                     'recipient': {w: ct['recipient'][w]['label'] for w in NR.SINGLE_VIEWS}, 'point': 'TWIN',
                     'pass': ct['benefit_positive'] and gt}
    if ct['benefit']['label'] == 'ALIAS_ZERO':
        flags.append('CAND_ALIAS_OF_TWIN')
    point = {'id': cand.get('config_id', 'CAND'), 'costs_vs': rec.get('costs_vs') or {}, 'grid_vs': grid}
    for name, key in D2_PATHS:
        if paths.get(key) is None:
            return incomplete(f'{key} path missing (not recorded as not run)')
        ids = control_path(paths.get(key), rec.get('alias_of'))
        if ids == ['TWIN']:
            flags.append(f'{key}_AT_IDENTITY')
        fr = NR.frontier_comparison(point, [{'id': i} for i in ids])
        if fr['label'] == 'INCOMPLETE':
            return incomplete(f'{name}: {fr.get("error")}')
        q = fr.get('bracketing_control')
        gate, rcp = None, None
        if q is not None:
            gate, cq = gate_of(q)
            if gate is None:
                return incomplete(f'{name}: recipient views vs bracketing control {q} missing or incomplete')
            rcp = {w: cq['recipient'][w]['label'] for w in NR.SINGLE_VIEWS}
        prim = fr.get('primary_cost_differences') or {}
        dominated = []
        for i in ids:
            pv = (prim.get(i) or {}).get('value')
            if _fin(pv) and pv > 0:                       # the control is cheaper than CAND
                neg = _resolved_negative((grid.get(i) or {}).get('pair'))
                if neg is None:
                    return incomplete(f'{name}: grid vs cheaper control point {i} missing')
                if neg:
                    dominated.append(i)
        ok = fr['label'] == 'DOMINATES_CONTROL_FRONTIER' and gate is True and not dominated
        reason = fr.get('reason')
        if fr['label'] == 'DOMINATES_CONTROL_FRONTIER' and gate is False:
            reason = 'RECIPIENT_GATE_FAILED_AT_BRACKETING_CONTROL'
            flags.append(f'RECIPIENT_GATE_FAILED_VS_{q}')
        if dominated:
            reason = 'DOMINATED_BY_CHEAPER_CONTROL' if fr['label'] == 'DOMINATES_CONTROL_FRONTIER' else reason
            flags += [f'DOMINATED_BY_CHEAPER_{i}' for i in dominated]
        if fr.get('reason') == 'CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT':
            flags.append(f'NO_BRACKET_{key}')
        comps[name] = {'label': fr['label'], 'reason': reason, 'path': ids, 'point': q, 'recipient_gate_vs_bracket': gate,
                       'recipient_vs_bracket': rcp, 'dominated_by_cheaper': dominated, 'pass': ok,
                       'benefit_mean': fr.get('benefit_mean'), 'benefit_se': fr.get('benefit_se'), 'frontier': fr}
    tg = {}
    for q in dict.fromkeys(c['point'] for c in comps.values() if c.get('point')):
        x = _point_positive((rec.get('nolookup') or {}).get(q))
        if x is None:
            return incomplete(f'reader-targeting input (non-table winner) vs {q} missing')
        tg[q] = x
    passed = []
    for n in D2_COMPARISONS:
        c = comps[n]
        c['targeting_pass'] = tg[c['point']]['pass'] if c.get('point') in tg else None
        if c['pass']:
            passed.append(n)
            if c['targeting_pass'] is False:
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
    return {'label': label, 'met': label == D2_MET, 'passed': passed, 'comparisons': comps, 'targeting': tg,
            'enhanced_reported_only': enh, 'flags': sorted(set(flags)), 'minimum_detectable': MDE_NOTE,
            'note': 'no partial outcome is a claim; "not separated" is the default reading'}


# ---------------------------------------------------------------- (4) D3, D4 with the reader-budget credit rule (frc logic)
def reference_credit(rec, refs):
    """A reference flagged READER_BUDGET_ASYMMETRIC (rec['reader_budget'][ref], ddr.readerbudget.contrast, pair view) is credited
    iff the non-table winner's point benefit against it ('nolookup') is > 0 in both version means; missing input -> INCOMPLETE
    (never 'not beaten'). Only flagged references appear."""
    out = {}
    for ref in refs:
        if ((rec or {}).get('reader_budget') or {}).get(ref) == 'READER_BUDGET_ASYMMETRIC':
            x = _point_positive(((rec or {}).get('nolookup') or {}).get(ref))
            out[ref] = 'INCOMPLETE' if x is None else ('CREDITED' if x['pass'] else 'NOT_CREDITED')
    return out


def _d3(rec, cl):
    """D3 = NRM1 G4 (all nine point-eligible references, intersection-union, recipient gates) with the reader-budget rule in
    three-valued form: NOT_CREDITED turns a met G4 into NOT_SUPERIOR; INCOMPLETE turns it into INCOMPLETE; a failing G4 stays."""
    lab = cl['G4']['label']
    cr = reference_credit(rec, NR.BENCHMARK)
    nc = sorted(r for r, x in cr.items() if x == 'NOT_CREDITED')
    inc = sorted(r for r, x in cr.items() if x == 'INCOMPLETE')
    if lab in ('SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES', 'MET_WITHOUT_TASK_FEASIBILITY'):
        lab = 'NOT_SUPERIOR' if nc else ('INCOMPLETE' if inc else lab)
    return {'label': lab, 'met': lab == 'SUPERIOR_TO_ALL_ELIGIBLE_REFERENCES',
            'failing': sorted(set(cl['G4'].get('failing') or []) | set(nc)), 'reader_budget_credit': cr,
            'not_credited': nc, 'incomplete_references': inc, 'missing': cl['G4'].get('missing')}


def _d4(rec, cl):
    """D4 = NRM1 G5 (0.020 vs NEURAL_RAW and H_LJ; levels kept) with the same rule for the two targets; met iff
    MATERIAL_0020_RESOLVED. Smaller gains keep their own level (POSITIVE_BELOW_MATERIAL_TARGET), never the criterion."""
    level = cl['G5'].get('level')
    cr = reference_credit(rec, NR.TARGET_REFS)
    if any(x == 'NOT_CREDITED' for x in cr.values()):
        level = 'NOT_MET'
    elif any(x == 'INCOMPLETE' for x in cr.values()) and level != 'NOT_MET':
        level = 'INCOMPLETE'
    feas = cl['G1']['label'] == 'TASK_FEASIBLE'
    lab = level if (feas or level in ('NOT_MET', 'INCOMPLETE')) else 'MET_WITHOUT_TASK_FEASIBILITY'
    return {'label': lab, 'level': level, 'met': lab == 'MATERIAL_0020_RESOLVED', 'reader_budget_credit': cr,
            'nrm_g5_label': cl['G5']['label']}


# ---------------------------------------------------------------- (7) registered vs adaptive readings
def adaptive_tag(rec):
    """None for a registered reading: revision 0 of DA / DB whose card was committed before any MODEL_SELECT outcome of the
    campaign. Otherwise '<KIND>_<parts>': KIND = REVISED if the card was committed before any MODEL_SELECT outcome, else ADAPTIVE;
    parts = FS (replacement family), R<n> (revision n >= 1), R0_LATE_CARD (a revision-0 card committed after an outcome or not
    recorded). Examples: ADAPTIVE_R1, REVISED_R1, ADAPTIVE_FS, REVISED_FS, ADAPTIVE_FS_R1, ADAPTIVE_R0_LATE_CARD."""
    n = int((rec or {}).get('revision') or 0)
    fs = str((rec or {}).get('family') or '').startswith('FS')
    early = (rec or {}).get('card_committed_before_any_model_select_outcome') is True
    if n == 0 and not fs and early:
        return None
    parts = (['FS'] if fs else []) + ([f'R{n}'] if n else ([] if fs else ['R0_LATE_CARD']))
    return f"{'REVISED' if early else 'ADAPTIVE'}_{'_'.join(parts)}"


def _mark(out, tag):
    for k in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'outcome_class'):
        x = out.get(k)
        if isinstance(x, dict) and 'label' in x:
            x['label_unprefixed'] = x['label']
            if tag:
                x['label'] = f'{tag}__{x["label"]}'


# ---------------------------------------------------------------- (2-7) decisions and the outcome class
def decisions(rec):
    """D1-D6 and the outcome class of one family reading (CAND of one card) (RULES.md 2-7). D3 and D4 are read at CAND (the D2
    configuration), never at another strength, the strongest release or the ablation. Outcome class, first applicable:
    INCOMPLETE (D1 incomplete; D2 incomplete or not computed for grid validity; D2 met and D3 undecided); TASK_INFEASIBLE (no
    feasible strength on MODEL_FIT, or CAND and every other fitted nonzero strength fail G1); SELECTED_STRENGTH_INFEASIBLE (CAND fails
    G1, another fitted nonzero strength passes: reported, never promoted); PROTECTION_NOT_ASSESSED (a control path not run for
    budget); CANDIDATE (D2 met and D3 met); COMPONENT (D2 met, D3 not); TASK_ONLY_FEASIBLE_NO_PROTECTION (CAND G1, D2 not met).
    Flags: D3_WITHOUT_D2, TWIN_INFEASIBLE, POINT_FEASIBLE_NOT_RESERVE_QUALIFIED, FAMILY_STOPS_EXACT_ALIAS, D2 / MC flags. A reading
    that is not registered (adaptive_tag) has every label prefixed '<tag>__' ('label_unprefixed' keeps the raw label)."""
    rec = rec or {}
    cand = rec.get('cand') or {}
    out = {'family': rec.get('family'), 'card': rec.get('card'), 'revision': int(rec.get('revision') or 0),
           'D1': d1(rec), 'D2': d2(rec)}
    d2lab = out['D2']['label']
    cg = NR.point_eligibility(cand.get('evals') or {})['label'] if cand.get('status') == 'FITTED' else None
    gates = ('NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'NOT_ASSESSED_EXACT_ALIAS', 'NOT_ASSESSED_TASK_INFEASIBLE',
             'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID')
    if d2lab in gates:
        out['D3'] = {'label': d2lab, 'met': False}
        out['D4'] = {'label': d2lab, 'met': False}
    elif cand.get('status') != 'FITTED' or cg != 'POINT_ELIGIBLE':
        out['D3'] = {'label': 'INCOMPLETE', 'met': False}
        out['D4'] = {'label': 'INCOMPLETE', 'met': False}
    else:
        cl = NR.claims({'evals': cand.get('evals') or {}, 'grid': rec.get('grid') or {}, 'grid_valid': rec.get('grid_valid'),
                        'frontier': None, 'matched': {}, 'joint': None})
        out['D3'] = _d3(rec, cl)
        out['D4'] = _d4(rec, cl)
    j = rec.get('joint')
    if not j or not all(j.get(k) for k in ('separate', 'income_first', 'occupation_first')):
        out['D5'] = {'label': 'NOT_ASSESSED_MISSING_CONTROLS', 'statement': 'no joint-design advantage is claimed'}
    else:
        cl = NR.claims({'evals': cand.get('evals') or {}, 'grid': rec.get('grid') or {}, 'grid_valid': rec.get('grid_valid'),
                        'matched': {}, 'joint': j})
        out['D5'] = {'label': cl['G6']['label'], 'detail': cl['G6']}
    nov = rec.get('novelty') or 'NOT_CLAIMED'
    out['D6'] = {'label': nov if nov in NR.NOVELTY_LABELS else 'NOT_CLAIMED',
                 'note': 'from a separate prior-art review; never inferred from performance'}
    # outcome class
    flags = list(out['D2'].get('flags') or []) + list(rec.get('mc_flags') or [])
    d1lab = out['D1']['label']
    d3lab = out['D3']['label']
    if out['D1'].get('twin') == 'POINT_INELIGIBLE':
        flags.append('TWIN_INFEASIBLE')
    if d2lab == 'NOT_ASSESSED_EXACT_ALIAS':
        flags.append('FAMILY_STOPS_EXACT_ALIAS')
    why = None
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
    elif d1lab == 'INCOMPLETE' or d2lab in ('INCOMPLETE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID'):
        oc, why = 'INCOMPLETE', f'D1 {d1lab}, D2 {d2lab}'
    elif d2lab == 'NOT_ASSESSED_CONTROLS_NOT_RUN':
        oc = 'PROTECTION_NOT_ASSESSED'
    elif out['D2']['met'] and d3lab in ('INCOMPLETE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID'):
        oc, why = 'INCOMPLETE', f'D2 met, D3 {d3lab}'
    elif out['D2']['met'] and out['D3']['met']:
        oc = 'CANDIDATE'
    elif out['D2']['met']:
        oc = 'COMPONENT'
    else:
        oc = 'TASK_ONLY_FEASIBLE_NO_PROTECTION'
    if out['D3'].get('met') and not out['D2'].get('met'):
        flags.append('D3_WITHOUT_D2')
    if cg == 'POINT_ELIGIBLE':
        try:
            if not NR.reserve_status(cand.get('evals') or {}).get('qualified'):
                flags.append('POINT_FEASIBLE_NOT_RESERVE_QUALIFIED')
        except (KeyError, ValueError, TypeError):
            flags.append('RESERVE_INCOMPLETE')
    tag = adaptive_tag(rec)
    out['outcome_class'] = {'label': oc, 'flags': sorted(set(flags)), 'reason': why,
                            'note': 'development evidence on adaptively used MODEL_SELECT records, never confirmation; a near-'
                                    'limit point estimate is never called robust'}
    out['registered'] = tag is None
    out['adaptive_tag'] = tag
    out['registration_note'] = ('registered reading (revision 0, card committed before any MODEL_SELECT outcome)' if tag is None
                                else 'NOT a registered result: ' + tag + ' (labels prefixed; report beside the registered reading '
                                     'with the number of looks)')
    _mark(out, tag)
    out['scope'] = list(NR.SCOPE) + ['Point feasibility and the reserve are not confidence preservation.',
                                     'A revision or replacement-family reading is never the registered result.']
    return out


def _raw(x):
    return (x or {}).get('label_unprefixed', (x or {}).get('label'))


def campaign_summary(decision_records):
    """The registered headline (one registered reading per family; a second one is refused), every non-registered reading with
    its tag, the number K of D2 looks (readings whose D2 was assessed), the readings with D2 met, and the multiplicity statement
    that any campaign-level sentence must carry. Flag D2_MET_ONLY_IN_NON_REGISTERED_LOOKS."""
    reg, adaptive, looks, met = {}, [], 0, []
    for d in decision_records or []:
        fam = d.get('family')
        lab = _raw(d.get('D2')) or ''
        assessed = lab in D2_ASSESSED or lab.startswith('PARTIAL_')
        looks += assessed
        row = {'family': fam, 'card': d.get('card'), 'revision': d.get('revision'), 'tag': d.get('adaptive_tag'),
               'D2': lab, 'outcome_class': _raw(d.get('outcome_class')), 'd2_assessed': assessed}
        if d.get('registered'):
            if fam in reg:
                raise ValueError(f'two registered readings for family {fam!r}')
            reg[fam] = row
        else:
            adaptive.append(row)
        if (d.get('D2') or {}).get('met'):
            met.append(row)
    k0 = sum(r['d2_assessed'] for r in reg.values())
    m0 = sum(1 for r in met if r['tag'] is None)
    flags = ['D2_MET_ONLY_IN_NON_REGISTERED_LOOKS'] if met and not m0 else []
    return {'registered': reg, 'non_registered': adaptive, 'd2_looks': looks, 'd2_met': met, 'flags': flags,
            'statement': (f'D2 was met in {m0} of {k0} registered family readings and in {len(met) - m0} of {looks - k0} '
                          f'non-registered readings ({looks} D2 looks in total); {PER_LOOK_FALSE_PASS}.')}


# ---------------------------------------------------------------- (8) strongest feasible nonzero release; D2R
def strongest_nonzero(configs):
    """The strongest feasible nonzero release of a family (frc.rules logic). configs = [{'config_id', 'strength' (number or per-
    version dict), 'revision', 'twin_id', 'evals': {v: utility}, 'pair_vs_twin': NR grid pair entry vs its own TWIN, 'alias_of':
    None or an earlier id}] of the family's OPERATION (kind candidate; controls and ablations excluded). Eligible: nonzero strength
    in every version, G1, not an alias of an earlier configuration, pair benefit vs TWIN neither ALIAS_ZERO nor INCOMPLETE.
    Choice: the largest registered-cell aggregate pair benefit x vs TWIN (it may be negative); |dx| <= 1e-12 ties go to the smaller
    strength, then the lower revision, then config_id. Chosen on registered MODEL_SELECT cells (winner's curse): descriptive; it
    never changes D1-D6. Flag MIXED_TWINS if eligible configurations have different twins."""
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
            b = NR.benefit_screen(c.get('pair_vs_twin'))
            r['benefit_label'] = b['label']
            if b['label'] in ('ALIAS_ZERO', 'INCOMPLETE'):
                r['status'] = 'ALIAS_OF_TWIN' if b['label'] == 'ALIAS_ZERO' else 'INCOMPLETE_GRID'
            else:
                r.update({'status': 'ELIGIBLE', 'x': b['aggregate']['mean'], 'twin_id': c.get('twin_id')})
                elig.append(r)
        rows.append(r)
    if not elig:
        return {'label': 'NO_FEASIBLE_NONZERO_RELEASE', 'selected': None, 'records': rows}
    best = max(r['x'] for r in elig)
    near = sorted((r for r in elig if r['x'] >= best - TIE_TOL),
                  key=lambda r: (_strength_value(r['strength']), r['revision'], str(r['config_id'])))
    twins = {r.get('twin_id') for r in elig}
    return {'label': 'SELECTED', 'selected': near[0]['config_id'], 'x': near[0]['x'], 'revision': near[0]['revision'],
            'tied': [r['config_id'] for r in near], 'records': rows, 'flags': ['MIXED_TWINS'] if len(twins) > 1 else [],
            'note': 'chosen on registered MODEL_SELECT cells (winner\'s curse); robustness cells and the ablation never change D1-D6'}


def d2r(rec):
    """D2R robustness to classifier refitting (descriptive; never changes D2). rec = {'grid_valid': bool, 'strongest':
    strongest_nonzero(...) record, 'refit': refit_effectiveness(...) record, 'rob': {comparator: NR grid entry of the strongest
    release vs it, the 8 registered and the robustness cells MERGED}, 'brackets': {'SBC': id | None, 'SPEND': id | None} (the
    strongest release's own bracketing controls, brackets())}. Per comparator: ROBUSTNESS_CONFIRMED iff NR.robustness is
    ROBUSTNESS_CONFIRMED (AGG_POSITIVE on the robustness cells and on all cells), the registered cells are present and the refit is
    REFIT_GENUINELY_EFFECTIVE; NOT_RUN if the entry is missing; else ROBUSTNESS_NOT_CONFIRMED (reasons). Family label over the
    decisive set {TWIN, bracketing controls}: CONFIRMED iff all CONFIRMED; NOT_RUN if any NOT_RUN; else NOT_CONFIRMED. NEURAL_RAW and
    H_LJ are reported beside."""
    rec = rec or {}
    why = _gate_label(rec.get('grid_valid'))
    if why:
        return {'label': why}
    s = rec.get('strongest') or {}
    if s.get('label') != 'SELECTED':
        return {'label': 'NOT_RUN', 'reason': 'NO_FEASIBLE_NONZERO_RELEASE'}
    refit_ok = (rec.get('refit') or {}).get('label') == 'REFIT_GENUINELY_EFFECTIVE'
    br = [(rec.get('brackets') or {}).get(k) for _, k in D2_PATHS]
    br = [(b.get('bracket') if isinstance(b, dict) else b) for b in br]          # an id, or a brackets() record
    decisive = list(dict.fromkeys(['TWIN'] + [q for q in br if q]))
    per = {}
    for c in decisive + [r for r in NR.TARGET_REFS if r not in decisive]:
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
        reasons = ([] if r['label'] == 'ROBUSTNESS_CONFIRMED' else ['NOT_AGG_POSITIVE_ON_ROBUSTNESS_OR_ALL_CELLS']) + \
                  ([] if refit_ok else ['REFIT_NOT_GENUINELY_EFFECTIVE']) + ([] if has_reg else ['REGISTERED_CELLS_ABSENT'])
        rob = r['robustness_cells']['aggregate']
        per[c] = {'label': 'ROBUSTNESS_CONFIRMED' if not reasons else 'ROBUSTNESS_NOT_CONFIRMED', 'reasons': reasons,
                  'nrm_label': r['label'], 'robustness_mean': rob.get('mean'), 'robustness_se': rob.get('se_agg'),
                  'n_eff_robustness_cells': rob.get('n_eff_distinct_cells'),
                  'reader_seed_qualifier': rob.get('reader_seed_qualifier'),
                  'all_cells_label': r['all_cells']['label']}
    labs = [per[c]['label'] for c in decisive]
    fam = ('NOT_RUN' if any(x == 'NOT_RUN' for x in labs) else
           'ROBUSTNESS_CONFIRMED' if all(x == 'ROBUSTNESS_CONFIRMED' for x in labs) else 'ROBUSTNESS_NOT_CONFIRMED')
    return {'label': fam, 'release': s.get('selected'), 'decisive': decisive, 'per_comparator': per,
            'refit': (rec.get('refit') or {}).get('label'), 'counted_once': (rec.get('refit') or {}).get('counted_once'),
            'note': 'descriptive; never changes D2; repeated reader splits and reader seeds on the same people are not replication'}
