"""Frozen statistical rules of the PCRL finite-solver adequacy campaign (`pcrl_finite_solver_adequacy_v1`; statistical and
challenger reviewer; written 2026-10-21 before the registration and before any outcome of this campaign; no label read).

Human-readable rules: results/pcrl_finite_solver_adequacy_v1/roles/statistical_reviewer/RULES.md
Machine-readable constants: .../RULES.json (schema FSA1-rules-v1); `load_rules()` refuses a file that disagrees with this module or
a changed inherited rule file.
Invented-data tests: `python3 -m fsa.test_rules` (prints JSON; exit 0 iff every test passes).

Pure functions on plain dicts / floats. Nothing here reads data or labels. Inherited machinery is imported UNCHANGED and never
monkeypatched: `nrm.rules` as NR (point eligibility, reserve, aggregate / benefit / recipient screens, comparison, frontier
comparison, claims G4-G6) and `frc.rules` as FR (f4, targeting, the reader-budget reference rule _d3 / _d4, strongest_nonzero,
_path). New here (RULES.md section in brackets):

  route_d(rec)                       advancement route D, independently verified implementation defect          [2]
  route_e(rec)                       advancement route E, exact-reference comparison on registered invented laws [3]
  route_s(rec)                       advancement route S, cached MODEL_FIT statistics + nested MODEL_FIT        [4]
  advancement(rec)                   the advancement decision and the closing rule                              [5]
  reader_validity(...), refit_effective(...)   FSA1 reader-grid validity (+800; 300-700 forbidden)            [6]
  operating_path(...), c1(...), c3(rec), revised_vs_original(rec), pilot(rec), diagnostics(rec)   the pilot   [7]
  p1_replay(recorded, replayed, verifier)      P1 replay acceptance                                            [8]

Sign convention: every difference is revised - original (or candidate - comparator) of a LOSS, so negative favours the revised
solver; accuracy enters as a loss; AUC differences are revised - original (negative = lower recovery). Pair benefits taken from
nrm.rules keep their own convention (comparator AUC - candidate AUC; positive favours the candidate). Versions are 0 and 1.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

from nrm import rules as NR                      # inherited claims machinery, imported unchanged (never monkeypatched)
from frc import rules as FR                      # FRC1 rules, imported unchanged (never monkeypatched)

RULES_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_finite_solver_adequacy_v1' / 'roles'
              / 'statistical_reviewer' / 'RULES.json')
SCHEMA = 'FSA1-rules-v1'
NRM1_RULES_SHA256 = '84f187768817e79ef9f2bb36151bea81f00b8f62d9179c8cd523069550674652'
FRC1_RULES_SHA256 = 'a0dc0fc7f377c26339df0c96e6abda9784d725ccab8cb204e1cfa7d789029510'     # the registered FRC1 RULES.json
P1_PRC_SHA256 = '1192f34dc931fee3426be3423b807296b7fd79c47b098f35ffbea3f1f7ad210a'         # nrm/fa/prc.py (imported unchanged)
P1_ROUTE_SHA256 = '933d0d6f9d8c97bf04bfd075ac4b1db166a81ae5921524748bb74bb414eb5988'       # frc/fa/route.py (the P1 estimator)
VERSIONS = NR.VERSIONS
Z = 2.0
# ---- tolerances (RULES.md section 1)
OBJ_TIE = 1e-9                  # |J_a - J_b| <= 1e-9 nats: an exact objective tie
AUC_MATERIAL = 0.001            # |AUC difference| < 0.001: no material change; <= -0.001: material reduction
TASK_NO_WORSE = 0.0005          # a task-loss increase < 0.0005 (nats NLL, accuracy loss, summed Brier) is 'no worse'
TASK_MATERIAL = 0.001           # a total-NLL (income + occupation) reduction >= 0.001 nats is a material task improvement
RECIPIENT_LIMIT = NR.RECIPIENT_LIMIT                                    # 0.010 single-view allowance (inherited)
EPS = 1e-12                     # numerical guard: a difference equal to a stated threshold up to rounding counts as equal to it
REPLAY_TOL = 1e-12              # coordinator's P1 replay vs the recorded SELECTION_V<v>.json (same deterministic code)
VERIFY_TOL = 1e-9               # verifier's independent recomputation vs the replay (other code, other summation order)
MAX_REVISIONS = 2               # registered solver variants per campaign (card FA_FSA_v2: SV1, and SV2 only by amendment)
LAMBDA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BUDGETS = ('S1', 'S2', 'S3')
GATING_BUDGET = 'S2'
MAX_NONZERO_POINTS = 3
TASK_METRICS = ('inc_nll', 'occ_nll', 'inc_acc_loss', 'occ_acc_loss', 'inc_brier', 'occ_brier', 'occ_brier_men',
                'occ_brier_women')
DOCUMENTED_APPROXIMATIONS = ('SHRUNK_ROUTING_TABLES', 'CELL_MEAN_CODEWORD_REFRESH', 'B1_EMISSION_DIFFERS_FROM_SEARCH_PROTOTYPES',
                             'SEVEN_TILT_FIRST_IMPROVEMENT_SEARCH')
SPEC_SOURCES = ('FA_PRC_v1 section 1', 'FA_FRC_v1', 'FA_FRC_R1', 'code docstring contract')
# ---- reader validity (RULES.md section 6)
SEED_OFFSET = 800
FORBIDDEN_OFFSETS = (300, 400, 500, 600, 700)
SPLIT_TAG = 'FSA1_devgrid_R{r}'
BOOT_SEED_BASE = 2026102200
LARGE_SUPPORT_MIN = 2048
# ---- labels
D_DEFECT_LABELS = ('NOT_A_DEFECT_DOCUMENTED_APPROXIMATION', 'NOT_A_DEFECT_NO_COMMITTED_SPECIFICATION',
                   'NOT_A_DEFECT_NO_CONTRADICTION', 'NOT_A_DEFECT_NOT_IN_REAL_PIPELINE', 'NOT_A_DEFECT_NO_EFFECT',
                   'DEFECT_UNVERIFIED', 'DEFECT_VERIFIED_IMMATERIAL', 'DEFECT_VERIFIED_MATERIAL')
ROUTE_LABELS = {
    'D': ('D_ADVANCES', 'D_NO_MATERIAL_DEFECT', 'INCOMPLETE'),
    'E': ('E_ADVANCES', 'E_OLD_MATCHES_REFERENCE', 'E_PROXY_ONLY', 'E_REVISED_WORSE_SOMEWHERE', 'E_REVISED_NOT_DEPLOYABLE',
          'E_GAP_WITHOUT_RUNNABLE_REVISION', 'E_NO_DISTINCT_SOURCE_POINT', 'E_INVALID', 'E_SUPERSEDED_BY_S', 'INCOMPLETE'),
    'S': ('S_ADVANCES', 'S_OLD_MATCHES_REFERENCE', 'S_PROXY_ONLY', 'S_REVISED_WORSE_SOMEWHERE', 'S_REVISED_NOT_DEPLOYABLE',
          'S_NO_FEASIBLE_REVISED_POINT', 'S_INVALID', 'INCOMPLETE')}
CLOSING = {'D': ('D_NO_MATERIAL_DEFECT',),
           'E': ('E_OLD_MATCHES_REFERENCE', 'E_PROXY_ONLY', 'E_REVISED_WORSE_SOMEWHERE', 'E_REVISED_NOT_DEPLOYABLE',
                 'E_GAP_WITHOUT_RUNNABLE_REVISION', 'E_NO_DISTINCT_SOURCE_POINT'),
           'S': ('S_OLD_MATCHES_REFERENCE', 'S_PROXY_ONLY', 'S_REVISED_WORSE_SOMEWHERE', 'S_REVISED_NOT_DEPLOYABLE',
                 'S_NO_FEASIBLE_REVISED_POINT')}
ADVANCEMENT_LABELS = ('ADVANCE', 'OPTIMIZER_REPAIR_CLOSED', 'INCOMPLETE')
C1_LABELS = ('BETTER_OPTIMIZATION', 'MIXED', 'NOT_BETTER', 'INCOMPLETE')
C3_COMPARISONS = ('TWIN', 'HQ_FRONTIER', 'CONTROL_FRONTIER')
RVO_LABELS = ('RECOVERY_GAIN_AT_MATCHED_SPEND', 'TASK_GAIN_AT_NO_WORSE_RECOVERY', 'NO_MEASURED_GAIN_OVER_ORIGINAL', 'INCOMPLETE')
OUTCOMES = ('METHOD_CANDIDATE', 'COMPONENT', 'ENGINEERING_RESULT', 'BOUNDED_NEGATIVE_RESULT', 'INCOMPLETE')
P1_LABELS = ('P1_REPLAY_VERIFIED', 'P1_NOT_REPRODUCED', 'P1_VERIFICATION_DISAGREES', 'P1_ESTIMATOR_CHANGED', 'INCOMPLETE')
P1_FIELDS = ('nested_means', 'audit_spend_over_twin', 'brackets', 'nested_lookup_pair', 'F4', 'selection')
P1_VERIFIER_MIN = ('audit_spend_over_twin', 'selection', 'F4', 'nested_lookup_pair')
P1_TEXT_KEYS = ('rule', 'bootstrap', 'verdict_note', 'note')
CLOSING_STATEMENT = ('the optimizer-repair hypothesis closes: the original solver matches the applicable reference, or a revised '
                     'solver improves only the surrogate objective without a usefulness / recovery rationale')
SCOPE = NR.SCOPE + ('The two source versions share every training row and the MODEL_SELECT people: procedure repeats, not '
                    'replications.',
                    'Cached MODEL_FIT out-of-fold statistics are derived from MODEL_FIT training labels (income, occupation, SEX).',
                    'A better optimizer of the stated objective is not a novel privacy algorithm.')

_cache = {}
_SEEDKEY = re.compile(r'S([01])k([0-9])')


# ---------------------------------------------------------------- rule file
def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_rules(path=RULES_PATH):
    """RULES.json with its SHA-256; refuses a schema or constant that disagrees with this module, or a changed NRM1 / FRC1 rule
    file."""
    p = Path(path)
    if str(p) not in _cache:
        raw = p.read_bytes()
        r = json.loads(raw)
        if r.get('schema') != SCHEMA:
            raise ValueError(f'unexpected rules schema {r.get("schema")!r}')
        if NR.load_rules().get('_sha256') != NRM1_RULES_SHA256:
            raise ValueError('the NRM1 rule file differs from the registered one')
        if FR.load_rules().get('_sha256') != FRC1_RULES_SHA256:
            raise ValueError('the FRC1 rule file differs from the registered one')
        t, rv, pi, p1 = r['tolerances'], r['reader_validity'], r['pilot'], r['p1_replay']
        checks = {
            'objective tie': (t['objective_tie_abs'], OBJ_TIE), 'auc material': (t['auc_material'], AUC_MATERIAL),
            'task no worse': (t['task_no_worse'], TASK_NO_WORSE), 'task material': (t['task_material_total_nll'], TASK_MATERIAL),
            'recipient limit': (t['recipient_allowance_auc'], RECIPIENT_LIMIT), 'z': (t['z'], Z),
            'numerical guard': (t['numerical_guard_abs'], EPS),
            'replay tolerance': (p1['reproduction_tolerance_abs'], REPLAY_TOL),
            'verifier tolerance': (p1['verifier_tolerance_abs'], VERIFY_TOL),
            'p1 code': ((p1['estimator_sha256']['nrm/fa/prc.py'], p1['estimator_sha256']['frc/fa/route.py']),
                        (P1_PRC_SHA256, P1_ROUTE_SHA256)),
            'max revisions': (r['advancement']['max_registered_revisions'], MAX_REVISIONS),
            'documented approximations': (tuple(r['advancement']['route_D']['documented_approximations']),
                                          DOCUMENTED_APPROXIMATIONS),
            'spec sources': (tuple(r['advancement']['route_D']['specification_sources']), SPEC_SOURCES),
            'lambda grid': (tuple(r['advancement']['registered_strengths']), LAMBDA_GRID),
            'task metrics': (tuple(r['advancement']['task_metrics']), TASK_METRICS),
            'seed offset': (rv['seed_offset'], SEED_OFFSET), 'forbidden offsets': (tuple(rv['forbidden_offsets']), FORBIDDEN_OFFSETS),
            'model seed base': (rv['model_seed_base'], NR.MODEL_SEED_BASE), 'split tag': (rv['split_tag'], SPLIT_TAG),
            'bootstrap seed base': (rv['bootstrap_seed_base'], BOOT_SEED_BASE),
            'large support': (rv['large_support_min_distinct_inputs'], LARGE_SUPPORT_MIN),
            'gating budget': (pi['gating_budget'], GATING_BUDGET), 'max nonzero points': (pi['max_nonzero_points_per_version'],
                                                                                      MAX_NONZERO_POINTS),
            'outcomes': (tuple(pi['outcome_classes']), OUTCOMES), 'c3 comparisons': (tuple(pi['C3']['comparisons']), C3_COMPARISONS),
            'p1 labels': (tuple(p1['labels']), P1_LABELS), 'advancement labels': (tuple(r['advancement']['labels']),
                                                                                ADVANCEMENT_LABELS),
        }
        bad = [k for k, (a, b) in checks.items() if a != b]
        if bad:
            raise ValueError(f'RULES.json disagrees with fsa.rules on {bad}')
        r['_sha256'] = hashlib.sha256(raw).hexdigest()
        _cache[str(p)] = r
    return _cache[str(p)]


# ---------------------------------------------------------------- small helpers
def _fin(*xs):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs)


def _v(d, v):
    if not isinstance(d, dict):
        return None
    return d[v] if v in d else d.get(str(v))


def _lam(d, lam):
    """Strength entry of a dict keyed by floats or their string forms ('8.0', '8')."""
    if not isinstance(d, dict):
        return None
    for k in (lam, str(lam), str(float(lam)), repr(float(lam))):
        if k in d:
            return d[k]
    return None


def _est(x):
    """(estimate, SE) from (est, se) or {'value', 'se'}; None unless both finite and SE >= 0."""
    if isinstance(x, dict):
        x = (x.get('value'), x.get('se'))
    if not isinstance(x, (tuple, list)) or len(x) < 2:
        return None
    e, s = x[0], x[1]
    return (float(e), float(s)) if _fin(e, s) and s >= 0 else None


def _deltas(rev, orig):
    """revised - original of every task loss and of the exact AUC of every view; None if anything is missing."""
    tr, to = (rev or {}).get('task') or {}, (orig or {}).get('task') or {}
    ar, ao = (rev or {}).get('auc') or {}, (orig or {}).get('auc') or {}
    if not all(_fin(tr.get(m), to.get(m)) for m in TASK_METRICS) or not all(_fin(ar.get(w), ao.get(w)) for w in NR.VIEWS):
        return None
    d = {m: float(tr[m]) - float(to[m]) for m in TASK_METRICS}
    d['tot_nll'] = d['inc_nll'] + d['occ_nll']
    d.update({f'auc_{w}': float(ar[w]) - float(ao[w]) for w in NR.VIEWS})
    return d


def usefulness(d):
    """Usefulness / recovery evidence from exact (invented-law) differences revised - original (RULES.md 3.4):
    RECOVERY iff the pair AUC falls by >= AUC_MATERIAL, each single-view AUC rises by <= RECIPIENT_LIMIT and every task loss rises by
    < TASK_NO_WORSE; TASK iff the total NLL falls by >= TASK_MATERIAL, every task loss rises by < TASK_NO_WORSE, the pair AUC rises by
    < AUC_MATERIAL and each single view by <= RECIPIENT_LIMIT. MISALIGNED (a flag) iff the pair AUC rises by >= AUC_MATERIAL."""
    if d is None:
        return None
    no_worse = all(_lt(d[m], TASK_NO_WORSE) for m in TASK_METRICS)
    views_ok = all(_le(d[f'auc_{w}'], RECIPIENT_LIMIT) for w in NR.SINGLE_VIEWS)
    rec = no_worse and views_ok and _le(d['auc_pair'], -AUC_MATERIAL)
    task = no_worse and views_ok and _le(d['tot_nll'], -TASK_MATERIAL) and _lt(d['auc_pair'], AUC_MATERIAL)
    return {'recovery': rec, 'task': task, 'useful': rec or task, 'misaligned': not _lt(d['auc_pair'], AUC_MATERIAL), 'deltas': d}


def _le(x, t):
    """x <= t with the numerical guard EPS (x equal to t up to rounding counts as equal)."""
    return x <= t + EPS


def _lt(x, t):
    """x < t with the numerical guard EPS (x equal to t up to rounding is not below it)."""
    return x < t - EPS


# ---------------------------------------------------------------- route D: an independently verified implementation defect
def defect_label(x):
    """One claimed defect (RULES.md 2). x = {'id', 'location', 'in_real_pipeline': bool, 'specification': {'source': one of
    SPEC_SOURCES, 'quote': committed text}, 'approximation': None or a DOCUMENTED_APPROXIMATIONS name, 'contradiction_demonstrated':
    bool, 'changes': {'partition_people_changed': int, 'release_max_abs_change': float, 'decision_inputs': [{'name', 'kind':
    'discrete' | 'auc' | 'task' | 'other', 'old', 'new'}]}, 'verified': bool (the verifier reproduced contradiction and change)}."""
    if (x or {}).get('approximation') in DOCUMENTED_APPROXIMATIONS:
        return 'NOT_A_DEFECT_DOCUMENTED_APPROXIMATION'
    sp = (x or {}).get('specification') or {}
    if sp.get('source') not in SPEC_SOURCES or not str(sp.get('quote') or '').strip():
        return 'NOT_A_DEFECT_NO_COMMITTED_SPECIFICATION'
    if x.get('contradiction_demonstrated') is not True:
        return 'NOT_A_DEFECT_NO_CONTRADICTION'
    if x.get('in_real_pipeline') is not True:
        return 'NOT_A_DEFECT_NOT_IN_REAL_PIPELINE'
    ch = x.get('changes') or {}
    people = ch.get('partition_people_changed') or 0
    rel = ch.get('release_max_abs_change') or 0.0
    inputs = ch.get('decision_inputs') or []
    moved = []
    for i in inputs:
        if i.get('kind') == 'discrete' or not _fin(i.get('old'), i.get('new')):
            if i.get('old') != i.get('new'):
                moved.append((i, math.inf))
        elif abs(float(i['new']) - float(i['old'])) > REPLAY_TOL:
            moved.append((i, abs(float(i['new']) - float(i['old']))))
    if not (people > 0 or (_fin(rel) and rel > REPLAY_TOL) or moved):
        return 'NOT_A_DEFECT_NO_EFFECT'
    if x.get('verified') is not True:
        return 'DEFECT_UNVERIFIED'
    material = people > 0 or any(
        i.get('kind') == 'discrete' or (i.get('kind') == 'auc' and not _lt(d, AUC_MATERIAL))
        or (i.get('kind') == 'task' and not _lt(d, TASK_NO_WORSE)) for i, d in moved)
    return 'DEFECT_VERIFIED_MATERIAL' if material else 'DEFECT_VERIFIED_IMMATERIAL'


def route_d(rec):
    """Route D (RULES.md 2): D_ADVANCES iff some claimed defect is DEFECT_VERIFIED_MATERIAL; INCOMPLETE iff none is and some claim
    is still DEFECT_UNVERIFIED; D_NO_MATERIAL_DEFECT otherwise (no claim, or every claim not a defect / immaterial)."""
    claims = (rec or {}).get('claims')
    if claims is None:
        return {'label': 'INCOMPLETE', 'error': 'defect claims not recorded (an empty list means none was claimed)'}
    per = {str(c.get('id', i)): defect_label(c) for i, c in enumerate(claims)}
    if any(x == 'DEFECT_VERIFIED_MATERIAL' for x in per.values()):
        lab = 'D_ADVANCES'
    elif any(x == 'DEFECT_UNVERIFIED' for x in per.values()):
        lab = 'INCOMPLETE'
    else:
        lab = 'D_NO_MATERIAL_DEFECT'
    return {'label': lab, 'per_claim': per, 'old_records_edited': False,
            'note': 'a verified defect never edits an earlier record; corrected values are reported beside in a dated note'}


# ---------------------------------------------------------------- route E: exact-reference comparison on registered invented laws
def _cell_key(c):
    return (str(c.get('law')), float(c.get('lam')) if _fin(c.get('lam')) else None)


def route_e(rec):
    """Route E for one registered revision (RULES.md 3). rec = {'revision_id', 'laws_registered_before_run': bool,
    'solver_registered_before_run': bool, 'runnable_on_source': bool, 'source_distinct': {v: bool}, 'registered_cells': [[law,
    lam], ...], 'cells': [{'law', 'lam', 'same_start', 'same_objective', 'same_capacity', 'same_rule_family': bools, 'original':
    point, 'revised': point, 'reference': None or point (+ 'same_class': bool), 'unrestricted': None or point}]}; point = {'J',
    'deployable': bool, 'auc': {view: exact Bayes AUC}, 'task': {metric: exact loss}}. Unrestricted points are reported only."""
    r = rec or {}
    errs = [f'{k} not true' for k in ('laws_registered_before_run', 'solver_registered_before_run') if r.get(k) is not True]
    cells = r.get('cells') or []
    reg = {(str(a), float(b)) for a, b in (r.get('registered_cells') or [])}
    got = [_cell_key(c) for c in cells]
    if not reg or set(got) != reg or len(got) != len(set(got)):
        errs.append('the reported cells differ from the registered (law, strength) cells (every registered cell is reported once)')
    rows, flags = [], []
    for c in cells:
        o, rv, ref = c.get('original') or {}, c.get('revised') or {}, c.get('reference')
        bad = [k for k in ('same_start', 'same_objective', 'same_capacity', 'same_rule_family') if c.get(k) is not True]
        if bad or not _fin(o.get('J'), rv.get('J')) or o.get('deployable') is not True:
            errs.append(f'cell {_cell_key(c)}: invalid comparison ({bad or "objective missing or original not deployable"})')
            continue
        best = [rv['J']] if rv.get('deployable') is True else []
        if isinstance(ref, dict) and ref.get('deployable') is True and ref.get('same_class') is True and _fin(ref.get('J')):
            best.append(float(ref['J']))
        jbest = min(best) if best else None
        u = usefulness(_deltas(rv, o))
        row = {'cell': list(_cell_key(c)), 'J_original': o['J'], 'J_revised': rv['J'], 'revised_deployable': rv.get('deployable') is True,
               'original_matches_reference': jbest is None or o['J'] <= jbest + OBJ_TIE,
               'revised_improves': rv.get('deployable') is True and rv['J'] < o['J'] - OBJ_TIE,
               'revised_worse': rv.get('deployable') is True and rv['J'] > o['J'] + OBJ_TIE, 'usefulness': u}
        unr = c.get('unrestricted')
        if isinstance(unr, dict) and _fin(unr.get('J')):
            row['unrestricted_gap_reported_only'] = o['J'] - float(unr['J'])
        if row['revised_improves'] and u is None:
            errs.append(f'cell {_cell_key(c)}: exact AUC or task metrics missing')
        if u and row['revised_improves'] and u['misaligned']:
            flags.append(f'OBJECTIVE_RECOVERY_MISALIGNED_{row["cell"][0]}@{row["cell"][1]}')
        rows.append(row)
    out = {'revision_id': r.get('revision_id'), 'cells': rows, 'flags': sorted(set(flags)),
           'note': 'exact law-level evaluation; statistical significance is not required; unrestricted partition enumeration is '
                   'reported only and never advances'}
    if errs:
        return {**out, 'label': 'E_INVALID', 'errors': errs}
    useful = [x for x in rows if x['revised_improves'] and x['usefulness']['useful']]
    if any(not x['revised_deployable'] for x in rows):
        lab = 'E_REVISED_NOT_DEPLOYABLE'
    elif any(x['revised_worse'] for x in rows):
        lab = 'E_REVISED_WORSE_SOMEWHERE'
    elif useful:
        if r.get('runnable_on_source') is not True:
            lab = 'E_GAP_WITHOUT_RUNNABLE_REVISION'
        elif not all(_v(r.get('source_distinct'), v) is True for v in VERSIONS):
            lab = 'E_NO_DISTINCT_SOURCE_POINT'
        else:
            lab = 'E_ADVANCES'
    elif all(x['original_matches_reference'] for x in rows):
        lab = 'E_OLD_MATCHES_REFERENCE'
    elif any(x['revised_improves'] for x in rows):
        lab = 'E_PROXY_ONLY'
    else:
        lab = 'E_GAP_WITHOUT_RUNNABLE_REVISION'
    return {**out, 'label': lab, 'useful_cells': [x['cell'] for x in useful],
            'improving_cells': [x['cell'] for x in rows if x['revised_improves']]}


# ---------------------------------------------------------------- route S: source-specific (cached MODEL_FIT statistics, nested)
def _bracket(path, r_spend):
    """The cheapest original path point whose nested spend is >= the revised point's spend - TASK_NO_WORSE (path order = increasing
    strength); the most-spending point when none is (then the no-worse spend check fails by construction)."""
    pts = [p for p in path or [] if _fin(p.get('spend'))]
    if not pts:
        return None
    ok = [p for p in pts if p['spend'] >= r_spend - TASK_NO_WORSE]
    return min(ok, key=lambda p: p['spend'])['id'] if ok else max(pts, key=lambda p: p['spend'])['id']


def route_s(rec):
    """Route S for one registered revision (RULES.md 4). rec = {'revision_id', 'card_committed_before_run': bool,
    'model_select_opened': bool, 'label_derived_statistics_disclosed': bool, 'fitted': {v: {lam: {'original': J, 'revised': J,
    'revised_deployable', 'distinct_from_original', 'distinct_from_twin': bools}}} (full-pool cached MODEL_FIT out-of-fold
    statistics), 'nested': {v: {'revised_point': id or None, 'revised_spend': float, 'original_s2': id, 'original_path': [{'id',
    'spend'}], 'paired': {original id: {'d_auc': (est, se), 'd_spend': (est, se)}}}} (MODEL_FIT nested held-out; revised - original;
    d_auc = pair LOOKUP AUC, paired household-bootstrap SE; d_spend = total NLL, household-clustered SE)}."""
    r = rec or {}
    errs = []
    if r.get('card_committed_before_run') is not True:
        errs.append('card not committed before the run')
    if r.get('model_select_opened') is not False:
        errs.append('MODEL_SELECT opened (or not recorded as unopened)')
    if r.get('label_derived_statistics_disclosed') is not True:
        errs.append('the record must state that the cached statistics are derived from MODEL_FIT training labels')
    if r.get('nested') is not None and r.get('original_entries_equal_replay') is not True:
        errs.append("the nested run's original-solver entries must equal the replay's per-person outputs (card FA_FSA_v2 3)")
    out = {'revision_id': r.get('revision_id'), 'statistics': 'cached MODEL_FIT out-of-fold statistics, derived from MODEL_FIT '
                                                              'training labels; nested estimates on MODEL_FIT only'}
    if errs:
        return {**out, 'label': 'S_INVALID', 'errors': errs}
    fit = r.get('fitted') or {}
    rows = {}
    for v in VERSIONS:
        fv = _v(fit, v)
        for lam in LAMBDA_GRID:
            e = _lam(fv, lam)
            if not isinstance(e, dict) or not _fin(e.get('original'), e.get('revised')):
                return {**out, 'label': 'INCOMPLETE', 'error': f'fitted objective missing (version {v}, strength {lam})'}
            rows[(v, lam)] = {'J_original': e['original'], 'J_revised': e['revised'], 'deployable': e.get('revised_deployable') is True,
                              'distinct': e.get('distinct_from_original') is True and e.get('distinct_from_twin') is True,
                              'improves': e['revised'] < e['original'] - OBJ_TIE, 'worse': e['revised'] > e['original'] + OBJ_TIE}
    out['fitted'] = {f'V{v}@{lam}': x for (v, lam), x in rows.items()}
    if any(not x['deployable'] for x in rows.values()):
        return {**out, 'label': 'S_REVISED_NOT_DEPLOYABLE'}
    if any(x['worse'] for x in rows.values()):
        return {**out, 'label': 'S_REVISED_WORSE_SOMEWHERE'}
    good = [lam for lam in LAMBDA_GRID if all(rows[(v, lam)]['improves'] and rows[(v, lam)]['distinct'] for v in VERSIONS)]
    out['improving_strengths_both_versions'] = good
    if not good:
        return {**out, 'label': 'S_OLD_MATCHES_REFERENCE'}
    nest, per, flags = r.get('nested') or {}, {}, []
    for v in VERSIONS:
        n = _v(nest, v)
        if not isinstance(n, dict):
            return {**out, 'label': 'INCOMPLETE', 'error': f'nested record of version {v} missing'}
        if n.get('revised_point') is None:
            return {**out, 'label': 'S_NO_FEASIBLE_REVISED_POINT', 'version': v}
        if not _fin(n.get('revised_spend')):
            return {**out, 'label': 'INCOMPLETE', 'error': f'revised nested spend of version {v} missing'}
        br = _bracket(n.get('original_path'), float(n['revised_spend']))
        o2 = n.get('original_s2')
        pr = n.get('paired') or {}
        a_br, a_o2 = pr.get(br) or {}, pr.get(o2) or {}
        need = [_est(a_br.get('d_auc')), _est(a_br.get('d_spend')), _est(a_o2.get('d_auc')), _est(a_o2.get('d_spend'))]
        views = {f'{w}|{nm}': _est(a.get(f'd_auc_{w}')) for w in NR.SINGLE_VIEWS for nm, a in (('bracket', a_br), ('s2', a_o2))}
        if br is None or o2 is None or any(x is None for x in need) or any(x is None for x in views.values()):
            return {**out, 'label': 'INCOMPLETE', 'error': f'version {v}: bracket, original S2 point or paired differences missing'}
        if n.get('revised_point') == o2 or n.get('alias_of_original_s2') is True:
            flags.append(f'ALIAS_OF_ORIGINAL_AT_S2_V{v}')
        per[v] = {'bracket': br, 'original_s2': o2, 'd_auc_vs_bracket': need[0], 'd_spend_vs_bracket': need[1],
                  'd_auc_vs_original_s2': need[2], 'd_spend_vs_original_s2': need[3], 'single_views': views,
                  'single_views_ok_bracket': all(_le(views[f'{w}|bracket'][0], RECIPIENT_LIMIT) for w in NR.SINGLE_VIEWS),
                  'single_views_ok_s2': all(_le(views[f'{w}|s2'][0], RECIPIENT_LIMIT) for w in NR.SINGLE_VIEWS)}
    pooled_auc = (sum(per[v]['d_auc_vs_bracket'][0] for v in VERSIONS) / 2, sum(per[v]['d_auc_vs_bracket'][1] for v in VERSIONS) / 2)
    pooled_sp = (sum(per[v]['d_spend_vs_original_s2'][0] for v in VERSIONS) / 2,
                 sum(per[v]['d_spend_vs_original_s2'][1] for v in VERSIONS) / 2)
    rec_ok = (all(_le(per[v]['d_auc_vs_bracket'][0], -AUC_MATERIAL) and _lt(per[v]['d_spend_vs_bracket'][0], TASK_NO_WORSE)
                  and per[v]['single_views_ok_bracket'] for v in VERSIONS) and pooled_auc[0] + Z * pooled_auc[1] < 0)
    task_ok = (all(_le(per[v]['d_spend_vs_original_s2'][0], -TASK_MATERIAL) and _lt(per[v]['d_auc_vs_original_s2'][0], AUC_MATERIAL)
                   and per[v]['single_views_ok_s2'] for v in VERSIONS) and pooled_sp[0] + Z * pooled_sp[1] < 0)
    if any(f.startswith('ALIAS_OF_ORIGINAL') for f in flags):
        rec_ok = task_ok = False
    if all(not _lt(per[v]['d_auc_vs_original_s2'][0], AUC_MATERIAL) for v in VERSIONS):
        flags.append('OBJECTIVE_RECOVERY_MISALIGNED')
    out.update({'per_version': per, 'pooled_d_auc_vs_bracket': pooled_auc, 'pooled_d_spend_vs_original_s2': pooled_sp,
                'recovery_rationale': rec_ok, 'task_rationale': task_ok, 'flags': sorted(set(flags)),
                'pooled_se_rule': 'mean of the version SEs (bound under any correlation of the versions, which share rows)'})
    return {**out, 'label': 'S_ADVANCES' if (rec_ok or task_ok) else 'S_PROXY_ONLY'}


# ---------------------------------------------------------------- the advancement decision and the closing rule
def advancement(rec):
    """RULES.md 5. rec = {'registered_revisions': [ids, registration order], 'attempted': {'D': bool, 'E': bool, 'S': bool},
    'withdrawn': [route names withdrawn by a dated record after an invalid evaluation], 'D': route_d input, 'E': {revision:
    route_e input}, 'S': {revision: route_s input}}. ADVANCE iff some attempted route advances (the pilot solver: the first
    registered revision that advances by E or S, with every verified material defect corrected; the corrected original solver if
    only D advances). OPTIMIZER_REPAIR_CLOSED iff no route advances and every attempted route reached a closing label (at least one
    route attempted). INCOMPLETE otherwise."""
    r = rec or {}
    revs = list(r.get('registered_revisions') or [])
    att = r.get('attempted') or {}
    wd = set(r.get('withdrawn') or [])
    flags, errs = [f'ROUTE_{x}_WITHDRAWN_AFTER_INVALID' for x in sorted(wd)], []
    if len(revs) > MAX_REVISIONS or len(set(revs)) != len(revs):
        errs.append(f'at most {MAX_REVISIONS} distinct registered revisions')
    res = {}
    if att.get('D') and 'D' not in wd:
        res['D'] = {'_': route_d(r.get('D'))}
    for route, fn in (('E', route_e), ('S', route_s)):
        if not att.get(route) or route in wd:
            continue
        got = r.get(route) or {}
        missing = [x for x in revs if x not in got]
        extra = [x for x in got if x not in revs]
        if missing or extra or not revs:
            errs.append(f'route {route}: every registered revision must be evaluated and only those (missing {missing}, '
                        f'unregistered {extra})')
        res[route] = {x: fn(got[x]) for x in revs if x in got}
    labels = {route: {x: y['label'] for x, y in d.items()} for route, d in res.items()}
    for x in revs:                    # source-specific evidence governs its revision: a closing S label overrides E (RULES.md 5.2)
        se, ee = labels.get('S', {}).get(x), labels.get('E', {}).get(x)
        if ee == 'E_ADVANCES' and se is not None and se != 'S_ADVANCES':
            labels['E'][x] = 'E_SUPERSEDED_BY_S' if se in CLOSING['S'] else 'INCOMPLETE'
            flags.append(f'E_SUPERSEDED_BY_S_{x}')
    ctl = {route: {c: fn(x) for c, x in ((r.get('controls') or {}).get(route) or {}).items()}
           for route, fn in (('E', route_e), ('S', route_s))}
    attribution = equal_effort_attribution(r.get('source_objectives'))
    out = {'labels': labels, 'routes': res, 'flags': flags, 'scope': list(SCOPE),
           'equal_effort_controls_reported_only': {route: {c: y['label'] for c, y in d.items()} for route, d in ctl.items()},
           'equal_effort_attribution': attribution,
           'note': 'an advancement is a gate for a development pilot, never a claim; equal-effort controls never advance'}
    if errs:
        return {**out, 'label': 'INCOMPLETE', 'errors': errs}
    pending = sorted(f'{route}:{x}' for route, d in labels.items() for x, y in d.items() if y in ('INCOMPLETE', f'{route}_INVALID'))
    if pending:                                   # decided only on a complete record (an invalid route may be withdrawn)
        return {**out, 'label': 'INCOMPLETE', 'errors': [f'attempted routes incomplete or invalid: {pending}']}
    adv = sorted(route for route, d in labels.items() if any(x == f'{route}_ADVANCES' for x in d.values()))
    if adv:
        pick = next((x for x in revs if any(labels.get(route, {}).get(x) == f'{route}_ADVANCES' for route in ('E', 'S'))), None)
        solver = pick if pick is not None else 'CORRECTED_ORIGINAL'
        return {**out, 'label': 'ADVANCE', 'advancing_routes': adv, 'pilot_solver': solver,
                'defects_corrected_in_pilot_solver': 'D' in adv}
    if not res:
        return {**out, 'label': 'INCOMPLETE', 'errors': ['no route attempted']}
    if all(x in CLOSING[route] + ('E_SUPERSEDED_BY_S',) for route, d in labels.items() for x in d.values()):
        return {**out, 'label': 'OPTIMIZER_REPAIR_CLOSED', 'statement': CLOSING_STATEMENT, 'routes_evaluated': sorted(res),
                'routes_not_attempted': sorted({'D', 'E', 'S'} - set(res))}
    return {**out, 'label': 'INCOMPLETE', 'errors': ['some attempted route is incomplete or invalid']}


def equal_effort_attribution(objs):
    """Descriptive attribution with the equal-effort control (RULES.md 5.3; never a decision input). objs = {v: {lam: {'original':
    J, 'revised': J, 'control': J}}} (the source stage: original, SV1, SV1_first on the full-pool cached statistics). Per cell:
    RESOLUTION = control < original - OBJ_TIE (the finer candidate set alone helps); SELECTION = revised < control - OBJ_TIE
    (best improvement adds to it); SELECTION_HURTS = revised > control + OBJ_TIE."""
    if not isinstance(objs, dict):
        return {'label': 'NOT_RECORDED', 'decision_input': False}
    cells = {}
    for v in VERSIONS:
        for lam in LAMBDA_GRID:
            e = _lam(_v(objs, v), lam)
            if not isinstance(e, dict) or not _fin(e.get('original'), e.get('revised'), e.get('control')):
                return {'label': 'INCOMPLETE', 'decision_input': False, 'error': f'version {v}, strength {lam} missing'}
            cells[f'V{v}@{lam}'] = {'resolution': e['control'] < e['original'] - OBJ_TIE,
                                    'selection': e['revised'] < e['control'] - OBJ_TIE,
                                    'selection_hurts': e['revised'] > e['control'] + OBJ_TIE}
    n = {k: sum(c[k] for c in cells.values()) for k in ('resolution', 'selection', 'selection_hurts')}
    return {'label': 'REPORTED', 'cells': cells, 'counts': n, 'decision_input': False,
            'note': 'separates best-vs-first selection from candidate resolution at equal candidate evaluations; descriptive'}


# ---------------------------------------------------------------- reader validity of a new (FSA1) reader grid
def forbidden_seeds():
    """Every earlier grid's reader seeds: MODEL_SEED_BASE + {300, 400, 500, 600, 700} + 10 v + k, k = 0..9 (100 seeds)."""
    return {NR.MODEL_SEED_BASE + off + 10 * v + k for off in FORBIDDEN_OFFSETS for v in VERSIONS for k in range(10)}


def registered_seed(v, k):
    return NR.MODEL_SEED_BASE + SEED_OFFSET + 10 * int(v) + int(k)


def registered_boot_seed(v, i):
    """Grid bootstrap seed of version v and repetition index i (R1 -> 0, R2 -> 1, ...)."""
    return BOOT_SEED_BASE + 100 * int(v) + int(i)


def _seed_errors(seeds, required_k):
    keys = [f'S{v}k{k}' for v in VERSIONS for k in required_k]
    s = seeds if isinstance(seeds, dict) else {}
    sv = [s.get(k) for k in keys] + [x for k, x in s.items() if k not in keys]
    if any(not isinstance(x, int) or isinstance(x, bool) for x in sv) or len(set(sv)) != len(sv):
        return [f'reader seeds must be recorded distinct integers, at least {keys}']
    errs = []
    bad = sorted(k for k in s if not _SEEDKEY.fullmatch(str(k)))
    if bad:
        errs.append(f'reader seed keys not of the form S<v>k<k>: {bad}')
    reused = sorted(set(sv) & forbidden_seeds())
    if reused:
        errs.append(f'reader seeds reuse earlier grid seeds {reused}')
    wrong = sorted(k for k, x in s.items() if _SEEDKEY.fullmatch(str(k)) and x != registered_seed(*_SEEDKEY.fullmatch(str(k)).groups()))
    if wrong:
        errs.append(f'reader seeds differ from the registered MODEL_SEED_BASE + {SEED_OFFSET} + 10 v + k: {wrong}')
    return errs


def _split_errors(splits, reps, boot_seeds):
    errs = []
    sp = splits if isinstance(splits, dict) else {}
    hs = [(sp.get(r) or {}).get('assignment_sha256') for r in reps]
    if not all(isinstance(h, str) and h for h in hs) or len(set(hs)) != len(hs):
        errs.append('household fitting splits not recorded or not different across repetitions')
    for i, r in enumerate(reps):
        e = sp.get(r) or {}
        if e.get('tag') != SPLIT_TAG.format(r=r[1:]):
            errs.append(f'{r}: split tag {e.get("tag")!r} is not {SPLIT_TAG.format(r=r[1:])!r}')
        if e.get('fit_score_household_overlap') != 0:
            errs.append(f'{r}: FIT / SCORE household overlap not recorded as 0')
        for q in reps:
            if q == r:
                continue
            o = (e.get('score_overlap') or {}).get(q)
            if o is None:
                o = ((sp.get(q) or {}).get('score_overlap') or {}).get(r)
            if not _fin(o) or not 0 <= o < 1:
                errs.append(f'SCORE household overlap {r}-{q} not recorded or identical')
        for v in VERSIONS:
            if (boot_seeds or {}).get(f'S{v}{r}') != registered_boot_seed(v, i):
                errs.append(f'bootstrap seed S{v}{r} is not the registered {registered_boot_seed(v, i)}')
    return sorted(set(errs))


def reader_validity(seeds, checks, distinct_inputs, splits, boot_seeds, recorded_errors, required_k=NR.REG_K, reps=NR.REG_REPS):
    """FSA1 reader-grid validity (RULES.md 6; prospective, never applied to FRC1 records). GRID_VALID iff
    (a) seeds: recorded distinct integers S<v>k<k> = MODEL_SEED_BASE + 800 + 10 v + k, none an earlier grid's (offsets 300-700);
    (b) splits: per repetition the tag FSA1_devgrid_R<r>, a recorded assignment hash distinct across repetitions, SCORE overlaps in
        [0, 1), FIT / SCORE household overlap 0, bootstrap seed 2026102200 + 100 v + repetition index;
    (c) recorded schema, access and score checks, each an empty error list;
    (d) stochastic recipes: every seed-dependent recipe is checked in every (version, repetition) block on at least one
        large-support view (recorded distinct-input count >= LARGE_SUPPORT_MIN) and differs between its two seeds on EVERY such
        check (an equality there, or no such check, is a seed-implementation error); an equality on a small-support (finite) view
        is expected seed invariance: it never invalidates and its duplicate cells count once (n_eff, nrm.rules.aggregate).
    A deterministic (seed-invariant) recipe is a legitimate reader; one that differs between seeds is flagged NONDETERMINISTIC."""
    errs = _seed_errors(seeds, required_k) + _split_errors(splits, reps, boot_seeds)
    for kind in ('schema', 'access', 'score'):
        xs = (recorded_errors or {}).get(kind)
        if xs is None:
            errs.append(f'{kind} check not recorded')
        elif xs:
            errs.append(f'{kind} errors: {list(xs)}')
    summ = {r: {'large_support': {'checks': 0, 'differ': 0}, 'small_support': {'checks': 0, 'differ': 0}} for r in NR.SEED_DEPENDENT}
    blocks = {(r, v, rp): 0 for r in NR.SEED_DEPENDENT for v in VERSIONS for rp in reps}
    expected, flags = [], []
    for c in checks or []:
        rcp = c.get('recipe')
        if rcp in NR.SEED_INVARIANT:
            if c.get('equal_k0_k1') is False:
                flags.append('NONDETERMINISTIC_RECIPE')
            continue
        if rcp not in NR.SEED_DEPENDENT:
            errs.append(f'unknown recipe {rcp!r}')
            continue
        n = ((distinct_inputs or {}).get(f"{c.get('wire')}|V{c.get('v')}") or {}).get(c.get('view'))
        if not isinstance(n, int) or isinstance(n, bool):
            errs.append(f"distinct-input count not recorded for {c.get('wire')}|V{c.get('v')}|{c.get('view')}")
            continue
        eq = c.get('equal_k0_k1')
        if eq not in (True, False):
            errs.append(f"seed check of {rcp} on {c.get('wire')}|{c.get('view')} not recorded")
            continue
        cls = 'large_support' if n >= LARGE_SUPPORT_MIN else 'small_support'
        summ[rcp][cls]['checks'] += 1
        summ[rcp][cls]['differ'] += (not eq)
        if cls == 'large_support':
            if (rcp, c.get('v'), c.get('R')) in blocks:
                blocks[(rcp, c.get('v'), c.get('R'))] += 1
            if eq:
                errs.append(f"seed-implementation error: {rcp} equal between seeds on the large-support view "
                            f"{c.get('wire')}|V{c.get('v')}|{c.get('R')}|{c.get('view')} ({n} distinct inputs)")
        elif eq:
            expected.append({k: c.get(k) for k in ('wire', 'view', 'v', 'R', 'recipe')})
    untested = sorted(f'{r}|V{v}|{rp}' for (r, v, rp), n in blocks.items() if n == 0)
    if untested:
        errs.append(f'seed implementation untested (no large-support check): {untested}')
    return {'label': 'GRID_VALID' if not errs else 'GRID_INVALID', 'valid': not errs, 'errors': errs,
            'flags': sorted(set(flags)), 'stochastic_recipes': summ, 'expected_small_support_equalities': len(expected),
            'expected_items': expected, 'seed_offset': SEED_OFFSET, 'forbidden_offsets': list(FORBIDDEN_OFFSETS),
            'note': 'expected seed invariance on finite views never invalidates; its duplicate cells count once (n_eff); '
                    'prospective rule, never applied to FRC1'}


def refit_effective(seeds, checks, distinct_inputs, splits, boot_seeds, recorded_errors):
    """Effective comparator refit of the diagnostics (RULES.md 7.8): reader_validity on repetitions R1-R4 and reader seeds k0-k3.
    Descriptive; never a decision input."""
    g = reader_validity(seeds, checks, distinct_inputs, splits, boot_seeds, recorded_errors, required_k=NR.REG_K + NR.ROB_K,
                        reps=NR.REG_REPS + NR.ROB_REPS)
    return {'label': 'REFIT_EFFECTIVE' if g['valid'] else 'REFIT_NOT_EFFECTIVE', 'validity': g, 'decision_input': False}


# ---------------------------------------------------------------- the pilot
def operating_path(selections, output_sha, chosen_on):
    """RULES.md 7.1. selections = {v: {'S1' | 'S2' | 'S3': point id or None}} from nrm.fa.prc.select_path on MODEL_FIT nested
    estimates; output_sha = {point id: {v: SHA-256 of its MODEL_FIT outputs}} including 'TWIN'; chosen_on must be
    'MODEL_FIT_NESTED'. Path per version = TWIN + the distinct nonzero selections (identical outputs count once; a selection
    identical to TWIN is an alias of the identity), at most three."""
    if chosen_on != 'MODEL_FIT_NESTED':
        return {'label': 'INVALID', 'error': 'operating points are chosen on MODEL_FIT nested estimates only'}
    out = {}
    for v in VERSIONS:
        sel = _v(selections, v) or {}
        tw = _v((output_sha or {}).get('TWIN'), v)
        if not isinstance(tw, str):
            return {'label': 'INCOMPLETE', 'error': f'TWIN output hash of version {v} missing'}
        seen, path, alias = {tw: 'TWIN'}, ['TWIN'], {}
        for b in BUDGETS:
            p = sel.get(b)
            if p is None:
                continue
            h = _v((output_sha or {}).get(p), v)
            if not isinstance(h, str):
                return {'label': 'INCOMPLETE', 'error': f'output hash of {p} (version {v}) missing'}
            if h in seen:
                if seen[h] != p:
                    alias[p] = seen[h]
                continue
            seen[h] = p
            path.append(p)
        if len(path) - 1 > MAX_NONZERO_POINTS:
            return {'label': 'INVALID', 'error': f'more than {MAX_NONZERO_POINTS} nonzero points in version {v}'}
        out[v] = {'path': path, 'aliases': alias, 'n_nonzero': len(path) - 1, 'primary': sel.get(GATING_BUDGET)}
    return {'label': 'PATH', 'per_version': out}


def c1(fitted, operating_lambda):
    """C1 better optimization (RULES.md 7.5): fitted = {v: {lam: {'original': J, 'revised': J}}} on the full-pool cached MODEL_FIT
    out-of-fold statistics (label-derived); operating_lambda = {v: strength of the revised primary point}. BETTER_OPTIMIZATION iff
    in both versions J_revised < J_original - OBJ_TIE at the operating strength and J_revised <= J_original + OBJ_TIE at every
    registered strength; MIXED if the first holds but the second fails; NOT_BETTER otherwise."""
    better, worse = True, False
    for v in VERSIONS:
        fv, lo = _v(fitted, v), _v(operating_lambda, v)
        if lo not in LAMBDA_GRID:
            return {'label': 'INCOMPLETE', 'error': f'operating strength of version {v} missing or unregistered'}
        for lam in LAMBDA_GRID:
            e = _lam(fv, lam)
            if not isinstance(e, dict) or not _fin(e.get('original'), e.get('revised')):
                return {'label': 'INCOMPLETE', 'error': f'fitted objective missing (version {v}, strength {lam})'}
            worse |= e['revised'] > e['original'] + OBJ_TIE
        e = _lam(fv, lo)
        better &= e['revised'] < e['original'] - OBJ_TIE
    lab = ('MIXED' if worse else 'BETTER_OPTIMIZATION') if better else 'NOT_BETTER'
    return {'label': lab, 'met': lab == 'BETTER_OPTIMIZATION',
            'note': 'the surrogate objective on label-derived training statistics; never usefulness or protection'}


def _effort_ok(e):
    p, h = (e or {}).get('PRC') or {}, (e or {}).get('HQ') or {}
    if not p.get('settings_sha256') or not h.get('settings_sha256') or p.get('stopped_by') not in ('CONVERGED', 'CAP') \
            or h.get('stopped_by') not in ('CONVERGED', 'CAP'):
        return None
    return p['settings_sha256'] == h['settings_sha256'] and not (h['stopped_by'] == 'CAP' and p['stopped_by'] == 'CONVERGED')


def c3(rec):
    """C3 protection beyond matched controls (RULES.md 7.5): the revised PRC primary point against the twin, the equally optimized
    revised HQ path and the registered untargeted / ordinary-compression control path. Conjuncts: (a) G1; (b) vs TWIN
    nrm.rules.comparison passes; (c) DOMINATES_CONTROL_FRONTIER vs the HQ path and the control path (an empty path is compared at
    TWIN, flag <name>_AT_IDENTITY); (d) F4 of the revised point (frc.rules.f4); (e) reader targeting vs TWIN and every bracketing
    point (frc.rules.targeting). Effort: the revised HQ runs the revised solver with identical settings and does not stop on a cap
    when the revised PRC converged (else NOT_ASSESSED_UNEQUAL_EFFORT)."""
    gv = rec.get('grid_valid')
    if gv is None:
        return {'label': 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'met': False}
    if gv is not True:
        return {'label': 'NOT_COMPUTED_GRID_INVALID', 'met': False}
    p = rec.get('revised_prc') or {}
    if p.get('status') == 'NOT_FEASIBLE':
        return {'label': 'NOT_ASSESSED_NO_FEASIBLE_POINT', 'met': False}
    if p.get('alias_of_twin') is True:
        return {'label': 'NOT_ASSESSED_EXACT_ALIAS', 'met': False}
    if p.get('alias_of_original') is True:
        return {'label': 'NOT_ASSESSED_ALIAS_OF_ORIGINAL', 'met': False}
    if p.get('status') != 'FITTED':
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'revised PRC status missing'}
    eff = _effort_ok(rec.get('effort'))
    if eff is None:
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'effort record of the revised PRC / HQ missing'}
    if not eff:
        return {'label': 'NOT_ASSESSED_UNEQUAL_EFFORT', 'met': False}
    g1 = NR.point_eligibility(p.get('evals') or {})
    if g1['label'] == 'INCOMPLETE':
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'revised PRC utility incomplete'}
    feas = g1['label'] == 'POINT_ELIGIBLE'
    grid, flags, comps = rec.get('grid') or {}, [], {}
    if 'TWIN' not in grid:
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'grid entry vs TWIN missing'}
    ct = NR.comparison(grid['TWIN'])
    if ct['benefit']['label'] == 'INCOMPLETE' or any(ct['recipient'][w]['label'] == 'INCOMPLETE' for w in NR.SINGLE_VIEWS):
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'grid entry vs TWIN incomplete'}
    comps['TWIN'] = {'point': 'TWIN', 'benefit': ct['benefit']['label'], 'pass': ct['benefit_positive'] and ct['recipient_gate_pass']}
    point = {'id': p.get('config_id', 'REVISED_PRC'), 'costs_vs': rec.get('costs_vs') or {}, 'grid_vs': grid}
    for name, key in (('HQ_FRONTIER', 'HQ'), ('CONTROL_FRONTIER', 'CONTROL')):
        ids = FR._path((rec.get('paths') or {}).get(key), rec.get('alias_of'))
        if not ids:
            ids = ['TWIN']
            flags.append(f'{key}_AT_IDENTITY')
        fr = NR.frontier_comparison(point, [{'id': i} for i in ids])
        if fr['label'] == 'INCOMPLETE':
            return {'label': 'INCOMPLETE', 'met': False, 'error': f'{name}: {fr.get("error")}'}
        comps[name] = {'label': fr['label'], 'reason': fr.get('reason'), 'path': ids, 'point': fr.get('bracketing_control'),
                       'pass': fr['label'] == 'DOMINATES_CONTROL_FRONTIER', 'frontier': fr}
    f = FR.f4(rec.get('f4'))
    if f is None:
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'F4 record missing or non-finite'}
    tg = {}
    for q in dict.fromkeys(c['point'] for c in comps.values() if c.get('point')):
        nl = (rec.get('nested_lookup') or {}).get(q)
        if nl is None and q == 'TWIN':
            nl = {v: f['per_version'][v]['diff'] for v in VERSIONS}
            flags.append('TWIN_NESTED_LOOKUP_FROM_F4')
        x = FR.targeting((rec.get('nolookup') or {}).get(q), nl)
        if x is None:
            return {'label': 'INCOMPLETE', 'met': False, 'error': f'reader-targeting inputs vs {q} missing'}
        tg[q] = x
    passed = [n for n in C3_COMPARISONS if comps[n]['pass']]
    allc = len(passed) == len(C3_COMPARISONS)
    targeted = not all(x['pass'] for x in tg.values())
    if not f['pass']:
        lab = 'NOT_MET_F4'
    elif allc and targeted:
        lab = 'READER_TARGETED'
    elif allc:
        lab = 'PROTECTION_BEYOND_MATCHED_CONTROLS' if feas else 'MET_WITHOUT_TASK_FEASIBILITY'
    elif not passed:
        lab = 'NO_REDUCTION_BEYOND_ANY_CONTROL'
    else:
        lab = 'PARTIAL_' + '+'.join(passed) + '_ONLY'
    se = [c['frontier'].get('benefit_se') for c in comps.values() if c.get('frontier') and _fin(c['frontier'].get('benefit_se'))]
    return {'label': lab, 'met': lab == 'PROTECTION_BEYOND_MATCHED_CONTROLS', 'task_feasible': feas, 'passed': passed, 'f4': f,
            'targeting': tg, 'comparisons': comps, 'flags': sorted(set(flags)),
            'minimum_detectable_pair_benefit': (Z * max(se)) if se else None, 'note': 'no partial outcome is a claim'}


def revised_vs_original(rec):
    """RULES.md 7.4: the revised PRC primary point against the original solver's path [TWIN, original PRC points] on MODEL_SELECT.
    RECOVERY_GAIN_AT_MATCHED_SPEND iff nrm.rules.frontier_comparison gives DOMINATES_CONTROL_FRONTIER; else
    TASK_GAIN_AT_NO_WORSE_RECOVERY iff against the original primary point the pooled paired total-NLL cost is resolved negative
    (value + 2 se < 0), the aggregate pair benefit is > -AUC_MATERIAL and the recipient gate passes; else
    NO_MEASURED_GAIN_OVER_ORIGINAL."""
    path = FR._path((rec.get('paths') or {}).get('ORIGINAL'), rec.get('alias_of'))
    o = rec.get('original_primary')
    grid = rec.get('grid') or {}
    if not path or o is None or o not in grid:
        return {'label': 'INCOMPLETE', 'error': 'original path, original primary point or its grid entry missing'}
    point = {'id': 'REVISED_PRC', 'costs_vs': rec.get('costs_vs') or {}, 'grid_vs': grid}
    fr = NR.frontier_comparison(point, [{'id': i} for i in path])
    if fr['label'] == 'INCOMPLETE':
        return {'label': 'INCOMPLETE', 'error': fr.get('error')}
    if fr['label'] == 'DOMINATES_CONTROL_FRONTIER':
        return {'label': 'RECOVERY_GAIN_AT_MATCHED_SPEND', 'frontier': fr}
    prim, _ = NR._primary((rec.get('costs_vs') or {}).get(o))
    cmp_ = NR.comparison(grid[o])
    if prim is None or cmp_['benefit']['label'] == 'INCOMPLETE':
        return {'label': 'INCOMPLETE', 'error': 'paired costs or grid entry vs the original primary point incomplete'}
    ben = cmp_['benefit']['aggregate']['mean']
    ok = prim['value'] + Z * prim['se'] < 0 and ben > -AUC_MATERIAL + EPS and cmp_['recipient_gate_pass']
    return {'label': 'TASK_GAIN_AT_NO_WORSE_RECOVERY' if ok else 'NO_MEASURED_GAIN_OVER_ORIGINAL', 'frontier': fr,
            'primary_cost_vs_original': prim, 'pair_benefit_vs_original': ben}


def pilot(rec):
    """Pilot claims C1-C7 and the outcome class (RULES.md 7). rec = {'advancement': advancement() record, 'grid_valid', 'c1':
    {'fitted', 'operating_lambda'}, 'revised_prc': {'status', 'config_id', 'evals', 'alias_of_twin', 'alias_of_original'}, 'grid':
    {id: grid entry of the revised point vs id, incl. TWIN, the HQ / control / original path points and the nine references},
    'paths': {'HQ', 'CONTROL', 'ORIGINAL'}, 'original_primary', 'alias_of', 'costs_vs', 'f4', 'nolookup', 'nested_lookup',
    'reader_budget', 'effort', 'joint'}. Diagnostics are never read here."""
    if ((rec.get('advancement') or {}).get('label')) != 'ADVANCE':
        return {'label': 'PILOT_NOT_AUTHORIZED', 'outcome_class': None,
                'note': 'the pilot runs only after an ADVANCE decision of the advancement rule'}
    p = rec.get('revised_prc') or {}
    c1rec = rec.get('c1') or {}
    out = {'C1': c1(c1rec.get('fitted'), c1rec.get('operating_lambda'))}
    g1 = NR.point_eligibility(p.get('evals') or {})
    out['C2'] = {'label': {'POINT_ELIGIBLE': 'TASK_FEASIBLE', 'POINT_INELIGIBLE': 'TASK_INFEASIBLE'}.get(g1['label'], 'INCOMPLETE'),
                 'detail': g1}
    out['C2']['met'] = out['C2']['label'] == 'TASK_FEASIBLE'
    try:
        out['C2']['reserve_reported_never_gating'] = NR.reserve_status(p.get('evals') or {})
    except (KeyError, ValueError, TypeError) as ex:
        out['C2']['reserve_reported_never_gating'] = {'status': 'INCOMPLETE', 'qualified': False, 'error': str(ex)}
    out['C3'] = c3(rec)
    out['R_VS_O'] = revised_vs_original(rec) if p.get('status') == 'FITTED' else {'label': 'INCOMPLETE'}
    flags = []
    gv = rec.get('grid_valid')
    if gv is True and p.get('status') == 'FITTED' and not p.get('alias_of_twin'):
        cl = NR.claims({'evals': p.get('evals') or {}, 'grid': rec.get('grid') or {}, 'grid_valid': gv, 'frontier': None,
                        'matched': {}, 'joint': None})
        out['C4'] = FR._d3(rec, cl)
        out['C5'] = FR._d4(rec, cl)
    else:
        why = ('NOT_COMPUTED_GRID_VALIDITY_UNKNOWN' if gv is None else 'NOT_COMPUTED_GRID_INVALID' if gv is not True
               else 'NOT_ASSESSED')
        out['C4'] = {'label': why, 'met': False}
        out['C5'] = {'label': why, 'met': False}
    j = rec.get('joint')
    if not j or not all(j.get(k) for k in ('separate', 'income_first', 'occupation_first')):
        out['C6'] = {'label': 'NOT_ASSESSED_MISSING_CONTROLS', 'statement': 'no joint-design claim without the separate design and '
                                                                             'both sequential orders'}
    else:
        cl = NR.claims({'evals': p.get('evals') or {}, 'grid': rec.get('grid') or {}, 'grid_valid': gv, 'matched': {}, 'joint': j})
        out['C6'] = {'label': cl['G6']['label'], 'detail': cl['G6']}
    out['C7'] = {'label': 'NOT_CLAIMED', 'statement': 'a better line search is not a novel privacy algorithm; the objective has named '
                                                      'prior art (privacy funnel, IBSI, FARE)'}
    bad = ('INCOMPLETE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID')
    c2m, c3m, c4m = out['C2']['met'], out['C3']['met'], bool(out['C4'].get('met'))
    if out['C2']['label'] == 'INCOMPLETE' or out['C3']['label'] in bad or out['C4']['label'] in bad:
        oc = 'INCOMPLETE'
    elif c2m and c3m and c4m:
        oc = 'METHOD_CANDIDATE'
    elif c2m and c3m:
        oc = 'COMPONENT'
    elif c2m and out['C1']['met'] and out['R_VS_O']['label'] in ('RECOVERY_GAIN_AT_MATCHED_SPEND', 'TASK_GAIN_AT_NO_WORSE_RECOVERY'):
        oc = 'ENGINEERING_RESULT'
    elif out['C1']['label'] == 'INCOMPLETE' or out['R_VS_O']['label'] == 'INCOMPLETE':
        oc = 'INCOMPLETE'
    else:
        oc = 'BOUNDED_NEGATIVE_RESULT'
    if c3m and not out['C1']['met']:
        flags.append('NOT_ATTRIBUTABLE_TO_BETTER_OPTIMIZATION')
    if c4m and not c3m:
        flags.append('C4_WITHOUT_C3')
    if out['C5'].get('met'):
        flags.append('MATERIAL_0020_MARGIN_MET')
    if c2m and not (out['C2']['reserve_reported_never_gating'] or {}).get('qualified'):
        flags.append('POINT_FEASIBLE_NOT_RESERVE_QUALIFIED')
    out['outcome_class'] = {'label': oc, 'flags': sorted(set(flags)),
                            'bound': out['C3'].get('minimum_detectable_pair_benefit'),
                            'note': 'development evidence on adaptively used MODEL_SELECT records; versions are procedure repeats'}
    out['scope'] = list(SCOPE)
    return out


def diagnostics(rec):
    """RULES.md 7.8: the strongest feasible nonzero point (frc.rules.strongest_nonzero, unchanged), one component ablation and the
    effective comparator refits, run whatever the first intervals show. Reported only: pilot() never reads them."""
    s = FR.strongest_nonzero(rec.get('configs') or [])
    ref = rec.get('refit')
    return {'strongest': s, 'ablation': rec.get('ablation'), 'refit': refit_effective(**ref) if isinstance(ref, dict) else None,
            'decision_input': False, 'note': 'diagnostics are never decision inputs'}


# ---------------------------------------------------------------- P1 replay acceptance
def _flat(x, pre=''):
    if isinstance(x, dict):
        out = {}
        for k in sorted(x, key=str):
            if str(k) not in P1_TEXT_KEYS:
                out.update(_flat(x[k], f'{pre}/{k}'))
        return out
    if isinstance(x, (list, tuple)):
        out = {}
        for i, y in enumerate(x):
            out.update(_flat(y, f'{pre}[{i}]'))
        return out
    return {pre: x}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _compare(a, b, fields, tol):
    """Mismatches of record b against record a over the given top-level fields: numeric leaves beyond tol (absolute; NaN equals
    NaN), any other leaf unequal, or a leaf missing on either side."""
    mism, worst = [], 0.0
    for f in fields:
        fa, fb = _flat((a or {}).get(f), f'/{f}'), _flat((b or {}).get(f), f'/{f}')
        for k in sorted(set(fa) | set(fb)):
            if k not in fa or k not in fb:
                mism.append({'path': k, 'kind': 'MISSING'})
                continue
            x, y = fa[k], fb[k]
            if _num(x) and _num(y):
                if math.isnan(x) and math.isnan(y):
                    continue
                d = abs(float(x) - float(y))
                if math.isfinite(d):
                    worst = max(worst, d)
                if not d <= tol:
                    mism.append({'path': k, 'kind': 'NUMERIC', 'a': x, 'b': y, 'abs_difference': d})
            elif x != y:
                mism.append({'path': k, 'kind': 'DISCRETE', 'a': x, 'b': y})
    return mism, worst


def _boundary(m, ref, tol):
    """True iff a discrete verifier mismatch sits on a decision boundary of the replayed record within tol."""
    path = m['path']
    if path.startswith('/selection/'):
        parts = path.split('/')
        fam, bud = (parts[2], parts[3]) if len(parts) > 3 else (None, None)
        lim = ((ref.get('budgets') or {}).get(bud) or {})
        aud = (ref.get('audit_spend_over_twin') or {}).get(fam) or {}
        return any(abs(float(es[0]) + Z * float(es[1]) - float(lim[k])) <= tol for c in aud.values() for k, es in c.items()
                   if k in lim and isinstance(es, (list, tuple)) and _fin(es[0], es[1]))
    if path == '/F4/pass_2se':
        f = ref.get('F4') or {}
        return _fin(f.get('diff_twin_minus_prc'), f.get('se_boot')) and abs(f['diff_twin_minus_prc'] - Z * f['se_boot']) <= tol
    if path.startswith('/brackets/brackets/'):
        b = ref.get('brackets') or {}
        fam, t = path.split('/')[-1], b.get('target_total_nll_spend')
        return _fin(t) and any(abs(float(s) - t) <= tol for s in ((b.get('spend') or {}).get(fam) or {}).values() if _fin(s))
    return False


def p1_replay(recorded, replayed, verifier):
    """RULES.md 8. recorded = {v: the published SELECTION_V<v>.json}; replayed = {v: the coordinator's replay in the same structure,
    plus 'estimator_unchanged': True (verifier-checked: nrm/fa/prc.py byte-identical and the frc P1 estimator functions textually
    identical) and 'per_person_outputs_saved': True}; verifier = {v: the verifier's independent recomputation, at least
    P1_VERIFIER_MIN}. P1_REPLAY_VERIFIED iff in both versions the replay matches the record (numbers within REPLAY_TOL, every
    discrete output equal) and the verifier matches the replay (numbers within VERIFY_TOL; a discrete difference counts only off a
    decision boundary). A mismatch never changes an FRC1 label."""
    per, labs = {}, []
    for v in VERSIONS:
        a, b, c = _v(recorded, v), _v(replayed, v), _v(verifier, v)
        if not isinstance(a, dict) or not isinstance(b, dict):
            return {'label': 'INCOMPLETE', 'error': f'recorded or replayed record of version {v} missing', 'old_labels_changed': False}
        if b.get('estimator_unchanged') is not True or b.get('per_person_outputs_saved') is not True:
            per[v] = {'label': 'P1_ESTIMATOR_CHANGED'}
            labs.append('P1_ESTIMATOR_CHANGED')
            continue
        rm, rw = _compare(a, b, [f for f in P1_FIELDS if f in a], REPLAY_TOL)
        if not isinstance(c, dict) or any(f not in c for f in P1_VERIFIER_MIN):
            per[v] = {'label': 'INCOMPLETE', 'reproduction_mismatches': rm, 'error': 'verifier recomputation missing fields'}
            labs.append('INCOMPLETE' if not rm else 'P1_NOT_REPRODUCED')
            continue
        vf = [f for f in P1_FIELDS if f in c]
        vm, vw = _compare(b, c, vf, VERIFY_TOL)
        ties = [m for m in vm if m['kind'] == 'DISCRETE' and _boundary(m, b, VERIFY_TOL)]
        real = [m for m in vm if m not in ties]
        lab = 'P1_NOT_REPRODUCED' if rm else ('P1_VERIFICATION_DISAGREES' if real else 'P1_REPLAY_VERIFIED')
        per[v] = {'label': lab, 'reproduction_mismatches': rm, 'reproduction_max_abs_difference': rw, 'verifier_mismatches': real,
                  'verifier_boundary_ties': ties, 'verifier_max_abs_difference': vw, 'verifier_fields': vf,
                  'verifier_vs_record_mismatches': _compare(a, c, [f for f in vf if f in a], VERIFY_TOL)[0]}
        labs.append(lab)
    order = ('P1_ESTIMATOR_CHANGED', 'P1_NOT_REPRODUCED', 'INCOMPLETE', 'P1_VERIFICATION_DISAGREES', 'P1_REPLAY_VERIFIED')
    lab = next(x for x in order if x in labs)
    cons = {'P1_REPLAY_VERIFIED': 'the P1 person-level spends, SEs, selection and F4 are independently recomputed; the replayed '
                                  'per-person outputs may serve as the original-solver baseline',
            'P1_NOT_REPRODUCED': 'the replayed outputs are not decision inputs; the original solver is rerun beside the revised one in '
                                 'the same nested run; a dated diagnosis precedes any route-S decision; the FRC1 record is reported '
                                 'as not reproduced',
            'P1_VERIFICATION_DISAGREES': 'the replayed outputs are not decision inputs until the disagreement is diagnosed in a dated '
                                         'note',
            'P1_ESTIMATOR_CHANGED': 'not a replay: the estimator or its saved outputs differ from the registered P1 estimator',
            'INCOMPLETE': 'awaiting the replay or the verifier recomputation'}[lab]
    return {'label': lab, 'per_version': per, 'consequence': cons, 'old_labels_changed': False,
            'tolerances': {'reproduction_abs': REPLAY_TOL, 'verifier_abs': VERIFY_TOL}}
