"""Frozen claims and selection rules of the PCRL finite-routing comparison campaign (`pcrl_finite_routing_comparison_v1`;
statistical reviewer; written before any outcome of this campaign existed; no label read).

Human-readable rules: results/pcrl_finite_routing_comparison_v1/roles/statistical_reviewer/RULES.md
Machine-readable constants: .../RULES.json (schema FRC1-rules-v1); `load_rules()` refuses a file that disagrees with this module.
Invented-data tests: `python3 -m frc.test_rules` (prints JSON; exit 0 iff every test passes).

Pure functions on plain dicts / floats. Nothing here reads data or labels. The inherited machinery (aggregate screen, benefit /
target / recipient screens, reserve_status, point_eligibility, frontier_comparison, reserve_frontier, claims G1-G7,
select_candidate, reader_stability, robustness, alias handling) is `nrm.rules`, imported UNCHANGED as NR. This module adds:

  qpe_label(nested, ms)                       Q-PE label of one capacity (PROTOCOL 6.1; SE_used = max(household SE, fold SE))
  construction_choice(nested)                 PROTOCOL 6.2 per capacity: 'B1' or 'A' (MODEL_FIT nested point estimates only)
  qual(evals)                                 PROTOCOL 5: Q1 (G1 per version) and Q2 (S2 slack per condition)
  f4(rec), targeting(nolookup, nested)        D2 conjuncts (d) and (e): falsifier F4, reader-targeting point conditions
  capacity_gate(spend_over_cont)              the original NRM1 capacity gate, REPORTED only (never gating)
  gate_reproduction(recomputed, gate, key)    T1 reproduction of a GATE record (max abs difference <= 1e-12)
  capacity_decision(by_capacity, repair)      capacity order, triggers T5 / T6, stop (PROTOCOL 6.3-6.6), deterministic
  d1(cap), d2(rec), decisions(rec)            D1-D6 and the outcome class (PROTOCOL 10)
  strongest_nonzero(configs)                  PROTOCOL 10.4 choice
  refit_effectiveness(splits, seeds, ...)     PROTOCOL 10.4 'genuinely effective classifier refit'
  forbidden_seeds(), grid_validity(...)       reader-seed validity of the FRC1 grid (+700; 300 / 400 / 500 / 600 forbidden)

Sign convention of every paired difference here: loss(first) - loss(second) on the same people (negative favours the first);
accuracy enters as a loss. Versions are the ints 0 and 1 (string keys '0' / '1' are accepted).
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

from nrm import rules as NR                      # inherited claims machinery, imported unchanged (never monkeypatched)
from ohn import rules as OR                      # condition parsing (refuses re-capped or malformed utility records)

RULES_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_finite_routing_comparison_v1' / 'roles'
              / 'statistical_reviewer' / 'RULES.json')
SCHEMA = 'FRC1-rules-v1'
NRM1_RULES_SHA256 = '84f187768817e79ef9f2bb36151bea81f00b8f62d9179c8cd523069550674652'
VERSIONS = NR.VERSIONS
Z = 2.0
# PROTOCOL 5, Q2: the S2 budget of FA_PRC_v1 per condition (the assessment budget of D2); the two summed-Brier clauses S2 does not
# budget get the guard budget (occupation) and half the income NLL budget (income). Keys: '<task>|<metric>' of the ohn.rules clause
# names, and 'guard'.
Q2_INCREMENTS = {'income|accuracy_loss': 0.004, 'income|nll_excess': 0.004, 'income|brier_excess': 0.0020,
                 'occupation|accuracy_loss': 0.006, 'occupation|nll_excess': 0.008, 'occupation|brier_excess': 0.0030,
                 'guard': 0.0030}                 # the S2 budget (adjudication finding 1, adopted 23:30Z, before any outcome)
N_OUTER = 5                                       # outer folds of the nested estimator (fold-level SE, adjudication finding 17)
CAPACITY_GATE_LIMITS = {'occ_nll': 0.004, 'inc_nll': 0.004, 'occ_acc': 0.003, 'inc_acc': 0.003}   # estimate + 2 SE (original)
REPRO_TOL = 1e-12
EXISTING = ('16x64', '24x128')
PROSPECTIVE = '32x256'
REPAIR = 'REPAIR'
NESTED_KEYS = ('occ_nll', 'inc_nll', 'tot_nll', 'occ_brier', 'inc_brier', 'occ_acc', 'inc_acc')
MS_KEY = {'occ_nll': 'occupation|nll', 'inc_nll': 'income|nll', 'tot_nll': 'total|nll', 'occ_brier': 'occupation|brier',
          'inc_brier': 'income|brier', 'occ_acc': 'occupation|acc_loss', 'inc_acc': 'income|acc_loss'}
MS_ONLY = ('occupation|men_brier', 'occupation|women_brier')
QPE_LABELS = ('PE_ERROR_SUPPORTED', 'PE_ERROR_POINT_ONLY', 'PE_ERROR_NOT_SUPPORTED', 'INCOMPLETE')
D1_LABELS = ('REPAIRED_TASK_ONLY_RELEASE', 'QUALIFIED_B1_REPAIR_NOT_RESOLVED', 'QUALIFIED_WITHOUT_PROBABILITY_REPAIR',
             'POST_HOC_REPAIR_QUALIFIED', 'NO_QUALIFYING_TASK_ONLY_RELEASE', 'INCOMPLETE')
D2_COMPARISONS = ('TWIN', 'SHUF_FRONTIER', 'HQ_FRONTIER', 'ORD_FRONTIER')
D2_PATHS = (('SHUF_FRONTIER', 'SHUF'), ('HQ_FRONTIER', 'HQ'), ('ORD_FRONTIER', 'ORD'))
OUTCOMES = ('CANDIDATE', 'COMPONENT', 'ENGINEERING_REPAIR', 'FINITE_NEGATIVE_RESULT', 'INCOMPLETE')
FORBIDDEN_OFFSETS = (300, 400, 500, 600)           # OHN1/CTS1 300, CTS 400, TSZ1 500, NRM1 600
SEED_OFFSET = 700                                  # this campaign: MODEL_SEED_BASE + 700 + 10 v + k
BOOT_SEED = 2026102100                             # + 100 v + repetition index (R1 0, R2 1, R3 2, R4 3)
TIE_TOL = 1e-12
STOP_STATEMENT = 'no task-only finite release qualifies: family FA stops and protected routing remains untested'

_cache = {}
_SEEDKEY = re.compile(r'S([01])k([0-9])')


# ---------------------------------------------------------------- rule file
def load_rules(path=RULES_PATH):
    """RULES.json with its SHA-256; refuses a schema or constant that disagrees with this module, or a changed NRM1 rule file."""
    p = Path(path)
    if str(p) not in _cache:
        raw = p.read_bytes()
        r = json.loads(raw)
        if r.get('schema') != SCHEMA:
            raise ValueError(f'unexpected rules schema {r.get("schema")!r}')
        if NR.load_rules().get('_sha256') != NRM1_RULES_SHA256:
            raise ValueError('the NRM1 rule file differs from the registered one')
        checks = {
            'q2 increments': (r['qual']['q2_increments'], Q2_INCREMENTS),
            'capacity gate limits': (r['original_capacity_gate']['limits_estimate_plus_2se'], CAPACITY_GATE_LIMITS),
            'z': (r['z'], Z), 'reproduction tolerance': (r['reproduction']['max_abs_difference'], REPRO_TOL),
            'capacity order': (tuple(r['capacities']['order']), EXISTING),
            'prospective capacity': (r['capacities']['prospective'], PROSPECTIVE),
            'forbidden offsets': (tuple(r['reader_seeds']['forbidden_offsets']), FORBIDDEN_OFFSETS),
            'seed offset': (r['reader_seeds']['offset_this_campaign'], SEED_OFFSET),
            'model seed base': (r['reader_seeds']['model_seed_base'], NR.MODEL_SEED_BASE),
            'qpe labels': (tuple(r['qpe']['labels']), QPE_LABELS), 'd1 labels': (tuple(r['decisions']['D1']['labels']), D1_LABELS),
            'd2 comparisons': (tuple(r['decisions']['D2']['comparisons']), D2_COMPARISONS),
            'outer folds': (r['qpe']['outer_folds'], N_OUTER),
            'outcomes': (tuple(r['outcome_class']['labels']), OUTCOMES),
        }
        bad = [k for k, (a, b) in checks.items() if a != b]
        if bad:
            raise ValueError(f'RULES.json disagrees with frc.rules on {bad}')
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


def _est(x):
    """(estimate, SE) from a tuple / list (est, se) or a dict {'value', 'se'}; None unless both finite and SE >= 0."""
    if isinstance(x, dict):
        x = (x.get('value'), x.get('se'))
    if not isinstance(x, (tuple, list)) or len(x) < 2:
        return None
    e, s = x[0], x[1]
    return (float(e), float(s)) if _fin(e, s) and s >= 0 else None


def _ms_value(ms, v, key):
    e = _est((_v(ms, v) or {}).get(key))
    return None if e is None else e[0]


def _utility(e):
    return e['utility'] if isinstance(e, dict) and 'utility' in e else e


# ---------------------------------------------------------------- (a) Q-PE labels per capacity
def _neg_label(pairs, z=Z):
    if any(p is None for p in pairs.values()):
        return 'MISSING'
    if any(p[0] >= 0 for p in pairs.values()):
        return 'NOT_NEGATIVE_SOME_VERSION'
    return 'RESOLVED_NEGATIVE_BOTH' if all(p[0] + z * p[1] < 0 for p in pairs.values()) else 'POINT_NEGATIVE_BOTH'


def fold_means(nested_v, key):
    """The N_OUTER outer-fold means of the paired B1 - A difference `key` of one version: from nested_v['folds'][i]
    ['paired_fold_means'][key] (the coordinator's record) or nested_v['fold_means'][key]; None unless exactly 5 finite values."""
    fm = (nested_v or {}).get('fold_means')
    if isinstance(fm, dict) and key in fm:
        xs = fm[key]
    else:
        folds = (nested_v or {}).get('folds')
        if not isinstance(folds, (list, tuple)):
            return None
        xs = [((f or {}).get('paired_fold_means') or {}).get(key) if isinstance(f, dict) else None for f in folds]
    if not isinstance(xs, (list, tuple)) or len(xs) != N_OUTER or not all(_fin(x) for x in xs):
        return None
    return [float(x) for x in xs]


def se_used(nested_v, key):
    """SE_used = max(household-clustered SE, SD of the 5 outer-fold means / sqrt(5)) for one version and metric (adjudication
    finding 17); sample SD (ddof 1). None if the estimate, its household SE or the fold means are missing."""
    e = _est(((nested_v or {}).get('B1-A') or {}).get(key))
    fm = fold_means(nested_v, key)
    if e is None or fm is None:
        return None
    m = sum(fm) / len(fm)
    se_f = math.sqrt(sum((x - m) ** 2 for x in fm) / (len(fm) - 1)) / math.sqrt(len(fm))
    return {'estimate': e[0], 'se_household': e[1], 'se_fold': se_f, 'se_used': max(e[1], se_f),
            'binding': 'fold' if se_f > e[1] else 'household', 'fold_means': fm}


def qpe_label(nested, ms=None, z=Z):
    """nested = {v: {'B1-A': {key: (est, se)}, 'B0-B1': {...}, 'spend_over_cont': {'A': {...}, 'B1': {...}, 'B0': {...}}}} (T1,
    MODEL_FIT nested held-out, household-clustered SEs, loss(first) - loss(second)); ms = {v: nrm.evaluate.paired_costs of B1
    vs A} (T3). PE_ERROR_SUPPORTED iff d_v + z SE_v < 0 in both versions and the MODEL_SELECT paired occupation NLL B1 - A < 0
    at the point in both versions; PE_ERROR_POINT_ONLY iff d_v < 0 in both versions otherwise; PE_ERROR_NOT_SUPPORTED iff
    d_v >= 0 in some version; INCOMPLETE if a needed input is missing (the MODEL_SELECT values are needed only when the nested
    differences are resolved). Reported beside (never gating): the same nested reading of every other metric, the MODEL_SELECT
    point values, B0 - B1, and the share of A's nested occupation NLL spend over CONT that B1 removes."""
    su = {v: se_used(_v(nested, v), 'occ_nll') for v in VERSIONS}
    rec = {'z': z, 'note': 'development label on adaptively used records; SE_used = max(household-clustered SE, SD of the 5 '
                           'outer-fold means / sqrt 5), still approximate (outer-training sets overlap); versions are procedure '
                           'repeats, not replications'}
    if any(x is None for x in su.values()):
        return {**rec, 'label': 'INCOMPLETE', 'error': 'nested B1 - A occupation NLL (estimate, household SE) or its 5 outer-fold '
                                                       'means missing or non-finite'}
    d = {v: (su[v]['estimate'], su[v]['se_used']) for v in VERSIONS}
    rec['occ_nll'] = {v: {**su[v], 'upper': d[v][0] + z * d[v][1]} for v in VERSIONS}
    msv = {v: _ms_value(ms, v, 'occupation|nll') for v in VERSIONS}
    rec['model_select_occ_nll'] = msv
    if any(d[v][0] >= 0 for v in VERSIONS):
        rec['label'] = 'PE_ERROR_NOT_SUPPORTED'
    elif all(d[v][0] + z * d[v][1] < 0 for v in VERSIONS):
        if any(x is None for x in msv.values()):
            return {**rec, 'label': 'INCOMPLETE', 'error': 'nested differences resolved; MODEL_SELECT paired occupation NLL missing'}
        rec['label'] = 'PE_ERROR_SUPPORTED' if all(msv[v] < 0 for v in VERSIONS) else 'PE_ERROR_POINT_ONLY'
    else:
        rec['label'] = 'PE_ERROR_POINT_ONLY'
    beside = {}
    for key in NESTED_KEYS:
        pairs, fl = {}, []
        for v in VERSIONS:
            s = se_used(_v(nested, v), key)
            if s is not None:
                pairs[v] = (s['estimate'], s['se_used'])
            else:
                pairs[v] = _est(((_v(nested, v) or {}).get('B1-A') or {}).get(key))
                fl.append(f'FOLD_MEANS_MISSING_V{v}')            # household SE only for this reported row
        beside[key] = {'nested': {v: pairs[v] for v in VERSIONS}, 'nested_reading': _neg_label(pairs, z), 'flags': fl,
                       'model_select_point': {v: _ms_value(ms, v, MS_KEY[key]) for v in VERSIONS}}
    for key in MS_ONLY:                                  # T1 reads no SEX: within-SEX rows come from MODEL_SELECT only
        beside[key] = {'model_select_point': {v: _ms_value(ms, v, key) for v in VERSIONS}}
    rec['reported_beside'] = beside
    rec['b0_minus_b1'] = {key: {v: _est(((_v(nested, v) or {}).get('B0-B1') or {}).get(key)) for v in VERSIONS}
                          for key in NESTED_KEYS}
    rec['share_removed'] = share_removed(nested)
    return rec


def share_removed(nested):
    """Share of A's nested held-out occupation NLL spend over CONT that B1 removes, per version: (spend_A - spend_B1) / spend_A,
    defined iff spend_A > 0; pooled = mean numerator / mean denominator (defined iff the mean denominator > 0). Reported only."""
    out, nums, dens = {}, [], []
    for v in VERSIONS:
        s = (_v(nested, v) or {}).get('spend_over_cont') or {}
        a, b = _est((s.get('A') or {}).get('occ_nll')), _est((s.get('B1') or {}).get('occ_nll'))
        if a is None or b is None:
            out[v] = {'share': None, 'reason': 'MISSING'}
            continue
        num, den = a[0] - b[0], a[0]
        nums.append(num)
        dens.append(den)
        out[v] = {'share': num / den if den > 0 else None, 'removed_nats': num, 'spend_A': den, 'spend_B1': b[0],
                  'reason': None if den > 0 else 'NOT_DEFINED_NONPOSITIVE_A_SPEND'}
    pooled = (sum(nums) / len(nums)) / (sum(dens) / len(dens)) if len(dens) == len(VERSIONS) and sum(dens) > 0 else None
    return {'per_version': out, 'pooled': pooled, 'note': 'reported, never gating'}


# ---------------------------------------------------------------- (b) construction choice
def construction_choice(nested):
    """PROTOCOL 6.2, applied per capacity (MODEL_FIT nested evidence only; MODEL_SELECT never chooses): 'B1' iff the nested paired
    occupation NLL difference and the nested paired total (income + occupation) NLL difference B1 - A are both < 0 at the point
    in both versions; otherwise 'A' (ties at exactly 0 go to A). If 'tot_nll' is absent the point total is inc_nll + occ_nll
    (exact for point estimates on the same people; flag TOTAL_FROM_TASK_SUM)."""
    pts, flags = {}, []
    for v in VERSIONS:
        b = (_v(nested, v) or {}).get('B1-A') or {}
        occ = _est(b.get('occ_nll'))
        tot = _est(b.get('tot_nll'))
        if tot is None and _est(b.get('inc_nll')) is not None and occ is not None:
            tot = (_est(b['inc_nll'])[0] + occ[0], math.nan)
            flags.append('TOTAL_FROM_TASK_SUM')
        if occ is None or tot is None:
            return {'choice': None, 'label': 'INCOMPLETE', 'error': f'version {v}: nested B1 - A occ_nll / tot_nll missing'}
        pts[v] = {'occ_nll': occ[0], 'tot_nll': tot[0]}
    b1 = all(pts[v]['occ_nll'] < 0 and pts[v]['tot_nll'] < 0 for v in VERSIONS)
    return {'choice': 'B1' if b1 else 'A', 'label': 'CHOSEN', 'points': pts, 'flags': sorted(set(flags)),
            'note': 'a construction choice from MODEL_FIT point estimates, not a claim'}


# ---------------------------------------------------------------- (c) QUAL
def q2_increment(name):
    """The Q2 increment of one ohn.rules condition name ('<task>|<reference>|<metric>' or 'guard|sex<g>')."""
    if name.startswith('guard|'):
        return Q2_INCREMENTS['guard']
    task, _, metric = name.split('|')
    return Q2_INCREMENTS[f'{task}|{metric}']


def qual(evals_by_version):
    """PROTOCOL 5 for one construction at one capacity: QUALIFIED iff in BOTH versions Q1 (G1 of the version: 12 clauses + 2
    guards at the point, original caps, inclusive) and Q2 (value <= cap - d for every condition, d = q2_increment = the S2 budget,
    cap - d in double precision). A gate for starting the comparison, never a claim and never confidence preservation; the
    reserve is reported separately."""
    q1 = NR.point_eligibility(evals_by_version)
    if q1['label'] == 'INCOMPLETE':
        return {'label': 'INCOMPLETE', 'error': q1.get('error')}
    per, reasons = {}, []
    for v in VERSIONS:
        conds = OR.conditions(_utility(_v(evals_by_version, v)), v)      # already validated by point_eligibility
        rows = {}
        for n, c in conds.items():
            d = q2_increment(n)
            lim = c['cap'] - d
            rows[n] = {'value': c['value'], 'cap': c['cap'], 'increment': d, 'slack_limit': lim, 'pass': c['value'] <= lim,
                       'slack_after_increment': lim - c['value']}
        failed = sorted(n for n, r in rows.items() if not r['pass'])
        per[v] = {'q1': q1['per_version'][v]['point_pass'], 'q1_failed': q1['per_version'][v]['failed'], 'q2': not failed,
                  'q2_failed': failed, 'binding': min(rows, key=lambda n: rows[n]['slack_after_increment']), 'conditions': rows}
        reasons += [f'Q1_FAILED_V{v}'] * (not per[v]['q1']) + [f'Q2_FAILED_V{v}'] * bool(failed)
    ok = q1['label'] == 'POINT_ELIGIBLE' and all(per[v]['q2'] for v in VERSIONS)
    return {'label': 'QUALIFIED' if ok else 'NOT_QUALIFIED', 'qualified': ok, 'reasons': reasons, 'per_version': per,
            'note': 'a gate for starting the comparison, not a claim; point slack is not confidence preservation'}


# ---------------------------------------------------------------- (d) original capacity gate (reported, never gating)
def capacity_gate(spend_over_cont, z=Z):
    """The NRM1 internal capacity gate in its original definition (FA_PRC_v1 section 4): per version, nested held-out spend of
    the construction over CONT, estimate + z SE <= 0.004 nats NLL and <= 0.003 accuracy loss per task. spend_over_cont = {v:
    {'occ_nll', 'inc_nll', 'occ_acc', 'inc_acc': (est, se)}}. ORIGINAL_CAPACITY_GATE_PASS iff both versions pass. REPORTED
    beside the decisions; no function of this module reads it."""
    per = {}
    for v in VERSIONS:
        s = _v(spend_over_cont, v)
        rows = {k: _est((s or {}).get(k)) for k in CAPACITY_GATE_LIMITS}
        if any(x is None for x in rows.values()):
            return {'label': 'INCOMPLETE', 'error': f'version {v}: spend over CONT missing', 'gating': False}
        per[v] = {k: {'estimate': e, 'se': s_, 'upper': e + z * s_, 'limit': CAPACITY_GATE_LIMITS[k],
                      'pass': e + z * s_ <= CAPACITY_GATE_LIMITS[k]} for k, (e, s_) in rows.items()}
        per[v]['_pass'] = all(r['pass'] for k, r in per[v].items() if k != '_pass')
    ok = all(per[v]['_pass'] for v in VERSIONS)
    return {'label': 'ORIGINAL_CAPACITY_GATE_PASS' if ok else 'ORIGINAL_CAPACITY_GATE_FAIL', 'per_version': per, 'gating': False,
            'note': 'the original NRM1 gate recomputed and reported; it is not this study\'s eligibility rule'}


def gate_reproduction(recomputed, gate_record, twin_key, tol=REPRO_TOL):
    """T1 reproduction: recomputed = {'nested_means': {'CONT': {...}, twin_key: {...}}, 'spend_over_cont': {metric: (est, se)}}
    against the published GATE record (its 'nested_means' and 'spend_twin_over_cont'). REPRODUCED iff every number of the record
    is matched within tol (absolute) and the recomputed gate label equals the record's."""
    diffs = []
    try:
        for name in ('CONT', twin_key):
            for k, x in gate_record['nested_means'][name].items():
                diffs.append(abs(float(recomputed['nested_means'][name][k]) - float(x)))
        for k, x in gate_record['spend_twin_over_cont'].items():
            r = recomputed['spend_over_cont'][k]
            diffs += [abs(float(r[0]) - float(x[0])), abs(float(r[1]) - float(x[1]))]
        lab = capacity_gate({0: recomputed['spend_over_cont'], 1: recomputed['spend_over_cont']})['label']
    except (KeyError, TypeError, IndexError, ValueError) as ex:
        return {'label': 'INCOMPLETE', 'error': repr(ex)}
    if not all(_fin(x) for x in diffs):
        return {'label': 'NOT_REPRODUCED', 'error': 'non-finite difference'}
    same_gate = (lab == 'ORIGINAL_CAPACITY_GATE_PASS') == (gate_record.get('gate') == 'PASS')
    worst = max(diffs) if diffs else math.inf
    return {'label': 'REPRODUCED' if worst <= tol and same_gate else 'NOT_REPRODUCED', 'max_abs_difference': worst,
            'gate_label_matches': same_gate, 'tol': tol}


# ---------------------------------------------------------------- (c) capacity order, triggers, stop
def _assess(rec, needs_reproduction):
    if not rec:
        return {'state': 'MISSING'}
    if needs_reproduction and rec.get('reproduced') is not True:
        return {'state': 'STOPPED_FOR_DIAGNOSIS', 'reason': 'T0 / T1 reproduction not recorded as passed'}
    ch = construction_choice(rec.get('nested'))
    if ch['choice'] is None:
        return {'state': 'INCOMPLETE', 'reason': ch.get('error')}
    ev = (rec.get('evals') or {}).get(ch['choice'])
    if ev is None:
        return {'state': 'INCOMPLETE', 'reason': f'MODEL_SELECT utility of the chosen construction {ch["choice"]} missing'}
    q = qual(ev)
    if q['label'] == 'INCOMPLETE':
        return {'state': 'INCOMPLETE', 'reason': q.get('error')}
    other = 'A' if ch['choice'] == 'B1' else 'B1'           # recorded, never used (adjudication finding 10)
    oev = (rec.get('evals') or {}).get(other)
    oq = qual(oev)['label'] if oev is not None else 'INCOMPLETE'
    return {'state': 'ASSESSED', 'choice': ch['choice'], 'qual': q['label'], 'qual_reasons': q['reasons'],
            'qpe': qpe_label(rec.get('nested'), rec.get('ms'))['label'], 'other_construction': other,
            'other_would_qualify': None if oq == 'INCOMPLETE' else oq == 'QUALIFIED',
            'other_note': 'recorded, never used: the chosen construction is final for this capacity'}


def capacity_decision(by_capacity, repair=None):
    """PROTOCOL 6.3-6.6 as a deterministic function of recorded inputs. by_capacity = {'16x64' | '24x128' | '32x256': {'reproduced':
    bool (existing capacities only), 'nested': qpe_label input, 'ms': MODEL_SELECT paired costs B1 vs A, 'evals': {'A': {v:
    utility}, 'B1': {v: utility}}}}; repair = None (not yet decided), 'DECLINED', or {'card_committed_before_model_select_read': True (or the stricter 'card_committed_before_fit'), 'nested',
    'ms', 'evals'} (judged by the same choice and QUAL rules). States: PROTECT (capacity, construction), TRIGGER_T5,
    TRIGGER_T6, STOP_FAMILY_FA, STOPPED_FOR_DIAGNOSIS, INCOMPLETE (awaiting). A result present without its trigger is ignored
    and flagged."""
    by = by_capacity or {}
    steps, flags = [], []

    def protect(cap, a):
        if cap in EXISTING and by.get(PROSPECTIVE):
            flags.append('T5_RESULT_WITHOUT_TRIGGER_IGNORED')
        if isinstance(repair, dict):
            flags.append('REPAIR_WITHOUT_TRIGGER_IGNORED')
        return {'state': 'PROTECT', 'capacity': cap, 'construction': a['choice'], 'qpe': a['qpe'], 'steps': steps,
                'flags': flags}
    for cap in EXISTING:
        a = _assess(by.get(cap), needs_reproduction=True)
        steps.append({'capacity': cap, **a})
        if a['state'] == 'MISSING':
            return {'state': 'INCOMPLETE', 'awaiting': cap, 'steps': steps, 'flags': flags}
        if a['state'] in ('STOPPED_FOR_DIAGNOSIS', 'INCOMPLETE'):
            return {'state': a['state'], 'awaiting': cap, 'steps': steps, 'flags': flags}
        if a['qual'] == 'QUALIFIED':
            return protect(cap, a)
    a = _assess(by.get(PROSPECTIVE), needs_reproduction=False)
    steps.append({'capacity': PROSPECTIVE, **a})
    if a['state'] == 'MISSING':
        return {'state': 'TRIGGER_T5', 'run': PROSPECTIVE, 'steps': steps, 'flags': flags}
    if a['state'] == 'INCOMPLETE':
        return {'state': 'INCOMPLETE', 'awaiting': PROSPECTIVE, 'steps': steps, 'flags': flags}
    if a['qual'] == 'QUALIFIED':
        if isinstance(repair, dict):
            flags.append('REPAIR_WITHOUT_TRIGGER_IGNORED')
        return {'state': 'PROTECT', 'capacity': PROSPECTIVE, 'construction': a['choice'], 'qpe': a['qpe'], 'steps': steps,
                'flags': flags}
    if repair is None:
        return {'state': 'TRIGGER_T6', 'note': 'one documented structural repair may be proposed (committed card first) or '
                                               'declined', 'steps': steps, 'flags': flags}
    if isinstance(repair, dict):
        if not (repair.get('card_committed_before_model_select_read') is True or repair.get('card_committed_before_fit') is True):
            flags.append('REPAIR_WITHOUT_COMMITTED_CARD_IGNORED')
        else:
            a = _assess(repair, needs_reproduction=False)
            steps.append({'capacity': REPAIR, **a})
            if a['state'] in ('INCOMPLETE', 'MISSING'):
                return {'state': 'INCOMPLETE', 'awaiting': REPAIR, 'steps': steps, 'flags': flags}
            if a['qual'] == 'QUALIFIED':
                return {'state': 'PROTECT', 'capacity': REPAIR, 'construction': a['choice'], 'qpe': a['qpe'], 'steps': steps,
                        'flags': flags}
    return {'state': 'STOP_FAMILY_FA', 'statement': STOP_STATEMENT, 'steps': steps, 'flags': flags}


# ---------------------------------------------------------------- (e) decisions D1-D6 and the outcome class
def d1(cap):
    """D1 from a capacity_decision record."""
    st = (cap or {}).get('state')
    if st == 'STOP_FAMILY_FA':
        return {'label': 'NO_QUALIFYING_TASK_ONLY_RELEASE', 'statement': STOP_STATEMENT}
    if st != 'PROTECT':
        return {'label': 'INCOMPLETE', 'state': st}
    if cap['capacity'] == REPAIR:                          # a T6 repair is POST_HOC: never REPAIRED_TASK_ONLY_RELEASE
        return {'label': 'POST_HOC_REPAIR_QUALIFIED', 'capacity': REPAIR, 'construction': cap['construction'],
                'qpe': cap.get('qpe')}
    if cap['construction'] == 'A':
        return {'label': 'QUALIFIED_WITHOUT_PROBABILITY_REPAIR', 'capacity': cap['capacity']}
    lab = {'PE_ERROR_SUPPORTED': 'REPAIRED_TASK_ONLY_RELEASE', 'PE_ERROR_POINT_ONLY': 'QUALIFIED_B1_REPAIR_NOT_RESOLVED'}
    return {'label': lab.get(cap.get('qpe'), 'INCOMPLETE'), 'capacity': cap['capacity'], 'qpe': cap.get('qpe')}


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


def targeting(nolookup_entry, nested_lookup):
    """Reader-targeting point conditions of one comparison (PRC vs a control point), adjudication finding 3: the non-table pair
    winner's benefit (the grid's 'nolookup' extra: per-cell control AUC - PRC AUC on the registered cells) > 0 in both version
    means, AND the nested MODEL_FIT held-out pair LOOKUP benefit (AUC_control - AUC_prc, P1) > 0 in both versions. None if an
    input is missing."""
    vm = _version_means(nolookup_entry)
    nl = {v: _v(nested_lookup, v) for v in VERSIONS}
    if vm is None or not all(_fin(x) for x in nl.values()):
        return None
    return {'pass': all(vm[v] > 0 for v in VERSIONS) and all(nl[v] > 0 for v in VERSIONS), 'nolookup_version_means': vm,
            'nested_lookup': nl}


def f4(rec, z=Z):
    """Falsifier F4 (card FA_FRC_v1): rec = {'version': {v: {'diff': AUC_twin - AUC_prc of the nested held-out pair LOOKUP,
    'se': household-bootstrap SE}}}; pass iff diff - z se > 0 in both versions. None if missing. Not a stop."""
    rows = {}
    for v in VERSIONS:
        p = _v((rec or {}).get('version'), v) or {}
        d, s = p.get('diff'), p.get('se')
        if not _fin(d, s) or s < 0:
            return None
        rows[v] = {'diff': d, 'se': s, 'lower': d - z * s, 'pass': d - z * s > 0}
    return {'pass': all(r['pass'] for r in rows.values()), 'per_version': rows}


def _path(ids, alias_of):
    """Control path ids that exist, aliases mapped to the saved id, duplicates removed (order kept)."""
    out = []
    for i in ids or []:
        j = (alias_of or {}).get(i, i) if i else None
        if j and j not in out:
            out.append(j)
    return out


def d2(rec):
    """D2 at the PRC configuration selected at S2 (an alias of an earlier budget's selection is that configuration). rec =
    {'grid_valid', 'prc_s2': {'status': 'FITTED' | 'NOT_FEASIBLE', 'config_id', 'evals': {v: utility},
    'every_nonzero_selection_alias_of_twin': bool}, 'grid': {id: nrm grid entry of PRC_S2 vs id}, 'paths': {'HQ': [HQ_S2, HQ_BR],
    'SHUF': [SHUF_S2, SHUF_BR], 'ORD': [TWIN, ORD_S1, ORD_S2, ORD_S3, ORD_BR]} (None for a point that does not exist),
    'alias_of': {id: saved id}, 'costs_vs': {id: MODEL_SELECT paired costs of PRC_S2 vs id}, 'f4': f4 record, 'nolookup': {id:
    per-cell pair benefits of the non-table winner}, 'nested_lookup': {id: {v: AUC_id - AUC_prc}}}. Conjuncts: (a) G1; (b) vs
    TWIN nrm comparison (AGG_POSITIVE + recipient gate); (c) DOMINATES_CONTROL_FRONTIER vs the SHUF, HQ and ORD paths (an empty
    HQ / SHUF path is compared at TWIN, flag <name>_AT_IDENTITY); (d) F4; (e) reader targeting vs TWIN and every bracketing
    point Q*. Labels in order: NOT_COMPUTED_GRID_*, NOT_ASSESSED_NO_FEASIBLE_STRENGTH, NOT_ASSESSED_EXACT_ALIAS (family stops),
    INCOMPLETE, NOT_MET_F4, READER_TARGETED (b and c all pass, e fails), COMPONENT_BEYOND_MATCHED_CONTROLS (met) /
    MET_WITHOUT_TASK_FEASIBILITY, NO_REDUCTION_BEYOND_ANY_CONTROL, PARTIAL_<passed>_ONLY. No partial outcome is a claim."""
    why = _gate_label(rec.get('grid_valid'))
    if why:
        return {'label': why, 'met': False}
    p = rec.get('prc_s2') or {}
    if p.get('status') == 'NOT_FEASIBLE':
        return {'label': 'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'met': False}
    if p.get('every_nonzero_selection_alias_of_twin') is True:
        return {'label': 'NOT_ASSESSED_EXACT_ALIAS', 'met': False, 'family_stops': True,
                'statement': "every nonzero selection transmits exactly the twin's vectors: the family stops"}
    if p.get('status') != 'FITTED':
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'PRC_S2 status missing'}
    g1 = NR.point_eligibility(p.get('evals') or {})
    if g1['label'] == 'INCOMPLETE':
        return {'label': 'INCOMPLETE', 'met': False, 'error': 'PRC_S2 utility incomplete'}
    feas = g1['label'] == 'POINT_ELIGIBLE'
    grid, flags, comps = rec.get('grid') or {}, [], {}

    def incomplete(msg):
        return {'label': 'INCOMPLETE', 'met': False, 'error': msg, 'comparisons': comps}
    if 'TWIN' not in grid:
        return incomplete('grid entry vs TWIN missing')
    ct = NR.comparison(grid['TWIN'])
    if ct['benefit']['label'] == 'INCOMPLETE' or any(ct['recipient'][w]['label'] == 'INCOMPLETE' for w in NR.SINGLE_VIEWS):
        return incomplete('grid entry vs TWIN incomplete')
    comps['TWIN'] = {'benefit': ct['benefit']['label'], 'recipient_gate_pass': ct['recipient_gate_pass'], 'point': 'TWIN',
                     'pass': ct['benefit_positive'] and ct['recipient_gate_pass']}
    if ct['benefit']['label'] == 'ALIAS_ZERO':
        flags.append('PRC_S2_ALIAS_OF_TWIN')
    point = {'id': p.get('config_id', 'PRC_S2'), 'costs_vs': rec.get('costs_vs') or {}, 'grid_vs': grid}
    for name, key in D2_PATHS:
        ids = _path((rec.get('paths') or {}).get(key), rec.get('alias_of'))
        if not ids:
            if key == 'ORD':
                return incomplete('ORD path missing (it starts at TWIN)')
            ids = ['TWIN']
            flags.append(f'{key}_AT_IDENTITY')
        fr = NR.frontier_comparison(point, [{'id': i} for i in ids])
        if fr['label'] == 'INCOMPLETE':
            return incomplete(f'{name}: {fr.get("error")}')
        comps[name] = {'label': fr['label'], 'reason': fr.get('reason'), 'path': ids, 'point': fr.get('bracketing_control'),
                       'pass': fr['label'] == 'DOMINATES_CONTROL_FRONTIER', 'frontier': fr}
    f = f4(rec.get('f4'))
    if f is None:
        return incomplete('F4 record missing or non-finite')
    tg = {}
    for q in dict.fromkeys(c['point'] for c in comps.values() if c.get('point')):
        nl = (rec.get('nested_lookup') or {}).get(q)
        if nl is None and q == 'TWIN':                      # the same quantity as F4's difference
            nl = {v: f['per_version'][v]['diff'] for v in VERSIONS}
            flags.append('TWIN_NESTED_LOOKUP_FROM_F4')
        x = targeting((rec.get('nolookup') or {}).get(q), nl)
        if x is None:
            return incomplete(f'reader-targeting inputs vs {q} missing')
        tg[q] = x
    passed = []
    for n in D2_COMPARISONS:
        c = comps[n]
        c['targeting_pass'] = tg[c['point']]['pass'] if c.get('point') in tg else None
        if c['pass']:
            passed.append(n)
            if c['targeting_pass'] is False:
                flags.append(f'READER_TARGETED_{n}')
    all4 = len(passed) == len(D2_COMPARISONS)
    targeted = not all(x['pass'] for x in tg.values())
    if not f['pass']:
        label = 'NOT_MET_F4'
    elif all4 and targeted:
        label = 'READER_TARGETED'
    elif all4:
        label = 'COMPONENT_BEYOND_MATCHED_CONTROLS' if feas else 'MET_WITHOUT_TASK_FEASIBILITY'
    elif not passed:
        label = 'NO_REDUCTION_BEYOND_ANY_CONTROL'
    else:
        label = 'PARTIAL_' + '+'.join(passed) + '_ONLY'
    return {'label': label, 'met': label == 'COMPONENT_BEYOND_MATCHED_CONTROLS', 'task_feasible': feas, 'g1': g1['label'],
            'passed': passed, 'f4': f, 'targeting': tg, 'comparisons': comps, 'flags': sorted(set(flags)),
            'minimum_detectable_pair_benefit': 'about 0.015 (2 se_agg); "not separated" is the default reading',
            'note': 'no partial outcome is a component claim'}


def reference_credit(rec, refs):
    """Reader-budget rule for references (coordinator correction 23:4xZ): a reference flagged READER_BUDGET_ASYMMETRIC (input
    rec['reader_budget'][ref], frc.readerbudget.contrast, pair view) is credited iff the NON-TABLE winner's point benefit (the
    grid's 'nolookup' extra vs that reference) is > 0 in both version means. The nested MODEL_FIT LOOKUP condition does not apply
    to references (they are never refitted on MODEL_FIT). Returns {ref: 'CREDITED' | 'NOT_CREDITED' | 'INCOMPLETE'} for the
    flagged references only (missing nolookup inputs: INCOMPLETE, never 'not beaten')."""
    out = {}
    for ref in refs:
        if (rec.get('reader_budget') or {}).get(ref) == 'READER_BUDGET_ASYMMETRIC':
            vm = _version_means((rec.get('nolookup') or {}).get(ref))
            out[ref] = 'INCOMPLETE' if vm is None else ('CREDITED' if all(vm[v] > 0 for v in VERSIONS) else 'NOT_CREDITED')
    return out


def _d3(rec, cl):
    """D3 = NRM1 G4 with the reader-budget rule (reference_credit). The rule can only withdraw credit, so it is applied in
    three-valued (Kleene) form: a NOT_CREDITED reference makes a met G4 NOT_SUPERIOR; an INCOMPLETE flagged reference makes a met
    G4 INCOMPLETE (it is never counted as not beaten); a G4 already failing on its own results stays NOT_SUPERIOR (the
    incomplete references are listed)."""
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
    """D4 = NRM1 G5 with the same reader-budget rule for NEURAL_RAW and H_LJ: a NOT_CREDITED target reference makes the level
    NOT_MET (every level needs both targets); an INCOMPLETE one makes a level above NOT_MET INCOMPLETE; a NOT_MET level stays."""
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


def decisions(rec):
    """D1-D6 and the outcome class. rec = {'capacity_decision': capacity_decision(...), the d2() inputs, 'grid' also holding
    the entries vs the 9 references, 'reader_budget': {ref: flag}, 'joint': None or the nrm G6 record}. D3 (NRM1 G4) and D4
    (NRM1 G5), both with the reader-budget rule (reference_credit), are read at the same configuration as D2, never another."""
    cap = rec.get('capacity_decision') or {}
    out = {'D1': d1(cap), 'D6': {'label': 'NOT_CLAIMED', 'note': 'deterministic privacy funnel / IBSI family; FARE is the '
                                                                   'published finite-cell input-routing method'}}
    flags = []
    if cap.get('state') == 'STOP_FAMILY_FA':
        for k in ('D2', 'D3', 'D4'):
            out[k] = {'label': 'NOT_ASSESSED_PROTECTED_ROUTING_UNTESTED', 'met': False}
        flags.append('PROTECTED_ROUTING_UNTESTED')
    elif cap.get('state') != 'PROTECT':
        for k in ('D2', 'D3', 'D4'):
            out[k] = {'label': 'INCOMPLETE', 'met': False}
    else:
        out['D2'] = d2(rec)
        p = rec.get('prc_s2') or {}
        if out['D2']['label'] in ('NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID',
                                  'NOT_ASSESSED_NO_FEASIBLE_STRENGTH', 'NOT_ASSESSED_EXACT_ALIAS') or p.get('status') != 'FITTED':
            lab = out['D2']['label'] if out['D2']['label'] != 'INCOMPLETE' else 'INCOMPLETE'
            out['D3'] = {'label': lab, 'met': False}
            out['D4'] = {'label': lab, 'met': False}
            if lab == 'NOT_ASSESSED_EXACT_ALIAS':
                flags.append('FAMILY_STOPS_EXACT_ALIAS')
        else:
            fr = (out['D2'].get('comparisons') or {}).get('ORD_FRONTIER', {}).get('frontier')
            cl = NR.claims({'evals': p.get('evals') or {}, 'grid': rec.get('grid') or {}, 'grid_valid': rec.get('grid_valid'),
                            'frontier': fr, 'matched': {}, 'joint': None})
            out['D3'] = _d3(rec, cl)
            out['D4'] = _d4(rec, cl)
            res = cl.get('reserve') or {}
            if cl['G1']['label'] == 'TASK_FEASIBLE' and not res.get('qualified'):
                flags.append('POINT_FEASIBLE_NOT_RESERVE_QUALIFIED')
    j = rec.get('joint')
    if not j or not all(j.get(k) for k in ('separate', 'income_first', 'occupation_first')):
        out['D5'] = {'label': 'NOT_ASSESSED_MISSING_CONTROLS', 'statement': 'no joint-design advantage is claimed'}
    else:
        p = rec.get('prc_s2') or {}
        cl = NR.claims({'evals': p.get('evals') or {}, 'grid': rec.get('grid') or {}, 'grid_valid': rec.get('grid_valid'),
                        'matched': {}, 'joint': j})
        out['D5'] = {'label': cl['G6']['label'], 'detail': cl['G6']}
    bad = ('INCOMPLETE', 'NOT_COMPUTED_GRID_VALIDITY_UNKNOWN', 'NOT_COMPUTED_GRID_INVALID')
    if out['D1']['label'] == 'INCOMPLETE' or out['D2']['label'] in bad or out['D3']['label'] in bad:
        oc = 'INCOMPLETE'
    elif out['D2']['met'] and out['D3']['met']:
        oc = 'CANDIDATE'
    elif out['D2']['met']:
        oc = 'COMPONENT'
    elif out['D1']['label'] == 'REPAIRED_TASK_ONLY_RELEASE':
        oc = 'ENGINEERING_REPAIR'
    else:
        oc = 'FINITE_NEGATIVE_RESULT'
    if out['D3'].get('met') and not out['D2'].get('met'):
        flags.append('D3_WITHOUT_D2')
    out['outcome_class'] = {'label': oc, 'flags': sorted(set(flags)),
                            'note': 'a near-limit point estimate is never promoted as robust; MODEL_SELECT is adaptively used '
                                    '(development evidence, never confirmation)'}
    out['scope'] = list(NR.SCOPE)
    return out


# ---------------------------------------------------------------- (f) strongest feasible nonzero release; effective refit
def strongest_nonzero(configs):
    """PROTOCOL 10.4. configs = [{'config_id', 'lam', 'evals': {v: utility}, 'pair_vs_twin': nrm grid pair entry, 'alias_of':
    None or an earlier id}]. Eligible: lam > 0, G1 (POINT_ELIGIBLE), not an alias of an earlier configuration, pair benefit vs
    TWIN neither ALIAS_ZERO nor INCOMPLETE. Choice: the largest registered-cell aggregate pair benefit x vs TWIN; |dx| <= 1e-12
    ties go to the smaller lambda, then config_id. Selected on registered MODEL_SELECT cells (winner's curse): descriptive."""
    elig, rows = [], []
    for c in configs or []:
        r = {'config_id': c.get('config_id'), 'lam': c.get('lam')}
        lam = c.get('lam')
        if not _fin(lam) or lam <= 0:
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
                r.update({'status': 'ELIGIBLE', 'x': b['aggregate']['mean']})
                elig.append(r)
        rows.append(r)
    if not elig:
        return {'label': 'NO_FEASIBLE_NONZERO_RELEASE', 'selected': None, 'records': rows}
    best = max(r['x'] for r in elig)
    near = sorted((r for r in elig if r['x'] >= best - TIE_TOL), key=lambda r: (r['lam'], str(r['config_id'])))
    return {'label': 'SELECTED', 'selected': near[0]['config_id'], 'x': near[0]['x'], 'tied': [r['config_id'] for r in near],
            'records': rows, 'note': 'chosen on registered MODEL_SELECT cells; robustness cells never change D1-D6'}


def refit_effectiveness(splits, seeds, checks, minimum_set, distinct_inputs):
    """PROTOCOL 10.4 'genuinely effective' classifier refit. splits = {'R1'..'R4': {'assignment_sha256': str, 'score_overlap':
    {other repetition: share of SCORE households shared}}} (recorded in READER_SPLITS.json); seeds = {'S<v>k<k>': int} for k =
    0..3; checks = recorded reader-seed checks (nrm format; 'equal_k0_k1' = the recorded equality of the check's two seeds);
    minimum_set = the strongest release and its matched controls. REFIT_GENUINELY_EFFECTIVE iff (a) R3 and R4 use different
    household splits (assignment hashes distinct from each other and from R1 / R2; every pairwise SCORE overlap recorded, finite,
    in [0, 1) ) and (b) grid_validity holds on all eight seeds (no seed-dependent equality on the minimum set, others explained).
    Otherwise REFIT_NOT_GENUINELY_EFFECTIVE: the robustness cells are counted once (per failed part)."""
    reps = ('R1', 'R2') + NR.ROB_REPS
    errs = []
    hs = [((splits or {}).get(r) or {}).get('assignment_sha256') for r in reps]
    if not all(isinstance(h, str) and h for h in hs) or len(set(hs)) != len(hs):
        errs.append('SPLITS_NOT_DIFFERENT_OR_NOT_RECORDED')
    else:
        for r in NR.ROB_REPS:
            for q in reps:
                if q == r:
                    continue
                o = (((splits or {}).get(r) or {}).get('score_overlap') or {}).get(q)
                if o is None:
                    o = (((splits or {}).get(q) or {}).get('score_overlap') or {}).get(r)
                if not _fin(o) or not 0 <= o < 1:
                    errs.append(f'SCORE_OVERLAP_{r}_{q}_NOT_RECORDED_OR_IDENTICAL')
    gv = grid_validity(seeds, checks, minimum_set, distinct_inputs, required_k=NR.REG_K + NR.ROB_K)
    if not gv['effective']:
        errs.append('READER_SEEDS_NOT_EFFECTIVE')
    count_once = ([] if not any(e.startswith(('SPLITS', 'SCORE')) for e in errs) else ['robustness repetitions']) + \
                 (['robustness reader seeds'] if not gv['effective'] else [])
    return {'label': 'REFIT_GENUINELY_EFFECTIVE' if not errs else 'REFIT_NOT_GENUINELY_EFFECTIVE', 'reasons': errs,
            'grid_validity': gv, 'counted_once': count_once}


# ---------------------------------------------------------------- (g) reader-seed validity of the FRC1 grid
def forbidden_seeds():
    """Every earlier grid's reader seeds: MODEL_SEED_BASE + {300, 400, 500, 600} + 10 v + k, k = 0..9 (a local equivalent of
    nrm.rules.forbidden_seeds, which reads its own module constant (300, 500); nrm is not monkeypatched)."""
    return {NR.MODEL_SEED_BASE + off + 10 * v + k for off in FORBIDDEN_OFFSETS for v in VERSIONS for k in range(10)}


def registered_seed(v, k):
    return NR.MODEL_SEED_BASE + SEED_OFFSET + 10 * int(v) + int(k)


def grid_validity(seeds, checks, minimum_set, distinct_inputs, required_k=NR.REG_K):
    """nrm.rules.grid_validity (A-SC1, unchanged) with its reader-seed part replaced by this grid's: the seeds of `required_k`
    for both versions must be recorded distinct integers, every recorded seed key must be S<v>k<k> and equal the registered
    MODEL_SEED_BASE + 700 + 10 v + k, and none may equal an earlier grid's seed (offsets 300, 400, 500, 600). The recipe-
    equality parts (ii) and (iii) are nrm's, unchanged."""
    base = NR.grid_validity(seeds, checks, minimum_set, distinct_inputs)
    errors = [e for e in base['errors'] if not e.startswith('reader seeds')]
    keys = [f'S{v}k{k}' for v in VERSIONS for k in required_k]
    s = seeds if isinstance(seeds, dict) else {}
    sv = [s.get(k) for k in keys] + [x for k, x in s.items() if k not in keys]
    local = []
    if any(not isinstance(x, int) or isinstance(x, bool) for x in sv) or len(set(sv)) != len(sv):
        local.append(f'reader seeds must be recorded distinct integers, at least {keys}')
    else:
        bad_keys = sorted(k for k in s if not _SEEDKEY.fullmatch(str(k)))
        if bad_keys:
            local.append(f'reader seed keys not of the form S<v>k<k>: {bad_keys}')
        reused = sorted(set(sv) & forbidden_seeds())
        if reused:
            local.append(f'reader seeds reuse earlier grid seeds {reused}')
        wrong = sorted(k for k, x in s.items() if _SEEDKEY.fullmatch(str(k))
                       and x != registered_seed(*_SEEDKEY.fullmatch(str(k)).groups()))
        if wrong:
            local.append(f'reader seeds differ from the registered MODEL_SEED_BASE + {SEED_OFFSET} + 10 v + k: {wrong}')
    errors = local + errors
    return {**base, 'label': 'READER_SEEDS_NOT_EFFECTIVE' if errors else 'READER_SEEDS_EFFECTIVE', 'effective': not errors,
            'errors': errors, 'forbidden_offsets': list(FORBIDDEN_OFFSETS), 'offset': SEED_OFFSET}
