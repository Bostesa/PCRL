"""Utility-reserve screen of the PCRL income-headroom successor development campaign (statistical reviewer's rule;
roles/statistical_reviewer/UTILITY_RESERVE_RULE.md, machine-readable RESERVE_RULE.json).

A DEVELOPMENT SELECTION HEURISTIC on adaptively used MODEL_SELECT data. It is not a confidence guarantee and it does not
change any cap: the original caps (accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005; within-SEX
occupation summed-Brier excess against U0 <= 0.005) stay the only acceptance limits, and point eligibility is judged separately.

Rule, per seed, on the dict returned by ihs.evaluate.measure (or any dict with the same 'utility' layout):
- income (6 conditions per seed, 12 per configuration): for each income clause (reference U0 / Ucal x accuracy_loss,
  nll_excess, brier_excess), margin = cap - value must be >= k * se_used, with k = 2.0 and
  se_used = max(own household-clustered paired SE, floor SE). The floor SE is the eligible reference H_LJ's MODEL_SELECT SE
  for the same condition and seed (H_LJ carries the inherited JOINT_TREE occupation output), rescaled by
  sqrt(5306 / households) if the evaluation has another household count. SEs are used at their measured scale (no projection).
- occupation clauses and within-SEX guards (6 + 2 per seed): if the occupation output is the inherited H_LJ / JOINT_TREE output
  (all 8 values equal the inherited values to 1e-12, or the occupation wire hash equals the inherited hash when the rule records
  one), they are NOT screened and are returned in an inherited-constraint report. Otherwise each must either meet the same
  reserve (margin >= k * max(own SE, inherited SE)) or keep at least the inherited output's margin (non-worsening).
- status per seed: RESERVE_MET when every income condition is met and no occupation condition is worsened; otherwise
  RESERVE_NOT_MET. two_seed_status: RESERVE_MET only when both seeds are met.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

RULE_PATH = (Path(__file__).resolve().parent.parent / 'results' / 'pcrl_income_headroom_successor_v1' / 'roles'
             / 'statistical_reviewer' / 'RESERVE_RULE.json')
SCHEMA = 'IHS1-reserve-rule-v1'
CAPS = {'accuracy_loss': 0.01, 'nll_excess': 0.01, 'brier_excess': 0.005}
GUARD_CAP = 0.005
REFS = ('U0', 'Ucal')
METRICS = ('accuracy_loss', 'nll_excess', 'brier_excess')
INCOME = tuple(f'income|{r}|{m}' for r in REFS for m in METRICS)
OCCUPATION = tuple(f'occupation|{r}|{m}' for r in REFS for m in METRICS)
GUARDS = ('guard|sex0', 'guard|sex1')
MET, NOT_MET = 'RESERVE_MET', 'RESERVE_NOT_MET'
TWO_SEED = ('RESERVE_MET', 'RESERVE_ONE_SEED_ONLY', 'RESERVE_NOT_MET', 'RESERVE_PENDING_SECOND_SEED')

_cache = {}


def load_rule(path=RULE_PATH):
    """The registered rule (RESERVE_RULE.json) with its SHA-256."""
    p = Path(path)
    key = str(p)
    if key not in _cache:
        raw = p.read_bytes()
        rule = json.loads(raw)
        if rule.get('schema') != SCHEMA:
            raise ValueError(f'unexpected reserve-rule schema {rule.get("schema")!r}')
        rule['_sha256'] = hashlib.sha256(raw).hexdigest()
        _cache[key] = rule
    return _cache[key]


def _finite(*xs):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _conditions(m, seed):
    """Name -> {value, cap, se} for the 12 clauses and 2 guards of one seed; refuses malformed or re-capped input."""
    u = m['utility']
    out = {}
    for c in u['clauses']:
        if 'seed' in c and int(c['seed']) != int(seed):
            raise ValueError(f'clause of seed {c["seed"]} passed to the seed-{seed} screen')
        name = f"{c['task']}|{c['reference']}|{c['metric']}"
        if c['task'] not in ('income', 'occupation') or c['reference'] not in REFS or c['metric'] not in CAPS:
            raise ValueError(f'unknown clause {name}')
        if name in out:
            raise ValueError(f'duplicate clause {name}')
        if float(c['cap']) != CAPS[c['metric']]:
            raise ValueError(f'{name}: cap {c["cap"]} differs from the original cap {CAPS[c["metric"]]} (caps are never changed)')
        out[name] = {'value': float(c['value']), 'cap': float(c['cap']), 'se': float(c['se'])}
    for g in u['guards']:
        if 'seed' in g and int(g['seed']) != int(seed):
            raise ValueError(f'guard of seed {g["seed"]} passed to the seed-{seed} screen')
        name = f"guard|{g['group']}"
        if name not in GUARDS or name in out:
            raise ValueError(f'unknown or duplicate guard {name}')
        if float(g['cap']) != GUARD_CAP:
            raise ValueError(f'{name}: cap {g["cap"]} differs from the original guard cap {GUARD_CAP}')
        out[name] = {'value': float(g['value']), 'cap': float(g['cap']), 'se': float(g['se'])}
    missing = [n for n in INCOME + OCCUPATION + GUARDS if n not in out]
    if missing:
        raise ValueError(f'missing conditions {missing}')
    return out


def _evaluate(name, c, floor_se, k):
    value, cap, own = c['value'], c['cap'], c['se']
    rec = {'condition': name, 'value': value, 'cap': cap, 'own_se': own, 'floor_se': floor_se}
    if not _finite(value, own, floor_se) or own < 0 or floor_se < 0:
        rec.update({'margin': cap - value if _finite(value) else -math.inf, 'se_used': None, 'se_source': None,
                    'required_margin': None, 'reserve': None, 'ratio': None, 'met': False, 'met_with_own_se': False,
                    'point_pass': bool(_finite(value) and value <= cap), 'nonfinite': True})
        return rec
    margin = cap - value
    se_used = max(own, floor_se)
    ratio = margin / se_used if se_used > 0 else (math.inf if margin > 0 else (0.0 if margin == 0 else -math.inf))
    rec.update({'margin': margin, 'se_used': se_used, 'se_source': 'own' if own >= floor_se else 'floor',
                'required_margin': k * se_used, 'reserve': margin - k * se_used, 'ratio': ratio,
                'met': bool(margin >= k * se_used and margin >= 0), 'met_with_own_se': bool(margin >= k * own and margin >= 0),
                'point_pass': bool(value <= cap), 'nonfinite': False})
    return rec


def _occupation_identity(m, conds, seed, rule):
    """('INHERITED' | 'CHANGED', method). Hash identity when the rule records the inherited wire hash and m carries one."""
    occ = rule['occupation']
    s = str(int(seed))
    inh_hash = (occ.get('inherited_wire_sha256') or {}).get(s)
    own_hash = (m.get('wire_sha256') or {}).get('occupation')
    vals = occ['inherited_values'][s]
    tol = float(occ['identity_tolerance'])
    same_values = all(_finite(conds[n]['value']) and abs(conds[n]['value'] - float(vals[n])) <= tol for n in OCCUPATION + GUARDS)
    if inh_hash and own_hash:
        same = own_hash == inh_hash
        if same != same_values:
            raise AssertionError('occupation wire-hash identity and value identity disagree')
        return ('INHERITED' if same else 'CHANGED'), 'wire_sha256'
    return ('INHERITED' if same_values else 'CHANGED'), 'values_1e-12'


def screen(m, seed, rule=None):
    """Reserve screen of one seed's MODEL_SELECT evaluation. Returns status ('RESERVE_MET' / 'RESERVE_NOT_MET'), failed
    (condition names), the per-condition numbers, the occupation side and the inherited-constraint report."""
    rule = load_rule() if rule is None else rule
    s = str(int(seed))
    if s not in ('0', '1'):
        raise ValueError(f'seed must be 0 or 1, not {seed!r}')
    k = float(rule['multiplier_k'])
    conds = _conditions(m, seed)
    G = int(m['utility']['households'])
    G_floor = int(rule['floor']['households'])
    scale = math.sqrt(G_floor / G)
    floors = rule['floor']['se'][s]

    income = [_evaluate(n, conds[n], float(floors[n]) * scale, k) for n in INCOME]
    failed = [r['condition'] + (':NONFINITE' if r['nonfinite'] else '') for r in income if not r['met']]

    side, method = _occupation_identity(m, conds, seed, rule)
    inh_margin = rule['occupation']['inherited_margins'][s]
    occ = []
    for n in OCCUPATION + GUARDS:
        r = _evaluate(n, conds[n], float(floors[n]) * scale, k)
        r['inherited_margin'] = float(inh_margin[n])
        r['non_worsening'] = bool(r['margin'] >= float(inh_margin[n])) if _finite(r['margin']) else False
        occ.append(r)
    report = None
    if side == 'INHERITED':
        below = [r['condition'] for r in occ if not r['met']]
        report = {'status': 'INHERITED_CONSTRAINTS_BIND' if below else 'INHERITED_CONSTRAINTS_HAVE_RESERVE',
                  'below_reserve': below, 'conditions': occ,
                  'note': 'inherited H_LJ / JOINT_TREE occupation output: reported, not screened (no income-side mechanism can change it)'}
        occupation_side = 'INHERITED'
    else:
        worse = [r for r in occ if not (r['met'] or r['non_worsening'])]
        failed += [r['condition'] + (':NONFINITE' if r['nonfinite'] else '') for r in worse]
        occupation_side = 'CHANGED_WORSENED' if worse else ('CHANGED_RESERVE_MET' if all(r['met'] for r in occ) else 'CHANGED_NON_WORSENING')

    ratios = [r['ratio'] for r in income if r['ratio'] is not None]
    return {'rule': SCHEMA, 'rule_sha256': rule.get('_sha256'), 'seed': int(seed), 'status': NOT_MET if failed else MET,
            'failed': failed, 'multiplier_k': k, 'households': G, 'floor_households': G_floor, 'floor_scale': scale,
            'income': income, 'income_met_with_own_se_only': [r['condition'] for r in income if r['met_with_own_se'] and not r['met']],
            'min_income_ratio': min(ratios) if len(ratios) == len(income) else None,
            'occupation_side': occupation_side, 'occupation_identity_method': method,
            'occupation': occ if side != 'INHERITED' else None, 'inherited_constraint_report': report,
            'point_pass_all': all(r['point_pass'] for r in income + occ),
            'note': 'development selection heuristic on adaptively used MODEL_SELECT data; not a confidence guarantee; caps unchanged'}


def two_seed_status(screens):
    """Configuration label from the per-seed screens (a list of screen dicts or {seed: screen}).
    RESERVE_MET only if seeds 0 and 1 are both RESERVE_MET; RESERVE_ONE_SEED_ONLY if both screened and exactly one met;
    RESERVE_NOT_MET if no screened seed met; RESERVE_PENDING_SECOND_SEED if only one seed was screened and it met."""
    items = list(screens.values()) if isinstance(screens, dict) else list(screens)
    if not items:
        raise ValueError('no screens')
    per = {}
    for sc in items:
        s = int(sc['seed'])
        if s not in (0, 1) or s in per:
            raise ValueError(f'invalid or duplicate seed {s}')
        if sc['status'] not in (MET, NOT_MET):
            raise ValueError(f'unknown per-seed status {sc["status"]!r}')
        per[s] = sc
    met = [s for s in sorted(per) if per[s]['status'] == MET]
    if len(per) == 2:
        status = 'RESERVE_MET' if len(met) == 2 else ('RESERVE_ONE_SEED_ONLY' if len(met) == 1 else 'RESERVE_NOT_MET')
    else:
        status = 'RESERVE_PENDING_SECOND_SEED' if met else 'RESERVE_NOT_MET'
    ratios = [per[s]['min_income_ratio'] for s in sorted(per)]
    return {'status': status, 'meets_reserve': status == 'RESERVE_MET',
            'failed': [f's{s}|{f}' for s in sorted(per) for f in per[s]['failed']],
            'per_seed': {s: per[s]['status'] for s in sorted(per)},
            'occupation_side': {s: per[s]['occupation_side'] for s in sorted(per)},
            'min_income_ratio': None if any(r is None for r in ratios) else min(ratios),
            'seeds_screened': sorted(per)}
