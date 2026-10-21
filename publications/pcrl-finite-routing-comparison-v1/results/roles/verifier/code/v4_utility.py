"""V4 MODEL_SELECT utility (T3 / E1), own code.

For every evaluated release (records .private/<campaign>/runs/FA/<cfg>/EVAL_MS*.json and runs/SOURCE/FA/<twin>/EVAL_MS*.json) and
the reference record runs/REFERENCES/EVAL_MS.json: recompute from the saved MODEL_SELECT wires and the MODEL_SELECT labels
(frc.common.load, operation 'verify', called from THIS script) the 12 clauses + 2 guards per version against U0_S<v> / Ucal_S<v>
(loaded with the admitted loader ohn.devrun.reference_release; row order checked against the admitted frozen-bank ids), the
household-clustered SEs, point status, the whole-release / occupation-side / income reserve (k = 2, H_LJ floor SE of the
occupation-headroom RULES.json read as data), absolute metrics, distinct vectors, wire hashes, the paired per-person costs against
every recorded comparator, and the governing reserve_rules block; compare with the recorded values.

Then the decision layer, from my own numbers only: QUAL (Q1: G1 per version; Q2: value <= cap - S2 increment) of A and B1 at
every capacity; the Q-PE label (my nested d_v and SE_used from records/V3_NESTED__<tag>.json and my own MODEL_SELECT paired
occupation NLL B1 - A); the construction choice; the capacity walk (16 x 64, then 24 x 128); D1; compared with
results/<campaign>/decisions/QUAL_DECISION.json.

    python3 .../v4_utility.py --tag V1 [--nested-tag V1]
    python3 .../v4_utility.py --selftest
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

TOL = 1e-12
TOL_RATIO = 1e-9
FLOOR_RULES = V.ROOT / 'results' / 'pcrl_occupation_headroom_overnight_v1' / 'roles' / 'statistical_reviewer' / 'RULES.json'
CAP_KEY = {'C64': '16x64', 'C128': '24x128', 'C256': '32x256'}


def load_labels(purpose):
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import frc.common as C
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify', purpose)
    return {'income': np.asarray(S['income']), 'occupation': np.asarray(S['occupation']), 'sex': np.asarray(S['sex']),
            'unit': np.asarray(S['unit']), 'row_id': np.asarray(S['row_id'])}


def floor_se():
    r = json.loads(FLOOR_RULES.read_text())
    fl = r['reserve']['floor']
    if int(fl['households']) != V.FLOOR_HOUSEHOLDS or float(r['reserve']['multiplier_k']) != V.K_RESERVE:
        raise ValueError('reserve floor constants differ')
    return fl['se']


_refs = {}


def references(v, row_id):
    """U0_S<v> and Ucal_S<v> MODEL_SELECT arrays (admitted loader), row order checked, wires against SOURCE_ADMISSION.json."""
    if v in _refs:
        return _refs[v]
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    from ohn.devrun import reference_release
    from ohn import common as OC
    _, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
    if not np.array_equal(np.asarray(ids['row_id__MODEL_SELECT']), row_id):
        raise PermissionError('frozen bank MODEL_SELECT row order differs from the admitted rows')
    adm = json.loads((V.RESULTS / 'SOURCE_ADMISSION.json').read_text())['reference_releases_MODEL_SELECT_wire_sha256']
    out, prov = {}, {}
    for name in V.REFS:
        r = reference_release(name, v, 'MODEL_SELECT')
        out[name] = {t: np.asarray(r[t], dtype=np.float64) for t in V.TASKS}
        prov[name] = all(V.wire_sha(out[name][t]) == adm[f'{name}/S{v}'][t] for t in V.TASKS)
    _refs[v] = (out, prov)
    return _refs[v]


def comparator(spec, v, row_id):
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    if spec == 'W':
        from tsz.evaluate import source_a1, h3_income
        return {'income': np.asarray(h3_income(v), dtype=np.float64), 'occupation': np.asarray(source_a1(v), dtype=np.float64)}
    if spec.startswith('REF:'):
        from ohn.devrun import reference_release
        r = reference_release(spec[4:], v, 'MODEL_SELECT')
        return {t: np.asarray(r[t], dtype=np.float64) for t in V.TASKS}
    if spec.startswith('NRM:'):
        fam, cfg = spec[4:].split('/')
        return release_arrays(V.SRC_RUNS / fam / cfg / f'V{v}')[0]
    fam, cfg = spec.split('/')
    return release_arrays(V.RUNS / fam / cfg / f'V{v}')[0]


def comparator_exists(spec, v):
    """Whether a release comparator ('FA/C' of this campaign or 'NRM:FA/C' of the source) has a saved release for version v."""
    if spec == 'W' or spec.startswith('REF:'):
        return True
    if spec.startswith('NRM:'):
        fam, cfg = spec[4:].split('/')
        return (V.SRC_RUNS / fam / cfg / f'V{v}' / 'RELEASE.json').is_file()
    fam, cfg = spec.split('/')
    return (V.RUNS / fam / cfg / f'V{v}' / 'RELEASE.json').is_file()


def release_arrays(d, role='MODEL_SELECT'):
    rec = json.loads((Path(d) / 'RELEASE.json').read_text())
    h = V.file_sha(Path(d) / 'RELEASE.npz')
    if h != rec['release_sha256']:
        raise PermissionError(f'{d}: RELEASE.npz differs from its record')
    with np.load(Path(d) / 'RELEASE.npz', allow_pickle=False) as z:
        arr = {t: z[f'{role}__{t}'].copy() for t in V.TASKS}
    return arr, rec, h


def compute(S, rel, refs, floor_v, households):
    labels = {t: S[t] for t in V.TASKS}
    conds, own = V.conditions_version(labels, rel, refs, S['sex'], S['unit'])
    res = V.reserve_version(conds, floor_v, households)
    absolute = {t: {'acc_loss': float(own[t]['err'].mean()), 'nll': float(own[t]['nll'].mean()), 'brier': float(own[t]['brier'].mean())}
                for t in V.TASKS}
    absolute['occupation']['men_brier'] = float(own['occupation']['brier'][S['sex'] == 0].mean())
    absolute['occupation']['women_brier'] = float(own['occupation']['brier'][S['sex'] == 1].mean())
    distinct = {t: V.row_ids(rel[t])[1] for t in V.TASKS}
    wires = {t: V.wire_sha(rel[t]) for t in V.TASKS}
    return {'conditions': conds, 'reserve': res, 'absolute': absolute, 'distinct': distinct, 'wires': wires, 'own': own}


def paired(S, own_a, own_b):
    """loss(a) - loss(b) per person (acc_loss = error), household-clustered SEs; men's / women's summed Brier (linearized subgroup
    SE); total NLL (income + occupation, per person). Deterministic releases: Monte Carlo SE 0."""
    out = {}
    for t in V.TASKS:
        for m, key in (('acc_loss', 'err'), ('nll', 'nll'), ('brier', 'brier')):
            d = own_a[t][key] - own_b[t][key]
            out[f'{t}|{m}'] = {'value': float(d.mean()), 'se': V.cluster_se(d, S['unit']), 'se_mc': 0.0}
            if m == 'brier':
                for g, name in ((0, 'men_brier'), (1, 'women_brier')):
                    mask = S['sex'] == g
                    out[f'{t}|{name}'] = {'value': float(d[mask].mean()), 'se': V.subgroup_se(d, mask, S['unit']), 'se_mc': 0.0}
    tot = (own_a['income']['nll'] + own_a['occupation']['nll']) - (own_b['income']['nll'] + own_b['occupation']['nll'])
    out['total|nll'] = {'value': float(tot.mean()), 'se': V.cluster_se(tot, S['unit']), 'se_mc': 0.0}
    return out


def compare_version(mine, recv, v, costs_mine):
    mm = []
    u = recv['utility']
    rec = {f"{c['task']}|{c['reference']}|{c['metric']}": c for c in u['clauses']}
    rec.update({f"guard|{g['group']}": g for g in u['guards']})
    if set(rec) != set(V.CONDITIONS) or len(u['clauses']) != 12 or len(u['guards']) != 2:
        mm.append({'what': 'condition set', 'recorded': sorted(rec)})
    dv = dse = 0.0
    for n in V.CONDITIONS:
        r, m = rec.get(n), mine['conditions'][n]
        if r is None:
            continue
        if int(r.get('seed', v)) != v:
            mm.append({'what': f'{n} seed', 'recorded': r.get('seed')})
        if float(r['cap']) != m['cap']:
            mm.append({'what': f'{n} cap', 'recorded': r['cap'], 'mine': m['cap']})
        dv = max(dv, abs(float(r['value']) - m['value']))
        dse = max(dse, abs(float(r['se']) - m['se']))
        if bool(r.get('point_pass')) != m['point_pass'] or (float(r['value']) <= float(r['cap'])) != m['point_pass']:
            mm.append({'what': f'{n} point status', 'recorded': r.get('point_pass'), 'mine': m['point_pass']})
    if dv > TOL:
        mm.append({'what': 'clause / guard values', 'max_abs_diff': dv})
    if dse > TOL:
        mm.append({'what': 'clause / guard SEs', 'max_abs_diff': dse})
    pt = recv.get('point', {})
    my_failed = sorted(n for n in V.CONDITIONS if not mine['conditions'][n]['point_pass'])
    point_all = not my_failed
    worst = max(mine['conditions'][n]['value'] / mine['conditions'][n]['cap'] for n in V.CONDITIONS)
    if bool(pt.get('point_pass_12_plus_2')) != point_all:
        mm.append({'what': 'point_pass_12_plus_2', 'recorded': pt.get('point_pass_12_plus_2'), 'mine': point_all})
    if sorted(pt.get('failed', [])) != my_failed:
        mm.append({'what': 'failed', 'recorded': pt.get('failed'), 'mine': my_failed})
    if not math.isclose(float(pt.get('max_value_over_cap', math.nan)), worst, rel_tol=0, abs_tol=1e-10):
        mm.append({'what': 'max_value_over_cap', 'recorded': pt.get('max_value_over_cap'), 'mine': worst})
    rs, me = recv.get('reserve_without_mc') or recv.get('reserve') or {}, mine['reserve']
    if not rs:
        mm.append({'what': 'no per-version reserve block'})
    else:
        for k, val in (('whole_release_met', me['met']), ('occ_side_met', me['occ_side_met']), ('income_reserve_met', me['income_met'])):
            if bool(rs.get(k)) != val:
                mm.append({'what': f'reserve {k}', 'recorded': rs.get(k), 'mine': val})
        for k, val in (('min_whole_ratio', me['min_ratio']), ('min_occ_ratio', me['min_occ_side_ratio']),
                       ('min_income_ratio', me['min_income_ratio'])):
            if not math.isclose(float(rs.get(k)), val, rel_tol=TOL_RATIO, abs_tol=TOL_RATIO):
                mm.append({'what': f'reserve {k}', 'recorded': rs.get(k), 'mine': val})
        wb = str(rs.get('whole_binding', ''))
        wb = wb.split('|', 1)[1] if wb.startswith('s') and '|' in wb else wb
        if wb != me['binding']:
            mm.append({'what': 'reserve binding', 'recorded': rs.get('whole_binding'), 'mine': me['binding']})
        if rs.get('occ_binding') != me['occ_binding']:
            mm.append({'what': 'reserve occupation binding', 'recorded': rs.get('occ_binding'), 'mine': me['occ_binding']})
    ab = recv.get('absolute')
    da = max(abs(float(ab[t][m]) - mine['absolute'][t][m]) for t in mine['absolute'] for m in mine['absolute'][t]) if ab else None
    if da is None or da > TOL:
        mm.append({'what': 'absolute metrics', 'max_abs_diff': da})
    dvr = recv.get('distinct_vectors')
    if dvr is None or {t: dvr[t] for t in V.TASKS} != mine['distinct']:
        mm.append({'what': 'distinct vectors', 'recorded': dvr, 'mine': mine['distinct']})
    wrec = recv.get('wire_sha256_eval') or recv.get('wire_sha256')
    if wrec is None or {t: wrec[t] for t in V.TASKS} != mine['wires']:
        mm.append({'what': 'wire hashes', 'recorded': wrec, 'mine': mine['wires']})
    cost_diff = 0.0
    for spec, mc in (costs_mine or {}).items():
        rc = recv['paired_costs'][spec]['costs']
        if set(rc) != set(mc):
            mm.append({'what': f'paired cost keys {spec}', 'recorded': sorted(rc), 'mine': sorted(mc)})
        for k, m in mc.items():
            r = rc.get(k)
            if r is None:
                continue
            cost_diff = max(cost_diff, abs(float(r['value']) - m['value']), abs(float(r['se']) - m['se']),
                            abs(float(r.get('se_mc', 0.0)) - m['se_mc']))
    if cost_diff > TOL:
        mm.append({'what': 'paired costs', 'max_abs_diff': cost_diff})
    return {'mismatches': mm, 'max_abs_value_diff': dv, 'max_abs_se_diff': dse, 'max_abs_cost_diff': cost_diff,
            'max_abs_absolute_diff': da, 'point_pass_12_plus_2': point_all, 'failed': my_failed, 'max_value_over_cap': worst,
            'reserve_met': me['met'], 'min_whole_ratio': me['min_ratio'], 'binding': me['binding'],
            'occ_side_met': me['occ_side_met'], 'min_occ_ratio': me['min_occ_side_ratio']}


def reserve_rules_mine(mine_by_v):
    per = {str(v): m['reserve'] for v, m in mine_by_v.items()}
    rv = sorted(r['ratio'] for v in per for r in per[v]['conditions'].values())
    return {'qualified': all(per[v]['met'] for v in per) and sorted(per) == ['0', '1'],
            'min_ratio_by_version': {v: per[v]['min_ratio'] for v in per}, 'ratio_vector': rv}


def compare_reserve_rules(mine, recd):
    mm = []
    if recd is None:
        return [{'what': 'reserve_rules missing'}]
    if bool(recd.get('qualified')) != mine['qualified']:
        mm.append({'what': 'reserve_rules qualified', 'recorded': recd.get('qualified'), 'mine': mine['qualified']})
    for v, x in mine['min_ratio_by_version'].items():
        r = (recd.get('min_ratio_by_version') or {}).get(v)
        if r is None or not math.isclose(float(r), x, rel_tol=TOL_RATIO, abs_tol=TOL_RATIO):
            mm.append({'what': f'reserve_rules min ratio V{v}', 'recorded': r, 'mine': x})
    vec = recd.get('ratio_vector')
    if vec is None or len(vec) != len(mine['ratio_vector']) or any(not math.isclose(float(a), b, rel_tol=TOL_RATIO, abs_tol=TOL_RATIO)
                                                                   for a, b in zip(vec, mine['ratio_vector'])):
        mm.append({'what': 'reserve_rules ratio vector'})
    return mm


def eval_files():
    out = []
    for p in sorted((V.RUNS / 'FA').glob('*/EVAL_MS*.json')):
        out.append(('frc', p))
    for p in sorted((V.RUNS / 'SOURCE' / 'FA').glob('*/EVAL_MS*.json')):
        out.append(('source', p))
    return out


def run_eval_file(S, kind, p, floor, tag, G):
    doc = json.loads(p.read_text())
    fam, cfg = doc['family'], doc['config_id']
    res, record, proc, mine_by_v = {}, {}, [], {}
    for vs, recv in doc['versions'].items():
        if recv.get('missing'):
            res[vs] = {'missing': True}
            continue
        v = int(vs)
        refs, rprov = references(v, S['row_id'])
        if not all(rprov.values()):
            proc.append(f'V{v}: reference wires differ from the admission record')
        d = (V.SRC_RUNS if kind == 'source' else V.RUNS) / fam / cfg / f'V{v}'
        rel, rrec, h = release_arrays(d)
        if h != recv.get('release_sha256'):
            proc.append(f'V{v}: evaluated release hash differs from the saved release')
        if rrec.get('randomized'):
            proc.append(f'V{v}: randomized release (not covered by this deterministic replay)')
        mine = compute(S, rel, refs, floor[vs], G)
        mine_by_v[v] = mine
        costs, absent = {}, {}
        for spec, pc in recv.get('paired_costs', {}).items():
            if isinstance(pc, dict) and pc.get('missing'):
                # the evaluator recorded the comparator as absent for this version: it must really be absent
                exists = comparator_exists(spec, v)
                absent[spec] = {'recorded_missing': True, 'comparator_release_exists': exists}
                if exists:
                    proc.append(f'V{v}: comparator {spec} recorded missing but its release exists')
                continue
            other = comparator(spec, v, S['row_id'])
            _, own_o = V.conditions_version({t: S[t] for t in V.TASKS}, other, refs, S['sex'], S['unit'])
            costs[spec] = paired(S, mine['own'], own_o)
        res[vs] = compare_version(mine, recv, v, costs)
        if absent:
            res[vs]['absent_comparators'] = absent
        record[vs] = {'conditions': {n: {'value': c['value'], 'se': c['se']} for n, c in mine['conditions'].items()},
                      'paired_costs': costs, 'absent_comparators': absent, 'release_sha256': h,
                      **{k: x for k, x in res[vs].items() if k != 'mismatches'}, 'mismatches': res[vs]['mismatches']}
    rr_mm = []
    if len(mine_by_v) == 2:
        rrm = reserve_rules_mine(mine_by_v)
        rr_mm = compare_reserve_rules(rrm, doc.get('reserve_rules'))
        record['reserve_rules_mine'] = rrm
    ok = all(not r.get('mismatches') for r in res.values() if not r.get('missing')) and not rr_mm
    chk = {'id': f'V4.UTILITY.{"SOURCE." if kind == "source" else ""}{fam}.{cfg}.{p.stem}__{tag}',
           'what': f'own recomputation of the 12 clauses + 2 guards, SEs, point status, reserve, absolute metrics, distinct vectors, '
                   f'wires, paired costs and reserve_rules of {"source " if kind == "source" else ""}{fam}/{cfg} vs {p.name}',
           'inputs': {V.rel(p): V.file_sha(p), 'releases': {vs: record[vs]['release_sha256'] for vs in record if vs in ('0', '1')}},
           'result': {**res, 'reserve_rules_mismatches': rr_mm}, 'numerical_status': 'PASS' if ok else 'FAIL',
           'procedural_status': 'PASS' if not proc else 'FAIL', 'procedural_findings': proc}
    return chk, record, mine_by_v


def run_references(S, floor, tag, G):
    p = V.RUNS / 'REFERENCES' / 'EVAL_MS.json'
    if not p.is_file():
        return []
    doc = json.loads(p.read_text())
    checks = []
    for name, per in doc['references'].items():
        res = {}
        for vs, recv in per.items():
            v = int(vs)
            refs, _ = references(v, S['row_id'])
            rel = comparator(name, v, S['row_id'])
            mine = compute(S, rel, refs, floor[vs], G)
            res[vs] = compare_version(mine, {**recv, 'paired_costs': {}}, v, {})
        ok = all(not r['mismatches'] for r in res.values())
        checks.append({'id': f'V4.UTILITY.REF.{name.replace(":", "_").replace("+", "_")}__{tag}',
                       'what': f'own recomputation of the 12 clauses + 2 guards, SEs, point status, reserve, absolute metrics, distinct '
                               f'vectors and wires of reference {name} vs runs/REFERENCES/EVAL_MS.json',
                       'inputs': {V.rel(p): V.file_sha(p)}, 'result': res, 'numerical_status': 'PASS' if ok else 'FAIL',
                       'procedural_status': 'NOT_APPLICABLE'})
    return checks


# ---------------------------------------------------------------- the decision layer (my own numbers)
def qual_label(conds_by_v):
    per = {str(v): V.qual_version(c) for v, c in conds_by_v.items()}
    for x in per.values():
        x['conditions'] = x['rows']                     # the recorded format's key (frc.rules.qual) for the same rows
    ok = sorted(per) == ['0', '1'] and all(x['q1'] and x['q2'] for x in per.values())
    reasons = [f'Q1_FAILED_V{v}' for v in sorted(per) if not per[v]['q1']] + [f'Q2_FAILED_V{v}' for v in sorted(per) if not per[v]['q2']]
    return {'label': 'QUALIFIED' if ok else 'NOT_QUALIFIED', 'reasons': reasons, 'per_version': per}


def compare_qual(mine, rec):
    mm = []
    if mine['label'] != rec.get('label'):
        mm.append({'what': 'label', 'mine': mine['label'], 'recorded': rec.get('label')})
    if sorted(mine['reasons']) != sorted(rec.get('reasons', [])):
        mm.append({'what': 'reasons', 'mine': mine['reasons'], 'recorded': rec.get('reasons')})
    worst = 0.0
    for v, m in mine['per_version'].items():
        r = (rec.get('per_version') or {}).get(v) or (rec.get('per_version') or {}).get(int(v)) or {}
        for k in ('q1', 'q2', 'binding'):
            if m[k] != r.get(k):
                mm.append({'what': f'V{v} {k}', 'mine': m[k], 'recorded': r.get(k)})
        if sorted(m['q2_failed']) != sorted(r.get('q2_failed', [])):
            mm.append({'what': f'V{v} q2_failed', 'mine': m['q2_failed'], 'recorded': r.get('q2_failed')})
        if sorted(m['q1_failed']) != sorted(r.get('q1_failed', [])):
            mm.append({'what': f'V{v} q1_failed', 'mine': m['q1_failed'], 'recorded': r.get('q1_failed')})
        for n, row in m['rows'].items():
            rr = (r.get('conditions') or {}).get(n) or {}
            for k in ('value', 'slack_after_increment', 'slack_limit', 'increment'):
                if rr.get(k) is None:
                    mm.append({'what': f'V{v} {n} {k} missing'})
                    continue
                worst = max(worst, abs(float(rr[k]) - row[k]))
            if rr.get('pass') is not None and bool(rr['pass']) != row['pass']:
                mm.append({'what': f'V{v} {n} pass', 'mine': row['pass'], 'recorded': rr['pass']})
    if worst > TOL:
        mm.append({'what': 'QUAL condition numbers', 'max_abs_diff': worst})
    return mm, worst


def decision_layer(qdec, my_conds, my_ms_occ, nested_rec):
    """QUAL of A and B1 per capacity, Q-PE and the construction choice, the capacity walk and D1, all from my numbers."""
    out, checks_mm = {'per_capacity': {}}, []
    walk = []
    for cid in ('C64', 'C128', 'C256'):
        if cid not in (qdec.get('per_capacity') or {}):
            continue
        rq = qdec['per_capacity'][cid]
        mine = {'qual': {}}
        for cons in ('A', 'B1'):
            key = (cid, cons)
            if key in my_conds and len(my_conds[key]) == 2:
                q = qual_label(my_conds[key])
                mm, worst = compare_qual(q, rq['qual'].get(cons, {}))
                mine['qual'][cons] = {**q, 'mismatches': mm, 'max_abs_diff': worst}
                checks_mm += [f'{cid} {cons}: {x}' for x in mm]
        nq = (nested_rec or {}).get('per', {}).get(f'{cid}_QPE')
        ms = my_ms_occ.get(cid)
        if nq is not None:
            d = {v: tuple(x) for v, x in nq['d_and_se_used'].items()}
            if any(x[0] >= 0 for x in d.values()):
                lab = 'PE_ERROR_NOT_SUPPORTED'
            elif all(x[0] + 2 * x[1] < 0 for x in d.values()):
                lab = ('INCOMPLETE' if ms is None or len(ms) < 2 else
                       'PE_ERROR_SUPPORTED' if all(ms[v] < 0 for v in ms) else 'PE_ERROR_POINT_ONLY')
            else:
                lab = 'PE_ERROR_POINT_ONLY'
            mine['qpe'] = {'label': lab, 'model_select_occ_nll_B1_minus_A_mine': ms, 'nested_from': 'records/V3_NESTED (my numbers)'}
            mine['choice'] = nq['construction_choice']
            if lab != (rq.get('qpe') or {}).get('label'):
                checks_mm.append(f'{cid}: Q-PE mine {lab} recorded {(rq.get("qpe") or {}).get("label")}')
            if mine['choice'] != (rq.get('construction_choice') or {}).get('choice'):
                checks_mm.append(f'{cid}: choice mine {mine["choice"]} recorded {(rq.get("construction_choice") or {}).get("choice")}')
            rms = (rq.get('qpe') or {}).get('model_select_occ_nll') or {}
            for v, x in (ms or {}).items():
                r = rms.get(str(v), rms.get(v))
                if r is None or abs(float(r) - x) > TOL:
                    checks_mm.append(f'{cid}: MODEL_SELECT occ NLL B1 - A V{v} mine {x} recorded {r}')
        out['per_capacity'][cid] = mine
        if cid in ('C64', 'C128'):
            ch = mine.get('choice')
            ql = (mine['qual'].get(ch) or {}).get('label') if ch else None
            other = 'A' if ch == 'B1' else 'B1'
            walk.append({'capacity': CAP_KEY[cid], 'choice': ch, 'qual': ql, 'qpe': (mine.get('qpe') or {}).get('label'),
                         'other_would_qualify': ((mine['qual'].get(other) or {}).get('label') == 'QUALIFIED'
                                                 if other in mine['qual'] else None)})
    state = None
    for st in walk:
        if st['choice'] is None or st['qual'] is None:
            state = {'state': 'INCOMPLETE', 'awaiting': st['capacity']}
            break
        if st['qual'] == 'QUALIFIED':
            state = {'state': 'PROTECT', 'capacity': st['capacity'], 'construction': st['choice'], 'qpe': st['qpe']}
            break
    if state is None:
        state = {'state': 'TRIGGER_T5'}
    if state['state'] == 'PROTECT':
        d1 = ('QUALIFIED_WITHOUT_PROBABILITY_REPAIR' if state['construction'] == 'A' else
              {'PE_ERROR_SUPPORTED': 'REPAIRED_TASK_ONLY_RELEASE', 'PE_ERROR_POINT_ONLY': 'QUALIFIED_B1_REPAIR_NOT_RESOLVED'}
              .get(state['qpe'], 'INCOMPLETE'))
    else:
        d1 = 'INCOMPLETE'
    out['walk'], out['capacity_decision'], out['d1'] = walk, state, d1
    rc = qdec.get('capacity_decision') or {}
    if state.get('state') != rc.get('state') or state.get('capacity') != rc.get('capacity') or \
            state.get('construction') != rc.get('construction'):
        checks_mm.append(f'capacity decision mine {state} recorded {{state: {rc.get("state")}, capacity: {rc.get("capacity")}, '
                         f'construction: {rc.get("construction")}}}')
    if d1 != (qdec.get('d1') or {}).get('label'):
        checks_mm.append(f'D1 mine {d1} recorded {(qdec.get("d1") or {}).get("label")}')
    for st, rs in zip(walk, rc.get('steps', [])):
        for k in ('choice', 'qual', 'qpe', 'other_would_qualify'):
            if st[k] != rs.get(k):
                checks_mm.append(f'step {st["capacity"]} {k} mine {st[k]} recorded {rs.get(k)}')
        if rs.get('state') == 'ASSESSED' and st['qual'] == 'QUALIFIED':
            break
    exp_outcome = 'QUALIFIED' if state['state'] == 'PROTECT' else state['state']
    if exp_outcome != qdec.get('outcome'):
        checks_mm.append(f'outcome mine {exp_outcome} recorded {qdec.get("outcome")}')
    if state['state'] == 'PROTECT':
        inv = {v: k for k, v in CAP_KEY.items()}
        cid = inv[state['capacity']]
        twin = {'source': 'frc', 'config': f'TWB_{cid}'} if state['construction'] == 'B1' else \
            ({'source': 'nrm', 'config': V.SOURCE_TWIN[cid]} if cid in V.SOURCE_TWIN else {'source': 'frc', 'config': f'TWA_{cid}'})
        if (qdec.get('capacity_id'), qdec.get('construction'), qdec.get('twin')) != (cid, state['construction'], twin):
            checks_mm.append(f'protection record mine {(cid, state["construction"], twin)} recorded '
                             f'{(qdec.get("capacity_id"), qdec.get("construction"), qdec.get("twin"))}')
    return out, checks_mm


def run(tag, nested_tag):
    floor = floor_se()
    S = load_labels('verify: independent recomputation of the MODEL_SELECT clauses, guards, SEs, reserve, paired costs and the '
                    'QUAL / Q2 / D1 decisions of the evaluated FA releases (V4, separate code)')
    G = len(np.unique(S['unit']))
    checks, record = [], {'tag': tag, 'utc': V.utc(), 'files': {}}
    my_conds, own_cache = {}, {}
    for kind, p in eval_files():
        chk, rec, mine_by_v = run_eval_file(S, kind, p, floor, tag, G)
        checks.append(chk)
        record['files'][V.rel(p)] = rec
        doc = json.loads(p.read_text())
        cfg = doc['config_id']
        if p.name == 'EVAL_MS.json':
            if kind == 'source':
                cid = {v: k for k, v in V.SOURCE_TWIN.items()}.get(cfg)
                if cid:
                    my_conds[(cid, 'A')] = {v: m['conditions'] for v, m in mine_by_v.items()}
                    own_cache[(cid, 'A')] = {v: m['own'] for v, m in mine_by_v.items()}
            elif cfg.startswith('TWB_'):
                cid = cfg[4:]
                my_conds[(cid, 'B1')] = {v: m['conditions'] for v, m in mine_by_v.items()}
                own_cache[(cid, 'B1')] = {v: m['own'] for v, m in mine_by_v.items()}
            elif cfg.startswith('TWA_'):
                cid = cfg[4:]
                my_conds[(cid, 'A')] = {v: m['conditions'] for v, m in mine_by_v.items()}
                own_cache[(cid, 'A')] = {v: m['own'] for v, m in mine_by_v.items()}
    checks += run_references(S, floor, tag, G)
    my_ms_occ = {}
    for cid in ('C64', 'C128', 'C256'):
        if (cid, 'A') in own_cache and (cid, 'B1') in own_cache:
            my_ms_occ[cid] = {v: paired(S, own_cache[(cid, 'B1')][v], own_cache[(cid, 'A')][v])['occupation|nll']['value']
                              for v in own_cache[(cid, 'B1')] if v in own_cache[(cid, 'A')]}
    qp = V.RESULTS / 'decisions' / 'QUAL_DECISION.json'
    if qp.is_file():
        qdec = json.loads(qp.read_text())
        np_ = V.RECORDS / f'V3_NESTED__{nested_tag}.json'
        nested = json.loads(np_.read_text()) if np_.is_file() else None
        dl, mm = decision_layer(qdec, my_conds, my_ms_occ, nested)
        record['decision_layer'] = dl
        checks.append({'id': f'V4.QUAL_DECISION__{tag}',
                       'what': 'QUAL (Q1 G1 and Q2 S2 slack per version) of A and B1 at each capacity, Q-PE (my nested d / SE_used and '
                               'my MODEL_SELECT paired occupation NLL B1 - A), construction choice, capacity walk, D1 and the '
                               'protection record, all from my own numbers, vs decisions/QUAL_DECISION.json',
                       'inputs': {**V.inputs_sha([qp]), **({V.rel(np_): V.file_sha(np_)} if nested else {})},
                       'result': {'mine': dl, 'mismatches': mm, 'nested_record_present': nested is not None},
                       'numerical_status': 'PASS' if (not mm and nested is not None) else ('NOT_RUN' if nested is None else 'FAIL'),
                       'procedural_status': 'NOT_APPLICABLE'})
    V.write_record(f'V4_UTILITY__{tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['procedural_status'],
              (c['result'].get('mismatches') if isinstance(c['result'].get('mismatches'), list) else ''))
    return 0


# ---------------------------------------------------------------- invented-data selftest
def selftest():
    rng = np.random.default_rng(11)
    n = 800
    S = {'income': rng.integers(0, 2, n), 'occupation': rng.integers(0, 5, n), 'sex': rng.integers(0, 2, n),
         'unit': (np.arange(n) // 2).astype(np.int64)}
    refs = {r: {'income': rng.dirichlet(np.ones(2), n), 'occupation': rng.dirichlet(np.ones(5), n)} for r in V.REFS}
    floor_v = {c: 0.002 for c in V.CONDITIONS}
    rel = {'income': rng.dirichlet(np.ones(2) * 3, n), 'occupation': rng.dirichlet(np.ones(5) * 3, n)}
    G = len(np.unique(S['unit']))
    mine = compute(S, rel, refs, floor_v, G)

    def recorded(m, costs=None, perturb=None):
        cl = []
        for c in V.CLAUSES:
            t, r, mt = c.split('|')
            x = m['conditions'][c]
            cl.append({'seed': 0, 'task': t, 'reference': r, 'metric': mt, 'value': x['value'], 'cap': x['cap'], 'se': x['se'],
                       'point_pass': x['point_pass']})
        gd = [{'seed': 0, 'group': g.split('|')[1], 'value': m['conditions'][g]['value'], 'cap': V.GUARD_CAP,
               'se': m['conditions'][g]['se'], 'point_pass': m['conditions'][g]['point_pass']} for g in V.GUARDS]
        rs = m['reserve']
        failed = [c for c in V.CONDITIONS if not m['conditions'][c]['point_pass']]
        out = {'utility': {'clauses': cl, 'guards': gd},
               'point': {'point_pass_12_plus_2': not failed, 'failed': failed,
                         'max_value_over_cap': max(m['conditions'][c]['value'] / m['conditions'][c]['cap'] for c in V.CONDITIONS)},
               'reserve_without_mc': {'whole_release_met': rs['met'], 'min_whole_ratio': rs['min_ratio'],
                                      'whole_binding': f's0|{rs["binding"]}', 'occ_side_met': rs['occ_side_met'],
                                      'min_occ_ratio': rs['min_occ_side_ratio'], 'occ_binding': rs['occ_binding'],
                                      'income_reserve_met': rs['income_met'], 'min_income_ratio': rs['min_income_ratio']},
               'absolute': json.loads(json.dumps(m['absolute'])), 'distinct_vectors': dict(m['distinct']),
               'wire_sha256_eval': dict(m['wires']), 'paired_costs': {k: {'costs': json.loads(json.dumps(c))} for k, c in (costs or {}).items()}}
        if perturb:
            perturb(out)
        return out
    other = {'income': rng.dirichlet(np.ones(2), n), 'occupation': rng.dirichlet(np.ones(5), n)}
    _, own_o = V.conditions_version({t: S[t] for t in V.TASKS}, other, refs, S['sex'], S['unit'])
    costs = {'X': paired(S, mine['own'], own_o)}
    ok = {'self_consistent': compare_version(mine, recorded(mine, costs), 0, costs)['mismatches'] == []}
    ok['flags_value'] = bool(compare_version(mine, recorded(mine, costs, lambda o: o['utility']['clauses'][4].update(
        value=o['utility']['clauses'][4]['value'] + 1e-9)), 0, costs)['mismatches'])
    ok['flags_guard_se'] = bool(compare_version(mine, recorded(mine, costs, lambda o: o['utility']['guards'][1].update(
        se=o['utility']['guards'][1]['se'] * 1.001)), 0, costs)['mismatches'])
    ok['flags_reserve'] = bool(compare_version(mine, recorded(mine, costs, lambda o: o['reserve_without_mc'].update(
        min_whole_ratio=o['reserve_without_mc']['min_whole_ratio'] + 1e-6)), 0, costs)['mismatches'])
    ok['flags_cost'] = bool(compare_version(mine, recorded(mine, costs, lambda o: o['paired_costs']['X']['costs']['total|nll'].update(
        value=o['paired_costs']['X']['costs']['total|nll']['value'] + 1e-9)), 0, costs)['mismatches'])
    ok['flags_wire'] = bool(compare_version(mine, recorded(mine, costs, lambda o: o['wire_sha256_eval'].update(income='0')), 0,
                                            costs)['mismatches'])
    z = paired(S, mine['own'], mine['own'])
    ok['paired_identical_zero'] = all(r['value'] == 0.0 and r['se'] == 0.0 for r in z.values())
    # clause value is the difference of means; accuracy loss = acc(ref) - acc(own)
    r0 = V.person_losses(S['income'], refs['U0']['income'], 2)
    o0 = V.person_losses(S['income'], rel['income'], 2)
    ok['accuracy_loss_sign'] = abs(mine['conditions']['income|U0|accuracy_loss']['value'] -
                                   ((1 - r0['err']).mean() - (1 - o0['err']).mean())) < 1e-15
    # QUAL: Q2 at the S2 increments; the published A (16, 64) version-1 guard arithmetic fails, B1-like value passes
    c = {nm: {'value': -0.001, 'cap': (V.GUARD_CAP if nm.startswith('guard') else V.CAPS[nm.split('|')[2]]), 'se': 0.001}
         for nm in V.CONDITIONS}
    c1 = json.loads(json.dumps(c))
    c1['guard|sex0']['value'] = 0.00255
    q = qual_label({0: c, 1: c1})
    ok['qual_fails_published_A_guard'] = q['label'] == 'NOT_QUALIFIED' and q['reasons'] == ['Q2_FAILED_V1'] and \
        q['per_version']['1']['q2_failed'] == ['guard|sex0']
    c2 = json.loads(json.dumps(c))
    c2['guard|sex0']['value'] = 0.002                    # exactly cap - d: inclusive pass
    ok['qual_inclusive_limit'] = qual_label({0: c, 1: c2})['label'] == 'QUALIFIED'
    c3 = json.loads(json.dumps(c))
    c3['occupation|U0|nll_excess']['value'] = 0.0105     # point failure
    ql = qual_label({0: c, 1: c3})
    ok['qual_q1_failure'] = ql['label'] == 'NOT_QUALIFIED' and 'Q1_FAILED_V1' in ql['reasons']
    # decision layer on an invented decision record
    nested = {'per': {'C64_QPE': {'d_and_se_used': {'0': [-0.004, 0.001], '1': [-0.005, 0.001]}, 'construction_choice': 'B1'},
                      'C128_QPE': {'d_and_se_used': {'0': [-0.008, 0.001], '1': [-0.009, 0.001]}, 'construction_choice': 'B1'}}}
    myc = {('C64', 'B1'): {0: c, 1: c}, ('C64', 'A'): {0: c, 1: c1}, ('C128', 'B1'): {0: c, 1: c}, ('C128', 'A'): {0: c, 1: c}}
    qrec = {k: qual_label(v) for k, v in myc.items()}
    qdec = {'outcome': 'QUALIFIED', 'capacity_id': 'C64', 'construction': 'B1', 'twin': {'source': 'frc', 'config': 'TWB_C64'},
            'd1': {'label': 'REPAIRED_TASK_ONLY_RELEASE'},
            'capacity_decision': {'state': 'PROTECT', 'capacity': '16x64', 'construction': 'B1',
                                  'steps': [{'capacity': '16x64', 'choice': 'B1', 'qual': 'QUALIFIED', 'qpe': 'PE_ERROR_SUPPORTED',
                                             'other_would_qualify': False, 'state': 'ASSESSED'}]},
            'per_capacity': {cid: {'qual': {cons: qrec[(cid, cons)] for cons in ('A', 'B1')},
                                   'qpe': {'label': 'PE_ERROR_SUPPORTED', 'model_select_occ_nll': {'0': -0.003, '1': -0.004}},
                                   'construction_choice': {'choice': 'B1'}} for cid in ('C64', 'C128')}}
    ms = {'C64': {0: -0.003, 1: -0.004}, 'C128': {0: -0.003, 1: -0.004}}
    dl, mm = decision_layer(qdec, myc, ms, nested)
    ok['decision_layer_consistent'] = mm == [] and dl['d1'] == 'REPAIRED_TASK_ONLY_RELEASE'
    ms2 = {'C64': {0: -0.003, 1: 0.001}, 'C128': {0: -0.003, 1: -0.004}}
    dl2, mm2 = decision_layer(qdec, myc, ms2, nested)
    ok['decision_layer_flags_ms_sign'] = dl2['d1'] == 'QUALIFIED_B1_REPAIR_NOT_RESOLVED' and bool(mm2)
    ok['all_ok'] = all(bool(x) for x in ok.values())
    return {k: bool(x) for k, x in ok.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--nested-tag', default='V1')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag, a.nested_tag)


if __name__ == '__main__':
    sys.exit(main())
