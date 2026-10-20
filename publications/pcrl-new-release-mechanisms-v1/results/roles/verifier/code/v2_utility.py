"""V2 task and subgroup metrics: own recomputation of the 12 clauses + 2 guards per version (accuracy loss, NLL excess, summed
Brier excess vs U0_S<v> and Ucal_S<v>; men's / women's occupation summed-Brier excess vs U0; household-clustered SEs), the
whole-release and occupation-side reserve, absolute metrics, distinct-vector counts and paired per-person costs, for the frozen
references (runs/REFERENCES/EVAL_MS.json) and for named releases (runs/<F>/<C>/EVAL_MS[__tag].json), compared with the recorded
values. Randomized releases: per-draw metrics averaged over draws (se = mean per-draw SE, se_mc = SD over draws / sqrt(R)).

Labels: MODEL_SELECT income, occupation, SEX through nrm.common.load(operation 'verify') called from THIS script (logged).

    python3 .../v2_utility.py --tag V1 --references
    python3 .../v2_utility.py --tag V2 --release FAMILY/CONFIG[@EVALTAG] [--release ...]
    python3 .../v2_utility.py --selftest        (invented data; comparison logic must flag planted mismatches)
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
import vref  # noqa: E402

TOL_VALUE = 1e-12
TOL_SE = 1e-12
TOL_RATIO = 1e-9


def load_labels(purpose):
    sys.path.insert(0, str(V.ROOT))
    import nrm.common as C
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify', purpose)
    return {'income': np.asarray(S['income']), 'occupation': np.asarray(S['occupation']), 'sex': np.asarray(S['sex']),
            'unit': np.asarray(S['unit']), 'row_id': np.asarray(S['row_id'])}


def floor_se():
    r = json.loads((V.ROOT / 'results' / 'pcrl_occupation_headroom_overnight_v1' / 'roles' / 'statistical_reviewer' / 'RULES.json').read_text())
    fl = r['reserve']['floor']
    if int(fl['households']) != V.FLOOR_HOUSEHOLDS or float(r['reserve']['multiplier_k']) != V.K_RESERVE:
        raise ValueError('reserve floor constants differ')
    return fl['se']


# ---------------------------------------------------------------- own computation for one release and version
def compute(S, rel, refs, v, floor):
    labels = {t: S[t] for t in V.TASKS}
    conds = V.conditions_table(labels, rel, refs, S['sex'], S['unit'])
    G = len(np.unique(S['unit']))
    res = V.reserve_version(conds, floor[str(v)], G)
    res_mc = V.reserve_version(conds, floor[str(v)], G, with_mc=True) if any('se_mc' in c for c in conds.values()) else None
    pp = V.per_person_avg(labels, rel)
    absolute = {t: {m: float(pp[t][m].mean()) for m in ('acc_loss', 'nll', 'brier')} for t in V.TASKS}
    absolute['occupation']['men_brier'] = float(pp['occupation']['brier'][S['sex'] == 0].mean())
    absolute['occupation']['women_brier'] = float(pp['occupation']['brier'][S['sex'] == 1].mean())
    ds = V.draws_of(rel)
    distinct = {t: (V.distinct_rows(ds[0][t]) if len(ds) == 1 else [V.distinct_rows(d[t]) for d in ds]) for t in V.TASKS}
    wires = {t: (V.wire_sha(rel[t]) if np.asarray(rel[t]).ndim == 2 else [V.wire_sha(x) for x in rel[t]]) for t in V.TASKS}
    return {'conditions': conds, 'reserve': res, 'reserve_with_mc': res_mc, 'absolute': absolute, 'distinct': distinct, 'pp': pp,
            'draws': len(ds), 'wires': wires}


def paired(S, pa, pb, mc_rule='paired_draws'):
    """loss(a) - loss(b) per person, household-clustered SEs. MC SE, two registered definitions (by evaluator version):
    'paired_draws' (before 705966524): SD over matched per-draw mean differences / sqrt(R) (one-draw side broadcast), reported only
    if a side is randomized; 'quadrature' (705966524, ENG1-06): sqrt(mc_a^2 + mc_b^2), each release's own SD over per-draw means /
    sqrt(R) (0 when deterministic), also for the men's / women's Brier and total NLL."""
    out = {}
    unit, sex = S['unit'], S['sex']

    def own_mc(p, t, m, mask=None):
        arr = p[t]['draw_arrays'][m]
        return V.mc_se([x[mask].mean() if mask is not None else x.mean() for x in arr])

    def quad(t, m, mask=None):
        return math.sqrt(own_mc(pa, t, m, mask) ** 2 + own_mc(pb, t, m, mask) ** 2)

    def total_mc(p):
        ti, to = list(p['income']['draw_means']['nll']), list(p['occupation']['draw_means']['nll'])
        k = max(len(ti), len(to))
        ti, to = (ti * k if len(ti) == 1 else ti), (to * k if len(to) == 1 else to)
        return V.mc_se(np.add(ti[:k], to[:k]))
    for t in V.TASKS:
        for m in ('acc_loss', 'nll', 'brier'):
            d = pa[t][m] - pb[t][m]
            rec = {'value': float(d.mean()), 'se': V.cluster_se(d, unit)}
            if mc_rule == 'quadrature':
                rec['se_mc'] = quad(t, m)
            else:
                da, db = pa[t]['draw_means'][m], pb[t]['draw_means'][m]
                if len(da) > 1 or len(db) > 1:
                    if len(da) == len(db):
                        dm = da - db
                    elif len(db) == 1:
                        dm = da - db[0]
                    elif len(da) == 1:
                        dm = da[0] - db
                    else:
                        dm = None
                    rec['se_mc'] = (float(np.std(dm, ddof=1) / math.sqrt(len(dm))) if dm is not None and len(dm) > 1 else None)
            out[f'{t}|{m}'] = rec
            if m == 'brier':
                for g, name in ((0, 'men_brier'), (1, 'women_brier')):
                    mask = sex == g
                    r2 = {'value': float(d[mask].mean()), 'se': V.subgroup_se(d, mask, unit)}
                    if mc_rule == 'quadrature':
                        r2['se_mc'] = quad(t, 'brier', mask)
                    out[f'{t}|{name}'] = r2
    tot = (pa['income']['nll'] + pa['occupation']['nll']) - (pb['income']['nll'] + pb['occupation']['nll'])
    out['total|nll'] = {'value': float(tot.mean()), 'se': V.cluster_se(tot, unit)}
    if mc_rule == 'quadrature':
        out['total|nll']['se_mc'] = math.sqrt(total_mc(pa) ** 2 + total_mc(pb) ** 2)
    return out


# ---------------------------------------------------------------- comparisons with the recorded evaluation
def compare(mine, recv, v, specs_costs=None):
    mm = []
    u = recv['utility']
    rec = {f"{c['task']}|{c['reference']}|{c['metric']}": c for c in u['clauses']}
    rec.update({f"guard|{g['group']}": g for g in u['guards']})
    if set(rec) != set(V.CONDITIONS) or len(u['clauses']) != 12 or len(u['guards']) != 2:
        mm.append({'what': 'condition set', 'recorded': sorted(rec)})
    dv = dse = dmc = 0.0
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
        if (float(r['value']) <= float(r['cap'])) != m['point_pass'] or ('point_pass' in r and bool(r['point_pass']) != m['point_pass']):
            mm.append({'what': f'{n} point status', 'recorded': r.get('point_pass'), 'mine': m['point_pass']})
        if m['draws'] > 1:
            if r.get('se_mc') is None:
                mm.append({'what': f'{n} se_mc missing'})
            else:
                dmc = max(dmc, abs(float(r['se_mc']) - m['se_mc']))
            if r.get('draw_values') is not None and np.abs(np.asarray(r['draw_values']) - np.asarray(m['draw_values'])).max() > TOL_VALUE:
                mm.append({'what': f'{n} draw values'})
    if dv > TOL_VALUE:
        mm.append({'what': 'clause / guard values', 'max_abs_diff': dv})
    if dse > TOL_SE:
        mm.append({'what': 'clause / guard SEs', 'max_abs_diff': dse})
    if dmc > TOL_SE:
        mm.append({'what': 'MC SEs', 'max_abs_diff': dmc})
    point_all = all(mine['conditions'][n]['point_pass'] for n in V.CONDITIONS)
    pt = recv.get('point', {})
    if pt and bool(pt.get('point_pass_12_plus_2')) != point_all:
        mm.append({'what': 'point_pass_12_plus_2', 'recorded': pt.get('point_pass_12_plus_2'), 'mine': point_all})
    my_failed = sorted(n for n in V.CONDITIONS if not mine['conditions'][n]['point_pass'])
    rec_failed = sorted(pt.get('failed', [])) if pt else None
    if pt and rec_failed != my_failed:
        mm.append({'what': 'failed clauses', 'recorded': rec_failed, 'mine': my_failed})
    rs, me = recv.get('reserve') or recv.get('reserve_without_mc') or {}, mine['reserve']
    if not rs:
        mm.append({'what': 'no per-version reserve block in the record'})
    if rs:
        pairs = (('whole_release_met', me['met']), ('occ_side_met', me['occ_side_met']), ('income_reserve_met', me['income_met']))
        for k, val in pairs:
            if bool(rs.get(k)) != val:
                mm.append({'what': f'reserve {k}', 'recorded': rs.get(k), 'mine': val})
        for k, val in (('min_whole_ratio', me['min_ratio']), ('min_occ_ratio', me['min_occ_side_ratio']),
                       ('min_income_ratio', me['min_income_ratio'])):
            if not math.isclose(float(rs.get(k)), val, rel_tol=TOL_RATIO, abs_tol=TOL_RATIO):
                mm.append({'what': f'reserve {k}', 'recorded': rs.get(k), 'mine': val})
        wb = str(rs.get('whole_binding', '')).split('|', 1)[-1] if str(rs.get('whole_binding', '')).startswith('s') else rs.get('whole_binding')
        if wb != me['binding']:
            mm.append({'what': 'reserve binding', 'recorded': rs.get('whole_binding'), 'mine': me['binding']})
        if rs.get('occ_binding') != me['occ_binding']:
            mm.append({'what': 'reserve occupation binding', 'recorded': rs.get('occ_binding'), 'mine': me['occ_binding']})
    ab = recv.get('absolute')
    if ab:
        da = max(abs(float(ab[t][m]) - mine['absolute'][t][m]) for t in mine['absolute'] for m in mine['absolute'][t])
        if da > TOL_VALUE:
            mm.append({'what': 'absolute metrics', 'max_abs_diff': da})
    dvr = recv.get('distinct_vectors')
    if dvr is not None and {t: dvr[t] for t in V.TASKS} != mine['distinct']:
        mm.append({'what': 'distinct vectors', 'recorded': dvr, 'mine': mine['distinct']})
    cost_diff = 0.0
    for spec, mine_costs in (specs_costs or {}).items():
        rc = recv['paired_costs'][spec]['costs']
        if set(rc) != set(mine_costs):
            mm.append({'what': f'paired costs keys {spec}'})
        for k, m in mine_costs.items():
            r = rc.get(k)
            if r is None:
                continue
            cost_diff = max(cost_diff, abs(float(r['value']) - m['value']), abs(float(r['se']) - m['se']))
            if m.get('se_mc') is not None and r.get('se_mc') is not None:
                cost_diff = max(cost_diff, abs(float(r['se_mc']) - m['se_mc']))
            elif (m.get('se_mc') is None) != (r.get('se_mc') is None):
                mm.append({'what': f'paired cost {spec} {k}: MC SE present on one side only', 'mine': m.get('se_mc'),
                           'recorded': r.get('se_mc')})
    wrec = recv.get('wire_sha256_eval') or recv.get('wire_sha256')
    if wrec is not None and mine.get('wires') is not None and {t: wrec[t] for t in V.TASKS} != mine['wires']:
        mm.append({'what': 'recorded evaluation wire hashes differ from the release arrays', 'recorded': wrec, 'mine': mine['wires']})
    if cost_diff > TOL_VALUE:
        mm.append({'what': 'paired costs', 'max_abs_diff': cost_diff})
    return {'mismatches': mm, 'max_abs_value_diff': dv, 'max_abs_se_diff': dse, 'max_abs_se_mc_diff': dmc,
            'max_abs_cost_diff': cost_diff, 'point_pass_12_plus_2': point_all, 'failed': my_failed,
            'reserve_met': me['met'], 'min_whole_ratio': me['min_ratio'], 'binding': me['binding'],
            'occ_side_met': me['occ_side_met'], 'min_occ_ratio': me['min_occ_side_ratio'],
            'reserve_with_mc_met': (mine['reserve_with_mc'] or {}).get('met'),
            'reserve_with_mc_min_ratio': (mine['reserve_with_mc'] or {}).get('min_ratio')}


def summary_values(mine):
    return {n: {'value': c['value'], 'se': c['se'], **({'se_mc': c['se_mc']} if 'se_mc' in c else {})} for n, c in mine['conditions'].items()}


# ---------------------------------------------------------------- targets
def run_references(S, tag, floor, fname='EVAL_MS.json'):
    p = V.RUNS / 'REFERENCES' / fname
    doc = json.loads(p.read_text())
    checks, record = [], {'file': V.rel(p), 'file_sha256': V.file_sha(p), 'references': {}}
    for name, per in doc['references'].items():
        res, provs, bad_prov = {}, {}, []
        for vs, recv in per.items():
            v = int(vs)
            refs, _ = vref.frozen_bank_ms(v)
            rel, prov = vref.reference(name, v)
            provs[vs] = prov
            if prov['wire_equals_admission'] is False:
                bad_prov.append(vs)
            mine = compute(S, rel, refs, v, floor)
            res[vs] = compare(mine, recv, v)
            record['references'].setdefault(name, {})[vs] = {'conditions': summary_values(mine), **{k: x for k, x in res[vs].items() if k != 'mismatches'},
                                                            'mismatches': res[vs]['mismatches'], 'provenance': prov}
        ok = all(not r['mismatches'] for r in res.values())
        checks.append({'id': f'V2.UTILITY.REF.{name.replace(":", "_").replace("+", "_")}__{tag}',
                       'what': f'own recomputation of the 12 clauses + 2 guards, SEs, point status, reserve, absolute metrics, distinct '
                               f'vectors and recorded wires of reference {name} (both versions) vs runs/REFERENCES/{fname}',
                       'inputs': {V.rel(p): record['file_sha256'], 'wires': {vs: provs[vs]['wire_sha256'] for vs in provs},
                                  'frozen_bank_sha256': {vs: provs[vs]['bank_sha256'] for vs in provs}},
                       'result': {vs: {k: x for k, x in r.items()} for vs, r in res.items()},
                       'numerical_status': 'PASS' if ok else 'FAIL',
                       'procedural_status': 'PASS' if not bad_prov else 'FAIL'})
    return checks, record


def run_release(S, spec, tag, floor):
    fc, _, etag = spec.partition('@')
    fam, cfg = fc.split('/')
    p = V.RUNS / fam / cfg / (f'EVAL_MS__{etag}.json' if etag else 'EVAL_MS.json')
    doc = json.loads(p.read_text())
    res, record, proc = {}, {}, []
    wires, mine_by_v = {}, {}
    new_format = any('reserve_without_mc' in x for x in doc['versions'].values() if isinstance(x, dict))
    mc_rule = 'quadrature' if new_format else 'paired_draws'
    for vs, recv in doc['versions'].items():
        if recv.get('missing'):
            res[vs] = {'missing': True}
            continue
        v = int(vs)
        refs, _ = vref.frozen_bank_ms(v)
        arr, rrec, prov = vref.release(fam, cfg, v)
        if prov['release_sha256'] != recv['release_sha256']:
            proc.append(f'V{v}: evaluated release hash {recv["release_sha256"]} differs from the saved release {prov["release_sha256"]}')
        rel = arr['MODEL_SELECT']
        wires[vs] = {t: ([V.wire_sha(x) for x in rel[t]] if np.asarray(rel[t]).ndim == 3 else V.wire_sha(rel[t])) for t in V.TASKS}
        mine = compute(S, rel, refs, v, floor)
        mine_by_v[v] = mine
        costs = {}
        for cspec in recv.get('paired_costs', {}):
            other, _ = vref.comparator(cspec, v)
            costs[cspec] = paired(S, mine['pp'], V.per_person_avg({t: S[t] for t in V.TASKS}, other), mc_rule)
        res[vs] = compare(mine, recv, v, costs)
        record[vs] = {'conditions': summary_values(mine), **{k: x for k, x in res[vs].items()}, 'paired_costs': costs,
                      'release': prov, 'randomized': bool(rrec.get('randomized')), 'draws': mine['draws'], 'mc_rule': mc_rule}
    rr = doc.get('reserve_rules')
    if rr is not None and len(mine_by_v) == 2:          # the governing MC-aware reserve (nrm.rules.reserve_status) per file
        gm = {}
        for lab, key in (('with_mc', 'reserve_with_mc'), ('without_mc', 'reserve')):
            per = {v: (m[key] if m[key] is not None else m['reserve']) for v, m in mine_by_v.items()}
            rv = sorted(r_['ratio'] if r_['ratio'] is not None else -math.inf for v in per for r_ in per[v]['conditions'].values())
            gm[lab] = {'qualified': all(per[v]['met'] for v in per), 'min_by_version': {str(v): per[v]['min_ratio'] for v in per},
                       'ratio_vector': rv}
        rec_w, rec_wo = rr, rr.get('without_mc') or {}
        for lab, recd in (('with_mc', rec_w), ('without_mc', rec_wo)):
            if bool(recd.get('qualified')) != gm[lab]['qualified']:
                res.setdefault('file', {'mismatches': []})['mismatches'].append({'what': f'reserve_rules {lab} qualified', 'recorded': recd.get('qualified'), 'mine': gm[lab]['qualified']})
            mb = recd.get('min_ratio_by_version') or {}
            for v in ('0', '1'):
                if v in mb and not math.isclose(float(mb[v]), gm[lab]['min_by_version'][v], rel_tol=TOL_RATIO, abs_tol=TOL_RATIO):
                    res.setdefault('file', {'mismatches': []})['mismatches'].append({'what': f'reserve_rules {lab} min ratio V{v}', 'recorded': mb[v], 'mine': gm[lab]['min_by_version'][v]})
            vec = recd.get('ratio_vector')
            if vec is not None and (len(vec) != len(gm[lab]['ratio_vector']) or any(not math.isclose(float(x), y, rel_tol=TOL_RATIO, abs_tol=TOL_RATIO) for x, y in zip(vec, gm[lab]['ratio_vector']))):
                res.setdefault('file', {'mismatches': []})['mismatches'].append({'what': f'reserve_rules {lab} ratio vector'})
        record['reserve_rules_mine'] = gm
    ok = all(not r.get('mismatches') for r in res.values() if not r.get('missing'))
    chk = {'id': f'V2.UTILITY.{fam}.{cfg}{("." + etag) if etag else ""}__{tag}',
           'what': f'own recomputation of clauses, guards, SEs (and MC SEs), point status, reserve, absolute metrics, distinct vectors '
                   f'and paired costs of {fam}/{cfg} vs {p.name}',
           'inputs': {V.rel(p): V.file_sha(p), 'wire_sha256': wires,
                      'releases': {vs: record[vs]['release']['release_sha256'] for vs in record if vs in ('0', '1')}},
           'result': res, 'numerical_status': 'PASS' if ok else 'FAIL', 'procedural_status': 'PASS' if not proc else 'FAIL',
           'procedural_findings': proc}
    return [chk], {f'{fam}/{cfg}': record}


# ---------------------------------------------------------------- invented-data selftest of the comparison logic
def selftest():
    rng = np.random.default_rng(11)
    n = 600
    S = {'income': rng.integers(0, 2, n), 'occupation': rng.integers(0, 5, n), 'sex': rng.integers(0, 2, n), 'unit': np.arange(n) // 2}
    refs = {r: {'income': rng.dirichlet(np.ones(2), n), 'occupation': rng.dirichlet(np.ones(5), n)} for r in V.REFS}
    floor = {str(v): {c: 0.002 for c in V.CONDITIONS} for v in (0, 1)}
    rel = {'income': rng.dirichlet(np.ones(2) * 3, n), 'occupation': rng.dirichlet(np.ones(5) * 3, n)}
    mine = compute(S, rel, refs, 0, floor)

    def recorded(m, perturb=None):
        cl = []
        for c in V.CLAUSES:
            t, r, mt = c.split('|')
            x = m['conditions'][c]
            cl.append({'seed': 0, 'task': t, 'reference': r, 'metric': mt, 'value': x['value'], 'cap': x['cap'], 'se': x['se'],
                       'point_pass': x['point_pass']})
        gd = [{'seed': 0, 'group': g.split('|')[1], 'value': m['conditions'][g]['value'], 'cap': V.GUARD_CAP,
               'se': m['conditions'][g]['se'], 'point_pass': m['conditions'][g]['point_pass']} for g in V.GUARDS]
        rs = m['reserve']
        out = {'utility': {'clauses': cl, 'guards': gd},
               'point': {'point_pass_12_plus_2': all(m['conditions'][c]['point_pass'] for c in V.CONDITIONS),
                         'failed': [c for c in V.CONDITIONS if not m['conditions'][c]['point_pass']]},
               'reserve': {'whole_release_met': rs['met'], 'min_whole_ratio': rs['min_ratio'], 'whole_binding': f's0|{rs["binding"]}',
                           'occ_side_met': rs['occ_side_met'], 'min_occ_ratio': rs['min_occ_side_ratio'], 'occ_binding': rs['occ_binding'],
                           'income_reserve_met': rs['income_met'], 'min_income_ratio': rs['min_income_ratio']},
               'absolute': json.loads(json.dumps(m['absolute'])), 'distinct_vectors': json.loads(json.dumps(m['distinct']))}
        if perturb:
            perturb(out)
        return out
    ok = {'self_consistent': compare(mine, recorded(mine), 0)['mismatches'] == []}
    ok['flags_value'] = bool(compare(mine, recorded(mine, lambda o: o['utility']['clauses'][3].update(value=o['utility']['clauses'][3]['value'] + 1e-9)), 0)['mismatches'])
    ok['flags_se'] = bool(compare(mine, recorded(mine, lambda o: o['utility']['guards'][1].update(se=o['utility']['guards'][1]['se'] * 1.01)), 0)['mismatches'])
    ok['flags_reserve'] = bool(compare(mine, recorded(mine, lambda o: o['reserve'].update(min_whole_ratio=o['reserve']['min_whole_ratio'] + 1e-6)), 0)['mismatches'])
    ok['flags_missing_guard'] = bool(compare(mine, recorded(mine, lambda o: o['utility']['guards'].pop()), 0)['mismatches'])
    ok['flags_distinct'] = bool(compare(mine, recorded(mine, lambda o: o['distinct_vectors'].update(income=1)), 0)['mismatches'])
    # randomized: three draws, recorded draw-averaged values must match; a draw value change must be flagged
    st = {'income': np.stack([rel['income'], refs['U0']['income'], rel['income']]), 'occupation': rel['occupation']}
    mr = compute(S, st, refs, 0, floor)
    rr = recorded(mr)
    for c in rr['utility']['clauses'] + rr['utility']['guards']:
        name = f"{c['task']}|{c['reference']}|{c['metric']}" if 'task' in c else f"guard|{c['group']}"
        c['se_mc'] = mr['conditions'][name]['se_mc']; c['draw_values'] = mr['conditions'][name]['draw_values']
    ok['randomized_self_consistent'] = compare(mr, rr, 0)['mismatches'] == []
    rr['utility']['clauses'][0]['se_mc'] += 1e-6
    ok['randomized_flags_se_mc'] = bool(compare(mr, rr, 0)['mismatches'])
    # paired costs: identical releases -> zero
    z = paired(S, mine['pp'], mine['pp'])
    ok['paired_identical_zero'] = all(r['value'] == 0.0 and r['se'] == 0.0 for r in z.values())
    pa3 = V.per_person_avg({t: S[t] for t in V.TASKS}, {t: np.stack([rel[t]] * 3) for t in V.TASKS})
    zq = paired(S, pa3, mine['pp'], 'quadrature')
    ok['quadrature_constant_draws_zero_mc'] = all(abs(r['se_mc']) < 1e-15 for r in zq.values())
    pn = V.per_person_avg({t: S[t] for t in V.TASKS}, st)
    zn = paired(S, pn, mine['pp'], 'quadrature')
    mc_inc = V.mc_se(pn['income']['draw_means']['nll'])
    ok['quadrature_matches_own_mc'] = abs(zn['income|nll']['se_mc'] - mc_inc) < 1e-15 and zn['total|nll']['se_mc'] > 0 and \
        zn['occupation|nll']['se_mc'] < 1e-15
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag'); ap.add_argument('--references', action='store_true'); ap.add_argument('--release', action='append', default=[])
    ap.add_argument('--ref-file', default='EVAL_MS.json')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        chk = {'id': f'V2.UTILITY_SELFTEST__{a.tag or "V1"}', 'what': 'utility replay comparison logic on invented data (self-consistency, '
               'planted value / SE / reserve / guard / distinct-vector / MC-SE mismatches flagged, identical paired costs zero)',
               'inputs': V.inputs_sha([Path(__file__), V.CODE_DIR / 'vlib.py']), 'result': ok,
               'numerical_status': 'PASS' if ok['all_ok'] else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'}
        print(json.dumps(V.append_checks([chk]), indent=1)); print(json.dumps(ok, indent=1, default=V._jd))
        return 0 if ok['all_ok'] else 1
    floor = floor_se()
    what = (['references'] if a.references else []) + a.release
    S = load_labels(f'verify: independent recomputation of clauses, guards, SEs, reserve and paired costs for {what} (separate code)')
    checks, record = [], {'tag': a.tag, 'utc': V.utc(), 'targets': what}
    if a.references:
        c, r = run_references(S, a.tag, floor, a.ref_file)
        checks += c; record['references'] = r
    for spec in a.release:
        c, r = run_release(S, spec, a.tag, floor)
        checks += c; record.setdefault('releases', {}).update(r)
    V.write_record(f'V2_UTILITY__{a.tag}.json', record)
    print(json.dumps(V.append_checks(checks), indent=1))
    for c in checks:
        print(c['id'], c['numerical_status'], c['procedural_status'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
