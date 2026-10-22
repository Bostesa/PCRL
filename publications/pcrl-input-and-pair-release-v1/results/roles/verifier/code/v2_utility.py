"""Verifier check 2 (MODEL_SELECT labels, read once through ipr.common.load, operation 'verify', inside a queue unit).

The 12 clauses + 2 guards per version (24 + 4) of every evaluated configuration of this campaign (CTL_TWIN, the IA and IB family
configurations, the shared CTL points through their role members resolved per version from paths/SHARED_ROLES.json), with own metric
code (accuracy loss = error(release) - error(reference), first argmax; NLL excess with the 1e-12 floor; summed Brier excess; references
U0_S<v> / Ucal_S<v> from the frozen bank (ohn.common.frozen_bank, MODEL_SELECT role only, hash-checked by its loader); men's and women's
occupation summed-Brier excess vs U0; household-clustered SEs, linearized subgroup SEs), compared with the saved EVAL_MS.json records
(tolerance 1e-9 on point values; SEs compared too). Every paired cost recorded in an EVAL_MS.json against a comparator of this campaign
(W skipped) is recomputed (11 metrics). The decisive paired costs of each reading (CAND vs TWIN and vs every gating path point) are
written for check 4. IA_SBC_LO (saved in version 0 only, not evaluated by the coordinator, not in the bank) is evaluated here
descriptively.

    python3 -m ipr.jobqueue run --unit IPR_VER_UTILITY_V1 --stage verify --rss-gib 2 -- python3 <this file>
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))

import numpy as np  # noqa: E402

import vlib as V  # noqa: E402

TOL = 1e-9
FAMILY_CONFIGS = (('CTL', 'CTL_TWIN'), ('IA', 'IA_SFG'), ('IA', 'IA_SBC_HI'), ('IA', 'IA_BLIND'), ('IB', 'IB_CSF'), ('IB', 'IB_CSFT'),
                  ('IB', 'IB_IND'))
DECISIVE = {'IA_SFG': ('CTL_TWIN', 'IA_SBC_HI', 'IA_SPD_HI', 'IA_RO_HI', 'IA_BLIND'),
            'IB_CSF': ('CTL_TWIN', 'IB_SPD_LO', 'IB_SPD_HI', 'IB_CSFT', 'IB_IND')}
METRIC_KEYS = (('acc_loss', 'err'), ('nll', 'nll'), ('brier', 'brier'))


def load_ms(fam, cfg, v):
    d = V.RUNS / fam / cfg / f'V{int(v)}'
    rec = json.loads((d / 'RELEASE.json').read_text())
    got = V.file_sha(d / 'RELEASE.npz')
    if got != rec['release_sha256']:
        raise RuntimeError(f'{fam}/{cfg}/V{v}: RELEASE.npz SHA-256 differs from RELEASE.json')
    with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
        return {t: z[f'MODEL_SELECT__{t}'].copy() for t in V.TASKS}, rec


def per_person(y, rel):
    return {t: V.losses(y[t], rel[t], V.K[t]) for t in V.TASKS}


def utility(own, ref_l, sex, unit):
    clauses = []
    for task in V.TASKS:
        for r in V.REFS:
            for m in V.METRICS:
                key = {'accuracy_loss': 'err', 'nll_excess': 'nll', 'brier_excess': 'brier'}[m]
                d = own[task][key] - ref_l[r][task][key]
                clauses.append({'task': task, 'reference': r, 'metric': m, 'value': float(d.mean()), 'se': V.clustered_se(d, unit),
                                'cap': V.CAPS[m]})
    d = own['occupation']['brier'] - ref_l['U0']['occupation']['brier']
    guards = []
    for g, name in ((0, 'sex0'), (1, 'sex1')):
        m_, se_ = V.subgroup_mean_se(d, sex == g, unit)
        guards.append({'group': name, 'value': m_, 'se': se_, 'cap': V.GUARD_CAP})
    for c in clauses + guards:
        c['point_pass'] = bool(c['value'] <= c['cap'])
    return clauses, guards


def costs(a, b, sex, unit):
    out = {}
    for task in V.TASKS:
        for m, key in METRIC_KEYS:
            d = a[task][key] - b[task][key]
            out[f'{task}|{m}'] = {'value': float(d.mean()), 'se': V.clustered_se(d, unit)}
            if m == 'brier':
                for g, name in ((0, 'men_brier'), (1, 'women_brier')):
                    mv, ms_ = V.subgroup_mean_se(d, sex == g, unit)
                    out[f'{task}|{name}'] = {'value': mv, 'se': ms_}
    tot = (a['income']['nll'] + a['occupation']['nll']) - (b['income']['nll'] + b['occupation']['nll'])
    out['total|nll'] = {'value': float(tot.mean()), 'se': V.clustered_se(tot, unit)}
    return out


def compare_eval(entry_clauses, entry_guards, ve):
    cmp_, worst_v, worst_s = [], 0.0, 0.0
    for c in entry_clauses:
        hit = [x for x in ve['utility']['clauses'] if x['task'] == c['task'] and x['reference'] == c['reference']
               and x['metric'] == c['metric']]
        if len(hit) != 1:
            cmp_.append({'clause': f"{c['task']}|{c['reference']}|{c['metric']}", 'error': f'{len(hit)} matches'})
            continue
        dv, ds = abs(hit[0]['value'] - c['value']), abs(hit[0]['se'] - c['se'])
        worst_v, worst_s = max(worst_v, dv), max(worst_s, ds)
        cmp_.append({'clause': f"{c['task']}|{c['reference']}|{c['metric']}", 'dv': dv, 'dse': ds,
                     'pass_agrees': bool(hit[0]['point_pass']) == c['point_pass']})
    for gd in entry_guards:
        hit = [x for x in ve['utility']['guards'] if x['group'] == gd['group']]
        dv, ds = abs(hit[0]['value'] - gd['value']), abs(hit[0]['se'] - gd['se'])
        worst_v, worst_s = max(worst_v, dv), max(worst_s, ds)
        cmp_.append({'clause': f"guard|{gd['group']}", 'dv': dv, 'dse': ds, 'pass_agrees': bool(hit[0]['point_pass']) == gd['point_pass']})
    ok = all(x.get('dv', 1) <= TOL and x.get('pass_agrees') for x in cmp_) and len(cmp_) == 14
    return cmp_, worst_v, worst_s, ok


def main():
    t0 = time.process_time()
    from ipr import common as C
    S = C.load('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'), 'verify',
               'independent verifier replay of the 24 clauses + 4 guards and the paired costs of every saved configuration of IA, IB '
               'and CTL (own metric code)')
    y = {'income': np.asarray(S['income']), 'occupation': np.asarray(S['occupation'])}
    sex = np.asarray(S['sex']).astype(int)
    unit = np.asarray(S['unit'])
    from ohn import common as OC                          # frozen U0 / Ucal release vectors (label-free, MODEL_SELECT role only)
    refs = {}
    for v in (0, 1):
        bank, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('frozen bank MODEL_SELECT row order differs')
        refs[v] = {r: per_person(y, {t: bank[r]['MODEL_SELECT'][t] for t in V.TASKS}) for r in V.REFS}
    roles = json.loads((V.RESULTS / 'paths' / 'SHARED_ROLES.json').read_text())['roles']
    members = {cfg: {v: (fam, cfg) for v in (0, 1)} for fam, cfg in FAMILY_CONFIGS}
    for name, per in roles.items():
        members[name] = {int(v): tuple(x.split('/')) for v, x in per.items()}
    out = {'unit_id': V.unit_id(), 'utc_start': V.utc(), 'people': int(len(unit)), 'households': int(len(np.unique(unit))),
           'n_men': int((sex == 0).sum()), 'n_women': int((sex == 1).sum()), 'members': {}, 'paired': [], 'errors': [],
           'decisive_costs': {}}
    own = {}
    worst = {'value': 0.0, 'se': 0.0}
    for name, per in members.items():
        ev_path = (V.RUNS / 'ROLES' / name / 'EVAL_MS.json') if name in roles else (V.RUNS / per[0][0] / name / 'EVAL_MS.json')
        ev = json.loads(ev_path.read_text()) if ev_path.is_file() else None
        ent = {'eval_file': str(ev_path.relative_to(V.ROOT)), 'eval_exists': ev is not None, 'versions': {}}
        for v in (0, 1):
            try:
                fam, cfg = per[v]
                rel, rec = load_ms(fam, cfg, v)
                o = per_person(y, rel)
                own[(name, v)] = o
                cl, gu = utility(o, refs[v], sex, unit)
                e = {'config': f'{fam}/{cfg}/V{v}', 'release_sha256': rec['release_sha256'],
                     'g1_version_pass': all(c['point_pass'] for c in cl + gu),
                     'max_value_over_cap': max(c['value'] / c['cap'] for c in cl + gu),
                     'clauses': {f"{c['task']}|{c['reference']}|{c['metric']}": c['value'] for c in cl},
                     'guards': {g['group']: g['value'] for g in gu}}
                if ev is not None:
                    ve = ev['versions'][str(v)]
                    e['release_sha_matches_eval'] = ve.get('release_sha256') == rec['release_sha256']
                    cmp_, wv, ws, ok = compare_eval(cl, gu, ve)
                    worst['value'] = max(worst['value'], wv); worst['se'] = max(worst['se'], ws)
                    e['max_abs_value_diff'] = wv; e['max_abs_se_diff'] = ws; e['all_within_tol'] = ok
                    e['point_pass_agrees'] = bool(ve['point']['point_pass_12_plus_2']) == e['g1_version_pass']
                ent['versions'][str(v)] = e
            except Exception as ex:
                ent['versions'][str(v)] = {'error': f'{type(ex).__name__}: {ex}'}
                out['errors'].append({'member': name, 'version': v, 'traceback': traceback.format_exc()[-1500:]})
        ent['g1_both_versions'] = all(ent['versions'].get(str(v), {}).get('g1_version_pass') for v in (0, 1))
        out['members'][name] = ent
    # IA_SBC_LO: saved in version 0 only (descriptive)
    try:
        rel, rec = load_ms('IA', 'IA_SBC_LO', 0)
        o = per_person(y, rel)
        own[('IA_SBC_LO', 0)] = o
        cl, gu = utility(o, refs[0], sex, unit)
        out['ia_sbc_lo_v0'] = {'g1_version_pass': all(c['point_pass'] for c in cl + gu),
                               'max_value_over_cap': max(c['value'] / c['cap'] for c in cl + gu),
                               'eval_exists': (V.RUNS / 'IA' / 'IA_SBC_LO' / 'EVAL_MS.json').is_file(),
                               'cost_IA_SFG_minus_IA_SBC_LO_v0': costs(own[('IA_SFG', 0)], o, sex, unit),
                               'cost_IA_SBC_LO_minus_TWIN_v0': costs(o, own[('CTL_TWIN', 0)], sex, unit)}
    except Exception as ex:
        out['errors'].append({'member': 'IA_SBC_LO', 'traceback': traceback.format_exc()[-1500:]})
    # paired costs recorded in the EVAL_MS records (comparators of this campaign; W skipped)
    worst_pc = {'value': 0.0, 'se': 0.0}
    for name, per in members.items():
        ev_path = (V.RUNS / 'ROLES' / name / 'EVAL_MS.json') if name in roles else (V.RUNS / per[0][0] / name / 'EVAL_MS.json')
        if not ev_path.is_file():
            continue
        ev = json.loads(ev_path.read_text())
        for v in (0, 1):
            for comp, ent in ev['versions'][str(v)].get('paired_costs', {}).items():
                if comp == 'W':
                    continue
                cname = comp[5:] if comp.startswith('ROLE:') else comp.split('/')[1]
                if (cname, v) not in own:
                    out['paired'].append({'release': name, 'version': v, 'comparator': comp, 'error': 'comparator not recomputed'})
                    continue
                mine = costs(own[(name, v)], own[(cname, v)], sex, unit)
                rec_c = ent['costs']
                diffs = {k: (abs(rec_c[k]['value'] - mine[k]['value']), abs(rec_c[k]['se'] - mine[k]['se'])) for k in mine if k in rec_c}
                missing = sorted(set(mine) - set(rec_c))
                wv = max(d[0] for d in diffs.values()); ws = max(d[1] for d in diffs.values())
                worst_pc['value'] = max(worst_pc['value'], wv); worst_pc['se'] = max(worst_pc['se'], ws)
                out['paired'].append({'release': name, 'version': v, 'comparator': comp, 'max_abs_value_diff': wv, 'max_abs_se_diff': ws,
                                      'missing': missing, 'within_tol': wv <= TOL and not missing})
    # decisive costs for check 4 (own values)
    for cand, comps in DECISIVE.items():
        out['decisive_costs'][cand] = {c: {str(v): costs(own[(cand, v)], own[(c, v)], sex, unit) for v in (0, 1)} for c in comps}
    out['worst_clause'] = worst
    out['worst_paired'] = worst_pc
    out['all_clauses_within_tol'] = all(e.get('all_within_tol') for m in out['members'].values() for e in m['versions'].values())
    out['all_paired_within_tol'] = all(p.get('within_tol') for p in out['paired'])
    out['cpu_s'] = round(time.process_time() - t0, 2)
    out['utc_end'] = V.utc()
    p = V.write_record(f'V2_UTILITY__{V.unit_id() or "NOQUEUE"}.json', out)
    summ = {'members': {k: {'g1': m['g1_both_versions'], 'eval': m['eval_exists'],
                            'within_tol': [m['versions'].get(str(v), {}).get('all_within_tol') for v in (0, 1)],
                            'max_over_cap': [round(m['versions'].get(str(v), {}).get('max_value_over_cap', float('nan')), 4) for v in (0, 1)]}
                        for k, m in out['members'].items()},
            'worst_clause': worst, 'worst_paired': worst_pc, 'n_paired': len(out['paired']),
            'all_clauses_within_tol': out['all_clauses_within_tol'], 'all_paired_within_tol': out['all_paired_within_tol'],
            'ia_sbc_lo_v0_g1': out.get('ia_sbc_lo_v0', {}).get('g1_version_pass'), 'errors': len(out['errors']), 'cpu_s': out['cpu_s'],
            'record': str(p)}
    print(json.dumps(summ, default=V._default))
    return 0 if not out['errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
