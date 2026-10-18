"""VER check 3 (and the per-seed part of check 4): the 12 utility clauses and 2 within-SEX guards of complete releases on
MODEL_SELECT, recomputed with the verifier's own metric and clustered-SE code from the stored arrays, compared with the
campaign's evaluator outputs; the paired comparisons with T_TREE / TASK_REFIT / JOINT, the per-SEX occupation excess, and the
cross-fitting diagnostics (deployed minus fold-average on MODEL_SELECT; out-of-fold MODEL_FIT occupation utility against the
in-sample U0 / T_TREE); the per-seed reserve (k = 2, H_LJ floor) against the stored per-seed reserve.

Usage: v_utility.py refs            (the 18 frozen references, both seeds, against runs/REF/S<s>/EVAL_MS_REFERENCES.json)
       v_utility.py configs [--only F/C ...]   (every scored family configuration not yet verified; EVAL_MS_H3 / _TW / _COMPLETE)
MODEL_SELECT labels and SEX are read through ohn.common.load_role with purposes starting "independent verification:"; MODEL_FIT
occupation labels (no SEX) are read the same way for the cross-fitted diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402

from ohn import common as C  # noqa: E402
from ohn import devrun as D  # noqa: E402  (stored-array loaders only)

RULES = V.load_rules()


def ms_data(seed, what):
    S = C.load_role('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'),
                    purpose=V.PURPOSE + f'utility clauses, guards and reserve of {what}, seed {seed} (VER check 3/4)')
    bank, ids = C.frozen_bank(seed, roles=('MODEL_SELECT',))
    if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']) or ('unit__MODEL_SELECT' in ids and not np.array_equal(ids['unit__MODEL_SELECT'], S['unit'])):
        raise AssertionError('frozen bank row order differs from MODEL_SELECT')
    refs = {r: {t: bank[r]['MODEL_SELECT'][t] for t in V.TASKS} for r in ('U0', 'Ucal')}
    others = {'T_TREE': bank['T_TREE']['MODEL_SELECT']['occupation'], 'TASK_REFIT': bank['TASK_REFIT']['MODEL_SELECT']['occupation'],
              'JOINT': bank['H_LJ']['MODEL_SELECT']['occupation']}
    return S, refs, others


def compare_eval(mine, ev, seed):
    """Max absolute differences between my 14 conditions and the evaluator's clauses / guards; per-seed reserve comparison."""
    diffs = {}
    for c in ev['utility']['clauses']:
        key = f"{c['task']}|{c['reference']}|{c['metric']}"
        m = mine[key]
        diffs[key] = {'value': abs(m['value'] - c['value']), 'se': abs(m['se'] - c['se']), 'point_pass_equal': (m['value'] <= m['cap']) == c['point_pass']}
    for g in ev['utility']['guards']:
        key = f"guard|{g['group']}"
        m = mine[key]
        diffs[key] = {'value': abs(m['value'] - g['value']), 'se': abs(m['se'] - g['se']), 'point_pass_equal': (m['value'] <= m['cap']) == g['point_pass']}
    res = {'n_conditions': len(diffs), 'max_abs_value_diff': max(d['value'] for d in diffs.values()),
           'max_abs_se_diff': max(d['se'] for d in diffs.values()), 'point_pass_all_equal': all(d['point_pass_equal'] for d in diffs.values())}
    my_res = V.reserve_seed(mine, seed, RULES, ev['utility']['households'])
    stored = ev.get('reserve', {}).get('per_seed', {}).get(str(seed), {}).get('conditions', {})
    if stored:
        res['reserve_max_abs_ratio_diff'] = max(abs(my_res[k]['ratio'] - stored[k]['ratio']) for k in my_res)
        res['reserve_met_all_equal'] = all(my_res[k]['met'] == stored[k]['met'] for k in my_res)
        res['reserve_se_source_all_equal'] = all(my_res[k]['se_source'] == stored[k]['se_source'] for k in my_res)
        res['reserve_floor_max_abs_diff'] = max(abs(my_res[k]['floor_se'] - stored[k]['floor_se']) for k in my_res)
    res['my_min_ratio'] = min(v['ratio'] for v in my_res.values())
    res['my_binding'] = min(my_res, key=lambda k: my_res[k]['ratio'])
    res['my_seed_met'] = all(v['met'] for v in my_res.values())
    res['stored_min_ratio'] = ev.get('reserve', {}).get('min_ratio')
    res['stored_binding'] = ev.get('reserve', {}).get('binding_condition')
    res['stored_meets_reserve_this_seed'] = ev.get('reserve', {}).get('meets_reserve')
    return res, my_res


def side_checks(S, occ, others, ev):
    """Paired occupation comparisons (candidate minus reference) and per-SEX excesses against the evaluator's numbers."""
    y = S['occupation']; unit = S['unit']
    own = V.losses(y, occ, 5)
    md = 0.0
    for name, q in others.items():
        r = V.losses(y, q, 5)
        for m, d in (('accuracy_loss', r['acc'] - own['acc']), ('nll_diff', own['nll'] - r['nll']), ('brier_diff', own['brier'] - r['brier'])):
            e = ev['occupation_versus_task_only_references'][f'{name}|{m}']
            md = max(md, abs(float(d.mean()) - e['value']), abs(V.clustered_se(d, unit) - e['se']))
    return md


def per_sex_check(S, occ, refs, ev):
    y = S['occupation']; unit = S['unit']; sex = S['sex']
    own = V.losses(y, occ, 5)
    md = 0.0
    for r in ('U0', 'Ucal'):
        rr = V.losses(y, refs[r]['occupation'], 5)
        for g, name in ((0, 'men'), (1, 'women')):
            for m, d in (('accuracy_loss', rr['acc'] - own['acc']), ('nll_excess', own['nll'] - rr['nll']), ('brier_excess', own['brier'] - rr['brier'])):
                v, se = V.subgroup_mean_se(d, sex == g, unit)
                e = ev['occupation_per_sex'][f'{name}|{r}|{m}']
                md = max(md, abs(v - e['value']), abs(se - e['se']))
    return md


def crossfit_check(seed, S, occ_ms, cf, ev, F):
    if cf is None or 'crossfit' not in ev or ev['crossfit'] is None:
        return None
    e = ev['crossfit']
    out = {}
    fa = cf['MODEL_SELECT__foldavg']
    y = S['occupation']; unit = S['unit']
    a, b = V.losses(y, occ_ms, 5), V.losses(y, fa, 5)
    md = 0.0
    for m, d in (('accuracy_loss', b['acc'] - a['acc']), ('nll_diff', a['nll'] - b['nll']), ('brier_diff', a['brier'] - b['brier'])):
        ee = e['deployed_minus_foldavg_on_MODEL_SELECT'][m]
        md = max(md, abs(float(d.mean()) - ee['value']), abs(V.clustered_se(d, unit) - ee['se']))
    md = max(md, abs(float(np.abs(occ_ms - fa).max()) - e['deployed_minus_foldavg_on_MODEL_SELECT']['max_abs_prob_diff']),
             abs(float(np.mean(occ_ms.argmax(1) == fa.argmax(1))) - e['deployed_minus_foldavg_on_MODEL_SELECT']['decision_agreement']))
    out['deployed_minus_foldavg_max_abs_diff'] = md
    out['deployed_minus_foldavg_nll'] = float((a['nll'] - b['nll']).mean())
    out['deployed_minus_foldavg_brier'] = float((a['brier'] - b['brier']).mean())
    if F is not None:
        oof = cf['MODEL_FIT__oof']
        yf = F['occupation']; uf = F['unit']
        sa = C.stage_a_outputs(seed)
        if not np.array_equal(sa['row_id__MODEL_FIT'], F['row_id']):
            raise AssertionError('stage A MODEL_FIT row order')
        own = V.losses(yf, oof, 5)
        md2 = max(abs(float(own[k].mean()) - e['oof_absolute'][m]) for k, m in (('acc', 'accuracy'), ('nll', 'nll'), ('brier', 'brier')))
        for name, key in (('U0_in_sample', 'U0__MODEL_FIT__occupation'), ('T_TREE_in_sample', 'T_TREE__MODEL_FIT__occupation')):
            r = V.losses(yf, sa[key], 5)
            for m, d in (('accuracy_loss', r['acc'] - own['acc']), ('nll_excess', own['nll'] - r['nll']), ('brier_excess', own['brier'] - r['brier'])):
                ee = e[f'oof_vs_{name}'][m]
                md2 = max(md2, abs(float(d.mean()) - ee['value']), abs(V.clustered_se(d, uf) - ee['se']))
        out['oof_max_abs_diff'] = md2
        out['fold_vector_equals_registered'] = bool(np.array_equal(cf['fold'], V.household_parts(uf, 5, 'OHN1_crossfit')))
    return out


def run_refs():
    rec = {'check': 'VER_UTILITY_REFS', 'utc': V.utc(), 'unit_id': V.unit_id(), 'seeds': {}}
    tables = {}
    for seed in (0, 1):
        S, refs, others = ms_data(seed, 'the 18 frozen references')
        ev_all = json.loads((V.RUNS / 'REF' / f'S{seed}' / 'EVAL_MS_REFERENCES.json').read_text())
        res = {}
        for name in D.REFERENCE_SET:
            rel = D.reference_release(name, seed, 'MODEL_SELECT')
            outs = {t: rel[t] for t in V.TASKS}
            mine = V.clause_table({t: S[t] for t in V.TASKS}, outs, refs, S['sex'], S['unit'])
            cmp_, my_res = compare_eval(mine, ev_all[name], seed)
            cmp_['wire_equal'] = {t: V.wire_hash(outs[t]) == ev_all[name]['wire_sha256'][t] for t in V.TASKS}
            cmp_['contract'] = {t: V.check_prob(outs[t], V.NCLASS[t])['contract_ok'] for t in V.TASKS}
            cmp_['versus_max_abs_diff'] = side_checks(S, outs['occupation'], others, ev_all[name])
            cmp_['per_sex_max_abs_diff'] = per_sex_check(S, outs['occupation'], refs, ev_all[name])
            res[name] = cmp_
            tables.setdefault(name, {})[seed] = {k: {'value': v['value'], 'se': v['se'], 'cap': v['cap']} for k, v in mine.items()}
        rec['seeds'][seed] = res
    rec['max_abs_value_diff'] = max(r['max_abs_value_diff'] for s in rec['seeds'].values() for r in s.values())
    rec['max_abs_se_diff'] = max(r['max_abs_se_diff'] for s in rec['seeds'].values() for r in s.values())
    rec['max_reserve_ratio_diff'] = max(r.get('reserve_max_abs_ratio_diff', 0) for s in rec['seeds'].values() for r in s.values())
    rec['all_point_pass_equal'] = all(r['point_pass_all_equal'] for s in rec['seeds'].values() for r in s.values())
    rec['all_reserve_met_equal'] = all(r.get('reserve_met_all_equal', True) for s in rec['seeds'].values() for r in s.values())
    rec['all_wires_equal'] = all(all(r['wire_equal'].values()) for s in rec['seeds'].values() for r in s.values())
    rec['max_side_diff'] = max(max(r['versus_max_abs_diff'], r['per_sex_max_abs_diff']) for s in rec['seeds'].values() for r in s.values())
    rec['tables'] = tables
    tol = 1e-10
    rec['status'] = 'PASS' if (rec['max_abs_value_diff'] < tol and rec['max_abs_se_diff'] < tol and rec['all_point_pass_equal'] and rec['all_reserve_met_equal']
                               and rec['all_wires_equal'] and rec['max_side_diff'] < tol and rec['max_reserve_ratio_diff'] < 1e-7) else 'ATTENTION'
    V.write_record('VER_UTILITY_REFS', rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ('seeds', 'tables')}, indent=1))


def scored_configs():
    out = []
    for p in sorted(V.RUNS.glob('*/*/S*/EVAL_MS_*.json')):
        fam, cfg, sdir = p.parts[-4], p.parts[-3], p.parts[-2]
        if fam in ('REF', 'GRID'):
            continue
        out.append((fam, cfg, int(sdir[1:]), p.stem.replace('EVAL_MS_', ''), p))
    return out


def run_configs(only, tag):
    prev = V.RECORDS / 'VER_UTILITY_CONFIGS.json'
    db = json.loads(prev.read_text()) if prev.is_file() else {'entries': {}, 'units': []}
    todo = []
    for fam, cfg, seed, anchor, p in scored_configs():
        key = f'{fam}/{cfg}/S{seed}/{anchor}'
        if only and f'{fam}/{cfg}' not in only:
            continue
        sha = V.file_sha256(p)
        if db['entries'].get(key, {}).get('eval_sha256') == sha:
            continue
        todo.append((fam, cfg, seed, anchor, p, sha, key))
    by_seed = {}
    for t in todo:
        by_seed.setdefault(t[2], []).append(t)
    F_cache = None
    for seed, items in sorted(by_seed.items()):
        S, refs, others = ms_data(seed, 'scored family configurations ' + ', '.join(sorted({f'{t[0]}/{t[1]}' for t in items})))
        for fam, cfg, seed_, anchor, p, sha, key in items:
            ev = json.loads(p.read_text())
            if anchor == 'COMPLETE':
                rel, rrec = D.load_complete(fam, cfg, seed, roles=('MODEL_SELECT',))
                outs = {t: rel['MODEL_SELECT'][t] for t in V.TASKS}
                cf = None
            else:
                occ, orec, cf = D.load_occupation(fam, cfg, seed, roles=('MODEL_SELECT',))
                outs = {'income': D.anchor_income(anchor, seed, 'MODEL_SELECT'), 'occupation': occ['MODEL_SELECT']}
            mine = V.clause_table({t: S[t] for t in V.TASKS}, outs, refs, S['sex'], S['unit'])
            cmp_, my_res = compare_eval(mine, ev, seed)
            cmp_['wire_equal'] = {t: V.wire_hash(outs[t]) == ev['wire_sha256'][t] for t in V.TASKS}
            cmp_['contract'] = {t: V.check_prob(outs[t], V.NCLASS[t]) for t in V.TASKS}
            cmp_['versus_max_abs_diff'] = side_checks(S, outs['occupation'], others, ev)
            cmp_['per_sex_max_abs_diff'] = per_sex_check(S, outs['occupation'], refs, ev)
            if cf is not None and ev.get('crossfit'):
                if F_cache is None:
                    F_cache = C.load_role('MODEL_FIT', ('features', 'occupation'),
                                          purpose=V.PURPOSE + 'out-of-fold MODEL_FIT occupation utility of scored configurations (no SEX; VER check 3)')
                cmp_['crossfit'] = crossfit_check(seed, S, outs['occupation'], cf, ev, F_cache)
            cmp_['eval_sha256'] = sha
            cmp_['table'] = {k: {'value': v['value'], 'se': v['se'], 'cap': v['cap']} for k, v in mine.items()}
            cmp_['verified_utc'] = V.utc()
            cmp_['unit_id'] = V.unit_id()
            db['entries'][key] = cmp_
    db['units'].append({'unit_id': V.unit_id(), 'utc': V.utc(), 'n_new': len(todo), 'tag': tag})
    tol = 1e-10
    for key, e in db['entries'].items():
        cf = e.get('crossfit') or {}
        e['status'] = 'PASS' if (e['max_abs_value_diff'] < tol and e['max_abs_se_diff'] < tol and e['point_pass_all_equal']
                                 and e.get('reserve_met_all_equal', True) and e.get('reserve_max_abs_ratio_diff', 0) < 1e-7
                                 and all(e['wire_equal'].values()) and e['versus_max_abs_diff'] < tol and e['per_sex_max_abs_diff'] < tol
                                 and cf.get('deployed_minus_foldavg_max_abs_diff', 0) < tol and cf.get('oof_max_abs_diff', 0) < tol
                                 and cf.get('fold_vector_equals_registered', True)
                                 and all(c['contract_ok'] for c in e['contract'].values())) else 'ATTENTION'
    db['n_entries'] = len(db['entries'])
    db['n_pass'] = sum(e['status'] == 'PASS' for e in db['entries'].values())
    V.write_record('VER_UTILITY_CONFIGS', db)
    print(json.dumps({'new': [t[6] for t in todo], 'n_entries': db['n_entries'], 'n_pass': db['n_pass'],
                      'attention': [k for k, e in db['entries'].items() if e['status'] != 'PASS']}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('refs', 'configs'))
    ap.add_argument('--only', nargs='*', default=[])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    if a.what == 'refs':
        run_refs()
    else:
        run_configs(set(a.only), a.tag)


if __name__ == '__main__':
    main()
