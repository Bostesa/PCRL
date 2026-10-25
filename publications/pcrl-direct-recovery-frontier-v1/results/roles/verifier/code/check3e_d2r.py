"""Verifier check 3e (DRF1; label-free): D2R robustness of a registered reading, by my reading of RULES.md 8 and NRM1 RULES.md 5.1,
from the registered grid and the robustness grid (per-cell SCORE AUCs, recorded paired cell SEs and per-recipe score fingerprints)
and my own split recomputation; compared with the reading's D2R.json when it exists (never used to compute a label).

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check3e_d2r.py --plan PLAN_DRF_R1 --main DRF1_R1 \
        --rob DRF1_R1_ROB --check2 OUT2 --out FILE

My reading:
  - cells: registered = S<v>R1/R2 k0/k1; robustness = every other cell of the merged grids (R3, R4 at any reader seed; R1, R2 at
    k2, k3); all = registered + robustness. (The narrower reading "R3 / R4 cells only" is reported beside.)
  - per comparator: robustness confirmed iff the pair benefit is AGG_POSITIVE on the robustness cells alone AND on all cells; the
    recipient gates (income, occupation) are read on the robustness cells; ROBUSTNESS_CONFIRMED (decisive comparators) iff
    confirmed, both robustness recipient gates pass, the registered cells are present and the refit is genuinely effective;
  - a sign change = sign(x registered) != sign(x robustness); a failed robustness recipient gate is flagged; both are carried as
    headline caveats for any comparator (decisive or beside);
  - refit effectiveness: R3 and / or R4 run (one only: flag), their household assignment hashes distinct from each other and from
    R1 / R2, every SCORE household overlap involving them finite and in [0, 1), and grid validity on the reader-seed pairs (k0, k1)
    and (k2, k3) of every repetition (seed-invariant equalities explained as in check 3c).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))                      # (added 04:28Z: the first run stopped at `import drf` before any output)
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

CAMPAIGN = 'pcrl_direct_recovery_frontier_v1'
RES = REPO / 'results' / CAMPAIGN
PRIV = REPO / '.private' / CAMPAIGN
SEED_DEP = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
SUBJECT = {'H_LJ': 'REF:H_LJ', 'NEURAL_RAW': 'REF:NEURAL_RAW', 'RETIRED_B3CTL': 'REF:RETIRED_B3CTL', 'SUCC': 'REF:SUCC',
           'TWB_C64': 'TWB_C64', 'B2TWIN': 'REF:B2TWIN', 'TWJ': 'REF:TWJ', 'W': 'W', 'TR100': 'REF:H3+TASK_REFIT',
           'TT100': 'REF:H3+T_TREE'}


def load(bank):
    g = {'cells': {}, 'paired': {}, 'files': {}}
    for v in (0, 1):
        p = PRIV / 'runs' / 'GRID' / bank / f'S{v}' / 'GRID.json'
        x = json.loads(p.read_text())
        g['files'][v] = {'path': str(p.relative_to(REPO)), 'sha256': V.sha256_file(p), 'unit_id': x.get('unit_id')}
        g['cells'].update(x['cells'])
        for cand, comps in x['paired'].items():
            for comp, views in comps.items():
                for view, e in views.items():
                    d = g['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells'])
                    d['cell_se'].update(e['cell_se'])
    return g


def cell_values(cells, cand, comp, view, use):
    out = {}
    for c in use:
        a, b = cells[c][cand][view]['score_auc'], cells[c][comp][view]['score_auc']
        out[c] = (b - a) if view == 'pair' else (a - b)
    return out


def sign(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


def seed_validity(cells, members, minimum, distinct, reps, pairs):
    fp = {}
    for cell, cres in cells.items():
        v, r, k = V.parse_cell(cell)
        for m, ent in cres.items():
            if m not in members:
                continue
            for view, e in ent.items():
                for rec, h in e['extras']['score_sha256'].items():
                    fp[(v, r, k, m, view, rec)] = h
    recipes = sorted({key[5] for key in fp})
    views = ('income', 'occupation', 'pair')
    out = {'checks': 0, 'equalities': [], 'unexplained': []}
    for (k0, k1) in pairs:
        def differs(v, r, m, view, rec):
            a_, b_ = fp.get((v, r, k0, m, view, rec)), fp.get((v, r, k1, m, view, rec))
            return None if a_ is None or b_ is None else a_ != b_
        for v in (0, 1):
            for r in reps:
                for m in sorted(members):
                    for view in views:
                        for rec in recipes:
                            d = differs(v, r, m, view, rec)
                            if d is None:
                                continue
                            out['checks'] += 1
                            if rec not in SEED_DEP or d:
                                continue
                            n = distinct[(m, v)][view]
                            a_ = any(differs(v, r, m, view, o) for o in SEED_DEP if o != rec)
                            b_ = any(differs(v, r, m, w, rec) for w in views if w != view)
                            iii = a_ and b_
                            few = n <= 320 and (any(differs(v, r, m2, view, rec) for m2 in members if m2 != m) or b_)
                            ni, no = distinct[(m, v)]['income'], distinct[(m, v)]['occupation']
                            pext = (view == 'pair' and n > 320 and ni <= 320 and no <= 320 and n <= ni * no <= 1024 and
                                    any(differs(v, r, m2, 'pair', rec) for m2 in members if m2 != m))
                            if m in minimum:
                                et = rec in ('ET5', 'ET25') and iii and n <= (1600 if rec == 'ET5' else 320)
                                expl = 'ER_EXTRA_TREES_SUPPORT' if et else 'FEW_VALUED' if few else 'FINITE_PAIR' if pext else None
                            else:
                                expl = 'NRM1_III' if iii else 'FEW_VALUED' if few else 'FINITE_PAIR' if pext else None
                            item = {'v': v, 'R': f'R{r}', 'k_pair': [k0, k1], 'wire': m, 'view': view, 'recipe': rec,
                                    'minimum_set': m in minimum, 'distinct': n, 'explanation': expl}
                            out['equalities'].append(item)
                            if expl is None:
                                out['unexplained'].append(item)
    out['valid'] = not out['unexplained']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='PLAN_DRF_R1')
    ap.add_argument('--main', default='DRF1_R1')
    ap.add_argument('--rob', default='DRF1_R1_ROB')
    ap.add_argument('--check2', required=True)
    ap.add_argument('--d2r', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    items, info = [], {'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    plan = json.loads((RES / 'claims' / f'{a.plan}.json').read_text())
    c2 = json.loads(Path(a.check2).read_text())
    spec = json.loads((RES / 'grid_specs' / f'{a.rob}.json').read_text())
    dec = json.loads((RES / 'claims' / plan['plan_id'] / 'DECISIONS.json').read_text())['decisions']
    q = (((dec.get('D2') or {}).get('comparisons') or {}).get('CTRL_FRONTIER') or {}).get('frontier', {}).get('bracketing_control')
    cand, obj, twin = plan['cand'], plan['obj_mi']['config_id'], plan['twin']['id']
    decisive = {'CTRL': q, 'OBJ': obj}
    items.append(V.cmp_eq('A:decisive_set_registered_in_bank_spec', spec.get('decisive'), decisive))
    gm, gr = load(a.main), load(a.rob)
    info['grids'] = {'main': gm['files'], 'rob': gr['files']}
    reg = list(V.REGISTERED)
    # the registered cells of the robustness grid are the registered grid's (cache re-use): exact equality of every member's AUCs
    members = [m for m in spec['members']]
    for c in reg:
        if c in gr['cells']:
            for m in members:
                for view in ('pair', 'income', 'occupation'):
                    items.append(V.cmp_eq(f'A:{c}:{m}:{view}:registered_cell_score_auc_equal_in_both_grids',
                                          gr['cells'][c][m][view]['score_auc'], gm['cells'][c][m][view]['score_auc']))
    cells = dict(gm['cells'])
    for c, x in gr['cells'].items():
        if c not in cells:
            cells[c] = x
    allc = sorted(c for c in cells if '@' not in c)
    rob = [c for c in allc if c not in reg]
    rob34 = [c for c in rob if V.parse_cell(c)[1] in (3, 4)]
    info['cells'] = {'registered': len([c for c in allc if c in reg]), 'robustness': len(rob), 'robustness_R3_R4_only': len(rob34)}
    se_all = {}
    for comp, views in gr['paired'].get(cand, {}).items():
        for view, e in views.items():
            for c in reg:
                if c in e['cell_se'] and comp in gm['paired'].get(cand, {}):
                    items.append(V.cmp_num(f'A:{c}:{cand}_vs_{comp}:{view}:registered_cell_se_equal_in_both_grids', e['cell_se'][c],
                                           gm['paired'][cand][comp][view]['cell_se'].get(c), 0.0))
    for g in (gm, gr):
        for comp, views in g['paired'].get(cand, {}).items():
            for view, e in views.items():
                se_all.setdefault(comp, {}).setdefault(view, {}).update(e['cell_se'])
    per, caveats = {}, []
    for comp in [x for x in dict.fromkeys([q, obj, twin, 'SUCC'] + list(spec.get('benchmark_references') or [])) if x]:
        if comp not in se_all:
            per[comp] = {'label': 'INCOMPLETE', 'reason': 'no paired record in the merged grids'}
            continue
        rec = {}
        for name, use in (('registered', reg), ('robustness', rob), ('all', allc), ('robustness_R3_R4_only', rob34)):
            pv = cell_values(cells, cand, comp, 'pair', use)
            ag = V.aggregate(pv, se_all[comp]['pair'], use=use)
            rec[name] = {'x': ag.get('x'), 'se_agg': ag.get('se_agg'), 'version_means': ag.get('version_means'),
                         'label': V.benefit_label(ag), 'n_cells': len(use)}
            for view in ('income', 'occupation'):
                rv = V.recipient(cell_values(cells, cand, comp, view, use), se_all[comp][view], use=use)
                rec[name][f'{view}_gate'] = rv['gate']
                rec[name][f'{view}_recipient_label'] = rv['label']
        confirmed = rec['robustness']['label'] == 'AGG_POSITIVE' and rec['all']['label'] == 'AGG_POSITIVE'
        gates = rec['robustness']['income_gate'] and rec['robustness']['occupation_gate']
        sc = (rec['registered']['x'] is not None and rec['robustness']['x'] is not None
              and sign(rec['registered']['x']) != sign(rec['robustness']['x']))
        rec['sign_change'] = bool(sc)
        rec['robustness_recipient_failure'] = not gates
        rec['nr_robustness'] = 'ROBUSTNESS_CONFIRMED' if confirmed else 'ROBUSTNESS_NOT_CONFIRMED'
        if sc:
            caveats.append(f'SIGN_CHANGE_{comp}')
        if not gates:
            caveats.append(f'ROBUSTNESS_RECIPIENT_FAILED_{comp}')
        per[comp] = rec
    # ---- refit effectiveness
    from drf import common as C                       # label-free MODEL_SELECT household ids (features)
    unit = C.features('MODEL_SELECT')['unit']
    man = json.loads((RES / 'READER_SPLITS.json').read_text())
    score_sets, hashes = {}, {}
    for rep in ('R1', 'R2', 'R3', 'R4'):
        part = V.household_parts(unit, 10, f'DRF1_devgrid_{rep}')
        sc = part >= 7
        hh = sorted(set(unit[sc].tolist()))
        score_sets[rep] = set(hh)
        hashes[rep] = hashlib.sha256(','.join(map(str, hh)).encode()).hexdigest()
        items.append(V.cmp_eq(f'R:{rep}_SCORE_hash_equals_manifest', hashes[rep], man['repetitions'][rep]['SCORE']['household_set_sha256']))
    run_reps = sorted({f'R{V.parse_cell(c)[1]}' for c in allc})
    items.append(V.cmp_eq('R:robustness_repetitions_run', [r for r in ('R3', 'R4') if r in run_reps], ['R3', 'R4']))
    distinct_hash = len({hashes[r] for r in ('R1', 'R2', 'R3', 'R4')}) == 4
    items.append(V.cmp_eq('R:assignment_hashes_distinct', distinct_hash, True))
    overlaps = {}
    for a_, b_ in (('R1', 'R3'), ('R1', 'R4'), ('R2', 'R3'), ('R2', 'R4'), ('R3', 'R4')):
        n = len(score_sets[a_] & score_sets[b_])
        f = n / len(score_sets[a_])
        overlaps[f'{a_}_vs_{b_}'] = {'households': n, 'fraction': f}
        items.append(V.cmp_eq(f'R:SCORE_overlap_{a_}_vs_{b_}_recorded', n, man['household_overlap_between_repetitions'].get(f'{a_}_vs_{b_}_SCORE')))
        items.append(V.cmp_eq(f'R:SCORE_overlap_{a_}_vs_{b_}_in_0_1', bool(math.isfinite(f) and 0 <= f < 1), True))
    distinct = {}
    for m in members:
        sid = SUBJECT.get(m, m)
        for v in (0, 1):
            distinct[(m, v)] = c2['per_subject'][sid][str(v)]['distinct_inputs']
    minimum = {x for x in (cand, twin, obj, 'SUCC', q) if x}
    sv = seed_validity(cells, set(members), minimum, distinct, reps=(1, 2, 3, 4), pairs=((0, 1), (2, 3)))
    eff = bool(distinct_hash and all(r in run_reps for r in ('R3', 'R4')) and sv['valid'] and
               all(0 <= o['fraction'] < 1 for o in overlaps.values()))
    refit = {'label': 'REFIT_GENUINELY_EFFECTIVE' if eff else 'REFIT_NOT_GENUINELY_EFFECTIVE', 'seed_checks': sv['checks'],
             'seed_equalities': len(sv['equalities']), 'unexplained': sv['unexplained'], 'overlaps': overlaps}
    for k in ('CTRL', 'OBJ'):
        comp = decisive[k]
        r_ = per.get(comp, {})
        ok = (r_.get('nr_robustness') == 'ROBUSTNESS_CONFIRMED' and not r_.get('robustness_recipient_failure', True) and eff
              and r_.get('registered', {}).get('n_cells') == 8)
        r_['d2r_label'] = 'ROBUSTNESS_CONFIRMED' if ok else 'ROBUSTNESS_NOT_CONFIRMED'
    d2r_label = ('ROBUSTNESS_CONFIRMED' if all(per.get(decisive[k], {}).get('d2r_label') == 'ROBUSTNESS_CONFIRMED' for k in decisive)
                 else 'ROBUSTNESS_NOT_CONFIRMED')
    mine = {'label': d2r_label, 'decisive': {k: per.get(decisive[k], {}).get('d2r_label') for k in decisive}, 'caveats': caveats,
            'sign_changes': [c.replace('SIGN_CHANGE_', '') for c in caveats if c.startswith('SIGN_CHANGE_')],
            'recipient_failures': [c.replace('ROBUSTNESS_RECIPIENT_FAILED_', '') for c in caveats if c.startswith('ROBUSTNESS_RECIPIENT')]}
    compared = None
    dp = Path(a.d2r) if a.d2r else RES / 'claims' / plan['plan_id'] / 'D2R.json'
    if dp.is_file():
        rec = json.loads(dp.read_text())
        info['d2r_record'] = {'path': str(dp.relative_to(REPO)), 'sha256': V.sha256_file(dp)}
        rd = rec['d2r']
        compared = {'label': [d2r_label, rd.get('label')], 'refit': [refit['label'], (rec.get('refit_effectiveness') or {}).get('label')],
                    'sign_changes': [sorted(mine['sign_changes']), sorted(rd.get('sign_changes') or [])],
                    'recipient_failures': [sorted(mine['recipient_failures']), sorted(rd.get('recipient_failures') or [])]}
        items.append(V.cmp_eq('D:d2r_label', d2r_label, rd.get('label')))
        items.append(V.cmp_eq('D:refit_effectiveness', refit['label'], (rec.get('refit_effectiveness') or {}).get('label')))
        items.append(V.cmp_eq('D:comparators', rec.get('comparators'), decisive))
        # (added 04:31Z) per-comparator values and labels against the record; the record's lists may repeat a comparator, so the
        # sign-change and recipient-failure lists are compared as sets
        items.append(V.cmp_eq('D:sign_change_set', sorted(set(mine['sign_changes'])), sorted(set(rd.get('sign_changes') or []))))
        items.append(V.cmp_eq('D:recipient_failure_set', sorted(set(mine['recipient_failures'])),
                              sorted(set(rd.get('recipient_failures') or []))))
        for comp, mr in per.items():
            rc = (rd.get('per_comparator') or {}).get(comp) or {}
            if not rc or rc.get('label') == 'NOT_RUN':
                items.append({'name': f'D:{comp}:record_entry_missing', 'result': 'FAIL'})
                continue
            items.append(V.cmp_num(f'D:{comp}:registered_mean', mr['registered']['x'], rc.get('registered_mean'), 1e-12))
            items.append(V.cmp_num(f'D:{comp}:robustness_mean', mr['robustness']['x'], rc.get('robustness_mean'), 1e-12))
            items.append(V.cmp_num(f'D:{comp}:robustness_se', mr['robustness']['se_agg'], rc.get('robustness_se'), 1e-12))
            items.append(V.cmp_eq(f'D:{comp}:all_cells_label', mr['all']['label'], rc.get('all_cells_label')))
            items.append(V.cmp_eq(f'D:{comp}:nrm_label', mr['nr_robustness'], rc.get('nrm_label')))
            items.append(V.cmp_eq(f'D:{comp}:sign_change', mr['sign_change'], rc.get('sign_change')))
            items.append(V.cmp_eq(f'D:{comp}:robustness_recipient_labels',
                                  [mr['robustness']['income_recipient_label'], mr['robustness']['occupation_recipient_label']],
                                  [(rc.get('recipient_robustness') or {}).get('income'), (rc.get('recipient_robustness') or {}).get('occupation')]))
            if comp in decisive.values():
                items.append(V.cmp_eq(f'D:{comp}:decisive_label', mr.get('d2r_label'), rc.get('label')))
        rr_ = rec.get('refit_effectiveness') or {}
        gv_ = rr_.get('grid_validity') or {}
        items.append(V.cmp_eq('D:refit_seed_equalities_count', len(sv['equalities']), (gv_.get('explained_equalities') or {}).get('count')))
    out = {'schema': 'DRF1-verifier-check3e-v1', 'info': info, 'decisive': decisive, 'mine': mine, 'per_comparator': per,
           'refit_effectiveness': refit, 'compared_with_d2r': compared}
    groups = {}
    for it in items:
        groups.setdefault(it['name'].split(':', 1)[0], []).append(it)
    out['summary'] = {g: V.summarize(x) for g, x in groups.items()}
    out['overall'] = V.summarize(items)
    out['items'] = items
    if a.out:
        V.dump(a.out, out)
    print(json.dumps(V.jsonable({'overall': out['overall'], 'mine': mine, 'refit': {k: refit[k] for k in ('label', 'seed_checks',
                                                                                                         'seed_equalities')},
                                 'per_comparator': {k: {'registered': v.get('registered', {}).get('x'),
                                                        'robustness': v.get('robustness', {}).get('x'),
                                                        'rob_label': v.get('robustness', {}).get('label'),
                                                        'all_label': v.get('all', {}).get('label'),
                                                        'rob_gates': [v.get('robustness', {}).get('income_gate'),
                                                                      v.get('robustness', {}).get('occupation_gate')],
                                                        'sign_change': v.get('sign_change'), 'd2r': v.get('d2r_label')}
                                                    for k, v in per.items()},
                                 'compared': compared}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
