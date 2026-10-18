"""Verifier checks 3-4: ladder costs (OOF, GUARD, CHECK, MODEL_SELECT; full and ensemble deployments), the witness baseline
slack, rule selections (OLD-B80 / OLD-B50 continuous projection, SIMPLE, UNCERTAINTY), realized outcomes and zero-correction
accounting, recomputed with the verifier's own metric and selection code (vlib) from the saved release arrays and labels.

PCRL correction-transfer campaign (CTS1; exploratory development on previously used ACS 2019 Florida PUMS person records).
Labels are read only through cts.common.load(operation='verify') from this script (D partitions: FIT, GUARD, diagnostic CHECK;
MODEL_SELECT). Repair panels are never requested here. Nothing person-level is printed or written.

    python3 results/pcrl_correction_transfer_stability_v1/roles/verifier/code/v_costs.py --partition D0 \
        --directions REFIT_PAIR,FROZEN_PAIR --tag V1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

ROOT = V.ROOT
sys.path.insert(0, str(ROOT))
from cts import common as C  # noqa: E402   (admitted-data loader and the logged label-access layer only)

CAMP = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
TRACKF = CAMP / 'runs' / 'TRACKF'
RES = ROOT / 'results' / 'pcrl_correction_transfer_stability_v1'
TOL = 1e-10
ESTS = ('OOF', 'GUARD', 'GUARD_ENS', 'CHECK', 'CHECK_ENS', 'MS', 'MS_ENS')


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def read_labels(P):
    """All label reads of this unit (operation 'verify'; called from this script so the access layer logs this entry point)."""
    out = {}
    for r in ('FIT', 'GUARD', 'CHECK'):
        out[r] = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'verify',
                        f'independent verifier: replay of Track F {P} {r} incremental costs and rule selections with separate code',
                        partition=P, prole=r)
    out['MS'] = C.load('MODEL_SELECT', ('features', 'occupation', 'sex'), 'verify',
                       f'independent verifier: replay of Track F {P} MODEL_SELECT costs, witness baseline and realized outcomes')
    return out


def frozen_refs(seed, ms_row_id):
    from ohn import common as OC          # admitted, hash-checked data loader (label-free reference outputs)
    bank, ids = OC.frozen_bank(seed, roles=('MODEL_SELECT',))
    if not np.array_equal(ids['row_id__MODEL_SELECT'], ms_row_id):
        raise AssertionError('frozen bank MODEL_SELECT row order differs')
    return {r: np.asarray(bank[r]['MODEL_SELECT']['occupation'], dtype=np.float64) for r in ('U0', 'Ucal')}


def my_conditions(p_w, refs, y, sex, unit):
    own = V.person_metrics(y, p_w)
    out = {}
    for ref in ('U0', 'Ucal'):
        r = V.person_metrics(y, refs[ref])
        for metric, d in (('accuracy_loss', r['accuracy'] - own['accuracy']), ('nll_excess', own['nll'] - r['nll']),
                          ('brier_excess', own['brier'] - r['brier'])):
            out[f'occupation|{ref}|{metric}'] = {'value': float(d.mean()), 'se': V.clustered_se(d, unit), 'cap': V.CAP[metric]}
    u0 = V.person_metrics(y, refs['U0'])
    d = own['brier'] - u0['brier']
    for g in (0, 1):
        m = np.asarray(sex) == g
        out[f'guard|sex{g}'] = {'value': float(d[m].mean()), 'se': V.subgroup_se(d, m, unit), 'cap': V.CAP['guard']}
    return out


def recorded_conditions(ev):
    out = {}
    for c in ev['eval']['utility']['clauses']:
        if c['task'] == 'occupation':
            out[f"occupation|{c['reference']}|{c['metric']}"] = {'value': c['value'], 'se': c['se'], 'cap': c['cap']}
    for g in ev['eval']['utility']['guards']:
        out[f"guard|{g['group']}"] = {'value': g['value'], 'se': g['se'], 'cap': g['cap']}
    return out


def ladder_mine(P, W, O, T, D, L, masks_ok):
    """Own recomputation of every estimator of the ladder."""
    res = {}
    F = L['FIT']
    z0 = V.centred_log(O['oof'])
    p0 = V.softmax_rows(z0)
    res['OOF'] = {str(s): V.incremental_costs(V.softmax_rows(z0 + s * T['delta_oof']), p0, F['occupation'], F['sex'], F['unit'])
                  for s in V.LADDER}
    for name in ('GUARD', 'CHECK'):
        X = L[name]; rows = X['rows']
        pw, dl = W['dep__MODEL_FIT'][rows], D['full__MODEL_FIT'][rows]
        pf, df = W['fold__MODEL_FIT'][:, rows], D['folds__MODEL_FIT'][:, rows]
        res[name] = {str(s): V.incremental_costs(V.release_full(pw, dl, s), pw, X['occupation'], X['sex'], X['unit']) for s in V.LADDER}
        e0 = V.release_ensemble(pf, df, 0.0)
        res[name + '_ENS'] = {str(s): V.incremental_costs(V.release_ensemble(pf, df, s), e0, X['occupation'], X['sex'], X['unit'])
                              for s in V.LADDER}
    S = L['MS']
    pw, dl = W['dep__MODEL_SELECT'], D['full__MODEL_SELECT']
    res['MS'] = {str(s): V.incremental_costs(V.release_full(pw, dl, s), pw, S['occupation'], S['sex'], S['unit']) for s in V.LADDER}
    pf, df = W['fold__MODEL_SELECT'], D['folds__MODEL_SELECT']
    e0 = V.release_ensemble(pf, df, 0.0)
    res['MS_ENS'] = {str(s): V.incremental_costs(V.release_ensemble(pf, df, s), e0, S['occupation'], S['sex'], S['unit'])
                     for s in V.LADDER}
    return res


def magnitude_defs(W, D, rows, s):
    """Correction magnitude on GUARD at strength s, both candidate definitions (label-free)."""
    pw, dl = W['dep__MODEL_FIT'][rows], D['full__MODEL_FIT'][rows]
    p = V.release_full(pw, dl, s)
    diff = np.abs(V.centred_log(p) - V.centred_log(pw))
    dc = s * dl - np.mean(s * dl, axis=1, keepdims=True)
    return {'sum_over_classes_mean_over_people': float(diff.sum(1).mean()),
            'mean_over_people_and_classes (RULES.md)': float(np.abs(dc).mean()),
            'flip_rate': float((np.argmax(p, 1) != np.argmax(pw, 1)).mean()), 'max_abs_prob_change': float(np.abs(p - pw).max())}


def compare_ladder(mine, rec):
    worst = {'stats': 0.0, 'se': 0.0}
    count_mismatch, missing = [], []
    for est in ESTS:
        if est not in rec['estimators']:
            missing.append(est); continue
        for s in map(str, V.LADDER):
            a, b = mine[est][s], rec['estimators'][est][s]
            for part in ('stats', 'se'):
                for k in V.STATS:
                    worst[part] = max(worst[part], abs(a[part][k] - b[part][k]))
            for k in ('n', 'n_men', 'n_women', 'households'):
                if a[k] != b[k]:
                    count_mismatch.append(f'{est}/{s}/{k}: {a[k]} vs {b[k]}')
    return worst, count_mismatch, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition', required=True); ap.add_argument('--directions', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    P = a.partition
    spec = json.loads((RES / 'PARTITIONS.json').read_text())
    rules = json.loads((RES / 'roles' / 'statistical_reviewer' / 'RULES.json').read_text())
    seed = int(spec['partitions'][P]['associated_source_seed'])
    Fall = C.features('MODEL_FIT')
    masks = V.partition_masks_from_json(Fall['unit'], spec)
    L = read_labels(P)
    out, record = [], {'partition': P, 'tag': a.tag}
    # ---------------- access layer returned exactly the reconstructed rows
    rows_ok = {r: bool(np.array_equal(L[r]['rows'], masks[P][r])) for r in ('FIT', 'GUARD', 'CHECK')}
    # ---------------- witness arrays and integrity
    wd = TRACKF / P / 'WITNESS'
    wj = json.loads((wd / 'WITNESS.json').read_text())
    integrity = [f for f in ('WITNESS.npz', 'OOF.npz') if V.file_sha(wd / f) != wj['files_sha256'][f]]
    W, O = npz(wd / 'WITNESS.npz'), npz(wd / 'OOF.npz')
    if not np.array_equal(O['row_id'], L['FIT']['row_id']):
        raise AssertionError('OOF row ids differ from the FIT rows')
    # ---------------- witness baseline (check: recorded evaluation and my recomputation)
    ev = json.loads((TRACKF / P / 'EVAL' / 'WITNESS_EVAL_MS.json').read_text())
    S = L['MS']
    refs = frozen_refs(seed, S['row_id'])
    mine_c = my_conditions(W['dep__MODEL_SELECT'], refs, S['occupation'], S['sex'], S['unit'])
    rec_c = recorded_conditions(ev)
    dv = max(abs(mine_c[c]['value'] - rec_c[c]['value']) for c in V.CONDITIONS)
    dse = max(abs(mine_c[c]['se'] - rec_c[c]['se']) for c in V.CONDITIONS)
    G = int(len(np.unique(S['unit'])))
    floor = rules['reserve']['floor_se'][str(seed)]
    base_mine = V.baseline_from_conditions(mine_c, floor, G)
    base_rec = V.baseline_from_conditions(rec_c, floor, G)
    n_ms = {'all': int(len(S['sex'])), 'men': int((S['sex'] == 0).sum()), 'women': int((S['sex'] == 1).sum())}
    base_cmp = {}
    out.append({'id': f'C3.{P}.WITNESS_BASELINE', 'kind': 'NUMERICAL_REPLAY',
                'what': f'{P} witness MODEL_SELECT occupation-side conditions (6 clauses vs U0 / Ucal and the men\'s / women\'s Brier '
                        'guards vs U0) and the reserve slack rho = margin - 2 max(own SE, floor SE) that every rule reads',
                'method': 'own per-person metrics and household-clustered / linearized subgroup SEs from the saved witness MODEL_SELECT '
                          'release, the admitted frozen-bank U0 / Ucal outputs and MODEL_SELECT labels (verify); floor SEs from RULES.json',
                'status': 'PASS' if (dv <= TOL and dse <= TOL and not integrity and all(rows_ok.values())) else 'FAIL',
                'numbers': {'max_abs_value_diff': dv, 'max_abs_se_diff': dse, 'households': G, 'n_ms': n_ms,
                            'conditions': {c: {'value': mine_c[c]['value'], 'se': mine_c[c]['se'], 'rho': base_mine[c]['rho'],
                                               'margin': base_mine[c]['margin'], 'se_used': base_mine[c]['se_used'],
                                               'se_source': base_mine[c].get('se_source')} for c in V.CONDITIONS},
                            'access_rows_equal_reconstruction': rows_ok, 'witness_file_integrity_problems': integrity}})
    n_guard = {'all': int(len(L['GUARD']['sex'])), 'men': int((L['GUARD']['sex'] == 0).sum()), 'women': int((L['GUARD']['sex'] == 1).sum())}
    record['n_guard'] = n_guard; record['n_ms'] = n_ms
    for did in [d for d in a.directions.split(',') if d]:
        dd = TRACKF / P / 'DIR' / did
        dj = json.loads((dd / 'DIR.json').read_text())
        di = [f for f in ('TRAIN.npz', 'DELTA.npz') if V.file_sha(dd / f) != dj['files_sha256'][f]]
        T, D = npz(dd / 'TRAIN.npz'), npz(dd / 'DELTA.npz')
        lp = TRACKF / P / 'LADDER' / f'{did}.json'
        sp = TRACKF / P / 'SELECT' / f'{did}.json'
        rp = TRACKF / P / 'SELECT' / f'{did}__REALIZED.json'
        mine = ladder_mine(P, W, O, T, D, L, rows_ok)
        # ---------------- check 3: ladder costs
        if lp.is_file():
            lad = json.loads(lp.read_text())
            worst, cm, missing = compare_ladder(mine, lad)
            mag = magnitude_defs(W, D, L['GUARD']['rows'], 1.0)
            rec_mag = lad['estimators']['GUARD']['1.0']['magnitude']
            out.append({'id': f'C3.{P}.{did}.LADDER_COSTS', 'kind': 'NUMERICAL_REPLAY',
                        'what': f'{P} {did}: incremental costs (men / women / all summed Brier, NLL, accuracy loss) and their household-'
                                'clustered SEs at all 11 ladder strengths for OOF (FIT), GUARD, GUARD_ENS, CHECK, CHECK_ENS, MS, MS_ENS',
                        'method': 'own release composition (softmax of centred log p_W + s Delta; ensemble = mean of the 5 fold releases) '
                                  'and own metrics / SEs, from WITNESS.npz, OOF.npz, TRAIN.npz, DELTA.npz and verify-operation labels; '
                                  f'tolerance {TOL} on every statistic and SE, exact people / household counts',
                        'status': 'PASS' if (worst['stats'] <= TOL and worst['se'] <= TOL and not cm and not missing and not di) else 'FAIL',
                        'numbers': {'max_abs_diff_stats': worst['stats'], 'max_abs_diff_se': worst['se'], 'count_mismatches': cm[:20],
                                    'estimators_missing_in_record': missing, 'direction_file_integrity_problems': di,
                                    'GUARD_s1.0_magnitude_mine': mag, 'GUARD_s1.0_magnitude_recorded': rec_mag}})
            # magnitude definition (diagnostic, never gating)
            ok_sum = abs(rec_mag['mean_abs_logit_change'] - mag['sum_over_classes_mean_over_people']) <= 1e-9
            reg_key = 'mean_abs_logit_change_per_class' if 'mean_abs_logit_change_per_class' in rec_mag else 'mean_abs_logit_change'
            ok_rules = abs(rec_mag[reg_key] - mag['mean_over_people_and_classes (RULES.md)']) <= 1e-9
            out.append({'id': f'C3.{P}.{did}.MAGNITUDE_DEFINITION', 'kind': 'NUMERICAL_REPLAY',
                        'what': f'{P} {did}: label-free correction magnitude recorded in the ladder (diagnostic, never gating)',
                        'method': 'recorded mean_abs_logit_change compared with both candidate definitions at s = 1.0 on GUARD; flip rate '
                                  'and max |prob change| compared directly',
                        'status': ('PASS' if ok_rules else 'ATTENTION') if (abs(rec_mag['flip_rate'] - mag['flip_rate']) <= 1e-12 and
                                                                           abs(rec_mag['max_abs_prob_change'] - mag['max_abs_prob_change']) <= 1e-9)
                        else 'FAIL',
                        'numbers': {'recorded': rec_mag, 'mine': mag, 'recorded_mean_abs_logit_change_matches_sum_over_classes': ok_sum,
                                    'registered_key_compared': reg_key,
                                    'registered_key_matches_RULES_md_mean_over_people_and_classes': ok_rules,
                                    'note': 'RULES.md section 5 defines the magnitude as the mean of |s x centred Delta| over people AND classes '
                                            '(cts.rules.correction_magnitude); cts.trackf.cost_summary sums over the 5 classes (5 times larger)'}})
        else:
            lad = None
            out.append({'id': f'C3.{P}.{did}.LADDER_COSTS', 'kind': 'NUMERICAL_REPLAY', 'what': f'{P} {did} ladder', 'method': '-',
                        'status': 'NOT_RUN', 'numbers': {'reason': 'LADDER record absent at run time'}})
        # ---------------- check 4: rule selections
        if sp.is_file():
            sel = json.loads(sp.read_text())
            # baseline the rules read (record) vs mine
            bd = max(abs(sel['baseline'][c]['slack'] - base_mine[c]['rho']) for c in V.CONDITIONS)
            bm = max(abs(sel['baseline'][c]['margin'] - base_mine[c]['margin']) for c in V.CONDITIONS)
            bs = max(abs(sel['baseline'][c]['se_used'] - base_mine[c]['se_used']) for c in V.CONDITIONS)
            rho = {c: base_mine[c]['rho'] for c in V.CONDITIONS}
            replay, mism = {}, []
            F = L['FIT']
            z0 = V.centred_log(O['oof'])
            for lvl, f in V.OLD_F.items():
                budget = V.stat_budget(rho, f)
                r = V.old_scale(z0, T['delta_oof'], F['occupation'], F['sex'], budget)
                recsel = sel['selections'][f'OLD-{lvl}']['selected']
                replay[f'OLD-{lvl}'] = {'mine': r['selected'], 'recorded': recsel, 'abs_diff': abs(r['selected'] - recsel),
                                        'budget_max_abs_diff': max(abs(budget[k] - sel['selections'][f'OLD-{lvl}']['budget'][k])
                                                                   for k in V.STATS)}
                if abs(r['selected'] - recsel) > 1e-9 or replay[f'OLD-{lvl}']['budget_max_abs_diff'] > TOL:
                    mism.append(f'OLD-{lvl}')
                for dep in ('full', 'ens'):
                    if abs(sel['selected'][f'OLD-{lvl}|{dep}'] - r['selected']) > 1e-9:
                        mism.append(f'OLD-{lvl}|{dep}')
            for dep, est in (('full', 'GUARD'), ('ens', 'GUARD_ENS')):
                for src_name, src in (('saved_ladder', (lad['estimators'][est] if lad else None)), ('own_ladder', mine[est])):
                    if src is None:
                        continue
                    gc = {s: src[str(s)]['stats'] for s in V.LADDER}
                    gs = {s: src[str(s)]['se'] for s in V.LADDER}
                    ng = {'all': src['0.0']['n'], 'men': src['0.0']['n_men'], 'women': src['0.0']['n_women']}
                    simple = V.simple_select(gc, rho)
                    unc = V.uncertainty_select(gc, gs, base_mine, ng, n_ms)
                    for rule, r in (('SIMPLE', simple), ('UNCERTAINTY', unc)):
                        recsel = sel['selected'][f'{rule}|{dep}']
                        replay[f'{rule}|{dep}|{src_name}'] = {'mine': r['selected'], 'recorded': recsel,
                                                              'first_failure': r['first_failure'],
                                                              'failing_conditions': r['failing_at_first_failure']}
                        if abs(r['selected'] - recsel) > 1e-12:
                            mism.append(f'{rule}|{dep}|{src_name}')
            n_ms_rec = sel.get('n_ms')
            if n_ms_rec and n_ms_rec != n_ms:
                mism.append(f'n_ms {n_ms_rec} vs {n_ms}')
            out.append({'id': f'C4.{P}.{did}.RULE_SELECTIONS', 'kind': 'NUMERICAL_REPLAY',
                        'what': f'{P} {did}: OLD-B80 / OLD-B50 (continuous projection on OOF FIT costs) and SIMPLE / UNCERTAINTY (fixed '
                                'ladder on GUARD costs, full and ensemble deployments) selections, and the baseline slack they read',
                        'method': 'own implementation of RULES.md sections 2-5 (101-point grid + 30 bisections; gamma 0.5; z 1.644853626951472; '
                                  'acc_loss SE floor 1 / n_GUARD; predictive inflation sqrt(1 + n_GUARD,g / n_MS,g)); SIMPLE / UNCERTAINTY '
                                  'replayed on both the saved ladder and the verifier\'s own ladder',
                        'status': 'PASS' if (not mism and bd <= TOL and bm <= TOL and bs <= TOL) else 'FAIL',
                        'numbers': {'replay': replay, 'mismatches': mism, 'baseline_rho_max_abs_diff': bd,
                                    'baseline_margin_max_abs_diff': bm, 'baseline_se_used_max_abs_diff': bs,
                                    'rho_used': rho}})
            # ---------------- realized outcomes and zero-correction accounting
            if rp.is_file():
                real = json.loads(rp.read_text())
                worst, flag_mism, ident, complete, missing_complete = 0.0, [], {}, {}, []
                sets = {'CHECK': L['CHECK'], 'MS': L['MS']}
                for name, X in sets.items():
                    if name == 'CHECK':
                        rows = X['rows']
                        pw, dl = W['dep__MODEL_FIT'][rows], D['full__MODEL_FIT'][rows]
                        pf, df = W['fold__MODEL_FIT'][:, rows], D['folds__MODEL_FIT'][:, rows]
                    else:
                        pw, dl, pf, df = W['dep__MODEL_SELECT'], D['full__MODEL_SELECT'], W['fold__MODEL_SELECT'], D['folds__MODEL_SELECT']
                    e0 = V.release_ensemble(pf, df, 0.0)
                    for key, s in sel['selected'].items():
                        dep = key.split('|')[1]
                        if dep == 'full':
                            c = V.incremental_costs(V.release_full(pw, dl, s), pw, X['occupation'], X['sex'], X['unit'])
                        else:
                            c = V.incremental_costs(V.release_ensemble(pf, df, s), e0, X['occupation'], X['sex'], X['unit'])
                        rr = real['realized'][name][key]
                        worst = max(worst, max(abs(c['stats'][k] - rr['cost']['stats'][k]) for k in V.STATS))
                        oc = V.outcome(c['stats'], base_mine)
                        ro = rr['outcome']
                        if oc['reserve_crossing'] != ro['reserve_crossing'] or oc['point_cap_crossing'] != ro['point_cap_crossing']:
                            flag_mism.append(f'{name}/{key}')
                        # complete-release outcome: the released vector against the deployed witness W (whose margins define rho)
                        if dep == 'full':
                            cw = c
                        else:
                            cw = V.incremental_costs(V.release_ensemble(pf, df, s), pw, X['occupation'], X['sex'], X['unit'])
                        if 'outcome_complete' in rr:
                            if 'cost_vs_witness' in rr:
                                worst = max(worst, max(abs(cw['stats'][k] - rr['cost_vs_witness']['stats'][k]) for k in V.STATS))
                            occ_ = V.outcome(cw['stats'], base_mine)
                            if (occ_['reserve_crossing'] != rr['outcome_complete']['reserve_crossing'] or
                                    occ_['point_cap_crossing'] != rr['outcome_complete']['point_cap_crossing']):
                                flag_mism.append(f'{name}/{key}/complete')
                            complete[f'{name}/{key}'] = {'selected': s, 'reserve_crossing': occ_['reserve_crossing'],
                                                         'point_cap_crossing': occ_['point_cap_crossing'],
                                                         'max_slack_use': occ_['max_slack_use'],
                                                         'men_brier_vs_W': cw['stats']['men_brier']}
                        else:
                            missing_complete.append(f'{name}/{key}')
                        if s == 0.0:
                            ident[f'{name}/{key}'] = {'identity_flag': ro.get('identity'), 'max_abs_cost': max(abs(rr['cost']['stats'][k])
                                                                                                               for k in V.STATS),
                                                      'reserve_crossing_flag': ro['reserve_crossing'],
                                                      'crossing_conditions_mine': oc['reserve_crossing_conditions']}
                id_ok = all(v['identity_flag'] is True and v['max_abs_cost'] <= 1e-12 for v in ident.values())
                id_cross = {k: v for k, v in ident.items() if v['reserve_crossing_flag']}
                out.append({'id': f'C4.{P}.{did}.REALIZED_AND_IDENTITY', 'kind': 'NUMERICAL_REPLAY',
                            'what': f'{P} {did}: realized CHECK and MODEL_SELECT incremental costs at every rule-selected strength, the '
                                    'reserve / point-cap crossing flags, and identity selections reported as identity with zero cost',
                            'method': 'own composition and metrics at the recorded selected strengths; own crossing test (cost > rho_c, '
                                      'cost > margin_c) against the own baseline',
                            'status': ('PASS' if (worst <= TOL and not flag_mism and id_ok) else 'FAIL') if not id_cross
                            else ('ATTENTION' if (worst <= TOL and not flag_mism and id_ok) else 'FAIL'),
                            'numbers': {'max_abs_cost_diff': worst, 'flag_mismatches': flag_mism, 'identity_selections': ident,
                                        'complete_release_outcomes_mine': complete,
                                        'records_without_outcome_complete': missing_complete,
                                        'identity_selections_flagged_as_reserve_crossing': sorted(id_cross),
                                        'note': ('an identity selection spends nothing; a reserve-crossing flag on identity can only come '
                                                 'from a non-positive baseline slack (rho_c <= 0: the uncorrected witness already fails the '
                                                 'reserve), not from the correction') if id_cross else ''}})
            else:
                out.append({'id': f'C4.{P}.{did}.REALIZED_AND_IDENTITY', 'kind': 'NUMERICAL_REPLAY', 'what': f'{P} {did} realized',
                            'method': '-', 'status': 'NOT_RUN', 'numbers': {'reason': 'REALIZED record absent at run time'}})
        else:
            out.append({'id': f'C4.{P}.{did}.RULE_SELECTIONS', 'kind': 'NUMERICAL_REPLAY', 'what': f'{P} {did} selections', 'method': '-',
                        'status': 'NOT_RUN', 'numbers': {'reason': 'SELECT record absent at run time'}})
        # ---------------- transfer summary at the ladder (descriptive; numbers for the coordinator)
        ss = {k: min(base_mine[c]['rho'] for c in V.CONDITIONS if V.STAT_FOR[c] == k) for k in V.STATS}
        record[f'{did}__men_brier_by_strength'] = {
            str(s): {e: mine[e][str(s)]['stats']['men_brier'] for e in ESTS} for s in V.LADDER[1:]}
        record[f'{did}__stat_slack'] = ss
    code = V.code_hashes(__file__, V.__file__)
    for c in out:
        c['id'] = f"{c['id']}__{a.tag}"
        c['code_sha256'] = code
    record['checks'] = out
    V.write_record(f'V_COSTS__{P}__{a.tag}.json', record)
    summ = V.append_checks(out)
    print(json.dumps({c['id']: c['status'] for c in out}, indent=1))
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
