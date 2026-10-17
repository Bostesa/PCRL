"""Unit FFV_REPORT: the ONE logged coordinator follow-up re-read of the already-exposed DEV_REPORT truth (after this campaign's
committed and published REPORTING_LOCK), then both frozen scorecards (PROTOCOL.md section 8) in this one run.

Census context: public-use ACS 2019 Florida person records; readers infer the recorded SEX column from the released income and
occupation predictions (each output alone and the linked pair). Development robustness evidence only.

- ORIGINAL scorecard: the source ANALYSIS_LOCK roster, conjunctions, margins and critical values (restricted to this bank: the two
  H_JJ secondary keys are out of the bank) with the source's original winners (stored source scores). Must reproduce the source.
- ENHANCED scorecard: the same roster and rules with the enhanced winners (source 8 + new 8 readers; ATTACK_VALIDATION selection).
- Reader-free utility (24) and subgroup-guard (4) endpoints are identical in both. CV uses the source-definition OUT_HGB controls.
- Household bootstrap (B = 2048, seed 2026101705, G = 3955) shared across releases, endpoints, scorecards and seeds; paired
  differences within replicate; Wald bounds with the source critical values (functions imported from amd.report unchanged).
- Also: paired reader-class changes (enhanced - original), benefit changes, interpretation labels, reader-seed variation, the
  new-recipe control diagnostics, and (private) the per-replicate endpoint draws and per-household utility contributions used by
  the confirmation-feasibility analysis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from amd.report import crit, endpoint
from task_headroom.inference import group_weights
from task_conditioned_representation.inference import auc, bootstrap_auc
from . import common as C

VIEWS = ('income', 'occupation', 'pair')
OUT_OF_BANK = ('H_JJ', 'T_TREE')


def _scores(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def evaluate_claims(lock_claims, eps, CV, avail):
    """The source claim engine (amd/report.py, verbatim logic), parameterized on one scorecard's endpoints and CV."""
    def category(ep, test):
        if test == 'point_le':
            return 'PASS' if ep['point'] <= ep['margin'] + 0.0 else 'POINT_FAILURE'
        if ep['availability'] != 'AVAILABLE':
            return 'UNAVAILABLE'
        if ep['category'] in ('PASS', 'IDENTICAL_RELEASE'):
            return 'PASS'
        return ep['category']

    def ev(c, claims_so_far):
        if 'claim_ref' in c:
            ok = claims_so_far[c['claim_ref']]['holds']
            return {'condition': f"claim {c['claim_ref']}", 'holds': ok, 'category': 'PASS' if ok else 'POINT_FAILURE'}
        if c['test'] == 'cv':
            return {'condition': 'CV (source-definition OUT_HGB planted and shuffled controls inside the frozen bands, every view and seed)',
                    'holds': CV, 'category': 'PASS' if CV else 'CONTROL_FAILURE'}
        if c['test'] == 'av':
            ok = c['role'] == 'TASKONLY' or bool(avail.get(c['role']))
            return {'condition': f"AV {c['role']}", 'holds': ok, 'category': 'PASS' if ok else 'UNAVAILABLE'}
        ep = eps[c['endpoint']]
        if c['test'] == 'point_le':
            ok = bool(ep['point'] <= ep['margin'])
        elif c['test'] == 'clause':
            ok = bool(ep['clause_pass']) and ep['availability'] == 'AVAILABLE'
        elif c['test'] == 'clause_at_0020':
            ok = ep['clause_L1'] is not None and bool(ep['clause_L1'] > 0.020)
        else:
            raise ValueError('unregistered condition test')
        cat = 'PASS' if ok else (category(ep, c['test']) if c['test'] != 'clause_at_0020' else
                                 ('ESTABLISHED_SHORTFALL' if ep['family_upper'] < 0.020 else ('POINT_SHORTFALL' if ep['point'] <= 0.020 else 'UNRESOLVED')))
        return {'condition': c['endpoint'], 'test': c['test'], 'gate': c.get('gate'), 'holds': ok, 'category': cat, 'point': ep['point'],
                'clause_L1': ep.get('clause_L1'), 'clause_U1': ep.get('clause_U1'), 'family': [ep.get('family_lower'), ep.get('family_upper')]}

    claims = {}
    for cl in lock_claims:
        name = cl['name']
        if cl['kind'] in ('not_claimed', 'descriptive'):
            claims[name] = {'label': cl['label'], 'holds': None}
            continue
        if cl['kind'] == 'gate':
            conds = [ev(c, claims) for c in cl['conditions']]
            ok = all(c['holds'] for c in conds)
            failed = sorted([c for c in conds if not c['holds']], key=lambda c: cl['failure_order'].index(c['gate']))
            claims[name] = {'label': cl['success_label'] if ok else cl['failure_label'], 'holds': ok, 'conditions': conds,
                            'failed_in_order': [c['condition'] for c in failed]}
            continue
        if cl['kind'] == 'fixed_sequence':
            steps, stop = [], False
            for st in cl['steps']:
                if stop:
                    steps.append({'name': st['name'], 'status': 'NOT_TESTED_SEQUENCE_STOPPED', 'holds': False}); continue
                conds = [ev(c, claims) for c in st['conditions']]
                ok = all(c['holds'] for c in conds)
                steps.append({'name': st['name'], 'holds': ok, 'conditions': conds, 'first_failing': next((c for c in conds if not c['holds']), None)})
                stop = not ok
            lab = cl['labels']['both'] if all(s_['holds'] for s_ in steps) else (cl['labels']['to_only'] if steps[0]['holds'] else cl['labels']['none'])
            claims[name] = {'label': lab, 'holds': lab == cl['labels']['both'], 'steps': steps}
            continue
        conds = [ev(c, claims) for c in cl['conditions']]
        ok = all(c['holds'] for c in conds)
        if name == 'B':
            if ok:
                lab = cl['success_label']
            elif not CV:
                lab = 'CONTROL_FAILURE'
            elif not claims['E']['holds']:
                gates = {c['gate'] for c in claims['E']['conditions'] if not c['holds']}
                lab = ('FINALIST_FAILS_DEV_UTILITY_GATE' if 'UG' in gates else 'FINALIST_FAILS_DEV_SUBGROUP_POINT_GUARD' if 'SG' in gates
                       else 'FINALIST_FAILS_DEV_RECIPIENT_POINT_GATE')
            else:
                lab = 'NOT_SUPERIOR_TO_ALL_ELIGIBLE'
            nested = [ev(c, claims) for c in cl['nested']['adds']]
            claims[name] = {'label': lab, 'holds': ok, 'conditions': conds, 'first_failing': next((c for c in conds if not c['holds']), None),
                            'nested_0020': {'label': cl['nested']['name'] if (ok and all(c['holds'] for c in nested)) else 'NOT_ESTABLISHED', 'conditions': nested}}
        else:
            claims[name] = {'label': cl['success_label'] if ok else cl['failure_label'], 'holds': ok, 'conditions': conds,
                            'first_failing': next((c for c in conds if not c['holds']), None)}
    return claims


def run(out_dir, truth=None, priv_dir=None, event_name='DEV_REPORT_TRUTH_REREAD_BY_FFV_COORDINATOR', purpose=None):
    names = ('PROTOCOL.md', 'READER_BANK_LOCK.json', 'REPORTING_LOCK.json', 'FROZEN_CANDIDATE_MANIFEST.json')
    if truth is None:
        for n in names:
            if not C.committed(C.RESULTS / n):
                raise PermissionError(f'report requires the committed {n}')
    out = Path(out_dir); priv = Path(priv_dir) if priv_dir else C.RUNS / 'REPORT'
    if (out / 'DEV_REPORT_RESULTS.json').exists() or (priv / 'ENDPOINT_DRAWS.npz').exists():
        raise PermissionError('report results exist (one evaluation)')
    src_lock = json.loads((C.SRC_RESULTS / 'ANALYSIS_LOCK.json').read_text())
    rlock = json.loads((C.RESULTS / 'REPORTING_LOCK.json').read_text()) if truth is None else None
    inf = src_lock['inference']
    T = truth if truth is not None else C.dev_report_truth(purpose or
        'frozen-finalist validation: the one coordinator follow-up evaluation of the frozen finalist, bank and enhanced readers', event_name)
    y = {t: T[t] for t in C.TASKS}; sex = T['sex']
    W, G = group_weights(T['unit'], seed=inf['bootstrap_seed'], draws=inf['bootstrap_B'])
    if G != 3955 or len(sex) != 5943:
        raise SystemExit(f'HALT: DEV_REPORT has {G} households and {len(sex)} persons, not 3955 / 5943')
    Wsum = W.sum(1)
    c1 = crit(G, inf['alpha_family_one_sided'] / inf['n_claims'])
    cK = {fam: crit(G, 0.05 / (2 * k)) for fam, k in inf['K'].items()}; cm = crit(G, 0.025)
    for name, val in (('c1', c1), ('c_P', cK['P']), ('c_S', cK['S']), ('marginal', cm)):
        if abs(val - inf['critical_values_expected'][name]) > 1e-6:
            raise AssertionError(f'critical value {name} differs from the source registration')
    res = {'schema': 'FFV1-dev-report-v1', 'utc': C.now(),
           'status': 'DEVELOPMENT ROBUSTNESS EVIDENCE (follow-up re-read of an already-exposed role; not confirmation; the source adaptive selection history is not covered)',
           'households': int(G), 'persons': int(len(sex)), 'critical': {'c1': c1, 't_family': cK, 'K': inf['K'], 't_marginal_95': cm},
           'locks_sha256': {n: C.sha(C.RESULTS / n) for n in names if (C.RESULTS / n).exists()}, 'per_seed': {}}
    auc_pt, auc_dr, util, per_vec, recs = {}, {}, {}, {}, {}
    hh_ids, hh_inv = np.unique(T['unit'], return_inverse=True)
    hh_contrib = {}
    for s in C.SEEDS:
        bank, ids, _ = C.service_bank(s)
        frozen, fids = C.frozen_bank(s)
        if not (np.array_equal(ids['row_id__DEV_REPORT'], T['row_id']) and np.array_equal(fids['row_id__DEV_REPORT'], T['row_id'])):
            raise AssertionError('bank / DEV_REPORT row order differs')
        src_rec, src_sc = C.source_readers(s)
        ed = C.RUNS / f'READERS_ENH_S{s}'
        if rlock is not None:
            for f, key in (('DEV_REPORT_SCORES.npz', 'dev_report_scores_sha256'), ('AV_SCORES.npz', 'av_scores_sha256'), ('READERS_ENH_SEED.json', 'record_sha256')):
                if C.sha(ed / f) != rlock['enhanced_readers'][f'S{s}'][key]:
                    raise PermissionError(f'enhanced reader file {f} differs from REPORTING_LOCK')
        enh = json.loads((ed / 'READERS_ENH_SEED.json').read_text()); enh_sc = _scores(ed / 'DEV_REPORT_SCORES.npz')
        if not (np.array_equal(src_sc['row_id'], T['row_id']) and np.array_equal(enh_sc['row_id'], T['row_id'])):
            raise AssertionError('score / DEV_REPORT row order differs')
        recs[s] = enh

        def score_of(key):
            origin, k = key.split('::', 1)
            return src_sc[k] if origin == 'SRC' else enh_sc[k]
        srec = {'services': {}, 'controls_cv': {}, 'controls_diagnostic': {}, 'variation': {}, 'reproduction': enh['reproduction']}
        refs = {r: {t: frozen[r]['DEV_REPORT'][t] for t in C.TASKS} for r in ('U0', 'Ucal')}
        u0 = {t: C.task_vectors(y[t], refs['U0'][t], C.K[t]) for t in C.TASKS}
        for svc in C.BANK:
            v = {'recovery_original': {}, 'recovery_enhanced': {}, 'reader_class_change': {}, 'utility': {}, 'subgroup': {}}
            for view in VIEWS:
                for tag, wkey in (('orig', 'original'), ('enh', 'enhanced')):
                    w = enh['winners'][svc][wkey][view]; sc = score_of(w['reader_key'])
                    auc_pt[(svc, s, view, tag)] = auc(sex, sc); auc_dr[(svc, s, view, tag)] = bootstrap_auc(sex, sc, W)
                    se = float(np.std(auc_dr[(svc, s, view, tag)], ddof=1)); pt = auc_pt[(svc, s, view, tag)]
                    v['recovery_original' if tag == 'orig' else 'recovery_enhanced'][view] = {
                        'winner': w['candidate'], 'origin': w['origin'], 'attack_validation_auc': w['attack_validation_auc'], 'auc': pt, 'auc_se': se,
                        'marginal95': [pt - cm * se, pt + cm * se]}
                d = auc_dr[(svc, s, view, 'enh')] - auc_dr[(svc, s, view, 'orig')]; pt = auc_pt[(svc, s, view, 'enh')] - auc_pt[(svc, s, view, 'orig')]
                se = float(np.std(d, ddof=1))
                v['reader_class_change'][view] = {'enhanced_minus_original': pt, 'se': se, 'marginal95': [pt - cm * se, pt + cm * se],
                                                 'same_reader': enh['winners'][svc]['enhanced'][view]['reader_key'] == enh['winners'][svc]['original'][view]['reader_key']}
            for t in C.TASKS:
                k = C.K[t]; vv = C.task_vectors(y[t], bank[svc]['DEV_REPORT'][t], k); per_vec[(svc, s, t)] = vv
                for ref in ('U0', 'Ucal'):
                    rv = C.task_vectors(y[t], refs[ref][t], k)
                    for metric, d in {'accuracy_loss': rv['accuracy'] - vv['accuracy'], 'nll_excess': vv['nll'] - rv['nll'], 'brier_excess': vv['brier'] - rv['brier']}.items():
                        dr = (W * d[None, :]).sum(1) / Wsum
                        util[(svc, s, t, ref, metric)] = (float(d.mean()), dr)
                        v['utility'][f'{t}|{ref}|{metric}'] = {'point': float(d.mean()), 'se': float(np.std(dr, ddof=1)), 'cap': C.CAPS[metric], 'point_pass': bool(d.mean() <= C.CAPS[metric])}
                        if svc == 'FINALIST':
                            hh_contrib[f'S{s}|{t}|{ref}|{metric}'] = np.bincount(hh_inv, weights=d, minlength=len(hh_ids))
            vv = C.task_vectors(y['occupation'], bank[svc]['DEV_REPORT']['occupation'], 5)
            for g in (0, 1):
                m_ = sex == g; d = vv['brier'] - u0['occupation']['brier']; Wg = W[:, m_]
                dr = (Wg * d[m_][None, :]).sum(1) / Wg.sum(1)
                util[(svc, s, 'guard', g)] = (float(d[m_].mean()), dr)
                v['subgroup'][f'sex{g}'] = {'occupation_brier_excess_vs_U0': float(d[m_].mean()), 'se': float(np.std(dr, ddof=1))}
                if svc == 'FINALIST':
                    hh_contrib[f'S{s}|guard|sex{g}|sum'] = np.bincount(hh_inv[m_], weights=d[m_], minlength=len(hh_ids))
                    hh_contrib[f'S{s}|guard|sex{g}|count'] = np.bincount(hh_inv[m_], minlength=len(hh_ids)).astype(np.float64)
            v['utility_point_feasible_12_this_seed'] = all(x['point_pass'] for x in v['utility'].values())
            v['guard_point_pass_this_seed'] = all(x['occupation_brier_excess_vs_U0'] <= C.GUARD for x in v['subgroup'].values())
            srec['services'][svc] = v
        # controls: CV conjunct from the source-definition OUT_HGB controls (source run and this run), diagnostics for new recipes
        for view in VIEWS:
            sc_src_ctl = {'shuffled': auc(sex, src_sc[f'CONTROL__{view}__SHUFFLED']),
                          'planted': auc(sex, np.where(sex == 1, src_sc[f'CONTROL__{view}__PLANTED_1'], src_sc[f'CONTROL__{view}__PLANTED_0']))}
            srec['controls_cv'][f'source_run|{view}'] = {**sc_src_ctl, 'av_shuffled': src_rec['controls'][view]['shuffled_attack_validation_auc'],
                                                         'av_planted': src_rec['controls'][view]['planted_attack_validation_auc']}
            for name, crec in enh['controls'][view].items():
                key = f'CONTROL__{view}__{name}'
                c = {'shuffled': auc(sex, enh_sc[key + '__SHUFFLED']),
                     'planted': auc(sex, np.where(sex == 1, enh_sc[key + '__PLANTED_1'], enh_sc[key + '__PLANTED_0'])),
                     'av_shuffled': crec['shuffled_attack_validation_auc'], 'av_planted': crec['planted_attack_validation_auc']}
                (srec['controls_cv'] if name == 'OUT_HGB' else srec['controls_diagnostic'])[f'this_run|{view}' if name == 'OUT_HGB' else f'{name}|{view}'] = c
        refit_auc = {}
        for kind_, recs_ in (('VAR', enh['variation']), ('AUX', enh['aux_sample_variation'])):
            for svc, vrec in recs_.items():
                j0 = auc_pt[(svc, s, 'pair', 'enh')]
                vals = [j0] + [auc(sex, enh_sc[f'{kind_}__{svc}__j{j}']) for j in (1, 2, 3, 4) if f'{kind_}__{svc}__j{j}' in enh_sc]
                refit_auc[(kind_, svc)] = vals
                srec['variation'].setdefault(svc, {})[kind_] = {'winner': vrec['winner'], 'dev_auc_by_refit': vals,
                                                                'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                                                                'note': vrec.get('seed_dependence') or vrec.get('note')}
        for kind_ in ('VAR', 'AUX'):
            for ref in ('NEURAL_RAW', 'H_LJ'):
                a_, b_ = refit_auc.get((kind_, ref)), refit_auc.get((kind_, 'FINALIST'))
                if a_ and b_ and len(a_) == len(b_) and len(a_) > 1:
                    bj = [x - y for x, y in zip(a_, b_)]
                    srec['variation'].setdefault('benefit_matched_refits', {})[f'{kind_}|{ref}'] = {'benefit_by_refit': bj, 'sd': float(np.std(bj, ddof=1))}
                else:
                    sa = np.std(a_, ddof=1) if a_ and len(a_) > 1 else 0.0; sb = np.std(b_, ddof=1) if b_ and len(b_) > 1 else 0.0
                    srec['variation'].setdefault('benefit_matched_refits', {})[f'{kind_}|{ref}'] = {
                        'benefit_by_refit': None, 'sd': float(np.sqrt(sa ** 2 + sb ** 2)),
                        'note': 'one side not refitted (seed-invariant or LOOKUP); sd combines the refitted side only'}
        # reader-matched descriptive benefits (review O3): the same recipe on FINALIST and on the comparator; never selection
        rm = {}
        for comp in ('NEURAL_RAW', 'H_LJ', 'TASKONLY', 'B2TWIN', 'PUB_IBSI'):
            for origin, names_, sc_ in (('source', ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP'), src_sc),
                                        ('new', ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1'), enh_sc)):
                for n_ in names_:
                    a_k, f_k = f'PAIR__{comp}__{n_}', f'PAIR__FINALIST__{n_}'
                    if a_k in sc_ and f_k in sc_:
                        dr_ = bootstrap_auc(sex, sc_[a_k], W) - bootstrap_auc(sex, sc_[f_k], W)
                        pt_ = auc(sex, sc_[a_k]) - auc(sex, sc_[f_k]); se_ = float(np.std(dr_, ddof=1))
                        rm[f'{comp}|{n_}'] = {'origin': origin, 'benefit': pt_, 'se': se_, 'marginal95': [pt_ - cm * se_, pt_ + cm * se_]}
        srec['reader_matched_benefits_descriptive'] = rm
        srec['near_ties'] = {svc: {v: {k: enh['winners'][svc]['enhanced'][v][k] for k in ('AV_NEAR_TIE', 'gap_to_runner_up', 'gap_to_original_winner', 'same_as_original')}
                                   for v in VIEWS} for svc in enh['winners']}
        srec['unseen_income_vectors'] = {k: d['unseen_income_vectors'] for k, d in enh['designs'].items() if d.get('unseen_income_vectors')}
        srec['not_converged'] = enh['not_converged']
        srec['controls_out_hgb_vs_source_bitwise'] = enh['controls_out_hgb_vs_source_bitwise']
        res['per_seed'][f'S{s}'] = srec

    bands = src_lock['control_validity_bands']
    def in_band(c):
        return (c['planted'] >= bands['planted_bit_auc_min'] and abs(c['shuffled'] - 0.5) <= bands['shuffled_null_auc_abs_minus_half_max']
                and c['av_planted'] >= bands['planted_bit_auc_min'] and abs(c['av_shuffled'] - 0.5) <= bands['shuffled_null_auc_abs_minus_half_max'])
    diag = {f'S{s}|{k}': in_band(c) for s in C.SEEDS for k, c in res['per_seed'][f'S{s}']['controls_diagnostic'].items()}
    out_hgb_this = {f'S{s}|OUT_HGB|{v}': in_band(res['per_seed'][f'S{s}']['controls_cv'][f'this_run|{v}']) for s in C.SEEDS for v in VIEWS}
    CV = {'ORIGINAL': all(in_band(res['per_seed'][f'S{s}']['controls_cv'][f'source_run|{v}']) for s in C.SEEDS for v in VIEWS),
          'ENHANCED': all(out_hgb_this.values()) and all(diag.values())}
    res['control_validity'] = {'CV': CV, 'bands': bands, 'ENHANCED_controls_in_band': {**out_hgb_this, **diag},
                               'rule': 'ORIGINAL: source-run OUT_HGB controls; ENHANCED: all 5 control recipes x 3 views x 2 seeds on ATTACK_VALIDATION and DEV_REPORT (review A5)'}

    # ---- endpoints: the source roster restricted to this bank, evaluated per scorecard
    wires = {}
    for s in C.SEEDS:
        b_s = C.service_bank(s)[0]
        wires[s] = {svc: {t: C.wire_sha(b_s[svc]['DEV_REPORT'][t]) for t in C.TASKS} for svc in C.BANK}
    roster = [e for e in src_lock['endpoints'] if not any(r in (e.get('a'), e.get('b'), e.get('service')) for r in OUT_OF_BANK)]
    scorecards, draws_out = {}, {}
    for tag, SC in (('orig', 'ORIGINAL'), ('enh', 'ENHANCED')):
        eps = {}
        for e in roster:
            s = e['seed']; fam = e['family']; identical = False
            if e['type'] == 'auc':
                a, b = e['a'], e['b']
                point = auc_pt[(a, s, e['view'], tag)] - auc_pt[(b, s, e['view'], tag)]; draws = auc_dr[(a, s, e['view'], tag)] - auc_dr[(b, s, e['view'], tag)]
                identical = a == b or (e['view'] in C.TASKS and wires[s][a][e['view']] == wires[s][b][e['view']])
            elif e['type'] == 'auc_views':
                a = e['service']
                point = auc_pt[(a, s, e['plus'], tag)] - auc_pt[(a, s, e['minus'], tag)]; draws = auc_dr[(a, s, e['plus'], tag)] - auc_dr[(a, s, e['minus'], tag)]
            elif e['type'] == 'utility':
                point, draws = util[(e['service'], s, e['task'], e['ref'], e['metric'])]
            elif e['type'] == 'guard':
                point, draws = util[(e['service'], s, 'guard', e['group'])]
            elif e['type'] == 'loss_diff':
                d = per_vec[(e['a'], s, e['task'])][e['metric']] - per_vec[(e['b'], s, e['task'])][e['metric']]
                point, draws = float(d.mean()), (W * d[None, :]).sum(1) / Wsum
                identical = wires[s][e['a']][e['task']] == wires[s][e['b']][e['task']]
            else:
                raise ValueError('unregistered endpoint type')
            if identical:
                rec = {'point': 0.0, 'se': 0.0, 'kind': e['kind'], 'margin': e['margin'], 'family_lower': 0.0, 'family_upper': 0.0,
                       'clause_L1': 0.0 if fam == 'P' else None, 'clause_U1': 0.0 if fam == 'P' else None,
                       'status': 'PASS' if e['kind'] == 'no_excess' else 'IDENTICAL_RELEASE', 'category': 'PASS' if e['kind'] == 'no_excess' else 'IDENTICAL_RELEASE',
                       'point_pass': e['kind'] == 'no_excess', 'clause_pass': e['kind'] == 'no_excess', 'at': {}}
                draws = np.zeros(len(Wsum))
            else:
                rec = endpoint(point, draws, e['kind'], e['margin'], c1, cK[fam], e.get('refs', ()))
                if fam != 'P':
                    rec['clause_L1'] = rec['clause_U1'] = rec['clause_pass'] = None
            flags = list(e.get('flags', [])) + (['IDENTICAL_RELEASE'] if identical and 'IDENTICAL_RELEASE' not in e.get('flags', []) else [])
            rec.update({'family': fam, 'claim': e.get('claim'), 'flags': flags, 'availability': 'COMPARATOR_INELIGIBLE' if 'COMPARATOR_INELIGIBLE' in flags else 'AVAILABLE',
                        'spec': {k: v for k, v in e.items() if k != 'flags'}})
            eps[e['key']] = rec; draws_out[f'{SC}::{e["key"]}'] = np.asarray(draws, dtype=np.float64)
        claims = evaluate_claims(src_lock['claims'], eps, CV[SC], src_lock['availability_ms_point_eligible'])
        P = {k: v for k, v in eps.items() if v['family'] == 'P'}
        uk = [k for k in P if k.startswith(('UTILITY|', 'SUBGROUP|'))]; rk = [k for k in P if k.startswith('COMPETITIVE|') and '_worsening_' in k]
        fam_status = [P[k]['status'] for k in uk]
        scorecards[SC] = {'endpoints': eps, 'claims': claims, 'CV': CV[SC],
                          'E_fields': {'dev_point_status': {'utility_24': sum(eps[k]['point'] <= eps[k]['margin'] for k in uk if k.startswith('UTILITY|')),
                                                            'guards_4': sum(eps[k]['point'] <= eps[k]['margin'] for k in uk if k.startswith('SUBGROUP|')),
                                                            'recipient_8': sum(eps[k]['point'] <= eps[k]['margin'] for k in rk)},
                                       'simultaneous_utility_confidence': ('CONFIDENCE_FEASIBLE' if all(x == 'PASS' for x in fam_status) else
                                                                           'ESTABLISHED_UTILITY_VIOLATION' if any(x == 'ESTABLISHED_VIOLATION' for x in fam_status) else 'CONFIDENCE_UNRESOLVED'),
                                       'simultaneous_utility_counts': {x: fam_status.count(x) for x in sorted(set(fam_status))}}}
    res['scorecards'] = scorecards
    # ---- reproduction of the source report by the ORIGINAL scorecard
    src_rep = json.loads((C.SRC_RESULTS / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    diffs = {k: abs(scorecards['ORIGINAL']['endpoints'][k]['point'] - src_rep['endpoints'][k]['point']) for k in scorecards['ORIGINAL']['endpoints']}
    sed = {k: abs(scorecards['ORIGINAL']['endpoints'][k]['se'] - src_rep['endpoints'][k]['se']) for k in scorecards['ORIGINAL']['endpoints']}
    res['original_reproduces_source'] = {'max_abs_point_diff': max(diffs.values()), 'max_abs_se_diff': max(sed.values()),
                                         'claim_labels_equal': {n: scorecards['ORIGINAL']['claims'][n]['label'] == src_rep['claims'][n]['label'] for n in src_rep['claims']},
                                         'nested_0020_equal': scorecards['ORIGINAL']['claims']['B']['nested_0020']['label'] == src_rep['claims']['B']['nested_0020']['label'],
                                         'endpoints_compared': len(diffs)}
    # ---- benefit changes and interpretation labels
    bc = {}
    for k in scorecards['ENHANCED']['endpoints']:
        if '|pair_benefit_vs_' in k:
            d = draws_out[f'ENHANCED::{k}'] - draws_out[f'ORIGINAL::{k}']
            pt = scorecards['ENHANCED']['endpoints'][k]['point'] - scorecards['ORIGINAL']['endpoints'][k]['point']; se = float(np.std(d, ddof=1))
            bc[k] = {'enhanced_minus_original_benefit': pt, 'se': se, 'marginal95': [pt - cm * se, pt + cm * se]}
    res['benefit_change'] = bc
    def side(cond):
        """Side attribution of a failing reader-dependent condition from the paired reader-class changes (review B1)."""
        if not cond or not isinstance(cond.get('condition'), str) or '|' not in cond['condition']:
            return None
        key = cond['condition']; spec = scorecards['ENHANCED']['endpoints'].get(key, {}).get('spec', {})
        if spec.get('type') != 'auc':
            return None
        s_, view = spec['seed'], spec['view']
        ch = {r: auc_pt[(r, s_, view, 'enh')] - auc_pt[(r, s_, view, 'orig')] for r in (spec['a'], spec['b'])}
        fin = 'FINALIST'; comp = spec['a'] if spec['b'] == fin else spec['b']
        up, down = ch[fin] > 0, ch[comp] < 0
        lab = 'BOTH' if (up and down) else 'FINALIST_READ_HIGHER' if up else 'COMPARATOR_READ_LOWER' if down else 'NEITHER'
        return {'label': lab, 'finalist_change': ch[fin], 'comparator_change': ch[comp],
                'note': 'COMPARATOR_READ_LOWER is ATTACK_VALIDATION selection instability, not added finalist exposure'}

    def interp_one(o_holds, e_holds, o_label, src_label, e_first):
        if o_label != src_label:
            return 'DISCREPANCY', None
        if not CV['ENHANCED']:
            return 'ENHANCED_NOT_INTERPRETABLE', None
        if o_holds and e_holds:
            return 'SURVIVED_THIS_READER_BANK', None
        if o_holds and not e_holds:
            return 'READER_SENSITIVE', side(e_first)
        if e_holds:
            return 'ENHANCED_ONLY_PASS', None
        return 'NOT_ESTABLISHED_EITHER_BANK', None
    interp = {}
    for n in ('B', 'J'):
        o, e_ = scorecards['ORIGINAL']['claims'][n], scorecards['ENHANCED']['claims'][n]
        lab, sd_ = interp_one(o['holds'], e_['holds'], o['label'], src_rep['claims'][n]['label'], e_.get('first_failing'))
        interp[n] = {'original_label': o['label'], 'enhanced_label': e_['label'], 'interpretation': lab, 'side': sd_,
                     'enhanced_first_failing': e_.get('first_failing')}
    ob, eb = scorecards['ORIGINAL']['claims']['B']['nested_0020'], scorecards['ENHANCED']['claims']['B']['nested_0020']
    lab, sd_ = interp_one(ob['label'] != 'NOT_ESTABLISHED', eb['label'] != 'NOT_ESTABLISHED', ob['label'], src_rep['claims']['B']['nested_0020']['label'],
                          next((c for c in eb['conditions'] if not c['holds']), None))
    interp['B_nested_0020'] = {'original_label': ob['label'], 'enhanced_label': eb['label'], 'interpretation': lab, 'side': sd_}
    om, em = scorecards['ORIGINAL']['claims']['M'], scorecards['ENHANCED']['claims']['M']
    steps = {}
    for so, se_ in zip(om['steps'], em['steps']):
        first = se_.get('first_failing')
        lab, sd_ = interp_one(so['holds'], se_['holds'], om['label'], src_rep['claims']['M']['label'], first)
        steps[so['name']] = {'original_holds': so['holds'], 'enhanced_holds': se_['holds'], 'interpretation': lab, 'side': sd_}
    lab, _ = interp_one(om['holds'], em['holds'], om['label'], src_rep['claims']['M']['label'], None)
    interp['M'] = {'original_label': om['label'], 'enhanced_label': em['label'], 'interpretation': lab, 'steps': steps}
    res['interpretation'] = interp
    # ---- private: per-replicate endpoint draws (feasibility) and per-household FINALIST utility contributions
    priv.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(priv / 'ENDPOINT_DRAWS.npz', 'xb') as f:
        np.savez(f, keys=np.array(sorted(draws_out)), draws=np.column_stack([draws_out[k] for k in sorted(draws_out)]))
    with open(priv / 'HOUSEHOLD_CONTRIBUTIONS.npz', 'xb') as f:
        np.savez(f, household_count=np.bincount(hh_inv).astype(np.float64), **{k.replace('|', '__'): v for k, v in hh_contrib.items()})
    res['private_feasibility_inputs_sha256'] = {n: C.sha(priv / n) for n in ('ENDPOINT_DRAWS.npz', 'HOUSEHOLD_CONTRIBUTIONS.npz')}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'DEV_REPORT_RESULTS.json', 'x') as f:
        json.dump(res, f, indent=1, sort_keys=True, default=float)
    print(json.dumps({'interpretation': interp, 'CV': CV, 'original_reproduces_source': res['original_reproduces_source']}, indent=1, default=float))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    a = ap.parse_args(); run(a.out)
