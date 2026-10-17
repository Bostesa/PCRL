"""Unit VER_DEVREPORT_V1: the ONE logged independent-verifier DEV_REPORT replay (checks 6 and 7, plus every other
label-dependent number this verification needs). Event: DEV_REPORT_TRUTH_REREAD_BY_FFV_INDEPENDENT_VERIFIER.

Public-use ACS 2019 Florida person records, development roles; DEV_REPORT is an already-exposed role (development robustness
evidence, not confirmation). Evaluation readers infer the recorded SEX column from the released income and occupation predictions;
utility endpoints compare the frozen finalist's income and occupation predictions with the U0 / Ucal references.

Own code throughout (AUC, household-bootstrap replicates of means and AUCs, endpoint bounds, claim engine, interpretation); the
allowed imports are task_headroom.inference.group_weights (bootstrap weights) and the admitted loaders. The enhanced winners are
re-derived here from the candidate AUCs (source-first order, exact ties to the earlier candidate).

Outputs (private, aggregate only; no person-level value): verifier/dev_results.json and verifier/dev_endpoint_draws.npz
(per-replicate endpoint draws for check 8). Comparisons with the coordinator's report are made afterwards by v6b_compare.py
without any label. `--invented` runs the identical pipeline on invented labels (no truth read) into verifier/dryrun/.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import vcommon as V
from ffv import common as C                       # admitted loaders and the one logged truth access
from task_headroom.inference import group_weights  # allowed by the brief

EVENT = 'DEV_REPORT_TRUTH_REREAD_BY_FFV_INDEPENDENT_VERIFIER'
PURPOSE = ('frozen-finalist validation: the one separately logged independent-verifier replay (checks 6 and 7: selection, primary '
           'endpoints, claim labels, utility and subgroup headroom) of the frozen finalist, bank and enhanced readers')
OUT_OF_BANK = ('H_JJ', 'T_TREE')
B_BOOT, BOOT_SEED = 2048, 2026101705


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def task_vectors(y, p, k):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), k) or not np.allclose(p.sum(1), 1.0, atol=1e-9, rtol=0):
        raise ValueError('probability vectors differ from the convention')
    tgt = (y[:, None] == np.arange(k)[None, :]).astype(np.float64)
    return {'accuracy': (np.argmax(p, axis=1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
            'brier': ((p - tgt) ** 2).sum(1)}


class Endpoints:
    def __init__(self, c1, cK):
        self.c1, self.cK = c1, cK

    def make(self, point, draws, kind, margin, fam, identical=False):
        if identical:
            r = {'point': 0.0, 'se': 0.0, 'family_lower': 0.0, 'family_upper': 0.0, 'clause_L1': 0.0 if fam == 'P' else None,
                 'clause_U1': 0.0 if fam == 'P' else None, 'status': 'PASS' if kind == 'no_excess' else 'IDENTICAL_RELEASE',
                 'category': 'PASS' if kind == 'no_excess' else 'IDENTICAL_RELEASE', 'point_pass': kind == 'no_excess',
                 'clause_pass': kind == 'no_excess' if fam == 'P' else None}
            return r, np.zeros(B_BOOT)
        draws = np.asarray(draws, dtype=np.float64)
        se = float(np.std(draws, ddof=1))
        cK = self.cK[fam]
        L, U = point - cK * se, point + cK * se
        L1, U1 = point - self.c1 * se, point + self.c1 * se
        if kind == 'benefit':
            status = 'ESTABLISHED' if L > margin else ('ESTABLISHED_SHORTFALL' if U < margin else 'UNRESOLVED')
            pp, cp = point > margin, L1 > margin
            cat = 'PASS' if cp else ('ESTABLISHED_SHORTFALL' if U < margin else ('POINT_SHORTFALL' if not pp else 'UNRESOLVED'))
        elif kind == 'no_excess':
            status = 'PASS' if U <= margin else ('ESTABLISHED_VIOLATION' if L > margin else 'UNRESOLVED')
            pp, cp = point <= margin, U1 <= margin
            cat = 'PASS' if cp else ('ESTABLISHED_VIOLATION' if L > margin else ('POINT_EXCESS' if not pp else 'UNRESOLVED'))
        else:
            status = 'ABOVE_ZERO' if L > margin else ('BELOW_ZERO' if U < margin else 'UNRESOLVED')
            pp = cp = None
            cat = 'DESCRIPTIVE'
        r = {'point': float(point), 'se': se, 'family_lower': float(L), 'family_upper': float(U), 'status': status, 'category': cat,
             'point_pass': None if pp is None else bool(pp)}
        if fam == 'P':
            r.update({'clause_L1': float(L1), 'clause_U1': float(U1), 'clause_pass': None if cp is None else bool(cp)})
        else:
            r.update({'clause_L1': None, 'clause_U1': None, 'clause_pass': None})
        return r, draws


def claim_engine(claims_spec, eps, CV, avail):
    def cond(c, done):
        if 'claim_ref' in c:
            return {'condition': f"claim {c['claim_ref']}", 'holds': bool(done[c['claim_ref']]['holds'])}
        if c['test'] == 'cv':
            return {'condition': 'CV', 'holds': bool(CV)}
        if c['test'] == 'av':
            return {'condition': f"AV {c['role']}", 'holds': bool(c['role'] == 'TASKONLY' or avail.get(c['role']))}
        e = eps[c['endpoint']]
        if c['test'] == 'point_le':
            ok = e['point'] <= e['margin']
        elif c['test'] == 'clause':
            ok = bool(e['clause_pass']) and e['availability'] == 'AVAILABLE'
        elif c['test'] == 'clause_at_0020':
            ok = e['clause_L1'] is not None and e['clause_L1'] > 0.020
        else:
            raise ValueError(c['test'])
        return {'condition': c['endpoint'], 'test': c['test'], 'gate': c.get('gate'), 'holds': bool(ok)}

    out = {}
    for cl in claims_spec:
        n = cl['name']
        if cl['kind'] in ('not_claimed', 'descriptive'):
            out[n] = {'label': cl['label'], 'holds': None}
        elif cl['kind'] == 'gate':
            cs = [cond(c, out) for c in cl['conditions']]
            ok = all(c['holds'] for c in cs)
            out[n] = {'label': cl['success_label'] if ok else cl['failure_label'], 'holds': ok,
                      'failed_gates': sorted({c['gate'] for c in cs if not c['holds']}), 'n_conditions': len(cs)}
        elif cl['kind'] == 'fixed_sequence':
            steps, stop = [], False
            for st in cl['steps']:
                if stop:
                    steps.append({'name': st['name'], 'holds': False, 'status': 'NOT_TESTED_SEQUENCE_STOPPED'})
                    continue
                cs = [cond(c, out) for c in st['conditions']]
                ok = all(c['holds'] for c in cs)
                steps.append({'name': st['name'], 'holds': ok, 'first_failing': next((c for c in cs if not c['holds']), None)})
                stop = not ok
            lab = cl['labels']['both'] if all(x['holds'] for x in steps) else (cl['labels']['to_only'] if steps[0]['holds'] else cl['labels']['none'])
            out[n] = {'label': lab, 'holds': lab == cl['labels']['both'], 'steps': steps}
        else:
            cs = [cond(c, out) for c in cl['conditions']]
            ok = all(c['holds'] for c in cs)
            first = next((c for c in cs if not c['holds']), None)
            if n == 'B':
                if ok:
                    lab = cl['success_label']
                elif not CV:
                    lab = 'CONTROL_FAILURE'
                elif not out['E']['holds']:
                    g = set(out['E']['failed_gates'])
                    lab = 'FINALIST_FAILS_DEV_UTILITY_GATE' if 'UG' in g else ('FINALIST_FAILS_DEV_SUBGROUP_POINT_GUARD' if 'SG' in g else 'FINALIST_FAILS_DEV_RECIPIENT_POINT_GATE')
                else:
                    lab = 'NOT_SUPERIOR_TO_ALL_ELIGIBLE'
                nested = [cond(c, out) for c in cl['nested']['adds']]
                out[n] = {'label': lab, 'holds': ok, 'first_failing': first,
                          'nested_0020': {'label': cl['nested']['name'] if (ok and all(c['holds'] for c in nested)) else 'NOT_ESTABLISHED',
                                          'conditions': nested}}
            else:
                out[n] = {'label': cl['success_label'] if ok else cl['failure_label'], 'holds': ok, 'first_failing': first}
    return out


def run(T, out_dir, invented):
    res = {'unit': 'VER_DEVREPORT_V1' if not invented else 'VER_DEVREPORT_DRYRUN', 'invented_labels': invented, 'errors': {}}
    sex = np.asarray(T['sex'], dtype=np.int64)
    y = {'income': np.asarray(T['income'], dtype=np.int64), 'occupation': np.asarray(T['occupation'], dtype=np.int64)}
    W, G = group_weights(T['unit'], seed=BOOT_SEED, draws=B_BOOT)
    if G != 3955 or len(sex) != 5943:
        raise SystemExit(f'HALT: {G} households, {len(sex)} persons')
    df = G - 1
    c1 = V.tcrit(df, 0.025 / 3)
    cK = {'P': V.tcrit(df, 0.05 / (2 * 52)), 'S': V.tcrit(df, 0.05 / (2 * 20))}
    cm = V.tcrit(df, 0.025)
    res['critical'] = {'c1': c1, 'c_P': cK['P'], 'c_S': cK['S'], 'marginal': cm, 'df': df,
                       'equal_registered_within_1e-6': all(abs(a - b) < 1e-6 for a, b in ((c1, 2.3949991), (cK['P'], 3.304031), (cK['S'], 3.025281), (cm, 1.9605641)))}
    res['households'], res['persons'] = int(G), int(len(sex))
    src_lock = V.jload(V.SRC_RES / 'ANALYSIS_LOCK.json')
    rl = V.jload(V.RES / 'REPORTING_LOCK.json')
    E = Endpoints(c1, cK)

    auc_cache = {}

    def auc_of(tag_key, score):
        if tag_key not in auc_cache:
            auc_cache[tag_key] = (V.auc(sex, score), V.auc_weighted(sex, score, W))
        return auc_cache[tag_key]

    data = {}
    winners = {}
    auc_pt, auc_dr = {}, {}
    per_vec, util = {}, {}
    controls = {'ORIGINAL': {}, 'ENHANCED': {}}
    for s in V.SEEDS:
        bank, ids, _ = C.service_bank(s)
        frozen, fids = C.frozen_bank(s)
        if not (np.array_equal(ids['row_id__DEV_REPORT'], T['row_id']) and np.array_equal(fids['row_id__DEV_REPORT'], T['row_id'])):
            raise AssertionError('bank row order differs')
        src_rec, src_sc = C.source_readers(s)
        ed = V.RUNS / f'READERS_ENH_S{s}'
        for f, k in (('DEV_REPORT_SCORES.npz', 'dev_report_scores_sha256'), ('AV_SCORES.npz', 'av_scores_sha256'), ('READERS_ENH_SEED.json', 'record_sha256')):
            if V.sha_file(ed / f) != rl['enhanced_readers'][f'S{s}'][k]:
                raise PermissionError(f'{f} differs from REPORTING_LOCK')
        enh = V.jload(ed / 'READERS_ENH_SEED.json')
        enh_sc = load_npz(ed / 'DEV_REPORT_SCORES.npz')
        if not (np.array_equal(src_sc['row_id'], T['row_id']) and np.array_equal(enh_sc['row_id'], T['row_id'])):
            raise AssertionError('score row order differs')
        data[s] = {'bank': bank, 'frozen': frozen, 'src_rec': src_rec, 'src_sc': src_sc, 'enh': enh, 'enh_sc': enh_sc}

        def score(key):
            o, k = key.split('::', 1)
            return src_sc[k] if o == 'SRC' else enh_sc[k]
        # winners: original (source record) and enhanced (re-derived here)
        for svc in V.BANK:
            for view in V.VIEWS:
                ow = src_rec['services'][svc]['winner'][view]
                pe = V.pick_max_first(V.enhanced_candidates(src_rec, enh, svc, view))
                winners[(svc, s, view)] = {'orig_key': 'SRC::' + ow['reader_id'], 'orig': ow['candidate'], 'enh_key': pe['key'],
                                           'enh': pe['candidate'] + ('' if pe['origin'] == 'source' else ' [new]'), 'enh_av_auc': pe['auc'],
                                           'orig_av_auc': ow.get('attack_validation_auc'), 'near_tie': pe['near_tie'], 'gap_to_runner_up': pe['gap_to_runner_up']}
                for tag, key in (('orig', 'SRC::' + ow['reader_id']), ('enh', pe['key'])):
                    pt, dr = auc_of(f'S{s}|{key}', score(key))
                    auc_pt[(svc, s, view, tag)], auc_dr[(svc, s, view, tag)] = pt, dr
        # task vectors and utility (every bank service; endpoints use FINALIST)
        refs = {r: {t: frozen[r]['DEV_REPORT'][t] for t in V.TASKS} for r in ('U0', 'Ucal')}
        rv = {r: {t: task_vectors(y[t], refs[r][t], V.K[t]) for t in V.TASKS} for r in refs}
        for svc in V.BANK:
            for t in V.TASKS:
                vv = task_vectors(y[t], bank[svc]['DEV_REPORT'][t], V.K[t])
                per_vec[(svc, s, t)] = vv
                for r in ('U0', 'Ucal'):
                    for metric, d in (('accuracy_loss', rv[r][t]['accuracy'] - vv['accuracy']), ('nll_excess', vv['nll'] - rv[r][t]['nll']),
                                      ('brier_excess', vv['brier'] - rv[r][t]['brier'])):
                        util[(svc, s, t, r, metric)] = (float(d.mean()), V.ratio_draws(d, W))
            d = per_vec[(svc, s, 'occupation')]['brier'] - rv['U0']['occupation']['brier']
            for g in (0, 1):
                m = sex == g
                util[(svc, s, 'guard', g)] = (float(d[m].mean()), V.ratio_draws(d, W, m))
        # controls
        bands = src_lock['control_validity_bands']
        for view in V.VIEWS:
            sh = V.auc(sex, src_sc[f'CONTROL__{view}__SHUFFLED'])
            pl = V.auc(sex, np.where(sex == 1, src_sc[f'CONTROL__{view}__PLANTED_1'], src_sc[f'CONTROL__{view}__PLANTED_0']))
            av = src_rec['controls'][view]
            controls['ORIGINAL'][f'S{s}|OUT_HGB|{view}'] = {'dev_shuffled': sh, 'dev_planted': pl, 'av_shuffled': av['shuffled_attack_validation_auc'],
                                                            'av_planted': av['planted_attack_validation_auc']}
            for name, cr in enh['controls'][view].items():
                k = f'CONTROL__{view}__{name}'
                sh = V.auc(sex, enh_sc[k + '__SHUFFLED'])
                pl = V.auc(sex, np.where(sex == 1, enh_sc[k + '__PLANTED_1'], enh_sc[k + '__PLANTED_0']))
                controls['ENHANCED'][f'S{s}|{name}|{view}'] = {'dev_shuffled': sh, 'dev_planted': pl, 'av_shuffled': cr['shuffled_attack_validation_auc'],
                                                               'av_planted': cr['planted_attack_validation_auc']}
    bands = src_lock['control_validity_bands']
    for sc_ in controls.values():
        for c in sc_.values():
            c['in_band'] = bool(c['dev_planted'] >= bands['planted_bit_auc_min'] and abs(c['dev_shuffled'] - 0.5) <= bands['shuffled_null_auc_abs_minus_half_max']
                                and c['av_planted'] >= bands['planted_bit_auc_min'] and abs(c['av_shuffled'] - 0.5) <= bands['shuffled_null_auc_abs_minus_half_max'])
    CV = {k: all(c['in_band'] for c in v.values()) for k, v in controls.items()}
    res['controls'] = controls
    res['CV'] = CV
    res['winners'] = {f'{a}|S{b}|{c}': v for (a, b, c), v in winners.items()}
    res['recovery'] = {f'{a}|S{b}|{c}|{t}': {'auc': auc_pt[(a, b, c, t)], 'se': float(np.std(auc_dr[(a, b, c, t)], ddof=1))} for (a, b, c, t) in auc_pt}

    # ---------------------------------------------------------------- endpoints, both scorecards
    wires = {s: {svc: {t: V.wire_hash(data[s]['bank'][svc]['DEV_REPORT'][t]) for t in V.TASKS} for svc in V.BANK} for s in V.SEEDS}
    roster = [e for e in src_lock['endpoints'] if not any(r in (e.get('a'), e.get('b'), e.get('service')) for r in OUT_OF_BANK)]
    res['roster_size'] = len(roster)
    scorecards, keys_out, draws_out = {}, [], []
    avail = src_lock['availability_ms_point_eligible']
    for tag, SC in (('orig', 'ORIGINAL'), ('enh', 'ENHANCED')):
        eps = {}
        for e in roster:
            s, fam, ident = e['seed'], e['family'], False
            if e['type'] == 'auc':
                a, b = e['a'], e['b']
                point = auc_pt[(a, s, e['view'], tag)] - auc_pt[(b, s, e['view'], tag)]
                dr = auc_dr[(a, s, e['view'], tag)] - auc_dr[(b, s, e['view'], tag)]
                ident = a == b or (e['view'] in V.TASKS and wires[s][a][e['view']] == wires[s][b][e['view']])
            elif e['type'] == 'auc_views':
                a = e['service']
                point = auc_pt[(a, s, e['plus'], tag)] - auc_pt[(a, s, e['minus'], tag)]
                dr = auc_dr[(a, s, e['plus'], tag)] - auc_dr[(a, s, e['minus'], tag)]
            elif e['type'] == 'utility':
                point, dr = util[(e['service'], s, e['task'], e['ref'], e['metric'])]
            elif e['type'] == 'guard':
                point, dr = util[(e['service'], s, 'guard', e['group'])]
            elif e['type'] == 'loss_diff':
                d = per_vec[(e['a'], s, e['task'])][e['metric']] - per_vec[(e['b'], s, e['task'])][e['metric']]
                point, dr = float(d.mean()), V.ratio_draws(d, W)
                ident = wires[s][e['a']][e['task']] == wires[s][e['b']][e['task']]
            else:
                raise ValueError(e['type'])
            r, dr2 = E.make(point, dr, e['kind'], e['margin'], fam, ident)
            flags = list(e.get('flags', [])) + (['IDENTICAL_RELEASE'] if ident and 'IDENTICAL_RELEASE' not in e.get('flags', []) else [])
            r.update({'kind': e['kind'], 'margin': e['margin'], 'family': fam, 'flags': flags,
                      'availability': 'COMPARATOR_INELIGIBLE' if 'COMPARATOR_INELIGIBLE' in flags else 'AVAILABLE', 'identical': ident})
            eps[e['key']] = r
            keys_out.append(f'{SC}::{e["key"]}')
            draws_out.append(dr2)
        claims = claim_engine(src_lock['claims'], eps, CV[SC], avail)
        P = {k: v for k, v in eps.items() if v['family'] == 'P'}
        uk = [k for k in P if k.startswith(('UTILITY|', 'SUBGROUP|'))]
        rk = [k for k in P if k.startswith('COMPETITIVE|') and '_worsening_' in k]
        fs = [P[k]['status'] for k in uk]
        scorecards[SC] = {'endpoints': eps, 'claims': claims, 'CV': CV[SC],
                          'E_fields': {'dev_point_status': {'utility_24': int(sum(eps[k]['point'] <= eps[k]['margin'] for k in uk if k.startswith('UTILITY|'))),
                                                            'guards_4': int(sum(eps[k]['point'] <= eps[k]['margin'] for k in uk if k.startswith('SUBGROUP|'))),
                                                            'recipient_8': int(sum(eps[k]['point'] <= eps[k]['margin'] for k in rk))},
                                       'simultaneous_utility_confidence': ('CONFIDENCE_FEASIBLE' if all(x == 'PASS' for x in fs) else
                                                                           'ESTABLISHED_UTILITY_VIOLATION' if any(x == 'ESTABLISHED_VIOLATION' for x in fs) else 'CONFIDENCE_UNRESOLVED'),
                                       'simultaneous_utility_counts': {x: fs.count(x) for x in sorted(set(fs))}}}
    res['scorecards'] = scorecards
    D = np.column_stack(draws_out)
    V.VPRIV.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'dev_endpoint_draws.npz', 'wb') as fh:
        np.savez(fh, keys=np.array(keys_out), draws=D)
    V.jdump(out_dir / 'dev_results.json', res)            # checkpoint: the core results are saved before the descriptive parts

    # ---------------------------------------------------------------- interpretation (PROTOCOL section 8 table) and side attribution
    try:
        src_rep = V.jload(V.SRC_RES / 'report' / 'DEV_REPORT_RESULTS.json')

        def side(first):
            if not first or first.get('condition') not in scorecards['ENHANCED']['endpoints']:
                return None
            spec = next(e for e in roster if e['key'] == first['condition'])
            if spec['type'] != 'auc':
                return None
            s_, v_ = spec['seed'], spec['view']
            ch = {r: auc_pt[(r, s_, v_, 'enh')] - auc_pt[(r, s_, v_, 'orig')] for r in (spec['a'], spec['b'])}
            comp = spec['a'] if spec['b'] == 'FINALIST' else spec['b']
            up, down = ch['FINALIST'] > 0, ch[comp] < 0
            return {'label': 'BOTH' if (up and down) else 'FINALIST_READ_HIGHER' if up else 'COMPARATOR_READ_LOWER' if down else 'NEITHER',
                    'finalist_change': ch['FINALIST'], 'comparator_change': ch[comp], 'condition': first['condition']}

        def interp(o_holds, e_holds, o_label, src_label, first):
            if o_label != src_label:
                return 'DISCREPANCY', None
            if not CV['ENHANCED']:
                return 'ENHANCED_NOT_INTERPRETABLE', None
            if o_holds and e_holds:
                return 'SURVIVED_THIS_READER_BANK', None
            if o_holds:
                return 'READER_SENSITIVE', side(first)
            if e_holds:
                return 'ENHANCED_ONLY_PASS', None
            return 'NOT_ESTABLISHED_EITHER_BANK', None
        O, En = scorecards['ORIGINAL']['claims'], scorecards['ENHANCED']['claims']
        it = {}
        for n in ('B', 'J'):
            lab, sd = interp(O[n]['holds'], En[n]['holds'], O[n]['label'], src_rep['claims'][n]['label'], En[n].get('first_failing'))
            it[n] = {'original_label': O[n]['label'], 'enhanced_label': En[n]['label'], 'interpretation': lab, 'side': sd,
                     'enhanced_first_failing': (En[n].get('first_failing') or {}).get('condition')}
        ob, eb = O['B']['nested_0020'], En['B']['nested_0020']
        lab, sd = interp(ob['label'] != 'NOT_ESTABLISHED', eb['label'] != 'NOT_ESTABLISHED', ob['label'], src_rep['claims']['B']['nested_0020']['label'],
                         next((c for c in eb['conditions'] if not c['holds']), None))
        it['B_nested_0020'] = {'original_label': ob['label'], 'enhanced_label': eb['label'], 'interpretation': lab, 'side': sd}
        steps = {}
        for so, se_ in zip(O['M']['steps'], En['M']['steps']):
            lab, sd = interp(so['holds'], se_['holds'], O['M']['label'], src_rep['claims']['M']['label'], se_.get('first_failing'))
            steps[so['name']] = {'original_holds': so['holds'], 'enhanced_holds': se_['holds'], 'interpretation': lab, 'side': sd,
                                 'enhanced_first_failing': (se_.get('first_failing') or {}).get('condition')}
        lab, _ = interp(O['M']['holds'], En['M']['holds'], O['M']['label'], src_rep['claims']['M']['label'], None)
        it['M'] = {'original_label': O['M']['label'], 'enhanced_label': En['M']['label'], 'interpretation': lab, 'steps': steps}
        res['interpretation'] = it
        res['original_vs_source_report'] = {
            'max_abs_point_diff': max(abs(scorecards['ORIGINAL']['endpoints'][k]['point'] - src_rep['endpoints'][k]['point']) for k in scorecards['ORIGINAL']['endpoints']),
            'max_abs_se_diff': max(abs(scorecards['ORIGINAL']['endpoints'][k]['se'] - src_rep['endpoints'][k]['se']) for k in scorecards['ORIGINAL']['endpoints']),
            'status_equal': all(scorecards['ORIGINAL']['endpoints'][k]['status'] == src_rep['endpoints'][k]['status'] for k in scorecards['ORIGINAL']['endpoints']),
            'labels_equal': {n: O[n]['label'] == src_rep['claims'][n]['label'] for n in src_rep['claims']},
            'nested_equal': O['B']['nested_0020']['label'] == src_rep['claims']['B']['nested_0020']['label'], 'n': len(scorecards['ORIGINAL']['endpoints'])}
    except Exception:
        res['errors']['interpretation'] = traceback.format_exc()

    # ---------------------------------------------------------------- reader-class changes and benefit changes
    try:
        rc = {}
        for (svc, s, view, tag) in auc_pt:
            if tag != 'enh':
                continue
            d = auc_dr[(svc, s, view, 'enh')] - auc_dr[(svc, s, view, 'orig')]
            pt = auc_pt[(svc, s, view, 'enh')] - auc_pt[(svc, s, view, 'orig')]
            rc[f'{svc}|S{s}|{view}'] = {'enhanced_minus_original': pt, 'se': float(np.std(d, ddof=1)),
                                        'same_reader': winners[(svc, s, view)]['enh_key'] == winners[(svc, s, view)]['orig_key']}
        res['reader_class_change'] = rc
        idx = {k: i for i, k in enumerate(keys_out)}
        bc = {}
        for k in scorecards['ENHANCED']['endpoints']:
            if '|pair_benefit_vs_' in k:
                d = D[:, idx[f'ENHANCED::{k}']] - D[:, idx[f'ORIGINAL::{k}']]
                bc[k] = {'enhanced_minus_original_benefit': scorecards['ENHANCED']['endpoints'][k]['point'] - scorecards['ORIGINAL']['endpoints'][k]['point'],
                         'se': float(np.std(d, ddof=1))}
        res['benefit_change'] = bc
    except Exception:
        res['errors']['changes'] = traceback.format_exc()

    # ---------------------------------------------------------------- check 7 detail: utility and subgroup headroom of FINALIST
    try:
        uh = {}
        for k, e in scorecards['ENHANCED']['endpoints'].items():
            if k.startswith(('UTILITY|', 'SUBGROUP|')):
                slack = e['margin'] - e['point']
                uh[k] = {'point': e['point'], 'se': e['se'], 'cap': e['margin'], 'slack': slack, 'slack_over_se': slack / e['se'] if e['se'] > 0 else None,
                         'clause_U1': e['clause_U1'], 'U1_le_cap': e['clause_U1'] <= e['margin'], 'family_status': e['status'], 'point_pass': e['point'] <= e['margin']}
        order = sorted(uh, key=lambda k: uh[k]['slack_over_se'])
        res['utility_headroom'] = {'endpoints': uh, 'n_utility': sum(k.startswith('UTILITY|') for k in uh), 'n_guards': sum(k.startswith('SUBGROUP|') for k in uh),
                                   'ascending_slack_over_se': [[k, uh[k]['slack_over_se']] for k in order[:8]],
                                   'all_point_pass': all(v['point_pass'] for v in uh.values()), 'n_U1_le_cap': sum(v['U1_le_cap'] for v in uh.values())}
        su = {}
        for svc in V.BANK:
            for s in V.SEEDS:
                su[f'{svc}|S{s}'] = {'utility_point_pass_12': all(util[(svc, s, t, r, m)][0] <= V.CAPS[m] for t in V.TASKS for r in ('U0', 'Ucal') for m in V.CAPS),
                                     'guard_point_pass_2': all(util[(svc, s, 'guard', g)][0] <= V.GUARD for g in (0, 1)),
                                     'points': {f'{t}|{r}|{m}': util[(svc, s, t, r, m)][0] for t in V.TASKS for r in ('U0', 'Ucal') for m in V.CAPS},
                                     'guards': {f'sex{g}': util[(svc, s, 'guard', g)][0] for g in (0, 1)}}
        res['utility_all_services'] = su
    except Exception:
        res['errors']['utility'] = traceback.format_exc()

    # ---------------------------------------------------------------- reader-matched descriptive benefits (review O3)
    try:
        rm = {}
        for s in V.SEEDS:
            for comp in ('NEURAL_RAW', 'H_LJ', 'TASKONLY', 'B2TWIN', 'PUB_IBSI'):
                for origin, names, sc_ in (('source', V.SRC8, data[s]['src_sc']), ('new', V.NEW8, data[s]['enh_sc'])):
                    for n in names:
                        a, f = f'PAIR__{comp}__{n}', f'PAIR__FINALIST__{n}'
                        if a in sc_ and f in sc_:
                            pa, da = auc_of(f'S{s}|{origin}|{a}', sc_[a])
                            pf, dfn = auc_of(f'S{s}|{origin}|{f}', sc_[f])
                            rm[f'S{s}|{comp}|{n}'] = {'origin': origin, 'benefit': pa - pf, 'se': float(np.std(da - dfn, ddof=1))}
        res['reader_matched_benefits'] = rm
    except Exception:
        res['errors']['reader_matched'] = traceback.format_exc()

    # ---------------------------------------------------------------- reader-refit variation (VAR, AUX) and the refit SDs for check 8
    try:
        var = {}
        for s in V.SEEDS:
            enh, es = data[s]['enh'], data[s]['enh_sc']
            vals = {}
            for kind, recs in (('VAR', enh['variation']), ('AUX', enh['aux_sample_variation'])):
                for svc in recs:
                    v0 = auc_pt[(svc, s, 'pair', 'enh')]
                    vals[(kind, svc)] = [v0] + [V.auc(sex, es[f'{kind}__{svc}__j{j}']) for j in (1, 2, 3, 4) if f'{kind}__{svc}__j{j}' in es]
                    var[f'S{s}|{kind}|{svc}'] = {'dev_auc_by_refit': vals[(kind, svc)], 'sd': float(np.std(vals[(kind, svc)], ddof=1)) if len(vals[(kind, svc)]) > 1 else None}
            for kind in ('VAR', 'AUX'):
                for ref in ('NEURAL_RAW', 'H_LJ'):
                    a, b = vals.get((kind, ref)), vals.get((kind, 'FINALIST'))
                    if a and b and len(a) == len(b) and len(a) > 1:
                        bj = [x - z for x, z in zip(a, b)]
                        var[f'S{s}|{kind}|benefit_vs_{ref}'] = {'benefit_by_refit': bj, 'sd': float(np.std(bj, ddof=1)), 'matched': True}
                    else:
                        sa = float(np.std(a, ddof=1)) if a and len(a) > 1 else 0.0
                        sb = float(np.std(b, ddof=1)) if b and len(b) > 1 else 0.0
                        var[f'S{s}|{kind}|benefit_vs_{ref}'] = {'benefit_by_refit': None, 'sd': float(np.sqrt(sa ** 2 + sb ** 2)), 'matched': False}
        res['variation'] = var
    except Exception:
        res['errors']['variation'] = traceback.format_exc()

    # ---------------------------------------------------------------- check 5, DEV half: AUC of own refits versus stored scores
    try:
        rf = {}
        for s in V.SEEDS:
            p = V.VPRIV / f'refit_S{s}.npz'
            if not p.is_file():
                continue
            mine = load_npz(p)
            if not np.array_equal(mine['row_id'], T['row_id']):
                raise AssertionError('refit row order differs')
            for k, sc_ in mine.items():
                if not k.startswith('REFIT__'):
                    continue
                key = k[len('REFIT__'):]
                stored = data[s]['enh_sc'][key] if key in data[s]['enh_sc'] and not key.endswith(tuple(V.SRC8)) else data[s]['src_sc'][key]
                a_m, a_s = V.auc(sex, sc_), V.auc(sex, stored)
                rf[f'S{s}|{key}'] = {'own_dev_auc': a_m, 'stored_dev_auc': a_s, 'diff': a_m - a_s, 'bitwise': bool(np.array_equal(sc_, stored))}
        res['refit_dev_auc'] = rf
    except Exception:
        res['errors']['refit'] = traceback.format_exc()
    V.jdump(out_dir / 'dev_results.json', res)
    print({'CV': CV, 'labels': {sc: {n: c['label'] for n, c in v['claims'].items()} for sc, v in scorecards.items()},
           'interp': {k: v['interpretation'] for k, v in res.get('interpretation', {}).items()}, 'errors': list(res['errors'])}, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--invented', action='store_true')
    a = ap.parse_args()
    if a.invented:
        F = C.load_role('DEV_REPORT', ('features',))
        rng = np.random.default_rng(424242)
        n = len(F['row_id'])
        T = {'row_id': F['row_id'], 'unit': F['unit'], 'sex': (rng.random(n) < 0.47).astype(np.int64),
             'income': (rng.random(n) < 0.4).astype(np.int64), 'occupation': rng.integers(0, 5, n)}
        run(T, V.VPRIV / 'dryrun', invented=True)
        return
    T = C.dev_report_truth(PURPOSE, EVENT)                 # the ONE logged verifier access (refuses a second use)
    ev = [e for e in C.dev_access_events() if e.get('event') == EVENT]
    V.VPRIV.mkdir(parents=True, exist_ok=True)
    V.jdump(V.VPRIV / 'dev_truth_event.json', {'utc': ev[-1]['utc'] if ev else None, 'purpose': PURPOSE, 'event': EVENT,
                                                'unit_id': ev[-1].get('unit_id') if ev else None,
                                                'reporting_lock_sha256': ev[-1].get('reporting_lock_sha256') if ev else None})
    run(T, V.VPRIV, invented=False)


if __name__ == '__main__':
    main()
