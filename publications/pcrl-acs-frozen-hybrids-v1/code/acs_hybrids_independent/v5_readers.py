"""V5 Reader selection (label-free REPLAY of the saved reader models; no reader is refit here).

Per model seed:
  - hashes of READERS_SEED.json, DEV_REPORT_SCORES.npz and all 119 MODEL__*.joblib files vs REPORTING_LOCK;
  - unique single-recipient wires recomputed from the bank (one marginal bank per unique wire, aliases), model inventory
    (70 marginal + 49 pair) and fit counts (+ 6 controls = 125);
  - each saved model's scaler moments equal the ATTACK_FIT reader-input moments (the input it was fitted on);
  - ATTACK_VALIDATION AUC with fixed orientation recomputed with the verifier's own AUC from each saved model applied to
    the bank's ATTACK_VALIDATION wires; orientation = 1 iff AUC >= 0.5;
  - DEV_REPORT scores replayed from each saved model (oriented) vs the stored DEV_REPORT_SCORES;
  - winners re-derived: max ATTACK_VALIDATION AUC, ties to the earlier candidate (pair readers in slate order, then the
    7 income-only aliases, then the 7 occupation-only aliases); candidate structure (7 + 14) checked;
  - no winner could use DEV_REPORT: the derivation uses only ATTACK_VALIDATION; timing of the first truth re-read;
  - the campaign's stored DEV_REPORT scores vs the restored source scores on the 105 shared wires.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import warnings

import joblib
import numpy as np

from acs_hybrids import common as C   # hash-checked data loading only

from . import vcommon as V

SOURCE_LABEL = {'H_LL': 'LOCAL_TREE', 'H_JJ': 'JOINT_TREE', 'NEURAL_RAW': 'NEURAL_RAW_rho1.0_E2', 'TASK_REFIT': 'TASK_REFIT',
                'T_TREE': 'T_TREE'}


def pick(cands):
    """First candidate attaining the maximum AUC (strictly greater required to replace)."""
    best = 0
    for i in range(1, len(cands)):
        if cands[i][1] > cands[best][1]:
            best = i
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(V.OUT / 'V5_reader_selection.json'))
    a = ap.parse_args()
    RL = V.reporting_lock()
    comps = V.components()
    A = C.load_role('ATTACK_FIT', ('features', 'sex'))
    Vv = C.load_role('ATTACK_VALIDATION', ('features', 'sex'))
    R = C.load_role('DEV_REPORT', ('features',))
    res = {'check': 'V5 Reader selection', 'scope': 'REPLAY of saved reader models and stored scores (no refit); label-free',
           'per_seed': {}}
    all_ok = True
    for s in V.SEEDS:
        rd = V.PRIV / 'runs' / f'READERS_{s}'
        rls = RL['readers'][f'S{s}']
        rs = V.jload(rd / 'READERS_SEED.json')
        rec = {'hashes': {'READERS_SEED.json': V.sha256_file(rd / 'READERS_SEED.json') == rls['readers_seed_json_sha256'],
                          'DEV_REPORT_SCORES.npz': V.sha256_file(rd / 'DEV_REPORT_SCORES.npz') == rls['dev_report_scores_sha256']}}
        bank, ids = C.load_bank(s, roles=('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
        rows_ok = all(np.array_equal(ids[f'row_id__{r}'], X['row_id']) and np.array_equal(ids[f'unit__{r}'], X['unit'])
                      for r, X in (('ATTACK_FIT', A), ('ATTACK_VALIDATION', Vv), ('DEV_REPORT', R)))
        with np.load(rd / 'DEV_REPORT_SCORES.npz', allow_pickle=False) as z:
            stored = {k: z[k].copy() for k in z.files}
        # ---- unique single-recipient wires (own grouping by bytes over the three reader roles)
        groups = {}
        for svc in V.ROSTER:
            for ti, t in enumerate(V.TASKS):
                key = (t, b''.join(np.ascontiguousarray(bank[svc][r][t]).tobytes() for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')))
                groups.setdefault(key, []).append(svc)
        wire_of, aliases, label_consistent = {}, {}, True
        for (t, _), svcs in groups.items():
            labs = {comps[x][V.TASKS.index(t)] for x in svcs}
            label_consistent &= len(labs) == 1
            wid = f'{sorted(labs)[0]}|{t}'
            aliases[wid] = svcs
            for x in svcs:
                wire_of[(x, t)] = wid
        rec['unique_wires'] = {'n': len(aliases), 'aliases': aliases, 'label_consistent': bool(label_consistent),
                               'equal_to_READERS_SEED_aliases': {k: sorted(v) for k, v in aliases.items()} == {k: sorted(v) for k, v in rs['aliases'].items()},
                               'equal_to_REPORTING_LOCK_aliases': {k: sorted(v) for k, v in aliases.items()} == {k: sorted(v) for k, v in rls['aliases'].items()}}
        # ---- inventory
        exp_ids = [f'MARG__{w}__{n}' for w in aliases for n in V.SLATE] + [f'PAIR__{x}__{n}' for x in V.ROSTER for n in V.SLATE]
        files = sorted(p.name[len('MODEL__'):-len('.joblib')] for p in rd.glob('MODEL__*.joblib'))
        hash_bad = [i for i in files if V.sha256_file(rd / f'MODEL__{i}.joblib') != rls['model_files_sha256'].get(i)
                    or rs['model_files_sha256'].get(i) != rls['model_files_sha256'].get(i)]
        ctrl_keys = sorted(k for k in stored if k.startswith('CONTROL__'))
        n_marg = sum(1 for i in exp_ids if i.startswith('MARG__'))
        n_pair = sum(1 for i in exp_ids if i.startswith('PAIR__'))
        n_ctrl_fits = len({k.split('__')[1] for k in ctrl_keys}) * 2       # per view: one shuffled + one planted fit
        rec['inventory'] = {'expected_models': len(exp_ids), 'model_files': len(files), 'missing': sorted(set(exp_ids) - set(files)),
                            'unexpected': sorted(set(files) - set(exp_ids)), 'model_hash_mismatch': hash_bad,
                            'control_score_keys': ctrl_keys,
                            'fit_counts_recomputed': {'marginal_unique_wire_fits': n_marg, 'pair_fits': n_pair, 'control_fits': n_ctrl_fits,
                                                      'effective_numerical_fits': n_marg + n_pair + n_ctrl_fits,
                                                      'aliased_service_views_not_refit': sum(len(v) - 1 for v in aliases.values()) * len(V.SLATE)},
                            'fit_counts_READERS_SEED': rs['fit_counts'], 'fit_counts_REPORTING_LOCK': rls['fit_counts']}
        fc = rec['inventory']['fit_counts_recomputed']
        rec['inventory']['fit_counts_agree'] = bool(fc == rs['fit_counts'] == rls['fit_counts'] and fc['effective_numerical_fits'] == 125)
        # ---- per model replay
        per = {}
        scaler_bad, auc_diffs, orient_bad, rep_nonbit, rep_maxdiff = [], [], [], [], 0.0
        for mid in exp_ids:
            kind_name = mid.split('__')[-1]
            kind = V.reader_kind(kind_name)
            if mid.startswith('MARG__'):
                wid = mid.split('__')[1]
                svc0 = aliases[wid][0]
                t = wid.split('|')[1]
                xs = {r: V.view_features(bank[svc0][r], t, kind) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')}
                ref = rs['marginal_banks'][wid]['readers'][kind_name]
            else:
                svc = mid.split('__')[1]
                xs = {r: V.view_features(bank[svc][r], 'pair', kind) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')}
                ref = rs['services'][svc]['pair_readers'][kind_name]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                recd = joblib.load(rd / f'MODEL__{mid}.joblib')
            m, sc = recd['model'], recd['scaler']
            xf = xs['ATTACK_FIT']
            sc_ok = bool(np.array_equal(sc.mean_, xf.mean(0)) and np.array_equal(sc.raw_scale_, xf.std(0))
                         and int(getattr(m, 'n_features_in_', xf.shape[1])) == xf.shape[1])
            if not sc_ok:
                scaler_bad.append(mid)
            raw_v = V.saved_predict(recd, xs['ATTACK_VALIDATION'])
            a_raw = V.auc(Vv['sex'], raw_v)
            sign = 1 if a_raw >= 0.5 else -1
            a_or = V.auc(Vv['sex'], raw_v if sign == 1 else 1.0 - raw_v)
            raw_r = V.saved_predict(recd, xs['DEV_REPORT'])
            s_r = raw_r if sign == 1 else 1.0 - raw_r
            st = stored[mid]
            bit = bool(np.array_equal(s_r, st))
            md = V.maxabs(s_r, st)
            rep_maxdiff = max(rep_maxdiff, md)
            if not bit:
                rep_nonbit.append(mid)
            d = abs(a_or - ref['attack_validation_auc'])
            auc_diffs.append(d)
            if sign != ref['orientation']:
                orient_bad.append(mid)
            per[mid] = {'av_auc': a_or, 'orientation': sign, 'campaign_av_auc': ref['attack_validation_auc'],
                        'abs_diff': d, 'dev_report_replay_bitwise': bit, 'dev_report_replay_max_abs_diff': md}
        rec['scaler_fitted_on_ATTACK_FIT'] = {'models': len(exp_ids), 'equal_moments': len(exp_ids) - len(scaler_bad), 'failures': scaler_bad}
        rec['attack_validation_auc_recomputed'] = {'models': len(per), 'max_abs_diff_vs_READERS_SEED': max(auc_diffs),
                                                    'exact_equal': sum(x == 0.0 for x in auc_diffs), 'orientation_mismatch': orient_bad,
                                                    'orientation_minus_one_count': sum(v['orientation'] == -1 for v in per.values())}
        rec['dev_report_score_replay'] = {'models': len(per), 'bitwise': len(per) - len(rep_nonbit), 'non_bitwise': rep_nonbit,
                                          'max_abs_diff': rep_maxdiff}
        # ---- winners
        win, cand_ok, tie_info, win_bad = {}, {}, {}, []
        for svc in V.ROSTER:
            win[svc] = {}
            for view in V.TASKS:
                wid = wire_of[(svc, view)]
                cands = [(n, per[f'MARG__{wid}__{n}']['av_auc'], f'MARG__{wid}__{n}') for n in V.SLATE]
                i = pick(cands)
                win[svc][view] = {'candidate': cands[i][0], 'reader_id': cands[i][2], 'attack_validation_auc': cands[i][1]}
                tie_info[f'{svc}|{view}'] = sum(c[1] == cands[i][1] for c in cands) - 1
            wi, wo = wire_of[(svc, 'income')], wire_of[(svc, 'occupation')]
            cands = [(n, per[f'PAIR__{svc}__{n}']['av_auc'], f'PAIR__{svc}__{n}') for n in V.SLATE]
            cands += [(f'IGNORE_occupation__{n}', per[f'MARG__{wi}__{n}']['av_auc'], f'MARG__{wi}__{n}') for n in V.SLATE]
            cands += [(f'IGNORE_income__{n}', per[f'MARG__{wo}__{n}']['av_auc'], f'MARG__{wo}__{n}') for n in V.SLATE]
            i = pick(cands)
            win[svc]['pair'] = {'candidate': cands[i][0], 'reader_id': cands[i][2], 'attack_validation_auc': cands[i][1]}
            tie_info[f'{svc}|pair'] = sum(c[1] == cands[i][1] for c in cands) - 1
            theirs = rs['services'][svc]['pair_candidates_attack_validation_auc']
            cand_ok[svc] = {'n_candidates': len(cands), 'names_equal_campaign': [c[0] for c in cands] == list(theirs.keys()) or sorted(c[0] for c in cands) == sorted(theirs),
                            'max_abs_auc_diff_vs_campaign_candidates': max(abs(c[1] - theirs[c[0]]) for c in cands),
                            'ignore_aliases_point_to_own_wires': {'income_only_wire': wi, 'occupation_only_wire': wo,
                                                                  'campaign_income_wire': rs['services'][svc]['income_wire'],
                                                                  'campaign_occupation_wire': rs['services'][svc]['occupation_wire']}}
            cand_ok[svc]['ok'] = bool(len(cands) == 21 and cand_ok[svc]['names_equal_campaign']
                                      and cand_ok[svc]['max_abs_auc_diff_vs_campaign_candidates'] <= 1e-12
                                      and wi == rs['services'][svc]['income_wire'] and wo == rs['services'][svc]['occupation_wire'])
            for view in V.VIEWS:
                c1, c2, c3 = win[svc][view], rs['services'][svc]['winner'][view], rls['winners'][svc][view]
                if not (c1['candidate'] == c2['candidate'] == c3['candidate'] and c1['reader_id'] == c2['reader_id'] == c3['reader_id']):
                    win_bad.append(f'{svc}|{view}')
        rec['winners_derived'] = win
        rec['winner_disagreements'] = win_bad
        rec['pair_candidates'] = cand_ok
        rec['exact_ties_at_max_per_service_view'] = tie_info
        # second-best margin for pair winners (how close the runner-up is)
        rec['pair_winner_margin_over_runner_up'] = {}
        for svc in V.ROSTER:
            wi, wo = wire_of[(svc, 'income')], wire_of[(svc, 'occupation')]
            vals = sorted([per[f'PAIR__{svc}__{n}']['av_auc'] for n in V.SLATE] + [per[f'MARG__{wi}__{n}']['av_auc'] for n in V.SLATE]
                          + [per[f'MARG__{wo}__{n}']['av_auc'] for n in V.SLATE], reverse=True)
            rec['pair_winner_margin_over_runner_up'][svc] = vals[0] - vals[1]
        # ---- stored score file structure
        sc_keys = sorted(k for k in stored if k not in ('row_id', 'unit'))
        rec['stored_scores'] = {'arrays': len(sc_keys), 'expected': len(exp_ids) + len(ctrl_keys),
                                'all_finite': all(bool(np.isfinite(stored[k]).all()) for k in sc_keys),
                                'rows_equal_DEV_REPORT_export': bool(np.array_equal(stored['row_id'], R['row_id']) and np.array_equal(stored['unit'], R['unit']))}
        # ---- replay vs the restored source scores on shared wires (label-free)
        with np.load(V.SRCP / 'runs' / f'READERS_{s}' / 'DEV_REPORT_SCORES.npz', allow_pickle=False) as z:
            src = {k: z[k].copy() for k in z.files}
        cmp_ = {}
        for wid in aliases:
            lab, t = wid.split('|')
            for n in V.SLATE:
                k = f'{lab}__{t}__{n}'
                if k in src:
                    cmp_[f'MARG__{wid}__{n}'] = bool(np.array_equal(stored[f'MARG__{wid}__{n}'], src[k]))
        for svc, lab in SOURCE_LABEL.items():
            for n in V.SLATE:
                k = f'{lab}__pair__{n}'
                if k in src:
                    cmp_[f'PAIR__{svc}__{n}'] = bool(np.array_equal(stored[f'PAIR__{svc}__{n}'], src[k]))
        rec['stored_vs_source_scores'] = {'compared': len(cmp_), 'bitwise': sum(cmp_.values()), 'campaign_reported': rls['replay_vs_source']}
        # ---- DEV_REPORT never used for selection
        rec['no_dev_report_in_selection'] = {
            'winners_rederived_from_ATTACK_VALIDATION_only_equal_campaign': not win_bad,
            'READERS_SEED_roles': rs['roles'], 'DEV_REPORT_labels_opened_or_hashed': rs['DEV_REPORT_labels_opened_or_hashed'],
            'reader_seed_rule': {'READERS_SEED': rs['reader_model_seed'], 'expected': V.MODEL_SEED_BASE + s}}
        sel_ok = (rs['roles'] == {'train': 'ATTACK_FIT', 'select': 'ATTACK_VALIDATION', 'score': 'DEV_REPORT'}
                  and rs['DEV_REPORT_labels_opened_or_hashed'] is False and rs['reader_model_seed'] == V.MODEL_SEED_BASE + s)
        seed_ok = bool(all(rec['hashes'].values()) and rows_ok and rec['unique_wires']['n'] == 10 and label_consistent
                       and rec['unique_wires']['equal_to_READERS_SEED_aliases'] and rec['unique_wires']['equal_to_REPORTING_LOCK_aliases']
                       and not rec['inventory']['missing'] and not rec['inventory']['unexpected'] and not hash_bad
                       and len(files) == 119 and rec['inventory']['fit_counts_agree'] and not scaler_bad
                       and max(auc_diffs) <= 1e-12 and not orient_bad and rep_maxdiff <= 1e-12 and not win_bad
                       and all(v['ok'] for v in cand_ok.values()) and rec['stored_scores']['arrays'] == rec['stored_scores']['expected'] == 128
                       and rec['stored_scores']['all_finite'] and rec['stored_scores']['rows_equal_DEV_REPORT_export']
                       and rec['stored_vs_source_scores']['compared'] == rec['stored_vs_source_scores']['bitwise'] == 105 and sel_ok)
        rec['pass'] = seed_ok
        all_ok &= seed_ok
        res['per_seed'][f'S{s}'] = rec
    # timing: readers finished and REPORTING_LOCK written before the first DEV_REPORT truth re-read
    ledger = [json.loads(l) for l in (V.PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines() if l.strip()]
    ends = {e['unit_id']: e['utc'] for e in ledger if e.get('event') == 'unit_end'}
    acc = [json.loads(l) for l in (V.PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl').read_text().splitlines() if l.strip()]
    first = min((e['utc'] for e in acc), default=None)
    t = lambda x: dt.datetime.fromisoformat(x)
    timing = {'READERS_S0_V1_end': ends.get('READERS_S0_V1'), 'READERS_S1_V1_end': ends.get('READERS_S1_V1'),
              'REPORTING_LOCK_V1_end': ends.get('REPORTING_LOCK_V1'), 'first_DEV_REPORT_truth_reread': first,
              'first_reread_unit': min(acc, key=lambda e: e['utc'])['unit_id'] if acc else None}
    timing['readers_and_reporting_lock_before_first_truth_reread'] = bool(
        first and all(ends.get(u) and t(ends[u]) <= t(first) for u in ('READERS_S0_V1', 'READERS_S1_V1', 'REPORTING_LOCK_V1')))
    res['timing'] = timing
    all_ok &= timing['readers_and_reporting_lock_before_first_truth_reread']
    res['derived_winner_reader_ids'] = {f'S{s}': {svc: {v: res['per_seed'][f'S{s}']['winners_derived'][svc][v]['reader_id'] for v in V.VIEWS}
                                                  for svc in V.ROSTER} for s in V.SEEDS}
    res['status'] = 'PASS' if all_ok else 'FAIL'
    V.write_json(a.out, res)
    print({'status': res['status'], **{k: v['pass'] for k, v in res['per_seed'].items()}, 'timing': timing['readers_and_reporting_lock_before_first_truth_reread']})


if __name__ == '__main__':
    main()
