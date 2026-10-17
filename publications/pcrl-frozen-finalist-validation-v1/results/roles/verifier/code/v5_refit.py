"""Units VER_REFIT_S{s}_V1: check 5 (representative reader refits) and the ATTACK_VALIDATION half of check 6, one model seed.

Public-use ACS 2019 Florida person records. Evaluation readers infer the recorded SEX column from the released income and
occupation predictions. Reads ATTACK_FIT SEX (fitting) and ATTACK_VALIDATION SEX (orientation, AUC recomputation) through the
admitted, logging loader; DEV_REPORT is scored from released outputs only (no DEV_REPORT label is read here).

A. Recompute, with own AUC code, the ATTACK_VALIDATION AUC of every new candidate and control from the stored AV scores.
B. Re-derive every original and enhanced winner from the candidate AUCs (source-first order; exact ties to the earlier one).
C. Refit, with own feature, scaling and estimator code following the lock text, the READER_BANK_LOCK verifier roster: the ET5,
   HGB31, MLP6432 and CATI1 pair readers of FINALIST, NEURAL_RAW and H_LJ, plus the enhanced pair winners of those three; compare
   AV AUCs, AV scores and DEV_REPORT scores with the stored ones (bitwise where possible).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import vcommon as V
from ffv import common as C        # admitted loaders only

ROSTER_NAMES = ('ET5', 'HGB31', 'MLP6432', 'CATI1')
ROSTER_SVCS = ('FINALIST', 'NEURAL_RAW', 'H_LJ')


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def lookup_fit_predict(keys_fit, y, keys_list, pseudo=2.0):
    pi = float(np.mean(y))
    tab = {}
    for k, v in zip(keys_fit, y):
        a, b = tab.get(k, (0, 0))
        tab[k] = (a + int(v), b + 1)
    return [np.array([(tab[k][0] + pseudo * pi) / (tab[k][1] + pseudo) if k in tab else pi for k in ks], dtype=np.float64) for ks in keys_list]


def rowkeys(*arrs):
    z = np.ascontiguousarray(np.column_stack([np.asarray(a, dtype=np.float64) for a in arrs]))
    return [r.tobytes() for r in z]


def main(seed):
    s = seed
    out = {'unit': f'VER_REFIT_S{s}_V1', 'seed': s, 'model_seed': V.MODEL_SEED_BASE + s}
    lock = V.jload(V.RES / 'READER_BANK_LOCK.json')
    import sklearn, scipy
    out['versions_equal_lock'] = {'sklearn': sklearn.__version__ == lock['versions']['sklearn'], 'numpy': np.__version__ == lock['versions']['numpy'],
                                  'scipy': scipy.__version__ == lock['versions']['scipy']}
    out['estimator_params_equal_lock'] = {n: V.symbolic_params(n) == lock['estimator_params'][n] for n in V.NEW8}
    ed = V.RUNS / f'READERS_ENH_S{s}'
    enh = V.jload(ed / 'READERS_ENH_SEED.json')
    av_sc = load_npz(ed / 'AV_SCORES.npz')
    dev_sc = load_npz(ed / 'DEV_REPORT_SCORES.npz')
    src_rec, src_sc = C.source_readers(s)
    bank, ids, _ = C.service_bank(s)
    A = C.load_role('ATTACK_FIT', ('features', 'sex'), purpose=f'FFV independent verifier: roster reader refits (fit), seed {s}')
    Vv = C.load_role('ATTACK_VALIDATION', ('features', 'sex'),
                     purpose=f'FFV independent verifier: orientation, AV AUC recomputation and winner re-derivation, seed {s}')
    R = C.load_role('DEV_REPORT', ('features',))
    for X in (A, Vv, R):                       # readers never see Census inputs: keep ids only
        del X['X'], X['feature_names']
    for r, X in (('ATTACK_FIT', A), ('ATTACK_VALIDATION', Vv), ('DEV_REPORT', R)):
        if not (np.array_equal(ids[f'row_id__{r}'], X['row_id']) and np.array_equal(ids[f'unit__{r}'], X['unit'])):
            raise AssertionError(f'bank row order differs from the {r} export')
    if not (np.array_equal(av_sc['row_id'], Vv['row_id']) and np.array_equal(dev_sc['row_id'], R['row_id'])):
        raise AssertionError('stored score row order differs')
    yA, yV = A['sex'], Vv['sex']

    # ---------------------------------------------------------------- A. AV AUCs of every stored new candidate and control
    rec_auc = {}
    for wid, mb in enh['marginal_banks'].items():
        for n, r in mb['readers'].items():
            rec_auc[f'MARG__{wid}__{n}'] = r['attack_validation_auc']
    for svc, pr in enh['pairs'].items():
        for n, r in pr.items():
            rec_auc[f'PAIR__{svc}__{n}'] = r['attack_validation_auc']
    for view, cr in enh['controls'].items():
        for n, c in cr.items():
            rec_auc[f'CONTROL__{view}__{n}__SHUFFLED'] = c['shuffled_attack_validation_auc']
            rec_auc[f'CONTROL__{view}__{n}__PLANTED'] = c['planted_attack_validation_auc']
    own_auc, diffs, below_half = {}, {}, []
    for k, a in rec_auc.items():
        own_auc[k] = V.auc(yV, av_sc[k])
        diffs[k] = abs(own_auc[k] - a)
        if not k.startswith('CONTROL__') and own_auc[k] < 0.5:
            below_half.append(k)
    unexpected = sorted(set(av_sc) - set(rec_auc) - {'row_id'})
    out['A_attack_validation_auc_recomputation'] = {
        'n_compared': len(rec_auc), 'n_exactly_equal': int(sum(d == 0.0 for d in diffs.values())), 'max_abs_diff': max(diffs.values()),
        'oriented_candidates_below_0.5': below_half, 'stored_av_keys_without_record': unexpected}
    ctl = {}
    for view, cr in enh['controls'].items():
        for n in cr:
            sh, pl = own_auc[f'CONTROL__{view}__{n}__SHUFFLED'], own_auc[f'CONTROL__{view}__{n}__PLANTED']
            ctl[f'{view}|{n}'] = {'shuffled': sh, 'planted': pl, 'in_band': bool(pl >= 0.95 and abs(sh - 0.5) <= 0.05)}
    out['A_controls_attack_validation'] = ctl

    # ---------------------------------------------------------------- B. winners re-derived
    rl = V.jload(V.RES / 'REPORTING_LOCK.json')['enhanced_readers'][f'S{s}']['winners']
    win, mism = {}, []
    for svc in V.BANK:
        for view in V.VIEWS:
            ce = V.enhanced_candidates(src_rec, enh, svc, view)
            pe = V.pick_max_first(ce)
            # the same rule with the own recomputed AV AUCs for the new candidates (source values stay stored, per the lock)
            ce_own = [(n, own_auc[k.split('::', 1)[1]] if o == 'new' else a, k, o) for n, a, k, o in ce]
            pe_own = V.pick_max_first(ce_own)
            co = V.source_candidates(src_rec, svc, view)
            po = V.pick_max_first(co)
            st = enh['winners'][svc]
            sw = src_rec['services'][svc]['winner'][view]
            ok = {'enhanced_candidate': pe['candidate'] == st['enhanced'][view]['candidate'],
                  'enhanced_key': pe['key'] == st['enhanced'][view]['reader_key'],
                  'enhanced_auc': pe['auc'] == st['enhanced'][view]['attack_validation_auc'],
                  'enhanced_gap': (pe['gap_to_runner_up'] == st['enhanced'][view]['gap_to_runner_up']),
                  'enhanced_near_tie_flag': pe['near_tie'] == st['enhanced'][view]['AV_NEAR_TIE'],
                  'enhanced_n_candidates': pe['n'] == st['enhanced'][view]['n_candidates'] == (16 if view in V.TASKS else 48),
                  'enhanced_same_with_own_av_aucs': pe_own['key'] == pe['key'],
                  'enhanced_equals_reporting_lock': (pe['candidate'] + ('' if pe['origin'] == 'source' else ' [new]')) == rl[svc]['enhanced'][view],
                  'original_candidate_equals_source_record': po['candidate'] == sw['candidate'] and po['key'] == sw['reader_id'],
                  'original_equals_campaign_record': st['original'][view]['reader_key'] == 'SRC::' + sw['reader_id'],
                  'original_equals_reporting_lock': po['candidate'] == rl[svc]['original'][view],
                  'gap_to_original': (pe['auc'] - po['auc']) == st['enhanced'][view]['gap_to_original_winner']}
            win[f'{svc}|{view}'] = {'enhanced': pe['candidate'] + ('' if pe['origin'] == 'source' else ' [new]'), 'enhanced_key': pe['key'],
                                    'enhanced_auc': pe['auc'], 'gap_to_runner_up': pe['gap_to_runner_up'], 'near_tie': pe['near_tie'],
                                    'original': po['candidate'], 'original_key': po['key'], 'original_auc': po['auc'],
                                    'gap_to_original': pe['auc'] - po['auc'], 'checks': ok}
            if not all(ok.values()):
                mism.append(f'{svc}|{view}')
    out['B_winners'] = {'per_release_view': win, 'mismatches': mism, 'n': len(win)}

    # ---------------------------------------------------------------- C. roster refits
    roster = [(svc, n, 'new', f'PAIR__{svc}__{n}') for svc in ROSTER_SVCS for n in ROSTER_NAMES]
    extra = []
    for svc in ROSTER_SVCS:
        w = win[f'{svc}|pair']
        origin, key = w['enhanced_key'].split('::', 1)
        if key in [r[3] for r in roster] and origin == 'NEW':
            extra.append({'service': svc, 'winner': w['enhanced'], 'note': 'already in the fixed roster'})
            continue
        roster.append((svc, w['enhanced'].replace(' [new]', ''), 'new' if origin == 'NEW' else 'source', key))
        extra.append({'service': svc, 'winner': w['enhanced'], 'key': key})
    out['C_roster'] = {'fixed': [r[3] for r in roster[:12]], 'enhanced_pair_winners': extra}
    seedm = V.MODEL_SEED_BASE + s
    refit_scores, res = {}, {}
    trip = {svc: [(bank[svc][r]['income'], bank[svc][r]['occupation']) for r in V.ROLES3] for svc in ROSTER_SVCS}
    cati_lock = lock['cati_fitted_state'][f'S{s}']
    for svc, cand, origin, key in roster:
        t0 = time.process_time()
        name = cand.split('__')[-1]
        view_used = 'pair' if not cand.startswith('IGNORE_') else ('income' if cand.startswith('IGNORE_occupation') else 'occupation')
        info = {'service': svc, 'candidate': cand, 'origin': origin, 'view_used': view_used, 'stored_key': key}
        T3 = trip[svc]
        if origin == 'new' and name.startswith('CATI'):
            if view_used != 'pair':
                raise NotImplementedError('single-view CATI refit not needed for this roster')
            dz = V.CatiPair().fit(T3[0][0])
            X3 = [dz.design(i, o) for i, o in T3]
            st = dz.state()
            lk = cati_lock[f'{svc}|pair']['income']
            info['cati_state_equals_lock'] = (st['categories_hex'] == lk['categories_hex']) if st['kind'] == 'categorical' else (st['cuts'] == lk['cuts'])
            info['n_columns'] = int(X3[0].shape[1])
        elif origin == 'new':
            def f(i, o):
                return V.rich(i) if view_used == 'income' else V.rich(o) if view_used == 'occupation' else np.column_stack((V.rich(i), V.rich(o)))
            X3 = [f(i, o) for i, o in T3]
            info['n_columns'] = int(X3[0].shape[1])
        elif name != 'LOOKUP':
            kd = V.SRC_KIND[name]

            def f(i, o):
                a, b = V.source_kind(i, kd), V.source_kind(o, kd)
                return a if view_used == 'income' else b if view_used == 'occupation' else np.column_stack((a, b))
            X3 = [f(i, o) for i, o in T3]
            info['n_columns'] = int(X3[0].shape[1])
        if name == 'LOOKUP':
            ks = [rowkeys(i) if view_used == 'income' else rowkeys(o) if view_used == 'occupation' else rowkeys(i, o) for i, o in T3]
            raw_v, raw_d = lookup_fit_predict(ks[0], yA, ks[1:])
            mdl = None
        else:
            (raw_v, raw_d), (mdl, _) = V.fit_predict(name, X3[0], yA, X3[1:], seedm)
        sign, sv, (sd,) = V.orient_scores(yV, raw_v, [raw_d])
        my_auc = V.auc(yV, sv)
        if origin == 'new':
            st_av, st_dev = av_sc[key], dev_sc[key]
            st_auc = rec_auc[key]
        else:
            st_av = None
            st_dev = src_sc[key]
            if key.startswith('PAIR__'):
                st_auc = src_rec['services'][svc]['pair_readers'][name]['attack_validation_auc']
            else:
                wid = key.split('__')[1]
                st_auc = src_rec['marginal_banks'][wid]['readers'][name]['attack_validation_auc']
        info.update({'orientation': sign, 'own_attack_validation_auc': my_auc, 'stored_attack_validation_auc': st_auc,
                     'attack_validation_auc_diff': my_auc - st_auc,
                     'av_scores_bitwise': None if st_av is None else bool(np.array_equal(sv, st_av)),
                     'av_scores_max_abs_diff': None if st_av is None else float(np.max(np.abs(sv - st_av))),
                     'dev_scores_bitwise': bool(np.array_equal(sd, st_dev)), 'dev_scores_max_abs_diff': float(np.max(np.abs(sd - st_dev))),
                     'n_iter': None if mdl is None else int(np.max(getattr(mdl, 'n_iter_', [-1]))), 'cpu_s': round(time.process_time() - t0, 2)})
        refit_scores[f'REFIT__{key}'] = sd
        res[key] = info
        print(key, origin, 'AV', round(my_auc, 6), 'diff', my_auc - st_auc, 'dev bitwise', info['dev_scores_bitwise'], info['dev_scores_max_abs_diff'], info['cpu_s'], flush=True)
    out['C_refits'] = res
    out['C_summary'] = {'n_refits': len(res), 'dev_bitwise': int(sum(r['dev_scores_bitwise'] for r in res.values())),
                        'av_bitwise_new': int(sum(bool(r['av_scores_bitwise']) for r in res.values() if r['av_scores_bitwise'] is not None)),
                        'max_abs_dev_diff': max(r['dev_scores_max_abs_diff'] for r in res.values()),
                        'max_abs_av_auc_diff': max(abs(r['attack_validation_auc_diff']) for r in res.values())}
    V.VPRIV.mkdir(parents=True, exist_ok=True)
    with open(V.VPRIV / f'refit_S{s}.npz', 'wb') as fh:
        np.savez(fh, row_id=R['row_id'], **refit_scores)
    V.jdump(V.VPRIV / f'check5_refit_S{s}.json', out)
    print({k: v for k, v in out.items() if k in ('A_attack_validation_auc_recomputation', 'C_summary', 'versions_equal_lock', 'estimator_params_equal_lock')})
    print('winner mismatches', mism)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    main(ap.parse_args().seed)
