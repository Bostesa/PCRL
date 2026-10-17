"""Units UM_READERS_S{s}: fresh evaluation readers for one model seed over this follow-up's final bank (PROTOCOL.md section 8).

Source slate, transformations and rules unchanged (`scm_bench.readers`): OUT_LR, OUT_MLP, OUT_HGB on probabilities + decision;
LOG_LR, LOG_MLP on log-probabilities + one-hot decision; DEC_LR on the one-hot decision; CONF_HGB on the max probability; and the
corrected exact-vector LOOKUP reader (exact float64 bytes, (n_female + 2 pi)/(n + 2), unseen key -> pi). Views: income recipient,
occupation recipient, linked pair. Fit on ATTACK_FIT with SEX; orientation and every winner on ATTACK_VALIDATION; DEV_REPORT
scored from released outputs alone (no DEV_REPORT label is opened here). Model seed MODEL_SEED_BASE + s for every reader.
- Marginal banks are fitted once per unique single-recipient wire; services sharing a wire alias it. Pair readers are fitted once
  per service (services are already one per effective release). Pair candidates: 8 pair readers + 16 ignore-one alternatives.
- Controls on the SCM service's wire, per view: shuffled-SEX null (seed 2026101602 + s; OUT_HGB) and planted SEX bit (OUT_HGB).
- Label-free replay: pair-reader DEV_REPORT scores of services also attacked by the source benchmark (historical SCM, C0,
  NEURAL_RAW, H_LJ) are compared with the source's stored scores (recorded only; never a reason to change a reader).
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

from scm_bench.readers import SLATE, NULL_SEED, VIEWS, kinds, lookup_keys, fit_lookup, predict_lookup
from . import common as C

ROLES3 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
SOURCE_NAME = {'HIST_SCM': 'SCM', 'C0': 'C0', 'NEURAL_RAW': 'NEURAL_RAW', 'H_LJ': 'H_LJ'}


def load_service_bank(seed):
    man = json.loads((C.RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    p = C.PRIV / 'bank' / f'SERVICE_BANK_S{seed}.npz'
    if C.sha(p) != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('service bank hash differs from SERVICE_BANK_MANIFEST')
    out, ids = {}, {}
    with np.load(p) as z:
        for key in z.files:
            parts = key.split('__')
            if parts[0] in ('row_id', 'unit'):
                ids[key] = z[key].copy(); continue
            svc, role, task = parts
            out.setdefault(svc, {}).setdefault(role, {})[task] = z[key].copy()
    return out, ids, man


def run(seed, out_dir):
    from task_conditioned_representation.attacks import MODEL_SEED_BASE, fit_reader, predict_reader, orientation, orient
    from task_conditioned_representation.inference import auc
    for name in ('ANALYSIS_LOCK.json', 'SELECTION_LOCK.json', 'SERVICE_BANK_MANIFEST.json'):
        if not C.committed(C.RESULTS / name):
            raise PermissionError(f'reader fits require the committed {name}')
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh reader directory required')
    out.mkdir(parents=True, mode=0o700)
    A = C.load_role('ATTACK_FIT', ('features', 'sex')); V = C.load_role('ATTACK_VALIDATION', ('features', 'sex'))
    R = C.load_role('DEV_REPORT', ('features',))
    bank, ids, man = load_service_bank(seed)
    ps = man['per_seed'][f'S{seed}']
    roster = ps['attacked_services']; svc_of = ps['service_of_role']; nominee = svc_of['SCM']
    for r, X in (('ATTACK_FIT', A), ('ATTACK_VALIDATION', V), ('DEV_REPORT', R)):
        if not (np.array_equal(ids[f'row_id__{r}'], X['row_id']) and np.array_equal(ids[f'unit__{r}'], X['unit'])):
            raise AssertionError(f'bank row order differs from the {r} export')
    if len(A['row_id']) != 8001 or len(V['row_id']) != 3000:
        raise AssertionError('attack role sizes differ from the source budgets (8,001 / 3,000)')
    seed_model = MODEL_SEED_BASE + seed
    rec = {'schema': 'SCMUM1-readers-v1', 'seed': seed, 'reader_model_seed': seed_model, 'rng_rule': 'MODEL_SEED_BASE + s for every service, view and reader',
           'roles': {'train': 'ATTACK_FIT', 'select': 'ATTACK_VALIDATION', 'score': 'DEV_REPORT'},
           'rows': {'ATTACK_FIT': int(len(A['row_id'])), 'ATTACK_VALIDATION': int(len(V['row_id'])), 'DEV_REPORT': int(len(R['row_id']))},
           'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'), 'selection_lock_sha256': C.sha(C.RESULTS / 'SELECTION_LOCK.json'),
           'bank_sha256': man['bank_files_sha256'][f'S{seed}'], 'roster': roster, 'service_of_role': svc_of,
           'marginal_banks': {}, 'services': {}, 'controls': {}, 'accounting': [], 'aliases': {}}
    scores, models = {}, {}

    def fit_one(x_fit, y_fit, x_val, x_rep, name, rid):
        t0 = time.process_time()
        if name == 'LOOKUP':
            m = fit_lookup(x_fit, y_fit); fit_pred = predict_lookup
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = fit_reader(x_fit, y_fit, name, seed_model)
            fit_pred = predict_reader
        mp = out / f'MODEL__{rid}.joblib'; joblib.dump(m, mp, compress=3); mp.chmod(0o600)
        raw = fit_pred(m, x_val); sign = orientation(V['sex'], raw)
        val_auc = auc(V['sex'], orient(raw, sign)); s_rep = orient(fit_pred(m, x_rep), sign)
        scores[rid] = s_rep; models[rid] = C.sha(mp)
        rec['accounting'].append({'reader_id': rid, 'reader': name, 'cpu_s': round(time.process_time() - t0, 4)})
        o = {'attack_validation_auc': float(val_auc), 'orientation': int(sign), 'model_sha256': models[rid]}
        if name == 'LOOKUP':
            o['distinct_fit_keys'] = len(m['table'])
            o['share_val_keys_seen_in_fit'] = float(np.mean([k in m['table'] for k in x_val]))
        return o

    wire_of, uniq = {}, {}
    for svc in roster:
        for t in C.TASKS:
            h = tuple(C.wire_sha(bank[svc][r][t]) for r in ROLES3)
            uniq.setdefault((t, h), []).append(svc)
    names8 = [n for n, _ in SLATE] + ['LOOKUP']
    for (t, h), svcs in uniq.items():
        k = C.K[t]; wid = f'{svcs[0]}|{t}'; svc0 = svcs[0]
        tf, tv, tr = (kinds(bank[svc0][r][t], k) for r in ROLES3)
        mb = {'task': t, 'services_aliasing_this_bank': svcs, 'readers': {}}
        for name, kind in SLATE:
            mb['readers'][name] = fit_one(tf[kind], A['sex'], tv[kind], tr[kind], name, f'MARG__{wid}__{name}')
        kf, kv, kr = (lookup_keys(bank[svc0][r][t]) for r in ROLES3)
        mb['readers']['LOOKUP'] = fit_one(kf, A['sex'], kv, kr, 'LOOKUP', f'MARG__{wid}__LOOKUP')
        rec['marginal_banks'][wid] = mb; rec['aliases'][wid] = svcs
        for svc in svcs:
            wire_of[(svc, t)] = wid

    def pick(cands):
        win = max(range(len(cands)), key=lambda i: (cands[i][1], -i))
        return {'candidate': cands[win][0], 'reader_id': cands[win][2], 'attack_validation_auc': cands[win][1], 'n_candidates': len(cands)}
    for svc in roster:
        srec = {'income_wire': wire_of[(svc, 'income')], 'occupation_wire': wire_of[(svc, 'occupation')], 'pair_readers': {}, 'winner': {}, 'winner_slate7': {}}
        tabs = {}
        for r in ROLES3:
            a, b = kinds(bank[svc][r]['income'], 2), kinds(bank[svc][r]['occupation'], 5)
            tabs[r] = {kk: np.column_stack((a[kk], b[kk])) for kk in a}
        for name, kind in SLATE:
            srec['pair_readers'][name] = fit_one(tabs['ATTACK_FIT'][kind], A['sex'], tabs['ATTACK_VALIDATION'][kind], tabs['DEV_REPORT'][kind], name, f'PAIR__{svc}__{name}')
        kf, kv, kr = (lookup_keys(bank[svc][r]['income'], bank[svc][r]['occupation']) for r in ROLES3)
        srec['pair_readers']['LOOKUP'] = fit_one(kf, A['sex'], kv, kr, 'LOOKUP', f'PAIR__{svc}__LOOKUP')
        for names, key in ((names8, 'winner'), (names8[:7], 'winner_slate7')):
            for view in ('income', 'occupation'):
                mb = rec['marginal_banks'][wire_of[(svc, view)]]['readers']
                srec[key][view] = pick([(n, mb[n]['attack_validation_auc'], f"MARG__{wire_of[(svc, view)]}__{n}") for n in names])
            cands = [(n, srec['pair_readers'][n]['attack_validation_auc'], f'PAIR__{svc}__{n}') for n in names]
            cands += [(f'IGNORE_occupation__{n}', rec['marginal_banks'][wire_of[(svc, 'income')]]['readers'][n]['attack_validation_auc'],
                       f"MARG__{wire_of[(svc, 'income')]}__{n}") for n in names]
            cands += [(f'IGNORE_income__{n}', rec['marginal_banks'][wire_of[(svc, 'occupation')]]['readers'][n]['attack_validation_auc'],
                       f"MARG__{wire_of[(svc, 'occupation')]}__{n}") for n in names]
            srec[key]['pair'] = pick(cands)
            if key == 'winner':
                srec['pair_candidates_attack_validation_auc'] = {c[0]: c[1] for c in cands}
        rec['services'][svc] = srec
    rec['coverage'] = {}
    for svc in roster:
        cov = {}
        for view in VIEWS:
            def keys_of(r):
                return lookup_keys(bank[svc][r]['income']) if view == 'income' else lookup_keys(bank[svc][r]['occupation']) if view == 'occupation' \
                    else lookup_keys(bank[svc][r]['income'], bank[svc][r]['occupation'])
            kf, kv, kd = keys_of('ATTACK_FIT'), keys_of('ATTACK_VALIDATION'), keys_of('DEV_REPORT')
            uf, cf = np.unique(np.array(kf, dtype=object), return_counts=True)
            seen = set(uf.tolist())
            cov[view] = {'distinct_vectors_ATTACK_FIT': int(len(uf)), 'distinct_vectors_ATTACK_VALIDATION': int(len(set(kv))),
                         'distinct_vectors_DEV_REPORT': int(len(set(kd))),
                         'median_ATTACK_FIT_people_per_vector': float(np.median(cf)), 'share_ATTACK_FIT_people_in_vectors_with_15_or_more': float(cf[cf >= 15].sum() / cf.sum()),
                         'share_ATTACK_VALIDATION_people_whose_vector_occurs_in_ATTACK_FIT': float(np.mean([k in seen for k in kv])),
                         'share_DEV_REPORT_people_whose_vector_occurs_in_ATTACK_FIT': float(np.mean([k in seen for k in kd]))}
        rec['coverage'][svc] = cov
    shuffled = np.random.default_rng(NULL_SEED + seed).permutation(A['sex'])
    for view in VIEWS:
        def tab(r):
            a, b = kinds(bank[nominee][r]['income'], 2)['out'], kinds(bank[nominee][r]['occupation'], 5)['out']
            return a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))
        x, xv, xr = tab('ATTACK_FIT'), tab('ATTACK_VALIDATION'), tab('DEV_REPORT')
        t0 = time.process_time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rs = fit_reader(x, shuffled, 'OUT_HGB', seed_model)
            rp = fit_reader(np.column_stack((x, A['sex'])), A['sex'], 'OUT_HGB', seed_model)
        scores[f'CONTROL__{view}__SHUFFLED'] = predict_reader(rs, xr)
        for bit in (0, 1):
            scores[f'CONTROL__{view}__PLANTED_{bit}'] = predict_reader(rp, np.column_stack((xr, np.full(len(xr), bit))))
        rec['controls'][view] = {'wire': nominee, 'shuffled_attack_validation_auc': auc(V['sex'], predict_reader(rs, xv)),
                                 'planted_attack_validation_auc': auc(V['sex'], predict_reader(rp, np.column_stack((xv, V['sex']))))}
        rec['accounting'].append({'reader_id': f'CONTROL__{view}', 'reader': 'OUT_HGB x2', 'cpu_s': round(time.process_time() - t0, 4)})
    replay = {}
    hp = C.SB_PRIV / 'runs' / f'READERS_{seed}' / 'DEV_REPORT_SCORES.npz'
    rl = json.loads((C.SB_RESULTS / 'REPORTING_LOCK.json').read_text())
    if hp.is_file() and C.sha(hp) == rl['readers'][f'S{seed}']['dev_report_scores_sha256']:
        with np.load(hp) as z:
            hs = {k: z[k].copy() for k in z.files}
        if np.array_equal(hs['row_id'], R['row_id']):
            for role, src in SOURCE_NAME.items():
                svc = svc_of.get(role)
                if svc is None:
                    continue
                for name in names8:
                    key = f'PAIR__{src}__{name}'
                    if key in hs:
                        a = scores[f'PAIR__{svc}__{name}']
                        replay[f'{role}|{name}'] = {'max_abs_diff': float(np.max(np.abs(a - hs[key]))), 'bitwise': bool(np.array_equal(a, hs[key]))}
    rec['replay_vs_source'] = {'compared': len(replay), 'bitwise': sum(v['bitwise'] for v in replay.values()),
                               'max_abs_diff': max((v['max_abs_diff'] for v in replay.values()), default=None), 'detail': replay}
    sp = out / 'DEV_REPORT_SCORES.npz'
    with sp.open('xb') as f:
        np.savez(f, row_id=R['row_id'], unit=R['unit'], **scores)
    sp.chmod(0o600)
    rec['dev_report_scores_sha256'] = C.sha(sp); rec['model_files_sha256'] = models
    n_marg = sum(len(m['readers']) for m in rec['marginal_banks'].values()); n_pair = sum(len(s['pair_readers']) for s in rec['services'].values())
    rec['fit_counts'] = {'marginal_unique_wire_fits': n_marg, 'pair_fits': n_pair, 'control_fits': 6, 'effective_numerical_fits': n_marg + n_pair + 6}
    rec['DEV_REPORT_labels_opened_or_hashed'] = False
    rec['sklearn_version'] = __import__('sklearn').__version__
    with (out / 'READERS_SEED.json').open('x') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.out)
    print(json.dumps({'seed': r['seed'], 'fit_counts': r['fit_counts'],
                      'winners': {k: {v: (w['candidate'], round(w['attack_validation_auc'], 4)) for v, w in s['winner'].items()} for k, s in r['services'].items()},
                      'controls': r['controls'], 'replay': {k: r['replay_vs_source'][k] for k in ('compared', 'bitwise', 'max_abs_diff')}}, indent=1))
