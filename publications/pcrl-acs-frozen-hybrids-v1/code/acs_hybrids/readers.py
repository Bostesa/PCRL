"""Fresh complete-view recovery readers for one model seed over the frozen bank (this campaign's common reader bank).

Source slate and transformations (acs_replication.readers, validated definitions in task_conditioned_representation.attacks):
OUT_LR, OUT_MLP, OUT_HGB on probabilities + decision; LOG_LR, LOG_MLP on log-probabilities + one-hot decision; DEC_LR on the
one-hot decision; CONF_HGB on the max probability. Views: income recipient, occupation recipient, coalition pair.
- Train on ATTACK_FIT with SEX; orientation and every winner on ATTACK_VALIDATION only; DEV_REPORT scored from released
  outputs alone (no DEV_REPORT label is opened here). No reader sees MODEL_SELECT, DEV_REPORT truth or sealed people.
- RNG rule: every reader uses model seed MODEL_SEED_BASE + s, independent of service, view and reader name.
- Marginal banks are fitted ONCE per unique single-recipient wire (FROZEN_BANK_MANIFEST); services sharing a wire alias it.
- Fresh pair readers for every roster service, including H_LL and H_JJ. Pair candidates: 7 pair readers + 14 ignore-one
  aliases (the service's own income-only and occupation-only marginal readers). Winner: max ATTACK_VALIDATION AUC with fixed
  orientation; ties go to the earlier candidate (pair readers in slate order, then income-only, then occupation-only).
- Real-data controls on the primary H_LJ wire, per view: shuffled-SEX null (fixed permutation of ATTACK_FIT SEX, seed
  2026101602 + s; OUT_HGB) and a planted SEX bit (OUT_HGB on the wire plus the SEX column).
- Label-free replay check: scores of readers on wires the source also attacked are compared with the source's stored
  DEV_REPORT reader scores (recorded only; never a reason to change a reader).
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

from . import common as C

SLATE = (('OUT_LR', 'out'), ('OUT_MLP', 'out'), ('OUT_HGB', 'out'), ('LOG_LR', 'log'), ('LOG_MLP', 'log'), ('DEC_LR', 'dec'), ('CONF_HGB', 'conf'))
NULL_SEED = 2026101602
SOURCE_LABEL = {'H_LL': 'LOCAL_TREE', 'H_JJ': 'JOINT_TREE', 'NEURAL_RAW': 'NEURAL_RAW_rho1.0_E2', 'TASK_REFIT': 'TASK_REFIT', 'T_TREE': 'T_TREE'}


def kinds(p, k):
    p = np.asarray(p, dtype=np.float64); d = p.argmax(1); one = np.eye(k)[d]
    return {'out': np.column_stack((p, d.astype(np.float64))), 'log': np.column_stack((np.log(np.maximum(p, 1e-300)), one)),
            'dec': one, 'conf': p.max(1, keepdims=True)}


def run(seed, out_dir):
    from task_conditioned_representation.attacks import MODEL_SEED_BASE, fit_reader, predict_reader, orientation, orient
    from task_conditioned_representation.inference import auc
    lock = C.RESULTS / 'ANALYSIS_LOCK.json'
    if not C.committed(lock):
        raise PermissionError('reader fits require the committed ANALYSIS_LOCK')
    man = json.loads((C.RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh reader directory required')
    out.mkdir(parents=True, mode=0o700)
    A = C.load_role('ATTACK_FIT', ('features', 'sex')); V = C.load_role('ATTACK_VALIDATION', ('features', 'sex'))
    R = C.load_role('DEV_REPORT', ('features',))
    bank, ids = C.load_bank(seed, roles=('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
    for r, X in (('ATTACK_FIT', A), ('ATTACK_VALIDATION', V), ('DEV_REPORT', R)):
        if not (np.array_equal(ids[f'row_id__{r}'], X['row_id']) and np.array_equal(ids[f'unit__{r}'], X['unit'])):
            raise AssertionError(f'bank row order differs from the {r} export')
    seed_model = MODEL_SEED_BASE + seed
    rec = {'schema': 'ACSH1-readers-v1', 'seed': seed, 'reader_model_seed': seed_model, 'rng_rule': 'MODEL_SEED_BASE + s for every service, view and reader',
           'roles': {'train': 'ATTACK_FIT', 'select': 'ATTACK_VALIDATION', 'score': 'DEV_REPORT'},
           'rows': {'ATTACK_FIT': int(len(A['row_id'])), 'ATTACK_VALIDATION': int(len(V['row_id'])), 'DEV_REPORT': int(len(R['row_id']))},
           'analysis_lock_sha256': C.sha(lock), 'bank_sha256': man['bank_files_sha256'][f'S{seed}'],
           'marginal_banks': {}, 'services': {}, 'controls': {}, 'accounting': [], 'aliases': {}}
    scores, models = {}, {}

    def fit_one(x_fit, x_val, x_rep, name, rid):
        t0 = time.process_time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = fit_reader(x_fit, A['sex'], name, seed_model)
        mp = out / f'MODEL__{rid}.joblib'; joblib.dump(m, mp, compress=3); mp.chmod(0o600)
        raw = predict_reader(m, x_val); sign = orientation(V['sex'], raw)
        val_auc = auc(V['sex'], orient(raw, sign)); s_rep = orient(predict_reader(m, x_rep), sign)
        scores[rid] = s_rep; models[rid] = C.sha(mp)
        rec['accounting'].append({'reader_id': rid, 'reader': name, 'cpu_s': round(time.process_time() - t0, 4)})
        return {'attack_validation_auc': float(val_auc), 'orientation': int(sign), 'model_sha256': models[rid]}

    # ---- marginal banks: once per unique single-recipient wire
    wire_of = {}
    for u in man['per_seed'][f'S{seed}']['unique_single_recipient_wires']:
        task, src = u['task'], u['source_service']; k = C.K[task]; wid = f'{src}|{task}'
        svc0 = u['services'][0]
        tf, tv, tr = (kinds(bank[svc0][r][task], k) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
        mb = {'task': task, 'source_service': src, 'services_aliasing_this_bank': u['services'], 'readers': {}}
        for name, kind in SLATE:
            mb['readers'][name] = fit_one(tf[kind], tv[kind], tr[kind], name, f'MARG__{wid}__{name}')
        rec['marginal_banks'][wid] = mb
        for svc in u['services']:
            wire_of[(svc, task)] = wid
            if svc != svc0:
                for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'):
                    if not np.array_equal(bank[svc][r][task], bank[svc0][r][task]):
                        raise AssertionError(f'alias {svc}/{task} is not the same wire as {svc0}')
        rec['aliases'][wid] = u['services']
    # ---- pair readers per service and winners
    for svc in C.ATTACK_ROSTER:
        srec = {'income_wire': wire_of[(svc, 'income')], 'occupation_wire': wire_of[(svc, 'occupation')], 'pair_readers': {}, 'winner': {}}
        tabs = {}
        for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'):
            a, b = kinds(bank[svc][r]['income'], 2), kinds(bank[svc][r]['occupation'], 5)
            tabs[r] = {k: np.column_stack((a[k], b[k])) for k in a}
        for name, kind in SLATE:
            srec['pair_readers'][name] = fit_one(tabs['ATTACK_FIT'][kind], tabs['ATTACK_VALIDATION'][kind], tabs['DEV_REPORT'][kind], name, f'PAIR__{svc}__{name}')
        for view in ('income', 'occupation'):
            mb = rec['marginal_banks'][wire_of[(svc, view)]]['readers']
            cands = [(n, mb[n]['attack_validation_auc'], f"MARG__{wire_of[(svc, view)]}__{n}") for n, _ in SLATE]
            win = max(range(len(cands)), key=lambda i: (cands[i][1], -i))
            srec['winner'][view] = {'candidate': cands[win][0], 'reader_id': cands[win][2], 'attack_validation_auc': cands[win][1]}
        cands = [(n, srec['pair_readers'][n]['attack_validation_auc'], f'PAIR__{svc}__{n}') for n, _ in SLATE]
        cands += [(f'IGNORE_occupation__{n}', rec['marginal_banks'][wire_of[(svc, 'income')]]['readers'][n]['attack_validation_auc'],
                   f"MARG__{wire_of[(svc, 'income')]}__{n}") for n, _ in SLATE]
        cands += [(f'IGNORE_income__{n}', rec['marginal_banks'][wire_of[(svc, 'occupation')]]['readers'][n]['attack_validation_auc'],
                   f"MARG__{wire_of[(svc, 'occupation')]}__{n}") for n, _ in SLATE]
        win = max(range(len(cands)), key=lambda i: (cands[i][1], -i))
        srec['winner']['pair'] = {'candidate': cands[win][0], 'reader_id': cands[win][2], 'attack_validation_auc': cands[win][1], 'n_candidates': len(cands)}
        srec['pair_candidates_attack_validation_auc'] = {c[0]: c[1] for c in cands}
        rec['services'][svc] = srec
    # ---- real-data controls on the primary H_LJ wire
    shuffled = np.random.default_rng(NULL_SEED + seed).permutation(A['sex'])
    for view in ('income', 'occupation', 'pair'):
        def tab(r):
            a, b = kinds(bank[C.PRIMARY][r]['income'], 2)['out'], kinds(bank[C.PRIMARY][r]['occupation'], 5)['out']
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
        rec['controls'][view] = {'wire': C.PRIMARY, 'shuffled_attack_validation_auc': auc(V['sex'], predict_reader(rs, xv)),
                                 'planted_attack_validation_auc': auc(V['sex'], predict_reader(rp, np.column_stack((xv, V['sex']))))}
        rec['accounting'].append({'reader_id': f'CONTROL__{view}', 'reader': 'OUT_HGB x2', 'cpu_s': round(time.process_time() - t0, 4)})
    # ---- label-free replay against the source's stored DEV_REPORT reader scores
    with np.load(C.SRCP / f'runs/READERS_{seed}/DEV_REPORT_SCORES.npz') as z:
        src = {k: z[k].copy() for k in z.files}
    if not np.array_equal(src['row_id'], R['row_id']):
        raise AssertionError('source score rows differ')
    replay = {}
    for wid in rec['marginal_banks']:
        s_label, task = wid.split('|')
        for name, _ in SLATE:
            key = f'{s_label}__{task}__{name}'
            if key in src:
                a = scores[f'MARG__{wid}__{name}']; replay[f'MARG__{wid}__{name}'] = {'source_key': key, 'max_abs_diff': float(np.max(np.abs(a - src[key]))), 'bitwise': bool(np.array_equal(a, src[key]))}
    for svc, s_label in SOURCE_LABEL.items():
        for name, _ in SLATE:
            key = f'{s_label}__pair__{name}'
            if key in src:
                a = scores[f'PAIR__{svc}__{name}']; replay[f'PAIR__{svc}__{name}'] = {'source_key': key, 'max_abs_diff': float(np.max(np.abs(a - src[key]))), 'bitwise': bool(np.array_equal(a, src[key]))}
    src_rec = json.loads((C.SRCP / f'runs/READERS_{seed}/READERS_SEED.json').read_text())
    win_agree = {}
    for svc, s_label in SOURCE_LABEL.items():
        for view in ('income', 'occupation', 'pair'):
            mine = rec['services'][svc]['winner'][view]['candidate']
            theirs = src_rec['services'][s_label]['winner'][view]['reader']
            win_agree[f'{svc}|{view}'] = {'this_campaign': mine, 'source': theirs, 'equal': mine == theirs}
    rec['replay_vs_source'] = {'compared': len(replay), 'bitwise': sum(v['bitwise'] for v in replay.values()),
                               'max_abs_diff': max((v['max_abs_diff'] for v in replay.values()), default=None), 'detail': replay, 'winners': win_agree}
    sp = out / 'DEV_REPORT_SCORES.npz'
    with sp.open('xb') as f:
        np.savez(f, row_id=R['row_id'], unit=R['unit'], **scores)
    sp.chmod(0o600)
    rec['dev_report_scores_sha256'] = C.sha(sp); rec['model_files_sha256'] = models
    n_marg = sum(len(m['readers']) for m in rec['marginal_banks'].values()); n_pair = sum(len(s['pair_readers']) for s in rec['services'].values())
    rec['fit_counts'] = {'marginal_unique_wire_fits': n_marg, 'pair_fits': n_pair, 'control_fits': 6, 'effective_numerical_fits': n_marg + n_pair + 6,
                         'aliased_service_views_not_refit': sum(len(v) - 1 for v in rec['aliases'].values()) * 7}
    rec['DEV_REPORT_labels_opened_or_hashed'] = False
    rec['sklearn_version'] = __import__('sklearn').__version__
    with (out / 'READERS_SEED.json').open('x') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.out)
    print(json.dumps({'seed': r['seed'], 'fit_counts': r['fit_counts'],
                      'winners': {k: {v: w['candidate'] for v, w in s['winner'].items()} for k, s in r['services'].items()},
                      'controls': r['controls'], 'replay': {k: r['replay_vs_source'][k] for k in ('compared', 'bitwise', 'max_abs_diff')},
                      'winner_agreement': sum(v['equal'] for v in r['replay_vs_source']['winners'].values())}, indent=1))
