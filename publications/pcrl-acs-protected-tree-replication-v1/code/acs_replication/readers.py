"""Fresh output-only readers, one seed, every service on the frozen SELECTION_LOCK attack roster.

Transport of task_headroom/reader_worker.py: the same 7-reader slate (OUT_LR, OUT_MLP, OUT_HGB on probabilities+decision;
LOG_LR, LOG_MLP on log-probabilities + one-hot decision; DEC_LR on the one-hot decision; CONF_HGB on the max probability)
x 3 views (income, occupation, coalition pair), validated sklearn reader definitions (task_conditioned_representation.attacks).
ACS roles: readers train on ATTACK_FIT with SEX; orientation and the per-view winner are chosen on ATTACK_VALIDATION only
(pair candidates include the 14 ignore-one-recipient aliases); DEV_REPORT is scored from released outputs alone
(DEV_REPORT labels are never opened here). Real-data controls on the T_TREE wire: a shuffled-SEX null (fixed-seed row
permutation of ATTACK_FIT SEX) and a planted SEX bit (OUT_HGB), per view.
The adversary sees auxiliary labelled releases (ATTACK_FIT) and the target release only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

from . import data as D
from . import registry

SLATE = (('OUT_LR', 'out'), ('OUT_MLP', 'out'), ('OUT_HGB', 'out'), ('LOG_LR', 'log'), ('LOG_MLP', 'log'), ('DEC_LR', 'dec'), ('CONF_HGB', 'conf'))
NULL_SEED = 2026101602


def kinds(p, k):
    p = np.asarray(p, dtype=np.float64); d = p.argmax(1); one = np.eye(k)[d]
    return {'out': np.column_stack((p, d.astype(np.float64))), 'log': np.column_stack((np.log(np.maximum(p, 1e-300)), one)),
            'dec': one, 'conf': p.max(1, keepdims=True)}


def tables(svc_role):
    a, b = kinds(svc_role['income'], 2), kinds(svc_role['occupation'], 5)
    return {'income': a, 'occupation': b, 'pair': {k: np.column_stack((a[k], b[k])) for k in a}}


def wire_sha(svc_role):
    h = hashlib.sha256()
    for t, k in (('income', 2), ('occupation', 5)):
        p = np.asarray(svc_role[t], dtype=np.float64)
        h.update(np.ascontiguousarray(p).tobytes()); h.update(np.ascontiguousarray(p.argmax(1).astype(np.int64)).tobytes())
    return h.hexdigest()


def run(seed, out_dir):
    from task_conditioned_representation.attacks import MODEL_SEED_BASE, fit_reader, predict_reader, orientation, orient
    from task_conditioned_representation.inference import auc
    sel_path = D.RESULTS / 'SELECTION_LOCK.json'
    if not D.lock_committed('SELECTION_LOCK.json'):
        raise PermissionError('reader fits require the committed SELECTION_LOCK')
    sel = json.loads(sel_path.read_text())
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh reader directory required')
    out.mkdir(parents=True, mode=0o700)
    A = D.load_role('ATTACK_FIT', ('features', 'sex')); V = D.load_role('ATTACK_VALIDATION', ('features', 'sex'))
    R = D.load_role('DEV_REPORT', ('features',))
    outs, hashes, _ = registry.load(seed, roles=('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
    seed_model = MODEL_SEED_BASE + seed
    receipt = {'schema': 'ACSR1-output-readers-v1', 'seed': seed, 'model_seed': seed_model, 'roles': {'train': 'ATTACK_FIT', 'select': 'ATTACK_VALIDATION', 'score': 'DEV_REPORT'},
               'rows': {'ATTACK_FIT': int(len(A['row_id'])), 'ATTACK_VALIDATION': int(len(V['row_id'])), 'DEV_REPORT': int(len(R['row_id']))},
               'services': {}, 'accounting': [], 'output_file_sha256': hashes, 'selection_lock_sha256': registry.sha(sel_path)}
    scores = {}
    for label in sel['attack_roster']:
        svc = outs[label]
        ta, tv, tr = tables(svc['ATTACK_FIT']), tables(svc['ATTACK_VALIDATION']), tables(svc['DEV_REPORT'])
        srec = {'wire_sha256': {r: wire_sha(svc[r]) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')}, 'validation': {}, 'winner': {}}
        single = {}
        for view in ('income', 'occupation', 'pair'):
            srec['validation'][view] = {}
            for name, kind in SLATE:
                t0 = time.process_time()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rec = fit_reader(ta[view][kind], A['sex'], name, seed_model)
                mp = out / f'MODEL__{label}__{view}__{name}.joblib'; joblib.dump(rec, mp, compress=3); mp.chmod(0o600)
                srec.setdefault('model_sha256', {})[mp.name] = registry.sha(mp)
                raw = predict_reader(rec, tv[view][kind]); sign = orientation(V['sex'], raw)
                srec['validation'][view][name] = {'fixed_orientation_auc': auc(V['sex'], orient(raw, sign)), 'orientation': sign}
                s_rep = orient(predict_reader(rec, tr[view][kind]), sign)
                scores[f'{label}__{view}__{name}'] = s_rep
                receipt['accounting'].append({'service': label, 'view': view, 'reader': name, 'cpu_s': time.process_time() - t0})
                if view != 'pair':
                    single[f'IGNORE_{"occupation" if view == "income" else "income"}__{name}'] = (srec['validation'][view][name]['fixed_orientation_auc'], s_rep)
            cands = {n: v['fixed_orientation_auc'] for n, v in srec['validation'][view].items()}
            if view == 'pair':
                for n, (a, s_rep) in single.items():
                    cands[n] = a; scores[f'{label}__pair__{n}'] = s_rep
            order = list(cands)
            win = max(order, key=lambda n: (cands[n], -order.index(n)))
            srec['winner'][view] = {'reader': win, 'attack_validation_auc': cands[win]}
        receipt['services'][label] = srec
    # Real-data controls on the T_TREE wire.
    svc = outs['T_TREE']; ta, tv, tr = tables(svc['ATTACK_FIT']), tables(svc['ATTACK_VALIDATION']), tables(svc['DEV_REPORT'])
    shuffled = np.random.default_rng(NULL_SEED + seed).permutation(A['sex'])
    controls = {}
    for view in ('income', 'occupation', 'pair'):
        x, xv, xr = ta[view]['out'], tv[view]['out'], tr[view]['out']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rs = fit_reader(x, shuffled, 'OUT_HGB', seed_model)
            rp = fit_reader(np.column_stack((x, A['sex'])), A['sex'], 'OUT_HGB', seed_model)
        scores[f'CONTROL__{view}__SHUFFLED'] = predict_reader(rs, xr)
        for bit in (0, 1):
            scores[f'CONTROL__{view}__PLANTED_{bit}'] = predict_reader(rp, np.column_stack((xr, np.full(len(xr), bit))))
        controls[view] = {'shuffled_attack_validation_auc': auc(V['sex'], predict_reader(rs, xv)),
                          'planted_attack_validation_auc': auc(V['sex'], predict_reader(rp, np.column_stack((xv, V['sex']))))}
    receipt['controls'] = controls
    sp = out / 'DEV_REPORT_SCORES.npz'
    with sp.open('xb') as f:
        np.savez(f, row_id=R['row_id'], unit=R['unit'], **scores)
    sp.chmod(0o600)
    receipt['dev_report_scores_sha256'] = registry.sha(sp)
    receipt['numerical_fit_count'] = len(receipt['accounting']) + 6
    receipt['DEV_REPORT_labels_opened_or_hashed'] = False
    receipt['sklearn_version'] = __import__('sklearn').__version__
    with (out / 'READERS_SEED.json').open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return receipt


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.out)
    print(json.dumps({'seed': r['seed'], 'winners': {k: {v: w['reader'] for v, w in s['winner'].items()} for k, s in r['services'].items()},
                      'controls': r['controls']}))
