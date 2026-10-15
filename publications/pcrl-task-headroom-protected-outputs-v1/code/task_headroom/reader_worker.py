"""Fresh output-only readers, one seed, every service on the frozen SELECTION_LOCK attack roster.

Readers train on ATT train groups with SEX, are oriented and the per-view
winner is selected on ATT validation groups only, then INNER is scored from
released outputs alone. INNER labels are never opened here. Every service gets
the same 21 direct fits (7 readers x 3 views); pair candidates also include the
14 ignore-one-recipient aliases. Controls on the T_TREE wire: group-shuffled SEX
and a planted SEX bit (OUT_HGB), per view.
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

SLATE = (('OUT_LR', 'out'), ('OUT_MLP', 'out'), ('OUT_HGB', 'out'), ('LOG_LR', 'log'), ('LOG_MLP', 'log'), ('DEC_LR', 'dec'), ('CONF_HGB', 'conf'))
K = {'income': 2, 'occupation': 6}


def kinds(rec, k):
    p = np.asarray(rec['probability'], dtype=np.float64); d = np.asarray(rec['decision'], dtype=np.int64)
    one = np.eye(k)[d]
    return {'out': np.column_stack((p, d.astype(np.float64))), 'log': np.column_stack((np.log(np.maximum(p, 1e-300)), one)),
            'dec': one, 'conf': p.max(1, keepdims=True)}


def tables(w):
    a, b = kinds(w[0], 2), kinds(w[1], 6)
    return {'income': a, 'occupation': b, 'pair': {k: np.column_stack((a[k], b[k])) for k in a}}


def wire_sha(w):
    h = hashlib.sha256()
    for rec in w:
        h.update(np.ascontiguousarray(rec['probability'], dtype=np.float64).tobytes()); h.update(np.ascontiguousarray(rec['decision'], dtype=np.int64).tobytes())
    return h.hexdigest()


def run(spec_path):
    from .evaluation_queue import queue_config
    from .data import load_role, load_npz
    from . import services
    from task_conditioned_representation.attacks import (MODEL_SEED_BASE, ATT_NULL_SEED, attack_split, fit_reader, predict_reader,
                                                         orientation, orient, group_shuffle)
    from task_conditioned_representation.inference import auc
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    sel = json.loads(verify(cfg['selection_lock'], cfg['selection_lock_sha256']).read_text())
    if sel.get('status') != 'NOMINATIONS_AND_ATTACK_ROSTER_FROZEN' or sel.get('INNER_truth_accessed') is not False:
        raise PermissionError('reader fits require the committed SELECTION_LOCK')
    seed = int(cfg['seed'])
    att = load_role('ATT', cfg['roles']['ATT'], verify, ('features', 'sex'))
    inner = load_role('INNER', cfg['roles']['INNER'], verify, ('features',))
    y = att['sex']; train, val = attack_split(att['unit'])
    out = Path(cfg['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private reader directory required')
    out.mkdir(parents=True, mode=0o700)
    seed_model = MODEL_SEED_BASE + seed
    receipt = {'schema': 'THP1-output-readers-v1', 'queue_unit_id': active['unit_id'], 'seed': seed, 'model_seed': seed_model,
               'ATT_split': {'train_rows': int(len(train)), 'validation_rows': int(len(val))}, 'services': {}, 'accounting': []}
    scores = {}
    for label, svc in cfg['services'].items():
        guard()
        wa = services.outputs(svc, att, verify, seed); wi = services.outputs(svc, inner, verify, seed)
        ta, ti = tables(wa), tables(wi)
        rel = out / f'INNER_RELEASE__{label}.npz'
        with rel.open('xb') as f:
            np.savez(f, row_id=inner['row_id'], p1=wi[0]['probability'], d1=wi[0]['decision'], p2=wi[1]['probability'], d2=wi[1]['decision'])
        srec = {'wire_sha256': {'ATT': wire_sha(wa), 'INNER': wire_sha(wi)}, 'validation': {}, 'winner': {}}
        single = {}
        for view in ('income', 'occupation', 'pair'):
            srec['validation'][view] = {}
            for name, kind in SLATE:
                t0 = time.process_time()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rec = fit_reader(ta[view][kind][train], y[train], name, seed_model)
                mp = out / f'MODEL__{label}__{view}__{name}.joblib'; joblib.dump(rec, mp, compress=3); mp.chmod(0o600)
                srec.setdefault('model_sha256', {})[mp.name] = hashlib.sha256(mp.read_bytes()).hexdigest()
                raw = predict_reader(rec, ta[view][kind][val]); sign = orientation(y[val], raw)
                srec['validation'][view][name] = {'fixed_orientation_auc': auc(y[val], orient(raw, sign)), 'orientation': sign}
                s_inner = orient(predict_reader(rec, ti[view][kind]), sign)
                scores[f'{label}__{view}__{name}'] = s_inner
                receipt['accounting'].append({'service': label, 'view': view, 'reader': name, 'cpu_s': time.process_time() - t0})
                if view != 'pair':
                    single[f'IGNORE_{"occupation" if view == "income" else "income"}__{name}'] = (srec['validation'][view][name]['fixed_orientation_auc'], s_inner)
            cands = {n: v['fixed_orientation_auc'] for n, v in srec['validation'][view].items()}
            if view == 'pair':
                for n, (a, s_inner) in single.items():
                    cands[n] = a; scores[f'{label}__pair__{n}'] = s_inner
            order = list(cands)
            win = max(order, key=lambda n: (cands[n], -order.index(n)))
            srec['winner'][view] = {'reader': win, 'att_validation_auc': cands[win]}
        receipt['services'][label] = srec
    # Controls on the T_TREE wire.
    guard()
    wa = services.outputs(cfg['services']['T_TREE'], att, verify, seed); wi = services.outputs(cfg['services']['T_TREE'], inner, verify, seed)
    ta, ti = tables(wa), tables(wi); shuffled = group_shuffle(y[train], att['unit'][train], ATT_NULL_SEED); controls = {}
    for view in ('income', 'occupation', 'pair'):
        x, xi = ta[view]['out'], ti[view]['out']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rs = fit_reader(x[train], shuffled, 'OUT_HGB', seed_model)
            rp = fit_reader(np.column_stack((x[train], y[train])), y[train], 'OUT_HGB', seed_model)
        scores[f'CONTROL__{view}__SHUFFLED'] = predict_reader(rs, xi)
        for bit in (0, 1):
            scores[f'CONTROL__{view}__PLANTED_{bit}'] = predict_reader(rp, np.column_stack((xi, np.full(len(xi), bit))))
        controls[view] = {'planted_att_val_auc': auc(y[val], predict_reader(rp, np.column_stack((x[val], y[val]))))}
    receipt['controls'] = controls
    sp = out / 'INNER_SCORES.npz'
    with sp.open('xb') as f:
        np.savez(f, row_id=inner['row_id'], unit=inner['unit'], **scores)
    receipt['inner_scores_sha256'] = hashlib.sha256(sp.read_bytes()).hexdigest()
    receipt['numerical_fit_count'] = len(receipt['accounting']) + 6
    receipt['INNER_labels_opened_or_hashed'] = False
    receipt['sklearn_version'] = __import__('sklearn').__version__
    with (out / 'FROZEN_COMPLETE.json').open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    with Path(cfg['output']).open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
