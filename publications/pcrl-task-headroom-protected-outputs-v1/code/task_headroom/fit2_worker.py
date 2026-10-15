"""FIT2 (internal selection role), one seed: utility vs N_EQ and the predeclared recovery proxy.

Opened after the committed TRAINING_COMPLETE_LOCK. For every saved checkpoint
of every arm and strength, the teacher, N_EQ, RAW_PROJ and every historical
comparator: 12 clauses per seed... (6 per seed: 2 tasks x 3 metrics) vs N_EQ,
labelled diagnostics vs in-sample U0/Ucal, task summaries, and the two-fold
group cross-fit FIT2 SEX proxy (LR_C1, HGB_15x200 on [p, decision]) for the
income, occupation and pair views. ATT and INNER are not opened.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PROXY_SEED = 2026101502
PROXY_READERS = ('LR_C1', 'HGB_15x200')


def proxy_folds(unit):
    return np.asarray([int(hashlib.sha256(f'THP1|FIT2-proxy|{int(u)}'.encode()).hexdigest(), 16) % 2 for u in unit])


def proxy(views, sex, folds):
    from task_conditioned_representation.attacks import fit_reader, predict_reader
    from task_conditioned_representation.inference import auc
    out = {}
    for name, x in views.items():
        vals = {}
        for reader in PROXY_READERS:
            oof = np.zeros(len(sex))
            for f in (0, 1):
                tr, te = folds != f, folds == f
                oof[te] = predict_reader(fit_reader(x[tr], sex[tr], reader, PROXY_SEED), x[te])
            vals[reader] = auc(sex, oof)
        out[name] = {'readers': vals, 'value': max(vals.values())}
    return out


def run(spec_path):
    import warnings
    warnings.filterwarnings('ignore')
    from .evaluation_queue import queue_config
    from .data import load_role, load_npz, clauses, summary, K
    from . import services
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    lock = json.loads(verify(cfg['training_complete_lock'], cfg['training_complete_lock_sha256']).read_text())
    if lock.get('status') != 'STAGE_B_TRAINING_FROZEN':
        raise PermissionError('FIT2 opens only after the committed TRAINING_COMPLETE_LOCK')
    seed = int(cfg['seed'])
    F = load_role('FIT2', cfg['roles']['FIT2'], verify, ('features', 'income', 'occupation', f'references_s{seed}'))
    sex = load_npz(cfg['fit2_sex'], verify, ('row_id', 'unit', 'sex'))
    if not (np.array_equal(sex['row_id'], F['row_id']) and np.array_equal(sex['unit'], F['unit'])):
        raise PermissionError('FIT2 SEX pairing differs')
    s = sex['sex'].astype(np.int64); folds = proxy_folds(F['unit'])
    labels = {'income': F['income'], 'occupation': F['occupation']}
    refs = {'U0': [F[f'references_s{seed}']['U0_1'], F[f'references_s{seed}']['U0_2']],
            'Ucal': [F[f'references_s{seed}']['Ucal_1'], F[f'references_s{seed}']['Ucal_2']]}
    neq = services.outputs(cfg['services']['N_EQ'], F, verify, seed)
    refs['N_EQ'] = [neq[0]['probability'], neq[1]['probability']]
    table = {}
    label_only = proxy({'labels': np.eye(12)[F['income'] * 6 + F['occupation']]}, s, folds)['labels']

    def score(name, svc):
        guard()
        w = services.outputs(svc, F, verify, seed)
        probs = [w[0]['probability'], w[1]['probability']]
        v1, v2 = services.wire(w[0]), services.wire(w[1])
        table[name] = {'internal': clauses(labels, probs, refs, seed, ('N_EQ',)),
                       'diagnostic_in_sample': clauses(labels, probs, refs, seed, ('U0', 'Ucal')),
                       'metrics': {t: summary(labels[t], probs[i], K[t]) for i, t in enumerate(('income', 'occupation'))},
                       'proxy': proxy({'income': v1, 'occupation': v2, 'pair': np.column_stack((v1, v2))}, s, folds)}
        table[name]['internal_point_feasible'] = all(c['point_pass'] for c in table[name]['internal'])
        table[name]['max_normalized_violation'] = max(c['normalized'] for c in table[name]['internal'])
        table[name]['summed_nll'] = sum(table[name]['metrics'][t]['nll'] for t in ('income', 'occupation'))
    for name, svc in cfg['services'].items():
        score(name, svc)
    for name, svc in cfg['checkpoints'].items():
        score(name, svc)
    result = {'schema': 'THP1-fit2-seed-v1', 'queue_unit_id': active['unit_id'], 'seed': seed, 'table': table,
              'label_only_reference_auc': label_only, 'proxy_definition': 'two-fold group cross-fit on FIT2, max of LR_C1/HGB_15x200 pooled OOF AUC',
              'fit2_rows': int(len(F['row_id'])), 'ATT_opened': False, 'INNER_opened': False}
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
