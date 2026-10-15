"""FIT2 (internal selection role), one seed: utility vs N_EQ and the frozen recovery proxy for the new controls.

Identical definitions to the original campaign (task_headroom.fit2_worker.proxy / proxy_folds and
task_headroom.data.clauses): 6 clauses per seed vs N_EQ (2 tasks x accuracy/NLL/Brier), labelled
in-sample U0/Ucal diagnostics, two-fold group cross-fit SEX proxy on FIT2 (LR_C1, HGB_15x200 on
[p, decision]) for the income, occupation and pair views. Opened only after the committed
COMPLETION_TRAINING_LOCK. ATT and INNER are not opened.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def run(spec_path):
    import warnings
    warnings.filterwarnings('ignore')
    from .evaluation_queue import queue_config
    from task_headroom.data import load_role, load_npz, clauses, summary, K
    from task_headroom.fit2_worker import proxy, proxy_folds
    from task_headroom import services
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    lock = json.loads(verify(cfg['training_lock'], cfg['training_lock_sha256']).read_text())
    if lock.get('status') != 'COMPLETION_TRAINING_FROZEN':
        raise PermissionError('FIT2 opens only after the committed COMPLETION_TRAINING_LOCK')
    seed = int(cfg['seed'])
    F = load_role('FIT2', cfg['roles']['FIT2'], verify, ('features', 'income', 'occupation', f'references_s{seed}'))
    sx = load_npz(cfg['fit2_sex'], verify, ('row_id', 'unit', 'sex'))
    if not (np.array_equal(sx['row_id'], F['row_id']) and np.array_equal(sx['unit'], F['unit'])):
        raise PermissionError('FIT2 SEX pairing differs')
    s = sx['sex'].astype(np.int64); folds = proxy_folds(F['unit'])
    labels = {'income': F['income'], 'occupation': F['occupation']}
    refs = {'U0': [F[f'references_s{seed}']['U0_1'], F[f'references_s{seed}']['U0_2']],
            'Ucal': [F[f'references_s{seed}']['Ucal_1'], F[f'references_s{seed}']['Ucal_2']]}
    neq = services.outputs(cfg['n_eq_service'], F, verify, seed)
    refs['N_EQ'] = [neq[0]['probability'], neq[1]['probability']]
    table = {}
    for name, svc in cfg['candidates'].items():
        guard()
        w = services.outputs(svc, F, verify, seed)
        probs = [w[0]['probability'], w[1]['probability']]
        v1, v2 = services.wire(w[0]), services.wire(w[1])
        t = {'internal': clauses(labels, probs, refs, seed, ('N_EQ',)),
             'diagnostic_in_sample': clauses(labels, probs, refs, seed, ('U0', 'Ucal')),
             'metrics': {tk: summary(labels[tk], probs[i], K[tk]) for i, tk in enumerate(('income', 'occupation'))},
             'proxy': proxy({'income': v1, 'occupation': v2, 'pair': np.column_stack((v1, v2))}, s, folds)}
        t['internal_point_feasible'] = all(c['point_pass'] for c in t['internal'])
        t['max_normalized_violation'] = max(c['normalized'] for c in t['internal'])
        t['summed_nll'] = sum(t['metrics'][tk]['nll'] for tk in ('income', 'occupation'))
        table[name] = t
    result = {'schema': 'TC1-fit2-seed-v1', 'queue_unit_id': active['unit_id'], 'seed': seed, 'table': table,
              'proxy_definition': 'identical to task_headroom.fit2_worker (two-fold group cross-fit on FIT2, max of LR_C1/HGB_15x200 pooled OOF AUC)',
              'ATT_opened': False, 'INNER_opened': False}
    with Path(cfg['output']).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
