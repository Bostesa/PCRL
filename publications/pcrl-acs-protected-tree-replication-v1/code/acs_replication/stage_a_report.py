"""Teacher headroom on MODEL_SELECT (both seeds) and the pre-registered usefulness gate.

Metrics per task and service (T_TREE, U0, Ucal): NLL, summed Brier, accuracy, gain over constant prediction
(accuracy, NLL, Brier), balanced accuracy, per-class recall. Headroom = U0 minus T_TREE (positive = teacher better).
Gate (SCIENCE_LOCK): the benchmark is SUPPORTED and the teacher USEFUL iff, for both tasks and both seeds, T_TREE has
positive NLL gain and positive accuracy gain over constant prediction on MODEL_SELECT and MODEL_FIT task-class support
passed. A useful teacher continues through the fixed protected recipe even if headroom over U0 is small or negative;
otherwise the study closes as UNSUPPORTED after this honest diagnostic comparison.
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from . import data as D
from . import registry


def run(out_path):
    if not D.lock_committed('SCIENCE_LOCK.json'):
        raise PermissionError('headroom report requires the committed SCIENCE_LOCK')
    S = D.load_role('MODEL_SELECT', ('features', 'income', 'occupation'))
    rec = {'schema': 'ACSR1-stage-a-headroom-v1', 'role': 'MODEL_SELECT', 'per_seed': {}, 'gate': {}}
    support_ok = D.receipt()['support']['status']['MODEL_FIT_task_classes']
    useful = []
    for seed in (0, 1):
        outs, _, _ = registry.load(seed, roles=('MODEL_SELECT',), families=('stage_a',))
        srec = {}
        for t in D.TASKS:
            k = D.K[t]
            sm = {svc: D.summary(S[t], outs[svc]['MODEL_SELECT'][t], k) for svc in ('T_TREE', 'U0', 'Ucal')}
            srec[t] = {'summaries': sm,
                       'headroom_vs_U0': {'nll': sm['U0']['nll'] - sm['T_TREE']['nll'], 'brier': sm['U0']['brier'] - sm['T_TREE']['brier'],
                                          'accuracy': sm['T_TREE']['accuracy'] - sm['U0']['accuracy'],
                                          'balanced_accuracy': sm['T_TREE']['balanced_accuracy'] - sm['U0']['balanced_accuracy']},
                       'headroom_vs_Ucal': {'nll': sm['Ucal']['nll'] - sm['T_TREE']['nll'], 'brier': sm['Ucal']['brier'] - sm['T_TREE']['brier'],
                                            'accuracy': sm['T_TREE']['accuracy'] - sm['Ucal']['accuracy']}}
            useful.append(sm['T_TREE']['nll_gain_over_constant'] > 0 and sm['T_TREE']['accuracy_gain_over_constant'] > 0)
        rec['per_seed'][str(seed)] = srec
    rec['gate'] = {'MODEL_FIT_task_class_support': bool(support_ok), 'teacher_useful_all': bool(all(useful)),
                   'decision': 'SUPPORTED_CONTINUE' if support_ok and all(useful) else 'UNSUPPORTED_CLOSE'}
    with open(out_path, 'x') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    r = run(ap.parse_args().out)
    print(json.dumps({'gate': r['gate'], 'headroom_vs_U0': {s: {t: r['per_seed'][s][t]['headroom_vs_U0'] for t in D.TASKS} for s in r['per_seed']}}, indent=1))
