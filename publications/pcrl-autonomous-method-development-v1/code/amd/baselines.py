"""MODEL_SELECT development table of the frozen historical releases (no fit; counts_toward_36 = 0).

Measures, with the same harness as every new configuration (amd.evaluate.measure), the eligible references NEURAL_RAW and
H_LJ, the tree recipe's releases (T_TREE teacher, TASK_REFIT, LOCAL_TREE = H_LL, JOINT_TREE = H_JJ, H_JL), U0 and Ucal, and the
SCM follow-up's SCM (t = 0.008) and PUB_IBSI (gamma = 2, MODEL_SELECT-matched) releases. Writes runs/BASELINES/BASELINES_MS_S<seed>.json.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

from . import common as C
from . import evaluate as EV
from . import ledger as L

FROZEN = ('NEURAL_RAW', 'H_LJ', 'H_JJ', 'H_LL', 'H_JL', 'T_TREE', 'TASK_REFIT', 'U0', 'Ucal')
UM = ('SCM', 'IBSI_NLLM')


def main(seeds):
    for s in seeds:
        out_p = C.RUNS / 'BASELINES' / f'BASELINES_MS_S{s}.json'
        res = {}
        S, refs, bank = EV.ms_context(s, purpose=f'baseline table of frozen releases S{s}')
        um = C.um_service_bank(s, roles=('MODEL_SELECT',))
        for name in FROZEN + UM:
            src = bank if name in FROZEN else um
            outs = {t: np.asarray(src[name]['MODEL_SELECT'][t], dtype=np.float64) for t in C.TASKS}
            m = EV.measure(s, outs, S, refs)
            res.setdefault(name, {})[str(s)] = m
            L.append({'family': 'REF', 'version': 'frozen', 'config_id': name, 'seed': s, 'kind': 'reference', 'counts_toward_36': 0,
                      'code_sha256': L.code_sha(__file__, EV.__file__), 'unit_id': os.environ.get('AMD_UNIT_ID', ''),
                      'hypothesis': 'historical frozen release (descriptive bar)', 'change_from_previous': 'none',
                      'roles_read': 'MODEL_SELECT (labels, SEX: evaluation)', 'training_effort': 'none (frozen)', 'selection_effort': 'none',
                      **L.summarize(m)})
            print(name, s, round(m['proxy']['pair']['max'], 4), m['point_pass_12'], m['guard_pass'], flush=True)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]] or list(C.SEEDS))
