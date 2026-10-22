"""Designer A (campaign pcrl_dual_design_release_v1): list the structure of the two cached teacher pickles named in the brief.
Label-free: prints keys, shapes and fitted-model settings only. No label file, nothing of MODEL_SELECT, no reader cache."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import pickle
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]          # the worktree root (results/<campaign>/roles/designer_a/this.py)
TASK = '.private/pcrl_new_release_mechanisms_v1/fa_cache/V{v}/TASK.pkl'
SIDE = '.private/pcrl_finite_routing_comparison_v1/fa_cache/V{v}/SEXSIDE.pkl'


def show(name, obj, depth=0):
    pad = '  ' * depth
    if isinstance(obj, dict):
        print(f'{pad}{name}: dict keys={list(obj.keys())}')
        for k, v in obj.items():
            show(str(k), v, depth + 1)
    elif isinstance(obj, np.ndarray):
        print(f'{pad}{name}: ndarray {obj.shape} {obj.dtype}')
    elif isinstance(obj, (list, tuple)):
        print(f'{pad}{name}: {type(obj).__name__} len={len(obj)} head={list(obj)[:8]}')
    else:
        print(f'{pad}{name}: {type(obj).__name__} {repr(obj)[:200]}')


if __name__ == '__main__':
    t0 = time.process_time()
    for v in (0, 1):
        for rel in (TASK.format(v=v), SIDE.format(v=v)):
            with open(ROOT / rel, 'rb') as f:
                obj = pickle.load(f)
            print('==', rel)
            show('root', obj)
    print(f'CPU_SECONDS {time.process_time() - t0:.2f}')
