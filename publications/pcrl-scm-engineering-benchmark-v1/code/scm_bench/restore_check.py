"""Restore-and-deploy check, run inside a restored copy (backup alone): the restored SCM and TWIN packages, given the restored
DEV_REPORT raw inputs (features and raw fields only; no labels), must reproduce the restored attacked DEV_REPORT wires bitwise."""
from __future__ import annotations
import json
import sys

import numpy as np

from . import common as C
from .deploy_scm import raw_rows, release


def main(out):
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    rows = raw_rows('DEV_REPORT'); res = {}
    for s in C.SEEDS:
        with np.load(C.PRIV / 'runs' / f'FIT_{s}' / 'ARM_OUTPUTS.npz') as z:
            for label in ('SCM', 'TWIN'):
                arm = sel['bank_arm_map'][label]
                r = release(C.PRIV / 'deploy' / f'{label}_S{s}', rows)
                res[f'{label}_S{s}'] = all(np.array_equal(r[t]['probability'], z[f'{arm}__DEV_REPORT__{t}']) for t in C.TASKS)
    rec = {'deployment_from_backup_reproduces_attacked_wire_bitwise': res, 'all_ok': all(res.values())}
    with open(out, 'w') as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec))


if __name__ == '__main__':
    main(sys.argv[1])
