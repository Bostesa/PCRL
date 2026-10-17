"""Restore-and-deploy check, run inside a restored copy (backup alone): every restored SCM-format package, given the restored
DEV_REPORT raw inputs (no labels), must reproduce the restored evaluated DEV_REPORT wires bitwise through the API and CSV paths."""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import common as C
from .deploy import raw_rows, release, release_csv, write_csv, PERMITTED
from .readers import load_service_bank


def main(out):
    rec = json.loads((C.RESULTS / 'DEPLOYMENT_RESULTS.json').read_text())
    rows = raw_rows(); tmp = Path(tempfile.mkdtemp()); p = tmp / 'raw.csv'; write_csv(p, rows, PERMITTED)
    res = {}
    for name, pk in rec['packages'].items():
        if 'release' not in pk:
            continue                                   # frozen references are checked by their own source code paths
        role, s = name.rsplit('_S', 1); s = int(s)
        bank, _, man = load_service_bank(s)
        ref = bank[man['per_seed'][f'S{s}']['service_of_role'][role]]['DEV_REPORT']
        d = C.PRIV / 'deploy' / name
        a, b = release(d, rows), release_csv(d, p)
        res[name] = all(np.array_equal(a[t]['probability'], ref[t]) and np.array_equal(b[t]['probability'], ref[t]) for t in C.TASKS)
    out_rec = {'deployment_from_backup_reproduces_evaluated_wire_bitwise': res, 'all_ok': bool(res) and all(res.values())}
    Path(out).write_text(json.dumps(out_rec, indent=1))
    print(json.dumps(out_rec))


if __name__ == '__main__':
    main(sys.argv[1])
