"""Dry run of the report on INVENTED labels (no DEV_REPORT label is opened): DEV_REPORT row ids and households with income,
occupation and SEX drawn at random (fixed seed), passed to amd.report.run as `truth`. Exercises every code path of the
frozen roster before the REPORTING_LOCK; its numbers mean nothing and are written to a private scratch directory only.
"""
from __future__ import annotations
import json
import sys
import tempfile

import numpy as np

from . import common as C
from . import report as R


def main(out=None):
    D = C.load_role('DEV_REPORT', ('features',))
    rng = np.random.default_rng(20261017)
    n = len(D['row_id'])
    truth = {'row_id': D['row_id'], 'unit': D['unit'], 'income': rng.integers(0, 2, n), 'occupation': rng.integers(0, 5, n),
             'sex': rng.integers(0, 2, n)}
    out = out or tempfile.mkdtemp(dir=C.PRIV)
    res = R.run(out, truth=truth)
    print(json.dumps({'dry_run': 'INVENTED LABELS', 'claims': {k: v['label'] for k, v in res['claims'].items()},
                      'n_endpoints': len(res['endpoints']), 'out': str(out)}))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
