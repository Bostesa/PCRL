"""Engineering rehearsal of report.run on SYNTHETIC labels: no DEV_REPORT truth is read and no access event is logged.

The truth loader is replaced by random labels on the real DEV_REPORT row and household structure, and the commit guard by
a stub, so the full report path (bootstrap, endpoints, claims, controls, tables) runs end to end before the real re-read.
Outputs go to a scratch directory and are never results. Inheritance checks against the source values fail by design here.
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from . import common as C
from . import report


def main(out):
    R = C.load_role('DEV_REPORT', ('features',))
    n = len(R['row_id']); rng = np.random.default_rng(20261016)
    hh_sex = {u: int(rng.integers(0, 2)) for u in np.unique(R['unit'])}
    fake = {'row_id': R['row_id'], 'unit': R['unit'], 'X': R['X'], 'feature_names': R['feature_names'],
            'income': rng.integers(0, 2, n).astype(np.int64), 'occupation': rng.integers(0, 5, n).astype(np.int64),
            'sex': np.array([hh_sex[u] if rng.random() < 0.5 else int(rng.integers(0, 2)) for u in R['unit']], dtype=np.int64),
            'weights': rng.integers(1, 200, n).astype(np.int64)}
    C.dev_report_truth = lambda purpose: fake
    C._SYNTHETIC_REHEARSAL = True
    report.HALT_ON_SOURCE_REPRODUCTION = False   # synthetic labels cannot reproduce the published source values
    C.committed = lambda p: True
    res = report.run(out)
    print(json.dumps({'SYNTHETIC': True, 'claims': {k: v['decision'] for k, v in res['claims'].items()}, 'K': [len(res['endpoints_primary']), len(res['endpoints_secondary'])],
                      'checks_at_truth': res['checks_at_truth'], 'controls_valid': res['controls']['CONTROLS_VALID'],
                      'central': res['central_question']['overall']}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    main(ap.parse_args().out)
