"""Dry runs on INVENTED labels (real features, invented task labels and SEX) so the selection and report code paths are exercised
end to end before they ever touch real labels. The real MODEL_SELECT and DEV_REPORT label files are never opened here.

    python -m scm_um.dryrun select OUT_JSON     # scm_um.select with invented MODEL_SELECT labels and SEX
    python -m scm_um.dryrun report OUT_DIR      # scm_um.report with an invented DEV_REPORT truth (no access event is logged)
Outputs go to a scratch location and are deleted after the structural check; their values mean nothing.
"""
from __future__ import annotations
import json
import sys

import numpy as np

from . import common as C


def invented(role, n, seed):
    r = np.random.default_rng(seed)
    return {'income': r.integers(0, 2, n).astype(np.int64), 'occupation': r.integers(0, 5, n).astype(np.int64), 'sex': r.integers(0, 2, n).astype(np.int64)}


def main(mode, out):
    real_load = C.load_role

    def fake_load(role, wanted=('features',)):
        base = real_load(role, ('features',) + tuple(w for w in wanted if w == 'raw_inputs'))
        inv = invented(role, len(base['row_id']), 20261017)
        for w in wanted:
            if w in ('income', 'occupation', 'sex'):
                base[w] = inv[w]
        return base
    if mode == 'select':
        from . import select as S
        S.C.load_role = fake_load
        r = S.run(out)
        print(json.dumps({'status': r['status'], 't_star': r['nomination']['t_star'], 'roles_seed0': sorted(r['roles']['0']),
                          'services_seed0': r['services']['0']['aliases'], 'DRY_RUN': 'invented labels; values meaningless'}, indent=1))
    elif mode == 'report':
        from . import report as R
        D = real_load('DEV_REPORT', ('features',))
        truth = {'row_id': D['row_id'], 'unit': D['unit'], **invented('DEV_REPORT', len(D['row_id']), 20261018)}
        r = R.run(out, truth=truth)
        print(json.dumps({'claims': {k: (v['decision'] if isinstance(v, dict) and 'decision' in v else v) for k, v in r['claims'].items()},
                          'n_primary': len(r['endpoints_primary']), 'n_secondary': len(r['endpoints_secondary']), 'controls': r['controls'],
                          'research_decision': r['research_decision'], 'DRY_RUN': 'invented labels; values meaningless'}, indent=1, default=str))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
