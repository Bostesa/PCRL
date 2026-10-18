"""Label-free check of how different family A's two seeds are (claims review, second pass, item 7).

    python3 -m ohn.seed_identity --configs A1_PO_16 A1_PO_8 --out results/<campaign>/checks/A_SEED_IDENTITY.json

The family-A learners are HistGradientBoosting fits on all MODEL_FIT people with early_stopping=False and no subsampling; only the
tree count (chosen on seed-specific household folds) and random_state differ by seed. If the fit is deterministic, the seed-1 model
equals the seed-0 boosting path truncated at the seed-1 tree count. This compares, on MODEL_SELECT features (no label), the seed-1
model's raw scores with the seed-0 model's staged raw scores at that iteration, and the parameters of the two fits.
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from . import common as C
from . import fam_a as A


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    M = C.load_role('MODEL_SELECT', ('features',))
    out = {'schema': 'OHN1-seed-identity-v1', 'utc': C.now(), 'labels_read': False, 'role': 'MODEL_SELECT features', 'configs': {}}
    for cfg in a.configs:
        (m0s, st0), (m1s, st1) = A.load_models(0, cfg), A.load_models(1, cfg)
        if len(m0s) != 1 or len(m1s) != 1 or st0['rep'] != st1['rep']:
            out['configs'][cfg] = {'comparable': False}; continue
        m0, m1 = m0s[0], m1s[0]
        Xr = A.rep_matrix(st0['rep'], M['X'], M['feature_names'])
        p0, p1 = m0.get_params(), m1.get_params()
        diff_params = {k: (p0[k], p1[k]) for k in p0 if repr(p0[k]) != repr(p1[k])}       # repr: array-valued parameters
        z1 = m1.decision_function(Xr)
        z0t = None
        for i, z in enumerate(m0.staged_decision_function(Xr), 1):
            if i == m1.n_iter_:
                z0t = z; break
        rec = {'comparable': True, 'n_iter': [int(m0.n_iter_), int(m1.n_iter_)],
               'params_that_differ': {k: [str(v[0]), str(v[1])] for k, v in diff_params.items()},
               'seed1_equals_seed0_truncated_bitwise': bool(z0t is not None and np.array_equal(z0t, z1)),
               'max_abs_raw_diff': None if z0t is None else float(np.max(np.abs(z0t - z1))),
               'release_seed1_vs_seed0_full_max_abs_prob_diff': float(np.max(np.abs(A.release_prob(m1s, Xr, st1.get('tau'))
                                                                                       - A.release_prob(m0s, Xr, st0.get('tau'))))),
               'decision_agreement_seed1_vs_seed0_full': float(np.mean(A.release_prob(m1s, Xr, st1.get('tau')).argmax(1)
                                                                       == A.release_prob(m0s, Xr, st0.get('tau')).argmax(1)))}
        out['configs'][cfg] = rec
    C.write_json(a.out, out)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
