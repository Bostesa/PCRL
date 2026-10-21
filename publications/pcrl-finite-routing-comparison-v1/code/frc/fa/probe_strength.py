"""Label-free design probe (coordinator, 2026-10-21 about 00:34Z; no label read): does the full-pool PRC continuation chain move beyond
the registered grid's top strength (8)? Uses only cached out-of-fold statistics (NRM1 task teachers, P1 SEX-side teachers) and the
saved twin book; continues the deterministic chain of nrm.fa.prc up an extended strength list and records, per strength, the people
re-routed relative to the twin, the fitted pi-weighted plug-in pair MI and the objective. It fits no release and saves nothing but its
record (results/<campaign>/fa/PROBE_STRENGTH_V<v>.json). Run through the queue:

    python3 -m frc.jobqueue run --unit FRC_PROBE_STRENGTH_V<v>_V1 --stage probe --rss-gib 2 -- python3 -m frc.fa.probe_strength <v>
"""
from __future__ import annotations
import json
import sys
import time

import numpy as np

from nrm.fa import prc as P
from frc import common as C
from frc.fa import route as R

EXTENDED = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def main(v):
    t0 = time.process_time()
    v = int(v)
    mf = C.features('MODEL_FIT')
    Z, _, unit, names, row_sha = R._prep(mf)
    E, task, ss = R.oof_statistics(v, row_sha, Z)
    dec = R.protection_decision()
    tb = R._twin_book_for(dec, v, E)
    k0, l0 = P.route(E, tb)
    saved = P.LAMBDA_GRID
    try:
        P.LAMBDA_GRID = EXTENDED                    # the source chain reads its module grid at call time (restored below)
        rp = P.route_params('PRC_S2', version=v)
        chain = P.protected_path(E, tb, k0, l0, list(EXTENDED), rp)
    finally:
        P.LAMBDA_GRID = saved
    out = {}
    prev = None
    for lam in EXTENDED:
        book, k, l = chain[lam]
        key = P.array_sha(k, l)
        out[str(lam)] = {'rerouted_vs_twin': float(((k != k0) | (l != l0)).mean()), 'income_rerouted': float((k != k0).mean()),
                         'occupation_rerouted': float((l != l0).mean()), 'fitted_pi_pair_mi': P.plugin_mi(k * 64 + l, E['pi']),
                         'objective': P.objective(E, book, k, l), 'tilt': float(book.get('tilt', 0.0)),
                         'same_partition_as_previous_strength': key == prev, 'partition_sha256': key}
        prev = key
    rec = {'stage': 'label_free_strength_probe', 'version': v, 'labels_read': False, 'extended_grid': list(EXTENDED),
           'twin_pi_pair_mi': P.plugin_mi(k0 * 64 + l0, E['pi']), 'per_strength': out, 'cpu_s': round(time.process_time() - t0, 1),
           'utc': R._utc(), 'note': 'design probe on cached MODEL_FIT out-of-fold statistics; no label, no release, no MODEL_SELECT read'}
    p = C.RESULTS / 'fa' / f'PROBE_STRENGTH_V{v}.json'
    C.write_json_x(p, rec)
    print(json.dumps({lam: (round(x['rerouted_vs_twin'], 4), round(x['fitted_pi_pair_mi'], 5), x['same_partition_as_previous_strength'])
                      for lam, x in out.items()}))


if __name__ == '__main__':
    main(sys.argv[1])
