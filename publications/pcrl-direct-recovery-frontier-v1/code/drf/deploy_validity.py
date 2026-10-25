"""Probability validity and input-name audit of the DRF1 deployment packages (coordinator; label-free; no record of any person is
read beyond the stored release wires the packages reproduce).

    python3 -m drf.deploy_validity --report results/<campaign>/deploy/DEPLOYMENT_VALIDITY.json

For every package under the campaign's private deploy/ folder: every stored wire of its release (all saved roles; the packages
reproduce these bitwise, deploy/DEPLOYMENT_CHECK*.json) is finite, inside [0, 1] and sums to one per row (worst absolute row-sum
error reported); the package's encoder has exactly the six permitted raw inputs and no encoded feature named after a forbidden field
(ohn.deploy.FORBIDDEN: SEX, the task targets, identifiers, weights, geography); no dependency file is label-like by name. Runtime
refusals are tested by `drf.deploy check`.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import deploy as D

PERMITTED = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')
LABEL_LIKE = re.compile(r'(?i)(label|y_inc|y_occ|pincp|occp|sex)')


def package(pkg: Path) -> dict:
    meta = json.loads((pkg / 'PACKAGE.json').read_text())
    with np.load(D.store_dir() / D._fname(meta['name'], meta['version'])) as z:
        arrays = {k: z[k].copy() for k in z.files}
    lo, hi, worst, finite = 1.0, 0.0, 0.0, True
    for a in arrays.values():
        finite &= bool(np.isfinite(a).all())
        lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
        worst = max(worst, float(np.abs(a.sum(1) - 1.0).max()))
    names = list(meta['encoder']['feature_names'])
    forbidden = [n for n in names if any(tok in D.OD.FORBIDDEN for tok in re.split(r'[^A-Za-z0-9]+', n.upper()))]
    deps = sorted(meta['depends_sha256'])
    label_like = [d for d in deps if LABEL_LIKE.search(d.split('::', 1)[1])]
    ok = (finite and lo >= 0.0 and hi <= 1.0 and worst <= 1e-12 and not forbidden and not label_like
          and tuple(meta['permitted_inputs']) == PERMITTED and not meta.get('randomized'))
    return {'name': meta['name'], 'version': meta['version'], 'arrays_checked': len(arrays), 'roles': sorted({k.split('__')[0] for k in arrays}),
            'all_finite': finite, 'min_probability': lo, 'max_probability': hi, 'worst_row_sum_error': worst,
            'permitted_inputs': meta['permitted_inputs'], 'n_encoded_features': len(names), 'forbidden_encoded_features': forbidden,
            'n_dependencies': len(deps), 'label_like_dependencies': label_like, 'randomized': bool(meta.get('randomized')), 'ok': ok}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--report', required=True)
    a = ap.parse_args(argv)
    t0 = time.process_time()
    pkgs = sorted(p.parent for p in (C.PRIV / 'deploy').glob('*/PACKAGE.json'))
    res = {p.name: package(p) for p in pkgs}
    out = {'schema': 'DRF1-deploy-validity-v1', 'utc': C.now(), 'labels_read': False, 'packages': res,
           'all_ok': bool(res) and all(r['ok'] for r in res.values()), 'cpu_s': round(time.process_time() - t0, 3),
           'note': 'stored wires = the packages\' outputs (bitwise, deploy/DEPLOYMENT_CHECK*.json); forbidden names checked in the encoder\'s '
                   'feature names and the dependency list; runtime refusals are in DEPLOYMENT_CHECK*.json'}
    C.write_json(Path(a.report) if Path(a.report).is_absolute() else C.ROOT / a.report, out)
    print(json.dumps({'packages': len(res), 'all_ok': out['all_ok']}))
    return 0 if out['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
