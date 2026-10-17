"""Admit the neural reference deployment package (NEURAL_RAW rho1.0 E2, seed 0; the source study packaged seed 0 only).

The package is copied from the source private tree after checking its PACKAGE.json hash against the public MODEL_MANIFEST.
It is then run on the DEV_REPORT raw inputs with the source deployment code, and its output is compared bitwise with the
frozen NEURAL_RAW DEV_REPORT wire in this benchmark's final bank. No label is read.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from . import common as C
from .deploy_scm import raw_rows, sha
from .readers import load_service_bank


def main(out_root, report):
    man = json.loads((C.SRC_RESULTS / 'MODEL_MANIFEST.json').read_text())
    src = C.SRC_PRIV / 'deploy' / 'NEURAL_RAW_rho1.0_E2_S0'
    if sha(src / 'PACKAGE.json') != man['deployment_packages']['NEURAL_RAW_rho1.0_E2_S0']['package_json_sha256']:
        raise PermissionError('neural reference package hash differs from the public MODEL_MANIFEST')
    dst = Path(out_root) / 'NEURAL_RAW_rho1.0_E2_S0'
    if dst.exists():
        raise PermissionError('package exists')
    shutil.copytree(src, dst)
    rows = raw_rows('DEV_REPORT')
    res = SRC_DEPLOY.release(dst, rows)
    bank, ids, _ = load_service_bank(0)
    ok = {t: bool(np.array_equal(res[t]['probability'], bank['NEURAL_RAW']['DEV_REPORT'][t])) for t in C.TASKS}
    close = {t: float(np.max(np.abs(res[t]['probability'] - bank['NEURAL_RAW']['DEV_REPORT'][t]))) for t in C.TASKS}
    dec = {t: bool(np.array_equal(res[t]['decision'], bank['NEURAL_RAW']['DEV_REPORT'][t].argmax(1))) for t in C.TASKS}
    rec = json.loads(Path(report).read_text())
    rec['neural_reference_S0'] = {'package_json_sha256': sha(dst / 'PACKAGE.json'), 'reproduces_frozen_DEV_REPORT_wire_bitwise': ok,
                                  'max_abs_diff': close, 'decisions_identical': dec,
                                  'note': 'source package for seed 0 only; seed 1 neural package was never built by the source study (coverage gap)'}
    Path(report).write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps(rec['neural_reference_S0'], indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
