"""PCRL recovery-estimator diagnostic (campaign `pcrl_estimator_diagnostic_v1`, EDG1): a synthetic measurement study of the
estimators DRF1 used to guide method development. Synthetic known-law data only: no census record, label, model or private array
is read anywhere in this package.

Imports the published DRF1 estimator code (publications/pcrl-direct-recovery-frontier-v1/code, commit c4896479e) and the vendored,
hash-checked DRF1 reader recipes (code/vendor; SOURCE_PROVENANCE.json)."""
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

CODE = Path(__file__).resolve().parent.parent             # results/pcrl_estimator_diagnostic_v1/code
RESULTS = CODE.parent                                      # results/pcrl_estimator_diagnostic_v1
WORKTREE = RESULTS.parent.parent
PUB_CODE = WORKTREE / 'publications' / 'pcrl-direct-recovery-frontier-v1' / 'code'
VENDOR = CODE / 'vendor'
OUT = Path(os.environ.get('EDG_OUT', str(RESULTS / 'out')))                 # override only for smoke tests
REPORT = Path(os.environ.get('EDG_REPORT', str(RESULTS)))                   # where analyze writes SUMMARY / TABLES / figures
for _p in (str(PUB_CODE), str(VENDOR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
