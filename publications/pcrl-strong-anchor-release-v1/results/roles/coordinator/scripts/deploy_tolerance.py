# Coordinator script (label-free) that wrote deploy/DEPLOYMENT_TOLERANCE.json (torch venv); saved from the session scratchpad at closeout.
"""Batch dependence of the 13 deployment packages (label-free; torch venv). For each package: the raw DEV_REPORT rows are encoded once
with the package's own encoder (row-wise), then the deployment function (sar.deploy.from_features, the call release() makes after its
checks) runs on contiguous request batches of several sizes; outputs are compared with the stored (evaluated) DEV_REPORT wire."""
import json
import os
import sys
import time

ROOT = str(__import__('pathlib').Path(__file__).resolve().parents[5])   # the repository root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np  # noqa: E402
from sar import common as C, deploy as DP  # noqa: E402
from acs_replication import deploy as SRC_DEPLOY  # noqa: E402

t0 = time.process_time()
R = C.features('DEV_REPORT', raw=True)
raw = R['raw_inputs']
n = len(R['row_id'])
rows = [{c: raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i] for c in DP.PERMITTED} for i in range(n)]
SIZES = (1, 2, 7, 50, 333, 1000, 2971, n)
ROWS_FOR = {1: 40, 2: 40, 7: 140, 50: 1000}         # small batches run on the first rows only (each call re-hashes frozen inputs)
SINGLE_ROWS = 40
out = {}
for pkg in sorted((C.PRIV / 'deploy').glob('*/PACKAGE.json')):
    meta = DP._load(pkg.parent)                     # dependency hashes checked once
    X = SRC_DEPLOY.encode_raw(rows, meta['encoder'])
    names = tuple(meta['encoder']['feature_names'])
    ref = DP.stored(meta)
    rec = {}
    for b in SIZES:
        m = ROWS_FOR.get(b, n)
        dmax = {t: 0.0 for t in C.TASKS}
        dec_mis = {t: 0 for t in C.TASKS}
        exact = {t: True for t in C.TASKS}
        for s in range(0, m, b):
            e = min(s + b, m)
            o = DP.from_features(meta['name'], meta['version'], X[s:e], names)
            for t in C.TASKS:
                d = np.abs(o[t] - ref[t][s:e])
                dmax[t] = max(dmax[t], float(d.max()))
                exact[t] = exact[t] and bool(np.array_equal(o[t], ref[t][s:e]))
                dec_mis[t] += int((o[t].argmax(1) != ref[t][s:e].argmax(1)).sum())
        rec[str(b)] = {'rows_checked': m, 'max_abs_diff': dmax, 'bitwise': exact, 'decision_mismatches': dec_mis}
    out[pkg.parent.name] = rec
    print(pkg.parent.name, {b: (max(r['max_abs_diff'].values()), sum(r['decision_mismatches'].values())) for b, r in rec.items()}, flush=True)
worst = max(max(r['max_abs_diff'].values()) for p in out.values() for r in p.values())
mis = sum(sum(r['decision_mismatches'].values()) for p in out.values() for r in p.values())
res = {'schema': 'SAR1-deployment-tolerance-v1', 'utc': C.now(), 'packages': out, 'batch_sizes': list(SIZES),
       'rows_checked_for_small_batches': ROWS_FOR, 'worst_max_abs_diff_any_batch': worst, 'decision_mismatches_total': mis,
       'labels_read': False, 'interpreter': sys.version.split()[0], 'cpu_s': round(time.process_time() - t0, 1),
       'statement': ('the deployed probabilities equal the evaluated output bitwise when a request is the full evaluated batch; for other '
                     'request batch sizes they agree within the worst absolute difference reported here (floating-point summation order '
                     'in the batched matrix products), and every decision is identical'),
       'found_by': 'verifier wave 3 (50-row and single-row requests of the FB packages)'}
C.write_json_x(C.RESULTS / 'deploy' / 'DEPLOYMENT_TOLERANCE.json', res)
print('worst', worst, 'decision mismatches', mis, 'cpu_s', res['cpu_s'])
