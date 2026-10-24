# Coordinator script (label-free) that wrote deploy/DEPLOYMENT_VALIDITY.json; saved from the session scratchpad at closeout.
"""Probability validity and forbidden-input audit of the 13 deployment packages (label-free).
The packages reproduce their stored wires bitwise (deploy/DEPLOYMENT_CHECK.json), so validity of the stored wires on all six admitted
feature sets is validity of the packages' outputs on those inputs."""
import json
import os
import sys
import time

ROOT = str(__import__('pathlib').Path(__file__).resolve().parents[5])   # the repository root
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np  # noqa: E402
from sar import common as C  # noqa: E402

t0 = time.process_time()
FORBIDDEN = ('SEX', 'PINCP', 'OCCP', 'PUMA', 'SERIALNO', 'RELP', 'ST', 'WAGP')
pk_dir = C.PRIV / 'deploy'
out = {}
for pkg in sorted(pk_dir.glob('*/PACKAGE.json')):
    meta = json.loads(pkg.read_text())
    name, v = meta['name'], int(meta['version'])
    stored = C.RUNS / 'STORED' / (name.replace('/', '__') + f'__V{v}.npz')
    rows = {}
    with np.load(stored, allow_pickle=False) as z:
        for key in sorted(z.files):
            p = z[key]
            s = p.sum(1)
            rows[key] = {'n': int(p.shape[0]), 'k': int(p.shape[1]), 'finite': bool(np.isfinite(p).all()),
                         'min': float(p.min()), 'max': float(p.max()), 'max_abs_row_sum_minus_1': float(np.abs(s - 1).max())}
    feats = [str(f) for f in meta['encoder']['feature_names']]
    bad_feats = sorted({f for f in feats for b in FORBIDDEN if f.upper().startswith(b + '_') or f.upper() == b})
    deps = sorted(meta['depends_sha256'])
    label_like = [d for d in deps if any(x in d.lower() for x in ('/labels', 'sex.npz', 'income.npz', 'occupation.npz', 'targets'))]
    ok = (all(r['finite'] and r['min'] >= 0 and r['max'] <= 1 and r['max_abs_row_sum_minus_1'] <= 1e-12 for r in rows.values())
          and not bad_feats and not label_like and meta['permitted_inputs'] == ['AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP'])
    out[pkg.parent.name] = {'name': name, 'version': v, 'arrays_checked': len(rows),
                            'worst_row_sum_error': max(r['max_abs_row_sum_minus_1'] for r in rows.values()),
                            'min_probability': min(r['min'] for r in rows.values()), 'max_probability': max(r['max'] for r in rows.values()),
                            'all_finite': all(r['finite'] for r in rows.values()), 'n_encoded_features': len(feats),
                            'forbidden_encoded_features': bad_feats, 'permitted_inputs': meta['permitted_inputs'],
                            'label_like_dependencies': label_like, 'n_dependencies': len(deps), 'ok': bool(ok)}
rec = {'schema': 'SAR1-deployment-validity-v1', 'utc': C.now(), 'packages': out, 'all_ok': all(r['ok'] for r in out.values()),
       'tolerance_row_sum': 1e-12, 'roles': list(C.ROLES6), 'labels_read': False,
       'note': 'stored wires = the packages\' outputs (bitwise, deploy/DEPLOYMENT_CHECK.json) on the admitted features of all six roles; '
               'forbidden names checked in the encoder\'s 145 feature names and the dependency list; runtime refusals are in '
               'DEPLOYMENT_CHECK.json', 'cpu_s': round(time.process_time() - t0, 2)}
C.write_json_x(C.RESULTS / 'deploy' / 'DEPLOYMENT_VALIDITY.json', rec)
print(json.dumps({k: (v['ok'], v['worst_row_sum_error'], v['arrays_checked']) for k, v in out.items()}, indent=1))
print('all_ok', rec['all_ok'], 'cpu_s', rec['cpu_s'])
