"""Deployment re-check after the D1 repair (no new packages; DEV_REPORT raw inputs only, no labels).

For each SCM and TWIN package (both seeds):
- the DEV_REPORT raw fields, written to a temporary CSV with the exact permitted header, pass through release_csv and must
  reproduce the attacked DEV_REPORT wire bitwise;
- the package path refuses a row carrying SEX, rows with an extra, a missing or a renamed field, and a CSV with a SEX
  column or reordered columns;
- six hidden-output requests are refused.
Writes DEPLOYMENT_RECHECK.json.
"""
from __future__ import annotations
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from . import common as C
from .deploy_scm import raw_rows, release, release_csv


def main(out):
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    rows = raw_rows('DEV_REPORT'); perm = list(SRC_DEPLOY.PERMITTED)
    tmp = Path(tempfile.mkdtemp())
    good = tmp / 'dev_raw.csv'
    with open(good, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(perm)
        for r in rows:
            w.writerow([r[c] for c in perm])
    with open(tmp / 'with_sex.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(perm + ['SEX']); w.writerow([rows[0][c] for c in perm] + ['1'])
    with open(tmp / 'reordered.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(list(reversed(perm))); w.writerow([rows[0][c] for c in reversed(perm)])
    rec = {'schema': 'SCM1-deployment-recheck-v1', 'utc': C.now(), 'packages': {}}
    for s in C.SEEDS:
        with np.load(C.PRIV / 'runs' / f'FIT_{s}' / 'ARM_OUTPUTS.npz') as z:
            for label in ('SCM', 'TWIN'):
                arm = sel['bank_arm_map'][label]; pkg = C.PRIV / 'deploy' / f'{label}_S{s}'
                res = release_csv(pkg, good)
                ok = {t: bool(np.array_equal(res[t]['probability'], z[f'{arm}__DEV_REPORT__{t}'])
                              and np.array_equal(res[t]['decision'], z[f'{arm}__DEV_REPORT__{t}'].argmax(1))) for t in C.TASKS}
                refusals = {}
                cases = {'row_with_SEX': [{**rows[0], 'SEX': '1'}], 'row_extra_field': [{**rows[0], 'PINCP': '1'}],
                         'row_missing_field': [{c: rows[0][c] for c in perm[:-1]}],
                         'row_renamed_field': [{**{c: rows[0][c] for c in perm[:-1]}, 'POB': rows[0][perm[-1]]}]}
                for name, bad in cases.items():
                    try:
                        release(pkg, bad); refusals[name] = False
                    except (PermissionError, ValueError):
                        refusals[name] = True
                for name in ('with_sex.csv', 'reordered.csv'):
                    try:
                        release_csv(pkg, tmp / name); refusals[f'csv_{name}'] = False
                    except (PermissionError, ValueError):
                        refusals[f'csv_{name}'] = True
                for req in ('cell_id', 'group_id', 'teacher_probability', 'propensity', 'raw_score', 'leaf_id'):
                    try:
                        release(pkg, rows[:2], ('probability', req)); refusals[f'hidden_{req}'] = False
                    except PermissionError:
                        refusals[f'hidden_{req}'] = True
                rec['packages'][f'{label}_S{s}'] = {'csv_path_reproduces_attacked_DEV_REPORT_wire_bitwise': ok, 'refusals': refusals,
                                                     'all_refused': all(refusals.values())}
    shutil.rmtree(tmp)
    rec['all_ok'] = all(all(v['csv_path_reproduces_attacked_DEV_REPORT_wire_bitwise'].values()) and v['all_refused'] for v in rec['packages'].values())
    Path(out).write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({'all_ok': rec['all_ok'], 'packages': {k: (v['all_refused'], v['csv_path_reproduces_attacked_DEV_REPORT_wire_bitwise']) for k, v in rec['packages'].items()}}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
