"""Restore check, run INSIDE a restored copy (backup alone): the 201 admitted mirror copies re-hash against SOURCE_ADMISSION.json;
every occupation release record (OCC.json -> OCC.npz) and complete release record (RELEASE.json -> RELEASE.npz) re-hashes; and
every deployment package under
runs/DEPLOY redeploys the restored DEV_REPORT raw inputs (no labels) and reproduces the restored evaluated release bitwise through
the wrapper API, the CSV path and the underlying function, with every refusal."""
from __future__ import annotations
import json
import sys
from pathlib import Path

from . import common as C
from . import deploy as DP


def main(out):
    adm = C.admission()['copies']
    bad = [rel for rel, r in adm.items() if C.sha(C.MIRROR / rel) != r['sha256']]
    rel_bad, n_rel = [], 0
    for p in sorted(C.RUNS.glob('*/*/S*/OCC.json')):
        rec = json.loads(p.read_text()); n_rel += 1
        if C.sha(p.parent / 'OCC.npz') != rec['occ_sha256']:
            rel_bad.append(str(p.parent.relative_to(C.RUNS)))
    for p in sorted(C.RUNS.glob('*/*/S*/RELEASE.json')):
        rec = json.loads(p.read_text()); n_rel += 1
        if C.sha(p.parent / 'RELEASE.npz') != rec['release_sha256']:
            rel_bad.append(str(p.parent.relative_to(C.RUNS)))
    pkgs = sorted(str(p.parent) for p in (C.RUNS / 'DEPLOY').glob('*/PACKAGE.json'))
    dep = DP.check(pkgs, Path(out).with_name('restore_deploy_detail.json')) if pkgs else {'all_bitwise': None, 'all_refused': None, 'packages': {}}
    rec = {'admitted_copies_rehash_ok': not bad, 'admitted_copies_checked': len(adm), 'mismatched': bad[:10],
           'release_records_checked': n_rel, 'release_records_mismatched': rel_bad[:10], 'deployment_packages': len(pkgs),
           'deployment_from_backup_bitwise': dep['all_bitwise'], 'deployment_from_backup_all_refused': dep.get('all_refused'),
           'labels_read': False}
    rec['all_ok'] = rec['admitted_copies_rehash_ok'] and not rel_bad and dep['all_bitwise'] is not False and dep.get('all_refused') is not False
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec))


if __name__ == '__main__':
    main(sys.argv[1])
