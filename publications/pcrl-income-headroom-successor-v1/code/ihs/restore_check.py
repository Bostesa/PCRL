"""Restore check, run INSIDE a restored copy (backup alone): the admitted copies re-hash against SOURCE_ADMISSION.json; every
release record re-hashes against its RELEASE.json; every deployment package under runs/DEPLOY redeploys the restored DEV_REPORT
raw inputs (no labels) and reproduces the restored evaluated release bitwise through the wrapper API, the CSV path and the
underlying function; and, if a REPORTING_LOCK exists, the final reader files re-hash against it."""
from __future__ import annotations
import json
import sys
from pathlib import Path

from . import common as C
from . import deploy as DP


def main(out):
    adm = C.admission()['copies']
    bad = [rel for rel, r in adm.items() if C.sha(C.ADM / rel) != r['sha256']]
    rel_bad, n_rel = [], 0
    for p in sorted(C.RUNS.glob('*/*/S*/RELEASE.json')):
        rec = json.loads(p.read_text()); n_rel += 1
        if C.sha(p.parent / 'RELEASE.npz') != rec['release_sha256']:
            rel_bad.append(str(p.parent.relative_to(C.RUNS)))
    readers_ok = None
    if (C.RESULTS / 'REPORTING_LOCK.json').is_file():
        rl = json.loads((C.RESULTS / 'REPORTING_LOCK.json').read_text())
        readers_ok = all(C.sha(C.RUNS / f'READERS_S{s}' / f) == rl['readers'][f'S{s}'][k] for s in C.SEEDS
                         for f, k in (('READERS_SEED.json', 'record_sha256'), ('AV_SCORES.npz', 'av_scores_sha256'), ('DEV_REPORT_SCORES.npz', 'dev_report_scores_sha256')))
    pkgs = sorted(str(p.parent) for p in (C.RUNS / 'DEPLOY').glob('*/PACKAGE.json'))
    dep = DP.check(pkgs, Path(out).with_name('restore_deploy_detail.json')) if pkgs else {'all_bitwise': None, 'packages': {}}
    rec = {'admitted_copies_rehash_ok': not bad, 'admitted_copies_checked': len(adm), 'mismatched': bad[:10],
           'release_records_checked': n_rel, 'release_records_mismatched': rel_bad[:10],
           'final_reader_files_match_reporting_lock': readers_ok, 'deployment_packages': len(pkgs),
           'deployment_from_backup_bitwise': dep['all_bitwise'], 'labels_read': False}
    rec['all_ok'] = rec['admitted_copies_rehash_ok'] and not rel_bad and readers_ok is not False and dep['all_bitwise'] is not False
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec))


if __name__ == '__main__':
    main(sys.argv[1])
