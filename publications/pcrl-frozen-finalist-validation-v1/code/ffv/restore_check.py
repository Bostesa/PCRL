"""Restore-and-deploy check, run INSIDE a restored copy (backup alone): every admitted package redeploys the restored DEV_REPORT
raw inputs (no labels) and must reproduce the restored evaluated outputs bitwise through the wrapper API, the CSV path and the
underlying function; the restored admitted copies, enhanced reader files and reporting-lock hashes must re-verify."""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from amd import deploy as AD
from . import common as C
from .deploy_check import PKGS, dev_rows, same, underlying, wrapper


def main(out):
    adm = C.admission()['copies']
    bad = [rel for rel, r in adm.items() if C.sha(C.ADM / rel) != r['sha256']]
    rl = json.loads((C.RESULTS / 'REPORTING_LOCK.json').read_text())
    rd_ok = all(C.sha(C.RUNS / f'READERS_ENH_S{s}' / f) == rl['enhanced_readers'][f'S{s}'][k]
                for s in C.SEEDS for f, k in (('READERS_ENH_SEED.json', 'record_sha256'), ('AV_SCORES.npz', 'av_scores_sha256'), ('DEV_REPORT_SCORES.npz', 'dev_report_scores_sha256')))
    rows = dev_rows(); tmp = Path(tempfile.mkdtemp()); p = tmp / 'raw.csv'; AD.write_csv(p, rows, AD.PERMITTED)
    res = {}
    for name in PKGS:
        d = C.ADM / f'deploy/{name}'; base, s = name.rsplit('_S', 1)
        bank, _, _ = C.service_bank(int(s)); ref = {t: bank[base]['DEV_REPORT'][t] for t in C.TASKS}
        w_api, w_csv = wrapper(name, d); u_api, _ = underlying(name, d)
        res[name] = all(all(v.values()) for v in (same(w_api(rows), ref), same(w_csv(p), ref), same(u_api(rows), ref)))
    rec = {'admitted_copies_rehash_ok': not bad, 'admitted_copies_checked': len(adm), 'mismatched': bad[:10],
           'enhanced_reader_files_match_reporting_lock': rd_ok,
           'deployment_from_backup_reproduces_evaluated_outputs_bitwise': res,
           'labels_read': False}
    rec['all_ok'] = rec['admitted_copies_rehash_ok'] and rd_ok and all(res.values())
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec))


if __name__ == '__main__':
    main(sys.argv[1])
