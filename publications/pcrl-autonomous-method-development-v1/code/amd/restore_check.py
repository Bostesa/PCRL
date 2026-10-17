"""Restore-and-deploy check, run inside a restored copy (backup alone): every packaged release of this campaign, given the restored
DEV_REPORT raw inputs (no labels), must reproduce the restored evaluated DEV_REPORT outputs bitwise through the API and CSV entry
points; the frozen reference packages are rerun through their own source deployment code behind the same input check."""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from acs_replication import deploy as SRC_DEPLOY
from . import common as C
from . import devrun as DR
from .deploy import raw_rows, release, release_csv, write_csv, check_rows, PERMITTED


def main(out):
    rec = json.loads((C.RESULTS / 'DEPLOYMENT_RESULTS.json').read_text())
    rows = raw_rows(); tmp = Path(tempfile.mkdtemp()); p = tmp / 'raw.csv'; write_csv(p, rows, PERMITTED)
    res = {}
    for name, pk in rec['packages'].items():
        d = C.PRIV / 'deploy' / name
        base, s = name.rsplit('_S', 1); s = int(s)
        if 'release' in pk:
            family, config_id = pk['release'].split('/')
            ref = DR.load_release(family, config_id, s, roles=('DEV_REPORT',))[0]['DEV_REPORT']
            a, b = release(d, rows), release_csv(d, p)
        else:
            bank, _ = C.frozen_bank(s, roles=('DEV_REPORT',))
            ref = {t: bank[base]['DEV_REPORT'][t] for t in C.TASKS}
            if base == 'H_LJ':
                from acs_hybrids import hybrid_deploy as HYB
                fn = HYB.release
            else:
                fn = SRC_DEPLOY.release
            check_rows(rows)
            a = fn(d, rows, ('probability', 'decision')); b = fn(d, SRC_DEPLOY.read_raw_csv(p), ('probability', 'decision'))
        res[name] = all(np.array_equal(a[t]['probability'], ref[t]) and np.array_equal(b[t]['probability'], ref[t]) and
                        np.array_equal(a[t]['decision'], ref[t].argmax(1)) for t in C.TASKS)
    out_rec = {'deployment_from_backup_reproduces_evaluated_outputs_bitwise': res, 'all_ok': bool(res) and all(res.values())}
    Path(out).write_text(json.dumps(out_rec, indent=1) + '\n')
    print(json.dumps(out_rec))


if __name__ == '__main__':
    main(sys.argv[1])
