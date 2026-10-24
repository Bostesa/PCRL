"""Verifier wave 3 diagnostic (label-free; torch venv): does sar.deploy.release reproduce the stored DEV_REPORT wire bitwise when it is
given ALL DEV_REPORT rows (the batch the stored wire was computed on), and how far do 50-row and 1-row batches differ? Run for the FB
packages (whose 50-row occupation outputs differed from the stored wire by a few ULP in v3_deploy.py) and one FA package as a contrast.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

sys.path.insert(0, str(V.ROOT))
import numpy as np  # noqa: E402

PERMITTED = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--names', nargs='*', default=None)
    a = ap.parse_args()
    t0 = time.process_time()
    from sar import common as C
    from sar import deploy as SD
    f = C.features('DEV_REPORT', raw=True)
    raw = f['raw_inputs']
    n = len(f['row_id'])
    rows = [{k: str(raw[k][i]) for k in PERMITTED} for i in range(n)]
    idx = np.sort(np.random.default_rng(20261026).choice(n, size=50, replace=False))
    out = {'utc_start': C.now(), 'packages': {}}
    for name in (a.names or ('FB_PFO_V0', 'FB_PFO_V1', 'FB_PFO_DQ_HI_V0', 'FB_PFO_RAWJ_HI_V0', 'FA_TFRB_R2_V0')):
        pkg = V.PRIV / 'deploy' / name
        meta = json.loads((pkg / 'PACKAGE.json').read_text())
        fam, cfg = meta['name'].split('/')
        with np.load(V.RUNS / 'STORED' / f'{fam}__{cfg}__V{int(meta["version"])}.npz') as z:
            st = {t: z[f'{t}__DEV_REPORT'].copy() for t in V.TASKS}
        full = SD.release(str(pkg), rows)
        sub = SD.release(str(pkg), [rows[i] for i in idx])
        one = [SD.release(str(pkg), [rows[i]]) for i in idx[:5]]
        r = {}
        for t in V.TASKS:
            r[t] = {'full_batch_bitwise': bool(np.array_equal(full[t]['probability'], st[t])),
                    'full_batch_max_abs': float(np.abs(full[t]['probability'] - st[t]).max()),
                    'full_batch_decisions_equal': bool(np.array_equal(full[t]['decision'], st[t].argmax(1))),
                    'sample50_vs_full_batch_max_abs': float(np.abs(sub[t]['probability'] - full[t]['probability'][idx]).max()),
                    'sample50_vs_stored_max_abs': float(np.abs(sub[t]['probability'] - st[t][idx]).max()),
                    'single_row_vs_stored_max_abs': float(max(np.abs(o[t]['probability'][0] - st[t][i]).max() for o, i in zip(one, idx[:5])))}
        out['packages'][name] = r
        print(json.dumps({name: r}), flush=True)
    out['cpu_s'] = time.process_time() - t0
    out['utc_end'] = C.now()
    V.jdump(Path(a.out), out)


if __name__ == '__main__':
    main()
