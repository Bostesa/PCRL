"""Verifier wave 3 addendum (label-free): append the full-batch diagnostic entries of the two FB packages not covered by the first
diagnostic run (work/deploy3_diag_b.json) to CHECKS.json, after checking the file against its hash at the end of v3_assemble.py.

    python3 .../roles/verifier/code/v3_append_b.py --utc <date -u> --prev-sha256 <sha>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--utc', required=True)
    ap.add_argument('--prev-sha256', required=True)
    a = ap.parse_args()
    cp = V.VDIR / 'CHECKS.json'
    if V.fsha(cp) != a.prev_sha256:
        raise PermissionError('CHECKS.json differs from its recorded content')
    W = json.loads(cp.read_text())
    gp = V.WORK / 'deploy3_diag_b.json'
    G = json.loads(gp.read_text())
    new = []
    for name, r in G['packages'].items():
        ok = all(t['full_batch_bitwise'] and t['full_batch_decisions_equal'] for t in r.values())
        new.append({'id': f'W3_DEPLOY_{name}_full_batch_bitwise', 'wave': 3,
                    'what': f'{name}: diagnostic: sar.deploy.release on ALL 5,943 DEV_REPORT rows reproduces the stored wire bitwise; 50-row '
                            'and 1-row batches differ by the recorded ULP-level amounts (addendum run, the two FB packages the first '
                            'diagnostic did not cover)',
                    'inputs_sha256': {str(gp.relative_to(V.ROOT)): V.fsha(gp)}, 'recomputed': r, 'record': 'stored DEV_REPORT wire',
                    'status': 'PASS' if ok else 'FAIL', 'tolerance': 'bitwise', 'unit': 'none (label-free)'})
    W['checks'] = W['checks'] + new
    w3 = W['waves']['3']
    nf = [c['id'] for c in W['checks'] if c.get('wave') == 3 and c['status'] != 'PASS']
    n3 = sum(1 for c in W['checks'] if c.get('wave') == 3)
    w3['totals'] = {'checks': n3, 'pass': n3 - len(nf), 'fail': len(nf)}
    w3['failed'] = nf
    w3['addendum_utc'] = a.utc
    w3['outside_queue_cpu_s'] = float(w3['outside_queue_cpu_s']) + float(G['cpu_s'])
    allf = [c['id'] for c in W['checks'] if c['status'] != 'PASS']
    W['totals_all_waves'] = {'checks': len(W['checks']), 'pass': len(W['checks']) - len(allf), 'fail': len(allf), 'failed': allf}
    V.jdump(cp, W)
    print(json.dumps({'wave3': w3['totals'], 'all': W['totals_all_waves']['checks'], 'all_fail': W['totals_all_waves']['fail']}))


if __name__ == '__main__':
    main()
