"""Custody for the correction pass: versioned snapshot, restore-and-rerun of the small exact checks, off-device copy.

It reuses mechanism_design/custody.py and changes two things:
- the private part of the snapshot is the correction pass's own queue ledger (.private/mechanism_design/correction_v1/compute);
- the restore check re-runs only the small exact checks named with --rerun MODULE=STORED_JSON (coordinator and
  verifier), from the restored code alone, and requires each re-run to equal the stored results minus provenance.
  It does not re-run the campaign's prototype.

snapshot --name NAME
restore-check --snapshot DIR --restore FRESH_DIR --rerun MODULE=results/...json [--rerun ...]
offdevice --snapshot DIR --drive-dir DIR
No real-person data exists in this campaign; nothing here reads any.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from mechanism_design import custody as C

C.PRIV = C.ROOT / '.private' / 'mechanism_design' / 'correction_v1'
# Run-specific fields excluded from the identity comparison: the coordinator's 'provenance' block, and the verifier's
# unit id, timestamp, interpreter version and timing fields. Every scientific value is compared.
VOLATILE_KEYS = ('provenance', 'unit_id', 'generated_utc', 'python', 'cpu_s', 'closed_form_large_cpu_s', 'process_cpu_s', 'wall_s')


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def restore_check(snapshot_dir: str, restore_dir: str, reruns: list) -> dict:
    snap, rest = Path(snapshot_dir), Path(restore_dir)
    if rest.exists():
        raise PermissionError('restore directory must be fresh')
    shutil.copytree(snap, rest)
    out = C._verify(rest)
    out['reruns'] = {}
    env = {**os.environ, 'PYTHONPATH': str(rest / 'code'), 'OMP_NUM_THREADS': '1'}
    for spec in reruns:
        module, stored_rel = spec.split('=', 1)
        stored = rest / stored_rel
        rerun = rest / ('rerun_' + Path(stored_rel).name)
        proc = subprocess.run([sys.executable, '-m', module, '--out', str(rerun)], cwd=str(rest), env=env, capture_output=True, text=True)
        rec = {'returncode': proc.returncode, 'stderr_tail': proc.stderr[-1500:]}
        if proc.returncode == 0 and stored.is_file():
            rec['identical_minus_provenance'] = _strip(json.loads(stored.read_text())) == _strip(json.loads(rerun.read_text()))
        out['reruns'][module] = rec
    out['all_reruns_identical'] = bool(reruns) and all(r.get('identical_minus_provenance') for r in out['reruns'].values())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('snapshot'); s.add_argument('--name', required=True)
    r = sub.add_parser('restore-check'); r.add_argument('--snapshot', required=True); r.add_argument('--restore', required=True)
    r.add_argument('--rerun', action='append', default=[])
    o = sub.add_parser('offdevice'); o.add_argument('--snapshot', required=True); o.add_argument('--drive-dir', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'snapshot':
        res = C.snapshot(a.name)
    elif a.cmd == 'restore-check':
        res = restore_check(a.snapshot, a.restore, a.rerun)
    else:
        res = C.offdevice(a.snapshot, a.drive_dir)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
