"""VER check 8 (restore): re-hash a campaign backup's MANIFEST.sha256 with own code, inspect the tar members (no absolute paths, no
'..', no links leaving the archive), and extract code.tar + private.tar into a NEW, EMPTY directory with the 'data' extraction filter.

    python3 -I v_restore.py --backup <backup dir> --dest <new empty dir> --out <record json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backup', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    b = Path(a.backup)
    rec = {'check': 'VER_RESTORE_EXTRACT', 'backup': str(b), 'dest': a.dest, 'utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    man = {}
    for line in (b / 'MANIFEST.sha256').read_text().splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            man[name.strip()] = h
    rec['manifest'] = {name: {'expected': h, 'actual': sha(b / name), 'equal': sha(b / name) == h} for name, h in man.items()}
    rec['manifest_all_equal'] = all(v['equal'] for v in rec['manifest'].values())
    rec['tip_commit'] = (b / 'TIP_COMMIT').read_text().strip()
    dest = Path(a.dest)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit('destination must be new and empty')
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    members = {}
    for name in ('code.tar', 'private.tar'):
        with tarfile.open(b / name) as tf:
            ms = tf.getmembers()
            unsafe = [m.name for m in ms if m.name.startswith('/') or '..' in Path(m.name).parts
                      or ((m.issym() or m.islnk()) and (os.path.isabs(m.linkname) or '..' in Path(m.linkname).parts))]
            members[name] = {'n_members': len(ms), 'n_files': sum(m.isfile() for m in ms), 'bytes': sum(m.size for m in ms if m.isfile()),
                             'unsafe_members': unsafe[:20], 'n_unsafe': len(unsafe)}
            if unsafe:
                raise SystemExit(f'unsafe members in {name}: {unsafe[:5]}')
            tf.extractall(dest, filter='data')
    rec['tar_members'] = members
    rec['restored_top_level'] = sorted(p.name for p in dest.iterdir())
    rec['cpu_s'] = round(time.process_time() - t0, 2)
    rec['utc_end'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'manifest_all_equal': rec['manifest_all_equal'], 'tip': rec['tip_commit'], 'members': members,
                      'top_level': rec['restored_top_level'][:30]}, indent=1))


if __name__ == '__main__':
    main()
