"""Verifier stage V6e: restore the campaign backup into a fresh directory of the verifier's own (label-free) and record the restore.

PCRL controlled training-size campaign (TSZ1). Steps: verify MANIFEST.sha256 of the backup with own SHA-256; extract code.tar and
private.tar into a new empty directory outside the campaign tree; record the file counts and the hashes of the 11 final package
PACKAGE.json files and stored wires in the restored copy against the live campaign tree. The deployment itself is then run by
v_deploy.py with TSZ_VERIFIER_ROOT = <restored root> and TSZ_VERIFIER_RECORD_DIR = the original verifier folder.

    python3 .../v6_restore.py --backup <dir> --dest <new empty dir> --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

FINAL = ('W100_V0_B2', 'W100_V1_B2', 'CTRL_S100_V0', 'CTRL_S100_V1', 'A1_S50_O0_V0', 'CORR_V0_LOC_s0p3', 'CORR_V0_PAIR_s0p3', 'CORR_V0_CONT_s0p3',
         'CORR_V1_LOC_s0p3', 'CORR_V1_PAIR_s0p3', 'CORR_V1_CONT_s0p3')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--backup', required=True); ap.add_argument('--dest', required=True); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    b, dest = Path(a.backup), Path(a.dest)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit('destination must be new and empty')
    dest.mkdir(parents=True, exist_ok=True)
    man = {}
    for line in (b / 'MANIFEST.sha256').read_text().splitlines():
        h, n = line.split()
        man[n] = h
    mine = {n: V.file_sha(b / n) for n in man}
    manifest_ok = mine == man
    counts = {}
    for t in ('code.tar', 'private.tar'):
        with tarfile.open(b / t) as tf:
            members = tf.getmembers()
            bad = [m.name for m in members if m.name.startswith('/') or '..' in Path(m.name).parts]
            if bad:
                raise SystemExit(f'unsafe member paths in {t}: {bad[:3]}')
            tf.extractall(dest, filter='data')
            counts[t] = len(members)
    live = V.ROOT / '.private' / 'pcrl_controlled_training_size_v1'
    rest = dest / '.private' / 'pcrl_controlled_training_size_v1'
    pk = {}
    for name in FINAL:
        p_live, p_rest = live / 'deploy' / name / 'PACKAGE.json', rest / 'deploy' / name / 'PACKAGE.json'
        pk[name] = {'restored_exists': p_rest.is_file(),
                    'package_json_equal_live': p_rest.is_file() and p_live.is_file() and V.file_sha(p_rest) == V.file_sha(p_live)}
    tip = (b / 'TIP_COMMIT').read_text().strip()
    ok = manifest_ok and all(x['restored_exists'] and x['package_json_equal_live'] for x in pk.values())
    chk = {'id': f'V6.RESTORE.BACKUP_EXTRACTED__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
           'what': 'the same-device backup B1 restored by the verifier into a fresh directory of its own: MANIFEST.sha256 re-hashed with own '
                   'SHA-256, code.tar and private.tar extracted (safe member paths), the 11 final packages present with PACKAGE.json equal to '
                   'the live campaign tree (warm same-device restore; not a cold or off-device check)',
           'inputs': {str(b / n): h for n, h in mine.items()}, 'expected': 'manifest equal; 11 packages present and equal',
           'observed': {'manifest_equal': manifest_ok, 'tip_commit': tip, 'members': counts, 'packages': pk, 'restored_root': str(dest)},
           'status': 'PASS' if ok else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}
    V.write_record(f'V6_RESTORE__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'manifest_equal': manifest_ok, 'members': counts, 'tip': tip}))


if __name__ == '__main__':
    main()
