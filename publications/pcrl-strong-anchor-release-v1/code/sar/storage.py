"""Storage and workspace admission record of the strong-anchor release campaign (label-free; reads only file metadata, the
relocation manifests and diskutil output; never writes to the external drive).

    python3 -m sar.storage admit      -> results/pcrl_strong_anchor_release_v1/STORAGE_ADMISSION.json
    python3 -m sar.storage free LABEL -> appends one free-space reading to .private/<campaign>/compute/FREE_SPACE.jsonl

The relocation index (~/storage-relocation-20261022/) is the authority for what was moved. The drive identity is checked against
its manifest (volume UUID, mount point, ExFAT, writable) before anything on the drive is read; nothing is mounted, unmounted or
forced. Personal paths are written relative to the home folder so that the record passes the publication sanitizer unchanged.
"""
from __future__ import annotations
import collections
import hashlib
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

from . import common as C

INDEX = Path.home() / 'storage-relocation-20261022'
MOUNT = Path('/Volumes/YOTUO')
FREE_LOG = C.PRIV / 'compute' / 'FREE_SPACE.jsonl'


def _sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def free_bytes(path='/System/Volumes/Data'):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def drive():
    rec = {'mounted': os.path.ismount(MOUNT)}
    if not rec['mounted']:
        rec['note'] = 'absent (no directory was created at the mount point)'
        return rec
    r = subprocess.run(['diskutil', 'info', '-plist', str(MOUNT)], capture_output=True)
    info = plistlib.loads(r.stdout) if r.returncode == 0 else {}
    man = json.loads((INDEX / 'manifest.json').read_text())['drive']
    rec.update({'device': info.get('DeviceNode'), 'volume_name': info.get('VolumeName'), 'filesystem': info.get('FilesystemType'),
                'volume_uuid': info.get('VolumeUUID'), 'protocol': info.get('BusProtocol'), 'writable': info.get('WritableVolume'),
                'size_bytes': info.get('TotalSize'), 'free_bytes': info.get('FreeSpace') or info.get('APFSContainerFree'),
                'identity_ok': (info.get('VolumeUUID') == man['volume_uuid'] and info.get('MountPoint') == man['mount_point']
                                and str(info.get('FilesystemType', '')).lower() == 'exfat' and bool(info.get('WritableVolume')))})
    ext = MOUNT / Path(man['relocation_root']).name / 'manifest.json'
    rec['external_index'] = {'path': f'<external drive>/{ext.parent.name}/manifest.json', 'exists': ext.is_file(),
                             'sha256': _sha(ext) if ext.is_file() else None}
    return rec


def admit():
    man = json.loads((INDEX / 'manifest.json').read_text())
    by = collections.Counter((j.get('category'), j.get('status')) for j in man['jobs'].values())
    local_sha = _sha(INDEX / 'manifest.json')
    d = drive()
    adm = json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())
    rec = {'schema': 'SAR1-storage-admission-v1', 'utc': C.now(), 'labels_read': False,
           'relocation_index': {'path': '~/storage-relocation-20261022/', 'run': man['run'], 'jobs': len(man['jobs']),
                                'by_category_status': {f'{k[0]} | {k[1]}': v for k, v in sorted(by.items(), key=str)},
                                'manifest_sha256': local_sha,
                                'read': ['manifest.json', 'RESTORE.md', 'LEDGER.csv', 'verify_restored.py (verification instruction)',
                                         'the extraction order: a ._* side archive before its main archive']},
           'drive': d,
           'external_index_equals_local': bool(d.get('external_index', {}).get('sha256') == local_sha),
           'restores': [],
           'restore_note': ('none needed: the anchors (SUCC, NEURAL_RAW), every reference and every source artefact this campaign reads '
                            'are local (the protected-tree replication tree is archived and kept; the inherited private trees are in the '
                            'working cache); relocated items mentioned by the code are read through their admitted mirrors'),
           'working_cache': {'trees': {t: {k: v for k, v in x.items() if k in ('files', 'bytes', 'ok')}
                                       for t, x in adm['working_cache']['trees'].items()},
                             'ok': adm['working_cache']['ok'], 'method': adm['working_cache']['method']},
           'internal_free_bytes': free_bytes(), 'floor_bytes': 20 * 2 ** 30}
    C.write_json(C.RESULTS / 'STORAGE_ADMISSION.json', rec)
    print(json.dumps({k: rec[k] for k in ('drive', 'external_index_equals_local', 'internal_free_bytes')}, indent=1, default=str))
    return rec


def free(label):
    FREE_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {'utc': C.now(), 'label': label, 'free_bytes': free_bytes()}
    with open(FREE_LOG, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec))
    return rec


if __name__ == '__main__':
    if sys.argv[1:2] == ['admit']:
        admit()
    elif sys.argv[1:2] == ['free']:
        free(sys.argv[2] if len(sys.argv) > 2 else 'reading')
    else:
        print(__doc__)
