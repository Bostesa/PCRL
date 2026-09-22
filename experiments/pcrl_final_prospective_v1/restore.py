"""Selective, hash-verified restore of the frozen task-directed release objects.

Only the members named by ``restore_prefixes`` are written; every member of a
downloaded part is still hashed against the pinned private manifest, and the
compressed part is checked against the pinned ARCHIVE_INDEX before use.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.pcrl_task_directed_release_v1.archive import chunk_records, verify_and_restore  # noqa: E402

OUT = ROOT/'results'/'pcrl_final_prospective_v1'
META = OUT/'private'/'archive_meta'
RESTORE = OUT/'private'/'restore'
BUCKET = 'pcrl-ux-archive-ed9d21fd'
INDEX = ROOT/'results'/'pcrl_task_directed_release_v1'/'ARCHIVE_INDEX.json'
RUN = 'results/pcrl_task_directed_release_v1/private/run'
MANIFEST_SHA256 = '504944807f9f9b9f087836d34768440647b6276db0fca2827be3084213b3e667'
CORE = ('H', 'J', 'T0_L_0.01_a17', 'T0_U_unconstrained_a17', 'T0_U_unconstrained_a33',
        'continuous_task', 'leace_supervised_mechanism40', 'splince_supervised_mechanism40',
        'T0_rr_0.75', 'T0_withhold_0.75')
MAPS = ('T0_L_0.01_a17', 'T0_U_unconstrained_a17', 'T0_U_unconstrained_a33')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def prefixes(audit_anchors=(0,)):
    out = []
    for a in (0, 1, 2):
        base = f'{RUN}/anchor_{a}'
        out += [f'{base}/encoder', f'{base}/erasers', f'{base}/baseline_supplement/mechanism40',
                f'{base}/prepared.joblib', f'{base}/PREPARED.json']
        out += [f'{base}/maps/{m}' for m in MAPS]
        out += [f'{base}/evaluation/{r}' for r in CORE]
        if a in audit_anchors:
            out += [f'{base}/audits/{r}' for r in CORE]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', type=int, nargs='+', required=True)
    args = ap.parse_args()
    manifest = META/'MANIFEST.private.json'
    if sha(manifest) != MANIFEST_SHA256:
        raise SystemExit('Pinned private manifest hash differs')
    records = json.loads(manifest.read_text())['files']
    index = json.loads(INDEX.read_text())
    groups = chunk_records(records)
    if len(groups) != len(index['chunks']):
        raise SystemExit('Chunk replay differs from archive index')
    wanted = prefixes()
    log = META/'RESTORE_LOG.jsonl'
    for i in args.parts:
        chunk = index['chunks'][i]
        local = META/f'part-{i:04d}.tar.zst'
        if not local.exists():
            subprocess.run(['aws', 's3api', 'get-object', '--bucket', BUCKET, '--key', chunk['key'],
                            '--version-id', chunk['version_id'], str(local)], check=True,
                           stdout=subprocess.DEVNULL)
        digest = sha(local)
        if digest != chunk['sha256'] or local.stat().st_size != chunk['bytes']:
            local.unlink()
            raise SystemExit(f'Compressed part {i} hash/size mismatch')
        result = verify_and_restore(local, groups[i], restore_root=RESTORE, restore_prefixes=wanted)
        entry = {'part': i, 'key': chunk['key'], 'version_id': chunk['version_id'],
                 'compressed_sha256': digest, 'verified_files': result['verified_files'],
                 'restored_files': result['restored_files']}
        with log.open('a') as f:
            f.write(json.dumps(entry) + '\n')
        local.unlink()
        print(json.dumps(entry), flush=True)


if __name__ == '__main__':
    main()
