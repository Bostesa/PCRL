"""Lossless publication/rehydration of completed, non-person-level JSON evidence.

Original local completion hashes stay authoritative. This never changes a fitted
artifact, candidate choice, score, or the original JSON bytes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compact(out):
    matrix = json.loads((out/'EXECUTED_MATRIX.json').read_text())
    assert matrix['status'] == 'complete', 'Wait for all scientific units'
    assert len(matrix['systems']) == 54
    assert sum(not e['reused'] for e in matrix['systems']) == 36
    assert all(e['training_complete'] and e['evaluation_complete'] for e in matrix['systems'])
    records = {}
    for entry in matrix['systems']:
        if entry['reused']:
            continue
        base = out/f"seed_{entry['seed']}"/entry['condition']
        completion_path = base/'complete.json'
        completion_bytes = completion_path.read_bytes()
        completion = json.loads(completion_bytes)
        for suffix in ('metrics.json', 'audits/audit_selection.json'):
            path = base/suffix
            raw = path.read_bytes()
            expected = completion['files_sha256'][str(path.resolve().relative_to(Path(__file__).resolve().parents[1]))]
            assert sha(raw) == expected, 'Immutable completed evidence changed'
            stream = io.BytesIO()
            with gzip.GzipFile(filename='', fileobj=stream, mode='wb', compresslevel=9, mtime=0) as archive:
                archive.write(raw)
            compressed = stream.getvalue()
            target = path.with_suffix(path.suffix+'.gz')
            if target.exists():
                assert target.read_bytes() == compressed, 'Existing public encoding differs'
            else:
                target.write_bytes(compressed)
            assert gzip.decompress(compressed) == raw
            records[str(path.relative_to(out))] = dict(plain_sha256=expected, gzip_sha256=sha(compressed),
                plain_bytes=len(raw), gzip_bytes=len(compressed), completion_path=str(completion_path.relative_to(out)),
                completion_sha256=sha(completion_bytes),
                completion_entry=str(path.resolve().relative_to(Path(__file__).resolve().parents[1])))
    assert len(records) == 72
    manifest = {'format': 'gzip level9, mtime0, empty filename', 'original_json_unchanged': True,
        'scope': '36 completed new systems; scores and selection metadata only; no person-level predictions',
        'files': records, 'source_sha256': sha(Path(__file__).read_bytes())}
    target = out/'COMPACT_UNIT_EVIDENCE.json'
    encoded = json.dumps(manifest, indent=2)+'\n'
    if target.exists():
        assert target.read_text() == encoded, 'Existing compact manifest differs'
    else:
        target.write_text(encoded)
    return manifest


def restore(out):
    manifest = json.loads((out/'COMPACT_UNIT_EVIDENCE.json').read_text())
    for relative, record in manifest['files'].items():
        path = (out/relative).resolve()
        assert path.is_relative_to(out.resolve()) and path.suffix == '.json'
        assert path.name in ('metrics.json', 'audit_selection.json')
        completion_path = (out/record['completion_path']).resolve()
        assert completion_path.is_relative_to(out.resolve()) and completion_path.name == 'complete.json'
        expected_base = path.parent if path.name == 'metrics.json' else path.parent.parent
        assert completion_path.parent == expected_base
        assert record['completion_entry'].endswith('/'+relative)
        completion_bytes = completion_path.read_bytes()
        assert sha(completion_bytes) == record['completion_sha256']
        completion = json.loads(completion_bytes)
        assert completion['files_sha256'][record['completion_entry']] == record['plain_sha256']
        archive = path.with_suffix('.json.gz').read_bytes()
        assert sha(archive) == record['gzip_sha256']
        raw = gzip.decompress(archive)
        assert len(raw) == record['plain_bytes'] and sha(raw) == record['plain_sha256']
        if path.exists():
            assert path.read_bytes() == raw, 'Refusing to overwrite differing local evidence'
        else:
            path.write_bytes(raw)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--restore', action='store_true', help='Restore exact JSON bytes for report regeneration')
    args = parser.parse_args()
    result = restore(args.out) if args.restore else compact(args.out)
    print(json.dumps({'files': len(result['files']), 'plain_bytes': sum(r['plain_bytes'] for r in result['files'].values()),
        'gzip_bytes': sum(r['gzip_bytes'] for r in result['files'].values()), 'all_hashes_verified': True}))
