"""A sealed-partition loader for a future confirmation year.

This is the interface that would enforce the lock before any final score exists.
It is deliberately small and has no dependency on any model, preprocessor or
scorer, so it can be exercised on synthetic fixtures without touching real data.

Contract
--------
* `seal(...)` writes a lock file recording the SHA256 of every declared input,
  the partition manifest hash, and the creation time. It refuses to overwrite.
* `open_final(...)` returns the final-partition rows **only if** every hashed input
  still matches and the lock exists. Any missing, extra or changed input aborts.
* Every successful open appends to an access log, with the lock digest and time.
  The first access must be strictly after the lock's creation.
* `amend(...)` may re-bind **code** files only. An amendment naming any non-code
  input, or any path not already in the lock, is rejected.

The loader never applies a model and never computes a score; it hands back row
indices and the arrays the caller declared, so the "no scoring before the lock"
rule is enforced by the file system, not by convention.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path

CODE_PREFIXES = ('experiments/', 'scripts/', 'tests/')


class SealError(PermissionError):
    """Raised whenever the sealed final partition may not be opened."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def lock_path(out):
    return Path(out) / 'YEAR_LOCK.json'


def access_log_path(out):
    return Path(out) / 'FINAL_ACCESS_LOG.json'


def seal(out, root, inputs, manifest, note=''):
    """Hash every declared input and write the lock. Refuses to overwrite."""
    out, root = Path(out), Path(root)
    path = lock_path(out)
    if path.exists():
        raise SealError('lock already exists; a second seal would reopen the year')
    files = {}
    for rel in sorted(dict.fromkeys(inputs)):
        target = root / rel
        if not target.is_file():
            raise SealError(f'declared input is missing: {rel}')
        files[rel] = sha256_file(target)
    payload = {'created_utc': _now(), 'root': str(root), 'files': files,
               'manifest_sha256': sha256_file(root / manifest), 'manifest': manifest,
               'note': note}
    payload['lock_digest'] = hashlib.sha256(
        json.dumps({k: payload[k] for k in ('files', 'manifest_sha256')}, sort_keys=True).encode()).hexdigest()
    _atomic_write_json(path, payload)
    return payload


def read_amendments(out):
    return [json.loads(p.read_text()) for p in sorted(Path(out).glob('YEAR_LOCK_AMENDMENT_*.json'))]


def amend(out, index, code_hashes, reason, original=None):
    """Re-bind named code files. Non-code paths and unknown paths are rejected here,
    so an invalid amendment can never reach `verify`."""
    out = Path(out)
    lock = json.loads(lock_path(out).read_text())
    for rel in code_hashes:
        if not rel.startswith(CODE_PREFIXES):
            raise SealError(f'amendment names a non-code input: {rel}')
        if rel not in lock['files']:
            raise SealError(f'amendment names a path that is not in the lock: {rel}')
    payload = {'created_utc': _now(), 'reason': reason, 'code_hashes': dict(code_hashes),
               'original_code_hashes': {k: lock['files'][k] for k in code_hashes},
               'supplied_original': original}
    _atomic_write_json(out / f'YEAR_LOCK_AMENDMENT_{index}.json', payload)
    return payload


def verify(out, root):
    """Return (lock, amendments) if every hashed input still matches, else raise."""
    out, root = Path(out), Path(root)
    path = lock_path(out)
    if not path.exists():
        raise SealError('final partition is sealed: YEAR_LOCK.json is missing')
    lock = json.loads(path.read_text())
    expected = dict(lock['files'])
    amendments = read_amendments(out)
    for a in amendments:
        for rel, digest in a['code_hashes'].items():
            if not rel.startswith(CODE_PREFIXES):
                raise SealError(f'invalid amendment: non-code input {rel}')
            if rel not in expected:
                raise SealError(f'invalid amendment: unknown path {rel}')
            expected[rel] = digest
    changed = []
    for rel, digest in expected.items():
        target = root / rel
        if not target.is_file():
            changed.append({'path': rel, 'reason': 'missing'})
        elif sha256_file(target) != digest:
            changed.append({'path': rel, 'reason': 'hash mismatch'})
    if changed:
        raise SealError('locked inputs changed: ' + json.dumps(changed))
    manifest = root / lock['manifest']
    if not manifest.is_file() or sha256_file(manifest) != lock['manifest_sha256']:
        raise SealError('partition manifest changed or missing')
    return lock, amendments


def open_final(out, root, manifest_key='final_evaluation', reason=''):
    """Verify the lock, log the access, and return the final partition's row index.

    Returns a dict with the row indices and the counts recorded at admission. It
    returns no label and computes no score: scoring is the caller's job, and the
    caller can only get here after the lock has verified.
    """
    lock, amendments = verify(out, root)
    manifest = json.loads((Path(root) / lock['manifest']).read_text())
    counts = manifest['partition_counts'][manifest_key]
    log_path = access_log_path(out)
    log = json.loads(log_path.read_text()) if log_path.exists() else {'accesses': []}
    entry = {'time_utc': _now(), 'lock_digest': lock['lock_digest'],
             'partition': manifest_key, 'reason': reason,
             'amendments': len(amendments)}
    if entry['time_utc'] <= lock['created_utc']:
        raise SealError('final access is not strictly after the lock')
    log['accesses'].append(entry)
    _atomic_write_json(log_path, log)
    return {'partition': manifest_key, 'counts': counts,
            'array_hash': manifest['array_hashes'][f'partition/{manifest_key}'],
            'lock_digest': lock['lock_digest'], 'access_index': len(log['accesses'])}
