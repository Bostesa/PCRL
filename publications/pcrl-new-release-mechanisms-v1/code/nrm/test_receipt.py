"""Data-free tests of nrm.receipt (temporary directories and a fake git; no person data, no network).

    python3 -m nrm.test_receipt

Checks the repaired defect: a FAILED verification appends an attempt but never creates the write-once record, so a later retry
can still succeed; a verified attempt creates the record exactly once; a second verified attempt is refused; every attempt is kept.
"""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

from . import receipt as R
from . import common as C


def fake_git(remote_tip, blob_ok=True, url=C.REMOTE_URL):
    def git(*a):
        if a[:2] == ('remote', 'get-url'):
            return 0, url, ''
        if a[0] == 'ls-remote':
            return (0, f'{remote_tip}\trefs/heads/x', '') if remote_tip else (2, '', 'network down')
        if a[0] == 'rev-parse':
            return 0, 'blob1', ''
        if a[0] == 'hash-object':
            return 0, 'blob1' if blob_ok else 'blob2', ''
        if a[0] == 'merge-base':
            return 0, '', ''
        raise AssertionError(a)
    return git


def main():
    out = {}
    with tempfile.TemporaryDirectory() as td:
        res, acc = Path(td) / 'results', Path(td) / 'access'
        res.mkdir()
        for f in C.REGISTRATION_FILES:
            (res / f).parent.mkdir(parents=True, exist_ok=True)
            (res / f).write_text('{"x": 1}\n')
        kw = dict(results=res, access=acc, committed=lambda p: True, now=lambda: '2026-10-18T19:00:00+00:00')
        r1 = R.verify('registration', 'aaa', 'bbb', git=fake_git('zzz'), **kw)          # remote tip differs -> failed
        out['failed_no_record'] = (not r1['remote_sha_verified']) and not (acc / 'REGISTRATION_RECEIPT_VERIFIED.json').exists()
        r2 = R.verify('registration', 'aaa', 'bbb', git=fake_git(None), **kw)           # network failure -> failed, retry allowed
        out['network_failure_no_record'] = (not r2['remote_sha_verified']) and not (acc / 'REGISTRATION_RECEIPT_VERIFIED.json').exists()
        r3 = R.verify('registration', 'aaa', 'bbb', git=fake_git('aaa', blob_ok=False), **kw)   # bytes differ -> failed
        out['bytes_differ_failed'] = not r3['remote_sha_verified']
        r4 = R.verify('registration', 'aaa', 'bbb', git=fake_git('aaa'), **kw)          # verified -> record written
        out['verified_record_written'] = r4['remote_sha_verified'] and (acc / 'REGISTRATION_RECEIPT_VERIFIED.json').exists()
        try:
            R.verify('registration', 'aaa', 'bbb', git=fake_git('aaa'), **kw)
            out['second_verified_refused'] = False
        except PermissionError:
            out['second_verified_refused'] = True
        attempts = [json.loads(l) for l in (acc / 'RECEIPT_ATTEMPTS.jsonl').read_text().splitlines()]
        out['all_attempts_logged'] = [a['outcome'] for a in attempts] == ['failed', 'failed', 'failed', 'verified', 'verified']
        pub = json.loads((res / 'RECEIPTS.json').read_text())
        out['public_summary_once'] = len(pub['receipts']) == 1
        r5 = R.verify('registration', 'aaa', 'bbb', git=fake_git('aaa', url='/local/mirror'), **kw)
        out['local_mirror_refused'] = not r5['remote_sha_verified']
    ok = all(out.values())
    print(json.dumps({'ok': ok, **out}, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
