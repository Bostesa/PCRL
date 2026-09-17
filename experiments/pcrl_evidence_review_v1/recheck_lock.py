"""Independent re-verification of TRANSPORT_LOCK.json with *all* amendments applied.

The study's published `INDEPENDENT_VERIFICATION.json` records
`code_only_amendments` for amendments 1 and 2 only, because it was produced before
`TRANSPORT_LOCK_AMENDMENT_3.json` was written. This script repeats the lock check
over all 17,639 hashed inputs against the current tree, applying every amendment
file present, and reports which inputs are covered by which amendment.

Read-only. One worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CODE_PREFIXES = ('experiments/', 'scripts/')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, help='directory holding TRANSPORT_LOCK.json and the amendments')
    ap.add_argument('--root', required=True, help='repository root the lock paths are relative to')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    study, root, out = Path(args.study), Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lock = json.loads((study / 'TRANSPORT_LOCK.json').read_text())
    amendment_files = sorted(study.glob('TRANSPORT_LOCK_AMENDMENT_*.json'))
    amendments = {p.name: json.loads(p.read_text()) for p in amendment_files}

    expected = dict(lock['files'])
    rebound = {}
    invalid = []
    for name, a in amendments.items():
        for rel, digest in a['code_hashes'].items():
            if not rel.startswith(CODE_PREFIXES) or rel not in lock['files']:
                invalid.append({'amendment': name, 'path': rel})
                continue
            expected[rel] = digest
            rebound.setdefault(name, []).append(rel)

    changed, missing = [], []
    for rel, digest in expected.items():
        target = root / rel
        if not target.is_file():
            missing.append(rel)
        elif sha256_file(target) != digest:
            changed.append(rel)

    # which of the published report's verification claims still hold
    published = json.loads((study / 'INDEPENDENT_VERIFICATION.json').read_text())
    published_amendments = next(
        c['detail'].get('code_only_amendments', {}) for c in published['checks']
        if c['check'] == 'lock_inputs_unchanged_or_amended')

    report = {
        'lock_files': len(lock['files']),
        'amendment_files_on_disk': [p.name for p in amendment_files],
        'amendments_covered_by_published_verification': sorted(published_amendments),
        'amendments_not_covered_by_published_verification': sorted(
            set(amendments) - set(published_amendments)),
        'rebound_paths_by_amendment': rebound,
        'invalid_amendment_entries': invalid,
        'inputs_missing': missing,
        'inputs_changed_after_all_amendments': changed,
        'pass_with_all_amendments': not (missing or changed or invalid),
        'non_code_inputs_rebound': [e for e in invalid],
        'note': ('Amendment 3 re-binds scripts/report_acs_spectral_transport.py from '
                 'a2705f89... to 38205cc6... for a CSV to gzip-CSV serialisation change. '
                 'The published verification artifact predates it, so its "changed: []" '
                 'verdict was computed against the pre-amendment-3 report code.'),
    }
    (out / 'LOCK_RECHECK.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rebound_paths_by_amendment'}, indent=2))


if __name__ == '__main__':
    main()
