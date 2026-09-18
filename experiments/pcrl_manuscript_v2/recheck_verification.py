"""Corrected companion to the locked study's INDEPENDENT_VERIFICATION.json.

The committed artifact was written before the third lock amendment existed, so its
`code_only_amendments` field lists two amendments and its `changed: []` verdict was
computed against the pre-amendment-3 report code. This script recomputes the lock
verification over every hashed input with **all three** amendments applied, and records
the ordering of the amendments against the final-partition access log.

It is read-only. It hashes files, writes nothing outside `--out`, runs on one worker,
and reruns no model prediction, fit or score. The historical artifact is never edited;
this is published beside it with its own provenance.

Usage:
    PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.recheck_verification \\
        --study-root <worktree holding the study's local artifacts> \\
        --published results/redesign_20260917_acs_spectral_transport_v1 \\
        --out results/pcrl_manuscript_review_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--study-root', required=True,
                    help='directory that relative locked paths resolve against')
    ap.add_argument('--published', default='results/redesign_20260917_acs_spectral_transport_v1')
    ap.add_argument('--out', default='results/pcrl_manuscript_review_v2')
    args = ap.parse_args()

    pub = Path(args.published)
    root = Path(args.study_root).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lock = json.loads((pub / 'TRANSPORT_LOCK.json').read_text())
    published_verification = json.loads((pub / 'INDEPENDENT_VERIFICATION.json').read_text())

    amendment_files = sorted(pub.glob('TRANSPORT_LOCK_AMENDMENT_*.json'))
    amendments = {p.name: json.loads(p.read_text()) for p in amendment_files}

    # which amendments does the committed artifact actually cover?
    covered = []
    for check in published_verification['checks']:
        if check['check'] == 'lock_inputs_unchanged_or_amended':
            covered = sorted(check['detail'].get('code_only_amendments', {}))
    not_covered = [n for n in amendments if n not in covered]

    # amendment validity: an amendment may only re-bind locked code paths
    invalid = []
    effective = dict(lock['files'])
    rebound: dict[str, list[str]] = {}
    for name in sorted(amendments):
        a = amendments[name]
        paths = sorted(a.get('code_hashes', {}))
        rebound[name] = paths
        for k in paths:
            if not (k.startswith('experiments/') or k.startswith('scripts/')):
                invalid.append({'amendment': name, 'path': k, 'why': 'not a code path'})
            elif k not in lock['files']:
                invalid.append({'amendment': name, 'path': k, 'why': 'not a locked input'})
        effective.update(a.get('code_hashes', {}))

    # the check itself
    changed, missing = [], []
    hashed_bytes = 0
    for name, digest in effective.items():
        p = Path(name) if name.startswith('/') else root / name
        if not p.exists():
            missing.append(name)
            continue
        hashed_bytes += p.stat().st_size
        if sha256(p) != digest:
            changed.append(name)

    # a locked input may only be re-bound if it is code
    non_code_rebound = sorted({k for paths in rebound.values() for k in paths
                               if not (k.startswith('experiments/') or k.startswith('scripts/'))})

    # ordering: did any final-partition read happen after the last amendment?
    log = json.loads((pub / 'FINAL_ACCESS_LOG.json').read_text())
    accesses = log if isinstance(log, list) else log.get('accesses', [])
    times = sorted(a.get('utc') or a.get('timestamp') or a.get('time') for a in accesses)
    amendment_times = {n: amendments[n]['created_utc'] for n in sorted(amendments)}
    reads_after = {n: [t for t in times if t > ts] for n, ts in amendment_times.items()}

    report = {
        'artifact': 'CORRECTED_INDEPENDENT_VERIFICATION.json',
        'purpose': 'companion to results/redesign_20260917_acs_spectral_transport_v1/'
                   'INDEPENDENT_VERIFICATION.json, which predates lock amendment 3. '
                   'The historical artifact is preserved unedited.',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'study_root_used_for_resolution': str(root),
        'lock_created_utc': lock['created_utc'],
        'lock_file_count': lock['file_count'],
        'effective_file_count_after_amendments': len(effective),
        'amendment_files_on_disk': sorted(amendments),
        'amendments_covered_by_the_published_artifact': covered,
        'amendments_NOT_covered_by_the_published_artifact': not_covered,
        'rebound_paths_by_amendment': rebound,
        'non_code_paths_rebound': non_code_rebound,
        'invalid_amendment_entries': invalid,
        'inputs_hashed': len(effective) - len(missing),
        'inputs_missing': missing,
        'bytes_hashed': hashed_bytes,
        'inputs_changed_after_all_amendments': changed,
        'pass_with_all_amendments': not changed and not missing and not invalid,
        'final_partition_accesses': len(times),
        'first_final_access_utc': times[0] if times else None,
        'last_final_access_utc': times[-1] if times else None,
        'amendment_created_utc': amendment_times,
        'final_reads_after_each_amendment': {n: len(v) for n, v in reads_after.items()},
        'scope': 'This recomputes the lock verification only. It reruns no prediction, no fit '
                 'and no score, and it establishes nothing about attack strength or about the '
                 'cause of the numerical corruption recorded in amendment 2.',
    }
    (out / 'CORRECTED_INDEPENDENT_VERIFICATION.json').write_text(
        json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('rebound_paths_by_amendment',)}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
