"""Evidence manifest: every read-only input consumed and every artifact produced, hashed.

Separates the three categories the protocol requires to stay separate: historical
inputs that were reused unchanged, artifacts this study published, and local
fitted objects that stay on this machine and are not published.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .inputs import OUT, read_json, sha_file, write_json

# Published: aggregate evidence, protocol, method, diagnostics and manifests.
PUBLISHED_GLOBS = ('*.md', '*.json', '*.csv', 'figures/*', 'seed_*/fit_complete.json',
                   'seed_*/fit_diagnostics.json', 'seed_*/anchor_parity.json',
                   'seed_*/diagnostics.json', 'seed_*/dev_2018_inputs.json',
                   'seed_*/*/complete.json', 'seed_*/*/selection_before_test.json')
# Local only: fitted objects and raw per-person predictions. Never published.
LOCAL_GLOBS = ('seed_*/maps.joblib', 'seed_*/releases/*/*.npz', 'seed_*/*/predictions.npz',
               'seed_*/*/metrics.json', 'seed_*/*/utility_state.joblib',
               'seed_*/*/fitted/**/*', 'seed_*/*/audits/**/*')


def collect(out: Path) -> dict:
    out = Path(out)
    inputs = {}
    for name in ('FIT_SUMMARY.json', 'DIAGNOSTICS_SUMMARY.json', 'DEVELOPMENT_2018.json'):
        path = out / name
        if not path.exists():
            continue
        record = read_json(path)
        for block in (record.get('inputs'), *(v.get('inputs', {}) for v in
                                              record.get('seeds', {}).values()
                                              if isinstance(v, dict))):
            if isinstance(block, dict):
                inputs.update(block.get('files', {}))
    for seed in (0, 1, 2):
        for name in ('dev_2018_inputs.json',):
            path = out / f'seed_{seed}' / name
            if path.exists():
                inputs.update(read_json(path).get('files', {}))

    published, local = {}, {}
    for pattern in PUBLISHED_GLOBS:
        for path in sorted(out.glob(pattern)):
            if path.is_file():
                published[str(path.relative_to(out))] = sha_file(path)
    seen_local = 0
    local_bytes = 0
    for pattern in LOCAL_GLOBS:
        for path in sorted(out.glob(pattern)):
            if path.is_file():
                seen_local += 1
                local_bytes += path.stat().st_size
                if seen_local <= 400:                 # hash a bounded sample; count them all
                    local[str(path.relative_to(out))] = sha_file(path)
    return {
        'historical_inputs': {
            'count': len(inputs), 'note': ('resolved read-only from the original checkout or the '
                                           'completed study worktree; nothing historical was '
                                           'refitted, moved or modified'),
            'files': dict(sorted(inputs.items()))},
        'published_artifacts': {'count': len(published), 'files': published},
        'local_only_artifacts': {
            'count': seen_local, 'total_bytes': local_bytes, 'hashed_sample': len(local),
            'note': ('fitted maps, releases, fitted candidates and raw per-person predictions. '
                     'These stay on this machine: they are excluded by .gitignore and are not '
                     'published. Person rows never leave the machine.'),
            'files': local},
    }


def run(out: Path = OUT) -> dict:
    record = collect(out)
    write_json(Path(out) / 'EVIDENCE_MANIFEST.json', record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    r = run(a.out)
    print('historical inputs %d | published %d | local-only %d (%.1f GB)' % (
        r['historical_inputs']['count'], r['published_artifacts']['count'],
        r['local_only_artifacts']['count'],
        r['local_only_artifacts']['total_bytes'] / 1e9))


if __name__ == '__main__':
    main()
