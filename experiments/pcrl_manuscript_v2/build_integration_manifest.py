"""Record what was integrated, from which commits, and the hash of every deliverable.

Usage:
    PYTHONPATH=. python -m experiments.pcrl_manuscript_v2.build_integration_manifest \\
        --repo . --out results/pcrl_manuscript_review_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

BASE = '349efa454afd907389760fd1f59fd8806a215efd'
EVIDENCE = '0d8f4b67b6d4961dfa133289d0167c874d2f4794'
NONLIN = 'c37807e4f568ef38e5528fc09c1506083278bf4d'

DELIVERABLES = [
    'papers/pcrl_manuscript_v2/main.tex',
    'papers/pcrl_manuscript_v2/main.pdf',
    'papers/pcrl_manuscript_v2/references.bib',
    'papers/pcrl_manuscript_v2/MANIFEST.json',
    'papers/pcrl_manuscript_v2/DERIVED_FACTS.json',
    'results/pcrl_manuscript_review_v2/CLAIM_LEDGER.csv',
    'results/pcrl_manuscript_review_v2/CLAIM_CHANGES.md',
    'results/pcrl_manuscript_review_v2/CORRECTED_RESEARCH_DECISION.md',
    'results/pcrl_manuscript_review_v2/CORRECTED_INDEPENDENT_VERIFICATION.json',
    'results/pcrl_manuscript_review_v2/RELATED_WORK_SCOPE.md',
    'results/pcrl_manuscript_review_v2/VERIFICATION_STATUS.md',
    'results/pcrl_manuscript_review_v2/REVIEWER_RISKS.md',
    'results/pcrl_manuscript_review_v2/REPRODUCE_PAPER.md',
    'results/pcrl_manuscript_review_v2/REVIEW_INDEX.md',
    'results/pcrl_manuscript_review_v2/RUN_STATUS.md',
    'results/pcrl_manuscript_review_v2/PENDING_ADDENDUM.md',
    'experiments/pcrl_manuscript_v2/make_assets_v2.py',
    'experiments/pcrl_manuscript_v2/build_claim_ledger.py',
    'experiments/pcrl_manuscript_v2/recheck_verification.py',
    'experiments/pcrl_manuscript_v2/build_integration_manifest.py',
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(root), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--out', default='results/pcrl_manuscript_review_v2')
    args = ap.parse_args()
    root = Path(args.repo).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def commits(rng: str) -> list[str]:
        raw = git(root, 'log', '--format=%H %s', rng)
        return [l for l in raw.splitlines() if l]

    deliverables = {}
    for rel in DELIVERABLES:
        p = root / rel
        deliverables[rel] = ({'sha256': sha256(p), 'bytes': p.stat().st_size}
                             if p.exists() else {'status': 'absent'})

    assets = json.loads((root / 'papers/pcrl_manuscript_v2/MANIFEST.json').read_text())

    manifest = {
        'branch': git(root, 'rev-parse', '--abbrev-ref', 'HEAD'),
        'head': git(root, 'rev-parse', 'HEAD'),
        'worktree': str(root),
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'sources': {
            'common_baseline': {'sha': BASE, 'role': 'the state both studies branched from'},
            'evidence_and_previous_manuscript': {
                'sha': EVIDENCE, 'branch': 'research/pcrl-evidence-paper-v1',
                'commits_since_baseline': commits(f'{BASE}..{EVIDENCE}'),
                'integrated': 'in full, by merge',
                'brings': ['results/pcrl_evidence_review_v1/', 'papers/pcrl_evidence_v1/',
                           'experiments/pcrl_evidence_review_v1/',
                           'tests/pcrl_evidence_review_v1/'],
            },
            'nonlinear_rank_study': {
                'sha': NONLIN, 'branch': 'research/pcrl-nonlinear-rank-v1',
                'commits_since_baseline': commits(f'{BASE}..{NONLIN}'),
                'integrated': 'in full, by merge',
                'brings': ['results/pcrl_nonlinear_rank_v1/',
                           'experiments/pcrl_nonlinear_rank_v1/',
                           'tests/pcrl_nonlinear_rank_v1/'],
            },
        },
        'integration_method': {
            'how': 'git merge --no-ff of the nonlinear/rank branch into the evidence branch, in a '
                   'fresh sibling worktree. The two branches descend from the common baseline and '
                   'touch disjoint paths; the merge produced no conflicts.',
            'conflicts': 0,
            'unrelated_commits_included': 'none: each source branch was merged at exactly the '
                                          'stated SHA, which is its tip',
            'original_worktrees_modified': 'none',
            'main_modified': 'no; never merged to, never force-pushed',
            'historical_artifacts_edited': 'none: results/redesign_20260917_acs_spectral_transport_v1, '
                                           'results/pcrl_evidence_review_v1 and '
                                           'results/pcrl_nonlinear_rank_v1 are byte-identical to '
                                           'their source commits',
        },
        'owned_paths': ['papers/pcrl_manuscript_v2/', 'results/pcrl_manuscript_review_v2/',
                        'experiments/pcrl_manuscript_v2/'],
        'terminal_1_experiment': {
            'included': False,
            'reason': 'no committed Terminal 1 result exists. Its worktree '
                      '/Users/nathansamson/PCRL-terminal-a is clean at ' + NONLIN + ' and no new '
                      'experiment directory, handoff or protocol has appeared.',
            'specification_for_later_integration':
                'results/pcrl_manuscript_review_v2/PENDING_ADDENDUM.md',
        },
        'deliverables': deliverables,
        'generated_assets': {k: v['generated_by'] for k, v in assets['assets'].items()},
        'asset_source_hashes': assets['assets'],
    }
    path = out / 'INTEGRATION_MANIFEST.json'
    path.write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'wrote {path}')
    print('head', manifest['head'])
    print('deliverables present:',
          sum(1 for v in deliverables.values() if 'sha256' in v), 'of', len(deliverables))
    absent = [k for k, v in deliverables.items() if 'sha256' not in v]
    if absent:
        print('ABSENT:', *absent, sep='\n  ')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
