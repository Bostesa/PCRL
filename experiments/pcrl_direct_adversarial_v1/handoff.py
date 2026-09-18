"""Atomic status updates for Terminal 2, plus the committed public handoff record.

Two destinations, deliberately different:

* the **private** shared directory under the resolved git common directory, which may
  carry absolute paths because it never leaves this machine;
* the **public** `HANDOFF.json` inside the results tree, which carries **no private
  paths and no data** and is committed.

Every write is atomic, so a reader never sees a partial file. The public record names
the **published evidence commit**, never a pre-commit internal HEAD.
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import tempfile
from pathlib import Path

from .inputs import OUT, read_json, sha_file, write_json

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'pcrl_direct_adversarial_v1'
BRANCH = 'research/pcrl-direct-adversarial-v1'

PREDECESSORS = {
    'transport_2017_locked': '349efa454afd907389760fd1f59fd8806a215efd',
    'nonlinear_rank': 'c37807e4f568ef38e5528fc09c1506083278bf4d',
    'invariant_baselines': '73903b7f28df68284285f0610a4036beb32b208f',
    'manuscript_integrated_v2': '3c6ada82e719489656da820b72ce8425d2079be0',
}

PUBLIC_FILES = (
    'PROTOCOL.md', 'METHOD.md', 'MATRIX.json', 'PROTOCOL_FREEZE.json', 'RUN_STATUS.md',
    'CORRECTIONS.md', 'REPRODUCE.md', 'BASELINE_TRANSPORT_COMPLETION.md',
    'DEVELOPMENT_2018.md', 'DEVELOPMENT_2018.json', 'EXPLORATORY_2017.md',
    'EXPLORATORY_2017.json', 'FRONTIER_ANALYSIS.md', 'ATTACK_STRENGTH.md',
    'OPTIMIZATION_STABILITY.md', 'VALIDATION.md', 'VALIDATION.json', 'MECHANISM.json',
    'RESEARCH_DECISION.md', 'PAPER_ADDENDUM.md', 'ARTIFACT_MANIFEST.json',
    'PER_SEED.csv', 'PAIRED_INTERVALS.csv', 'PER_SEED_SIGNS.csv', 'SCORE_REPLAY.csv',
    'EXPLORATORY_2017.csv', 'OPTNET_RECONSTRUCTION.json', 'NEW_ERASURE.json',
)


def git(*args, cwd=ROOT) -> str:
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True,
                          text=True, check=False).stdout.strip()


def common_dir() -> Path:
    path = Path(git('rev-parse', '--path-format=absolute', '--git-common-dir'))
    return path if path.is_absolute() else (ROOT / path).resolve()


def private_dir() -> Path:
    destination = common_dir() / 'pcrl_parallel_handoff_v3' / 'terminal_1'
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_hashes(out: Path) -> dict:
    return {name: {'sha256': sha_file(out / name), 'bytes': (out / name).stat().st_size}
            for name in PUBLIC_FILES if (out / name).exists()}


def publish(milestone: str, completed, outstanding, notes=None, out: Path = OUT,
            evidence_commit: str = None) -> dict:
    """Write one atomic status update to both destinations."""
    out = Path(out)
    head = git('rev-parse', 'HEAD')
    record = {
        'terminal': 1, 'role': 'experiments', 'study': STUDY, 'branch': BRANCH,
        'milestone': milestone, 'updated_utc': now(),
        'evidence_commit': evidence_commit,
        'internal_head_at_write_time': head,
        'evidence_commit_note': (
            'evidence_commit is the COMMIT THAT CONTAINS the published evidence. '
            'internal_head_at_write_time is whatever HEAD happened to be when this record '
            'was written and is NOT a substitute for it. When evidence_commit is null the '
            'evidence for this milestone is not yet committed.'),
        'predecessor_commits': PREDECESSORS,
        'completed_units': completed,
        'outstanding_work': outstanding,
        'artifact_hashes': file_hashes(out),
        'notes': notes or [],
        'boundaries': [
            '2016 remains SEALED and UNUSED. Nothing in this study touches it.',
            'Every number here is DEVELOPMENT on repeatedly used pools.',
            'The historical first locked 2017 result keeps its original status and is '
            'neither restated nor overwritten.',
            'Terminal 2 may revise prose but must not silently change any scientific '
            'endpoint, selection, decision rule or reported sign.',
        ],
    }
    write_json(out / 'HANDOFF.json', record)

    private = dict(record)
    private['private_paths'] = {
        'worktree': str(ROOT), 'results': str(out),
        'logs': str(out / 'logs'),
        'invariant_worktree_read_only': os.environ.get(
            'PCRL_DAX_INVARIANT_ROOT', '/Users/nathansamson/PCRL-terminal-1-invariant'),
    }
    destination = private_dir() / 'STATUS.json'
    fd, tmp = tempfile.mkstemp(dir=destination.parent, suffix='.tmp')
    with os.fdopen(fd, 'w') as handle:
        import json
        handle.write(json.dumps(private, indent=2, sort_keys=False) + '\n')
    os.replace(tmp, destination)

    history = private_dir() / 'HISTORY.jsonl'
    with open(history, 'a') as handle:
        import json
        handle.write(json.dumps({'milestone': milestone, 'updated_utc': record['updated_utc'],
                                 'evidence_commit': evidence_commit,
                                 'completed_units': completed}) + '\n')
    print('HANDOFF', milestone, '->', destination, flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--milestone', required=True)
    p.add_argument('--completed', nargs='*', default=[])
    p.add_argument('--outstanding', nargs='*', default=[])
    p.add_argument('--notes', nargs='*', default=[])
    p.add_argument('--evidence-commit')
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    publish(a.milestone, a.completed, a.outstanding, a.notes, a.out, a.evidence_commit)


if __name__ == '__main__':
    main()
