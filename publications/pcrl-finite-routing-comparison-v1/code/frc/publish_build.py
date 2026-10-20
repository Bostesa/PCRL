"""Build the sanitized public package of the PCRL finite-routing comparison campaign and commit it to its publication branch
(derived from nrm.publish_build).

    python3 -m frc.publish_build build DEST                 # copy results + frc/**/*.py, sanitize, leak scan; refuses on any problem
    python3 -m frc.publish_build commit DEST BRANCH MESSAGE_FILE [PARENT]

Copies results/pcrl_finite_routing_comparison_v1/ and frc/**/*.py into DEST/publications/pcrl-finite-routing-comparison-v1/
{results,code}, applying the established publication sanitizer (mechanism_design.publish_build.sanitise: personal paths, private
sources, submission numbers, the advisor's name), then its leak scan (mechanism_design.publish_check.scan). Private arrays, models,
reader artifacts and the .private/ tree are never copied. Gated registration files must come out of the sanitizer unchanged (the
receipt check compares their bytes with the research copy). Commits use git plumbing (hash-object, mktree, commit-tree,
update-ref); no working tree is switched; earlier branches and history are untouched. Pushing is a separate, explicit step.
"""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
from pathlib import Path

from mechanism_design import publish_check as PC
from mechanism_design.publish_build import sanitise

from . import common as C

NAME = C.PUBLICATION_NAME
UNPUBLISHED_SUFFIXES = ('.lock', '.jsonl')   # not accepted by the leak scan; left unpublished openly (listed in the build output)


def build(dest: Path, committed_only: bool = True) -> dict:
    """committed_only: publish only files that are tracked, committed and clean (the package then equals a research commit)."""
    out = dest / 'publications' / NAME
    if out.exists():
        shutil.rmtree(out)
    copied, changed, skipped = [], [], []
    for f in sorted(C.RESULTS.rglob('*')):
        if not f.is_file() or f.name in ('.DS_Store',) or '__pycache__' in f.parts:
            continue
        if f.suffix in UNPUBLISHED_SUFFIXES:          # recorded: the verifier's lock file and its append-only .jsonl notes log
            skipped.append(f'{f.relative_to(C.ROOT)} (unpublished type {f.suffix}; content carried by CHECKS.json / REPORT.md)'); continue
        if committed_only and not C.committed(f):
            skipped.append(str(f.relative_to(C.ROOT))); continue
        d = out / 'results' / f.relative_to(C.RESULTS); d.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix.lower() in PC.TEXT_SUFFIXES:
            t = f.read_text(); s = sanitise(t); d.write_text(s)
            if s != t:
                changed.append(str(f.relative_to(C.RESULTS)))
        else:
            shutil.copy2(f, d)
        copied.append(str(d.relative_to(out)))
    for f in sorted((C.ROOT / 'frc').rglob('*.py')):
        if '__pycache__' in f.parts:
            continue
        if committed_only and not C.committed(f):
            skipped.append(str(f.relative_to(C.ROOT))); continue
        d = out / 'code' / f.relative_to(C.ROOT); d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(sanitise(f.read_text())); copied.append(str(d.relative_to(out)))
    gated_changed = [f for f in C.REGISTRATION_FILES if f in changed]
    problems = PC.scan(out)
    if gated_changed:
        problems.append(f'gated files changed by the sanitizer (must be publishable verbatim): {gated_changed}')
    return {'out': str(out), 'files': len(copied), 'sanitized_files': changed, 'skipped_uncommitted': skipped, 'problems': problems}


def _git(*args, inp=None):
    r = subprocess.run(['git', '-C', str(C.ROOT), *args], input=inp, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'git {args[0]} failed: {r.stderr.strip()}')
    return r.stdout.strip()


def _tree(path: Path) -> str:
    entries = []
    for p in sorted(path.iterdir(), key=lambda q: q.name):
        if p.is_dir():
            entries.append(f'040000 tree {_tree(p)}\t{p.name}')
        else:
            entries.append(f'100644 blob {_git("hash-object", "-w", str(p))}\t{p.name}')
    return _git('mktree', inp='\n'.join(entries) + '\n')


def commit(dest: Path, branch: str, message_file: str, parent: str | None = None) -> dict:
    tree = _tree(dest)
    args = ['commit-tree', tree, '-F', message_file] + (['-p', parent] if parent else [])
    sha = _git(*args)
    _git('update-ref', f'refs/heads/{branch}', sha, *([parent] if parent else ['0' * 40]))
    return {'tree': tree, 'commit': sha, 'branch': branch, 'parent': parent}


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv[0] == 'build':
        res = build(Path(argv[1]))
        print(json.dumps({'files': res['files'], 'sanitized_files': res['sanitized_files'][:40], 'skipped_uncommitted': res['skipped_uncommitted'][:60], 'problem_count': len(res['problems']),
                          'problems': res['problems'][:40]}, indent=1))
        return 1 if res['problems'] else 0
    if argv[0] == 'commit':
        print(json.dumps(commit(Path(argv[1]), argv[2], argv[3], argv[4] if len(argv) > 4 else None), indent=1))
        return 0
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main())
