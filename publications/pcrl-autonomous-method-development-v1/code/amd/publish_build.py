"""Build the sanitised public package and commit it to an orphan publication branch (private tool; also published).

    python -m amd.publish_build build DEST          # copy + sanitise + leak scan; refuses on any problem
    python -m amd.publish_build commit DEST BRANCH MESSAGE_FILE [PARENT]
        # hash the built tree into this repository with git plumbing (hash-object, mktree, commit-tree, update-ref);
        # no working tree is switched. Without PARENT the commit is a root (orphan) commit.

Copies results/pcrl_autonomous_method_development_v1/ and amd/*.py into
DEST/publications/pcrl-autonomous-method-development-v1/{results,code}, applying the replacements of the established
mechanism-design publication workflow (personal paths, private sources, submission numbers, the advisor's name), then
runs its leak scan (mechanism_design.publish_check.scan). Private arrays, models and the .private/ tree are never copied.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from mechanism_design import publish_check as PC
from mechanism_design.publish_build import sanitise

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results' / 'pcrl_autonomous_method_development_v1'
NAME = 'pcrl-autonomous-method-development-v1'


def build(dest: Path) -> dict:
    out = dest / 'publications' / NAME
    if out.exists():
        shutil.rmtree(out)
    copied = []
    for f in sorted(RES.rglob('*')):
        if not f.is_file() or f.name in ('.DS_Store',) or '__pycache__' in f.parts:
            continue
        d = out / 'results' / f.relative_to(RES); d.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix.lower() in PC.TEXT_SUFFIXES:
            d.write_text(sanitise(f.read_text()))
        else:
            shutil.copy2(f, d)
        copied.append(str(d.relative_to(out)))
    for f in sorted((ROOT / 'amd').glob('*.py')):
        d = out / 'code' / 'amd' / f.name; d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(sanitise(f.read_text())); copied.append(str(d.relative_to(out)))
    return {'out': str(out), 'files': len(copied), 'problems': PC.scan(out)}


def _git(*args, inp=None):
    r = subprocess.run(['git', '-C', str(ROOT), *args], input=inp, capture_output=True, text=True)
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
    _git('update-ref', f'refs/heads/{branch}', sha, *( [parent] if parent else ['0' * 40] ))
    return {'tree': tree, 'commit': sha, 'branch': branch, 'parent': parent}


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv[0] == 'build':
        res = build(Path(argv[1]))
        print(json.dumps({'files': res['files'], 'problem_count': len(res['problems']), 'problems': res['problems'][:40]}, indent=1))
        return 1 if res['problems'] else 0
    if argv[0] == 'commit':
        print(json.dumps(commit(Path(argv[1]), argv[2], argv[3], argv[4] if len(argv) > 4 else None), indent=1))
        return 0
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main())
