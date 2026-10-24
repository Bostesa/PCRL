"""Build the sanitized public package of the PCRL strong-anchor release campaign and commit it to its publication branch
(derived from nrm.publish_build).

    python3 -m sar.publish_build build DEST                 # copy results + sar/**/*.py, sanitize, leak scan; refuses on any problem
    python3 -m sar.publish_build commit DEST BRANCH MESSAGE_FILE [PARENT]

Copies results/pcrl_strong_anchor_release_v1/ and sar/**/*.py into DEST/publications/pcrl-strong-anchor-release-v1/
{results,code}, applying the established publication sanitizer (mechanism_design.publish_build.sanitise: personal paths, private
sources, submission numbers, the advisor's name), then its leak scan (mechanism_design.publish_check.scan). Private arrays, models,
reader artifacts and the .private/ tree are never copied. Gated registration files must come out of the sanitizer unchanged (the
receipt check compares their bytes with the research copy). Commits use git plumbing (hash-object, mktree, commit-tree,
update-ref); no working tree is switched; earlier branches and history are untouched. Pushing is a separate, explicit step.
"""
from __future__ import annotations
import json
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from mechanism_design import publish_check as PC
from mechanism_design.publish_build import sanitise

from . import common as C

NAME = C.PUBLICATION_NAME
UNPUBLISHED_SUFFIXES = ('.lock', '.jsonl')   # not accepted by the leak scan; left unpublished openly (listed in the build output)


def owner_patterns():
    """Owner identifiers, derived at run time so that no form of them is written into this code: the account's full name (spaced and
    joined), the home folder name and the host names. Each becomes a case-insensitive redaction applied after the shared sanitizer.
    Added 2026-10-24 after verifier wave 3 found the owner's name in a drive relocation folder name and the host name in the coordinator
    log (PROTOCOL_DEVIATIONS item 13); the shared sanitizer has no pattern for either."""
    names, hosts = set(), set()
    full = subprocess.run(['id', '-F'], capture_output=True, text=True).stdout.strip()
    if full:
        names |= {full, full.replace(' ', ''), full.replace(' ', '-'), full.replace(' ', '_')} | set(full.split())
    names.add(Path.home().name)
    mail = subprocess.run(['git', '-C', str(C.ROOT), 'config', 'user.email'], capture_output=True, text=True).stdout.strip()
    if '@' in mail:
        names.add(mail.split('@')[0])
    h = socket.gethostname()
    hosts |= {h, h.split('.')[0]}
    cn = subprocess.run(['scutil', '--get', 'ComputerName'], capture_output=True, text=True).stdout.strip()
    if cn:
        hosts.add(cn)
    pats = [(re.compile(re.escape(x), re.I), '<owner host>') for x in sorted(hosts, key=len, reverse=True) if len(x) >= 4]
    pats += [(re.compile(re.escape(x), re.I), '<owner>') for x in sorted(names, key=len, reverse=True) if len(x) >= 4]
    return pats


# Leak-scan hits reviewed by the coordinator (2026-10-24, before 07:29Z by `date -u`) and judged harmless: the verifier's own
# description, table and search patterns name the credential words its scan looked for; no credential is present. Matched on file and
# problem.
REVIEWED_SCAN_HITS = {('results/roles/verifier/VERIFICATION_REPORT.md', 'credential-like string'),
                      ('results/roles/verifier/code/v3_assemble.py', 'credential-like string'),
                      ('results/roles/verifier/code/v3_pubscan.py', 'credential-like string')}   # its own search regexes


def redact_owner(text: str, pats) -> str:
    for p, r in pats:
        text = p.sub(r, text)
    return text


def build(dest: Path, committed_only: bool = True) -> dict:
    """committed_only: publish only files that are tracked, committed and clean (the package then equals a research commit)."""
    out = dest / 'publications' / NAME
    if out.exists():
        shutil.rmtree(out)
    copied, changed, skipped, owner_redacted = [], [], [], []
    pats = owner_patterns()
    for f in sorted(C.RESULTS.rglob('*')):
        if not f.is_file() or f.name in ('.DS_Store',) or '__pycache__' in f.parts:
            continue
        if f.suffix in UNPUBLISHED_SUFFIXES:          # recorded: the verifier's lock file and its append-only .jsonl notes log
            skipped.append(f'{f.relative_to(C.ROOT)} (unpublished type {f.suffix}; content carried by CHECKS.json / REPORT.md)'); continue
        if committed_only and not C.committed(f):
            skipped.append(str(f.relative_to(C.ROOT))); continue
        d = out / 'results' / f.relative_to(C.RESULTS); d.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix.lower() in PC.TEXT_SUFFIXES:
            t = f.read_text(); s = sanitise(t); s2 = redact_owner(s, pats); d.write_text(s2)
            if s != t:
                changed.append(str(f.relative_to(C.RESULTS)))
            if s2 != s:
                owner_redacted.append(str(f.relative_to(C.RESULTS)))
        else:
            shutil.copy2(f, d)
        copied.append(str(d.relative_to(out)))
    for f in sorted((C.ROOT / 'sar').rglob('*.py')):
        if '__pycache__' in f.parts:
            continue
        if committed_only and not C.committed(f):
            skipped.append(str(f.relative_to(C.ROOT))); continue
        d = out / 'code' / f.relative_to(C.ROOT); d.parent.mkdir(parents=True, exist_ok=True)
        s = sanitise(f.read_text()); s2 = redact_owner(s, pats); d.write_text(s2); copied.append(str(d.relative_to(out)))
        if s2 != s:
            owner_redacted.append(str(f.relative_to(C.ROOT)))
    gated_changed = [f for f in C.REGISTRATION_FILES if f in changed]
    gated_owner_redacted = [f for f in C.REGISTRATION_FILES if f in owner_redacted]   # differ from the registered bytes only there
    scan = PC.scan(out)
    reviewed = [p for p in scan if isinstance(p, dict) and (p.get('file'), p.get('problem')) in REVIEWED_SCAN_HITS]
    problems = [p for p in scan if p not in reviewed]
    if gated_changed:
        problems.append(f'gated files changed by the sanitizer (must be publishable verbatim): {gated_changed}')
    left = [str(p.relative_to(out)) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix.lower() in PC.TEXT_SUFFIXES
            and any(q.search(p.read_text()) for q, _ in pats)]
    if left:
        problems.append(f'owner identifiers left after redaction: {left}')
    return {'out': str(out), 'files': len(copied), 'sanitized_files': changed, 'owner_redacted_files': owner_redacted,
            'gated_files_owner_redacted': gated_owner_redacted, 'reviewed_scan_hits': reviewed, 'skipped_uncommitted': skipped,
            'problems': problems}


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
        print(json.dumps({'files': res['files'], 'sanitized_files': res['sanitized_files'][:40],
                          'owner_redacted_files': res['owner_redacted_files'], 'gated_files_owner_redacted': res['gated_files_owner_redacted'],
                          'reviewed_scan_hits': res['reviewed_scan_hits'],
                          'skipped_uncommitted': res['skipped_uncommitted'][:60], 'problem_count': len(res['problems']),
                          'problems': res['problems'][:40]}, indent=1))
        return 1 if res['problems'] else 0
    if argv[0] == 'commit':
        print(json.dumps(commit(Path(argv[1]), argv[2], argv[3], argv[4] if len(argv) > 4 else None), indent=1))
        return 0
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main())
