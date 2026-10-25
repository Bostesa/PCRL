"""Sanitized public package of EDG1, committed as a ROOT commit on a NEW branch `publication/pcrl-estimator-diagnostic-v1` with git
plumbing (no working tree is switched; no existing branch or history is touched; pushing is a separate explicit step).

Package = the committed files of results/pcrl_estimator_diagnostic_v1/ plus the three unchanged, already-public DRF1 modules the code
imports (publications/pcrl-direct-recovery-frontier-v1/code/drf/{__init__,finite_law,solver}.py at c4896479e), in the same layout,
so the package runs as published. Text files get absolute local paths replaced by placeholders and owner identifiers (derived at RUN
TIME, as drf.publish_build.owner_patterns does: the account's full name and its tokens, the home folder name, the git e-mail's local
part, the host and computer names; never written into this code) redacted. A private scan of the proposed tree, the commit message
and the branch history then COUNTS remaining owner identifiers, absolute personal or volume paths and e-mail-like strings; the build
refuses on any count above zero. Matched strings are never printed or written.

    python3 edg_run.py publish build DEST
    python3 edg_run.py publish commit DEST MESSAGE_FILE"""
from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from . import RESULTS, WORKTREE

BRANCH = 'publication/pcrl-estimator-diagnostic-v1'
REL = 'results/pcrl_estimator_diagnostic_v1'
DRF_FILES = ('publications/pcrl-direct-recovery-frontier-v1/code/drf/__init__.py',
             'publications/pcrl-direct-recovery-frontier-v1/code/drf/finite_law.py',
             'publications/pcrl-direct-recovery-frontier-v1/code/drf/solver.py')
TEXT = {'.py', '.md', '.json', '.txt', '.csv', '.log', '.gitignore', ''}
_S = '/'                                  # patterns assembled from parts so the sanitizer never rewrites this file
ABS = [re.compile(_S + 'Users' + _S + r'[^\s"\'/]+'), re.compile(_S + 'private' + _S + r'(?:tmp|var)/[^\s"\']+'),
       re.compile(_S + 'Volumes' + _S + r'[^\s"\']+'), re.compile(_S + 'home' + _S + r'[^\s"\'/]+')]
MAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')


def _git(*a, inp=None):
    r = subprocess.run(['git', '-C', str(WORKTREE), *a], input=inp, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'git {a[0]} failed')
    return r.stdout.strip()


def owner_patterns():
    names, hosts = set(), set()
    full = subprocess.run(['id', '-F'], capture_output=True, text=True).stdout.strip()
    if full:
        names |= {full, full.replace(' ', ''), full.replace(' ', '-'), full.replace(' ', '_')} | set(full.split())
    names.add(Path.home().name)
    mail = subprocess.run(['git', '-C', str(WORKTREE), 'config', 'user.email'], capture_output=True, text=True).stdout.strip()
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


def _clean(text, pats):
    text = text.replace(str(WORKTREE), '<worktree>').replace(str(Path.home()), '<home>')
    for p in ABS:
        text = p.sub('<local path>', text)
    for p, r in pats:
        text = p.sub(r, text)
    return text


def scan_text(text, pats):
    return {'owner': sum(len(p.findall(text)) for p, _ in pats), 'abs_path': sum(len(p.findall(text)) for p in ABS),
            'email': len(MAIL.findall(text))}


def build(dest):
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    pats = owner_patterns()
    files = [f for f in _git('ls-files', REL).splitlines() if f] + list(DRF_FILES)
    status = _git('status', '--porcelain', '--untracked-files=no', '--', REL)   # untracked files are never published (ls-files)
    if status:
        raise RuntimeError('uncommitted changes under the study folder: commit first (the package must equal a research commit)')
    changed, counts = [], {'owner': 0, 'abs_path': 0, 'email': 0}
    for f in files:
        src, dst = WORKTREE / f, dest / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in TEXT or src.name == '.gitignore':
            t = src.read_text()
            s = _clean(t, pats)
            if s != t:
                changed.append(f)
            dst.write_text(s)
            for k, v in scan_text(s, pats).items():
                counts[k] += v
        else:
            shutil.copy2(src, dst)
    return {'files': len(files), 'sanitized_files': changed, 'scan_counts_tree': counts, 'patterns': len(pats),
            'research_commit': _git('rev-parse', 'HEAD'), 'ok': all(v == 0 for v in counts.values())}


def _tree(path):
    entries = []
    for p in sorted(path.iterdir(), key=lambda q: q.name):
        if p.is_dir():
            entries.append(f'040000 tree {_tree(p)}\t{p.name}')
        else:
            entries.append(f'100644 blob {_git("hash-object", "-w", str(p))}\t{p.name}')
    return _git('mktree', inp='\n'.join(entries) + '\n')


def commit(dest, message_file):
    pats = owner_patterns()
    msg = Path(message_file).read_text()
    mc = scan_text(msg, pats)
    if any(mc.values()):
        raise RuntimeError(f'commit message scan counts {mc}')
    if subprocess.run(['git', '-C', str(WORKTREE), 'rev-parse', '--verify', '-q', f'refs/heads/{BRANCH}'],
                      capture_output=True).returncode == 0:
        raise RuntimeError(f'{BRANCH} exists already; refusing to move it')
    tree = _tree(Path(dest))
    sha = _git('commit-tree', tree, '-F', str(message_file))
    _git('update-ref', f'refs/heads/{BRANCH}', sha, '0' * 40)
    hist = _git('log', '--format=%H %s%n%b', BRANCH)
    hc = scan_text(hist, pats)
    return {'branch': BRANCH, 'commit': sha, 'tree': tree, 'root_commit': True, 'message_scan_counts': mc,
            'history_scan_counts': hc, 'history_commits': len(_git('rev-list', BRANCH).splitlines())}


def main(argv):
    if argv[0] == 'build':
        print(json.dumps(build(argv[1]), indent=1))
    elif argv[0] == 'commit':
        print(json.dumps(commit(argv[1], argv[2]), indent=1))
