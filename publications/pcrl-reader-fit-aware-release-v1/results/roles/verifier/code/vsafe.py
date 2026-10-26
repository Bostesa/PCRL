"""RFA1 verifier, wave 5: run-time derivation of identifying strings and a printing guard (counts-only scans).

Nothing identifying is written into this file: every owner identifier is derived at run time (the account's full name in its
spaced / joined / hyphenated / underscored forms and its parts, the account and home-folder names, the git user e-mail and its local
part, the host names). The git user name (the public repository handle, which also appears in the registered remote URL) is kept as a
separate category. safe() replaces any identifying substring before anything is printed or written.
"""
from __future__ import annotations

import os
import pwd
import re
import socket
import subprocess
from pathlib import Path

SCRATCH_ID = None


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ''


def identifiers(git_email_file=None, git_name_file=None):
    """{category: sorted set of strings}; categories: owner_name, user_name, email_full, email_local, host, git_handle."""
    full = _run(['id', '-F'])
    owner = set()
    if full:
        owner |= {full, full.replace(' ', ''), full.replace(' ', '-'), full.replace(' ', '_'), full.replace(' ', '.')} | set(full.split())
    users = {Path.home().name, pwd.getpwuid(os.getuid()).pw_name, os.environ.get('USER', ''), os.environ.get('LOGNAME', ''), _run(['id', '-un'])}
    hosts = set()
    h = socket.gethostname()
    hosts |= {h, h.split('.')[0]}
    for key in ('ComputerName', 'LocalHostName', 'HostName'):
        v = _run(['scutil', '--get', key])
        if v and 'not set' not in v:
            hosts |= {v, v.replace(' ', '-'), v.replace(' ', '')}
    email = Path(git_email_file).read_text().strip() if git_email_file and Path(git_email_file).is_file() else ''
    gname = Path(git_name_file).read_text().strip() if git_name_file and Path(git_name_file).is_file() else ''
    out = {'owner_name': owner, 'user_name': users, 'host': hosts,
           'email_full': {email} if '@' in email else set(), 'email_local': {email.split('@')[0]} if '@' in email else set(),
           'git_handle': {gname} if gname else set()}
    return {k: sorted({x for x in v if x and len(x) >= 4}, key=len, reverse=True) for k, v in out.items()}


def compile_ids(ids):
    return {k: [re.compile(re.escape(x), re.I) for x in v] for k, v in ids.items()}


def count_ids(text, cids):
    return {k: sum(len(p.findall(text)) for p in pats) for k, pats in cids.items()}


def safe(s, cids):
    """Replace every identifying substring (any category) before printing or writing."""
    s = str(s)
    for k, pats in cids.items():
        for p in pats:
            s = p.sub(f'<{k}>', s)
    return s
