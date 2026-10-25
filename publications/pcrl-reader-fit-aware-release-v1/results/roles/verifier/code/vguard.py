"""RFA1 verifier: label-free guard for wave-1 code (invented data only).

install() adds a Python audit hook that refuses (PermissionError) any open() of a path under a `.private` directory, under
<worktrees>/, or of a saved model file (MODEL.pkl, *.pt, RELEASE.npz, GRID.json, BOOT.npz, devgrid_cache),
and records every refused attempt. Later waves that are authorized to read labels through rfa.common.load(..., operation='verify')
do NOT install this guard. One numerical thread is enforced by environment variables before numpy / torch are imported.
"""
from __future__ import annotations

import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

REFUSED = []
_FORBIDDEN_PARTS = ('/.private/', '/.private', '<worktrees>/')
_FORBIDDEN_NAMES = ('MODEL.pkl', 'RELEASE.npz', 'GRID.json', 'BOOT.npz', 'BOOT_EXTRAS.npz', 'LABEL_ACCESS.jsonl')
_FORBIDDEN_SUFFIX = ('.pt',)
_INSTALLED = [False]


_INVENTED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out', 'invented_')


def _bad(path):
    p = str(path)
    if any(x in p for x in _FORBIDDEN_PARTS):
        return True                                   # never relaxed
    if p.startswith(_INVENTED_DIR):
        return False                                  # the verifier's own invented-data outputs (grid files of invented arrays)
    if 'devgrid_cache' in p:
        return True
    base = os.path.basename(p)
    return base in _FORBIDDEN_NAMES or base.endswith(_FORBIDDEN_SUFFIX)


def _hook(event, args):
    if event == 'open' and args:
        path = args[0]
        if isinstance(path, (str, bytes, os.PathLike)):
            p = os.fsdecode(path) if not isinstance(path, str) else path
            if _bad(os.path.abspath(p)):
                REFUSED.append(p)
                raise PermissionError(f'verifier wave-1 guard: refused to open {p} (label-free wave)')


def install():
    if not _INSTALLED[0]:
        sys.addaudithook(_hook)
        _INSTALLED[0] = True
    return True


def selfcheck():
    """The guard refuses a forbidden path (nothing is created) and allows an ordinary one."""
    ok = {}
    probe = '<repository>/.claude/worktrees/pcrl-reader-fit-aware-release-v1/.private/__verifier_guard_probe__'
    try:
        open(probe, 'rb')
        ok['refuses_private'] = False
    except PermissionError:
        ok['refuses_private'] = True
    except FileNotFoundError:
        ok['refuses_private'] = False
    try:
        with open(__file__, 'rb') as fh:
            fh.read(1)
        ok['allows_code'] = True
    except PermissionError:
        ok['allows_code'] = False
    return ok
