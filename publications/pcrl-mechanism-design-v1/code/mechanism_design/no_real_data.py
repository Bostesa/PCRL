"""Refuse real-person data in this campaign's processes.

This campaign is authorised to use published aggregates, code and provenance metadata only. `install()` adds a Python
audit hook that raises PermissionError when the process tries to open:
- anything under a `.private/` directory other than `.private/mechanism_design/` (role exports, predictions, models,
  reader banks and labels of every earlier campaign live there);
- the private evaluation cache and backups (`~/PCRL_eval_cache_private`, except this campaign's own
  `mechanism_design_v1/` backups) or the external backup drive (this campaign writes its off-device copy there with
  the custody tool, which does not install the guard);
- raw ACS files (`psam_*`, `csv_p*.zip`);
- array or model files (.npz, .npy, .parquet, .pkl, .joblib, .pt, .cbm, .feather) outside this campaign's own
  synthetic area (`/mechanism_design/` paths) and the interpreter's own installation.

The hook is a guard rail, not a sandbox: it stops accidental reads by this campaign's code (prototype, verifier,
pilot rehearsal on synthetic stand-ins). The real-data pilot entry point refuses to run at all in this campaign
(see mechanism_design/pilot/authorization.py).
"""
from __future__ import annotations

import os
import sys

_ARRAY_SUFFIXES = ('.npz', '.npy', '.parquet', '.pkl', '.joblib', '.pt', '.cbm', '.feather')
_HOME = os.path.expanduser('~')
_DENY_PREFIXES = (os.path.join(_HOME, 'PCRL_eval_cache_private'), '/Volumes/YOTUO')
_OWN_BACKUPS = os.path.join(_HOME, 'PCRL_eval_cache_private', 'mechanism_design_v1')   # this campaign's own backups only
_ALLOW_PREFIXES = tuple(p for p in {sys.prefix, sys.base_prefix, getattr(sys, 'exec_prefix', sys.prefix), _OWN_BACKUPS} if p)
_installed = False


def refusal(path: str) -> str | None:
    """Return the reason a path is refused, or None if it may be opened."""
    p = os.path.abspath(os.path.expanduser(path))
    if any(p.startswith(a + os.sep) for a in _ALLOW_PREFIXES) or '/site-packages/' in p:
        return None
    if '/.private/' in p and '/.private/mechanism_design/' not in p:
        return 'earlier campaigns\' private role data'
    if any(p == d or p.startswith(d + os.sep) for d in _DENY_PREFIXES):
        return 'private evaluation cache or external backup'
    base = os.path.basename(p)
    if base.startswith('psam_') or (base.startswith('csv_p') and base.endswith('.zip')):
        return 'raw ACS file'
    if p.lower().endswith(_ARRAY_SUFFIXES) and '/mechanism_design' not in p:
        return 'array/model file outside this campaign\'s synthetic area'
    return None


def _hook(event, args):
    if event != 'open' or not args:
        return
    path = args[0]
    if isinstance(path, int):
        return
    try:
        path = os.fsdecode(path)
    except TypeError:
        return
    reason = refusal(path)
    if reason is not None:
        raise PermissionError(f'no_real_data guard: refused to open {path!r} ({reason})')


def install() -> None:
    global _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True
