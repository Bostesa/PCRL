"""Real-data refusal for the prepared (unrun) pilot.

This campaign is authorised for data-free work only. The pilot's real-data path refuses unless ALL hold:
1. results/pcrl_mechanism_design_v1/pilot/PILOT_AUTHORIZATION.json exists, is committed and clean in git;
2. it names this pilot's protocol hash, an explicit user authorisation statement and a future campaign id;
3. it keeps SEALED_ASSESSMENT closed.
This campaign never creates that file. SEALED_ASSESSMENT is refused unconditionally, and DEV_REPORT truth is refused
unless a committed pilot REPORTING_LOCK exists (also never created here).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / 'results' / 'pcrl_mechanism_design_v1' / 'pilot'
AUTH = PILOT_DIR / 'PILOT_AUTHORIZATION.json'
REPORTING_LOCK = PILOT_DIR / 'PILOT_REPORTING_LOCK.json'
REQUIRED_KEYS = ('protocol_sha256', 'user_authorization_statement', 'future_campaign_id', 'sealed_assessment')


def _committed_clean(path: Path) -> bool:
    if not path.is_file():
        return False
    log = subprocess.run(['git', '-C', str(ROOT), 'log', '-1', '--format=%H', '--', str(path)], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', str(path)], capture_output=True, text=True)
    return bool(log.stdout.strip()) and not dirty.stdout.strip()


def require_real_data_authorization(protocol_sha256: str) -> dict:
    if not _committed_clean(AUTH):
        raise PermissionError('real-data pilot refused: no committed PILOT_AUTHORIZATION.json (this campaign is data-free)')
    rec = json.loads(AUTH.read_text())
    if any(k not in rec for k in REQUIRED_KEYS) or rec['protocol_sha256'] != protocol_sha256 or rec['sealed_assessment'] != 'closed':
        raise PermissionError('real-data pilot refused: authorization record incomplete or for a different protocol')
    return rec


def refuse_role(role: str) -> None:
    if role == 'SEALED_ASSESSMENT':
        raise PermissionError('SEALED_ASSESSMENT is closed')


def require_reporting_lock() -> None:
    if not _committed_clean(REPORTING_LOCK):
        raise PermissionError('DEV_REPORT truth refused: no committed pilot REPORTING_LOCK')
