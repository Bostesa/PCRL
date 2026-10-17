"""Closeout: write RESOURCE_INVENTORY.json and HANDOFF.json for the structural-mechanism campaign.

    python -m structural_pilot.closeout EXTRA_JSON
EXTRA_JSON carries run-time facts (backup and off-device paths and checks, publication commit and remote SHA). Hashes,
ledger totals and commit lists are computed here. Nothing here reads a real-person array.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from . import jobqueue as Q

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results' / 'pcrl_structural_mechanism_pilot_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def main(extra_path):
    extra = json.loads(Path(extra_path).read_text())
    s = Q.summary(); start = dt.datetime.fromisoformat(s['start_utc']); now = dt.datetime.now(dt.timezone.utc)
    inv = {'schema': 'SMP1-resource-inventory-v1', 'utc': now.isoformat(timespec='seconds'),
           'elapsed_h': round((now - start).total_seconds() / 3600, 2), 'elapsed_limit_h': Q.ELAPSED_LIMIT_H,
           'charged_cpu_h': s['charged_cpu_h'], 'cpu_limit_h': Q.CPU_LIMIT_S / 3600, 'units_done': s['units_done'], 'units_failed': s['units_failed'],
           'agent_and_light_script_charges': [{k: e[k] for k in ('label', 'charged_cpu_s', 'note')} for e in s['agent_charges']],
           'heavy_process_limit': 2, 'threads_per_job': 1, 'cloud_or_paid_compute_usd': 0,
           'agents': extra.get('agents'), 'processes_left_running': extra.get('processes_left_running', 'none')}
    (RES / 'RESOURCE_INVENTORY.json').write_text(json.dumps(inv, indent=1) + '\n')
    files = sorted(str(p.relative_to(ROOT)) for p in RES.rglob('*') if p.is_file() and p.name != 'HANDOFF.json')
    files += sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'structural_pilot').glob('*') if p.is_file() and p.suffix in ('.py', '.sh'))
    rec = {'schema': 'SMP1-handoff-v1', 'branch': 'research/pcrl-structural-mechanism-pilot-v1', 'base': 'da5f5fe89 (research/pcrl-mechanism-design-correction-v1)',
           'status': ('NO_CANDIDATE_SURVIVED at candidate selection; no pilot; no defense or reader fit; no MODEL_SELECT label array loaded, '
                      'no DEV_REPORT truth re-read, no sealed access'),
           'failed_prerequisite': ('candidate selection: both proposals fail S3 (distinct ingredient) as committed, which alone triggers rule 1; '
                                   'a separate post-reading ruling places both outside the intent of S2 (substantive structure)'),
           'not_this': ['external service refusal', 'missing artifact', 'implementation failure', 'failed utility prerequisite', 'negative scientific result'],
           'commits': git('log', '--format=%h %s', 'da5f5fe89..HEAD').splitlines(),
           'proposal_sha256': {'designer_1_SCM': sha(RES / 'roles/designer_1/PROPOSAL.md'), 'designer_2_SCAR': sha(RES / 'roles/designer_2/PROPOSAL.md')},
           'criteria_commit': '2d96c471e (before either proposal existed)',
           'file_sha256': {f: sha(ROOT / f) for f in files}, 'resources': inv}
    rec.update(extra)
    (RES / 'HANDOFF.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: rec[k] for k in ('status', 'failed_prerequisite')} | {'elapsed_h': inv['elapsed_h'], 'charged_cpu_h': inv['charged_cpu_h']}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
