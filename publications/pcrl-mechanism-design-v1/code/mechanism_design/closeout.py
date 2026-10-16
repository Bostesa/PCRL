"""Closeout helpers: resource inventory from the append-only queue ledger.

`resources` writes results/pcrl_mechanism_design_v1/RESOURCE_INVENTORY.json: every queue unit (done and failed),
charged CPU (queue units plus logged agent/reviewer allowances), elapsed time from the campaign start, peak sampled
RSS, and the budget limits. Agent (LLM) work is not local CPU; it is accounted by an explicit allowance entry.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from . import jobqueue as Q

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'pcrl_mechanism_design_v1'


def resources() -> dict:
    es = Q.entries()
    ends = [e for e in es if e.get('event') == 'unit_end']
    starts = {e['unit_id']: e for e in es if e.get('event') == 'unit_start'}
    units = [{'unit_id': e['unit_id'], 'stage': e['stage'], 'status': e['status'], 'returncode': e['returncode'], 'wall_s': e['wall_s'],
              'charged_cpu_s': e['charged_cpu_s'], 'sampled_peak_rss_mib': round((e.get('sampled_peak_rss_bytes') or 0) / 2 ** 20, 1),
              'started_utc': starts.get(e['unit_id'], {}).get('utc'), 'ended_utc': e['utc']} for e in ends]
    charges = [e for e in es if e.get('event') == 'charge']
    start = Q.start_utc()
    now = dt.datetime.now(dt.timezone.utc)
    out = {
        'schema': 'MD1-resource-inventory-v1',
        'campaign_start_utc': start.isoformat(),
        'inventory_utc': now.isoformat(timespec='seconds'),
        'elapsed_h_at_inventory': round((now - start).total_seconds() / 3600, 3),
        'limits': {'cpu_h': Q.CPU_LIMIT_S / 3600, 'elapsed_h': Q.ELAPSED_LIMIT_H, 'slots': Q.SLOTS, 'rss_gib': Q.RSS_LIMIT_GIB,
                   'threads_per_job': 1, 'cloud_usd': 0},
        'units_done': sum(u['status'] == 'done' for u in units),
        'units_failed': sum(u['status'] != 'done' for u in units),
        'failed_units': [u['unit_id'] for u in units if u['status'] != 'done'],
        'queue_cpu_s': round(sum(u['charged_cpu_s'] for u in units), 3),
        'allowance_charges': [{'label': c['label'], 'cpu_s': c['charged_cpu_s'], 'note': c['note'], 'utc': c['utc']} for c in charges],
        'charged_cpu_h_total': round(Q.charged_cpu_s() / 3600, 4),
        'peak_unit_rss_mib': max([u['sampled_peak_rss_mib'] for u in units] or [0]),
        'cloud_spend_usd': 0,
        'units': units,
    }
    return out


ARTIFACTS = ['START_HERE.md', 'ADVISOR_BRIEF.md', 'ADVERSARY_AND_MOTIVATION.md', 'MANUSCRIPT_ADVERSARY_PATCH.diff', 'CLOSED_LANES.md',
             'CLOSEST_PRIOR_WORK.md', 'PROBLEM_SPECIFICATION.md', 'EVIDENCE_PACK.md', 'SELECTION_DECISION.md', 'METHOD_SPECIFICATION.md',
             'PILOT_PROTOCOL_DRAFT.md', 'DECISION_LOG.md', 'AGENT_ROSTER.json', 'RESOURCE_INVENTORY.json',
             'prototype/PROTOTYPE_PREREGISTRATION.json', 'prototype/PROTOTYPE_RESULTS.json', 'prototype/PROTOTYPE_REPORT.md',
             'prototype/DEMONSTRATIONS.json', 'pilot/PILOT_REHEARSAL_RESULTS.json', 'roles/designer_1/PROPOSAL.md',
             'roles/designer_2/PROPOSAL.md', 'roles/math_verifier/MATH_ASSESSMENT.md', 'roles/math_verifier/PROTOTYPE_VERIFICATION.md',
             'roles/adversary_reviewer/PROPOSAL_THREAT_ASSESSMENT.md', 'roles/prior_work_reviewer/PROPOSAL_COMPARISON.md',
             'BACKUP_VERIFICATION.json', 'PUBLICATION_RECORD.json']


def handoff(extra_path=None) -> dict:
    import hashlib
    import subprocess
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    git = lambda *a: subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    extra = json.loads(Path(extra_path).read_text()) if extra_path else {}
    return {
        'schema': 'MD1-handoff-v1',
        'written_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'head_at_write': git('rev-parse', 'HEAD'),
        'campaign_commits': git('log', '--format=%h %cI %s', '54f089d3e..HEAD').splitlines(),
        'artifacts_sha256': {a: sha(RESULTS / a) for a in ARTIFACTS},
        **extra,
    }


def main(argv=None):
    argv = argv or sys.argv[1:] or ['resources']
    cmd = argv[0]
    if cmd == 'resources':
        res = resources()
        (RESULTS / 'RESOURCE_INVENTORY.json').write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != 'units'}, indent=1))
    elif cmd == 'handoff':
        res = handoff(argv[1] if len(argv) > 1 else None)
        (RESULTS / 'HANDOFF.json').write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ('branch', 'head_at_write')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
