"""Closeout: RESOURCE_ACCOUNTING.json and HANDOFF.json (aggregates and hashes only; no person-level data).

    python -m amd.closeout EXTRA_JSON
EXTRA_JSON supplies what only the coordinator knows (roles actually performed, custody, publication, decision, open items).
"""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from . import common as C
from . import jobqueue as Q


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True).stdout.strip()


def main(extra_path):
    extra = json.loads(Path(extra_path).read_text())
    s = Q.summary(); start = dt.datetime.fromisoformat(s['start_utc']); now = dt.datetime.now(dt.timezone.utc)
    ends = [e for e in Q.entries() if e.get('event') == 'unit_end']
    with open(C.RESULTS / 'EXPERIMENT_LEDGER.csv') as f:
        ledger = list(csv.DictReader(f))
    inv = {'schema': 'AMD1-resource-accounting-v1', 'utc': now.isoformat(timespec='seconds'), 'start_utc': s['start_utc'],
           'elapsed_h': round((now - start).total_seconds() / 3600, 2), 'elapsed_limit_h': Q.ELAPSED_LIMIT_H,
           'charged_cpu_h': s['charged_cpu_h'], 'cpu_limit_h': Q.CPU_LIMIT_S / 3600, 'units_done': s['units_done'], 'units_failed': s['units_failed'],
           'failed_units': s['failed_units'],
           'new_seed_specific_configurations': sum(int(r['counts_toward_36'] or 0) for r in ledger), 'configuration_limit': 36,
           'configurations_by_family': {fam: sum(int(r['counts_toward_36'] or 0) for r in ledger if r['family'] == fam) for fam in sorted({r['family'] for r in ledger})},
           'units': [{k: e.get(k) for k in ('unit_id', 'stage', 'status', 'wall_s', 'charged_cpu_s', 'utc')} for e in ends],
           'charges': [{k: e[k] for k in ('label', 'charged_cpu_s', 'note')} for e in s['agent_charges']],
           'heavy_process_limit': Q.SLOTS, 'threads_per_job': 1, 'cloud_or_paid_compute_usd': 0, 'roles': extra.get('roles'),
           'processes_left_running': extra.get('processes_left_running', 'none')}
    (C.RESULTS / 'RESOURCE_ACCOUNTING.json').write_text(json.dumps(inv, indent=1) + '\n')
    files = sorted(str(p.relative_to(C.ROOT)) for p in C.RESULTS.rglob('*') if p.is_file() and p.name != 'HANDOFF.json')
    files += sorted(str(p.relative_to(C.ROOT)) for p in (C.ROOT / 'amd').glob('*') if p.is_file() and p.suffix in ('.py', '.sh'))
    rec = {'schema': 'AMD1-handoff-v1', 'branch': 'research/pcrl-autonomous-method-development-v1', 'base': '482808212 (SCM utility-matched research tip)',
           'nature': 'exploratory method-development campaign on previously used Florida ACS 2019 development records; development evidence, not confirmation',
           'locks_sha256': {n: sha(C.RESULTS / n) for n in ('PROTOCOL.md', 'SOURCE_ADMISSION.json', 'SOURCE_ADMISSION_2.json', 'PROMOTION.json', 'FINALIST_LOCK.json',
                                                           'ANALYSIS_LOCK.json', 'SERVICE_BANK_MANIFEST.json', 'REPORTING_LOCK.json') if (C.RESULTS / n).exists()},
           'commits': git('log', '--format=%h %s', '482808212..HEAD').splitlines(),
           'file_sha256': {f: sha(C.ROOT / f) for f in files}, 'resources': {k: v for k, v in inv.items() if k != 'units'}}
    rec.update(extra)
    (C.RESULTS / 'HANDOFF.json').write_text(json.dumps(rec, indent=1, default=float) + '\n')
    print(json.dumps({'elapsed_h': inv['elapsed_h'], 'charged_cpu_h': inv['charged_cpu_h'], 'configurations': inv['new_seed_specific_configurations']}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
