"""Closeout: RESOURCE_INVENTORY.json and HANDOFF.json for the SCM engineering benchmark (aggregates and hashes only).

    python -m scm_bench.closeout EXTRA_JSON
"""
from __future__ import annotations
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
    inv = {'schema': 'SCM1-resource-inventory-v1', 'utc': now.isoformat(timespec='seconds'), 'elapsed_h': round((now - start).total_seconds() / 3600, 2),
           'elapsed_limit_h': Q.ELAPSED_LIMIT_H, 'charged_cpu_h': s['charged_cpu_h'], 'cpu_limit_h': Q.CPU_LIMIT_S / 3600,
           'units_done': s['units_done'], 'units_failed': s['units_failed'], 'failed_units': s['failed_units'],
           'charges': [{k: e[k] for k in ('label', 'charged_cpu_s', 'note')} for e in s['agent_charges']],
           'heavy_process_limit': 2, 'threads_per_job': 1, 'cloud_or_paid_compute_usd': 0, 'roles': extra.get('roles'),
           'processes_left_running': extra.get('processes_left_running', 'none')}
    (C.RESULTS / 'RESOURCE_INVENTORY.json').write_text(json.dumps(inv, indent=1) + '\n')
    rep = json.loads((C.RESULTS / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    files = sorted(str(p.relative_to(C.ROOT)) for p in C.RESULTS.rglob('*') if p.is_file() and p.name != 'HANDOFF.json')
    files += sorted(str(p.relative_to(C.ROOT)) for p in (C.ROOT / 'scm_bench').glob('*') if p.is_file() and p.suffix in ('.py', '.sh'))
    rec = {'schema': 'SCM1-handoff-v1', 'branch': 'research/pcrl-scm-engineering-benchmark-v1', 'base': 'ec20c8bdf (stopped structural campaign tip)',
           'nature': 'existing-method engineering benchmark of Shared-Cell Merging; no novelty claim; exploratory development evidence',
           'scientific_status': {k: (v['decision'] if isinstance(v, dict) else v) for k, v in rep['claims'].items()},
           'separate_status_fields': rep['separate_status_fields'], 'controls': rep['controls'],
           'selection': {k: sel[k] for k in ('status', 'nominee', 'twin', 'ibsi', 'shuf', 'eligible_scm_settings', 'twin_matched_capacity', 'falsifier_F1_check')},
           'source_hashes': json.loads((C.RESULTS / 'ADMISSION.json').read_text()),
           'locks_sha256': {n: sha(C.RESULTS / n) for n in ('ANALYSIS_LOCK.json', 'TRAINING_COMPLETE_LOCK.json', 'SELECTION_LOCK.json', 'REPORTING_LOCK.json',
                                                           'SERVICE_BANK_MANIFEST.json', 'PROTOCOL.md')},
           'commits': git('log', '--format=%h %s', 'ec20c8bdf..HEAD').splitlines(),
           'file_sha256': {f: sha(C.ROOT / f) for f in files}, 'resources': inv}
    rec.update(extra)
    (C.RESULTS / 'HANDOFF.json').write_text(json.dumps(rec, indent=1, default=float) + '\n')
    print(json.dumps({'scientific_status': rec['scientific_status'], 'elapsed_h': inv['elapsed_h'], 'charged_cpu_h': inv['charged_cpu_h']}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
