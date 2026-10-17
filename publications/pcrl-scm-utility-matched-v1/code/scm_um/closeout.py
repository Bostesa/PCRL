"""Closeout: RESOURCE_ACCOUNTING.json and HANDOFF.json for the SCM utility-matched follow-up (aggregates and hashes only).

    python -m scm_um.closeout EXTRA_JSON
EXTRA_JSON supplies what only the coordinator knows (roles actually performed, custody, publication, open items).
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
    ends = [e for e in Q.entries() if e.get('event') == 'unit_end']
    inv = {'schema': 'SCMUM1-resource-accounting-v1', 'utc': now.isoformat(timespec='seconds'), 'start_utc': s['start_utc'],
           'elapsed_h': round((now - start).total_seconds() / 3600, 2), 'elapsed_limit_h': Q.ELAPSED_LIMIT_H,
           'charged_cpu_h': s['charged_cpu_h'], 'cpu_limit_h': Q.CPU_LIMIT_S / 3600, 'units_done': s['units_done'], 'units_failed': s['units_failed'],
           'failed_units': s['failed_units'],
           'units': [{k: e.get(k) for k in ('unit_id', 'stage', 'status', 'wall_s', 'charged_cpu_s', 'sampled_peak_rss_bytes', 'utc')} for e in ends],
           'charges': [{k: e[k] for k in ('label', 'charged_cpu_s', 'note')} for e in s['agent_charges']],
           'heavy_process_limit': Q.SLOTS, 'threads_per_job': 1, 'cloud_or_paid_compute_usd': 0, 'roles': extra.get('roles'),
           'processes_left_running': extra.get('processes_left_running', 'none')}
    (C.RESULTS / 'RESOURCE_ACCOUNTING.json').write_text(json.dumps(inv, indent=1) + '\n')
    rep = json.loads((C.RESULTS / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    files = sorted(str(p.relative_to(C.ROOT)) for p in C.RESULTS.rglob('*') if p.is_file() and p.name != 'HANDOFF.json')
    files += sorted(str(p.relative_to(C.ROOT)) for p in (C.ROOT / 'scm_um').glob('*') if p.is_file() and p.suffix in ('.py', '.sh'))
    rec = {'schema': 'SCMUM1-handoff-v1', 'branch': 'research/pcrl-scm-utility-matched-v1', 'base': '2738cc30b (SCM engineering benchmark research tip)',
           'nature': 'exploratory engineering comparison of existing methods on previously used Florida ACS 2019 development records; no novelty claim',
           'scientific_status': {k: (v['decision'] if isinstance(v, dict) and 'decision' in v else v) for k, v in rep['claims'].items()},
           'research_decision': rep['research_decision'], 'separate_status_fields': rep['separate_status_fields'], 'controls': rep['controls'],
           'selection': {'status': sel['status'], 't_star': sel['nomination']['t_star'], 'K_star': sel['nomination']['K_star'], 'e_star': sel['nomination']['e_star'],
                         'matching': {k: v for k, v in sel['matching'].items() if k != 'candidates'}, 'selection_effort': sel['selection_effort']},
           'locks_sha256': {n: sha(C.RESULTS / n) for n in ('PROTOCOL.md', 'UNIT_INVENTORY.json', 'ANALYSIS_LOCK.json', 'TRAJECTORY_COMPLETE_LOCK.json',
                                                           'SELECTION_LOCK.json', 'SERVICE_BANK_MANIFEST.json', 'REPORTING_LOCK.json') if (C.RESULTS / n).exists()},
           'commits': git('log', '--format=%h %s', '2738cc30b..HEAD').splitlines(),
           'file_sha256': {f: sha(C.ROOT / f) for f in files}, 'resources': inv}
    rec.update(extra)
    (C.RESULTS / 'HANDOFF.json').write_text(json.dumps(rec, indent=1, default=float) + '\n')
    print(json.dumps({'scientific_status': rec['scientific_status'], 'elapsed_h': inv['elapsed_h'], 'charged_cpu_h': inv['charged_cpu_h']}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
