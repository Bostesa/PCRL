"""Closeout receipts: RESOURCE_ACCOUNTING.json (queue ledger totals, failed attempts, agent charges, processes left running),
ACCESS_RECEIPT.json (DEV_REPORT follow-up events, ATTACK SEX reads, what was never read) and HANDOFF.json (branch, commits, locks,
decision, custody, open items). No label is read."""
from __future__ import annotations
import datetime as dt
import json
import subprocess

from . import common as C
from . import jobqueue as Q


def _git(*a):
    return subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True).stdout.strip()


def main():
    es = Q.entries(); ends = [e for e in es if e.get('event') == 'unit_end']
    start = Q.start_utc(); now = dt.datetime.now(dt.timezone.utc)
    by_stage = {}
    for e in ends:
        by_stage[e['stage']] = round(by_stage.get(e['stage'], 0.0) + float(e['charged_cpu_s']), 1)
    ps = subprocess.run(['ps', '-axo', 'pid,command'], capture_output=True, text=True).stdout.splitlines()
    mine = [l.strip() for l in ps if ('ffv.' in l or 'ffv/' in l) and 'closeout' not in l and 'ps -axo' not in l]
    res = {'schema': 'FFV1-resource-accounting-v1', 'utc': now.isoformat(timespec='seconds'), 'start_utc': start.isoformat(),
           'elapsed_h': round((now - start).total_seconds() / 3600, 3), 'elapsed_limit_h': Q.ELAPSED_LIMIT_H,
           'charged_cpu_h': round(Q.charged_cpu_s() / 3600, 3), 'cpu_limit_h': Q.CPU_LIMIT_S / 3600,
           'units_done': sum(e['status'] == 'done' for e in ends), 'units_failed': sum(e['status'] != 'done' for e in ends),
           'failed_units': [{'unit': e['unit_id'], 'stage': e['stage'], 'charged_cpu_s': e['charged_cpu_s']} for e in ends if e['status'] != 'done'],
           'cpu_s_by_stage': by_stage, 'agent_charges': [{k: e[k] for k in ('label', 'charged_cpu_s', 'note')} for e in es if e.get('event') == 'charge'],
           'units': [{k: e.get(k) for k in ('unit_id', 'stage', 'status', 'charged_cpu_s', 'wall_s', 'utc')} for e in ends],
           'heavy_process_limit': Q.SLOTS, 'threads_per_job': 1, 'cloud_or_paid_compute_usd': 0,
           'processes_left_running': mine}
    C.write_json(C.RESULTS / 'RESOURCE_ACCOUNTING.json', res)
    dev = C.dev_access_events()
    lab = [json.loads(l) for l in C.LABEL_LOG.read_text().splitlines() if l.strip()] if C.LABEL_LOG.is_file() else []
    acc = {'schema': 'FFV1-access-receipt-v1', 'utc': now.isoformat(timespec='seconds'),
           'dev_report_follow_up_events': [{k: e.get(k) for k in ('event', 'utc', 'unit_id', 'purpose', 'reporting_lock_sha256')} for e in dev],
           'dev_report_note': "already-exposed development role; new follow-up event names; appended to this campaign's log and to the lineage-wide log; the source campaign's logs and one-use counters untouched",
           'attack_sex_reads': [{k: e.get(k) for k in ('utc', 'role', 'purpose', 'unit_id')} for e in lab],
           'never_read': ['MODEL_FIT, CALIBRATION and MODEL_SELECT task labels and SEX (not admitted)', 'the sealed assessment (refused by every loader)',
                          'any new population, download or person-level file outside the admitted Florida 2019 development exports'],
           'network': 'git push of the publication branch to the existing GitHub remote only'}
    C.write_json(C.RESULTS / 'ACCESS_RECEIPT.json', acc)
    locks = {n: C.sha(C.RESULTS / n) for n in ('PROTOCOL.md', 'READER_BANK_LOCK.json', 'REPORTING_LOCK.json', 'FROZEN_CANDIDATE_MANIFEST.json', 'SOURCE_ADMISSION.json')}
    commits = _git('log', '--format=%H %cI %s', f'{C.SRC_TIP}..HEAD').splitlines()
    hand = {'schema': 'FFV1-handoff-v1', 'utc': now.isoformat(timespec='seconds'), 'branch': 'research/pcrl-frozen-finalist-validation-v1',
            'base': C.SRC_TIP, 'publication_branch': 'publication/pcrl-frozen-finalist-validation-v1',
            'nature': 'development robustness evidence on already-used Florida ACS 2019 development records; validation of a frozen candidate; not confirmation',
            'commits': [c[:160] for c in commits], 'locks_sha256': locks,
            'decision': json.loads((C.RESULTS / 'HANDOFF_DECISION.json').read_text()) if (C.RESULTS / 'HANDOFF_DECISION.json').exists() else None,
            'resources': {k: res[k] for k in ('elapsed_h', 'charged_cpu_h', 'units_done', 'units_failed', 'cloud_or_paid_compute_usd')},
            'custody': json.loads((C.RESULTS / 'BACKUP_VERIFICATION.json').read_text()) if (C.RESULTS / 'BACKUP_VERIFICATION.json').exists() else None,
            'processes_left_running': mine}
    C.write_json(C.RESULTS / 'HANDOFF.json', hand)
    print(json.dumps({'elapsed_h': res['elapsed_h'], 'charged_cpu_h': res['charged_cpu_h'], 'units': (res['units_done'], res['units_failed']),
                      'dev_events': [e['event'] for e in dev], 'processes_left_running': mine}, indent=1))


if __name__ == '__main__':
    main()
