"""Verifier check 8: access (no early DEV_REPORT access, no sealed access, ledger order). Own code; reads git metadata, the
queue ledger and both DEV_REPORT access logs (this campaign's and the cross-campaign source log). No data file is read.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

SRC_LOG = Path('<worktrees>/pcrl-acs-protected-tree-replication-v1/.private/acs_replication/access/DEV_REPORT_ACCESS.jsonl')
LOG = V.PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl'
LEDGER = V.PRIV / 'compute' / 'ledger.jsonl'
R = 'results/pcrl_scm_utility_matched_v1/'


def git(*a, timeout=30):
    r = subprocess.run(['git', '-C', str(V.ROOT), *a], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def T(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(dt.timezone.utc)


def commit_times(path):
    out = git('log', '--format=%h %cI', '--', path)
    return [(l.split()[0], T(l.split()[1])) for l in out.splitlines()]


def main():
    t0 = time.process_time()
    out = {'check': 8, 'title': 'access', 'order': {}, 'discrepancies': []}
    files = ['PROTOCOL.md', 'UNIT_INVENTORY.json', 'ANALYSIS_LOCK.json', 'SOURCE_ADMISSION.json', 'TRAJECTORY_COMPLETE_LOCK.json', 'TRAJECTORY_SUMMARY.json',
             'SELECTION_LOCK.json', 'SERVICE_BANK_MANIFEST.json', 'REPORTING_LOCK.json', 'READERS_SUMMARY.json', 'report/DEV_REPORT_RESULTS.json',
             'report/PRIMARY_ENDPOINTS.csv', 'report/SECONDARY_ENDPOINTS.csv', 'DEPLOYMENT_RESULTS.json']
    ct = {f: commit_times(R + f) for f in files}
    out['commits'] = {f: [(h, t.isoformat()) for h, t in v] for f, v in ct.items()}
    out['lock_files_committed_once'] = {f: len(v) == 1 for f, v in ct.items()}
    first = {f: v[-1][1] for f, v in ct.items() if v}
    ents = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    start = {e['unit_id']: T(e['utc']) for e in ents if e['event'] == 'unit_start'}
    end = {e['unit_id']: T(e['utc']) for e in ents if e['event'] == 'unit_end'}
    status = {e['unit_id']: e['status'] for e in ents if e['event'] == 'unit_end'}
    checks = {
        'TRAJ units start after ANALYSIS_LOCK commit': all(start[u] >= first['ANALYSIS_LOCK.json'] for u in ('UM_TRAJ_S0_V1', 'UM_TRAJ_S1_V1')),
        'SELECT (real labels) starts after TRAJECTORY_COMPLETE_LOCK commit': start['UM_SELECT_V1'] >= first['TRAJECTORY_COMPLETE_LOCK.json'],
        'SELECT dry run (invented labels) starts after TRAJECTORY_COMPLETE_LOCK commit': start['UM_SELECT_DRYRUN_V1'] >= first['TRAJECTORY_COMPLETE_LOCK.json'],
        'BANK starts after SELECTION_LOCK commit': start['UM_BANK_V1'] >= first['SELECTION_LOCK.json'],
        'READERS start after SERVICE_BANK_MANIFEST commit': all(start[u] >= first['SERVICE_BANK_MANIFEST.json'] for u in ('UM_READERS_S0_V1', 'UM_READERS_S1_V1')),
        'REPORTING_LOCK committed after READERS end': first['REPORTING_LOCK.json'] >= max(end['UM_READERS_S0_V1'], end['UM_READERS_S1_V1']),
        'REPORT starts after REPORTING_LOCK commit': start['UM_REPORT_V1'] >= first['REPORTING_LOCK.json'],
        'report results committed after REPORT end': first['report/DEV_REPORT_RESULTS.json'] >= end['UM_REPORT_V1'],
        'DEPLOY starts after REPORTING_LOCK commit': start['UM_DEPLOY_V1'] >= first['REPORTING_LOCK.json'],
        'verifier replay starts after the committed coordinator report': start['VERIFY_CHECK5_V1'] >= first['report/DEV_REPORT_RESULTS.json'],
        'verifier dry run precedes verifier replay': end['VERIFY_CHECK5_DRYRUN_V1'] <= start['VERIFY_CHECK5_V1'],
        'coordinator report dry run precedes report': end['UM_REPORT_DRYRUN_V1'] <= start['UM_REPORT_V1'],
        'all queue units done (no failed attempt)': all(v == 'done' for v in status.values())}
    out['order'] = checks
    out['unit_windows'] = {u: [start[u].isoformat(), end.get(u).isoformat() if end.get(u) else None] for u in start}
    # access logs
    log = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    src = [json.loads(l) for l in SRC_LOG.read_text().splitlines() if l.strip()]
    mine_src = [e for e in src if e.get('campaign') == 'pcrl_scm_utility_matched_v1']
    pub = V.public_export_sha()['DEV_REPORT']
    rl_sha = V.sha_file(V.RES / 'REPORTING_LOCK.json')
    acc = {'campaign_log_events': [(e['event'], e['unit_id'], e['utc']) for e in log],
           'source_log_events_of_this_campaign': [(e['event'], e['unit_id'], e['utc']) for e in mine_src],
           'logs_identical_for_this_campaign': log == mine_src,
           'exactly_one_coordinator_and_one_verifier_event': sorted(e['event'] for e in log) == ['DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP', 'DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER'],
           'no_crash_repair_event': not any(e['event'].endswith('CRASH_REPAIR') for e in log),
           'events_cite_committed_reporting_lock_hash': all(e['reporting_lock_sha256'] == rl_sha for e in log),
           'events_cite_public_truth_hashes': all(e['files_sha256'] == {k: pub[k] for k in ('income', 'occupation', 'sex')} for e in log),
           'events_inside_their_unit_windows': all(start[e['unit_id']] <= T(e['utc']) <= end[e['unit_id']] for e in log),
           'events_after_REPORTING_LOCK_commit': all(T(e['utc']) >= first['REPORTING_LOCK.json'] for e in log),
           'no_event_of_this_campaign_before_REPORTING_LOCK_in_source_log': all(T(e['utc']) >= first['REPORTING_LOCK.json'] for e in mine_src)}
    out['access_logs'] = acc
    # sealed
    sealed_hits = [l for l in LEDGER.read_text().splitlines() if 'SEALED' in l.upper()]
    sealed_log = [e for e in log + mine_src if 'SEALED' in json.dumps(e).upper() and 'sealed assessment untouched' not in e.get('note', '')]
    out['sealed'] = {'ledger_lines_mentioning_sealed': len(sealed_hits), 'access_events_touching_sealed': len(sealed_log),
                     'tracked_files_named_sealed': [f for f in git('ls-files').splitlines() if 'SEALED' in f.upper() and 'pcrl_scm_utility_matched_v1' in f],
                     'private_paths_named_sealed': [str(p) for p in V.PRIV.rglob('*') if 'SEALED' in p.name.upper()]}
    # lock utc fields vs commit times
    utc = {}
    for f in ('ANALYSIS_LOCK.json', 'TRAJECTORY_COMPLETE_LOCK.json', 'SELECTION_LOCK.json', 'REPORTING_LOCK.json'):
        u = T(V.jload(V.RES / f)['utc'])
        utc[f] = {'utc_field': u.isoformat(), 'commit': first[f].isoformat(), 'commit_not_before_utc_field': first[f] >= u}
    out['lock_utc_vs_commit'] = utc
    # publication receipt of the reporting lock
    rec = {}
    for c in ('3c7d69317',):
        rec[c] = {'exists_locally': git('cat-file', '-t', c) == 'commit', 'time': git('log', '-1', '--format=%cI', c),
                  'subject': git('log', '-1', '--format=%s', c)[:200], 'branches': git('branch', '-a', '--contains', c)[:400]}
        try:
            remote = git('ls-remote', 'origin', timeout=20)
            rec[c]['remote_refs_containing_prefix'] = [l for l in remote.splitlines() if l.startswith(git('rev-parse', c))]
        except Exception as e:  # network may be unavailable in the sandbox
            rec[c]['remote_check'] = f'not checked: {type(e).__name__}'
    out['reporting_receipt'] = rec
    # budget
    cpu = sum(float(e.get('charged_cpu_s', 0.0)) for e in ents if e['event'] in ('unit_end', 'charge'))
    ver = sum(float(e.get('charged_cpu_s', 0.0)) for e in ents if e['event'] == 'unit_end' and e['unit_id'].startswith('VERIFY_'))
    out['budget'] = {'ledger_charged_cpu_h_total': round(cpu / 3600, 4), 'verifier_queue_cpu_s': round(ver, 1),
                     'elapsed_h_since_registered_start': round((dt.datetime.now(dt.timezone.utc) - T('2026-10-17T04:38:00+00:00')).total_seconds() / 3600, 3)}
    ok = all(checks.values()) and all(v for k, v in acc.items() if isinstance(v, bool)) and not out['sealed']['access_events_touching_sealed'] \
        and not out['sealed']['private_paths_named_sealed'] and not out['sealed']['ledger_lines_mentioning_sealed']
    out['status'] = 'PASS' if ok else 'FAIL_OR_REVIEW'
    out['cpu_s'] = round(time.process_time() - t0, 2)
    V.write_json('check8_access.json', out)
    print(json.dumps({k: out[k] for k in ('status', 'order', 'access_logs', 'sealed', 'lock_utc_vs_commit', 'lock_files_committed_once', 'reporting_receipt', 'budget')}, indent=1, default=str))


if __name__ == '__main__':
    main()
