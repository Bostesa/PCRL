"""Closeout receipts of the PCRL correction-transfer campaign (no label is read).

    python3 -m cts.closeout

-> results/<campaign>/RESOURCE_ACCOUNTING.json  elapsed time, queue CPU by stage and unit prefix, units done / failed (all attempts
   kept), agent charges, fitted release specifications against the 72 ceiling, internal fits (model, cross-fitting, routing,
   critics, calibrator, reader) with cache hits counted separately, people / training repetitions / model and reader seeds,
   processes left running;
-> results/<campaign>/ACCESS_RECEIPT.json        every label read by operation, entry point, role, field and partition role (from
   the single access log), repair-panel reads and their timing relative to the committed REPAIR_FREEZE, the lineage DEV_REPORT log
   against its campaign-start snapshot, what was never read.
Local paths are reduced to module or script names.
"""
from __future__ import annotations
import datetime as dt
import json
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path

from scm_bench import common as SB
from . import common as C
from . import jobqueue as Q

LINEAGE_SNAPSHOT = 'c7829e99e6af303cdf070f698b6088e024f3c2b806974a6b1f711785bb50c567'     # 19:26Z, campaign start
CEILING = 72


def _git(*a):
    return subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True).stdout.strip()


def freeze_commit_utc(first=True):
    """Committer time of the first (or last) commit touching REPAIR_FREEZE.json as an aware UTC datetime; None if never committed.
    git's %cI carries the committer's local offset (here -04:00), so it must not be string-compared with the UTC access log."""
    out = _git('log', '--format=%ct', '--', str(C.RESULTS / C.FREEZE_FILE)).split()
    if not out:
        return None
    return dt.datetime.fromtimestamp(int(out[-1] if first else out[0]), dt.timezone.utc)


def panel_reads_before_freeze(panel_summary, freeze_utc):
    """Repair-panel reads by new-release operations (not the whole-role source replay / support count) at or before the first
    freeze commit; with no committed freeze, every such read counts."""
    return [e for e in panel_summary if e['operation'] not in ('source_replay', 'support_count')
            and (freeze_utc is None or dt.datetime.fromisoformat(e['utc']) <= freeze_utc)]


def fitted_specs():
    """Fitted release specifications per training repetition: witness fits, correction directions, compression codes, nested
    variant corrections, Track R income lineages, sensitivity witness."""
    specs = []
    for P in sorted(p.name for p in (C.RUNS / 'TRACKF').iterdir() if p.is_dir()) if (C.RUNS / 'TRACKF').is_dir() else []:
        d = C.RUNS / 'TRACKF' / P
        if (d / 'WITNESS' / 'WITNESS.json').is_file():
            specs.append({'partition': P, 'kind': 'witness', 'id': json.loads((d / 'WITNESS' / 'WITNESS.json').read_text()).get('recipe', 'A1_PO_16')})
        for dd in sorted((d / 'DIR').iterdir()) if (d / 'DIR').is_dir() else []:
            if (dd / 'DIR.json').is_file():
                specs.append({'partition': P, 'kind': 'direction', 'id': dd.name})
        for dd in sorted((d / 'COMP').iterdir()) if (d / 'COMP').is_dir() else []:
            if (dd / 'COMP.json').is_file():
                specs.append({'partition': P, 'kind': 'compression', 'id': dd.name})
        for f in sorted((d / 'NESTED').glob('NESTED_*.json')) if (d / 'NESTED').is_dir() else []:
            specs.append({'partition': P, 'kind': 'nested_correction', 'id': f.stem})
    for P in sorted(p.name for p in (C.RUNS / 'TRACK_R').iterdir() if p.is_dir()) if (C.RUNS / 'TRACK_R').is_dir() else []:
        if (C.RUNS / 'TRACK_R' / P / 'INCOME' / 'INCOME.json').is_file():
            specs.append({'partition': P, 'kind': 'track_r_income_lineage', 'id': 'H3R+HCR'})
    return specs


def main():
    es = Q.entries()
    ends = [e for e in es if e.get('event') == 'unit_end']
    starts = {e['unit_id']: e for e in es if e.get('event') == 'unit_start'}
    now = dt.datetime.now(dt.timezone.utc)
    by_stage, by_prefix = Counter(), Counter()
    stats = Counter()
    for e in ends:
        by_stage[e['stage']] += float(e['charged_cpu_s'])
        by_prefix['_'.join(e['unit_id'].split('_')[:2])] += float(e['charged_cpu_s'])
        for k, v in (e.get('unit_stats') or {}).items():
            if isinstance(v, (int, float)):
                stats[k] += v
    specs = fitted_specs()
    ps = subprocess.run(['ps', '-axo', 'pid,command'], capture_output=True, text=True).stdout.splitlines()
    mine = [l.strip()[:120] for l in ps if ('cts.' in l or 'cts/' in l or 'roles/verifier/code' in l) and 'closeout' not in l
            and 'ps -axo' not in l]
    res = {'schema': 'CTS1-resource-accounting-v1', 'utc': now.isoformat(timespec='seconds'), 'start_utc': Q.CAMPAIGN_START_UTC,
           'elapsed_h': round((now - Q.start_utc()).total_seconds() / 3600, 3), 'research_limit_h': Q.RESEARCH_LIMIT_H,
           'elapsed_limit_h': Q.ELAPSED_LIMIT_H, 'charged_cpu_h': round(Q.charged_cpu_s() / 3600, 3), 'cpu_limit_h': Q.CPU_LIMIT_S / 3600,
           'units_done': sum(e['status'] == 'done' for e in ends), 'units_failed': sum(e['status'] != 'done' for e in ends),
           'units_started_without_end': sorted(set(starts) - {e['unit_id'] for e in ends}),
           'failed_units': [{'unit': e['unit_id'], 'stage': e['stage'], 'charged_cpu_s': e['charged_cpu_s'], 'returncode': e.get('returncode')}
                            for e in ends if e['status'] != 'done'],
           'cpu_s_by_stage': {k: round(v, 1) for k, v in sorted(by_stage.items())},
           'cpu_s_by_unit_prefix': {k: round(v, 1) for k, v in sorted(by_prefix.items())},
           'fitted_release_specifications': {'count': len(specs), 'ceiling': CEILING, 'list': specs,
                                             'counting_rule': 'one per fitted release-producing pipeline per training repetition '
                                                              '(witness, direction, compression code, nested correction, Track R '
                                                              'income lineage); ladder strengths and rule selections of a fitted '
                                                              'direction are recorded selections, not fits'},
           'internal_fit_counters_from_units': dict(stats),
           'counts': {'training_repetitions': ['D0', 'D1', 'R0', 'R1'], 'sensitivity_alias': 'D0B (same households as D0)',
                      'reader_household_repetitions': ['R1', 'R2'], 'reader_seeds': ['k0', 'k1'],
                      'people': {'MODEL_FIT': 20000, 'CALIBRATION': 2982, 'MODEL_SELECT': 8001}},
           'agent_charges': [{k: e.get(k) for k in ('label', 'charged_cpu_s', 'note')} for e in es if e.get('event') == 'charge'],
           'heavy_process_limit': Q.SLOTS, 'threads_per_job': 1, 'cloud_or_paid_compute_usd': 0, 'processes_left_running': mine}
    C.write_json(C.RESULTS / 'RESOURCE_ACCOUNTING.json', res)

    acc = C.access_events()
    freeze_first, freeze_last = freeze_commit_utc(True), freeze_commit_utc(False)
    freeze_c = freeze_first.isoformat() if freeze_first else None
    by = Counter((e.get('operation'), e.get('role'), ','.join(e.get('fields') or []), e.get('entry_point', '').split(':')[0]) for e in acc)
    panel = [e for e in acc if e.get('touches_repair_panel')]
    panel_summary = [{'utc': e['utc'], 'operation': e['operation'], 'entry_point': e.get('entry_point'), 'partition': e.get('partition'),
                      'unit_id': e.get('unit_id')} for e in panel]
    lineage_now = C.sha(SB.SRC_ACCESS_LOG) if Path(SB.SRC_ACCESS_LOG).is_file() else None
    rec = {'schema': 'CTS1-access-receipt-v1', 'utc': now.isoformat(timespec='seconds'), 'label_reads': len(acc),
           'by_operation_role_fields_entry': [{'operation': k[0], 'role': k[1], 'fields': k[2], 'entry_point': k[3], 'reads': v}
                                              for k, v in sorted(by.items(), key=lambda kv: tuple(map(str, kv[0])))],
           'repair_panel_reads': panel_summary, 'repair_freeze_committed_utc': freeze_c,
           'repair_freeze_last_commit_utc': freeze_last.isoformat() if freeze_last else None,
           'repair_panel_reads_before_freeze_by_new_release_operations': panel_reads_before_freeze(panel_summary, freeze_first),
           'ATTACK_or_DEV_REPORT_label_reads': [e for e in acc if e.get('role') in C.CLOSED_LABEL_ROLES],
           'lineage_dev_report_log': {'campaign_start_snapshot_sha256': LINEAGE_SNAPSHOT, 'now_sha256': lineage_now,
                                      'unchanged': lineage_now == LINEAGE_SNAPSHOT},
           'never_read': ['ATTACK_FIT labels', 'ATTACK_VALIDATION labels', 'DEV_REPORT labels', 'sealed assessment', 'capped-out records']}
    C.write_json(C.RESULTS / 'ACCESS_RECEIPT.json', rec)
    print(json.dumps({'elapsed_h': res['elapsed_h'], 'charged_cpu_h': res['charged_cpu_h'], 'units_done': res['units_done'],
                      'units_failed': res['units_failed'], 'fitted_specs': len(specs), 'label_reads': len(acc),
                      'lineage_unchanged': rec['lineage_dev_report_log']['unchanged'], 'processes_left': mine}))


if __name__ == '__main__':
    main()
