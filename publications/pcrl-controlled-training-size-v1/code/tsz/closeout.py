"""Closeout receipts of the PCRL controlled training-size campaign (no label is read).

    python3 -m tsz.closeout

-> results/<campaign>/RESOURCE_ACCOUNTING.json  elapsed time, queue CPU by stage and unit prefix, units done / failed (all attempts
   kept), agent charges; actual fits, configurations, fitting people and households, aliases and cache hits counted separately;
   processes left running (campaign-owned only);
-> results/<campaign>/ACCESS_RECEIPT.json        every label read by operation, entry point, role, field and subset (from the single
   access log), reads before the registration receipt, closed-role reads, the lineage DEV_REPORT log against its campaign-start
   snapshot, what was never read.
Local paths are reduced to module or script names.
"""
from __future__ import annotations
import datetime as dt
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from scm_bench import common as SB
from . import common as C
from . import jobqueue as Q

LINEAGE_SNAPSHOT = 'c7829e99e6af303cdf070f698b6088e024f3c2b806974a6b1f711785bb50c567'     # 00:12:45Z (campaign start; unchanged since CTS1)


def _json(p):
    return json.loads(Path(p).read_text())


def fits():
    """Actual fitted pipelines by kind, with their training people / households and aliases."""
    out = {'size_A1': [], 'size_A1_encoding': [], 'control': [], 'correction_base': [], 'correction_direction': [], 'nested': []}
    for d in sorted((C.RUNS / 'SIZE').iterdir()) if (C.RUNS / 'SIZE').is_dir() else []:
        if (d / 'FIT.json').is_file():
            r = _json(d / 'FIT.json')
            row = {'id': d.name, 'subset': r.get('subset'), 'version': r.get('version'), 'people': r.get('n_people'),
                   'households': r.get('n_households'), 'T_star': r.get('T_star'),
                   'aliased_to_orderings': list(C.ORDERINGS) if r.get('subset') == 'S100' else None}
            out['size_A1_encoding' if d.name.endswith('__ADM') else 'size_A1'].append(row)
    for d in sorted((C.RUNS / 'CONTROL').iterdir()) if (C.RUNS / 'CONTROL').is_dir() else []:
        recs = sorted(p.name for p in d.glob('*.json'))
        out['control'].append({'id': d.name, 'records': recs})
    for d in sorted((C.RUNS / 'CORRECT').iterdir()) if (C.RUNS / 'CORRECT').is_dir() else []:
        for p in sorted(d.rglob('*.json')):
            name = p.stem.upper()
            if 'BASE' in name:
                out['correction_base'].append({'version': d.name, 'record': p.name})
            elif 'DIR' in name or 'DIRECTION' in name:
                out['correction_direction'].append({'version': d.name, 'record': p.name})
    for d in sorted((C.RUNS / 'NESTED').iterdir()) if (C.RUNS / 'NESTED').is_dir() else []:
        out['nested'].append({'version': d.name, 'records': sorted(p.name for p in d.glob('*.json'))})
    return out


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
    f = fits()
    ps = subprocess.run(['ps', '-axo', 'pid,command'], capture_output=True, text=True).stdout.splitlines()
    owned = re.compile(r'(-m tsz\.|/tsz/|pcrl_controlled_training_size_v1/roles/verifier/code)')   # campaign modules only
    mine = [l.strip()[:120] for l in ps if owned.search(l) and 'closeout' not in l and 'ps -axo' not in l]
    res = {'schema': 'TSZ1-resource-accounting-v1', 'utc': now.isoformat(timespec='seconds'), 'start_utc': Q.CAMPAIGN_START_UTC,
           'elapsed_h': round((now - Q.start_utc()).total_seconds() / 3600, 3), 'research_limit_h': Q.RESEARCH_LIMIT_H,
           'elapsed_limit_h': Q.ELAPSED_LIMIT_H, 'charged_cpu_h': round(Q.charged_cpu_s() / 3600, 3), 'cpu_limit_h': Q.CPU_LIMIT_S / 3600,
           'units_done': sum(e['status'] == 'done' for e in ends), 'units_failed': sum(e['status'] != 'done' for e in ends),
           'units_started_without_end': sorted(set(starts) - {e['unit_id'] for e in ends}),
           'failed_units': [{'unit': e['unit_id'], 'stage': e['stage'], 'charged_cpu_s': e['charged_cpu_s'], 'returncode': e.get('returncode')}
                            for e in ends if e['status'] != 'done'],
           'cpu_s_by_stage': {k: round(v, 1) for k, v in sorted(by_stage.items())},
           'cpu_s_by_unit_prefix': {k: round(v, 1) for k, v in sorted(by_prefix.items())},
           'fits': f,
           'counts': {
               'actual_fitted_pipelines': {k: len(v) for k, v in f.items()},
               'configurations': {'task recipe': 'A1_PO_16 (one)', 'encoding sensitivity': 'A1_PO_16 + admitted dictionary (one)',
                                  'control': 'T_TREE -> TASK_REFIT (one)', 'correction arms': 'CONT, LOC, PAIR of the source P2 recipe'},
               'aliases': {'S100': 'one fit per version serves all four orderings (no duplicate fit)',
                           'identity selections': 'a selected strength 0 equals W100_v exactly (no fit)'},
               'internal_fit_counters_from_units': dict(stats),
               'training_subsets': list(C.subset_ids()), 'versions': list(C.VERSIONS),
               'reader_household_repetitions': ['R1', 'R2'], 'reader_seeds': ['k0', 'k1'],
               'people': {'MODEL_FIT': 20000, 'CALIBRATION': 2982, 'MODEL_SELECT': 8001},
               'households': {'MODEL_FIT': 13337, 'CALIBRATION': 1974, 'MODEL_SELECT': 5306}},
           'agent_charges': [{k: e.get(k) for k in ('label', 'charged_cpu_s', 'note')} for e in es if e.get('event') == 'charge'],
           'heavy_process_limit': Q.SLOTS, 'threads_per_job': 1, 'cloud_or_paid_compute_usd': 0, 'processes_left_running': mine}
    C.write_json(C.RESULTS / 'RESOURCE_ACCOUNTING.json', res)

    acc = C.access_events()
    reg = _json(C.REG_RECEIPT) if C.REG_RECEIPT.is_file() else {}
    reg_utc = reg.get('verified_utc')
    before = [e for e in acc if reg_utc and dt.datetime.fromisoformat(e['utc']) < dt.datetime.fromisoformat(reg_utc)]
    by = Counter((e.get('operation'), e.get('role'), ','.join(e.get('fields') or []), e.get('entry_point', '').split(':')[0],
                  e.get('subset') or '') for e in acc)
    lineage_now = C.sha(SB.SRC_ACCESS_LOG) if Path(SB.SRC_ACCESS_LOG).is_file() else None
    fit_ms = [e for e in acc if e.get('role') == 'MODEL_SELECT' and e.get('operation') not in ('benchmark', 'readers', 'verify')]
    rec = {'schema': 'TSZ1-access-receipt-v1', 'utc': now.isoformat(timespec='seconds'), 'label_reads': len(acc),
           'registration_receipt_verified_utc': reg_utc, 'label_reads_before_registration_receipt': len(before),
           'by_operation_role_fields_entry_subset': [{'operation': k[0], 'role': k[1], 'fields': k[2], 'entry_point': k[3], 'subset': k[4],
                                                      'reads': v} for k, v in sorted(by.items(), key=lambda kv: tuple(map(str, kv[0])))],
           'MODEL_SELECT_reads_outside_benchmark_readers_verify': fit_ms,
           'ATTACK_or_DEV_REPORT_label_reads': [e for e in acc if e.get('role') in C.CLOSED_LABEL_ROLES],
           'lineage_dev_report_log': {'campaign_start_snapshot_sha256': LINEAGE_SNAPSHOT, 'now_sha256': lineage_now,
                                      'unchanged': lineage_now == LINEAGE_SNAPSHOT},
           'never_read': ['ATTACK_FIT labels', 'ATTACK_VALIDATION labels', 'DEV_REPORT labels', 'CALIBRATION SEX (not admitted)',
                          'sealed assessment', 'capped-out records']}
    C.write_json(C.RESULTS / 'ACCESS_RECEIPT.json', rec)
    print(json.dumps({'elapsed_h': res['elapsed_h'], 'charged_cpu_h': res['charged_cpu_h'], 'units_done': res['units_done'],
                      'units_failed': res['units_failed'], 'fits': res['counts']['actual_fitted_pipelines'], 'label_reads': len(acc),
                      'reads_before_receipt': len(before), 'lineage_unchanged': rec['lineage_dev_report_log']['unchanged'],
                      'processes_left': mine}))


if __name__ == '__main__':
    main()
