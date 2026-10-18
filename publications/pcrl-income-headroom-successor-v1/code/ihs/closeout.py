"""Closeout receipts: RESOURCE_ACCOUNTING.json (queue totals by stage and family, configuration and individual fit counts, reader
fits, failed attempts, agent charges, processes left running), ACCESS_RECEIPT.json (MODEL_SELECT uses, ATTACK SEX reads,
DEV_REPORT follow-up events, what was never read) and HANDOFF.json (branch, commits, locks, decision, custody, open items).
No label is read."""
from __future__ import annotations
import datetime as dt
import json
import subprocess

from . import common as C
from . import jobqueue as Q
from . import ledger as L


def _git(*a):
    return subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True).stdout.strip()


def _jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def main():
    es = Q.entries(); ends = [e for e in es if e.get('event') == 'unit_end']
    start = Q.start_utc(); now = dt.datetime.now(dt.timezone.utc)
    by_stage, by_prefix = {}, {}
    for e in ends:
        by_stage[e['stage']] = round(by_stage.get(e['stage'], 0.0) + float(e['charged_cpu_s']), 1)
        pre = '_'.join(e['unit_id'].split('_')[:2])
        by_prefix[pre] = round(by_prefix.get(pre, 0.0) + float(e['charged_cpu_s']), 1)
    rows = L.rows()
    fits = [r for r in rows if r['counts_toward_36'] == '1']
    cfg_counts = {}
    for r in fits:
        cfg_counts.setdefault(r['family'], set()).add((r['config_id'], r['seed']))
    ps = subprocess.run(['ps', '-axo', 'pid,command'], capture_output=True, text=True).stdout.splitlines()
    mine = [l.strip() for l in ps if ('ihs.' in l or 'ihs/' in l) and 'closeout' not in l and 'ps -axo' not in l]
    res = {'schema': 'IHS1-resource-accounting-v1', 'utc': now.isoformat(timespec='seconds'), 'start_utc': start.isoformat(),
           'elapsed_h': round((now - start).total_seconds() / 3600, 3), 'elapsed_limit_h': Q.ELAPSED_LIMIT_H,
           'charged_cpu_h': round(Q.charged_cpu_s() / 3600, 3), 'cpu_limit_h': Q.CPU_LIMIT_S / 3600,
           'units_done': sum(e['status'] == 'done' for e in ends), 'units_failed': sum(e['status'] != 'done' for e in ends),
           'failed_units': [{'unit': e['unit_id'], 'stage': e['stage'], 'charged_cpu_s': e['charged_cpu_s']} for e in ends if e['status'] != 'done'],
           'cpu_s_by_stage': by_stage, 'cpu_s_by_unit_prefix': by_prefix,
           'configurations_counted_toward_36': {f: len(v) for f, v in cfg_counts.items()},
           'configurations_counted_total': sum(len(v) for v in cfg_counts.values()),
           'individual_model_fits_by_family': {f: sum(int(r['n_model_fits'] or 0) for r in fits if r['family'] == f) for f in cfg_counts},
           'development_reader_fits_requested': sum(int(r['reader_fits'] or 0) for r in rows),
           'development_reader_fits_note': 'requested cells x 2 directions, counting cache hits; distinct fits are fewer (identical views are cached)',
           'agent_charges': [{k: e[k] for k in ('label', 'charged_cpu_s', 'note')} for e in es if e.get('event') == 'charge'],
           'units': [{k: e.get(k) for k in ('unit_id', 'stage', 'status', 'charged_cpu_s', 'wall_s', 'utc')} for e in ends],
           'heavy_process_limit': Q.SLOTS, 'threads_per_job': 1, 'cloud_or_paid_compute_usd': 0, 'processes_left_running': mine}
    C.write_json(C.RESULTS / 'RESOURCE_ACCOUNTING.json', res)
    ms = _jsonl(C.MS_LOG); lab = _jsonl(C.LABEL_LOG); dev = C.dev_access_events()
    acc = {'schema': 'IHS1-access-receipt-v1', 'utc': now.isoformat(timespec='seconds'),
           'model_select_label_reads': {'count': len(ms), 'first_utc': ms[0]['utc'] if ms else None, 'last_utc': ms[-1]['utc'] if ms else None,
                                        'every_read_has_a_purpose': all(bool(e.get('purpose')) for e in ms)},
           'attack_sex_reads': [{k: e.get(k) for k in ('utc', 'role', 'purpose', 'unit_id')} for e in lab],
           'dev_report_follow_up_events': [{k: e.get(k) for k in ('event', 'utc', 'unit_id', 'purpose', 'reporting_lock_sha256')} for e in dev],
           'dev_report_note': "already-exposed development role; new follow-up event names; appended to this campaign's log and to the lineage-wide log; earlier logs and one-use counters untouched",
           'training_and_calibration_exposure': {
               'H': 'MODEL_FIT income labels (HistGradientBoosting, household-separated 5-fold CV); CALIBRATION income labels (STRONG stack, Platt maps, '
                    'beta budget); no SEX in any new fit (the transplanted LOCAL / JOINT_TREE income corrections were trained with MODEL_FIT SEX by the replication)',
               'P': 'MODEL_FIT income labels and SEX (online critics; SEX a training-time label only); CALIBRATION income labels for the optional '
                    '2-parameter Platt map (off in P1); label-free distortion budget on MODEL_FIT people (projection toward the teacher); no MODEL_SELECT fitting',
               'O': 'CALIBRATION occupation labels (temperature; OC_TBCAL bias); MODEL_FIT occupation labels and SEX (O1 bias, SEX a training-time label only; OC_BFIT without SEX)',
               'CMB': 'no fit (combinations of stored releases)',
               'MODEL_SELECT': 'labels and SEX only through ihs.references and ihs.devrun score (every read logged); guided adaptive choices and revisions '
                               '(budgets, selection rules, the next configuration); never a gradient-fitting or calibration role for any release parameter',
               'DEV_REPORT': 'lineage events before this campaign: 12; this campaign: one coordinator evaluation and one verifier replay'},
           'never_read': ['the sealed assessment (refused by every loader)', 'any new population, download or person-level file outside the admitted Florida 2019 development exports',
                          'ATTACK_FIT / ATTACK_VALIDATION SEX before a committed FINALIST_LOCK (refused by the loader)',
                          'DEV_REPORT truth outside the two registered follow-up events'],
           'network': 'git push of the publication branch to the existing GitHub remote only'}
    C.write_json(C.RESULTS / 'ACCESS_RECEIPT.json', acc)
    names = ('PROTOCOL.md', 'SOURCE_ADMISSION.json', 'DEV_EVALUATOR_FINGERPRINT.json', 'NOMINATION.json', 'FINALIST_LOCK.json',
             'ANALYSIS_LOCK.json', 'SERVICE_BANK_MANIFEST.json', 'READER_PLAN.json', 'REPORTING_LOCK.json')
    locks = {n: C.sha(C.RESULTS / n) for n in names if (C.RESULTS / n).exists()}
    commits = _git('log', '--format=%H %cI %s', f'{C.FFV_TIP}..HEAD').splitlines()
    hand = {'schema': 'IHS1-handoff-v1', 'utc': now.isoformat(timespec='seconds'), 'branch': 'research/pcrl-income-headroom-successor-v1',
            'base': C.FFV_TIP, 'publication_branch': 'publication/pcrl-income-headroom-successor-v1',
            'nature': 'exploratory development on already-used Florida ACS 2019 development records; not confirmation',
            'commits': [c[:160] for c in commits], 'locks_sha256': locks,
            'decision': json.loads((C.RESULTS / 'HANDOFF_DECISION.json').read_text()) if (C.RESULTS / 'HANDOFF_DECISION.json').exists() else None,
            'resources': {k: res[k] for k in ('elapsed_h', 'charged_cpu_h', 'units_done', 'units_failed', 'configurations_counted_total', 'cloud_or_paid_compute_usd')},
            'custody': json.loads((C.RESULTS / 'BACKUP_VERIFICATION.json').read_text()) if (C.RESULTS / 'BACKUP_VERIFICATION.json').exists() else None,
            'processes_left_running': mine}
    C.write_json(C.RESULTS / 'HANDOFF.json', hand)
    print(json.dumps({'elapsed_h': res['elapsed_h'], 'charged_cpu_h': res['charged_cpu_h'], 'units': (res['units_done'], res['units_failed']),
                      'configurations': res['configurations_counted_total'], 'dev_events': [e['event'] for e in dev], 'processes_left_running': mine}, indent=1))


if __name__ == '__main__':
    main()
