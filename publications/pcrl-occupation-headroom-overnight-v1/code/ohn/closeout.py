"""Closeout receipts of the overnight occupation-headroom campaign (adapted from ihs.closeout; no label is read).

    python -m ohn.closeout

-> results/<campaign>/RESOURCE_ACCOUNTING.json  elapsed time, queue CPU by stage and unit prefix, units done / failed, configurations
   by budget class against the registered ceilings, model / calibrator / cross-fit fit counts, development reader fits and cache
   hits, agent charges, processes left running;
-> results/<campaign>/ACCESS_RECEIPT.json        MODEL_SELECT label reads by entry point (the registered paths and every other unit,
   with purposes), training-label reads by role and field, ATTACK SEX reads, DEV_REPORT events, the lineage DEV log against its
   admission snapshot, what was never read;
-> results/<campaign>/HANDOFF.json              branch, base, publication branch, commits, lock hashes, decision, resources, custody,
   open items.
Local paths are reduced to module or script names before anything is written.
"""
from __future__ import annotations
import datetime as dt
import json
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path

from . import common as C
from . import jobqueue as Q
from . import ledger as L

CEILINGS = {'OCC36': 36, 'PRIV24': 24, 'INC8': 8, 'CTRL12': 12, 'TOTAL': 80}
BASE = '2af3c9db2d65e052dbd9d3d6a040227e04d09854'


def _git(*a):
    return subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True).stdout.strip()


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).is_file() else []


def _entry_point(argv):
    """Module or script name only (no local path)."""
    argv = list(argv or [])
    if '-m' in argv:
        return argv[argv.index('-m') + 1]
    scripts = [a for a in argv if a.endswith('.py')]
    return Path(scripts[0]).name if scripts else (Path(argv[0]).name if argv else '')


def _class(ep):
    if ep == 'ohn.devrun':
        return 'registered: ohn.devrun score / references'
    if ep == 'ohn.devreaders':
        return 'registered: ohn.devreaders (development reader grid)'
    if ep.startswith('v_'):
        return 'independent verifier (separate code; PROTOCOL section 7)'
    if ep.startswith('rr_'):
        return 'reader-robustness diagnostic (never selection)'
    if ep == 'ohn.decompose':
        return 'fallback cost decomposition (PROTOCOL section 9)'
    return 'other'


def main():
    es = Q.entries()
    ends = [e for e in es if e.get('event') == 'unit_end']
    starts = {e['unit_id']: e for e in es if e.get('event') == 'unit_start'}
    start = Q.start_utc(); now = dt.datetime.now(dt.timezone.utc)
    by_stage, by_prefix = Counter(), Counter()
    for e in ends:
        by_stage[e['stage']] += float(e['charged_cpu_s'])
        by_prefix['_'.join(e['unit_id'].split('_')[:2])] += float(e['charged_cpu_s'])
    rows = L.rows()
    cfg = OrderedDict()
    for r in rows:
        if r['kind'] in ('reader_grid', 'reference', 'decision') or r['budget_class'] in ('', 'NONE'):
            continue
        cfg[(r['family'], r['config_id'], r['seed'])] = r                  # last row per configuration-seed
    by_class = Counter(r['budget_class'] for r in cfg.values())
    fits = {k: sum(int(r.get(k) or 0) for r in cfg.values()) for k in ('n_model_fits', 'n_calibrator_fits', 'n_crossfit_fits')}
    fam_fits = Counter()
    for (f, _, _), r in cfg.items():
        fam_fits[f] += int(r.get('n_model_fits') or 0)
    grid_rows = [r for r in rows if r['kind'] == 'reader_grid']
    ps = subprocess.run(['ps', '-axo', 'pid,command'], capture_output=True, text=True).stdout.splitlines()
    mine = [l.strip()[:120] for l in ps if ('ohn.' in l or 'ohn/' in l) and 'closeout' not in l and 'ps -axo' not in l]
    res = {'schema': 'OHN1-resource-accounting-v1', 'utc': now.isoformat(timespec='seconds'), 'start_utc': start.isoformat(),
           'elapsed_h': round((now - start).total_seconds() / 3600, 3), 'research_limit_h': Q.RESEARCH_LIMIT_H,
           'elapsed_limit_h': Q.ELAPSED_LIMIT_H, 'charged_cpu_h': round(Q.charged_cpu_s() / 3600, 3), 'cpu_limit_h': Q.CPU_LIMIT_S / 3600,
           'units_done': sum(e['status'] == 'done' for e in ends), 'units_failed': sum(e['status'] != 'done' for e in ends),
           'failed_units': [{'unit': e['unit_id'], 'stage': e['stage'], 'charged_cpu_s': e['charged_cpu_s'], 'returncode': e.get('returncode')}
                            for e in ends if e['status'] != 'done'],
           'cpu_s_by_stage': {k: round(v, 1) for k, v in sorted(by_stage.items())},
           'cpu_s_by_unit_prefix': {k: round(v, 1) for k, v in sorted(by_prefix.items())},
           'configurations_by_class': dict(by_class), 'configurations_total': sum(by_class.values()), 'ceilings': CEILINGS,
           'configurations': [{'family': f, 'config_id': c, 'seed': int(s), 'class': r['budget_class'], 'kind': r['kind']}
                              for (f, c, s), r in cfg.items()],
           'fits': fits, 'model_fits_by_family': dict(fam_fits),
           'development_reader_fits_requested': sum(int(r.get('reader_fits') or 0) for r in grid_rows),
           'development_reader_cache_hits': sum(int(r.get('cache_hits') or 0) for r in grid_rows),
           'development_reader_note': 'grid ledger rows: requested reader fits and cache hits (identical views are read from the cache); '
                                      'diagnostic probes, learning-curve fixed-recipe reads and verifier refits are counted in their units',
           'agent_charges': [{k: e.get(k) for k in ('label', 'charged_cpu_s', 'note')} for e in es if e.get('event') == 'charge'],
           'heavy_process_limit': Q.SLOTS, 'threads_per_job': 1, 'cloud_or_paid_compute_usd': 0, 'processes_left_running': mine}
    C.write_json(C.RESULTS / 'RESOURCE_ACCOUNTING.json', res)

    ms = _jsonl(C.MS_LOG); train = _jsonl(C.TRAIN_LOG); lab = _jsonl(C.LABEL_LOG); dev = C.dev_access_events()
    ms_units = OrderedDict()
    for e in ms:
        u = e.get('unit_id') or ''
        ep = _entry_point((starts.get(u) or {}).get('argv'))
        d = ms_units.setdefault(u, {'unit_id': u, 'entry_point': ep, 'class': _class(ep), 'reads': 0, 'fields': set(), 'purposes': set()})
        d['reads'] += 1; d['fields'] |= set(e.get('fields') or []); d['purposes'].add(e.get('purpose') or '')
    ms_list = [{**d, 'fields': sorted(d['fields']), 'purposes': sorted(d['purposes'])} for d in ms_units.values()]
    by_cls = Counter()
    for d in ms_list:
        by_cls[d['class']] += d['reads']
    tr = Counter((e.get('role'), ','.join(e.get('fields') or [])) for e in train)
    adm = C.admission().get('access_log_snapshots') or {}
    src_now = _jsonl(C.SRC_ACCESS_LOG)
    snap = adm.get('lineage_access/ACCESS_LOG_snapshot.jsonl') or {}
    lineage_unchanged = bool(snap) and Path(C.SRC_ACCESS_LOG).is_file() and C.sha(C.SRC_ACCESS_LOG) == snap.get('sha256')
    acc = {'schema': 'OHN1-access-receipt-v1', 'utc': now.isoformat(timespec='seconds'),
           'model_select_label_reads': {'count': len(ms), 'first_utc': ms[0]['utc'] if ms else None, 'last_utc': ms[-1]['utc'] if ms else None,
                                        'every_read_has_a_purpose': all(bool(e.get('purpose')) for e in ms),
                                        'reads_by_entry_class': dict(by_cls),
                                        'units_outside_registered_paths': [d for d in ms_list if not d['class'].startswith('registered')],
                                        'registered_path_units': len([d for d in ms_list if d['class'].startswith('registered')])},
           'training_label_reads': [{'role': r, 'fields': f, 'reads': n} for (r, f), n in sorted(tr.items())],
           'training_label_note': ('task-only fits read MODEL_FIT / CALIBRATION task labels only; protection fits read MODEL_FIT occupation '
                                   'and SEX (critics; SEX a training-time label only); the scorer reads MODEL_FIT SEX for its oof_by_sex '
                                   'diagnostic (PROTOCOL_DEVIATIONS item 4)'),
           'attack_sex_reads': [{k: e.get(k) for k in ('utc', 'role', 'purpose', 'unit_id')} for e in lab],
           'dev_report_events': [{k: e.get(k) for k in ('event', 'utc', 'unit_id', 'purpose')} for e in dev],
           'lineage_dev_log': {'lines_now': len(src_now), 'unchanged_since_admission': lineage_unchanged,
                               'note': 'the lineage-wide DEV_REPORT access log equals its admission snapshot byte for byte (SHA-256) iff '
                                       'unchanged_since_admission; no DEV_REPORT truth was read in this campaign',
                               'admission_snapshots': adm},
           'never_read': ['ATTACK_FIT / ATTACK_VALIDATION SEX (no FINALIST_LOCK; the loader refuses them)',
                          'DEV_REPORT truth (no REPORTING_LOCK; the loader refuses it); DEV_REPORT raw inputs and features only, label-free, '
                          'for the deployment checks', 'the sealed assessment and any capped-out population (refused by every loader)',
                          'any new population, download or person-level file outside the admitted Florida 2019 development exports'],
           'network': 'git push of the publication branch to the existing GitHub remote and its read-back only'}
    C.write_json(C.RESULTS / 'ACCESS_RECEIPT.json', acc)

    names = ('PROTOCOL.md', 'DEV_GRID.json', 'DEVGRID_CODE_LOCK.json', 'EXPOSURE_TABLE.md', 'SOURCE_ADMISSION.json',
             'roles/statistical_reviewer/RULES.json', 'roles/statistical_reviewer/SELECTION_RULE.md', 'WITNESS_SELECTION.json',
             'NOMINATION.json', 'FINALIST_LOCK.json', 'REPORTING_LOCK.json')
    locks = {n: C.sha(C.RESULTS / n) for n in names if (C.RESULTS / n).exists()}
    commits = _git('log', '--format=%h %cI %s', f'{BASE}..HEAD').splitlines()
    nom = json.loads((C.RESULTS / 'NOMINATION.json').read_text()) if (C.RESULTS / 'NOMINATION.json').exists() else {}
    hand = {'schema': 'OHN1-handoff-v1', 'utc': now.isoformat(timespec='seconds'), 'branch': 'research/pcrl-occupation-headroom-overnight-v1',
            'base': BASE, 'publication_branch': 'publication/pcrl-occupation-headroom-overnight-v1',
            'nature': 'exploratory development on already-used Florida ACS 2019 development records; not confirmation',
            'decision': {'nomination_outcome': nom.get('outcome'), 'nominated': nom.get('nominated'), 'dev_report': nom.get('dev_report'),
                         'documents': ['START_HERE.md', 'RESEARCH_DECISION.md', 'ADVISOR_BRIEF.md', 'MANUSCRIPT_CLAIMS_HANDOFF.md']},
            'commits': [c[:200] for c in commits], 'locks_sha256': locks,
            'resources': {k: res[k] for k in ('elapsed_h', 'charged_cpu_h', 'units_done', 'units_failed', 'configurations_total',
                                              'cloud_or_paid_compute_usd')},
            'custody': json.loads((C.RESULTS / 'BACKUP_VERIFICATION.json').read_text()) if (C.RESULTS / 'BACKUP_VERIFICATION.json').exists() else None,
            'open_items': ['off-device backup copy pending (no writable external drive attached; command in BACKUP_VERIFICATION.json)',
                           'no final reader evaluation, ATTACK or DEV_REPORT read: no configuration qualified',
                           'receipt write-once-on-failure defect (PROTOCOL_DEVIATIONS item 21) before any successor reporting stage'],
            'processes_left_running': mine}
    C.write_json(C.RESULTS / 'HANDOFF.json', hand)
    print(json.dumps({'elapsed_h': res['elapsed_h'], 'charged_cpu_h': res['charged_cpu_h'], 'units': (res['units_done'], res['units_failed']),
                      'configurations': dict(by_class), 'ms_reads_by_class': dict(by_cls), 'attack_reads': len(lab), 'dev_events': len(dev),
                      'processes_left_running': mine}, indent=1))


if __name__ == '__main__':
    main()
