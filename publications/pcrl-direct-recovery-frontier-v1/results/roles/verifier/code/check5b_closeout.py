"""Verifier check 5b (DRF1; label-free, read-only): closeout cross-checks.
  (a) each reading's plan and each robustness bank spec committed before the first MODEL_SELECT read of its own units;
  (b) ACCESS_RECEIPT.json recounted from the access log; (c) RESOURCE_ACCOUNTING.json recounted from the queue ledger and the
  configuration ledger; (d) custody: BACKUP_VERIFICATION.json and OFFDEVICE_COPY.json against the RESTORE_CHECK.json, TIP_COMMIT and
  MANIFEST.sha256 they summarize (the drive is only read; nothing is written, mounted or unmounted).

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check5b_closeout.py --out FILE
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

CAMPAIGN = 'pcrl_direct_recovery_frontier_v1'
RES = REPO / 'results' / CAMPAIGN
PRIV = REPO / '.private' / CAMPAIGN
BACKUP = REPO.parent / 'pcrl-direct-recovery-frontier-v1-backup' / 'pcrl_direct_recovery_frontier_v1_20261025T051138Z'
DRIVE = Path('<external drive>)
CLOSED = {'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'CALIBRATION'}


def t(s):
    return dt.datetime.fromisoformat(str(s).replace('Z', '+00:00')).astimezone(dt.timezone.utc)


def first_commit(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'log', '--diff-filter=A', '--format=%cI', '--', rel], capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    return t(lines[-1]) if lines else None


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    items, info = [], {'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    access = jl(PRIV / 'access' / 'LABEL_ACCESS.jsonl')
    queue = jl(PRIV / 'compute' / 'ledger.jsonl')
    cfg = jl(PRIV / 'runs' / 'CONFIG_LEDGER.jsonl')
    ms = [e for e in access if e['role'] == 'MODEL_SELECT']
    # (a) plans and robustness bank specs before their own readings' first MODEL_SELECT read
    for rel, pred in ((f'results/{CAMPAIGN}/claims/PLAN_DRF_R2.json', lambda u: 'DRF_R2' in u and 'ROB' not in u),
                      (f'results/{CAMPAIGN}/grid_specs/DRF1_R1_ROB.json', lambda u: 'DRF1_R1_ROB' in u),
                      (f'results/{CAMPAIGN}/grid_specs/DRF1_R2_ROB.json', lambda u: 'DRF1_R2_ROB' in u)):
        when = first_commit(rel)
        reads = sorted(t(e['utc']) for e in ms if pred(str(e.get('unit_id'))))
        info.setdefault('ordering', {})[rel] = {'first_commit_utc': when.isoformat() if when else None,
                                                'first_own_model_select_read_utc': reads[0].isoformat() if reads else None,
                                                'own_model_select_reads': len(reads)}
        items.append(V.cmp_eq(f'a:{Path(rel).name}:committed_before_its_first_MODEL_SELECT_read',
                              bool(when and reads and when < reads[0]), True))
    # (b) the access receipt
    ar = json.loads((RES / 'ACCESS_RECEIPT.json').read_text())
    rec_utc = t(ar['registration_receipt_verified_utc'])
    items.append(V.cmp_eq('b:label_reads', len(access), ar.get('label_reads')))
    items.append(V.cmp_eq('b:label_reads_before_registration_receipt', sum(t(e['utc']) < rec_utc for e in access),
                          ar.get('label_reads_before_registration_receipt')))
    items.append(V.cmp_eq('b:ATTACK_or_DEV_REPORT_label_reads', len([e for e in access if e['role'] in CLOSED - {'CALIBRATION'}]),
                          len(ar.get('ATTACK_or_DEV_REPORT_label_reads') or [])))
    items.append(V.cmp_eq('b:CALIBRATION_label_reads', len([e for e in access if e['role'] == 'CALIBRATION']),
                          len(ar.get('CALIBRATION_label_reads') or [])))
    items.append(V.cmp_eq('b:MODEL_SELECT_reads_outside_benchmark_readers_verify',
                          len([e for e in ms if e['operation'] not in ('benchmark', 'readers', 'verify')]),
                          len(ar.get('MODEL_SELECT_reads_outside_benchmark_readers_verify') or [])))
    late = []
    for e in access:
        if e['operation'] == 'fit':
            w = first_commit(e['card'])
            if not (w and w < t(e['utc'])):
                late.append(e['utc'])
    items.append(V.cmp_eq('b:fitting_reads_with_card_committed_after_the_read', len(late),
                          len(ar.get('fitting_reads_with_card_committed_after_the_read') or [])))
    # (c) the resource accounting (as of its own utc)
    ra = json.loads((RES / 'RESOURCE_ACCOUNTING.json').read_text())
    cut = t(ra['utc'])
    ends = [e for e in queue if e.get('event') == 'unit_end']
    starts = [e for e in queue if e.get('event') == 'unit_start']
    items.append(V.cmp_eq('c:units_done', sum(e['status'] == 'done' for e in ends), ra.get('units_done')))
    items.append(V.cmp_eq('c:units_failed', sum(e['status'] != 'done' for e in ends), ra.get('units_failed')))
    items.append(V.cmp_eq('c:units_started_without_end', sorted({e['unit_id'] for e in starts} - {e['unit_id'] for e in ends}),
                          sorted(ra.get('units_started_without_end') or [])))
    cpu = sum(float(e.get('charged_cpu_s', 0)) for e in queue if e.get('event') in ('unit_end', 'charge') and
              t(e.get('utc')) <= cut)
    items.append(V.cmp_num('c:charged_cpu_h_to_its_utc', round(cpu / 3600, 3), ra.get('charged_cpu_h'), 0.0005))
    items.append(V.cmp_num('c:elapsed_h_to_its_utc', round((cut - t(ra['start_utc'])).total_seconds() / 3600, 3), ra.get('elapsed_h'), 0.0015))
    n_cfg = len({(e['family'], e['config_id'], int(e['version'])) for e in cfg})
    info['configurations_recounted'] = n_cfg
    items.append(V.cmp_eq('c:configurations_at_most_32', n_cfg <= 32, True))
    closed = [e for e in queue if e.get('event') == 'queue_closed']
    info['queue_closed_utc'] = closed[0]['utc'] if closed else None
    items.append(V.cmp_eq('c:queue_closed_recorded', bool(closed), True))
    items.append(V.cmp_eq('c:no_unit_start_after_close', [e['unit_id'] for e in starts if closed and t(e['utc']) > t(closed[0]['utc'])], []))
    # (d) custody summaries against the files they summarize (read-only)
    for name, rec_path, root, key in (('backup', RES / 'custody' / 'BACKUP_VERIFICATION.json', BACKUP, 'restore_check_from_backup'),
                                      ('drive_copy', RES / 'custody' / 'OFFDEVICE_COPY.json', DRIVE, 'restore_check_from_copy')):
        rec = json.loads(rec_path.read_text())
        rc = json.loads((root / 'RESTORE_CHECK.json').read_text())
        items.append(V.cmp_eq(f'd:{name}:all_ok_record_equals_restore_check', rec[key].get('all_ok'), rc.get('all_ok')))
        items.append(V.cmp_eq(f'd:{name}:restore_check_all_ok', rc.get('all_ok'), True))
        for k in ('models_rerun_bitwise', 'models_never_rerun', 'readers_never_bitwise'):
            items.append(V.cmp_eq(f'd:{name}:{k}', rec[key].get(k), rc.get(k)))
        tip = (root / 'TIP_COMMIT').read_text().strip()
        items.append(V.cmp_eq(f'd:{name}:tip_commit', tip, rec.get('tip_commit')))
        man = {}
        for line in (root / 'MANIFEST.sha256').read_text().splitlines():
            if line.strip():
                h, f = line.split(None, 1)
                man[f.strip().lstrip('*')] = h
        for f, h in man.items():
            items.append(V.cmp_eq(f'd:{name}:manifest_rehash:{f}', V.sha256_file(root / f), h))
        size = sum((root / f).stat().st_size for f in man)
        info.setdefault('custody', {})[name] = {'files': sorted(man), 'size_bytes_manifest_files': size, 'record_size_bytes': rec.get('size_bytes')}
    # (e) corrected custody comparison (added 05:58Z after the first run): RESTORE_CHECK.json nests the restore detail under
    # 'restore_check_from_backup' (group d compared three of its keys at the top level and failed; that run is kept as recorded)
    for name, rec_path, root, key in (('backup', RES / 'custody' / 'BACKUP_VERIFICATION.json', BACKUP, 'restore_check_from_backup'),
                                      ('drive_copy', RES / 'custody' / 'OFFDEVICE_COPY.json', DRIVE, 'restore_check_from_copy')):
        rec = json.loads(rec_path.read_text())
        rc = json.loads((root / 'RESTORE_CHECK.json').read_text())
        items.append(V.cmp_eq(f'e:{name}:nested_restore_check_equal', rec[key], rc.get('restore_check_from_backup')))
        for k in ('tip_commit', 'manifest_rehash_ok', 'bundle_verify', 'checked_utc', 'read_back', 'all_ok'):
            items.append(V.cmp_eq(f'e:{name}:{k}', rec.get(k), rc.get(k)))
    # (f) (added 05:58Z after group e) the summaries are abridged: every field a summary carries must equal the RESTORE_CHECK.json
    # field at the same path (fields only in the file are listed, not failures)
    def subset_diffs(a, b, path=''):
        bad, extra = [], []
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in b:
                    bad.append(f'{path}/{k}: only in the summary')
                elif k not in a:
                    extra.append(f'{path}/{k}')
                else:
                    x, y = subset_diffs(a[k], b[k], f'{path}/{k}')
                    bad += x
                    extra += y
        elif a != b:
            bad.append(f'{path}: summary {json.dumps(a)[:80]} vs file {json.dumps(b)[:80]}')
        return bad, extra
    for name, rec_path, root, key in (('backup', RES / 'custody' / 'BACKUP_VERIFICATION.json', BACKUP, 'restore_check_from_backup'),
                                      ('drive_copy', RES / 'custody' / 'OFFDEVICE_COPY.json', DRIVE, 'restore_check_from_copy')):
        rec = json.loads(rec_path.read_text())
        rc = json.loads((root / 'RESTORE_CHECK.json').read_text())
        bad, extra = subset_diffs(rec[key], rc.get('restore_check_from_backup'))
        items.append(V.cmp_eq(f'f:{name}:every_summarized_field_equals_the_file', bad, []))
        info.setdefault('custody_fields_only_in_file', {})[name] = extra
    bv = json.loads((RES / 'custody' / 'BACKUP_VERIFICATION.json').read_text())
    man_b = {}
    for line in (BACKUP / 'MANIFEST.sha256').read_text().splitlines():
        if line.strip():
            h, f = line.split(None, 1)
            man_b[f.strip().lstrip('*')] = h
    items.append(V.cmp_eq('d:backup_record_manifest_sha256_equals_manifest_file', dict(sorted(bv['manifest_sha256'].items())),
                          dict(sorted(man_b.items()))))
    out = {'schema': 'DRF1-verifier-check5b-v1', 'info': info, 'overall': V.summarize(items), 'items': items}
    groups = {}
    for it in items:
        groups.setdefault(it['name'].split(':', 1)[0], []).append(it)
    out['summary'] = {g: V.summarize(x) for g, x in groups.items()}
    if a.out:
        V.dump(a.out, out)
    print(json.dumps(V.jsonable({'overall': out['overall'], 'summary': out['summary'], 'info': info}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
