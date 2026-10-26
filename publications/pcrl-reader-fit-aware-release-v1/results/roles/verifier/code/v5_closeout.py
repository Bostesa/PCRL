#!/usr/bin/env python3
"""RFA1 verifier, wave 5 task 5: closeout cross-checks (label-free, read-only).

RESOURCE_ACCOUNTING.json against the queue ledger; ACCESS_RECEIPT.json against the label access log (and each read against its
queue unit); custody/BACKUP_VERIFICATION.json against the final backup on disk (manifest re-hash, tip, sizes, bundle check output,
code.tar members against the tip's tree blobs, private.tar.gz member list against the private tree); custody/OFFDEVICE_COPY.json
against the drive copy when the drive is mounted (never mounted, unmounted or forced here; when absent, only internal consistency
with the backup). Archive member contents are never parsed; only names, sizes and hashes.

    <repository>/.venv/bin/python <this file> <scratch folder with bundle_verify.txt and tip_ls.txt>
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT))
import vsafe as V  # noqa: E402

RES = ROOT / 'results' / 'pcrl_reader_fit_aware_release_v1'
PRIV = ROOT / '.private' / 'pcrl_reader_fit_aware_release_v1'
BK_DIR = ROOT.parent / 'pcrl-reader-fit-aware-release-v1-backup' / 'pcrl_reader_fit_aware_release_v1_20261025T232718Z'
DRIVE = Path('/Volumes/YOTUO')
OUT = HERE.parent / 'out' / 'V5_CLOSEOUT.json'
FILES = ('TIP_COMMIT', 'branch_since_base.bundle', 'code.tar', 'private.tar.gz')
CODE = ['rfa', 'drf', 'sar', 'ipr', 'ddr', 'fsa', 'frc', 'nrm', 'tsz', 'cts', 'ohn', 'ihs', 'ffv', 'amd', 'scm_bench', 'acs_replication',
        'acs_hybrids', 'task_conditioned_representation', 'task_headroom', 'mechanism_design', 'results/']   # results/<listed campaigns>


def jl(p):
    return json.loads(Path(p).read_text())


def ts(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00'))


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def blob_id(b):
    return hashlib.sha1(b'blob %d\x00' % len(b) + b).hexdigest()


def main():
    t0 = time.process_time()
    scr = Path(sys.argv[1])
    cids = V.compile_ids(V.identifiers(scr / 'git_email.txt', scr / 'git_name.txt'))
    out = {'schema': 'RFA1-verifier-closeout-v5', 'checks': {}}
    chk = out['checks']
    ev = [json.loads(x) for x in (PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines() if x.strip()]
    for e in ev:                                         # the init event carries queue_opened_utc instead of utc
        e.setdefault('utc', e.get('queue_opened_utc'))
    acct = jl(RES / 'RESOURCE_ACCOUNTING.json')
    cut = ts(acct['utc'])
    evc = [e for e in ev if ts(e['utc']) <= cut]
    init = [e for e in ev if e['event'] == 'init'][0]
    starts = {}
    for e in ev:
        if e['event'] == 'unit_start':
            starts.setdefault(e['unit_id'], []).append(e)
    ends = [e for e in evc if e['event'] == 'unit_end']
    charges = [e for e in evc if e['event'] == 'charge']
    closed = [e for e in ev if e['event'] == 'queue_closed']
    # ------------------------------------------------------------------ A. RESOURCE_ACCOUNTING vs ledger
    by_stage, by_prefix, stats = Counter(), Counter(), Counter()
    for e in ends:
        by_stage[e['stage']] += float(e['charged_cpu_s'])
        by_prefix['_'.join(e['unit_id'].split('_')[:2])] += float(e['charged_cpu_s'])
        for k, v in (e.get('unit_stats') or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                stats[k] += v
    tot = sum(float(e['charged_cpu_s']) for e in ends) + sum(float(e['charged_cpu_s']) for e in charges)
    chk['A: one start per unit, every end after its start, no unit started without an end'] = (
        all(len(v) == 1 for v in starts.values()) and all(ts(e['utc']) >= ts(starts[e['unit_id']][0]['utc']) for e in ends)
        and sorted(starts) == sorted(e['unit_id'] for e in ends) and acct['units_started_without_end'] == [])
    chk['A: units done / failed'] = (acct['units_done'] == sum(e['status'] == 'done' for e in ends) == 46
                                     and acct['units_failed'] == sum(e['status'] != 'done' for e in ends) == 2)
    chk['A: failed units list'] = acct['failed_units'] == [{'unit': e['unit_id'], 'stage': e['stage'], 'charged_cpu_s': e['charged_cpu_s'],
                                                           'returncode': e.get('returncode')} for e in ends if e['status'] != 'done']
    chk['A: CPU by stage (1 dp)'] = acct['cpu_s_by_stage'] == {k: round(v, 1) for k, v in sorted(by_stage.items())}
    chk['A: CPU by unit prefix (1 dp)'] = acct['cpu_s_by_unit_prefix'] == {k: round(v, 1) for k, v in sorted(by_prefix.items())}
    chk['A: agent charges equal the ledger charge events'] = acct['agent_charges'] == [{k: e.get(k) for k in ('label', 'charged_cpu_s', 'note', 'utc')}
                                                                                     for e in charges]
    chk['A: charged CPU-h = (unit charges + agent charges) / 3600'] = acct['charged_cpu_h'] == round(tot / 3600, 3)
    chk['A: elapsed h = record time - campaign start'] = (acct['elapsed_h'] == round((cut - ts(acct['start_utc'])).total_seconds() / 3600, 3)
                                                          and acct['start_utc'] == init['start_utc'])
    chk['A: limits as initialized (6 h, 8 CPU-h, 2 slots, 1 thread)'] = (acct['elapsed_limit_h'] == init['elapsed_limit_h'] and
                                                                         acct['cpu_limit_h'] * 3600 == init['cpu_limit_s'] and
                                                                         acct['heavy_process_limit'] == init['slots'] == 2 and acct['threads_per_job'] == 1)
    chk['A: internal fit counters = sums of unit_stats'] = acct['internal_fit_counters_from_units'] == dict(stats)
    chk['A: queue closed at the record time; no unit started after the close'] = (len(closed) == 1 and closed[0]['utc'] == acct['utc']
                                                                                 and not any(e['event'] == 'unit_start' and ts(e['utc']) > ts(closed[0]['utc']) for e in ev))
    post = [e for e in ev if ts(e['utc']) > cut]
    out['A'] = {'ledger_events': len(ev), 'events_after_record': [(e['utc'], e['event']) for e in post], 'charged_cpu_s_total': tot,
                'units': len(ends), 'charges': len(charges)}
    fits = {k: jl(RES / 'fam' / f'FIT_RF_{k}.json') for k in ('V0', 'V1', 'R1_V0', 'R1_V1')}
    saved = sorted((cfg, f['version']) for f in fits.values() for cfg in f['saved'])
    confs = acct['configurations']['configurations']
    kind_of = lambda c: 'candidate' if c.endswith('_FULL') else ('twin' if c.endswith('_TASK') else 'control')  # noqa: E731
    chk['A: 24 configurations = the fit records\' saved configurations; kinds; cards'] = (
        sorted((x['config_id'], x['version']) for x in confs) == saved and len(saved) == 24
        and all(x['kind'] == kind_of(x['config_id']) for x in confs)
        and all(x['card'].endswith('RF_RFIT_R1.md' if x['config_id'].startswith('RF1_') else 'RF_RFIT_v1.md') for x in confs)
        and acct['configurations']['by_family_and_kind'] == {'FAM|candidate': 4, 'FAM|control': 16, 'FAM|twin': 4})
    ps = subprocess.run(['ps', '-axo', 'pid,command'], capture_output=True, text=True).stdout.splitlines()
    owned = re.compile(r'(-m rfa\.|/rfa/|pcrl_reader_fit_aware_release_v1/roles/verifier/code)')
    mine = [x for x in ps if owned.search(x) and 'v5_closeout' not in x and 'ps -axo' not in x]
    chk['A: no campaign process running now (other than this check)'] = not mine and acct['processes_left_running'] == []
    # ------------------------------------------------------------------ B. ACCESS_RECEIPT vs access log
    acc = [json.loads(x) for x in (PRIV / 'access' / 'LABEL_ACCESS.jsonl').read_text().splitlines() if x.strip()]
    ar = jl(RES / 'ACCESS_RECEIPT.json')
    accc = [e for e in acc if ts(e['utc']) <= ts(ar['utc'])]
    reg = jl(PRIV / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json')
    rcpt = jl(RES / 'RECEIPTS.json')['receipts'][0]
    by = Counter((e.get('operation'), e.get('role'), ','.join(e.get('fields') or []), e.get('entry_point', '').split(':')[0], e.get('card') or '')
                 for e in accc)
    mine_by = sorted(({'operation': k[0], 'role': k[1], 'fields': k[2], 'entry_point': k[3], 'card': k[4], 'reads': v} for k, v in by.items()),
                     key=lambda d: tuple(map(str, (d['operation'], d['role'], d['fields'], d['entry_point'], d['card']))))
    rec_by = sorted(ar['by_operation_role_fields_entry_card'], key=lambda d: tuple(map(str, (d['operation'], d['role'], d['fields'], d['entry_point'], d['card']))))
    norm = lambda d: {**d, 'entry_point': re.sub(r'^.*?(results/pcrl_reader_fit_aware_release_v1/)', r'\1', d['entry_point'])}  # noqa: E731
    chk['B: label reads = log entries up to the receipt time (40); none after'] = ar['label_reads'] == len(accc) == len(acc) == 40
    chk['B: reads by operation / role / fields / entry point / card'] = [norm(d) for d in mine_by] == [norm(d) for d in rec_by]
    chk['B: registration receipt time = RECEIPTS.json = the private verification record; zero reads before it'] = (
        ar['registration_receipt_verified_utc'] == reg.get('verified_utc') == rcpt['verified_utc']
        and ar['label_reads_before_registration_receipt'] == sum(ts(e['utc']) < ts(rcpt['verified_utc']) for e in accc) == 0)
    chk['B: no CALIBRATION, ATTACK or DEV_REPORT read; no MODEL_SELECT read outside benchmark / readers / verify'] = (
        ar['CALIBRATION_label_reads'] == [] == [e for e in accc if e['role'] == 'CALIBRATION']
        and ar['ATTACK_or_DEV_REPORT_label_reads'] == [] == [e for e in accc if e['role'] in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')]
        and ar['MODEL_SELECT_reads_outside_benchmark_readers_verify'] == []
        == [e for e in accc if e['role'] == 'MODEL_SELECT' and e['operation'] not in ('benchmark', 'readers', 'verify')])
    late = [e for e in accc if e.get('card_commit_utc') and ts(e['card_commit_utc']) > ts(e['utc'])]
    chk['B: every fitting read under a card committed before the read; card hashes equal the card files'] = (
        ar['fitting_reads_with_card_committed_after_the_read'] == [] == late
        and all(e['card'] and hashlib.sha256((ROOT / e['card']).read_bytes()).hexdigest() == e['card_sha256'] for e in accc if e['operation'] == 'fit'))
    unit_win = {}
    for e in ev:
        if e['event'] == 'unit_start':
            unit_win.setdefault(e['unit_id'], {})['start'] = ts(e['utc'])
            unit_win[e['unit_id']]['stage'] = e['stage']
        elif e['event'] == 'unit_end':
            unit_win.setdefault(e['unit_id'], {})['end'] = ts(e['utc'])
    stage_of_op = {'fit': 'fit', 'readers': 'readers', 'verify': 'verify', 'benchmark': 'evaluate'}
    bad_unit = [e['unit_id'] for e in accc if e.get('unit_id') not in unit_win
                or not (unit_win[e['unit_id']]['start'] <= ts(e['utc']) <= unit_win[e['unit_id']].get('end', cut))
                or unit_win[e['unit_id']]['stage'] != stage_of_op.get(e['operation'], e['operation'])]
    chk['B: every read inside its queue unit\'s start-end window, in a unit of the matching stage'] = not bad_unit
    try:
        from scm_bench import common as SB
        lineage_now = sha256(SB.SRC_ACCESS_LOG) if Path(SB.SRC_ACCESS_LOG).is_file() else None
    except Exception:
        lineage_now = None
    from rfa import closeout as CO
    chk['B: lineage DEV_REPORT log unchanged (its SHA-256 now = the campaign-start snapshot = the record)'] = (
        lineage_now == CO.LINEAGE_SNAPSHOT == ar['lineage_dev_report_log']['campaign_start_snapshot_sha256'] == ar['lineage_dev_report_log']['now_sha256']
        and ar['lineage_dev_report_log']['unchanged'] is True)
    out['B'] = {'log_entries': len(acc), 'roles': dict(Counter(e['role'] for e in acc)), 'operations': dict(Counter(e['operation'] for e in acc)),
                'reads_outside_unit_window_or_stage': bad_unit}
    # ------------------------------------------------------------------ C. the final backup
    bv = jl(RES / 'custody' / 'BACKUP_VERIFICATION.json')
    files = {p.name: p.stat().st_size for p in BK_DIR.iterdir() if p.is_file()}
    man = dict(x.split('  ', 1)[::-1] for x in (BK_DIR / 'MANIFEST.sha256').read_text().splitlines() if x.strip())
    rehash = {f: sha256(BK_DIR / f) for f in FILES}
    chk['C: backup folder holds the four payload files, MANIFEST.sha256 and RESTORE_CHECK.json only'] = sorted(files) == sorted(list(FILES) + ['MANIFEST.sha256', 'RESTORE_CHECK.json'])
    chk['C: manifest = record manifest = fresh re-hash of the four payload files'] = man == bv['manifest_sha256'] == rehash
    chk['C: tip commit (file) = record = 42ef29832'] = ((BK_DIR / 'TIP_COMMIT').read_text().strip() == bv['tip_commit']
                                                      == '42ef29832191a797a44bc10ed5f61073b23d8d63')
    payload = sum(files[f] for f in FILES)
    chk['C: size_bytes = every file in the backup folder'] = bv['size_bytes'] == sum(files.values())
    out['C'] = {'files': files, 'payload_bytes': payload, 'payload_plus_manifest': payload + files['MANIFEST.sha256'],
                'all_files_bytes': sum(files.values()), 'record_size_bytes': bv['size_bytes']}
    bvt = (scr / 'bundle_verify.txt').read_text()
    chk['C: bundle verifies; holds the research branch at the tip; requires the base 913eb5c91'] = (
        'is okay' in bvt and '42ef29832191a797a44bc10ed5f61073b23d8d63 refs/heads/research/pcrl-reader-fit-aware-release-v1' in bvt
        and '913eb5c918837b909237527ddba8192d1521c347' in bvt and bv['bundle_verify'] is True)
    rc_disk = jl(BK_DIR / 'RESTORE_CHECK.json')
    chk['C: RESTORE_CHECK.json on disk agrees with the record (tip, rehash, bundle, all_ok, passes)'] = (
        rc_disk.get('tip_commit') == bv['tip_commit'] and rc_disk.get('manifest_rehash_ok') == bv['manifest_rehash_ok'] is True
        and rc_disk.get('bundle_verify') == bv['bundle_verify'] and rc_disk.get('all_ok') is True and bv['all_ok'] is True
        and rc_disk.get('restore_check_from_backup') == bv['restore_check_from_backup'] and rc_disk.get('checked_utc') == bv['checked_utc']
        and rc_disk.get('read_back') == bv['read_back'])
    out['C']['restore_check_keys_disk'] = sorted(rc_disk)
    # code.tar against the tip's tree blobs
    tip_ls = {}
    for x in (scr / 'tip_ls.txt').read_text().splitlines():
        meta, path = x.split('\t', 1)
        tip_ls[path] = meta.split()[2]
    with tarfile.open(BK_DIR / 'code.tar') as t:
        code = {m.name: blob_id(t.extractfile(m).read()) for m in t.getmembers() if m.isfile()}
    scope = {p: s for p, s in tip_ls.items() if p.split('/')[0] in CODE[:-1] or any(p.startswith(f'{c}/') for c in bv_campaigns(ROOT))}
    chk['C: code.tar = git archive of the tip (every member equals its tip blob)'] = all(tip_ls.get(p) == s for p, s in code.items()) and bool(code)
    out['C']['code_tar_members'] = len(code)
    out['C']['code_tar_members_not_in_tip'] = len([p for p in code if p not in tip_ls])
    out['C']['tip_entries_in_code_scope_missing_from_tar'] = len([p for p in scope if p not in code])
    # private.tar.gz member names against the private tree (names and sizes only; contents never parsed)
    # run 1 of this check compared macOS AppleDouble entries ('._<name>', written by bsdtar for extended attributes) as if they were
    # private files and failed on them (out/V5_CLOSEOUT_run1_CHECK_DESIGN_APPLEDOUBLE_AND_UNIT_FIELD.json); they are counted apart here
    with tarfile.open(BK_DIR / 'private.tar.gz', 'r:gz') as t:
        allm = [m for m in t.getmembers() if m.isfile()]
    apple = [m.name for m in allm if Path(m.name).name.startswith('._')]
    pm = {m.name: m.size for m in allm if not Path(m.name).name.startswith('._')}
    camp = {n: s for n, s in pm.items() if n.startswith('.private/pcrl_reader_fit_aware_release_v1/')}
    now = {}
    for dp, _, fns in os.walk(PRIV):
        for fn in fns:
            p = Path(dp) / fn
            now[str(p.relative_to(ROOT))] = p.stat().st_size
    bk_t = ts('2026-10-25T23:27:18+00:00')
    changed = [n for n, s in camp.items() if n in now and now[n] != s]
    gone = [n for n in camp if n not in now]
    new = [n for n in now if n not in camp and 'compute/tickets' not in n]
    new_after = [n for n in new if dt.datetime.fromtimestamp((ROOT / n).stat().st_mtime, dt.timezone.utc) >= bk_t]
    changed_after = [n for n in changed if dt.datetime.fromtimestamp((ROOT / n).stat().st_mtime, dt.timezone.utc) >= bk_t]
    chk['C: private.tar.gz holds the campaign private tree as of the backup (later-changed files only differ)'] = (
        not gone and sorted(new) == sorted(new_after) and sorted(changed) == sorted(changed_after)
        and not any('compute/tickets' in n for n in camp))
    cat = Counter()
    for n in pm:
        if '/runs/FAM/' in n and n.endswith('.pt'):
            cat['FAM model .pt'] += 1
        if n.endswith('GRID.json') and '/runs/GRID/' in n:
            cat['GRID.json'] += 1
        if '/runs/GRID/' in n and n.endswith('BOOT.npz'):
            cat['BOOT.npz'] += 1
        if n.endswith('RELEASE.npz'):
            cat['RELEASE.npz'] += 1
        if n.endswith('.pkl'):
            cat['reader .pkl'] += 1
    out['C']['private_tar'] = {'members': len(pm), 'appledouble_metadata_entries_excluded': len(apple),
                               'campaign_members': len(camp), 'inherited_members': len(pm) - len(camp),
                               'categories': dict(cat), 'campaign_files_changed_after_backup': len(changed),
                               'campaign_files_new_after_backup': len(new), 'campaign_files_gone': len(gone),
                               'changed_after_backup_names': [V.safe(n, cids) for n in sorted(changed)],
                               'new_after_backup_names': [V.safe(n, cids) for n in sorted(new)][:40]}
    rcb = bv['restore_check_from_backup']['passes']
    chk['C: restore-check counts consistent with the archive (24 family models, 24 releases, 12 grids)'] = (
        all(p.get('family_models_rehashed') == 24 and p.get('releases_rehashed_ok') == 24 and len(p.get('reader_grids') or {}) == 12
            for p in rcb.values()) and cat['GRID.json'] >= 12)
    # run 1 compared the records' descriptive 'unit' field ('<unit id> (queue; ...)') with the unit id as a whole string
    unit_b = [e for e in ev if e.get('unit_id') == (bv.get('unit') or '').split()[0]]
    chk['C: record written by its custody unit (ledger window) and checked after the backup'] = (
        len(unit_b) == 2 and ts(unit_b[0]['utc']) <= ts(bv['checked_utc']) <= ts(unit_b[1]['utc']) and unit_b[1]['status'] == 'done')
    # ------------------------------------------------------------------ D. the drive copy
    oc = jl(RES / 'custody' / 'OFFDEVICE_COPY.json')
    present = DRIVE.is_dir() and os.path.ismount(DRIVE)
    out['D'] = {'drive_mounted_now': present, 'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    if present:
        cp = DRIVE / 'PCRL_private_backups' / 'pcrl_reader_fit_aware_release_v1' / BK_DIR.name
        cfiles = {p.name: p.stat().st_size for p in cp.iterdir() if p.is_file() and not p.name.startswith('._')}
        cman = dict(x.split('  ', 1)[::-1] for x in (cp / 'MANIFEST.sha256').read_text().splitlines() if x.strip())
        chk['D: drive copy manifest = backup manifest = fresh re-hash'] = cman == man and all(sha256(cp / f) == man[f] for f in FILES)
        out['D']['files'] = cfiles
    else:
        out['D']['note'] = 'drive not mounted: nothing mounted, unmounted or forced; only the record and its consistency with the backup are checked'
    chk['D: record internally consistent with the backup (tip, size = payload + MANIFEST.sha256, WARM, all_ok, same drive identity as PROTOCOL)'] = (
        oc['tip_commit'] == bv['tip_commit'] and oc['size_bytes'] == payload + files['MANIFEST.sha256'] and oc['read_back'].startswith('WARM')
        and oc['all_ok'] is True and oc['manifest_equal_to_source'] is True and oc['manifest_rehash_ok'] is True and oc['bundle_verify'] is True
        and oc['drive']['volume_uuid'] == '3D74932B-33DD-3896-8CC6-0D2FDDAD6B8D' and '3D74932B-33DD-3896-8CC6-0D2FDDAD6B8D' in (RES / 'PROTOCOL.md').read_text()
        and oc['source_backup'] == bv['backup'] and oc['copy'].endswith(BK_DIR.name))
    unit_o = [e for e in ev if e.get('unit_id') == (oc.get('unit') or '').split()[0]]
    cpu_o = re.search(r'([0-9.]+) CPU-s', oc.get('unit') or '')
    chk['D: CPU stated in the record = the unit\'s ledger charge (0.1 s)'] = bool(cpu_o) and len(unit_o) == 2 and abs(
        float(cpu_o.group(1)) - float(unit_o[1]['charged_cpu_s'])) <= 0.05
    chk['D: record written by its custody unit (ledger window)'] = (len(unit_o) == 2 and ts(unit_o[0]['utc']) <= ts(oc['checked_utc']) <= ts(unit_o[1]['utc'])
                                                                     and unit_o[1]['status'] == 'done')
    out['all_pass'] = all(chk.values())
    out['cpu_s'] = time.process_time() - t0
    out['code_sha256'] = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ('v5_closeout.py', 'vsafe.py')}
    text = json.dumps(out, indent=1, sort_keys=True, default=str)
    if V.safe(text, cids) != text:
        raise SystemExit('refused: output would contain an identifying string')
    OUT.write_text(text + '\n')
    print(json.dumps({'all_pass': out['all_pass'], 'failed': [k for k, x in chk.items() if not x], 'A': out['A'], 'B': out['B'],
                      'C': {k: v for k, v in out['C'].items() if k not in ('new_after_backup_names',)}, 'D': out['D'], 'cpu_s': out['cpu_s']},
                     indent=1, default=str))


def bv_campaigns(root):
    """results/<campaign> prefixes archived in code.tar (rfa.custody.CODE)."""
    from rfa import custody as CU
    return [c for c in CU.CODE if c.startswith('results/')]


if __name__ == '__main__':
    main()
