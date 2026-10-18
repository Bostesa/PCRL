"""VER check 2: exposure and label access (reads JSON logs and records only; no data array).

- label-access logs of this campaign: every label read after the verified registration receipt; MODEL_SELECT labels read only by
  scoring / reader-grid / verification units (queue stage and command); no ATTACK or DEV_REPORT label read before the gates;
- no SEX in a task-only fit: every OCC.json of kind task_model (or a task-only control) against the TRAINING_LABEL_READS lines of
  its unit; every SEX read on MODEL_FIT attributed to its unit's stage and command;
- the source's ihs loaders: no access directory in the admitted mirror; the original source access logs and the lineage DEV_REPORT
  log unchanged since their admission snapshots;
- queue units started before the receipt verification and their stages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402

from ohn import common as C  # noqa: E402  (paths only)

ACCESS = V.PRIV / 'access'
LEDGER = V.PRIV / 'compute' / 'ledger.jsonl'


def jl(p):
    if not p.is_file():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def module_of(argv):
    a = list(argv or [])
    if '-m' in a:
        i = a.index('-m')
        return ' '.join(a[i + 1:i + 3])
    return ' '.join(Path(x).name for x in a[:2])


def main():
    rec = {'check': 'VER_ACCESS', 'utc': V.utc()}
    receipt = json.loads((ACCESS / 'REGISTRATION_RECEIPT_VERIFIED.json').read_text())
    t_ver = receipt['verified_utc']
    rec['receipt_verified_utc'] = t_ver
    units = {}
    for e in jl(LEDGER) or []:
        if e.get('event') == 'unit_start':
            units.setdefault(e['unit_id'], {}).update({'stage': e['stage'], 'argv': e['argv'], 'start': e['utc']})
        elif e.get('event') == 'unit_end':
            units.setdefault(e['unit_id'], {}).update({'end': e['utc'], 'status': e['status']})
    logs = {n: jl(ACCESS / n) for n in ('MODEL_SELECT_USE.jsonl', 'TRAINING_LABEL_READS.jsonl', 'ATTACK_LABEL_READS.jsonl',
                                         'DEV_REPORT_REREAD.jsonl')}
    rec['log_lines'] = {n: (None if v is None else len(v)) for n, v in logs.items()}
    issues, notes = [], []
    all_reads = [(n, e) for n in ('MODEL_SELECT_USE.jsonl', 'TRAINING_LABEL_READS.jsonl', 'ATTACK_LABEL_READS.jsonl') for e in (logs[n] or [])]
    early = [(n, e) for n, e in all_reads if e['utc'] < t_ver]
    rec['label_reads_before_receipt_verification'] = len(early)
    if early:
        issues.append(f'{len(early)} label reads before {t_ver}')
    rec['first_label_read_utc'] = min((e['utc'] for _, e in all_reads), default=None)
    if logs['ATTACK_LABEL_READS.jsonl']:
        lock = V.RES / 'FINALIST_LOCK.json'
        if not lock.is_file():
            issues.append('ATTACK label reads without FINALIST_LOCK')
        notes.append(f"{len(logs['ATTACK_LABEL_READS.jsonl'])} ATTACK label reads logged")
    if logs['DEV_REPORT_REREAD.jsonl']:
        notes.append(f"{len(logs['DEV_REPORT_REREAD.jsonl'])} DEV_REPORT events logged: " +
                     ', '.join(e.get('event', '?') for e in logs['DEV_REPORT_REREAD.jsonl']))
    # MODEL_SELECT readers
    ms = []
    for e in logs['MODEL_SELECT_USE.jsonl'] or []:
        u = units.get(e.get('unit_id'), {})
        mod = module_of(u.get('argv'))
        kind = ('verification' if (e.get('unit_id') or '').startswith('VER_') else
                'scoring' if (mod.startswith('ohn.devrun score') or mod.startswith('ohn.devrun references')) else
                'reader_grid' if mod.startswith('ohn.devreaders run') else
                'reader_robustness_probe' if (mod.split(' ')[-1].startswith('rr_') and mod.split(' ')[-1].endswith('.py')
                                              and (e.get('unit_id') or '').startswith('OHN_RR_')) else
                'fallback_decomposition' if mod.startswith('ohn.decompose') else 'OTHER')
        okp = kind != 'verification' or e.get('purpose', '').startswith(V.PURPOSE)
        if kind == 'fallback_decomposition':
            # PROTOCOL section 9 (fallback programme) decomposition of occupation costs by class and SEX on MODEL_SELECT; diagnostic,
            # logged; outside the common brief's "only through ohn.devrun score and ohn.devreaders" wording -> deviation note
            ok = u.get('stage') == 'diagnostic' and bool(e.get('purpose'))
            notes.append(f"MODEL_SELECT {','.join(e['fields'])} read by fallback decomposition {e.get('unit_id')} (ohn.decompose; PROTOCOL "
                         "section 9 diagnostic; outside the 'only through ohn.devrun score / ohn.devreaders' wording)")
        elif kind == 'reader_robustness_probe':
            # coordinator-authorized diagnostic (READER_ROBUSTNESS_BRIEF item 4): MODEL_SELECT SEX only, never selection; it is
            # outside the common brief's "only through ohn.devrun score and ohn.devreaders" wording -> recorded as a deviation note
            ok = (u.get('stage') == 'diagnostic' and e['fields'] == ['sex'] and 'never selection' in e.get('purpose', ''))
            notes.append(f"MODEL_SELECT SEX read by reader-robustness diagnostic {e.get('unit_id')} (custom script {mod.split(' ')[-1]}; "
                         "reader-robustness brief / coordinator request; outside the 'only through ohn.devrun score / ohn.devreaders' wording)")
        else:
            ok = kind != 'OTHER' and u.get('stage') in ('score', 'readers', 'verification') and okp
        # REGISTERED rule (rule v1, unchanged since 05:51Z): MODEL_SELECT labels / SEX only through ohn.devrun score / references,
        # ohn.devreaders run or the verifier's own units (AGENT_BRIEF_COMMON; PROTOCOL section 3; VERIFIER_BRIEF check 2)
        ok_registered = kind in ('scoring', 'reader_grid', 'verification') and u.get('stage') in ('score', 'readers', 'verification') and okp
        ms.append({'unit_id': e.get('unit_id'), 'stage': u.get('stage'), 'module': mod, 'kind': kind, 'fields': e['fields'], 'ok': ok,
                   'ok_registered_rule': ok_registered, 'utc': e['utc'], 'purpose': e.get('purpose', '')})
        if not ok:
            issues.append(f"MODEL_SELECT read by unit {e.get('unit_id')} (stage {u.get('stage')}, {mod})")
    rec['model_select_reads'] = {'n': len(ms), 'by_kind': {k: sum(r['kind'] == k for r in ms) for k in sorted({r['kind'] for r in ms})},
                                 'not_ok': [r for r in ms if not r['ok']],
                                 'outside_registered_entry_points': [r for r in ms if not r['ok_registered_rule']]}
    rec['rule_versions'] = {'registered_rule': 'v1 (05:51Z): scoring / reader-grid / verification units only',
                            'amended_rule': 'v4 (07:17:37Z): v1 plus authorized diagnostic classes (reader-robustness rr_*.py units OHN_RR_*, '
                                            'stage diagnostic, SEX only, purpose "never selection"; ohn.decompose fallback, stage diagnostic); '
                                            'see records/ACCESS_CHECK_AMENDMENT.json'}
    # SEX on MODEL_FIT and training reads
    tr = logs['TRAINING_LABEL_READS.jsonl'] or []
    sex_reads = []
    for e in tr:
        if 'sex' in e['fields']:
            u = units.get(e.get('unit_id'), {})
            sex_reads.append({'unit_id': e.get('unit_id'), 'role': e['role'], 'stage': u.get('stage'), 'module': module_of(u.get('argv')),
                              'purpose': e['purpose']})
    rec['sex_reads_in_training_log'] = sex_reads
    # OCC.json records vs logs
    occ = {}
    for p in sorted(V.RUNS.glob('*/*/S*/OCC.json')) + sorted(V.RUNS.glob('*/*/S*/RELEASE.json')):
        r = json.loads(p.read_text())
        uid = r.get('unit_id')
        reads = [e for e in tr if e.get('unit_id') == uid]
        msr = [e for e in (logs['MODEL_SELECT_USE.jsonl'] or []) if e.get('unit_id') == uid]
        logged = sorted({f"{e['role']}:{','.join(e['fields'])}" for e in reads})
        sex_in_fit = any('sex' in e['fields'] for e in reads)
        task_only = r.get('kind') == 'task_model' or (r.get('kind') == 'control' and bool(r.get('task_only')))
        rr = json.dumps(r.get('roles_read', ''))
        mentions = {role: (role in rr) for role in ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT')}
        roles_logged = sorted({e['role'] for e in reads})
        consistent = all(mentions[x] for x in roles_logged)
        row = {'unit_id': uid, 'kind': r.get('kind'), 'task_only': task_only, 'logged_reads': logged, 'sex_read_by_fit_unit': sex_in_fit,
               'model_select_reads_by_fit_unit': len(msr), 'roles_read_mentions': mentions, 'logged_roles_mentioned': consistent,
               'record_says_sex': ('sex' in rr.lower() and 'no sex' not in rr.lower())}
        occ[str(p.relative_to(V.RUNS))] = row
        if task_only and sex_in_fit:
            issues.append(f'SEX read by task-only fit {p.parent}')
        if 'NO SEX' in rr.upper() and sex_in_fit:
            issues.append(f'record declares no SEX but its fit unit read SEX: {p.parent}')
        if sex_in_fit and any(e['role'] != 'MODEL_FIT' for e in reads if 'sex' in e['fields']):
            issues.append(f'SEX read outside MODEL_FIT by fit unit {uid}')
        row['sex_read_roles'] = sorted({e['role'] for e in reads if 'sex' in e['fields']})
        if msr:
            issues.append(f'MODEL_SELECT read by fit unit {uid} ({p.parent})')
        if uid is None:
            issues.append(f'no unit id in {p}')
        if not consistent:
            notes.append(f'{p.relative_to(V.RUNS)}: logged roles {roles_logged} not all named in roles_read')
    rec['release_records'] = occ
    # the source's ihs loaders
    mirror_access = V.MIRROR / 'access'
    rec['mirror_access_dir_exists'] = mirror_access.exists()
    if rec['mirror_access_dir_exists']:
        issues.append('the admitted mirror has an access directory (an ihs loader was used)')
    adm = json.loads((V.RES / 'SOURCE_ADMISSION.json').read_text())['access_log_snapshots']
    src_logs = {'ihs_access/DEV_REPORT_REREAD.jsonl': C.IHS_SRC_PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl',
                'ihs_access/MODEL_SELECT_USE.jsonl': C.IHS_SRC_PRIV / 'access' / 'MODEL_SELECT_USE.jsonl',
                'ihs_access/ATTACK_LABEL_READS.jsonl': C.IHS_SRC_PRIV / 'access' / 'ATTACK_LABEL_READS.jsonl',
                'lineage_access/ACCESS_LOG_snapshot.jsonl': C.SRC_ACCESS_LOG}
    snap = {}
    for k, p in src_logs.items():
        now_sha = V.file_sha256(p) if p.is_file() else None
        n_lines = len([l for l in p.read_text().splitlines() if l.strip()]) if p.is_file() else None
        snap[k] = {'unchanged': now_sha == adm[k].get('sha256'), 'lines_now': n_lines, 'lines_at_admission': adm[k].get('lines')}
        if now_sha != adm[k].get('sha256'):
            lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            snap[k]['new_events'] = [{'event': e.get('event'), 'campaign': e.get('campaign'), 'utc': e.get('utc')} for e in lines[adm[k]['lines']:]]
            notes.append(f'source log changed since admission: {k}')
    rec['source_logs_vs_admission_snapshot'] = snap
    early_units = sorted((u.get('start', ''), uid, u.get('stage')) for uid, u in units.items() if u.get('start', '9') < t_ver)
    rec['units_started_before_receipt_verification'] = [{'start': s, 'unit_id': uid, 'stage': st} for s, uid, st in early_units]
    rec['issues'], rec['notes'] = issues, notes
    outside = rec['model_select_reads']['outside_registered_entry_points']
    other_issues = [i for i in issues if not i.startswith('MODEL_SELECT read by unit')]
    # two statuses, never merged: the registered rule's verdict stands as the finding; the amended rule only separates
    # coordinator-authorized diagnostic reads from unauthorized ones
    rec['status_registered_rule'] = 'PASS' if (not outside and not issues) else 'FAIL'
    rec['status_amended_rule'] = 'PASS' if not issues else 'FAIL'
    rec['n_reads_outside_registered_entry_points'] = len(outside)
    rec['status'] = ('PASS' if (not outside and not issues) else
                     'FAIL_REGISTERED_RULE__DEVIATIONS_AUTHORIZED_DIAGNOSTIC' if (outside and not issues) else 'FAIL')
    rec['other_issues'] = other_issues
    V.write_record('VER_ACCESS', rec)
    print(json.dumps({'status': rec['status'], 'issues': issues, 'notes': notes, 'log_lines': rec['log_lines'],
                      'first_read': rec['first_label_read_utc'], 'ms_by_kind': rec['model_select_reads']['by_kind'],
                      'sex_reads': [(r['unit_id'], r['role'], r['stage']) for r in sex_reads], 'n_release_records': len(occ),
                      'mirror_access_dir_exists': rec['mirror_access_dir_exists'], 'status_registered_rule': rec['status_registered_rule'], 'status_amended_rule': rec['status_amended_rule'], 'n_outside': rec['n_reads_outside_registered_entry_points'],
                      'src_logs_unchanged': {k: v['unchanged'] for k, v in snap.items()}}, indent=1))


if __name__ == '__main__':
    main()
