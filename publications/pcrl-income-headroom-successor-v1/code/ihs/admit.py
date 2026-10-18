"""Unit IHS_ADMIT: admit the lineage's private artifacts read-only, by hash (no label is opened here).

Every copy lands at .private/pcrl_income_headroom_successor_v1/admitted/<path> and is accepted only when its SHA-256 equals the
hash recorded by a committed record of its source:
- exports/, bank/, stage_a/, stage_b/, um_bank/, packages/: the source campaign's SOURCE_ADMISSION(_2).json (research
  @ 9e2b6c567), cross-checked for exports against the public extraction summary and for the frozen bank against the
  frozen-hybrids FROZEN_BANK_MANIFEST;
- amd_bank/SERVICE_BANK_S<s>.npz: the source campaign's SERVICE_BANK_MANIFEST;
- amd_runs/B/<config>/S<s>/RELEASE.npz: the source FINALIST_LOCK release hashes (as admitted by the validation); RELEASE.json:
  its release_sha256 must equal the admitted RELEASE.npz;
- amd_deploy/<package>/...: the source DEPLOYMENT_RESULTS package hashes and each package's file map (as admitted by the
  validation).
Access logs are snapshot-copied and hashed only (they are append-only and carry no earlier expected hash).

Output: results/<campaign>/SOURCE_ADMISSION.json (write-once).
"""
from __future__ import annotations
import json
import shutil

from . import common as C

AMD_RUNS = ('B3CTL_UNCOUPLED_MIX50_C6A', 'B3CTL_TASKONLY_TR_C6A', 'B2CTL_UNCOUPLED_C6A')
AMD_PACKAGES = ('FINALIST_S0', 'FINALIST_S1', 'TASKONLY_S0', 'TASKONLY_S1', 'NEURAL_RAW_S0', 'H_LJ_S0', 'H_LJ_S1')


def plan():
    """(destination path, source path, expected sha256 or None, basis)."""
    amd_adm = json.loads((C.AMD_RESULTS / 'SOURCE_ADMISSION.json').read_text())['copies']
    amd_adm.update(json.loads((C.AMD_RESULTS / 'SOURCE_ADMISSION_2.json').read_text())['copies'])
    ffv_adm = json.loads((C.FFV_RESULTS / 'SOURCE_ADMISSION.json').read_text())['copies']
    pub = C.public_export_sha()
    hyb = json.loads((C.SB.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256']
    sbm = json.loads((C.AMD_RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())['bank_files_sha256']
    out = []
    for rel, rec in sorted(amd_adm.items()):
        e = rec['sha256']
        if rel.startswith('exports/'):
            role, f = rel.split('/')[1], rel.split('/')[2][:-4]
            if e != pub[role][f]:
                raise PermissionError(f'source admission and public extraction summary disagree: {rel}')
        if rel.startswith('bank/FROZEN_BANK_S'):
            if e != hyb[f"S{rel[-5]}"]:
                raise PermissionError('frozen bank hash disagreement')
        out.append((rel, C.AMD_PRIV / 'admitted' / rel, e, 'source SOURCE_ADMISSION(_2).json'))
    for s in C.SEEDS:
        out.append((f'amd_bank/SERVICE_BANK_S{s}.npz', C.AMD_PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz', sbm[f'S{s}'],
                    'source SERVICE_BANK_MANIFEST'))
        for cfg in AMD_RUNS:
            k = f'runs/B/{cfg}/S{s}/RELEASE.npz'
            out.append((f'amd_{k}', C.AMD_PRIV / k, ffv_adm[k]['expected_sha256'], 'source FINALIST_LOCK release_sha256 (validation admission)'))
            kj = f'runs/B/{cfg}/S{s}/RELEASE.json'
            out.append((f'amd_{kj}', C.AMD_PRIV / kj, ffv_adm[kj]['sha256'], 'release record as admitted by the validation'))
    for k, rec in sorted(ffv_adm.items()):
        if k.startswith('deploy/') and k.split('/')[1] in AMD_PACKAGES:
            out.append((f'amd_{k}', C.AMD_PRIV / k, rec['expected_sha256'], rec['basis'] + ' (validation admission)'))
    return out


def run():
    rows, problems = {}, []
    for dst, src, expected, basis in plan():
        p = C.ADM / dst
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not src.is_file():
            problems.append(f'missing source file: {dst}'); continue
        if not p.exists():
            shutil.copyfile(src, p)
        h = C.sha(p)
        if expected is not None and h != expected:
            problems.append(f'hash differs: {dst}')
        rows[dst] = {'sha256': h, 'expected_sha256': expected, 'basis': basis, 'bytes': p.stat().st_size}
    for s in C.SEEDS:                                   # release records must point at the admitted release arrays
        for cfg in AMD_RUNS:
            rec = json.loads((C.ADM / f'amd_runs/B/{cfg}/S{s}/RELEASE.json').read_text())
            if rec['release_sha256'] != rows[f'amd_runs/B/{cfg}/S{s}/RELEASE.npz']['sha256']:
                problems.append(f'release record does not match its array: {cfg} S{s}')
    logs = {'amd_access/DEV_REPORT_REREAD.jsonl': C.AMD_PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl',
            'amd_access/MODEL_SELECT_USE.jsonl': C.AMD_PRIV / 'access' / 'MODEL_SELECT_USE.jsonl',
            'ffv_access/DEV_REPORT_REREAD.jsonl': C.FFV_PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl',
            'ffv_access/ATTACK_LABEL_READS.jsonl': C.FFV_PRIV / 'access' / 'ATTACK_LABEL_READS.jsonl',
            'lineage_access/ACCESS_LOG_snapshot.jsonl': C.SRC_ACCESS_LOG}
    snapshots = {}
    for dst, src in logs.items():
        if src.is_file():
            p = C.ADM / dst
            p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(src, p)
            lines = [l for l in p.read_text().splitlines() if l.strip()]
            snapshots[dst] = {'sha256': C.sha(p), 'lines': len(lines), 'bytes': p.stat().st_size}
        else:
            snapshots[dst] = {'status': 'NOT_PRESENT'}
    lineage = [json.loads(l) for l in (C.ADM / 'lineage_access/ACCESS_LOG_snapshot.jsonl').read_text().splitlines() if l.strip()]
    dev_events = [e.get('event') for e in lineage if 'DEV_REPORT' in str(e.get('event', ''))]
    rec = {'schema': 'IHS1-source-admission-v1', 'campaign': C.CAMPAIGN, 'utc': C.now(),
           'sources': {'source_campaign': {'branch': 'research/pcrl-autonomous-method-development-v1', 'tip': C.AMD_TIP,
                                           'publication': '5901f7c737c111c9228a3c2035bb77aa77a20f0c'},
                       'validation': {'branch': 'research/pcrl-frozen-finalist-validation-v1', 'tip': C.FFV_TIP,
                                      'publication': '2abfaefdc69725907730d15bf75d24a59d5c20ca'}},
           'copies': rows, 'n_copies': len(rows), 'access_log_snapshots': snapshots,
           'lineage_dev_report_events_before_this_campaign': {'count': len(dev_events), 'events': dev_events},
           'labels_read': 'none (exports are copied and hashed; no label array is opened by this unit)',
           'sealed_assessment': 'not admitted; every loader refuses it',
           'problems': problems, 'status': 'PASS' if not problems else 'FAIL'}
    C.write_json_x(C.RESULTS / 'SOURCE_ADMISSION.json', rec)
    print(json.dumps({'n_copies': len(rows), 'problems': problems, 'lineage_dev_events': len(dev_events)}))
    return rec


if __name__ == '__main__':
    run()
