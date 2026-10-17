"""Unit FFV_ADMIT: admit the source campaign's private artifacts read-only, by hash, and freeze the candidate manifest.

Every copy lands at .private/pcrl_frozen_finalist_validation_v1/admitted/amd/<source-relative path> and is accepted only when its
SHA-256 equals the hash recorded by the source's own committed records (public extraction summary, source admission, frozen-bank
and service-bank manifests, finalist lock, reporting lock, reader record, deployment record and package file maps). Access logs
are snapshot-copied and hashed (they carry no earlier expected hash). Nothing is refitted and no label is read here (the
export copies include ATTACK_FIT / ATTACK_VALIDATION SEX files, which are hashed but not opened).

Outputs: results/<campaign>/SOURCE_ADMISSION.json and FROZEN_CANDIDATE_MANIFEST.json.
"""
from __future__ import annotations
import json
import shutil

import numpy as np

from . import common as C

RELEASE_DIRS = {'FINALIST': 'B3CTL_UNCOUPLED_MIX50_C6A', 'TASKONLY': 'B3CTL_TASKONLY_TR_C6A', 'B2TWIN': 'B2CTL_UNCOUPLED_C6A'}
PACKAGES = ('FINALIST_S0', 'FINALIST_S1', 'TASKONLY_S0', 'TASKONLY_S1', 'NEURAL_RAW_S0', 'H_LJ_S0', 'H_LJ_S1')


def _src_json(name):
    return json.loads((C.SRC_RESULTS / name).read_text())


def plan():
    """(source-relative path, expected sha256 or None, basis) for every file admitted."""
    amd_adm = _src_json('SOURCE_ADMISSION.json')['copies']
    amd_adm.update(_src_json('SOURCE_ADMISSION_2.json')['copies'])
    pub = C.public_export_sha()
    hyb = json.loads((C.SB.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256']
    sbm = _src_json('SERVICE_BANK_MANIFEST.json')['bank_files_sha256']
    rlock = _src_json('REPORTING_LOCK.json')['readers']
    dep = _src_json('DEPLOYMENT_RESULTS.json')['packages']
    out = []
    for role, fields in C.ADMITTED.items():
        for f in fields:
            e = amd_adm[f'exports/{role}/{f}.npz']['sha256']
            if e != pub[role][f]:
                raise PermissionError(f'source admission and public extraction summary disagree: {role}/{f}')
            out.append((f'admitted/exports/{role}/{f}.npz', e, 'public EXTRACTION_SUMMARY export hash = source SOURCE_ADMISSION'))
    for s in C.SEEDS:
        e = amd_adm[f'bank/FROZEN_BANK_S{s}.npz']['sha256']
        if e != hyb[f'S{s}']:
            raise PermissionError('frozen bank hash disagreement')
        out.append((f'admitted/bank/FROZEN_BANK_S{s}.npz', e, 'FROZEN_BANK_MANIFEST = source SOURCE_ADMISSION'))
        for t in C.TASKS:
            out.append((f'admitted/stage_a/S{s}/TREE_TEACHER_{t}.json', amd_adm[f'stage_a/S{s}/TREE_TEACHER_{t}.json']['sha256'], 'source SOURCE_ADMISSION'))
        for arm in ('LOCAL_TREE', 'JOINT_TREE', 'TASK_REFIT'):
            out.append((f'admitted/stage_b/S{s}/{arm}__E20.npz', amd_adm[f'stage_b/S{s}/{arm}__E20.npz']['sha256'], 'source SOURCE_ADMISSION'))
        out.append((f'admitted/packages/T_TREE_S{s}/PACKAGE.json', amd_adm[f'packages/T_TREE_S{s}/PACKAGE.json']['sha256'], 'source SOURCE_ADMISSION_2'))
        out.append((f'bank/SERVICE_BANK_S{s}.npz', sbm[f'S{s}'], 'source SERVICE_BANK_MANIFEST'))
        out.append((f'runs/READERS_{s}/READERS_SEED.json', rlock[f'S{s}']['readers_seed_sha256'], 'source REPORTING_LOCK'))
        out.append((f'runs/READERS_{s}/DEV_REPORT_SCORES.npz', rlock[f'S{s}']['dev_report_scores_sha256'], 'source REPORTING_LOCK'))
        rr = json.loads((C.SRC_PRIV / f'runs/READERS_{s}/READERS_SEED.json').read_text())
        if C.sha(C.SRC_PRIV / f'runs/READERS_{s}/READERS_SEED.json') != rlock[f'S{s}']['readers_seed_sha256']:
            raise PermissionError('source reader record differs from the source REPORTING_LOCK')
        for rid, h in rr['model_files_sha256'].items():
            out.append((f'runs/READERS_{s}/MODEL__{rid}.joblib', h, 'source READERS_SEED.json model_files_sha256 (under the source REPORTING_LOCK)'))
        flock = _src_json('FINALIST_LOCK.json')['release_sha256']
        for role, cfg in RELEASE_DIRS.items():
            out.append((f'runs/B/{cfg}/S{s}/RELEASE.npz', flock[f'B/{cfg}'][f'S{s}'], 'source FINALIST_LOCK release_sha256'))
            out.append((f'runs/B/{cfg}/S{s}/RELEASE.json', None, 'release record (its release_sha256 is checked against FINALIST_LOCK)'))
    for pkg in PACKAGES:
        meta_sha = dep[pkg].get('package_json_sha256') or dep[pkg].get('source_package_json_sha256')
        out.append((f'deploy/{pkg}/PACKAGE.json', meta_sha, 'source DEPLOYMENT_RESULTS package hash'))
        meta = json.loads((C.SRC_PRIV / f'deploy/{pkg}/PACKAGE.json').read_text())
        for f, h in meta['files'].items():
            out.append((f'deploy/{pkg}/{f}', h, 'package file map in the hash-checked PACKAGE.json'))
    for log in ('DEV_REPORT_REREAD.jsonl', 'MODEL_SELECT_USE.jsonl'):
        out.append((f'access/{log}', None, 'snapshot of the source campaign access log (read-only; never modified)'))
    return out


def main():
    if (C.RESULTS / 'SOURCE_ADMISSION.json').exists():
        raise PermissionError('admission exists')
    copies, problems = {}, []
    for rel, expected, basis in plan():
        src, dst = C.SRC_PRIV / rel, C.ADM / rel
        if not src.is_file():
            problems.append({'path': rel, 'problem': 'missing in the source private tree'}); continue
        h_src = C.sha(src)
        if expected is not None and h_src != expected:
            problems.append({'path': rel, 'problem': 'source file hash differs from its recorded hash', 'expected': expected, 'found': h_src}); continue
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(src, dst); dst.chmod(0o400)
        h_dst = C.sha(dst)
        if h_dst != h_src:
            problems.append({'path': rel, 'problem': 'copy hash differs'}); continue
        copies[rel] = {'sha256': h_dst, 'expected_sha256': expected, 'basis': basis, 'bytes': dst.stat().st_size}
    # release records: the stored release_sha256 equals FINALIST_LOCK, and the stored evaluated-role outputs equal the bank bitwise
    checks = {}
    flock = _src_json('FINALIST_LOCK.json')['release_sha256']
    for s in C.SEEDS:
        with np.load(C.ADM / f'bank/SERVICE_BANK_S{s}.npz') as z:
            bank = {k: z[k].copy() for k in z.files}
        for role, cfg in RELEASE_DIRS.items():
            rec = json.loads((C.ADM / f'runs/B/{cfg}/S{s}/RELEASE.json').read_text())
            with np.load(C.ADM / f'runs/B/{cfg}/S{s}/RELEASE.npz') as z:
                rel_arr = {k: z[k].copy() for k in z.files}
            same = all(np.array_equal(rel_arr[f'{r}__{t}'], bank[f'{role}__{r}__{t}']) for r in ('MODEL_SELECT',) + C.ROLES3 for t in C.TASKS)
            checks[f'{role}|S{s}'] = {'release_record_sha256_equals_finalist_lock': rec['release_sha256'] == flock[f'B/{cfg}'][f'S{s}'],
                                       'evaluated_role_outputs_equal_bank_bitwise': bool(same),
                                       'roles_in_release_record': sorted({k.split('__')[0] for k in rel_arr})}
            if not all(v for k, v in checks[f'{role}|S{s}'].items() if k != 'roles_in_release_record'):
                problems.append({'path': f'runs/B/{cfg}/S{s}', 'problem': 'release record does not match the lock or the bank'})
        # bank: six admitted releases; wires equal the source manifest; probability contract
        man = _src_json('SERVICE_BANK_MANIFEST.json')['per_seed'][f'S{s}']['wire_sha256']
        for svc in C.BANK:
            for r in ('MODEL_SELECT',) + C.ROLES3:
                for t in C.TASKS:
                    p = bank[f'{svc}__{r}__{t}']
                    ok = (C.wire_sha(p) == man[f'{svc}|{t}'][r] and p.dtype == np.float64 and p.shape[1] == C.K[t]
                          and np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-9, rtol=0))
                    if not ok:
                        problems.append({'path': f'bank S{s} {svc} {r} {t}', 'problem': 'wire hash or probability contract differs'})
        checks[f'bank|S{s}'] = {'services_admitted': list(C.BANK), 'services_in_source_bank_not_admitted': sorted({k.split('__')[0] for k in bank if '__' in k
                                and k.split('__')[0] not in ('row_id', 'unit')} - set(C.BANK))}
    rec = {'schema': 'FFV1-source-admission-v1', 'campaign': C.CAMPAIGN, 'utc': C.now(), 'source_branch': 'research/pcrl-autonomous-method-development-v1',
           'source_tip': C.SRC_TIP, 'source_private_tree': '<worktrees>/pcrl-autonomous-method-development-v1/.private/' + C.SRC_CAMPAIGN,
           'copies': copies, 'n_copies': len(copies), 'checks': checks, 'problems': problems, 'status': 'ADMITTED' if not problems else 'PROBLEMS',
           'labels_read': 'none (ATTACK_FIT / ATTACK_VALIDATION SEX files are copied and hashed, not opened; no other role label is copied)',
           'sealed_assessment': 'not admitted; refused by every loader'}
    C.write_json_x(C.RESULTS / 'SOURCE_ADMISSION.json', rec)
    print(json.dumps({'n_copies': len(copies), 'problems': problems, 'checks': checks}, indent=1))
    return rec


if __name__ == '__main__':
    main()
