"""Admission (no labels read, nothing fitted): copy the needed source artifacts read-only into this campaign's private tree,
each checked against a hash published by its own campaign, and write results/<campaign>/SOURCE_ADMISSION.json.

Sources:
- Role exports (Florida ACS 2019 development roles) and the frozen-hybrids bank: the SCM utility-matched follow-up's admitted
  copies; expected hashes from EXTRACTION_SUMMARY.json (public) and FROZEN_BANK_MANIFEST.json (public).
- Stage A (CatBoost teachers T_TREE as routed tree JSON, their stored outputs) and Stage B (LOCAL_TREE / JOINT_TREE /
  TASK_REFIT leaf-score corrections, all checkpoints, logs and seed records) of the protected-tree replication; expected hashes
  from its public TRAINING_COMPLETE_LOCK.json.
- The SCM utility-matched follow-up's frozen service bank (SCM, PUB_IBSI, ORD, SHUF, C0 releases); expected hash from its
  public SERVICE_BANK_MANIFEST.json.
"""
from __future__ import annotations
import json
import shutil

from . import common as C

EXPORTS = {'MODEL_FIT': ('features', 'income', 'occupation', 'sex', 'raw_inputs'), 'CALIBRATION': ('features', 'income', 'occupation'),
           'MODEL_SELECT': ('features', 'income', 'occupation', 'sex'), 'ATTACK_FIT': ('features', 'sex'),
           'ATTACK_VALIDATION': ('features', 'sex'), 'DEV_REPORT': ('features', 'raw_inputs')}
STAGE_A = ('TREE_TEACHER_income.json', 'TREE_TEACHER_occupation.json', 'STAGE_A_OUTPUTS.npz', 'STAGE_A_SEED.json',
           'CATBOOST_income.cbm', 'CATBOOST_occupation.cbm')
STAGE_B = tuple(f'{arm}__{x}' for arm in ('JOINT_TREE', 'LOCAL_TREE', 'TASK_REFIT')
                for x in ('E0.npz', 'E2.npz', 'E5.npz', 'E10.npz', 'E20.npz', 'LOGS.json')) + ('STAGE_B_SEED.json', 'STAGE_B_OUTPUTS_E20.npz')


def main():
    pub = C.public_export_sha()
    hyb = json.loads((C.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256']
    tcl = json.loads((C.REP_RESULTS / 'TRAINING_COMPLETE_LOCK.json').read_text())['frozen_files_sha256']
    umb = json.loads((C.UM_RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())['bank_files_sha256']
    plan = []
    for role, names in EXPORTS.items():
        for n in names:
            plan.append((f'exports/{role}/{n}.npz', C.UM_PRIV / 'source' / 'exports' / role / f'{n}.npz', pub[role][n],
                         'EXTRACTION_SUMMARY.json export_file_sha256'))
    for s in C.SEEDS:
        plan.append((f'bank/FROZEN_BANK_S{s}.npz', C.UM_PRIV / 'source' / 'bank' / f'FROZEN_BANK_S{s}.npz', hyb[f'S{s}'],
                     'pcrl_acs_frozen_hybrids_v1/FROZEN_BANK_MANIFEST.json'))
        for f in STAGE_A:
            plan.append((f'stage_a/S{s}/{f}', C.REP_PRIV / 'runs' / f'STAGE_A_{s}' / f, tcl[f'STAGE_A_{s}/{f}'],
                         'pcrl_acs_protected_tree_replication_v1/TRAINING_COMPLETE_LOCK.json'))
        for f in STAGE_B:
            plan.append((f'stage_b/S{s}/{f}', C.REP_PRIV / 'runs' / f'STAGE_B_{s}' / f, tcl[f'STAGE_B_{s}/{f}'],
                         'pcrl_acs_protected_tree_replication_v1/TRAINING_COMPLETE_LOCK.json'))
        plan.append((f'um_bank/SERVICE_BANK_S{s}.npz', C.UM_PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz', umb[f'S{s}'],
                     'pcrl_scm_utility_matched_v1/SERVICE_BANK_MANIFEST.json'))
    copies, problems = {}, []
    for rel, src, expected, basis in plan:
        if not src.is_file() or C.sha(src) != expected:
            problems.append(rel); continue
        dst = C.SRCP / rel
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not dst.exists():
            shutil.copyfile(src, dst); dst.chmod(0o400)
        if C.sha(dst) != expected:
            problems.append(rel); continue
        copies[rel] = {'sha256': expected, 'hash_basis': basis, 'bytes': dst.stat().st_size}
    rec = {'schema': 'AMD1-source-admission-v1', 'utc': C.now(), 'campaign': C.CAMPAIGN,
           'status': 'ADMITTED' if not problems else 'ADMISSION_FAILED', 'problems': problems, 'n_copies': len(copies),
           'copies': copies,
           'sources': {'latest_public_record': 'publication/pcrl-scm-utility-matched-v1 @ 8cb70b2901f7360ffcc946cee23664f4233bf504',
                       'earlier_scm_public_record': 'publication/pcrl-scm-engineering-benchmark-v1 @ 6fc06021d3ce42e1b281e92702b54bfe1cf384c8',
                       'research_base': 'research/pcrl-scm-utility-matched-v1 @ 482808212',
                       'tree_teachers_and_leaf_corrections': 'pcrl_acs_protected_tree_replication_v1 (public f4ea199)',
                       'frozen_references': 'pcrl_acs_frozen_hybrids_v1 (public 3a1f509)'},
           'labels_read': False, 'fits': 0, 'sealed_assessment': 'closed (no export exists; every loader refuses it)'}
    C.write_json_x(C.RESULTS / 'SOURCE_ADMISSION.json', rec)
    print(json.dumps({'status': rec['status'], 'n_copies': len(copies), 'problems': problems}))


if __name__ == '__main__':
    main()
