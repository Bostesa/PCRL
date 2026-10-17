"""Unit RESTORE: copy the admitted source artifacts read-only into this campaign's private tree and check every hash.

- Role exports (only the fields this campaign's protocol admits; DEV_REPORT truth is NOT copied): checked against the public
  EXTRACTION_SUMMARY.json export_file_sha256 of the source study (f4ea199).
- Source model outputs (U0, Ucal, T_TREE, TASK_REFIT, LOCAL_TREE, JOINT_TREE, the NEURAL_RAW grid) and the U0 weights: checked
  against the frozen-hybrids RESTORE_RECEIPT, which was itself checked against the published MODEL_MANIFEST.
- The frozen-hybrids bank: checked against the public FROZEN_BANK_MANIFEST.json (3a1f509).
Nothing is fitted and no label is summarised here. The copy is append-only: an existing destination file must already match.
"""
from __future__ import annotations
import json
import shutil

from . import common as C


def run():
    pub = C.public_export_sha()
    hyb = json.loads((C.HYB_PRIV / 'RESTORE_RECEIPT.json').read_text())['files']
    bank_man = json.loads((C.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256']
    C.SRCP.mkdir(parents=True, exist_ok=True, mode=0o700)
    rec = {'schema': 'SMP1-restore-receipt-v1', 'utc': C.now(), 'source_private_root': 'source worktree .private/acs_replication (read-only)',
           'files': {}, 'not_copied': ['DEV_REPORT income/occupation/sex/weights (re-read in place only after the committed REPORTING_LOCK)',
                                       'any SEALED_ASSESSMENT record (no export exists)', 'weights of every role']}

    def copy(src, rel, expected, basis):
        dst = C.SRCP / rel
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if C.sha(src) != expected:
            raise PermissionError(f'source hash differs from {basis}: {rel}')
        if dst.exists():
            if C.sha(dst) != expected:
                raise PermissionError(f'existing restored file differs: {rel}')
        else:
            shutil.copyfile(src, dst); dst.chmod(0o600)
        if C.sha(dst) != expected:
            raise PermissionError(f'copy hash differs: {rel}')
        rec['files'][rel] = {'sha256': expected, 'basis': basis, 'bytes': dst.stat().st_size}

    for role, fields in C.ADMITTED.items():
        for f in fields:
            rel = f'exports/{role}/{f}.npz'
            copy(C.SRC_PRIV / rel, rel, pub[role][f], 'public EXTRACTION_SUMMARY.json export_file_sha256 (f4ea199)')
    for rel in C.RUN_FILES:
        copy(C.SRC_PRIV / rel, rel, hyb[rel]['sha256'], 'frozen-hybrids RESTORE_RECEIPT (checked there against the published MODEL_MANIFEST)')
    sel = json.loads((C.SRC_RESULTS / 'SELECTION_LOCK.json').read_text())['output_file_sha256']
    for s in C.SEEDS:
        rel = f'bank/FROZEN_BANK_S{s}.npz'
        copy(C.HYB_PRIV / rel, rel, bank_man[f'S{s}'], 'public FROZEN_BANK_MANIFEST.json (3a1f509)')
        for rel in (f'runs/FARE_{s}/FARE_OUTPUTS.npz', f'runs/FAGTB_{s}/FAGTB_OUTPUTS.npz'):   # utility-ineligible published baselines
            copy(C.SRC_PRIV / rel, rel, sel[rel], 'public SELECTION_LOCK.json output_file_sha256 (f4ea199)')
    rec['n_files'] = len(rec['files'])
    (C.PRIV / 'RESTORE_RECEIPT.json').write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


if __name__ == '__main__':
    r = run()
    print(json.dumps({'n_files': r['n_files'], 'bytes': sum(v['bytes'] for v in r['files'].values())}))
