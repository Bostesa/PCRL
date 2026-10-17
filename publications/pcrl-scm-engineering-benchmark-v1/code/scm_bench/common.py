"""Shared definitions for the PCRL SCM engineering benchmark (EXPLORATORY DEVELOPMENT PILOT; not confirmation).

Source: the Florida ACS 2019 development roles of the protected-tree replication (public f4ea199) and the frozen comparator
bank of the frozen-hybrids follow-up (public 3a1f509). Source artifacts are copied read-only into
.private/pcrl_scm_engineering_benchmark_v1/source by `restore`, each checked against the published hashes
(EXTRACTION_SUMMARY.json export_file_sha256 for role exports; the hybrids RESTORE_RECEIPT / FROZEN_BANK_MANIFEST for model
outputs).

Guards:
- SEALED_ASSESSMENT has no export and is refused by every loader.
- Role access follows the campaign protocol: MODEL_FIT (features, both task labels, SEX) for defense fitting; CALIBRATION
  (features, task labels) for calibration; MODEL_SELECT (features, task labels, SEX) for selection under the frozen rule;
  ATTACK_FIT / ATTACK_VALIDATION (features, SEX) for readers only; DEV_REPORT (features, raw inputs) label-free.
- DEV_REPORT truth (labels, SEX) is re-read only through `dev_report_truth`, which requires THIS campaign's committed
  REPORTING_LOCK and appends a logged re-read event to this campaign's access log and to the source access log. The source's
  one-time-opening receipt is never modified or reset.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRIV = ROOT / '.private' / 'pcrl_scm_engineering_benchmark_v1'
SRCP = PRIV / 'source'
RESULTS = ROOT / 'results' / 'pcrl_scm_engineering_benchmark_v1'
PCRL = Path(os.environ.get('SMP_PCRL_ROOT', str(ROOT.parents[2])))
SRC_ROOT = PCRL / '.worktrees' / 'pcrl-acs-protected-tree-replication-v1'      # source worktree (read-only)
SRC_PRIV = SRC_ROOT / '.private' / 'acs_replication'
HYB_ROOT = PCRL / '.worktrees' / 'pcrl-acs-frozen-hybrids-v1'                 # frozen-hybrids worktree (read-only)
HYB_PRIV = HYB_ROOT / '.private' / 'acs_frozen_hybrids'
SRC_RESULTS = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'      # inherited public records
HYB_RESULTS = ROOT / 'results' / 'pcrl_acs_frozen_hybrids_v1'
ACCESS_LOG = PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl'
SRC_ACCESS_LOG = SRC_PRIV / 'access' / 'DEV_REPORT_ACCESS.jsonl'

TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
SEEDS = (0, 1)
ROLES = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
ADMITTED = {'MODEL_FIT': ('features', 'income', 'occupation', 'sex', 'raw_inputs'),
            'CALIBRATION': ('features', 'income', 'occupation'),
            'MODEL_SELECT': ('features', 'income', 'occupation', 'sex'),
            'ATTACK_FIT': ('features', 'sex'),
            'ATTACK_VALIDATION': ('features', 'sex'),
            'DEV_REPORT': ('features', 'raw_inputs')}
FIELDS = {'features': ('row_id', 'unit', 'X', 'feature_names'), 'income': ('row_id', 'unit', 'income'),
          'occupation': ('row_id', 'unit', 'occupation'), 'sex': ('row_id', 'unit', 'sex'), 'weights': ('row_id', 'unit', 'pwgtp')}
RUN_FILES = [f'runs/{d}_{s}/{f}' for s in SEEDS for d, f in (('STAGE_A', 'STAGE_A_OUTPUTS.npz'), ('STAGE_A', 'STAGE_A_SEED.json'),
                                                             ('STAGE_A', 'U0_MODEL.pt'), ('STAGE_B', 'STAGE_B_OUTPUTS_E20.npz'),
                                                             ('STAGE_B', 'STAGE_B_SEED.json'), ('NEURAL_RAW', 'NEURAL_RAW_OUTPUTS.npz'),
                                                             ('NEURAL_RAW', 'NEURAL_RAW_SEED.json'))]


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def arr_sha(a):
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def wire_sha(p):
    """Hash of one recipient's released wire: float64 probability vector and int64 argmax decision."""
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes()); h.update(np.ascontiguousarray(p.argmax(1).astype(np.int64)).tobytes())
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def committed(path):
    """True iff `path` is tracked, committed and clean in this worktree."""
    p = Path(path)
    if not p.is_file():
        return False
    out = subprocess.run(['git', '-C', str(ROOT), 'log', '-1', '--format=%H', '--', str(p)], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', str(p)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not dirty.stdout.strip()


def public_export_sha():
    return json.loads((SRC_RESULTS / 'EXTRACTION_SUMMARY.json').read_text())['export_file_sha256']


def restore_receipt():
    return json.loads((PRIV / 'RESTORE_RECEIPT.json').read_text())


def _npz(path, expected_sha, fields):
    if sha(path) != expected_sha:
        raise PermissionError(f'hash differs: {path}')
    with ZipFile(path) as z:
        if {x.filename for x in z.infolist()} != {f + '.npy' for f in fields}:
            raise PermissionError('unexpected export fields: ' + path.name)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in fields}


def load_role(role, wanted=('features',)):
    """Restored, hash-checked development exports under this campaign's role-access rules."""
    if role == 'SEALED_ASSESSMENT' or role not in ADMITTED:
        raise PermissionError(f'role {role} is not admitted (the sealed assessment stays closed)')
    if set(wanted) - set(ADMITTED[role]):
        raise PermissionError(f'{role}: fields {sorted(set(wanted) - set(ADMITTED[role]))} are not admitted through load_role')
    pub = public_export_sha()[role]
    out = _npz(SRCP / 'exports' / role / 'features.npz', pub['features'], FIELDS['features'])
    out['feature_names'] = tuple(map(str, out['feature_names']))
    for name in wanted:
        if name in ('features', 'raw_inputs'):
            continue
        z = _npz(SRCP / 'exports' / role / f'{name}.npz', pub[name], FIELDS[name])
        if not (np.array_equal(z['row_id'], out['row_id']) and np.array_equal(z['unit'], out['unit'])):
            raise PermissionError(f'{role} {name} pairing differs')
        out[name] = z[name].astype(np.int64)
    if 'raw_inputs' in wanted:
        p = SRCP / 'exports' / role / 'raw_inputs.npz'
        if sha(p) != pub['raw_inputs']:
            raise PermissionError('raw_inputs hash differs')
        with np.load(p, allow_pickle=False) as z:
            out['raw_inputs'] = {k: z[k].copy() for k in z.files}
    return out


def dev_report_truth(purpose):
    """Re-read the ALREADY-OPENED DEV_REPORT truth (labels, SEX) for this campaign's frozen final comparison only."""
    lock = RESULTS / 'REPORTING_LOCK.json'
    if not committed(lock):
        raise PermissionError("DEV_REPORT truth re-read requires this campaign's committed REPORTING_LOCK")
    pub = public_export_sha()['DEV_REPORT']
    # Truth files live where the source extractor wrote them (exports/DEV_REPORT_TRUTH/, per the source extraction receipt).
    # Repair found by the independent verifier (C1): existence and published hashes are checked BEFORE any log event is
    # written, so a missing or altered file can no longer produce a false re-read record.
    src_files = json.loads((SRC_PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())['files']['DEV_REPORT']
    truth = {name: SRC_PRIV / Path(src_files[name]['path']).relative_to('.private/acs_replication') for name in ('income', 'occupation', 'sex')}
    for name, p in truth.items():
        if not p.is_file() or sha(p) != pub[name]:
            raise PermissionError(f'DEV_REPORT {name} truth file missing or hash differs from the public extraction summary')
    event = {'event': 'DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP', 'campaign': 'pcrl_scm_engineering_benchmark_v1', 'utc': now(),
             'purpose': purpose, 'reporting_lock_sha256': sha(lock), 'unit_id': os.environ.get('SCM_UNIT_ID'),
             'files_sha256': {k: pub[k] for k in ('income', 'occupation', 'sex')},
             'note': 'already-exposed development role; not a first opening; sealed assessment untouched'}
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for log in (ACCESS_LOG, SRC_ACCESS_LOG):
        with open(log, 'a') as f:
            f.write(json.dumps(event, sort_keys=True) + '\n')
    out = _npz(SRCP / 'exports' / 'DEV_REPORT' / 'features.npz', pub['features'], FIELDS['features'])
    for name in ('income', 'occupation', 'sex'):
        z = _npz(truth[name], pub[name], FIELDS[name])
        if not (np.array_equal(z['row_id'], out['row_id']) and np.array_equal(z['unit'], out['unit'])):
            raise PermissionError(f'DEV_REPORT {name} pairing differs')
        out[name] = z[name].astype(np.int64)
    return out


def source_outputs(seed, roles=ROLES, baselines=False):
    """Stored source outputs for one seed (restored, hash-checked): {service: {role: {task: probs}}} and row ids.
    baselines=True adds the utility-ineligible published FARE and FAGTB outputs (diagnostic only)."""
    rr = restore_receipt()
    out, rows = {}, {}
    rels = [f'runs/STAGE_A_{seed}/STAGE_A_OUTPUTS.npz', f'runs/STAGE_B_{seed}/STAGE_B_OUTPUTS_E20.npz',
            f'runs/NEURAL_RAW_{seed}/NEURAL_RAW_OUTPUTS.npz']
    if baselines:
        rels += [f'runs/FARE_{seed}/FARE_OUTPUTS.npz', f'runs/FAGTB_{seed}/FAGTB_OUTPUTS.npz']
    for rel in rels:
        p = SRCP / rel
        if sha(p) != rr['files'][rel]['sha256']:
            raise PermissionError(f'restored output hash differs: {rel}')
        with np.load(p) as z:
            for key in z.files:
                parts = key.split('__')
                if parts[0] in ('row_id', 'unit'):
                    rows[f'{rel}|{key}'] = z[key].copy(); continue
                svc, role, task = parts
                if role in roles:
                    out.setdefault(svc, {}).setdefault(role, {})[task] = np.asarray(z[key], dtype=np.float64)
    return out, rows


def frozen_bank(seed, roles=('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'MODEL_SELECT')):
    """The frozen-hybrids bank (H_LL, H_LJ, H_JL, H_JJ, NEURAL_RAW, TASK_REFIT, T_TREE, U0, Ucal), hash-checked."""
    man = json.loads((HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())
    p = SRCP / 'bank' / f'FROZEN_BANK_S{seed}.npz'
    if sha(p) != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('frozen bank hash differs from FROZEN_BANK_MANIFEST')
    out, ids = {}, {}
    with np.load(p) as z:
        for key in z.files:
            parts = key.split('__')
            if parts[0] in ('row_id', 'unit'):
                ids[key] = z[key].copy(); continue
            svc, role, task = parts
            if role in roles:
                out.setdefault(svc, {}).setdefault(role, {})[task] = z[key].copy()
    return out, ids


def task_vectors(y, p, k):
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), k) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
        raise ValueError('probability convention differs')
    target = y[:, None] == np.arange(k)[None, :]
    return {'accuracy': (p.argmax(1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
            'brier': ((p - target) ** 2).sum(1)}
