"""Shared definitions for the ACS frozen-hybrids follow-up (EXPLORATORY DEVELOPMENT FOLLOW-UP; not confirmation).

Source: publication f4ea199 (pcrl-acs-protected-tree-replication-v1) and its private artifacts, restored read-only into
.private/acs_frozen_hybrids/source with sha256 checks against the published records. No defense model is fitted here.

Guards:
- SEALED_ASSESSMENT has no export and is refused by every loader.
- DEV_REPORT truth (labels, SEX, weights) is read only through `dev_report_truth`, which requires THIS campaign's committed
  REPORTING_LOCK and appends a logged re-read event to this campaign's access log and to the source access log. The source's
  one-time-opening receipt is never modified or reset, and no guard protecting another role is bypassed.
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
PRIV = ROOT / '.private' / 'acs_frozen_hybrids'
SRCP = PRIV / 'source'                     # restored source artifacts, same relative layout as the source private root
RESULTS = ROOT / 'results' / 'pcrl_acs_frozen_hybrids_v1'
SRC_RESULTS = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'   # inherited; byte-identical to the pinned publication
SRC_ROOT = ROOT.parent / 'pcrl-acs-protected-tree-replication-v1'           # source worktree (read-only)
SRC_PRIV = SRC_ROOT / '.private' / 'acs_replication'
PUBLISHED_SOURCE_COMMIT = 'f4ea19982bce3dc3c88c5951ad34c4cf84620a20'
PUBLISHED_SOURCE_PATH = 'publications/pcrl-acs-protected-tree-replication-v1'
ACCESS_LOG = PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl'
SRC_ACCESS_LOG = SRC_PRIV / 'access' / 'DEV_REPORT_ACCESS.jsonl'

TASKS = ('income', 'occupation')
K = {'income': 2, 'occupation': 5}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
SEEDS = (0, 1)
BANK_ROLES = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'MODEL_SELECT')

# The fixed four-combination bank: (income source, occupation source). One global choice, never per person.
COMPONENTS = {'H_LL': ('LOCAL_TREE', 'LOCAL_TREE'), 'H_LJ': ('LOCAL_TREE', 'JOINT_TREE'),
              'H_JL': ('JOINT_TREE', 'LOCAL_TREE'), 'H_JJ': ('JOINT_TREE', 'JOINT_TREE')}
PRIMARY = 'H_LJ'
# External controls: bank label -> source service label (selected NEURAL_RAW, matched TASK_REFIT E20, diagnostic teacher).
CONTROLS = {'NEURAL_RAW': 'NEURAL_RAW_rho1.0_E2', 'TASK_REFIT': 'TASK_REFIT', 'T_TREE': 'T_TREE'}
REFERENCES = ('U0', 'Ucal')
ATTACK_ROSTER = ('H_LL', 'H_LJ', 'H_JL', 'H_JJ', 'NEURAL_RAW', 'TASK_REFIT', 'T_TREE')
SOURCE_OF = {**{h: c for h, c in COMPONENTS.items()}, **{b: (s, s) for b, s in CONTROLS.items()}}   # per-task source service
OUTPUT_FILES = {'stage_a': 'runs/STAGE_A_{s}/STAGE_A_OUTPUTS.npz', 'stage_b': 'runs/STAGE_B_{s}/STAGE_B_OUTPUTS_E20.npz',
                'neural_raw': 'runs/NEURAL_RAW_{s}/NEURAL_RAW_OUTPUTS.npz'}
SERVICE_FILE = {'T_TREE': 'stage_a', 'U0': 'stage_a', 'Ucal': 'stage_a', 'LOCAL_TREE': 'stage_b', 'JOINT_TREE': 'stage_b',
                'TASK_REFIT': 'stage_b', 'NEURAL_RAW_rho1.0_E2': 'neural_raw'}


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


def receipt():
    return json.loads((SRCP / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())


def restore_receipt():
    return json.loads((PRIV / 'RESTORE_RECEIPT.json').read_text())


def _npz(path, expected_sha, fields):
    if sha(path) != expected_sha:
        raise PermissionError(f'hash differs: {path.name}')
    with ZipFile(path) as z:
        if {x.filename for x in z.infolist()} != {f + '.npy' for f in fields}:
            raise PermissionError('unexpected export fields: ' + path.name)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in fields}


FIELDS = {'features': ('row_id', 'unit', 'X', 'feature_names'), 'income': ('row_id', 'unit', 'income'),
          'occupation': ('row_id', 'unit', 'occupation'), 'sex': ('row_id', 'unit', 'sex'), 'weights': ('row_id', 'unit', 'pwgtp'),
          'raw_inputs': None}
LABEL_FREE = {'ATTACK_FIT': ('features', 'sex'), 'ATTACK_VALIDATION': ('features', 'sex'),
              'DEV_REPORT': ('features', 'raw_inputs'), 'MODEL_SELECT': ('features', 'income', 'occupation')}


def load_role(role, wanted=('features',)):
    """Restored, hash-checked development exports. DEV_REPORT truth is NOT available here (see dev_report_truth)."""
    if role == 'SEALED_ASSESSMENT' or role not in LABEL_FREE:
        raise PermissionError(f'role {role} is not admitted in this campaign (the sealed assessment stays closed)')
    if set(wanted) - set(LABEL_FREE[role]):
        raise PermissionError(f'{role}: fields {sorted(set(wanted) - set(LABEL_FREE[role]))} are not admitted through load_role')
    files = receipt()['files'][role]
    def path(f):
        return SRCP / Path(files[f]['path']).relative_to('.private/acs_replication')
    out = _npz(path('features'), files['features']['sha256'], FIELDS['features'])
    out['feature_names'] = tuple(map(str, out['feature_names']))
    for name in wanted:
        if name in ('features', 'raw_inputs'):
            continue
        z = _npz(path(name), files[name]['sha256'], FIELDS[name])
        if not (np.array_equal(z['row_id'], out['row_id']) and np.array_equal(z['unit'], out['unit'])):
            raise PermissionError(f'{role} {name} pairing differs')
        out[name] = z[name].astype(np.int64)
    if 'raw_inputs' in wanted:
        p = path('raw_inputs')
        if sha(p) != files['raw_inputs']['sha256']:
            raise PermissionError('raw_inputs hash differs')
        with np.load(p, allow_pickle=False) as z:
            out['raw_inputs'] = {k: z[k].copy() for k in z.files}
    return out


def dev_report_truth(purpose):
    """Re-read the ALREADY-OPENED DEV_REPORT truth (labels, SEX, weights) for this follow-up's reporting step only.

    Requires this campaign's committed REPORTING_LOCK. Bound to the original export hashes (source extraction receipt, equal
    to the public EXTRACTION_SUMMARY). Appends a re-read event to this campaign's access log and to the source access log;
    the source's one-time-opening receipt is never modified, reset or deleted."""
    lock = RESULTS / 'REPORTING_LOCK.json'
    if not committed(lock):
        raise PermissionError('DEV_REPORT truth re-read requires this campaign\'s committed REPORTING_LOCK')
    files = receipt()['files']['DEV_REPORT']
    event = {'event': 'DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP', 'campaign': 'pcrl_acs_frozen_hybrids_v1', 'utc': now(),
             'purpose': purpose, 'reporting_lock_sha256': sha(lock), 'unit_id': os.environ.get('ACSH_UNIT_ID'),
             'files_sha256': {k: files[k]['sha256'] for k in ('income', 'occupation', 'sex', 'weights')},
             'note': 'already-exposed development role; not a first opening; sealed assessment untouched'}
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for log in (ACCESS_LOG, SRC_ACCESS_LOG):
        with open(log, 'a') as f:
            f.write(json.dumps(event, sort_keys=True) + '\n')
    out = _npz(SRCP / Path(files['features']['path']).relative_to('.private/acs_replication'), files['features']['sha256'], FIELDS['features'])
    for name in ('income', 'occupation', 'sex', 'weights'):
        z = _npz(SRC_PRIV / Path(files[name]['path']).relative_to('.private/acs_replication'), files[name]['sha256'], FIELDS[name])
        if not (np.array_equal(z['row_id'], out['row_id']) and np.array_equal(z['unit'], out['unit'])):
            raise PermissionError(f'DEV_REPORT {name} pairing differs')
        out[name] = z['pwgtp' if name == 'weights' else name].astype(np.int64)
    return out


def source_outputs(seed, roles=BANK_ROLES):
    """Stored source outputs for one seed from the restored, hash-checked files: {service: {role: {task: probs}}} + row ids."""
    rr = restore_receipt()
    out, rows = {}, {}
    for fam, pat in OUTPUT_FILES.items():
        rel = pat.format(s=seed); p = SRCP / rel
        if sha(p) != rr['files'][rel]['sha256']:
            raise PermissionError(f'restored output hash differs: {rel}')
        with np.load(p) as z:
            for key in z.files:
                parts = key.split('__')
                if parts[0] in ('row_id', 'unit'):
                    rows.setdefault(fam, {})[key] = z[key].copy(); continue
                svc, role, task = parts
                if role in roles:
                    out.setdefault(svc, {}).setdefault(role, {})[task] = z[key].copy()
    return out, rows


def task_vectors(y, p, k):
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    if p.shape != (len(y), k) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
        raise ValueError('probability convention differs')
    target = y[:, None] == np.arange(k)[None, :]
    return {'accuracy': (p.argmax(1) == y).astype(np.float64),
            'nll': -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)),
            'brier': ((p - target) ** 2).sum(1)}


def bank_path(seed):
    return PRIV / 'bank' / f'FROZEN_BANK_S{seed}.npz'


def load_bank(seed, roles=BANK_ROLES):
    """The hashed four-combination bank (+ controls and references): {service: {role: {task: probs}}}, row ids, units."""
    man = json.loads((RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())
    p = bank_path(seed)
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
