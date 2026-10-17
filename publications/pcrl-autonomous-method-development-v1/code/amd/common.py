"""Shared definitions for the PCRL autonomous method-development campaign (EXPLORATORY DEVELOPMENT; not confirmation).

Census context: Florida ACS 2019 public-use person records (development roles of the protected-tree replication). The service
predicts each person's income class (2 classes) and occupation group (5 classes) from six raw fields (AGEP, WKHP, COW, SCHL,
MAR, POBP) and gives each recipient only its own probability vector and argmax decision. Evaluation readers try to infer the
recorded SEX column from those released outputs (each output alone and the two outputs aligned to the same record).

Sources are admitted read-only into .private/pcrl_autonomous_method_development_v1/admitted by `amd.admit`, each copy checked
against a hash recorded by its own manifest. Role access (PROTOCOL.md section 3):
- MODEL_FIT (features, task labels, SEX): fitting of release mechanisms and task models; household-separated internal folds.
- CALIBRATION (features, task labels): calibrators only.
- MODEL_SELECT (features, task labels, SEX): development evaluation, reader proxies and selection feedback. Every read that
  returns MODEL_SELECT labels is appended to the MODEL_SELECT use log. Never used as gradient-fitting labels.
- ATTACK_FIT / ATTACK_VALIDATION (features, SEX): final readers only, after the finalist freeze.
- DEV_REPORT (features, raw inputs): label-free; truth only through `dev_report_truth` after this campaign's committed and
  published REPORTING_LOCK (one coordinator evaluation, one independent-verifier replay; nothing else).
- SEALED_ASSESSMENT: refused by every loader; other projects' reserved records are never read.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scm_bench import common as SB          # source definitions (constants, hash helpers, public export hashes)

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = 'pcrl_autonomous_method_development_v1'
PRIV = ROOT / '.private' / CAMPAIGN
SRCP = PRIV / 'admitted'
RUNS = PRIV / 'runs'
RESULTS = ROOT / 'results' / CAMPAIGN
PCRL = ROOT.parents[2]
UM_PRIV = PCRL / '.claude' / 'worktrees' / 'pcrl-scm-engineering-benchmark-v1' / '.private' / 'pcrl_scm_utility_matched_v1'
UM_RESULTS = ROOT / 'results' / 'pcrl_scm_utility_matched_v1'
REP_PRIV = SB.SRC_PRIV                      # protected-tree replication private tree (read-only)
REP_RESULTS = SB.SRC_RESULTS
HYB_RESULTS = SB.HYB_RESULTS
ACCESS_LOG = PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl'
MS_LOG = PRIV / 'access' / 'MODEL_SELECT_USE.jsonl'
SRC_ACCESS_LOG = SB.SRC_ACCESS_LOG

TASKS, K, CAPS, SEEDS, FIELDS = SB.TASKS, SB.K, SB.CAPS, SB.SEEDS, SB.FIELDS
GUARD = 0.005                               # within-SEX occupation summed-Brier excess vs U0, each SEX group
RAW_FIELDS = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')
ADMITTED = {'MODEL_FIT': ('features', 'income', 'occupation', 'sex', 'raw_inputs'),
            'CALIBRATION': ('features', 'income', 'occupation'),
            'MODEL_SELECT': ('features', 'income', 'occupation', 'sex'),
            'ATTACK_FIT': ('features', 'sex'),
            'ATTACK_VALIDATION': ('features', 'sex'),
            'DEV_REPORT': ('features', 'raw_inputs')}
DEV_EVENTS = ('DEV_REPORT_TRUTH_REREAD_BY_AMD_COORDINATOR', 'DEV_REPORT_TRUTH_REREAD_BY_AMD_INDEPENDENT_VERIFIER')
SELECT_LABEL_ROLES = ('MODEL_SELECT',)

sha, arr_sha, wire_sha, task_vectors, public_export_sha = SB.sha, SB.arr_sha, SB.wire_sha, SB.task_vectors, SB.public_export_sha


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def committed(path):
    return SB.committed(path)


def admission():
    return json.loads((RESULTS / 'SOURCE_ADMISSION.json').read_text())


def admitted(rel):
    """Path of an admitted private copy after re-checking its hash against SOURCE_ADMISSION.json."""
    rec = admission()['copies'][rel]
    p = SRCP / rel
    if sha(p) != rec['sha256']:
        raise PermissionError(f'admitted copy hash differs: {rel}')
    return p


def _log_ms(purpose):
    MS_LOG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(MS_LOG, 'a') as f:
        f.write(json.dumps({'utc': now(), 'purpose': purpose, 'unit_id': os.environ.get('AMD_UNIT_ID')}, sort_keys=True) + '\n')


def load_role(role, wanted=('features',), purpose=None):
    """Admitted, hash-checked development exports under this campaign's role-access rules."""
    if role == 'SEALED_ASSESSMENT' or role not in ADMITTED:
        raise PermissionError(f'role {role} is not admitted (the sealed assessment stays closed)')
    if set(wanted) - set(ADMITTED[role]):
        raise PermissionError(f'{role}: fields {sorted(set(wanted) - set(ADMITTED[role]))} are not admitted')
    labels = set(wanted) & {'income', 'occupation', 'sex'}
    if role in SELECT_LABEL_ROLES and labels:
        if not purpose:
            raise PermissionError('MODEL_SELECT labels require a logged purpose')
        _log_ms(f'{purpose} | fields {sorted(labels)}')
    pub = public_export_sha()[role]
    out = SB._npz(SRCP / 'exports' / role / 'features.npz', pub['features'], FIELDS['features'])
    out['feature_names'] = tuple(map(str, out['feature_names']))
    for name in wanted:
        if name in ('features', 'raw_inputs'):
            continue
        z = SB._npz(SRCP / 'exports' / role / f'{name}.npz', pub[name], FIELDS[name])
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


def frozen_bank(seed, roles=('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')):
    """Frozen-hybrids bank (H_LL, H_LJ, H_JL, H_JJ, NEURAL_RAW, TASK_REFIT, T_TREE, U0, Ucal), hash-checked."""
    man = json.loads((HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())
    p = admitted(f'bank/FROZEN_BANK_S{seed}.npz')
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


def um_service_bank(seed, roles=('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')):
    """The SCM utility-matched follow-up's frozen service bank (SCM, PUB_IBSI, ORD, SHUF, C0 ... releases), hash-checked."""
    p = admitted(f'um_bank/SERVICE_BANK_S{seed}.npz')
    out = {}
    with np.load(p) as z:
        for key in z.files:
            parts = key.split('__')
            if len(parts) != 3:
                continue
            svc, role, task = parts
            if role in roles:
                out.setdefault(svc, {}).setdefault(role, {})[task] = z[key].copy()
    return out


def teacher(seed, task):
    """The replication's frozen CatBoost teacher (T_TREE) for one task, as the routed tree structure (acs_replication.trees)."""
    from acs_replication import trees
    return trees.TreeTeacher(json.loads(admitted(f'stage_a/S{seed}/TREE_TEACHER_{task}.json').read_text()))


def stage_a_outputs(seed):
    with np.load(admitted(f'stage_a/S{seed}/STAGE_A_OUTPUTS.npz')) as z:
        return {k: z[k].copy() for k in z.files}


def stage_b_theta(seed, arm, label='E20'):
    with np.load(admitted(f'stage_b/S{seed}/{arm}__{label}.npz')) as z:
        return [z['theta_income'].copy(), z['theta_occupation'].copy()]


def household_folds(unit, k, tag):
    """Deterministic household-separated fold id per person (label-free; SHA-256 order of household ids)."""
    groups = np.unique(unit)
    keys = [hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest() for g in groups]
    order = np.argsort(keys)
    fold_of_group = np.empty(len(groups), dtype=np.int64)
    fold_of_group[order] = np.arange(len(groups)) % k
    return fold_of_group[np.searchsorted(groups, unit)]


def dev_access_events():
    if not ACCESS_LOG.is_file():
        return []
    return [json.loads(l) for l in ACCESS_LOG.read_text().splitlines() if l.strip()]


def dev_report_truth(purpose, event_name):
    """Re-read the ALREADY-OPENED DEV_REPORT truth for this campaign's frozen finalist evaluation only. Registered events: one
    coordinator evaluation and one independent-verifier replay; each at most once; requires the committed REPORTING_LOCK."""
    if event_name not in DEV_EVENTS:
        raise PermissionError('unregistered DEV_REPORT access event')
    if any(e.get('event') == event_name for e in dev_access_events()):
        raise PermissionError(f'{event_name} already used once in this campaign')
    lock = RESULTS / 'REPORTING_LOCK.json'
    if not committed(lock):
        raise PermissionError("DEV_REPORT truth re-read requires this campaign's committed REPORTING_LOCK")
    pub = public_export_sha()['DEV_REPORT']
    src_files = json.loads((REP_PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())['files']['DEV_REPORT']
    truth = {name: REP_PRIV / Path(src_files[name]['path']).relative_to('.private/acs_replication') for name in ('income', 'occupation', 'sex')}
    for name, p in truth.items():          # existence and published hashes are checked BEFORE any log event is written
        if not p.is_file() or sha(p) != pub[name]:
            raise PermissionError(f'DEV_REPORT {name} truth file missing or hash differs from the public extraction summary')
    event = {'event': event_name, 'campaign': CAMPAIGN, 'utc': now(), 'purpose': purpose, 'reporting_lock_sha256': sha(lock),
             'unit_id': os.environ.get('AMD_UNIT_ID'), 'files_sha256': {k: pub[k] for k in ('income', 'occupation', 'sex')},
             'note': 'already-exposed development role; not a first opening; sealed assessment untouched'}
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for log in (ACCESS_LOG, SRC_ACCESS_LOG):
        with open(log, 'a') as f:
            f.write(json.dumps(event, sort_keys=True) + '\n')
    out = SB._npz(SRCP / 'exports' / 'DEV_REPORT' / 'features.npz', pub['features'], FIELDS['features'])
    for name in ('income', 'occupation', 'sex'):
        z = SB._npz(truth[name], pub[name], FIELDS[name])
        if not (np.array_equal(z['row_id'], out['row_id']) and np.array_equal(z['unit'], out['unit'])):
            raise PermissionError(f'DEV_REPORT {name} pairing differs')
        out[name] = z[name].astype(np.int64)
    return out


def write_json_x(path, obj):
    """Write-once JSON (refuses to overwrite)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=float)


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float) + '\n')
