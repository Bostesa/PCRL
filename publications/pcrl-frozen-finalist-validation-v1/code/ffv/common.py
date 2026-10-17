"""Shared definitions for the PCRL frozen-finalist validation and confirmation-readiness study (DEVELOPMENT ROBUSTNESS
EVIDENCE on already-used records; not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, in the development roles of the
protected-tree replication. The service predicts each person's income class (2 classes: PINCP x ADJINC / 10^6 above $50,000) and
occupation group (5 classes: OCCP mapped to the five civilian broad groups) from six permitted raw fields (AGEP, WKHP, COW, SCHL,
MAR, POBP). Each recipient receives only its own probability vector and argmax decision. SEX is the dataset's recorded
demographic column; separate evaluation classifiers ("readers") try to infer SEX from the released predictions: each output
alone and the two outputs aligned to the same record (the linked pair).

The frozen candidate is the source campaign's finalist B3CTL_UNCOUPLED_MIX50_C6A (research/pcrl-autonomous-method-development-v1
@ 9e2b6c567). Nothing here refits, retunes or reselects a release. Source artifacts are admitted read-only, by hash, into
.private/pcrl_frozen_finalist_validation_v1/admitted/amd/<source-relative path> by `ffv.admit`.

Role access (PROTOCOL.md section 3):
- ATTACK_FIT (features, SEX): reader fitting only (8,001 people). ATTACK_VALIDATION (features, SEX): reader selection and
  orientation only (3,000 people). Every SEX read is logged.
- DEV_REPORT (features, raw inputs): label-free. Truth only through `dev_report_truth` after this campaign's committed
  REPORTING_LOCK: one coordinator evaluation and one independent-verifier replay, under NEW follow-up event names. The source
  campaign's access logs and one-use counters are preserved and never reset.
- MODEL_FIT, CALIBRATION, MODEL_SELECT: features, raw inputs and household ids only (output-mapping and exposure checks).
  No task label or SEX column of these roles is admitted.
- SEALED_ASSESSMENT: refused by every loader. No new population, download or label is opened.
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
CAMPAIGN = 'pcrl_frozen_finalist_validation_v1'
PRIV = ROOT / '.private' / CAMPAIGN
ADM = PRIV / 'admitted' / 'amd'              # admitted copies of the source campaign's private files (source-relative paths)
RUNS = PRIV / 'runs'
RESULTS = ROOT / 'results' / CAMPAIGN
SRC_CAMPAIGN = 'pcrl_autonomous_method_development_v1'
SRC_RESULTS = ROOT / 'results' / SRC_CAMPAIGN                       # the source campaign's committed public records
PCRL = ROOT.parents[2]
SRC_WORKTREE = PCRL / '.claude' / 'worktrees' / 'pcrl-autonomous-method-development-v1'
SRC_PRIV = SRC_WORKTREE / '.private' / SRC_CAMPAIGN                  # read-only; used by `ffv.admit` only
SRC_TIP = '9e2b6c567d7d10dbfae75133e8def490556360b6'
REP_PRIV = SB.SRC_PRIV                                               # protected-tree replication private tree (DEV_REPORT truth)
SRC_ACCESS_LOG = SB.SRC_ACCESS_LOG                                   # lineage-wide DEV_REPORT access log (append-only)
ACCESS_LOG = PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl'
LABEL_LOG = PRIV / 'access' / 'ATTACK_LABEL_READS.jsonl'

TASKS, K, CAPS, SEEDS = SB.TASKS, SB.K, SB.CAPS, SB.SEEDS
GUARD = 0.005
RAW_FIELDS = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
ROLES3 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
ADMITTED = {'MODEL_FIT': ('features', 'raw_inputs'), 'CALIBRATION': ('features',), 'MODEL_SELECT': ('features',),
            'ATTACK_FIT': ('features', 'sex'), 'ATTACK_VALIDATION': ('features', 'sex'), 'DEV_REPORT': ('features', 'raw_inputs')}
# the compact bank (brief section 3): at most six effective releases per seed, source roles and source eligibility labels
BANK = ('FINALIST', 'NEURAL_RAW', 'H_LJ', 'TASKONLY', 'B2TWIN', 'PUB_IBSI')
ELIGIBLE_REFERENCES = ('NEURAL_RAW', 'H_LJ')
DEV_EVENTS = ('DEV_REPORT_TRUTH_REREAD_BY_FFV_COORDINATOR', 'DEV_REPORT_TRUTH_REREAD_BY_FFV_INDEPENDENT_VERIFIER')

sha, wire_sha, task_vectors, public_export_sha = SB.sha, SB.wire_sha, SB.task_vectors, SB.public_export_sha


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def committed(path):
    return SB.committed(path)


def admission():
    return json.loads((RESULTS / 'SOURCE_ADMISSION.json').read_text())


def admitted(rel):
    """Path of an admitted private copy after re-checking its hash against this campaign's SOURCE_ADMISSION.json."""
    rec = admission()['copies'][rel]
    p = ADM / rel
    if sha(p) != rec['sha256']:
        raise PermissionError(f'admitted copy hash differs: {rel}')
    return p


def _log_labels(role, purpose):
    LABEL_LOG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(LABEL_LOG, 'a') as f:
        f.write(json.dumps({'utc': now(), 'role': role, 'field': 'sex', 'purpose': purpose, 'unit_id': os.environ.get('FFV_UNIT_ID')},
                           sort_keys=True) + '\n')


def load_role(role, wanted=('features',), purpose=None):
    """Admitted, hash-checked development exports under this campaign's role-access rules."""
    if role == 'SEALED_ASSESSMENT' or role not in ADMITTED:
        raise PermissionError(f'role {role} is not admitted (the sealed assessment stays closed)')
    if set(wanted) - set(ADMITTED[role]):
        raise PermissionError(f'{role}: fields {sorted(set(wanted) - set(ADMITTED[role]))} are not admitted in this campaign')
    pub = public_export_sha()[role]
    out = SB._npz(admitted(f'admitted/exports/{role}/features.npz'), pub['features'], SB.FIELDS['features'])
    out['feature_names'] = tuple(map(str, out['feature_names']))
    if 'sex' in wanted:
        if not purpose:
            raise PermissionError('SEX reads require a logged purpose')
        _log_labels(role, purpose)
        z = SB._npz(admitted(f'admitted/exports/{role}/sex.npz'), pub['sex'], SB.FIELDS['sex'])
        if not (np.array_equal(z['row_id'], out['row_id']) and np.array_equal(z['unit'], out['unit'])):
            raise PermissionError(f'{role} sex pairing differs')
        out['sex'] = z['sex'].astype(np.int64)
    if 'raw_inputs' in wanted:
        p = admitted(f'admitted/exports/{role}/raw_inputs.npz')
        if sha(p) != pub['raw_inputs']:
            raise PermissionError('raw_inputs hash differs')
        with np.load(p, allow_pickle=False) as z:
            out['raw_inputs'] = {k: z[k].copy() for k in z.files}
    return out


def _bank_npz(p):
    out, ids = {}, {}
    with np.load(p) as z:
        for key in z.files:
            parts = key.split('__')
            if parts[0] in ('row_id', 'unit'):
                ids[key] = z[key].copy(); continue
            svc, role, task = parts
            out.setdefault(svc, {}).setdefault(role, {})[task] = z[key].copy()
    return out, ids


def service_bank(seed, services=BANK):
    """The source campaign's frozen final bank (hash-checked against its SERVICE_BANK_MANIFEST), restricted to this campaign's
    six releases. Probability vectors are the exact float64 arrays the recipients received; decisions are their argmax."""
    man = json.loads((SRC_RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    p = admitted(f'bank/SERVICE_BANK_S{seed}.npz')
    if sha(p) != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('service bank hash differs from the source SERVICE_BANK_MANIFEST')
    out, ids = _bank_npz(p)
    return {s: out[s] for s in services}, ids, man


def frozen_bank(seed, roles=('DEV_REPORT',)):
    """The frozen-hybrids bank (U0 and Ucal utility references), hash-checked against FROZEN_BANK_MANIFEST."""
    man = json.loads((SB.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())
    p = admitted(f'admitted/bank/FROZEN_BANK_S{seed}.npz')
    if sha(p) != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('frozen bank hash differs from FROZEN_BANK_MANIFEST')
    out, ids = _bank_npz(p)
    return {s: {r: v for r, v in d.items() if r in roles} for s, d in out.items()}, ids


def source_readers(seed):
    """The source campaign's fitted reader record and stored DEV_REPORT reader scores (hash-checked against its REPORTING_LOCK)."""
    rl = json.loads((SRC_RESULTS / 'REPORTING_LOCK.json').read_text())['readers'][f'S{seed}']
    rp, sp = admitted(f'runs/READERS_{seed}/READERS_SEED.json'), admitted(f'runs/READERS_{seed}/DEV_REPORT_SCORES.npz')
    if sha(rp) != rl['readers_seed_sha256'] or sha(sp) != rl['dev_report_scores_sha256']:
        raise PermissionError('source reader record or scores differ from the source REPORTING_LOCK')
    with np.load(sp) as z:
        scores = {k: z[k].copy() for k in z.files}
    return json.loads(rp.read_text()), scores


def dev_access_events():
    if not ACCESS_LOG.is_file():
        return []
    return [json.loads(l) for l in ACCESS_LOG.read_text().splitlines() if l.strip()]


def dev_report_truth(purpose, event_name):
    """Re-read the ALREADY-OPENED DEV_REPORT truth for this campaign's frozen evaluation only. Two registered follow-up events
    (coordinator evaluation, verifier replay), each at most once, after this campaign's committed REPORTING_LOCK. The event is
    appended to this campaign's log and to the lineage-wide replication log; no earlier log line is changed."""
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
             'unit_id': os.environ.get('FFV_UNIT_ID'), 'files_sha256': {k: pub[k] for k in ('income', 'occupation', 'sex')},
             'note': 'follow-up development robustness evaluation of an already-exposed role; not a first opening; sealed assessment untouched'}
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for log in (ACCESS_LOG, SRC_ACCESS_LOG):
        with open(log, 'a') as f:
            f.write(json.dumps(event, sort_keys=True) + '\n')
    out = SB._npz(admitted('admitted/exports/DEV_REPORT/features.npz'), pub['features'], SB.FIELDS['features'])
    for name in ('income', 'occupation', 'sex'):
        z = SB._npz(truth[name], pub[name], SB.FIELDS[name])
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


def file_sha_tree(d):
    d = Path(d)
    return {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob('*')) if p.is_file()}


def household_folds(unit, k, tag):
    groups = np.unique(unit)
    keys = [hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest() for g in groups]
    order = np.argsort(keys)
    fold_of_group = np.empty(len(groups), dtype=np.int64)
    fold_of_group[order] = np.arange(len(groups)) % k
    return fold_of_group[np.searchsorted(groups, unit)]
