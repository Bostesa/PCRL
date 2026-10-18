"""Shared definitions for the PCRL income-headroom successor development campaign (EXPLORATORY DEVELOPMENT on previously used
records; not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, in the development roles of the
protected-tree replication. The service predicts each person's income class (2 classes: PINCP x ADJINC / 10^6 above $50,000) and
occupation group (5 classes: OCCP mapped to the five civilian broad groups) from six permitted raw fields (AGEP, WKHP, COW, SCHL,
MAR, POBP; admitted 145-column encoding). Each recipient receives only its own float64 probability vector and argmax decision.
SEX is the dataset's recorded demographic column; separate evaluation classifiers ("readers") estimate SEX from the released
probability vectors: each output alone and the two outputs aligned to the same record (the linked pair).

The retired candidate B3CTL_UNCOUPLED_MIX50_C6A (source research/pcrl-autonomous-method-development-v1 @ 9e2b6c567; validation
research/pcrl-frozen-finalist-validation-v1 @ 0be8c6a27) is a diagnostic baseline only. Source artifacts are admitted read-only,
by hash, into .private/pcrl_income_headroom_successor_v1/admitted/<path> by `ihs.admit`.

Role access (PROTOCOL.md section 3):
- MODEL_FIT (features, raw inputs, income, occupation, SEX): fitting release mechanisms and task models; household-separated
  internal folds for development diagnosis. SEX only as a training-time label, never a runtime input or an output.
- CALIBRATION (features, income, occupation): calibrators only (or a household-separated internal split for a calibrator).
  No SEX.
- MODEL_SELECT (features, income, occupation, SEX): development evaluation, reader proxies, adaptive selection and revision
  feedback. Every read that returns MODEL_SELECT labels is logged (MODEL_SELECT_USE.jsonl). Never gradient-fitting labels,
  never calibration labels, never used to fit release parameters.
- ATTACK_FIT / ATTACK_VALIDATION (features, SEX): final readers only, and only after this campaign's committed FINALIST_LOCK.
  Every SEX read is logged.
- DEV_REPORT (features, raw inputs): label-free. Truth only through `dev_report_truth` after this campaign's committed and
  published REPORTING_LOCK: one coordinator evaluation and one independent-verifier replay under NEW follow-up event names. The
  lineage's access logs and one-use counters are preserved and never reset.
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
CAMPAIGN = 'pcrl_income_headroom_successor_v1'
PRIV = ROOT / '.private' / CAMPAIGN
ADM = PRIV / 'admitted'
RUNS = PRIV / 'runs'
RESULTS = ROOT / 'results' / CAMPAIGN
PCRL = ROOT.parents[2]
AMD_CAMPAIGN = 'pcrl_autonomous_method_development_v1'
FFV_CAMPAIGN = 'pcrl_frozen_finalist_validation_v1'
AMD_RESULTS = ROOT / 'results' / AMD_CAMPAIGN
FFV_RESULTS = ROOT / 'results' / FFV_CAMPAIGN
AMD_PRIV = PCRL / '.claude' / 'worktrees' / 'pcrl-autonomous-method-development-v1' / '.private' / AMD_CAMPAIGN   # read-only
FFV_PRIV = PCRL / '.claude' / 'worktrees' / 'pcrl-frozen-finalist-validation-v1' / '.private' / FFV_CAMPAIGN     # read-only
AMD_TIP = '9e2b6c567d7d10dbfae75133e8def490556360b6'
FFV_TIP = '0be8c6a27cccd6f1aa53b76305166c2a83a8975e'
REP_PRIV = SB.SRC_PRIV                      # protected-tree replication private tree (DEV_REPORT truth; read-only)
SRC_ACCESS_LOG = SB.SRC_ACCESS_LOG          # lineage-wide DEV_REPORT access log (append-only)
ACCESS_LOG = PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl'
MS_LOG = PRIV / 'access' / 'MODEL_SELECT_USE.jsonl'
LABEL_LOG = PRIV / 'access' / 'ATTACK_LABEL_READS.jsonl'

TASKS, K, CAPS, SEEDS, FIELDS = SB.TASKS, SB.K, SB.CAPS, SB.SEEDS, SB.FIELDS
GUARD = 0.005                               # within-SEX occupation summed-Brier excess vs U0, each SEX group
RAW_FIELDS = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
ADMITTED = {'MODEL_FIT': ('features', 'income', 'occupation', 'sex', 'raw_inputs'),
            'CALIBRATION': ('features', 'income', 'occupation'),
            'MODEL_SELECT': ('features', 'income', 'occupation', 'sex'),
            'ATTACK_FIT': ('features', 'sex'),
            'ATTACK_VALIDATION': ('features', 'sex'),
            'DEV_REPORT': ('features', 'raw_inputs')}
ATTACK_ROLES = ('ATTACK_FIT', 'ATTACK_VALIDATION')
DEV_EVENTS = ('DEV_REPORT_TRUTH_REREAD_BY_IHS_COORDINATOR', 'DEV_REPORT_TRUTH_REREAD_BY_IHS_INDEPENDENT_VERIFIER')
UNIT_ENV = 'IHS_UNIT_ID'

sha, arr_sha, wire_sha, task_vectors, public_export_sha = SB.sha, SB.arr_sha, SB.wire_sha, SB.task_vectors, SB.public_export_sha


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


REGISTRATION_FILES = ('PROTOCOL.md', 'roles/statistical_reviewer/RESERVE_RULE.json', 'roles/statistical_reviewer/NOMINATION_RULE.md')


def registered():
    """True once the registration files are committed. No new release fit and no further development label read happens
    before that. (This gate was added at about 02:12Z, after the two pre-registration reference units IHS_REFERENCES_S0_V1 /
    S1_V1 had read MODEL_SELECT labels to evaluate historical releases only; no release was fitted before registration.)"""
    return all(committed(RESULTS / f) for f in REGISTRATION_FILES)


def _log(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path, 'a') as f:
        f.write(json.dumps({'utc': now(), 'unit_id': os.environ.get(UNIT_ENV), **rec}, sort_keys=True) + '\n')


def load_role(role, wanted=('features',), purpose=None):
    """Admitted, hash-checked development exports under this campaign's role-access rules."""
    if role == 'SEALED_ASSESSMENT' or role not in ADMITTED:
        raise PermissionError(f'role {role} is not admitted (the sealed assessment stays closed)')
    if set(wanted) - set(ADMITTED[role]):
        raise PermissionError(f'{role}: fields {sorted(set(wanted) - set(ADMITTED[role]))} are not admitted')
    labels = set(wanted) & {'income', 'occupation', 'sex'}
    if labels and role in ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT') and not registered():
        raise PermissionError(f'{role} labels: no fit or label-based development work before the committed registration '
                              '(PROTOCOL.md, RESERVE_RULE.json, NOMINATION_RULE.md)')
    if role == 'MODEL_SELECT' and labels:
        if not purpose:
            raise PermissionError('MODEL_SELECT labels require a logged purpose')
        _log(MS_LOG, {'role': role, 'fields': sorted(labels), 'purpose': purpose})
    if role in ATTACK_ROLES and labels:
        if not committed(RESULTS / 'FINALIST_LOCK.json'):
            raise PermissionError(f'{role} SEX is for final readers only, after the committed FINALIST_LOCK')
        if not purpose:
            raise PermissionError('ATTACK SEX reads require a logged purpose')
        _log(LABEL_LOG, {'role': role, 'fields': sorted(labels), 'purpose': purpose})
    pub = public_export_sha()[role]
    out = SB._npz(admitted(f'exports/{role}/features.npz'), pub['features'], FIELDS['features'])
    out['feature_names'] = tuple(map(str, out['feature_names']))
    for name in wanted:
        if name in ('features', 'raw_inputs'):
            continue
        z = SB._npz(admitted(f'exports/{role}/{name}.npz'), pub[name], FIELDS[name])
        if not (np.array_equal(z['row_id'], out['row_id']) and np.array_equal(z['unit'], out['unit'])):
            raise PermissionError(f'{role} {name} pairing differs')
        out[name] = z[name].astype(np.int64)
    if 'raw_inputs' in wanted:
        p = admitted(f'exports/{role}/raw_inputs.npz')
        if sha(p) != pub['raw_inputs']:
            raise PermissionError('raw_inputs hash differs')
        with np.load(p, allow_pickle=False) as z:
            out['raw_inputs'] = {k: z[k].copy() for k in z.files}
    return out


def _bank_npz(p, roles=None):
    out, ids = {}, {}
    with np.load(p) as z:
        for key in z.files:
            parts = key.split('__')
            if parts[0] in ('row_id', 'unit'):
                ids[key] = z[key].copy(); continue
            if len(parts) != 3:
                continue
            svc, role, task = parts
            if roles is None or role in roles:
                out.setdefault(svc, {}).setdefault(role, {})[task] = z[key].copy()
    return out, ids


def frozen_bank(seed, roles=('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')):
    """Frozen-hybrids bank (H_LL, H_LJ, H_JL, H_JJ, NEURAL_RAW, TASK_REFIT, T_TREE, U0, Ucal), hash-checked."""
    man = json.loads((SB.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())
    p = admitted(f'bank/FROZEN_BANK_S{seed}.npz')
    if sha(p) != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('frozen bank hash differs from FROZEN_BANK_MANIFEST')
    return _bank_npz(p, roles)


def amd_service_bank(seed, roles=('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')):
    """The source campaign's final service bank (FINALIST = retired B3CTL, NEURAL_RAW, H_LJ, TASKONLY, B2TWIN, PUB_IBSI ...),
    hash-checked against its committed SERVICE_BANK_MANIFEST."""
    man = json.loads((AMD_RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    p = admitted(f'amd_bank/SERVICE_BANK_S{seed}.npz')
    if sha(p) != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('service bank hash differs from the source SERVICE_BANK_MANIFEST')
    return _bank_npz(p, roles)


def um_service_bank(seed, roles=('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')):
    """The SCM utility-matched follow-up's frozen service bank (SCM, PUB_IBSI, ...), hash-checked by admission."""
    return _bank_npz(admitted(f'um_bank/SERVICE_BANK_S{seed}.npz'), roles)[0]


def amd_release(config_id, seed, roles=None):
    """A source-campaign development release (RELEASE.npz: '<ROLE>__<task>' float64 probability vectors), hash-checked."""
    p = admitted(f'amd_runs/B/{config_id}/S{seed}/RELEASE.npz')
    out = {}
    with np.load(p) as z:
        for k in z.files:
            role, t = k.split('__')
            if roles is None or role in roles:
                out.setdefault(role, {})[t] = z[k].copy()
    return out


def teacher(seed, task):
    """The replication's frozen CatBoost teacher (T_TREE) for one task, as the routed tree structure (acs_replication.trees)."""
    from acs_replication import trees
    return trees.TreeTeacher(json.loads(admitted(f'stage_a/S{seed}/TREE_TEACHER_{task}.json').read_text()))


def stage_a_outputs(seed):
    with np.load(admitted(f'stage_a/S{seed}/STAGE_A_OUTPUTS.npz')) as z:
        return {k: z[k].copy() for k in z.files}


def stage_b_outputs(seed):
    with np.load(admitted(f'stage_b/S{seed}/STAGE_B_OUTPUTS_E20.npz')) as z:
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
    """Re-read the ALREADY-OPENED DEV_REPORT truth for this campaign's frozen successor evaluation only. Two registered
    follow-up events (coordinator evaluation, verifier replay), each at most once, after this campaign's committed
    REPORTING_LOCK. The event is appended to this campaign's log and to the lineage-wide replication log; no earlier log line
    is changed."""
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
             'unit_id': os.environ.get(UNIT_ENV), 'files_sha256': {k: pub[k] for k in ('income', 'occupation', 'sex')},
             'note': 'follow-up development evaluation of an already-exposed role; not a first opening; sealed assessment untouched'}
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for log in (ACCESS_LOG, SRC_ACCESS_LOG):
        with open(log, 'a') as f:
            f.write(json.dumps(event, sort_keys=True) + '\n')
    out = SB._npz(admitted('exports/DEV_REPORT/features.npz'), pub['features'], FIELDS['features'])
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
