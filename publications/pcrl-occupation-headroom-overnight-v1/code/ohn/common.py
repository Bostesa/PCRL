"""Shared definitions for the PCRL overnight occupation-headroom and robust-release campaign (EXPLORATORY DEVELOPMENT on
previously used records; not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged
25-64, civilian employed), in the development roles of the protected-tree replication. The service predicts each person's
income class (2 classes: PINCP x ADJINC / 10^6 above $50,000) and occupation group (5 classes: OCCP mapped to the five official
broad civilian groups) from six permitted raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column encoding whose
dictionary and moments were learned label-free on MODEL_FIT). Each recipient receives only its own float64 probability vector
and argmax decision. SEX is the dataset's recorded demographic column (sex0 = SEX code 1, men; sex1 = SEX code 2, women);
separate evaluation classifiers ("readers") estimate SEX from the released probability vectors: each output alone and the two
outputs aligned to the same record (the linked pair). SEX is never a runtime input or an output.

Source: the income-headroom successor campaign (research/pcrl-income-headroom-successor-v1 @ 2af3c9db2; public package
89fd2f6e6). Its private artifacts are admitted read-only, by hash, into a mirror of its own private layout
(.private/pcrl_income_headroom_successor_v1/ inside this worktree) so that its released code (ihs.*) runs unchanged for the
income anchors; `ohn.admit` records every copy in results/<campaign>/SOURCE_ADMISSION.json.

Role access (PROTOCOL.md section 3):
- MODEL_FIT (features, raw inputs, income, occupation, SEX): fitting release parameters and task models; household-separated
  internal folds for cross-fitting checks. SEX only as a training-time label.
- CALIBRATION (features, income, occupation): calibrators only (or household-separated internal splits). No SEX.
- MODEL_SELECT (features, income, occupation, SEX): logged development selection, diagnosis and the development reader grid
  (household-disjoint FIT / VALIDATION / SCORE reader roles). Never gradient-fitting or calibration labels.
- ATTACK_FIT / ATTACK_VALIDATION (features, SEX): final readers only, after this campaign's committed FINALIST_LOCK.
- DEV_REPORT (features, raw inputs): label-free. Truth only through `dev_report_truth` after this campaign's committed
  REPORTING_LOCK and its verified public receipt, under two NEW follow-up events. Closed unless a finalist is promoted.
- SEALED_ASSESSMENT and capped-out records: refused by every loader. No new population, download or label is opened.
Label reads are refused until the registration is committed AND its sanitized public receipt has been verified on the GitHub
remote (`ohn.receipt verify`), so no fit can precede the public receipt.
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
CAMPAIGN = 'pcrl_occupation_headroom_overnight_v1'
PRIV = ROOT / '.private' / CAMPAIGN
RUNS = PRIV / 'runs'
RESULTS = ROOT / 'results' / CAMPAIGN
PCRL = ROOT.parents[2]

IHS_CAMPAIGN = 'pcrl_income_headroom_successor_v1'
IHS_TIP = '2af3c9db2d65e052dbd9d3d6a040227e04d09854'          # research tip (local); public package below
IHS_PUBLICATION = '89fd2f6e649547f65cf1b69a9e8e8b51f854d25e'
IHS_RESULTS = ROOT / 'results' / IHS_CAMPAIGN
IHS_SRC_PRIV = PCRL / '.claude' / 'worktrees' / 'pcrl-income-headroom-successor-v1' / '.private' / IHS_CAMPAIGN   # read-only
MIRROR = ROOT / '.private' / IHS_CAMPAIGN                    # verified admitted mirror (ihs.* code reads it unchanged)
ADM = MIRROR / 'admitted'

REP_PRIV = SB.SRC_PRIV                      # protected-tree replication private tree (DEV_REPORT truth; read-only)
SRC_ACCESS_LOG = SB.SRC_ACCESS_LOG          # lineage-wide DEV_REPORT access log (append-only)
ACCESS_LOG = PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl'
MS_LOG = PRIV / 'access' / 'MODEL_SELECT_USE.jsonl'
LABEL_LOG = PRIV / 'access' / 'ATTACK_LABEL_READS.jsonl'
TRAIN_LOG = PRIV / 'access' / 'TRAINING_LABEL_READS.jsonl'
REG_RECEIPT = PRIV / 'access' / 'REGISTRATION_RECEIPT_VERIFIED.json'
REP_RECEIPT = PRIV / 'access' / 'REPORTING_RECEIPT_VERIFIED.json'

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
DEV_EVENTS = ('DEV_REPORT_TRUTH_REREAD_BY_OHN_COORDINATOR', 'DEV_REPORT_TRUTH_REREAD_BY_OHN_INDEPENDENT_VERIFIER')
UNIT_ENV = 'OHN_UNIT_ID'
PUBLICATION_BRANCH = 'publication/pcrl-occupation-headroom-overnight-v1'
REMOTE_URL = 'https://github.com/Bostesa/PCRL.git'

sha, arr_sha, wire_sha, task_vectors, public_export_sha = SB.sha, SB.arr_sha, SB.wire_sha, SB.task_vectors, SB.public_export_sha


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def committed(path):
    return SB.committed(path)


def admission():
    return json.loads((RESULTS / 'SOURCE_ADMISSION.json').read_text())


def admitted(rel):
    """Path of an admitted mirror copy (relative to the mirror root) after re-checking its hash against SOURCE_ADMISSION.json."""
    rec = admission()['copies'][rel]
    p = MIRROR / rel
    if sha(p) != rec['sha256']:
        raise PermissionError(f'admitted copy hash differs: {rel}')
    return p


REGISTRATION_FILES = ('PROTOCOL.md', 'DEV_GRID.json', 'EXPOSURE_TABLE.md', 'roles/statistical_reviewer/RULES.json',
                      'roles/statistical_reviewer/SELECTION_RULE.md')


def _receipt_ok(path, files):
    if not path.is_file():
        return False
    rec = json.loads(path.read_text())
    if not rec.get('remote_sha_verified') or rec.get('remote_sha') != rec.get('receipt_commit'):
        return False
    return all(rec.get('files_sha256', {}).get(f) == sha(RESULTS / f) for f in files)


def registered():
    """True once the registration files are committed AND the sanitized public receipt holding exactly these files has been
    verified on the GitHub remote (ohn.receipt verify writes REGISTRATION_RECEIPT_VERIFIED.json after `git ls-remote`)."""
    return all(committed(RESULTS / f) for f in REGISTRATION_FILES) and _receipt_ok(REG_RECEIPT, REGISTRATION_FILES)


def _log(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path, 'a') as f:
        f.write(json.dumps({'utc': now(), 'unit_id': os.environ.get(UNIT_ENV), **rec}, sort_keys=True) + '\n')


def load_role(role, wanted=('features',), purpose=None):
    """Admitted, hash-checked development exports under this campaign's role-access rules."""
    if role == 'SEALED_ASSESSMENT' or role not in ADMITTED:
        raise PermissionError(f'role {role} is not admitted (the sealed assessment and capped-out records stay closed)')
    if set(wanted) - set(ADMITTED[role]):
        raise PermissionError(f'{role}: fields {sorted(set(wanted) - set(ADMITTED[role]))} are not admitted')
    labels = set(wanted) & {'income', 'occupation', 'sex'}
    if labels and role in ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT') and not registered():
        raise PermissionError(f'{role} labels: no fit or label-based development work before the committed registration and its '
                              'verified public receipt')
    if labels and role in ('MODEL_FIT', 'CALIBRATION'):
        _log(TRAIN_LOG, {'role': role, 'fields': sorted(labels), 'purpose': purpose or ''})
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
    out = SB._npz(admitted(f'admitted/exports/{role}/features.npz'), pub['features'], FIELDS['features'])
    out['feature_names'] = tuple(map(str, out['feature_names']))
    for name in wanted:
        if name in ('features', 'raw_inputs'):
            continue
        z = SB._npz(admitted(f'admitted/exports/{role}/{name}.npz'), pub[name], FIELDS[name])
        if not (np.array_equal(z['row_id'], out['row_id']) and np.array_equal(z['unit'], out['unit'])):
            raise PermissionError(f'{role} {name} pairing differs')
        out[name] = z[name].astype(np.int64)
    if 'raw_inputs' in wanted:
        p = admitted(f'admitted/exports/{role}/raw_inputs.npz')
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
    """Frozen-hybrids bank (H_LL, H_LJ, H_JL, H_JJ, NEURAL_RAW, TASK_REFIT, T_TREE, U0, Ucal), hash-checked twice."""
    man = json.loads((SB.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())
    p = admitted(f'admitted/bank/FROZEN_BANK_S{seed}.npz')
    if sha(p) != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('frozen bank hash differs from FROZEN_BANK_MANIFEST')
    return _bank_npz(p, roles)


def amd_release(config_id, seed, roles=None):
    """A source-lineage development release (retired B3CTL, its task-only twin, B2TWIN), hash-checked."""
    p = admitted(f'admitted/amd_runs/B/{config_id}/S{seed}/RELEASE.npz')
    out = {}
    with np.load(p) as z:
        for k in z.files:
            role, t = k.split('__')
            if roles is None or role in roles:
                out.setdefault(role, {})[t] = z[k].copy()
    return out


def ihs_release(family, config_id, seed, roles=None):
    """A source-campaign (income-headroom successor) release, e.g. H/H3_STRONG_LJDIR_PB, H/HC_STRONG, O/O1_SEXBAL; the array
    is checked against its admitted RELEASE.json and SOURCE_ADMISSION.json."""
    rec = json.loads(admitted(f'runs/{family}/{config_id}/S{seed}/RELEASE.json').read_text())
    p = admitted(f'runs/{family}/{config_id}/S{seed}/RELEASE.npz')
    if sha(p) != rec['release_sha256']:
        raise PermissionError('source release array differs from its record')
    out = {}
    with np.load(p) as z:
        for k in z.files:
            role, t = k.split('__')
            if roles is None or role in roles:
                out.setdefault(role, {})[t] = z[k].copy()
    return out, rec


def teacher(seed, task):
    """The replication's frozen CatBoost teacher (T_TREE) for one task, as the routed tree structure (acs_replication.trees)."""
    from acs_replication import trees
    return trees.TreeTeacher(json.loads(admitted(f'admitted/stage_a/S{seed}/TREE_TEACHER_{task}.json').read_text()))


def stage_a_outputs(seed):
    with np.load(admitted(f'admitted/stage_a/S{seed}/STAGE_A_OUTPUTS.npz')) as z:
        return {k: z[k].copy() for k in z.files}


def stage_b_outputs(seed):
    with np.load(admitted(f'admitted/stage_b/S{seed}/STAGE_B_OUTPUTS_E20.npz')) as z:
        return {k: z[k].copy() for k in z.files}


def stage_b_theta(seed, arm, label='E20'):
    with np.load(admitted(f'admitted/stage_b/S{seed}/{arm}__{label}.npz')) as z:
        return [z['theta_income'].copy(), z['theta_occupation'].copy()]


def household_folds(unit, k, tag):
    """Deterministic household-separated fold id per person (label-free; SHA-256 order of household ids)."""
    groups = np.unique(unit)
    keys = [hashlib.sha256(f'{tag}|{g}'.encode()).hexdigest() for g in groups]
    order = np.argsort(keys)
    fold_of_group = np.empty(len(groups), dtype=np.int64)
    fold_of_group[order] = np.arange(len(groups)) % k
    return fold_of_group[np.searchsorted(groups, unit)]


CROSSFIT_TAG = 'OHN1_crossfit'              # household 5-fold cross-fitting split of MODEL_FIT (registered in DEV_GRID.json)
CROSSFIT_K = 5


def crossfit_folds(unit):
    return household_folds(unit, CROSSFIT_K, CROSSFIT_TAG)


def dev_access_events():
    if not ACCESS_LOG.is_file():
        return []
    return [json.loads(l) for l in ACCESS_LOG.read_text().splitlines() if l.strip()]


def dev_report_truth(purpose, event_name):
    """Re-read the ALREADY-OPENED DEV_REPORT truth for this campaign's frozen finalist evaluation only. Two registered follow-up
    events (coordinator evaluation, verifier replay), each at most once, after this campaign's committed REPORTING_LOCK and its
    verified public reporting receipt. The event is appended to this campaign's log and to the lineage-wide replication log; no
    earlier log line is changed."""
    if event_name not in DEV_EVENTS:
        raise PermissionError('unregistered DEV_REPORT access event')
    if any(e.get('event') == event_name for e in dev_access_events()):
        raise PermissionError(f'{event_name} already used once in this campaign')
    lock = RESULTS / 'REPORTING_LOCK.json'
    if not committed(lock):
        raise PermissionError("DEV_REPORT truth re-read requires this campaign's committed REPORTING_LOCK")
    if not _receipt_ok(REP_RECEIPT, ('REPORTING_LOCK.json',)):
        raise PermissionError('DEV_REPORT truth re-read requires the verified public reporting receipt')
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
    out = SB._npz(admitted('admitted/exports/DEV_REPORT/features.npz'), pub['features'], FIELDS['features'])
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
