"""Shared definitions for the PCRL SCM utility-matched follow-up (EXPLORATORY DEVELOPMENT COMPARISON; not confirmation).

Source: the SCM engineering benchmark (research/pcrl-scm-engineering-benchmark-v1, recorded tip 2738cc30b; public package
6fc06021d). Its admitted artifacts (role exports, frozen comparator bank, teachers, out-of-fold predictions, fine partition)
are copied read-only into .private/pcrl_scm_utility_matched_v1/source by `scm_um.admit`, each checked against the hash
recorded by its own manifest. Nothing is retrained.

Guards (unchanged from the source):
- SEALED_ASSESSMENT has no export and is refused by every loader. Other projects' reserved records are never read.
- Role access: MODEL_FIT (features, task labels, SEX) only through the admitted source statistics; MODEL_SELECT (features,
  task labels, SEX) for utility clauses, proxies, nomination and matching; ATTACK_FIT / ATTACK_VALIDATION (features, SEX) for
  final readers only; DEV_REPORT (features, raw inputs) label-free until the committed REPORTING_LOCK.
- DEV_REPORT truth is re-read only through `dev_report_truth`, which requires THIS campaign's committed REPORTING_LOCK, checks
  the published hashes BEFORE logging, and appends one event to this campaign's access log and to the source access log.
  Registered events: one coordinator evaluation and one independent-verifier replay; any further call is refused.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np

from scm_bench import common as SB          # source campaign definitions (read-only use)

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = 'pcrl_scm_utility_matched_v1'
PRIV = ROOT / '.private' / CAMPAIGN
SRCP = PRIV / 'source'
RESULTS = ROOT / 'results' / CAMPAIGN
SB_RESULTS = SB.RESULTS                     # results/pcrl_scm_engineering_benchmark_v1 (public records of the source)
SB_PRIV = SB.PRIV                           # .private/pcrl_scm_engineering_benchmark_v1 (read-only)
ACCESS_LOG = PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl'
SRC_ACCESS_LOG = SB.SRC_ACCESS_LOG          # the protected-tree replication's cross-campaign access log

TASKS, K, CAPS, SEEDS = SB.TASKS, SB.K, SB.CAPS, SB.SEEDS
ADMITTED, FIELDS = SB.ADMITTED, SB.FIELDS
OUT_ROLES = ('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
DEV_EVENTS = ('DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP', 'DEV_REPORT_TRUTH_REREAD_BY_INDEPENDENT_VERIFIER',
              'DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP_CRASH_REPAIR')
REPORT_RESULTS = RESULTS / 'report' / 'DEV_REPORT_RESULTS.json'

sha, arr_sha, wire_sha, now, task_vectors = SB.sha, SB.arr_sha, SB.wire_sha, SB.now, SB.task_vectors
public_export_sha = SB.public_export_sha


def committed(path):
    return SB.committed(path)


def admission():
    return json.loads((RESULTS / 'SOURCE_ADMISSION.json').read_text())


def _admitted_copy(rel):
    """Path of an admitted private copy, after re-checking its hash against SOURCE_ADMISSION.json."""
    rec = admission()['copies'][rel]
    p = SRCP / rel
    if sha(p) != rec['sha256']:
        raise PermissionError(f'admitted copy hash differs: {rel}')
    return p


def load_role(role, wanted=('features',)):
    """Admitted, hash-checked development exports under this campaign's role-access rules."""
    if role == 'SEALED_ASSESSMENT' or role not in ADMITTED:
        raise PermissionError(f'role {role} is not admitted (the sealed assessment stays closed)')
    if set(wanted) - set(ADMITTED[role]):
        raise PermissionError(f'{role}: fields {sorted(set(wanted) - set(ADMITTED[role]))} are not admitted through load_role')
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


def frozen_bank(seed, roles=OUT_ROLES):
    """The frozen-hybrids bank (NEURAL_RAW, H_LJ, U0, Ucal, ...), admitted copy, hash-checked against the public manifest."""
    man = json.loads((SB.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())
    p = _admitted_copy(f'bank/FROZEN_BANK_S{seed}.npz')
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


def source_fit(seed):
    """Admitted source fit artifacts for one seed: PARTITION arrays, historical ARM_TABLES and FIT_SEED record."""
    part = _admitted_copy(f'fit/FIT_{seed}/PARTITION.npz')
    tabs = _admitted_copy(f'fit/FIT_{seed}/ARM_TABLES.npz')
    rec = json.loads(_admitted_copy(f'fit/FIT_{seed}/FIT_SEED.json').read_text())
    with np.load(part) as z:
        P = {k: z[k].copy() for k in z.files}
    with np.load(tabs) as z:
        T = {k: z[k].copy() for k in z.files}
    return P, T, rec


def source_arm_outputs(seed, arms=None):
    p = _admitted_copy(f'fit/FIT_{seed}/ARM_OUTPUTS.npz')
    out, ids = {}, {}
    with np.load(p) as z:
        for key in z.files:
            parts = key.split('__')
            if parts[0] in ('row_id', 'unit'):
                ids[key] = z[key].copy(); continue
            arm, role, task = parts
            if arms is None or arm in arms:
                out.setdefault(arm, {}).setdefault(role, {})[task] = np.asarray(z[key], dtype=np.float64)
    return out, ids


def routing(seed):
    """Admitted runtime routing (initial fine cell per person) for the four output roles, hash-checked."""
    p = _admitted_copy(f'routing/ROUTING_S{seed}.npz')
    with np.load(p) as z:
        return {k: z[k].copy() for k in z.files}


def dev_access_events():
    if not ACCESS_LOG.is_file():
        return []
    return [json.loads(l) for l in ACCESS_LOG.read_text().splitlines() if l.strip()]


def dev_report_truth(purpose, event_name='DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP'):
    """Re-read the ALREADY-OPENED DEV_REPORT truth (labels, SEX) for this campaign's frozen evaluation only.
    One coordinator evaluation and one independent-verifier replay are registered; a second call of either kind is refused."""
    if event_name not in DEV_EVENTS:
        raise PermissionError('unregistered DEV_REPORT access event')
    if any(e.get('event') == event_name for e in dev_access_events()):
        raise PermissionError(f'{event_name} already used once in this campaign (one evaluation per registered role)')
    if event_name == 'DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP_CRASH_REPAIR':
        # registered contingency (PROTOCOL.md section 10): only after a logged coordinator re-read whose run crashed before
        # writing any result; the failed attempt is preserved in the queue ledger
        if not any(e.get('event') == 'DEV_REPORT_TRUTH_REREAD_BY_FOLLOWUP' for e in dev_access_events()) or REPORT_RESULTS.exists():
            raise PermissionError('crash-repair re-read is allowed only after a coordinator re-read that produced no result')
    lock = RESULTS / 'REPORTING_LOCK.json'
    if not committed(lock):
        raise PermissionError("DEV_REPORT truth re-read requires this campaign's committed REPORTING_LOCK")
    pub = public_export_sha()['DEV_REPORT']
    src_files = json.loads((SB.SRC_PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())['files']['DEV_REPORT']
    truth = {name: SB.SRC_PRIV / Path(src_files[name]['path']).relative_to('.private/acs_replication') for name in ('income', 'occupation', 'sex')}
    for name, p in truth.items():          # existence and published hashes are checked BEFORE any log event is written
        if not p.is_file() or sha(p) != pub[name]:
            raise PermissionError(f'DEV_REPORT {name} truth file missing or hash differs from the public extraction summary')
    event = {'event': event_name, 'campaign': CAMPAIGN, 'utc': now(), 'purpose': purpose, 'reporting_lock_sha256': sha(lock),
             'unit_id': os.environ.get('SCMUM_UNIT_ID'), 'files_sha256': {k: pub[k] for k in ('income', 'occupation', 'sex')},
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
