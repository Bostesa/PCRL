"""Shared definitions and the single logged label-access layer of the PCRL correction-transfer campaign
(`pcrl_correction_transfer_stability_v1`; EXPLORATORY DEVELOPMENT on previously used records, not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed), in the development roles of the protected-tree replication. The service predicts an income class (2) and an
occupation group (5 official broad civilian groups) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column
encoding X). Each recipient receives only its own float64 probability vector and argmax decision. SEX is the recorded demographic
column (sex 0 = SEX code 1, men; sex 1 = SEX code 2, women): a training-time label on authorized training households and an
evaluation label for development readers; never a runtime input or output.

Sources (admitted read-only, by hash; `cts.admit`):
- the occupation-headroom campaign (research tip ca8725e72, local; publication 1435c395d): its private tree is copied to
  .private/pcrl_occupation_headroom_overnight_v1/ so that its released code (ohn.*) runs unchanged on the copies;
- the income-headroom successor (H3 source; publication 89fd2f6e6) through the occupation-headroom campaign's verified mirror
  .private/pcrl_income_headroom_successor_v1/ (ohn.common.admitted re-checks every file against SOURCE_ADMISSION.json).

Label access is defined by ROLE x FIELD x OPERATION x PURPOSE (POLICY below), never by script name. Every label read, by every
entry point (coordinator, investigators, statistical reviewer, reader investigator, independent verifier), goes through `load`,
which refuses reads that the policy does not allow and appends one line to .private/<campaign>/access/LABEL_ACCESS.jsonl
(utc, unit id, calling entry point, role, fields, operation, purpose, partition, partition role, people, households, row-set hash).
- No label is read before the registration is committed and its sanitized public receipt is verified on the GitHub remote.
- ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT labels are refused for the whole campaign (features stay available label-free for
  deployment parity). The sealed assessment and capped-out records are refused by every loader.
- The repair CHECK panels C0 / C1 (MODEL_FIT households) are refused for every operation except the source replay, the support
  count and, after the committed REPAIR_FREEZE lock, the repair check and the independent verification.
- MODEL_FIT labels are released only for the rows of the requested partition role (FIT, GUARD or CHECK of D0, D1, R0, R1; C0; C1),
  except for the registered whole-role operations (source replay, support count).
"""
from __future__ import annotations
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from scm_bench import common as SB
from ohn import common as OC                 # admitted-copy hash checks (SOURCE_ADMISSION.json of the source campaign)

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = 'pcrl_correction_transfer_stability_v1'
PRIV = ROOT / '.private' / CAMPAIGN
RUNS = PRIV / 'runs'
RESULTS = ROOT / 'results' / CAMPAIGN
ACCESS = PRIV / 'access'
LABEL_LOG = ACCESS / 'LABEL_ACCESS.jsonl'
REG_RECEIPT = ACCESS / 'REGISTRATION_RECEIPT_VERIFIED.json'
FREEZE_RECEIPT = ACCESS / 'REPAIR_FREEZE_RECEIPT_VERIFIED.json'
RECEIPT_ATTEMPTS = ACCESS / 'RECEIPT_ATTEMPTS.jsonl'
UNIT_ENV = 'CTS_UNIT_ID'
STATS_ENV = 'CTS_UNIT_STATS'
CATBOOST_PY = PRIV / 'venv_catboost' / 'bin' / 'python'     # CatBoost 1.2.8 (fit-time only; inference is numpy)
TORCH_PY = Path('<repository>/.venv/bin/python')  # torch 2.10.0 (correction training)

OHN_CAMPAIGN = 'pcrl_occupation_headroom_overnight_v1'
OHN_TIP = 'ca8725e72'                                         # local research tip of the source campaign
OHN_PUBLICATION = '1435c395d3f753a9bf0b553d14d264ca9e991421'
IHS_PUBLICATION = '89fd2f6e649547f65cf1b69a9e8e8b51f854d25e'
OHN_RESULTS = ROOT / 'results' / OHN_CAMPAIGN
OHN_RUNS = OC.RUNS                                           # admitted copy of the source campaign's runs (read-only)
PUBLICATION_BRANCH = 'publication/pcrl-correction-transfer-stability-v1'
PUBLICATION_NAME = 'pcrl-correction-transfer-stability-v1'
REMOTE_URL = 'https://github.com/Bostesa/PCRL.git'

TASKS, K, CAPS, SEEDS = SB.TASKS, SB.K, SB.CAPS, SB.SEEDS
K_OCC = 5
GUARD_CAP = 0.005
RAW_FIELDS = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
ADMITTED = OC.ADMITTED
LABELS = ('income', 'occupation', 'sex')
CLOSED_LABEL_ROLES = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')

sha, arr_sha, task_vectors = SB.sha, SB.arr_sha, SB.task_vectors


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def committed(path):
    return SB.committed(path)


# ------------------------------------------------------------------ partitions (label-free; frozen in the registration)
N_BLOCKS = 20
BLOCK_TAG = 'CTS1_blocks'
PANELS = {'C0': (0, 1, 2), 'C1': (3, 4, 5)}                  # repair CHECK panels (reserved before any fit)
POOL_BLOCKS = tuple(range(6, 20))
PARTITIONS = {
    'D0': {'kind': 'diagnosis', 'seed': 0, 'tag': 'CTS1_D0', 'parts': 10, 'FIT': range(0, 6), 'GUARD': range(6, 8), 'CHECK': range(8, 10)},
    'D1': {'kind': 'diagnosis', 'seed': 1, 'tag': 'CTS1_D1', 'parts': 10, 'FIT': range(0, 6), 'GUARD': range(6, 8), 'CHECK': range(8, 10)},
    'R0': {'kind': 'repair', 'seed': 0, 'tag': 'CTS1_R0', 'parts': 4, 'FIT': range(0, 3), 'GUARD': range(3, 4), 'CHECK': 'C0'},
    'R1': {'kind': 'repair', 'seed': 1, 'tag': 'CTS1_R1', 'parts': 4, 'FIT': range(0, 3), 'GUARD': range(3, 4), 'CHECK': 'C1'},
    # bounded sensitivity check: the same households as D0, with the second source witness recipe (B2_SINGLE) instead of A1_PO_16
    'D0B': {'kind': 'diagnosis', 'seed': 0, 'tag': 'CTS1_D0', 'parts': 10, 'FIT': range(0, 6), 'GUARD': range(6, 8), 'CHECK': range(8, 10),
            'witness': 'B2_SINGLE', 'same_rows_as': 'D0'},
}
TAG_SUFFIX = {}        # support-check repairs (filled from PARTITION_SUPPORT.json if a partition needed the next tag suffix)
PROLES = ('FIT', 'GUARD', 'CHECK')


def household_folds(unit, k, tag):
    return OC.household_folds(unit, k, tag)


def blocks(unit):
    return household_folds(unit, N_BLOCKS, BLOCK_TAG)


def partition_masks(unit):
    """{'C0': mask, 'C1': mask, 'POOL': mask, 'D0': {'FIT': mask, 'GUARD': ..., 'CHECK': ...}, ...} over MODEL_FIT people."""
    b = blocks(unit)
    out = {name: np.isin(b, bl) for name, bl in PANELS.items()}
    pool = np.isin(b, POOL_BLOCKS)
    out['POOL'] = pool
    for pid, spec in PARTITIONS.items():
        tag = spec['tag'] + TAG_SUFFIX.get(pid, '')
        part = household_folds(unit[pool], spec['parts'], tag)
        full = np.full(len(unit), -1, dtype=np.int64)
        full[pool] = part
        m = {r: np.isin(full, list(spec[r])) & pool for r in ('FIT', 'GUARD')}
        if spec['kind'] == 'diagnosis':
            m['CHECK'] = np.isin(full, list(spec['CHECK'])) & pool
        else:
            m['CHECK'] = out[spec['CHECK']].copy()
        out[pid] = m
    return out


def partition_rows(unit, pid, prole):
    m = partition_masks(unit)
    if pid in PANELS:
        if prole not in (None, 'CHECK'):
            raise ValueError('panels have no FIT / GUARD role')
        return m[pid]
    return m[pid][prole]


def check_partitions(unit):
    """Label-free structural checks: disjoint roles inside each partition, panels disjoint from the pool, whole households."""
    m = partition_masks(unit)
    problems = []
    if (m['C0'] & m['C1']).any() or ((m['C0'] | m['C1']) & m['POOL']).any() or not (m['C0'] | m['C1'] | m['POOL']).all():
        problems.append('panels / pool not a partition of MODEL_FIT')
    for pid in PARTITIONS:
        f, g, c = m[pid]['FIT'], m[pid]['GUARD'], m[pid]['CHECK']
        if (f & g).any() or (f & c).any() or (g & c).any():
            problems.append(f'{pid}: roles overlap')
        for r, mm in m[pid].items():
            hh = set(np.unique(unit[mm]))
            others = set(np.unique(unit[~mm]))
            if hh & others:
                problems.append(f'{pid}:{r} splits a household')
        if ((m[pid]['FIT'] | m[pid]['GUARD']) & (m['C0'] | m['C1'])).any():
            problems.append(f'{pid}: FIT/GUARD touch a repair panel')
    return problems


def partition_summary(unit):
    m = partition_masks(unit)

    def desc(mask):
        hh = np.unique(unit[mask])
        return {'people': int(mask.sum()), 'households': int(len(hh)),
                'household_set_sha256': hashlib.sha256(','.join(map(str, hh)).encode()).hexdigest()}
    out = {'C0': desc(m['C0']), 'C1': desc(m['C1']), 'POOL': desc(m['POOL'])}
    for pid in PARTITIONS:
        out[pid] = {r: desc(m[pid][r]) for r in PROLES}
    fits = {pid: set(np.unique(unit[m[pid]['FIT']])) for pid in PARTITIONS}
    out['FIT_overlap_households'] = {f'{a}|{b}': len(fits[a] & fits[b]) for a in PARTITIONS for b in PARTITIONS if a < b}
    return out


# ------------------------------------------------------------------ registration / freeze state
REGISTRATION_FILES = ('PROTOCOL.md', 'PARTITIONS.json', 'EXPOSURE_MAP.md', 'roles/statistical_reviewer/RULES.json',
                      'roles/statistical_reviewer/RULES.md', 'ACCESS_POLICY.json')
FREEZE_FILE = 'REPAIR_FREEZE.json'


def _receipt_ok(path, files):
    if not path.is_file():
        return False
    rec = json.loads(path.read_text())
    if not rec.get('remote_sha_verified') or rec.get('remote_sha') != rec.get('receipt_commit'):
        return False
    return all(rec.get('files_sha256', {}).get(f) == sha(RESULTS / f) for f in files)


def registered():
    return all(committed(RESULTS / f) for f in REGISTRATION_FILES) and _receipt_ok(REG_RECEIPT, REGISTRATION_FILES)


def repair_frozen():
    return committed(RESULTS / FREEZE_FILE)


# ------------------------------------------------------------------ access policy (ROLE x FIELD x OPERATION x PURPOSE)
POLICY = {
    'source_replay': {'roles': ('MODEL_FIT', 'MODEL_SELECT'), 'fields': ('occupation', 'sex', 'income'), 'rows': 'ALL',
                      'what': 'recompute source records (cost curve, projection decision, routing reproduction) on the rows the '
                              'source used; no new release is evaluated on a repair panel'},
    'support_count': {'roles': ('MODEL_FIT',), 'fields': ('occupation', 'sex'), 'rows': 'ALL',
                      'what': 'class x SEX counts per partition role before any fit (no outcome)'},
    'fit': {'roles': ('MODEL_FIT',), 'fields': ('occupation', 'sex', 'income'), 'rows': ('FIT',),
            'what': 'task learning, routing, correction directions, training critics, cross-fitting, Track R income lineage'},
    'calibrate': {'roles': ('CALIBRATION',), 'fields': ('occupation', 'income'), 'rows': 'ALL',
                  'what': 'permitted task calibration / Track R CALIBRATION stack, beta budget and Platt map'},
    'select': {'roles': ('MODEL_FIT',), 'fields': ('occupation', 'sex'), 'rows': ('GUARD',),
               'what': 'GUARD incremental costs for choosing a strength from the fixed ladder'},
    'check_diag': {'roles': ('MODEL_FIT',), 'fields': ('occupation', 'sex'), 'rows': ('CHECK',), 'partitions': ('D0', 'D1', 'D0B'),
                   'what': 'diagnostic CHECK realized costs (D partitions only)'},
    'check_repair': {'roles': ('MODEL_FIT',), 'fields': ('occupation', 'sex'), 'rows': ('CHECK',), 'partitions': ('R0', 'R1', 'C0', 'C1'),
                     'requires_freeze': True, 'what': 'repair CHECK realized costs, after the committed REPAIR_FREEZE'},
    'benchmark': {'roles': ('MODEL_SELECT',), 'fields': ('income', 'occupation', 'sex'), 'rows': 'ALL',
                  'what': 'original-benchmark utility (24 clauses + 4 guards) and witness baseline slack (logged design input)'},
    'readers': {'roles': ('MODEL_SELECT',), 'fields': ('sex',), 'rows': 'ALL',
                'what': 'development reader grid (FIT / VALIDATION / SCORE reader roles inside MODEL_SELECT)'},
    'screen_study': {'roles': ('MODEL_SELECT',), 'fields': ('sex',), 'rows': 'ALL',
                     'what': 'operating characteristics of recipient screens on frozen source releases and aliases'},
    'verify': {'roles': ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT'), 'fields': ('income', 'occupation', 'sex'),
               'rows': ('FIT', 'GUARD', 'CHECK', 'ALL'), 'panels_require_freeze': True,
               'what': 'independent verifier replay with separate metric and selection code'},
}


ENTRY_POINTS = {      # operation -> authorized entry points (module names, or repository-relative script paths for the verifier)
    'source_replay': ('cts.preflight', 'cts.trackf'),
    'support_count': ('cts.partitions',),
    'fit': ('cts.trackf', 'cts.nested', 'cts.select', 'cts.track_r', 'cts.sensitivity', 'cts.redesign', 'cts.compression'),
    'calibrate': ('cts.trackf', 'cts.track_r', 'cts.sensitivity'),
    'select': ('cts.trackf', 'cts.select', 'cts.compression', 'cts.redesign'),
    'check_diag': ('cts.trackf', 'cts.select', 'cts.compression', 'cts.redesign'),
    'check_repair': ('cts.trackf', 'cts.select', 'cts.compression', 'cts.redesign'),
    'benchmark': ('cts.evaluate', 'cts.trackf', 'cts.select', 'cts.compression', 'cts.redesign'),
    'readers': ('cts.devreaders',),
    'screen_study': ('cts.screen_study',),
    'verify': ('results/pcrl_correction_transfer_stability_v1/roles/verifier/code/',),
}
AMENDMENTS = 'ACCESS_AMENDMENTS.json'      # dated, committed additions of entry points (before their first use)


def _caller():
    """(module or repository-relative script path, function) of the first frame outside this module."""
    for fr in inspect.stack()[2:12]:
        g = fr.frame.f_globals
        mod = g.get('__name__', '?')
        if mod == __name__:
            continue
        if mod == '__main__':
            spec = g.get('__spec__')
            if spec is not None and getattr(spec, 'name', None):
                mod = spec.name
            else:
                try:
                    mod = str(Path(g.get('__file__', '?')).resolve().relative_to(ROOT))
                except ValueError:
                    mod = str(g.get('__file__', '?'))
        return mod, fr.function
    return '?', '?'


def authorized_entry_points():
    eps = {k: list(v) for k, v in ENTRY_POINTS.items()}
    p = RESULTS / AMENDMENTS
    if p.is_file() and committed(p):
        for a in json.loads(p.read_text()).get('amendments', []):
            eps.setdefault(a['operation'], []).append(a['entry_point'])
    return eps


def _check_entry(operation):
    mod, fn = _caller()
    allowed = authorized_entry_points().get(operation, [])
    if not any(mod == e or mod.startswith(e if e.endswith('/') else e + '.') or (e.endswith('/') and mod.startswith(e)) for e in allowed):
        raise PermissionError(f'entry point {mod} is not authorized for operation {operation} (authorize it by a dated, committed '
                              f'{AMENDMENTS} entry before use)')
    return f'{mod}:{fn}'


def _log(rec):
    ACCESS.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(LABEL_LOG, 'a') as f:
        f.write(json.dumps({'utc': now(), 'unit_id': os.environ.get(UNIT_ENV), **rec}, sort_keys=True) + '\n')


def _features(role):
    pub = SB.public_export_sha()[role]
    out = SB._npz(OC.admitted(f'admitted/exports/{role}/features.npz'), pub['features'], SB.FIELDS['features'])
    out['feature_names'] = tuple(map(str, out['feature_names']))
    return out


def _label(role, name, out):
    pub = SB.public_export_sha()[role]
    z = SB._npz(OC.admitted(f'admitted/exports/{role}/{name}.npz'), pub[name], SB.FIELDS[name])
    if not (np.array_equal(z['row_id'], out['row_id']) and np.array_equal(z['unit'], out['unit'])):
        raise PermissionError(f'{role} {name} pairing differs')
    return z[name].astype(np.int64)


def features(role, raw=False):
    """Label-free admitted features (any role, any operation; no log needed)."""
    if role == 'SEALED_ASSESSMENT' or role not in ADMITTED:
        raise PermissionError(f'role {role} is not admitted')
    out = _features(role)
    if raw:
        if 'raw_inputs' not in ADMITTED[role]:
            raise PermissionError(f'{role} has no admitted raw inputs')
        p = OC.admitted(f'admitted/exports/{role}/raw_inputs.npz')
        if sha(p) != SB.public_export_sha()[role]['raw_inputs']:
            raise PermissionError('raw_inputs hash differs')
        with np.load(p, allow_pickle=False) as z:
            out['raw_inputs'] = {k: z[k].copy() for k in z.files}
    return out


def load(role, fields, operation, purpose, partition=None, prole=None):
    """Labels (and features) of one role under the access policy. MODEL_FIT reads return ONLY the rows of (partition, prole)
    unless the operation is a registered whole-role operation. Returns a dict with X, feature_names, row_id, unit, the requested
    labels and 'rows' (boolean mask over the role's people)."""
    fields = tuple(fields)
    if role == 'SEALED_ASSESSMENT' or role not in ADMITTED:
        raise PermissionError(f'role {role} is not admitted')
    labels = tuple(f for f in fields if f in LABELS)
    if set(fields) - set(ADMITTED[role]) - {'features'}:
        raise PermissionError(f'{role}: fields {sorted(set(fields) - set(ADMITTED[role]))} are not admitted')
    out = features(role, raw='raw_inputs' in fields)
    n = len(out['row_id'])
    rows = np.ones(n, dtype=bool)
    if not labels:
        out['rows'] = rows
        return out
    if role in CLOSED_LABEL_ROLES:
        raise PermissionError(f'{role} labels are closed for this campaign')
    if not registered():
        raise PermissionError('no label read before the committed registration and its verified public receipt')
    pol = POLICY.get(operation)
    if pol is None:
        raise PermissionError(f'unregistered operation {operation!r}')
    if role not in pol['roles'] or set(labels) - set(pol['fields']):
        raise PermissionError(f'operation {operation} does not allow {role} {labels}')
    if not purpose or len(purpose) < 12:
        raise PermissionError('a label read needs a stated purpose')
    entry = _check_entry(operation)
    unit = out['unit']
    panel_mask = None
    if role == 'MODEL_FIT':
        pm = partition_masks(unit)
        panel_mask = pm['C0'] | pm['C1']
        if pol['rows'] == 'ALL' and partition is None:
            rows = np.ones(n, dtype=bool)
        else:
            allowed_r = pol['rows'] if pol['rows'] != 'ALL' else PROLES + ('ALL',)
            if partition is None:
                raise PermissionError(f'operation {operation} needs a partition and partition role')
            if pol.get('partitions') and partition not in pol['partitions']:
                raise PermissionError(f'operation {operation} is not allowed on partition {partition}')
            if partition in PANELS:
                prole = 'CHECK' if prole is None else prole
                if prole != 'CHECK':
                    raise PermissionError('panels have only a CHECK role')
                rows = pm[partition].copy()
            elif partition in PARTITIONS:
                if prole not in PROLES or (prole not in allowed_r):
                    raise PermissionError(f'operation {operation} does not allow partition role {prole}')
                rows = pm[partition][prole].copy()
            else:
                raise PermissionError(f'unknown partition {partition}')
        touches_panel = bool((rows & panel_mask).any())
        if touches_panel and operation not in ('source_replay', 'support_count', 'check_repair', 'verify'):
            raise PermissionError(f'operation {operation} may not read repair-panel labels')
        if touches_panel and operation in ('check_repair', 'verify') and not repair_frozen():
            raise PermissionError('repair CHECK panels stay closed until the committed REPAIR_FREEZE')
    elif pol['rows'] != 'ALL' and pol['rows'] != ('FIT', 'GUARD', 'CHECK', 'ALL'):
        raise PermissionError(f'operation {operation} reads MODEL_FIT partition rows only')
    for name in labels:
        out[name] = _label(role, name, out)
    keep = ('X', 'feature_names', 'row_id', 'unit', 'raw_inputs') + labels
    res = {k: (v[rows] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in out.items() if k in keep}
    if 'raw_inputs' in res:
        res['raw_inputs'] = {k: v[rows] for k, v in res['raw_inputs'].items()}
    res['rows'] = rows
    _log({'role': role, 'fields': sorted(labels), 'operation': operation, 'purpose': purpose, 'partition': partition,
          'prole': prole, 'people': int(rows.sum()), 'households': int(len(np.unique(unit[rows]))),
          'rows_sha256': arr_sha(np.asarray(out['row_id'][rows])), 'entry_point': entry,
          'touches_repair_panel': bool(panel_mask is not None and (rows & panel_mask).any())})
    return res


def access_events():
    if not LABEL_LOG.is_file():
        return []
    return [json.loads(l) for l in LABEL_LOG.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ small IO helpers
def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float) + '\n')


def write_json_x(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=float)


def unit_stats(**counts):
    """Accumulate the unit's own fit counters (read by cts.jobqueue at unit end)."""
    p = os.environ.get(STATS_ENV)
    if not p:
        return
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.is_file() else {}
    for k, v in counts.items():
        cur[k] = cur.get(k, 0) + v
    p.write_text(json.dumps(cur, sort_keys=True))
