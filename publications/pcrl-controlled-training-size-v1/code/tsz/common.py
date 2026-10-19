"""Shared definitions and the single logged label-access layer of the PCRL controlled training-size campaign
(`pcrl_controlled_training_size_v1`; EXPLORATORY DEVELOPMENT on previously used records, not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed), in the development roles of the protected-tree replication. The service predicts an income class (2) and an
occupation group (5 official broad civilian groups) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column
encoding X). Each recipient receives only its own float64 probability vector and argmax decision. SEX is the recorded demographic
column (sex 0 = SEX code 1, men; sex 1 = SEX code 2, women): a training-time label on MODEL_FIT and an evaluation label on
MODEL_SELECT for development readers; never a runtime input or output.

Sources (admitted read-only, by hash; `tsz.admit`): the correction-transfer campaign (publication e32a185b4, research tip 507024649;
its code `cts.*` is imported unchanged where reused), the occupation-headroom campaign (publication 1435c395d; private copy
.private/pcrl_occupation_headroom_overnight_v1/, re-checked by ohn.common.admitted) and the H3 income source (publication 89fd2f6e6;
verified mirror .private/pcrl_income_headroom_successor_v1/).

Label access is defined by ROLE x FIELD x OPERATION x PURPOSE (POLICY below), never by script name. Every label read, by every
entry point, goes through `load`, which refuses reads the policy does not allow and appends one line to
.private/<campaign>/access/LABEL_ACCESS.jsonl (utc, unit id, entry point, role, fields, operation, purpose, subset, people,
households, row-set hash).
- No label is read before the registration is committed and its sanitized public receipt is verified on the GitHub remote.
- ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT labels are refused for the whole campaign (features stay available label-free for
  deployment parity). The sealed assessment and capped-out records are refused by every loader.
- MODEL_FIT labels are released only for the rows of the requested registered subset (S50_O<o>, S70_O<o>, S100).
"""
from __future__ import annotations
import datetime as dt
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np

from scm_bench import common as SB
from ohn import common as OC                 # admitted-copy hash checks (SOURCE_ADMISSION.json of the source campaign)

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = 'pcrl_controlled_training_size_v1'
PRIV = ROOT / '.private' / CAMPAIGN
RUNS = PRIV / 'runs'
RESULTS = ROOT / 'results' / CAMPAIGN
ACCESS = PRIV / 'access'
LABEL_LOG = ACCESS / 'LABEL_ACCESS.jsonl'
REG_RECEIPT = ACCESS / 'REGISTRATION_RECEIPT_VERIFIED.json'
RECEIPT_ATTEMPTS = ACCESS / 'RECEIPT_ATTEMPTS.jsonl'
UNIT_ENV = 'TSZ_UNIT_ID'
STATS_ENV = 'TSZ_UNIT_STATS'
CTS_PRIV = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
CATBOOST_PY = CTS_PRIV / 'venv_catboost' / 'bin' / 'python'       # CatBoost 1.2.8 (fit-time only; inference is numpy)
TORCH_PY = Path('<repository>/.venv/bin/python')       # torch (Stage B task refit, corrections)

CTS_PUBLICATION = 'e32a185b4fe921cba474f392659f493d0658b771'
CTS_TIP = '507024649'
OHN_CAMPAIGN = 'pcrl_occupation_headroom_overnight_v1'
OHN_PUBLICATION = '1435c395d3f753a9bf0b553d14d264ca9e991421'
IHS_PUBLICATION = '89fd2f6e649547f65cf1b69a9e8e8b51f854d25e'
PUBLICATION_BRANCH = 'publication/pcrl-controlled-training-size-v1'
PUBLICATION_NAME = 'pcrl-controlled-training-size-v1'
REMOTE_URL = 'https://github.com/Bostesa/PCRL.git'

TASKS, K, CAPS, SEEDS = SB.TASKS, SB.K, SB.CAPS, SB.SEEDS
VERSIONS = (0, 1)                             # source versions = source seeds (procedure seeds, references, H3 income)
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


# ------------------------------------------------------------------ training subsets (label-free; frozen in the registration)
ORDERINGS = ('O0', 'O1', 'O2', 'O3')
ORDER_TAG = 'TSZ1_order'                     # households sorted by SHA-256('TSZ1_order_<o>|<household id>')
FRACTIONS = (50, 70, 100)
NCV_K = 10
NCV_TAG = 'TSZ1_ncv10'                       # outer folds of the nested strength-selection estimator (conditional stage)


def subset_ids():
    return tuple(f'S{f}_{o}' for f in (50, 70) for o in ORDERINGS) + ('S100',)


def household_order(groups, o):
    keys = [hashlib.sha256(f'{ORDER_TAG}_{o}|{g}'.encode()).hexdigest() for g in groups]
    return np.asarray(groups)[np.argsort(keys)]


def n_households(G, f):
    return G if f == 100 else int(math.ceil(f / 100.0 * G))


def subset_mask(unit, sid):
    """Boolean mask over MODEL_FIT people of the registered subset `sid` (whole households; nested within an ordering)."""
    groups = np.unique(unit)
    if sid == 'S100':
        return np.ones(len(unit), dtype=bool)
    f, o = sid[1:].split('_')
    f = int(f)
    if f not in (50, 70) or o not in ORDERINGS:
        raise ValueError(f'unknown subset {sid}')
    keep = household_order(groups, o)[:n_households(len(groups), f)]
    return np.isin(unit, keep)


def ncv_folds(unit):
    return OC.household_folds(unit, NCV_K, NCV_TAG)


def check_subsets(unit):
    """Label-free structural checks: whole households, nesting S50 < S70 < S100 within every ordering, exact sizes."""
    problems = []
    G = len(np.unique(unit))
    for o in ORDERINGS:
        m50, m70 = subset_mask(unit, f'S50_{o}'), subset_mask(unit, f'S70_{o}')
        if (m50 & ~m70).any():
            problems.append(f'{o}: S50 not inside S70')
        for f, m in ((50, m50), (70, m70)):
            hh = set(np.unique(unit[m])); others = set(np.unique(unit[~m]))
            if hh & others:
                problems.append(f'{o}: S{f} splits a household')
            if len(hh) != n_households(G, f):
                problems.append(f'{o}: S{f} has {len(hh)} households')
    return problems


def subset_summary(unit):
    out = {}
    hh = {}
    for sid in subset_ids():
        m = subset_mask(unit, sid)
        h = np.unique(unit[m]); hh[sid] = set(h.tolist())
        out[sid] = {'people': int(m.sum()), 'households': int(len(h)),
                    'household_set_sha256': hashlib.sha256(','.join(map(str, h)).encode()).hexdigest()}
    ov = {}
    for f in (50, 70):
        for i, a in enumerate(ORDERINGS):
            for b in ORDERINGS[i + 1:]:
                A, B = hh[f'S{f}_{a}'], hh[f'S{f}_{b}']
                ov[f'S{f}_{a}|S{f}_{b}'] = {'shared_households': len(A & B), 'jaccard': len(A & B) / len(A | B)}
    return {'subsets': out, 'pairwise_overlap': ov, 'model_fit_people': int(len(unit)),
            'model_fit_households': int(len(np.unique(unit)))}


# ------------------------------------------------------------------ registration state
REGISTRATION_FILES = ('PROTOCOL.md', 'SUBSETS.json', 'EXPOSURE_MAP.md', 'ACCESS_POLICY.json', 'SOURCE_ADMISSION.json',
                      'READER_SPLITS.json', 'DEVGRID_CODE_LOCK.json', 'roles/statistical_reviewer/RULES.md',
                      'roles/statistical_reviewer/RULES.json')


def _receipt_ok(path, files):
    if not path.is_file():
        return False
    rec = json.loads(path.read_text())
    if not rec.get('remote_sha_verified') or rec.get('remote_sha') != rec.get('receipt_commit'):
        return False
    return all(rec.get('files_sha256', {}).get(f) == sha(RESULTS / f) for f in files)


def registered():
    return all(committed(RESULTS / f) for f in REGISTRATION_FILES) and _receipt_ok(REG_RECEIPT, REGISTRATION_FILES)


# ------------------------------------------------------------------ access policy (ROLE x FIELD x OPERATION x PURPOSE)
POLICY = {
    'fit': {'roles': ('MODEL_FIT',), 'fields': ('occupation', 'income', 'sex'), 'rows': 'SUBSET',
            'what': 'task learning on a registered subset (A1_PO_16; T_TREE income / occupation teachers; Stage B TASK_REFIT, whose '
                    'warm critics read SEX as in the source loop); full-pool correction fitting on S100 (witness cross-fitting base, '
                    'routing, CONT / LOC / PAIR critics with SEX as a training label)'},
    'calibrate': {'roles': ('CALIBRATION',), 'fields': ('occupation',), 'rows': 'ALL',
                  'what': 'the A1_PO_16 diagnostic temperature on CALIBRATION (recorded, never applied, as in the source)'},
    'select_ncv': {'roles': ('MODEL_FIT',), 'fields': ('occupation', 'income', 'sex'), 'rows': 'SUBSET', 'subsets': ('S100',),
                   'what': 'nested 10-fold household cross-fitting of the conditional strength rule: every upstream layer refitted on '
                           '90%, out-of-fold incremental task and within-SEX costs on the held-out 10% (fitting information only)'},
    'benchmark': {'roles': ('MODEL_SELECT',), 'fields': ('income', 'occupation', 'sex'), 'rows': 'ALL',
                  'what': 'original-benchmark utility (24 clauses + 4 guards), the development reserve, paired size contrasts, the '
                          'registered task-stage trigger, realized costs of selected releases (never a strength-rule input)'},
    'readers': {'roles': ('MODEL_SELECT',), 'fields': ('sex',), 'rows': 'ALL',
                'what': 'fresh development reader grid (FIT / VALIDATION / SCORE reader roles inside MODEL_SELECT)'},
    'verify': {'roles': ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT'), 'fields': ('income', 'occupation', 'sex'), 'rows': 'ANY',
               'what': 'independent verifier replay with separate metric, subset and selection code'},
}

ENTRY_POINTS = {      # operation -> authorized entry points (module names, or repository-relative script paths for the verifier)
    'fit': ('tsz.sizefit', 'tsz.control', 'tsz.correct'),
    'calibrate': ('tsz.sizefit',),
    'select_ncv': ('tsz.nested',),
    'benchmark': ('tsz.evaluate', 'tsz.correct'),
    'readers': ('tsz.devreaders',),
    'verify': ('results/pcrl_controlled_training_size_v1/roles/verifier/code/',),
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
    """Label-free admitted features (any admitted role, any operation; no log needed)."""
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


def load(role, fields, operation, purpose, subset=None):
    """Labels (and features) of one role under the access policy. MODEL_FIT reads return ONLY the rows of the registered subset
    `subset` (S50_O<o>, S70_O<o>, S100). Returns a dict with X, feature_names, row_id, unit, the requested labels and 'rows'
    (boolean mask over the role's people)."""
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
    if role == 'MODEL_FIT':
        if pol['rows'] == 'ANY' and subset is None:
            rows = np.ones(n, dtype=bool)
        else:
            if subset not in subset_ids():
                raise PermissionError(f'operation {operation} needs a registered subset (got {subset!r})')
            if pol.get('subsets') and subset not in pol['subsets']:
                raise PermissionError(f'operation {operation} is not allowed on subset {subset}')
            rows = subset_mask(unit, subset)
    elif subset is not None:
        raise PermissionError(f'{role} has no subsets (the log would misstate the rows)')
    for name in labels:
        out[name] = _label(role, name, out)
    keep = ('X', 'feature_names', 'row_id', 'unit', 'raw_inputs') + labels
    res = {k: (v[rows] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in out.items() if k in keep}
    if 'raw_inputs' in res:
        res['raw_inputs'] = {k: v[rows] for k, v in res['raw_inputs'].items()}
    res['rows'] = rows
    _log({'role': role, 'fields': sorted(labels), 'operation': operation, 'purpose': purpose, 'subset': subset,
          'people': int(rows.sum()), 'households': int(len(np.unique(unit[rows]))),
          'rows_sha256': arr_sha(np.asarray(out['row_id'][rows])), 'entry_point': entry})
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


def save_npz_x(path, **arrs):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path, 'xb') as f:
        np.savez(f, **arrs)
    return sha(path)


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def unit_stats(**counts):
    """Accumulate the unit's own fit counters (read by tsz.jobqueue at unit end)."""
    p = os.environ.get(STATS_ENV)
    if not p:
        return
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.is_file() else {}
    for k, v in counts.items():
        cur[k] = cur.get(k, 0) + v
    p.write_text(json.dumps(cur, sort_keys=True))
