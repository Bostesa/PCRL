"""Shared definitions and the single logged label-access layer of the PCRL finite-output solver-adequacy campaign
(`pcrl_finite_solver_adequacy_v1`; EXPLORATORY DEVELOPMENT on previously used records, not confirmation).

Census context: public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida (housing-unit residents aged 25-64,
civilian employed), in the development roles of the protected-tree replication. The service predicts an income class (2) and an
occupation group (5 official broad civilian groups) from six raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP; admitted 145-column
encoding X). Each recipient receives only its own float64 probability vector and its argmax decision. SEX is the dataset's recorded
sex column (sex 0 = SEX code 1, men; sex 1 = SEX code 2, women): a training-time label on MODEL_FIT only where a committed mechanism
card says so, an evaluation label on MODEL_SELECT for development readers; never a runtime input or output.

Derived from `frc.common` (the finite-routing comparison campaign, publication 04ac9161f, research tip 278cb1ed6 = the base of this
branch; itself derived from `nrm.common`) with this campaign's paths, names, entry points and limits. The source code (`frc.*`,
`nrm.*` and their sources) is imported unchanged where reused; the sources' private artefacts are read-only inputs pinned by hash in
SOURCE_ADMISSION.json.

Label access is defined by ROLE x FIELD x OPERATION x PURPOSE (POLICY below), never by script name. Every label read, by every
entry point, goes through `load`, which refuses reads the policy does not allow and appends one line to
.private/<campaign>/access/LABEL_ACCESS.jsonl (utc, unit id, entry point, role, fields, operation, purpose, people, households,
row-set hash, mechanism card).
- No label is read before the registration is committed and its sanitized public receipt is verified on the GitHub remote.
- Fitting reads ('fit', 'calibrate') also need a committed, unchanged mechanism card under results/<campaign>/cards/ (the card
  governing the fit is committed before the fit; its path and SHA-256 are logged with the read).
- ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT labels are refused for the whole campaign (features stay available label-free for
  deployment parity). The sealed assessment and capped-out records are refused by every loader.
- MODEL_SELECT labels are read only by the coordinator's evaluation and reader entry points and by the verifier: never by a fitting
  module.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from scm_bench import common as SB
from ohn import common as OC                 # admitted-copy hash checks (SOURCE_ADMISSION.json of the occupation-headroom source)

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN = 'pcrl_finite_solver_adequacy_v1'
PRIV = ROOT / '.private' / CAMPAIGN
RUNS = PRIV / 'runs'
RESULTS = ROOT / 'results' / CAMPAIGN
CARDS = RESULTS / 'cards'
ACCESS = PRIV / 'access'
LABEL_LOG = ACCESS / 'LABEL_ACCESS.jsonl'
REG_RECEIPT = ACCESS / 'REGISTRATION_RECEIPT_VERIFIED.json'
UNIT_ENV = 'FSA_UNIT_ID'
STATS_ENV = 'FSA_UNIT_STATS'
CTS_PRIV = ROOT / '.private' / 'pcrl_correction_transfer_stability_v1'
CATBOOST_PY = CTS_PRIV / 'venv_catboost' / 'bin' / 'python'       # CatBoost 1.2.8 (fit-time only)
TORCH_PY = Path('<repository>/.venv/bin/python')       # torch 2.10.0

# the source campaign (read-only inputs)
NRM_CAMPAIGN = 'pcrl_new_release_mechanisms_v1'
NRM_PUBLICATION = '682068d52147bc1bdfc75d9e49199115b1deaaed'
NRM_TIP = 'fd933a824'                                               # research tip of the source = the base of this branch
NRM_PRIV = ROOT / '.private' / NRM_CAMPAIGN
NRM_RUNS = NRM_PRIV / 'runs'
NRM_FA_CACHE = NRM_PRIV / 'fa_cache'
NRM_RESULTS = ROOT / 'results' / NRM_CAMPAIGN
# the finite-routing comparison campaign (read-only inputs: its repaired twin, protected releases, caches and records)
FRC_CAMPAIGN = 'pcrl_finite_routing_comparison_v1'
FRC_PUBLICATION = '04ac9161fa1dd32cbf5dea4ce2175a6b8f8cf067'
FRC_TIP = '278cb1ed6'                                               # its research tip = the base of this branch
FRC_PRIV = ROOT / '.private' / FRC_CAMPAIGN
FRC_RUNS = FRC_PRIV / 'runs'
FRC_FA_CACHE = FRC_PRIV / 'fa_cache'
FRC_RESULTS = ROOT / 'results' / FRC_CAMPAIGN

PUBLICATION_BRANCH = 'publication/pcrl-finite-solver-adequacy-v1'
PUBLICATION_NAME = 'pcrl-finite-solver-adequacy-v1'
REMOTE_URL = 'https://github.com/Bostesa/PCRL.git'
CAMPAIGN_START_UTC = '2026-10-21T02:22:58+00:00'   # first `date -u` of this campaign (memory notes in context before it)

TASKS, K, CAPS, SEEDS = SB.TASKS, SB.K, SB.CAPS, SB.SEEDS
VERSIONS = (0, 1)                             # source versions: training seeds / internal folds of version v; U0_S<v>, Ucal_S<v>
K_OCC = 5
GUARD_CAP = 0.005
MATERIAL_TARGET = 0.020
RAW_FIELDS = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
ADMITTED = OC.ADMITTED
LABELS = ('income', 'occupation', 'sex')
CLOSED_LABEL_ROLES = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
CONFIG_LIMIT = 32                              # new real-data release configurations (controls and genuine training repetitions included)

sha, arr_sha, wire_sha, task_vectors = SB.sha, SB.arr_sha, SB.wire_sha, SB.task_vectors
household_folds = OC.household_folds


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def committed(path):
    return SB.committed(path)


def git(*a):
    r = subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


# ------------------------------------------------------------------ registration state
REGISTRATION_FILES = ('PROTOCOL.md', 'EXPOSURE_MAP.md', 'ACCESS_POLICY.json', 'SOURCE_ADMISSION.json', 'READER_SPLITS.json',
                      'FAILURE_MAP.md', 'roles/statistical_reviewer/RULES.md', 'roles/statistical_reviewer/RULES.json')


def _receipt_ok(path, files):
    if not path.is_file():
        return False
    rec = json.loads(path.read_text())
    if not rec.get('remote_sha_verified') or rec.get('remote_sha') != rec.get('receipt_commit'):
        return False
    return all(rec.get('files_sha256', {}).get(f) == sha(RESULTS / f) for f in files)


def registered():
    return all(committed(RESULTS / f) for f in REGISTRATION_FILES) and _receipt_ok(REG_RECEIPT, REGISTRATION_FILES)


# ------------------------------------------------------------------ mechanism cards (committed before the fit they govern)
CARD_PREFIX = {'FA': 'FA_', 'CTL': 'CTL_', 'FS': 'FS_'}     # family -> card file prefix (FS: a successor family, if any)


def family_of_module(mod):
    """'fsa.fa.route' -> 'FA' (fitting entry points only), else None."""
    parts = str(mod).split('.')
    return parts[1].upper() if len(parts) > 1 and parts[0] == 'fsa' and parts[1].upper() in CARD_PREFIX else None


def card_info(card, family=None):
    """Committed, unchanged mechanism card under results/<campaign>/cards/: its repository path, SHA-256, commit and commit time.
    With `family`, the card's file name must carry that family's prefix; the template README.md is never a card."""
    if not card:
        raise PermissionError('a fitting read needs the committed mechanism card that governs the fit (card=...)')
    p = Path(card)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(CARDS.resolve()) or not p.is_file():
        raise PermissionError(f'{card}: mechanism cards live under {CARDS.relative_to(ROOT)}/')
    if p.name == 'README.md':
        raise PermissionError('cards/README.md is the template, not a mechanism card')
    if family is not None and not p.name.startswith(CARD_PREFIX.get(str(family).upper(), '\0')):
        raise PermissionError(f'{p.name} is not a card of family {family} (expected prefix {CARD_PREFIX.get(str(family).upper())})')
    rel = str(p.relative_to(ROOT))
    if not committed(p):
        raise PermissionError(f'{rel} is not committed (commit the card before the fit it governs)')
    log = git('log', '-1', '--format=%H %cI', '--', rel) or ''
    commit, when = (log.split() + [None, None])[:2]
    return {'card': rel, 'card_sha256': sha(p), 'card_commit': commit, 'card_commit_utc': when}


# ------------------------------------------------------------------ access policy (ROLE x FIELD x OPERATION x PURPOSE)
FIT_ENTRY = ('fsa.fa', 'fsa.ctl', 'fsa.fs')
POLICY = {
    'fit': {'roles': ('MODEL_FIT',), 'fields': ('occupation', 'income', 'sex'), 'card': True,
            'what': 'fitting release parameters of the family, its task-only twin or a control on the full MODEL_FIT pool (household-'
                    'grouped internal folds for any cross-fitting); SEX only as a training label of a protection objective (propensity '
                    'and SEX-conditional occupation statistics) or as the evaluation label of the MODEL_FIT cross-fitted check'},
    'calibrate': {'roles': ('CALIBRATION',), 'fields': ('occupation', 'income'), 'card': True,
                  'what': 'calibration parameters only (CALIBRATION has no SEX label); not planned by the registered card'},
    'benchmark': {'roles': ('MODEL_SELECT',), 'fields': ('income', 'occupation', 'sex'),
                  'what': 'original-benchmark utility (24 clauses + 4 guards), the development reserve and paired utility costs of saved '
                          'releases (logged adaptive evaluation; never a fitting input)'},
    'readers': {'roles': ('MODEL_SELECT',), 'fields': ('sex',),
                'what': 'fresh development reader grid (FIT / VALIDATION / SCORE reader roles inside MODEL_SELECT)'},
    'verify': {'roles': ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT'), 'fields': ('income', 'occupation', 'sex'),
               'what': 'independent verifier replay with separate metric, selection and reader code'},
}
ENTRY_POINTS = {      # operation -> authorized entry points (module names, or repository-relative script paths for the verifier)
    'fit': FIT_ENTRY,
    'calibrate': FIT_ENTRY,
    'benchmark': ('fsa.evaluate',),
    'readers': ('fsa.devreaders', 'fsa.readercustody'),
    'verify': (f'results/{CAMPAIGN}/roles/verifier/code/',),
}
AMENDMENTS = 'ACCESS_AMENDMENTS.json'      # dated, committed additions of entry points (before their first use)


def _caller():
    """(module or repository-relative script path, function) of the first frame outside this module."""
    for fr in inspect.stack()[2:14]:
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


def load(role, fields, operation, purpose, card=None):
    """Labels (and features) of one role under the access policy (all of the role's people). Returns a dict with X,
    feature_names, row_id, unit and the requested labels."""
    fields = tuple(fields)
    if role == 'SEALED_ASSESSMENT' or role not in ADMITTED:
        raise PermissionError(f'role {role} is not admitted')
    labels = tuple(f for f in fields if f in LABELS)
    if set(fields) - set(ADMITTED[role]) - {'features'}:
        raise PermissionError(f'{role}: fields {sorted(set(fields) - set(ADMITTED[role]))} are not admitted')
    out = features(role, raw='raw_inputs' in fields)
    n = len(out['row_id'])
    if not labels:
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
    chain = _module_chain()
    if role == 'MODEL_SELECT' and any(family_of_module(m) for m in chain):
        raise PermissionError(f'MODEL_SELECT labels are never read inside a fitting entry point (call chain {chain})')
    if not os.environ.get(UNIT_ENV):
        raise PermissionError('label reads run only inside a queue unit (fsa.jobqueue sets FSA_UNIT_ID)')
    cinfo = card_info(card, family=family_of_module(entry.split(':')[0])) if pol.get('card') else {}
    for name in labels:
        out[name] = _label(role, name, out)
    keep = ('X', 'feature_names', 'row_id', 'unit', 'raw_inputs') + labels
    res = {k: v for k, v in out.items() if k in keep}
    _log({'role': role, 'fields': sorted(labels), 'operation': operation, 'purpose': purpose, 'people': int(n),
          'households': int(len(np.unique(out['unit']))), 'rows_sha256': arr_sha(np.asarray(out['row_id'])), 'entry_point': entry,
          'module_chain': chain, **cinfo})
    return res


def _module_chain():
    """Every module on the call stack outside this module, plus the __main__ module's spec name (a fitting module calling a
    coordinator helper is still visible)."""
    mods = []
    for fr in inspect.stack()[1:40]:
        g = fr.frame.f_globals
        mod = g.get('__name__', '?')
        if mod == '__main__':
            spec = g.get('__spec__')
            mod = spec.name if spec is not None and getattr(spec, 'name', None) else str(g.get('__file__', '?'))
        if mod != __name__ and mod not in mods:
            mods.append(mod)
    return mods


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
    """Accumulate the unit's own fit counters (read by fsa.jobqueue at unit end)."""
    p = os.environ.get(STATS_ENV)
    if not p:
        return
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.is_file() else {}
    for k, v in counts.items():
        cur[k] = cur.get(k, 0) + v if isinstance(v, (int, float)) else v
    p.write_text(json.dumps(cur, sort_keys=True))
