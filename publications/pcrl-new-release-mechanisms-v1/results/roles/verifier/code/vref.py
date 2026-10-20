"""Label-free loaders of the frozen references and of this campaign's saved releases, with the verifier's own hash checks.

- frozen_bank_ms(v): the frozen bank file read directly (np.load), its SHA-256 checked against BOTH the occupation-headroom
  SOURCE_ADMISSION.json copy record and FROZEN_BANK_MANIFEST.json; MODEL_SELECT row ids checked against the admitted features.
- reference(name, v): 'W' (A1_PO_16 S<v> occupation + H3 income S<v>) and 'REF:<name>'; frozen-bank members come from the direct
  load, the others (B2TWIN, RETIRED_B3CTL, SUCC, TWJ) and W's parts through the source campaigns' admitted-copy DATA loaders
  (ohn.devrun.reference_release, tsz.evaluate.source_a1 / h3_income; no metric code); every MODEL_SELECT wire is checked against this
  campaign's SOURCE_ADMISSION.json wire hashes where recorded.
- release(family, config, v): RELEASE.npz read directly, SHA-256 checked against RELEASE.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

IHS_MIRROR = V.ROOT / '.private' / 'pcrl_income_headroom_successor_v1'
OHN_ADMISSION = V.ROOT / 'results' / 'pcrl_occupation_headroom_overnight_v1' / 'SOURCE_ADMISSION.json'
BANK_MANIFEST = V.ROOT / 'results' / 'pcrl_acs_frozen_hybrids_v1' / 'FROZEN_BANK_MANIFEST.json'
NRM_ADMISSION = V.RESULTS / 'SOURCE_ADMISSION.json'
FROZEN = ('U0', 'Ucal', 'T_TREE', 'TASK_REFIT', 'NEURAL_RAW', 'H_LJ', 'H_JJ', 'H_LL', 'H_JL')
_bank = {}


def ms_row_id():
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    import nrm.common as C
    return np.asarray(C.features('MODEL_SELECT')['row_id'])


def frozen_bank_ms(v):
    if v in _bank:
        return _bank[v]
    rel = f'admitted/bank/FROZEN_BANK_S{v}.npz'
    p = IHS_MIRROR / rel
    h = V.file_sha(p)
    adm = json.loads(OHN_ADMISSION.read_text())['copies'][rel]['sha256']
    man = json.loads(BANK_MANIFEST.read_text())['bank_files_sha256'][f'S{v}']
    if not (h == adm == man):
        raise PermissionError(f'frozen bank S{v}: file {h} admission {adm} manifest {man}')
    out = {}
    with np.load(p, allow_pickle=False) as z:
        rid = z['row_id__MODEL_SELECT'].copy()
        for k in z.files:
            parts = k.split('__')
            if len(parts) == 3 and parts[1] == 'MODEL_SELECT':
                out.setdefault(parts[0], {})[parts[2]] = z[k].copy()
    if not np.array_equal(rid, ms_row_id()):
        raise PermissionError('frozen bank MODEL_SELECT row order differs from the admitted features')
    _bank[v] = (out, {'bank_file': str(p.relative_to(V.ROOT)), 'bank_sha256': h})
    return _bank[v]


def admitted_wire(name, v):
    adm = json.loads(NRM_ADMISSION.read_text())['reference_releases_MODEL_SELECT_wire_sha256']
    return adm.get(f'{name}/S{v}')


def reference(name, v):
    """({'income', 'occupation'} MODEL_SELECT arrays, provenance with wire checks)."""
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    bank, bprov = frozen_bank_ms(v)
    if name == 'W':
        from tsz.evaluate import source_a1, h3_income
        rel, short = {'income': h3_income(v), 'occupation': source_a1(v)}, 'W'
    else:
        short = name[4:] if name.startswith('REF:') else name
        if short in FROZEN:
            rel = {t: bank[short][t] for t in V.TASKS}
        else:
            from ohn.devrun import reference_release
            rel = reference_release(short, v, 'MODEL_SELECT')
    rel = {t: np.asarray(rel[t], dtype=np.float64) for t in V.TASKS}
    exp = admitted_wire(short, v)
    wires = {t: V.wire_sha(rel[t]) for t in V.TASKS}
    prov = {'name': name, 'wire_sha256': wires, 'admitted_wire_sha256': {t: exp[t] for t in V.TASKS} if exp else None,
            'wire_equals_admission': (all(wires[t] == exp[t] for t in V.TASKS) if exp else None), **bprov}
    return rel, prov


def release(family, config, v, roles=('MODEL_SELECT',)):
    d = V.RUNS / family / config / f'V{int(v)}'
    rec = json.loads((d / 'RELEASE.json').read_text())
    h = V.file_sha(d / 'RELEASE.npz')
    if h != rec['release_sha256']:
        raise PermissionError(f'{family}/{config}/V{v}: RELEASE.npz {h} differs from its record')
    out = {}
    with np.load(d / 'RELEASE.npz', allow_pickle=False) as z:
        for k in z.files:
            role, task = k.split('__')
            if roles is None or role in roles:
                out.setdefault(role, {})[task] = z[k].copy()
    return out, rec, {'release_dir': str(d.relative_to(V.ROOT)), 'release_sha256': h}


def comparator(spec, v):
    """An nrm.evaluate comparator spec: 'W', 'REF:<name>' or 'FAMILY/CONFIG' (MODEL_SELECT arrays)."""
    if spec == 'W' or spec.startswith('REF:'):
        rel, prov = reference(spec, v)
        return rel, prov
    fam, cfg = spec.split('/')
    arr, rec, prov = release(fam, cfg, v)
    return arr['MODEL_SELECT'], {**prov, 'kind': rec.get('kind')}
