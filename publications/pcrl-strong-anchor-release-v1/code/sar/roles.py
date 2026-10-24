"""Version-specific role members of the PCRL strong-anchor release campaign (label-free; coordinator).

The shared SPEND and RO path points are saved once under the family CTL with strength-named configuration ids (card CTL_PATHS_v1:
CTL_DQ_<step>, CTL_RO_<strength>), so a family's bracket point of one path can be a different configuration in each version. A role
member names that point by its role, for example `IA_SPD_LO` = {version 0: CTL/CTL_DQ_0p750000, version 1: CTL/CTL_DQ_0p825000},
written to `results/<campaign>/paths/SHARED_ROLES.json` by `sar.plan` from the family fit records before any grid output is read.

    spec 'ROLE:<name>'   resolved per version by `resolve(spec, v)` -> 'CTL/<config>' (or None when the role is absent in v)
"""
from __future__ import annotations
import json
from pathlib import Path

from . import common as C

ROLES_PATH = C.RESULTS / 'paths' / 'SHARED_ROLES.json'


def role_map(path=ROLES_PATH):
    p = Path(path)
    return json.loads(p.read_text())['roles'] if p.is_file() else {}


def resolve(spec, v, path=ROLES_PATH):
    """'ROLE:<name>' -> the 'FAMILY/CONFIG' of that role in version v; any other spec is returned unchanged."""
    if not str(spec).startswith('ROLE:'):
        return spec
    m = role_map(path).get(spec[5:])
    if m is None:
        raise KeyError(f'unknown role member {spec}')
    return m.get(str(int(v)))


def array_member(name, path=ROLES_PATH):
    """The devreaders 'arrays' member of a role: per version, the MODEL_SELECT arrays of the role's configuration (hash-pinned)."""
    from . import release as RL
    m = role_map(path)[name]
    seeds = {}
    for v, spec in m.items():
        fam, cfg = spec.split('/')
        d = RL.rdir(fam, cfg, int(v))
        sha = json.loads((d / 'RELEASE.json').read_text())['release_sha256']
        rel = str((d / 'RELEASE.npz').relative_to(C.ROOT))
        seeds[str(v)] = {'income': {'path': rel, 'key': 'MODEL_SELECT__income', 'sha256': sha},
                         'occupation': {'path': rel, 'key': 'MODEL_SELECT__occupation', 'sha256': sha}}
    return {'type': 'arrays', 'seeds': seeds, 'role': name}
