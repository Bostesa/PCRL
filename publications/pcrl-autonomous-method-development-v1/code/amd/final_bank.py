"""Final release bank (only after a committed FINALIST_LOCK.json). Writes .private/.../bank/SERVICE_BANK_S<s>.npz per seed and
results/<campaign>/SERVICE_BANK_MANIFEST.json.

Each registered role in FINALIST_LOCK['bank'] names its source: a frozen release of the frozen-hybrids bank ('frozen'), a
release of the SCM utility-matched follow-up's bank ('um'), or a release file of this campaign ('release', family/config). Exact
aliases (identical income and occupation wires on every role) are collapsed into one attacked service; at most 8 effective
releases per seed. Roles: MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION, DEV_REPORT, each in the admitted row order. No label is
read here.
"""
from __future__ import annotations
import json

import numpy as np

from . import common as C
from . import devrun as DR

ROLES4 = ('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
MAX_EFFECTIVE = 8


def build():
    lock_p = C.RESULTS / 'FINALIST_LOCK.json'
    if not C.committed(lock_p):
        raise PermissionError('the final bank requires the committed FINALIST_LOCK.json')
    lock = json.loads(lock_p.read_text())
    ids_ref = {r: C.load_role(r, ('features',)) for r in ROLES4}
    man = {'schema': 'AMD1-service-bank-manifest-v1', 'utc': C.now(), 'finalist_lock_sha256': C.sha(lock_p),
           'release_contract': 'each recipient: its own float64 probability vector and argmax decision; coalition: the aligned pair; nothing else',
           'bank_files_sha256': {}, 'per_seed': {}}
    for s in C.SEEDS:
        frozen, fids = C.frozen_bank(s, roles=ROLES4)
        um = C.um_service_bank(s, roles=ROLES4)
        outs = {}
        for entry in lock['bank']:
            src = entry['source']
            if src['kind'] == 'frozen':
                o = {r: {t: np.asarray(frozen[src['name']][r][t], dtype=np.float64) for t in C.TASKS} for r in ROLES4}
            elif src['kind'] == 'um':
                o = {r: {t: np.asarray(um[src['name']][r][t], dtype=np.float64) for t in C.TASKS} for r in ROLES4}
            elif src['kind'] == 'release':
                rel, rec = DR.load_release(src['family'], src['config_id'], s, roles=ROLES4)
                want = lock['release_sha256'][f"{src['family']}/{src['config_id']}"][f'S{s}']
                if rec['release_sha256'] != want:
                    raise PermissionError(f"{src['family']}/{src['config_id']} S{s} release differs from FINALIST_LOCK")
                o = rel
            else:
                raise ValueError('unregistered bank source')
            for r in ROLES4:
                for t in C.TASKS:
                    p = o[r][t]
                    if p.shape != (len(ids_ref[r]['row_id']), C.K[t]) or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
                        raise AssertionError(f"{entry['role']} {r} {t}: contract or row count differs")
            outs[entry['role']] = o
        for r in ROLES4:
            if not np.array_equal(fids[f'row_id__{r}'], ids_ref[r]['row_id']):
                raise AssertionError(f'frozen bank row order differs on {r}')
        key_of = {role: tuple(C.wire_sha(o[r][t]) for r in ROLES4 for t in C.TASKS) for role, o in outs.items()}
        service_of_role, aliases, attacked = {}, {}, []
        for entry in lock['bank']:
            role = entry['role']
            match = next((svc for svc in attacked if key_of[svc] == key_of[role]), None)
            if match is None:
                attacked.append(role); aliases[role] = [role]; service_of_role[role] = role
            else:
                aliases[match].append(role); service_of_role[role] = match
        if len(attacked) > MAX_EFFECTIVE:
            raise AssertionError(f'{len(attacked)} effective releases exceed the registered maximum of {MAX_EFFECTIVE}')
        arrays = {}
        for svc in attacked:
            for r in ROLES4:
                for t in C.TASKS:
                    arrays[f'{svc}__{r}__{t}'] = np.ascontiguousarray(outs[svc][r][t], dtype=np.float64)
        for r in ROLES4:
            arrays[f'row_id__{r}'] = ids_ref[r]['row_id']; arrays[f'unit__{r}'] = ids_ref[r]['unit']
        p = C.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz'
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(p, 'xb') as f:
            np.savez(f, **arrays)
        p.chmod(0o600)
        man['bank_files_sha256'][f'S{s}'] = C.sha(p)
        man['per_seed'][f'S{s}'] = {'attacked_services': attacked, 'service_of_role': service_of_role, 'aliases': aliases,
                                    'wire_sha256': {f'{svc}|{t}': {r: C.wire_sha(outs[svc][r][t]) for r in ROLES4} for svc in attacked for t in C.TASKS},
                                    'checks': {'at_most_8_effective': len(attacked) <= MAX_EFFECTIVE, 'row_order_matches_exports': True,
                                               'float64_probability_vectors': True}}
    C.write_json_x(C.RESULTS / 'SERVICE_BANK_MANIFEST.json', man)
    print(json.dumps({s: v['attacked_services'] for s, v in man['per_seed'].items()}))


if __name__ == '__main__':
    build()
