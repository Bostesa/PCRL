"""Unit UM_BANK: assemble the final evaluation bank (PROTOCOL.md section 8) from the committed SELECTION_LOCK and hash it.

Per seed (seed s pairs only with seed s): one service per effective release among the ten roles (roles sharing a release are
aliases of one service, named by its first role in the fixed order); outputs on the four output roles rebuilt from the locked
release tables and the admitted routing; frozen NEURAL_RAW, H_LJ, U0 and Ucal copied bitwise from the admitted frozen bank.
Checks: row order, float64 probability convention, frozen bitwise, HIST_SCM and C0 bitwise equal to the historical outputs.
Each recipient receives only its own float64 probability vector and argmax decision; the coalition gets the aligned pair.
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from scm_bench import engine as E
from . import common as C
from .select import load_traj

FROZEN = ('NEURAL_RAW', 'H_LJ', 'U0', 'Ucal')


def build(seed, sel, tl):
    rec, tabs = load_traj(seed, tl)
    route = C.routing(seed)
    roles = {r: C.load_role(r, ('features',)) for r in C.OUT_ROLES}
    frozen, hids = C.frozen_bank(seed)
    hist, _ = C.source_arm_outputs(seed, arms={'SCM_b1.0', 'C0'})
    svc_roles = sel['services'][str(seed)]['aliases']
    rmap = sel['roles'][str(seed)]
    bank, svc_release = {}, {}
    for svc, rnames in svc_roles.items():
        rid = rmap[rnames[0]]['release']
        if str(rid).startswith('FROZEN_'):
            continue
        svc_release[svc] = rid
        bank[svc] = {r: E.release(tabs[rid], route[f'init__{r}']) for r in C.OUT_ROLES}
    for svc in FROZEN:
        bank[svc] = {r: {t: np.array(frozen[svc][r][t], dtype=np.float64, copy=True) for t in C.TASKS} for r in C.OUT_ROLES}
    checks = {'row_order_matches_exports': all(np.array_equal(route[f'row_id__{r}'], roles[r]['row_id']) and np.array_equal(hids[f'row_id__{r}'], roles[r]['row_id'])
                                               for r in C.OUT_ROLES),
              'frozen_bitwise_equal_to_hybrids_bank': all(np.array_equal(bank[s][r][t], frozen[s][r][t]) for s in FROZEN for r in C.OUT_ROLES for t in C.TASKS)}
    hs = sel['services'][str(seed)]['service_of_role']
    checks['HIST_SCM_bitwise_equal_to_historical_SCM_b1.0'] = all(np.array_equal(bank[hs['HIST_SCM']][r][t], hist['SCM_b1.0'][r][t]) for r in C.OUT_ROLES for t in C.TASKS)
    checks['C0_bitwise_equal_to_historical_C0'] = all(np.array_equal(bank[hs['C0']][r][t], hist['C0'][r][t]) for r in C.OUT_ROLES for t in C.TASKS)
    conv = True
    for svc in bank:
        for r in C.OUT_ROLES:
            for t in C.TASKS:
                p = bank[svc][r][t]
                conv &= bool(p.dtype == np.float64 and p.shape == (len(roles[r]['row_id']), C.K[t]) and np.isfinite(p).all()
                             and (p >= 0).all() and np.allclose(p.sum(1), 1.0, atol=1e-9, rtol=0))
    checks['float64_probability_vectors_original_class_order'] = conv
    checks['at_most_ten_attacked_releases'] = len([s for s in bank if s not in ('U0', 'Ucal')]) <= 10
    wires = {f'{svc}|{t}': {r: C.wire_sha(bank[svc][r][t]) for r in C.OUT_ROLES} for svc in bank for t in C.TASKS}
    return bank, roles, checks, wires, svc_release


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    for name in ('SELECTION_LOCK.json', 'TRAJECTORY_COMPLETE_LOCK.json'):
        if not C.committed(C.RESULTS / name):
            raise PermissionError(f'bank requires the committed {name}')
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    tl = json.loads((C.RESULTS / 'TRAJECTORY_COMPLETE_LOCK.json').read_text())
    (C.PRIV / 'bank').mkdir(parents=True, exist_ok=True, mode=0o700)
    man = {'schema': 'SCMUM1-service-bank-manifest-v1', 'utc': C.now(), 'selection_lock_sha256': C.sha(C.RESULTS / 'SELECTION_LOCK.json'),
           'frozen': list(FROZEN), 'release_contract': 'each recipient: its own float64 probability vector and argmax decision; coalition: the aligned pair; nothing else',
           'per_seed': {}, 'bank_files_sha256': {}}
    ok = True
    for s in C.SEEDS:
        bank, roles, checks, wires, svc_release = build(s, sel, tl)
        arrays = {f'{svc}__{r}__{t}': p for svc, rr in bank.items() for r, tt in rr.items() for t, p in tt.items()}
        for r in C.OUT_ROLES:
            arrays[f'row_id__{r}'] = roles[r]['row_id']; arrays[f'unit__{r}'] = roles[r]['unit']
        p = C.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz'
        if p.exists():
            raise PermissionError('bank file exists (append-only)')
        with open(p, 'xb') as f:
            np.savez(f, **arrays)
        p.chmod(0o600)
        man['bank_files_sha256'][f'S{s}'] = C.sha(p)
        man['per_seed'][f'S{s}'] = {'checks': checks, 'wire_sha256': wires, 'service_release': svc_release,
                                    'service_of_role': sel['services'][str(s)]['service_of_role'], 'aliases': sel['services'][str(s)]['aliases'],
                                    'missing_roles': sel['services'][str(s)]['missing_roles'],
                                    'attacked_services': [svc for svc in bank if svc not in ('U0', 'Ucal')]}
        ok &= all(checks.values())
    man['status'] = 'BANK_FROZEN' if ok else 'BANK_CHECK_FAILED'
    with open(a.out, 'x') as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(json.dumps({'status': man['status'], 'bank_sha256': man['bank_files_sha256'], 'checks': {k: v['checks'] for k, v in man['per_seed'].items()},
                      'services': {k: v['attacked_services'] for k, v in man['per_seed'].items()}}, indent=1))
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
