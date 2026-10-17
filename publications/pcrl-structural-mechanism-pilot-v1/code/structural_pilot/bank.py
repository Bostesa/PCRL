"""Unit BANK: assemble this campaign's SERVICE_BANK for readers and the report, and hash it.

Per model seed s (seed s pairs only with seed s; seeds are never averaged):
- fitted arms of this campaign (the nominee, its task-only twin and any other arm listed in ANALYSIS_LOCK), read from the
  hashed arm-output files written by the fit units (`runs/FIT_{s}/ARM_OUTPUTS.npz`);
- frozen comparators copied bitwise from the frozen-hybrids bank (hash-checked against the public FROZEN_BANK_MANIFEST):
  NEURAL_RAW (selected rho1.0 E2), TASK_REFIT (E20), T_TREE, H_LL (= LOCAL_TREE), H_LJ, H_JJ (= JOINT_TREE), U0 and Ucal;
- utility-ineligible published baselines (diagnostic only) from the restored source outputs: FARE_F1 and FAGTB_lam0.03.
Each recipient receives only its own float64 probability vector; its decision is the argmax. Nothing else is appended.
Checks: row order per role equals the export order; probability convention; frozen comparators bitwise equal to the
hybrids bank; unique single-recipient wire hashes.
"""
from __future__ import annotations
import argparse
import json

import numpy as np

from . import common as C

ROLES4 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'MODEL_SELECT')
FROZEN = ('NEURAL_RAW', 'TASK_REFIT', 'T_TREE', 'H_LL', 'H_LJ', 'H_JJ', 'U0', 'Ucal')
BASELINES = {'FARE_F1': 'FARE_F1', 'FAGTB': 'FAGTB_lam0.03'}


def arm_outputs(seed):
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    rel = f'runs/FIT_{seed}/ARM_OUTPUTS.npz'; p = C.PRIV / rel
    if C.sha(p) != sel['arm_output_files_sha256'][rel]:
        raise PermissionError('arm output hash differs from SELECTION_LOCK')
    out, ids = {}, {}
    with np.load(p) as z:
        for key in z.files:
            parts = key.split('__')
            if parts[0] in ('row_id', 'unit'):
                ids[key] = z[key].copy(); continue
            arm, role, task = parts
            out.setdefault(arm, {}).setdefault(role, {})[task] = np.asarray(z[key], dtype=np.float64)
    return out, ids


def build(seed, arm_map):
    """arm_map: {bank service name: fitted arm name} for this campaign's arms (from SELECTION_LOCK)."""
    roles = {r: C.load_role(r, ('features',)) for r in ROLES4}
    fitted, fids = arm_outputs(seed)
    frozen, hids = C.frozen_bank(seed, roles=ROLES4)
    base, _ = C.source_outputs(seed, roles=ROLES4, baselines=True)
    bank, checks = {}, {}
    for svc, arm in arm_map.items():
        bank[svc] = {r: {t: fitted[arm][r][t] for t in C.TASKS} for r in ROLES4}
    for svc in FROZEN:
        bank[svc] = {r: {t: np.array(frozen[svc][r][t], dtype=np.float64, copy=True) for t in C.TASKS} for r in ROLES4}
    for svc, src in BASELINES.items():
        bank[svc] = {r: {t: np.array(base[src][r][t], dtype=np.float64, copy=True) for t in C.TASKS} for r in ROLES4}
    checks['row_order_matches_exports'] = all(np.array_equal(fids[f'row_id__{r}'], roles[r]['row_id']) and np.array_equal(hids[f'row_id__{r}'], roles[r]['row_id'])
                                              for r in ROLES4)
    checks['frozen_bitwise_equal_to_hybrids_bank'] = all(np.array_equal(bank[s][r][t], frozen[s][r][t]) for s in FROZEN for r in ROLES4 for t in C.TASKS)
    conv = True
    for svc in bank:
        for r in ROLES4:
            for t in C.TASKS:
                p = bank[svc][r][t]
                conv &= bool(p.dtype == np.float64 and p.shape == (len(roles[r]['row_id']), C.K[t]) and np.isfinite(p).all()
                             and (p >= 0).all() and np.allclose(p.sum(1), 1.0, atol=1e-9, rtol=0))
    checks['float64_probability_vectors_original_class_order'] = conv
    wires = {f'{svc}|{t}': {r: C.wire_sha(bank[svc][r][t]) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')} for svc in bank for t in C.TASKS}
    return bank, roles, checks, wires


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    sel = json.loads((C.RESULTS / 'SELECTION_LOCK.json').read_text())
    if not C.committed(C.RESULTS / 'SELECTION_LOCK.json'):
        raise PermissionError('bank requires the committed SELECTION_LOCK')
    arm_map = sel['bank_arm_map']
    (C.PRIV / 'bank').mkdir(parents=True, exist_ok=True, mode=0o700)
    man = {'schema': 'SMP1-service-bank-manifest-v1', 'utc': C.now(), 'arm_map': arm_map, 'frozen': list(FROZEN), 'baselines_diagnostic': BASELINES,
           'release_contract': ('each recipient: its own float64 probability vector and argmax decision; coalition: aligned concatenation of the two '
                                'recipient wires; nothing else'), 'per_seed': {}, 'bank_files_sha256': {}}
    ok = True
    for s in C.SEEDS:
        bank, roles, checks, wires = build(s, arm_map)
        arrays = {f'{svc}__{r}__{t}': p for svc, rr in bank.items() for r, tt in rr.items() for t, p in tt.items()}
        for r in ROLES4:
            arrays[f'row_id__{r}'] = roles[r]['row_id']; arrays[f'unit__{r}'] = roles[r]['unit']
        p = C.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz'
        if p.exists():
            raise PermissionError('bank file exists (append-only)')
        with open(p, 'xb') as f:
            np.savez(f, **arrays)
        p.chmod(0o600)
        man['bank_files_sha256'][f'S{s}'] = C.sha(p)
        man['per_seed'][f'S{s}'] = {'checks': checks, 'wire_sha256': wires, 'rows': {r: int(len(roles[r]['row_id'])) for r in ROLES4}}
        ok &= all(checks.values())
    man['status'] = 'BANK_FROZEN' if ok else 'BANK_CHECK_FAILED'
    with open(a.out, 'x') as f:
        json.dump(man, f, indent=2, sort_keys=True)
    print(json.dumps({'status': man['status'], 'bank_sha256': man['bank_files_sha256'], 'checks': {k: v['checks'] for k, v in man['per_seed'].items()}}, indent=1))
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
