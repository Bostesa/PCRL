"""Unit BANK: construct the fixed four-combination bank (plus external controls and utility references) and hash it.

Per existing model seed s (seed s pairs only with seed s; seeds are never averaged):
  H_LL = LOCAL income + LOCAL occupation      H_LJ = LOCAL income + JOINT occupation (PRIMARY)
  H_JL = JOINT income + LOCAL occupation      H_JJ = JOINT income + JOINT occupation
  controls: NEURAL_RAW (frozen selected rho1.0 E2), TASK_REFIT (matched E20), T_TREE (unchanged teacher, diagnostic)
  references (utility only, never attacked): U0, Ucal
Each recipient receives only its own probability vector; its decision is the argmax. The coalition receives the aligned
concatenation of the two recipients' wires. Nothing else is appended (no teacher outputs, leaf ids, hidden activations,
source-selection flags, record identifiers, weights or labels). Component choice is global, never per person.
Checks: bitwise identity of every recipient array to its chosen stored source output; H_LL and H_JJ reproduce the source
LOCAL_TREE and JOINT_TREE complete wires; each recipient swap changes exactly the intended wire; one row order per role
(the export order, verified at admission by recomputation from the frozen models); float64 probability vectors in the
original class order. The private bank files and every wire are hashed into the public FROZEN_BANK_MANIFEST.json.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from . import common as C


def build(seed):
    outs, _ = C.source_outputs(seed)
    roles = {r: C.load_role(r, ('features',)) for r in C.BANK_ROLES}
    bank, checks = {}, {}
    for svc, (src_i, src_o) in C.SOURCE_OF.items():
        bank[svc] = {r: {'income': np.array(outs[src_i][r]['income'], dtype=np.float64, copy=True),
                         'occupation': np.array(outs[src_o][r]['occupation'], dtype=np.float64, copy=True)} for r in C.BANK_ROLES}
    for ref in C.REFERENCES:
        bank[ref] = {r: {t: np.array(outs[ref][r][t], dtype=np.float64, copy=True) for t in C.TASKS} for r in C.BANK_ROLES}
    # 1. every recipient array is bitwise its chosen source output
    ident = {}
    for svc, src in C.SOURCE_OF.items():
        for r in C.BANK_ROLES:
            for t, s in zip(C.TASKS, src):
                st = outs[s][r][t]
                ident[f'{svc}|{r}|{t}<-{s}'] = bool(st.dtype == np.float64 and np.array_equal(bank[svc][r][t], st) and bank[svc][r][t].tobytes() == st.tobytes())
    checks['recipient_bitwise_source_identity'] = {'n': len(ident), 'all': all(ident.values()), 'failures': [k for k, v in ident.items() if not v]}
    # 2. H_LL / H_JJ reproduce the source complete wires
    checks['H_LL_equals_source_LOCAL_TREE_complete_wire'] = all(np.array_equal(bank['H_LL'][r][t], outs['LOCAL_TREE'][r][t]) for r in C.BANK_ROLES for t in C.TASKS)
    checks['H_JJ_equals_source_JOINT_TREE_complete_wire'] = all(np.array_equal(bank['H_JJ'][r][t], outs['JOINT_TREE'][r][t]) for r in C.BANK_ROLES for t in C.TASKS)
    # 3. a recipient swap changes exactly the intended wire
    def same(a, b, t):
        return all(np.array_equal(bank[a][r][t], bank[b][r][t]) for r in C.BANK_ROLES)
    swaps = {'H_LL->H_LJ (occupation swapped)': (same('H_LL', 'H_LJ', 'income'), not same('H_LL', 'H_LJ', 'occupation')),
             'H_LL->H_JL (income swapped)': (not same('H_LL', 'H_JL', 'income'), same('H_LL', 'H_JL', 'occupation')),
             'H_JJ->H_JL (occupation swapped)': (same('H_JJ', 'H_JL', 'income'), not same('H_JJ', 'H_JL', 'occupation')),
             'H_JJ->H_LJ (income swapped)': (not same('H_JJ', 'H_LJ', 'income'), same('H_JJ', 'H_LJ', 'occupation'))}
    checks['swap_changes_only_intended_wire'] = {k: bool(a and b) for k, (a, b) in swaps.items()}
    # 4. probability convention, class order, precision
    conv = True
    for svc in bank:
        for r in C.BANK_ROLES:
            for t in C.TASKS:
                p = bank[svc][r][t]
                conv &= bool(p.dtype == np.float64 and p.shape == (len(roles[r]['row_id']), C.K[t]) and np.isfinite(p).all()
                             and (p >= 0).all() and np.allclose(p.sum(1), 1.0, atol=1e-9, rtol=0))
    checks['float64_probability_vectors_original_class_order'] = conv
    # 5. unique single-recipient wires (reader aliasing) and per-service wire hashes
    wires, uniq = {}, {}
    for svc in C.ATTACK_ROSTER:
        for t in C.TASKS:
            h = {r: C.wire_sha(bank[svc][r][t]) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')}
            wires[f'{svc}|{t}'] = h
            uniq.setdefault((t, tuple(h.values())), []).append(svc)
    unique_wires = [{'task': t, 'services': svcs, 'source_service': C.SOURCE_OF[svcs[0]][C.TASKS.index(t)]} for (t, _), svcs in uniq.items()]
    checks['unique_single_recipient_wires'] = len(unique_wires)
    return bank, roles, checks, wires, unique_wires


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = Path(a.out); (C.PRIV / 'bank').mkdir(parents=True, exist_ok=True, mode=0o700)
    man = {'schema': 'ACSH1-frozen-bank-manifest-v1', 'utc': C.now(), 'status': None,
           'definition': {h: {'income_source': a_, 'occupation_source': b_} for h, (a_, b_) in C.COMPONENTS.items()},
           'primary': C.PRIMARY, 'controls': C.CONTROLS, 'references_for_utility_only': list(C.REFERENCES), 'roles': list(C.BANK_ROLES),
           'release_contract': ('each recipient: its own float64 probability vector and argmax decision; coalition: aligned concatenation of '
                                'the two recipient wires; no teacher outputs, leaf ids, activations, selection flags, identifiers, weights or labels'),
           'seed_pairing': 'seed s components come only from seed s source files; the two model seeds are never averaged',
           'per_seed': {}, 'bank_files_sha256': {}}
    ok = True
    for s in C.SEEDS:
        bank, roles, checks, wires, unique_wires = build(s)
        arrays = {}
        for svc, rr in bank.items():
            for r, tt in rr.items():
                for t, p in tt.items():
                    arrays[f'{svc}__{r}__{t}'] = p
        for r in C.BANK_ROLES:
            arrays[f'row_id__{r}'] = roles[r]['row_id']; arrays[f'unit__{r}'] = roles[r]['unit']
        p = C.bank_path(s)
        if p.exists():
            raise PermissionError('bank file exists (append-only)')
        with open(p, 'xb') as f:
            np.savez(f, **arrays)
        p.chmod(0o600)
        man['bank_files_sha256'][f'S{s}'] = C.sha(p)
        man['per_seed'][f'S{s}'] = {'checks': checks, 'wire_sha256': wires, 'unique_single_recipient_wires': unique_wires,
                                    'rows': {r: int(len(roles[r]['row_id'])) for r in C.BANK_ROLES},
                                    'households': {r: int(len(np.unique(roles[r]['unit']))) for r in C.BANK_ROLES},
                                    'row_id_sha256': {r: C.arr_sha(roles[r]['row_id']) for r in C.BANK_ROLES},
                                    'source_output_files_sha256': {k: v['sha256'] for k, v in C.restore_receipt()['files'].items()
                                                                   if k.startswith(('runs/STAGE_A_%d/' % s, 'runs/STAGE_B_%d/' % s, 'runs/NEURAL_RAW_%d/' % s)) and k.endswith('.npz') and 'OUTPUTS' in k}}
        ok &= (checks['recipient_bitwise_source_identity']['all'] and checks['H_LL_equals_source_LOCAL_TREE_complete_wire']
               and checks['H_JJ_equals_source_JOINT_TREE_complete_wire'] and all(checks['swap_changes_only_intended_wire'].values())
               and checks['float64_probability_vectors_original_class_order'])
    man['status'] = 'BANK_FROZEN' if ok else 'BANK_CHECK_FAILED'
    (out / 'FROZEN_BANK_MANIFEST.json').write_text(json.dumps(man, indent=2, sort_keys=True))
    print(json.dumps({'status': man['status'], 'bank_sha256': man['bank_files_sha256'],
                      'checks': {k: v['checks'] for k, v in man['per_seed'].items()},
                      'unique_wires': {k: [(u['task'], u['services']) for u in v['unique_single_recipient_wires']] for k, v in man['per_seed'].items()}}, indent=1, default=str))
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
