"""V2 Within-seed row pairing and wire inheritance (label-free).

Source outputs are read here directly from the restored files (each sha256 checked against the published
TRAINING_COMPLETE_LOCK); the bank is read through acs_hybrids.common.load_bank (hash-checked loader) and its file hash is
also compared with ANALYSIS_LOCK / REPORTING_LOCK. Checks per model seed s:
  1. bank row_id / unit equal the export order for every role (and the stored source row ids where present);
  2. every bank recipient array is bitwise its chosen seed-s source output (hybrids, controls, references);
  3. no bank array equals the other seed's source output (seed s components come only from seed s files);
  4. H_LL / H_JJ equal the complete LOCAL_TREE / JOINT_TREE wires; each hybrid differs from the other component;
  5. FROZEN_BANK_MANIFEST wire and row hashes recomputed here; probability conventions; no extra bank fields.
"""
from __future__ import annotations

import argparse
import hashlib
from zipfile import ZipFile

import numpy as np

from acs_hybrids import common as C   # hash-checked data loading only

from . import vcommon as V

FILES = {'stage_a': ('STAGE_A_{s}/STAGE_A_OUTPUTS.npz', ('T_TREE', 'U0', 'Ucal')),
         'stage_b': ('STAGE_B_{s}/STAGE_B_OUTPUTS_E20.npz', ('LOCAL_TREE', 'JOINT_TREE', 'TASK_REFIT')),
         'neural_raw': ('NEURAL_RAW_{s}/NEURAL_RAW_OUTPUTS.npz', ('NEURAL_RAW_rho1.0_E2',))}


def load_sources(seed, tl):
    out, rows, hashes = {}, {}, {}
    for fam, (pat, labels) in FILES.items():
        rel = pat.format(s=seed)
        p = V.SRCP / 'runs' / rel
        h = V.sha256_file(p)
        hashes[rel] = {'sha256': h, 'equals_TRAINING_COMPLETE_LOCK': h == tl.get(rel)}
        with np.load(p, allow_pickle=False) as z:
            for key in z.files:
                parts = key.split('__')
                if parts[0] in ('row_id', 'unit'):
                    rows.setdefault(fam, {})[key] = z[key].copy()
                    continue
                lab, role, task = parts
                if lab in labels and role in V.ROLES:
                    out.setdefault(lab, {}).setdefault(role, {})[task] = z[key].copy()
    return out, rows, hashes


def wire_hash(p):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    h = hashlib.sha256(p.tobytes())
    h.update(np.ascontiguousarray(np.argmax(p, axis=1).astype(np.int64)).tobytes())
    return h.hexdigest()


def arr_hash(a):
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def same(a, b):
    return bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(V.OUT / 'V2_row_pairing_and_inheritance.json'))
    a = ap.parse_args()
    L = V.analysis_lock()
    RL = V.reporting_lock()
    man = V.jload(V.RES / 'FROZEN_BANK_MANIFEST.json')
    tl = V.jload(V.SRC_RES / 'TRAINING_COMPLETE_LOCK.json')['frozen_files_sha256']
    comps = V.components()
    expected_services = sorted(comps)
    exports = {r: C.load_role(r, ('features',)) for r in V.ROLES}
    src = {}
    src_rows = {}
    src_hash = {}
    for s in V.SEEDS:
        src[s], src_rows[s], src_hash[s] = load_sources(s, tl)
    res = {'check': 'V2 Within-seed row pairing and wire inheritance', 'scope': 'array identity replay (label-free)',
           'combinations_from_lock': {k: list(v) for k, v in comps.items()}, 'per_seed': {}}
    ok = True
    for s in V.SEEDS:
        o = 1 - s
        bfile = C.bank_path(s)
        bh = V.sha256_file(bfile)
        bank, ids = C.load_bank(s)
        rec = {'bank_file_sha256': bh,
               'bank_hash_equals': {'ANALYSIS_LOCK': bh == L['bank']['bank_files_sha256'][f'S{s}'],
                                    'REPORTING_LOCK': bh == RL['bank_files_sha256'][f'S{s}'],
                                    'FROZEN_BANK_MANIFEST': bh == man['bank_files_sha256'][f'S{s}']},
               'source_files': src_hash[s]}
        # 1. rows
        rows = {}
        for r in V.ROLES:
            e = exports[r]
            rows[r] = {'n': int(len(e['row_id'])),
                       'bank_row_id_equals_export': same(ids[f'row_id__{r}'], e['row_id']),
                       'bank_unit_equals_export': same(ids[f'unit__{r}'], e['unit']),
                       'stage_a_stored_row_id_equals_export': same(src_rows[s]['stage_a'][f'row_id__{r}'], e['row_id']),
                       'stage_a_stored_unit_equals_export': same(src_rows[s]['stage_a'][f'unit__{r}'], e['unit']),
                       'row_id_sha256_equals_manifest': arr_hash(e['row_id']) == man['per_seed'][f'S{s}']['row_id_sha256'][r],
                       'row_ids_unique': bool(len(np.unique(e['row_id'])) == len(e['row_id']))}
            for fam in ('stage_b', 'neural_raw'):
                rows[r][f'{fam}_file_has_row_ids'] = bool(src_rows[s].get(fam))
        rec['rows'] = rows
        rows_ok = all(v['bank_row_id_equals_export'] and v['bank_unit_equals_export'] and v['stage_a_stored_row_id_equals_export']
                      and v['stage_a_stored_unit_equals_export'] and v['row_id_sha256_equals_manifest'] and v['row_ids_unique'] for v in rows.values())
        # 2./3. bitwise identity to the chosen seed-s source, never to the other seed's
        ident, other_seed_equal, conv_bad, failures = 0, [], [], []
        n = 0
        for svc in expected_services:
            for r in V.ROLES:
                for ti, t in enumerate(V.TASKS):
                    lab = comps[svc][ti]
                    b = bank[svc][r][t]
                    st = src[s][lab][r][t]
                    n += 1
                    if b.dtype == np.float64 and same(b, st):
                        ident += 1
                    else:
                        failures.append(f'{svc}|{r}|{t}<-{lab}')
                    if same(b, src[o][lab][r][t]):
                        other_seed_equal.append(f'{svc}|{r}|{t}')
                    k = V.NCLASS[t]
                    if not (b.shape == (len(exports[r]['row_id']), k) and np.isfinite(b).all() and (b >= 0).all()
                            and np.max(np.abs(b.sum(1) - 1.0)) <= 1e-9):
                        conv_bad.append(f'{svc}|{r}|{t}')
        rec['recipient_bitwise_identity'] = {'arrays': n, 'identical': ident, 'failures': failures,
                                             'services': expected_services}
        rec['seed_provenance'] = {'arrays_equal_to_other_seed_source': other_seed_equal,
                                  'note': 'an array equal to the other seed\'s source would make the seed of origin ambiguous'}
        rec['probability_convention_failures'] = conv_bad
        # 4. complete wires and hybrid swaps
        full = {'H_LL_equals_LOCAL_TREE_complete_wire': all(same(bank['H_LL'][r][t], src[s]['LOCAL_TREE'][r][t]) for r in V.ROLES for t in V.TASKS),
                'H_JJ_equals_JOINT_TREE_complete_wire': all(same(bank['H_JJ'][r][t], src[s]['JOINT_TREE'][r][t]) for r in V.ROLES for t in V.TASKS)}
        swaps = {}
        for h in V.HYBRIDS:
            for ti, t in enumerate(V.TASKS):
                chosen = comps[h][ti]
                other = 'JOINT_TREE' if chosen == 'LOCAL_TREE' else 'LOCAL_TREE'
                swaps[f'{h}|{t}'] = {'chosen': chosen,
                                     'equals_chosen_all_roles': all(same(bank[h][r][t], src[s][chosen][r][t]) for r in V.ROLES),
                                     'differs_from_other_component_all_roles': all(not same(bank[h][r][t], src[s][other][r][t]) for r in V.ROLES)}
        rec['complete_wires'] = full
        rec['hybrid_component_choice'] = swaps
        # 5. manifest wire hashes, extra fields
        wh_bad = []
        for svc in V.ROSTER:
            for t in V.TASKS:
                for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'):
                    if wire_hash(bank[svc][r][t]) != man['per_seed'][f'S{s}']['wire_sha256'][f'{svc}|{t}'][r]:
                        wh_bad.append(f'{svc}|{t}|{r}')
        rec['manifest_wire_hashes'] = {'checked': len(V.ROSTER) * 2 * 3, 'unequal': wh_bad}
        with ZipFile(bfile) as z:
            keys = {x.filename[:-4] for x in z.infolist()}
        expect = {f'{svc}__{r}__{t}' for svc in expected_services for r in V.ROLES for t in V.TASKS}
        expect |= {f'{k}__{r}' for k in ('row_id', 'unit') for r in V.ROLES}
        rec['bank_fields'] = {'n': len(keys), 'unexpected': sorted(keys - expect), 'missing': sorted(expect - keys)}
        seed_ok = (rows_ok and ident == n and not failures and not other_seed_equal and not conv_bad and all(full.values())
                   and all(v['equals_chosen_all_roles'] and v['differs_from_other_component_all_roles'] for v in swaps.values())
                   and not wh_bad and not rec['bank_fields']['unexpected'] and not rec['bank_fields']['missing']
                   and all(rec['bank_hash_equals'].values()) and all(v['equals_TRAINING_COMPLETE_LOCK'] for v in src_hash[s].values()))
        rec['pass'] = bool(seed_ok)
        ok &= seed_ok
        res['per_seed'][f'S{s}'] = rec
    # rows identical across seeds (one row order per role)
    b0, i0 = C.load_bank(0)
    b1, i1 = C.load_bank(1)
    res['row_order_identical_across_seeds'] = all(same(i0[k], i1[k]) for k in i0)
    ok &= res['row_order_identical_across_seeds']
    res['status'] = 'PASS' if ok else 'FAIL'
    V.write_json(a.out, res)
    print({'status': res['status'], **{k: v['pass'] for k, v in res['per_seed'].items()}})


if __name__ == '__main__':
    main()
