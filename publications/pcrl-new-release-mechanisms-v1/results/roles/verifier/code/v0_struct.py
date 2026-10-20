"""V0 label-free structure replay: the frozen reader splits (NRM1_devgrid_R1..R4 FIT / VALIDATION / SCORE), the NRM1_screen halves,
their household-set hashes, people / household counts, disjointness, overlaps between repetitions and with the earlier OHN1 / TSZ1
grid splits, reader seeds, bootstrap seed rule and the 16 recipes, all rebuilt with the verifier's own code from MODEL_SELECT
household ids (label-free features only; no label is read).

    python3 results/pcrl_new_release_mechanisms_v1/roles/verifier/code/v0_struct.py --tag V1
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

sys.path.insert(0, str(V.ROOT))
import nrm.common as C  # noqa: E402  (label-free features only)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    man_path = V.RESULTS / 'READER_SPLITS.json'
    man = json.loads(man_path.read_text())
    unit = np.asarray(C.features('MODEL_SELECT')['unit'])
    diffs = []
    roles = {}
    for rep in V.ALL_REPS:
        rr = V.reader_roles(unit, rep)
        roles[rep] = rr
        for role, m in rr.items():
            got = {'people': int(m.sum()), 'households': int(len(np.unique(unit[m]))), 'household_set_sha256': V.hh_set_sha(unit[m])}
            exp = man['repetitions'][rep][role]
            if got != exp:
                diffs.append({'what': f'{rep} {role}', 'got': got, 'expected': exp})
        hh = {r: set(unit[m].tolist()) for r, m in rr.items()}
        disjoint = not (hh['FIT'] & hh['VALIDATION'] or hh['FIT'] & hh['SCORE'] or hh['VALIDATION'] & hh['SCORE'])
        if disjoint != man['repetitions'][rep]['disjoint'] or not disjoint:
            diffs.append({'what': f'{rep} disjoint', 'got': disjoint})
        if man['repetitions'][rep]['tag'] != f'NRM1_devgrid_{rep}':
            diffs.append({'what': f'{rep} tag', 'got': man['repetitions'][rep]['tag']})
    for i, r1 in enumerate(V.ALL_REPS):
        for r2 in V.ALL_REPS[i + 1:]:
            for role in ('FIT', 'VALIDATION', 'SCORE'):
                got = len(set(unit[roles[r1][role]].tolist()) & set(unit[roles[r2][role]].tolist()))
                exp = man['household_overlap_between_repetitions'][f'{r1}_vs_{r2}_{role}']
                if got != exp:
                    diffs.append({'what': f'overlap {r1} {r2} {role}', 'got': got, 'expected': exp})
    for rep in V.ALL_REPS:
        for camp in ('OHN1', 'TSZ1'):
            for old in ('R1', 'R2'):
                part = V.household_part(unit, 10, f'{camp}_devgrid_{old}')
                theirs = {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}
                for role in ('FIT', 'VALIDATION', 'SCORE'):
                    got = len(set(unit[roles[rep][role]].tolist()) & set(unit[theirs[role]].tolist()))
                    exp = man['household_overlap_with_earlier_grid_splits'][f'{rep}_{role}_vs_{camp}_{old}_{role}']
                    if got != exp:
                        diffs.append({'what': f'earlier overlap {rep} {role} {camp} {old}', 'got': got, 'expected': exp})
    A, B = V.screen_halves(unit)
    for nm, m in (('A', A), ('B', B)):
        got = {'people': int(m.sum()), 'households': int(len(np.unique(unit[m]))), 'household_set_sha256': V.hh_set_sha(unit[m])}
        if got != man['screening_split'][nm]:
            diffs.append({'what': f'screen {nm}', 'got': got, 'expected': man['screening_split'][nm]})
    if set(unit[A].tolist()) & set(unit[B].tolist()) or not man['screening_split']['household_disjoint']:
        diffs.append({'what': 'screen halves not household-disjoint'})
    seeds = {f'V{v}k{k}': V.reader_seed(v, k) for v in (0, 1) for k in range(4)}
    if seeds != man['reader_seeds']:
        diffs.append({'what': 'reader seeds', 'got': seeds, 'expected': man['reader_seeds']})
    earlier = {V.MODEL_SEED_BASE + off + 10 * v + k for off in (300, 400, 500) for v in (0, 1) for k in range(10)}
    if set(seeds.values()) & earlier or len(set(seeds.values())) != 8:
        diffs.append({'what': 'reader seeds collide with earlier campaigns or each other'})
    if man['recipes'] != {'source8': list(V.SRC8), 'new8': list(V.NEW8)}:
        diffs.append({'what': 'recipes', 'got': man['recipes']})
    cells = [f'S{v}{r}k{k}' for v in (0, 1) for r in V.REG_REPS for k in (0, 1)]
    if man['cells'] != cells or man['registered_repetitions'] != ['R1', 'R2'] or man['registered_reader_seed_indices'] != [0, 1]:
        diffs.append({'what': 'registered cells / repetitions / seed indices', 'got': [man['cells'], man['registered_repetitions']]})
    if man['bootstrap']['B'] != V.BOOT_B or not man['bootstrap']['seed'].startswith(f'{V.BOOT_BASE} + 100 x version'):
        diffs.append({'what': 'bootstrap rule', 'got': man['bootstrap']})
    res = {'n_differences': len(diffs), 'differences': diffs, 'people': int(len(unit)), 'households': int(len(np.unique(unit))),
           'reader_seeds': seeds, 'cells': cells}
    chk = {'id': f'V0.STRUCT.READER_SPLITS__{a.tag}',
           'what': 'label-free rebuild of the frozen reader splits R1..R4 (FIT / VALIDATION / SCORE), screening halves, household-set '
                   'hashes, counts, disjointness, overlaps (between repetitions and with OHN1 / TSZ1 splits), reader seeds, bootstrap '
                   'rule and recipes, against READER_SPLITS.json',
           'inputs': {**V.inputs_sha([man_path, V.CODE_DIR / 'vlib.py', Path(__file__)]),
                      'MODEL_SELECT unit array_sha256': V.arr_sha(unit)},
           'result': res, 'numerical_status': 'PASS' if not diffs else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps(res, indent=1, default=V._jd)[:3000])
    return 0 if not diffs else 1


if __name__ == '__main__':
    sys.exit(main())
