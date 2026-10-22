"""Verifier check 3b (label-free): reader-seed effectiveness on the minimum sets of IA_REG and IB_REG, from the recorded per-recipe
SCORE fingerprints (GRID.json cells[...]['extras']['score_sha256']) of the registered grid, with own code; and the recorded reader /
bootstrap seeds against the registered formula (MODEL_SEED_BASE + 900 + 10 v + k; 2026102300 + 100 v + repetition index; none at the
forbidden offsets 300-800).

    python3 <this file>        (label-free record arithmetic)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vlib as V  # noqa: E402

SEED_DEPENDENT = ('ET5', 'ET25', 'MLP64', 'MLP6432', 'OUT_MLP', 'LOG_MLP')
MINIMUM = {'IA_REG': ('IA_SFG', 'CTL_TWIN', 'IA_SBC_HI', 'IA_SPD_HI', 'IA_RO_HI'),
           'IB_REG': ('IB_CSF', 'CTL_TWIN', 'IB_SPD_LO', 'IB_SPD_HI')}


def main():
    t0 = time.process_time()
    out = {'utc': V.utc(), 'readings': {}}
    cells = {}
    for v in (0, 1):
        cells.update(json.loads((V.GRID_DIR / f'S{v}' / 'GRID.json').read_text())['cells'])
    for rid, mem in MINIMUM.items():
        equal, checked, all_recipes_equal = [], 0, []
        for v in (0, 1):
            for rep in ('R1', 'R2'):
                c0, c1 = f'S{v}{rep}k0', f'S{v}{rep}k1'
                for m in mem:
                    for view in V.VIEWS:
                        f0 = cells[c0][m][view]['extras']['score_sha256']
                        f1 = cells[c1][m][view]['extras']['score_sha256']
                        for r in SEED_DEPENDENT:
                            checked += 1
                            if f0[r] == f1[r]:
                                equal.append(f'S{v}{rep}|{m}|{view}|{r}')
                        if all(f0[r] == f1[r] for r in f0):
                            all_recipes_equal.append(f'S{v}{rep}|{m}|{view}')
        out['readings'][rid] = {'minimum_set': list(mem), 'seed_dependent_checks': checked,
                                'seed_dependent_equal_k0_k1': equal, 'views_with_every_recipe_equal': all_recipes_equal}
    dec = json.loads((V.RESULTS / 'claims' / 'IA_REG' / 'DECISIONS.json').read_text())
    seeds, boot = dec['seeds'], dec['boot_seeds']
    want = {f'S{v}k{k}': V.reader_seed(v, k) for v in (0, 1) for k in (0, 1)}
    wantb = {f'S{v}{r}': V.BOOT_BASE + 100 * v + V.REP_INDEX[r] for v in (0, 1) for r in ('R1', 'R2')}
    forbidden = {V.MODEL_SEED_BASE + o + 10 * v + k for o in (300, 400, 500, 600, 700, 800) for v in (0, 1) for k in range(10)}
    man = json.loads((V.RESULTS / 'READER_SPLITS.json').read_text())
    out['seeds'] = {'recorded': seeds, 'expected': want, 'match': seeds == want, 'boot_recorded': boot, 'boot_expected': wantb,
                    'boot_match': boot == wantb, 'any_forbidden': any(s in forbidden for s in seeds.values()),
                    'manifest_reader_seeds': {k: man['reader_seeds'][k] for k in sorted(man['reader_seeds']) if k.endswith(('k0', 'k1'))}}
    out['cpu_s'] = round(time.process_time() - t0, 3)
    p = V.write_record(f'V6_SEEDS__{V.unit_id() or "NOQUEUE"}.json', out)
    print(json.dumps({rid: {'checks': x['seed_dependent_checks'], 'equal': len(x['seed_dependent_equal_k0_k1']),
                            'equal_items': x['seed_dependent_equal_k0_k1'][:10], 'all_equal_views': len(x['views_with_every_recipe_equal'])}
                      for rid, x in out['readings'].items()} | {'seeds_match': out['seeds']['match'], 'boot_match': out['seeds']['boot_match'],
                                                                'any_forbidden': out['seeds']['any_forbidden'], 'cpu_s': out['cpu_s'],
                                                                'record': str(p)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
