"""Independent verifier, check 10 helper: MODEL_SELECT development numbers quoted in RESEARCH_DECISION.md, recomputed from the
stored EVAL_MS.json records (aggregate values only; no label is read here). Pair proxy per seed = mean over the 4 grid cells of
the pair 'max_incl_single'; income Brier vs Ucal and the men's (sex0) guard from the utility block.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
RUNS = ROOT / '.private' / 'pcrl_income_headroom_successor_v1' / 'runs'

CONFIGS = [('H', 'H3_STRONG_LJDIR_PB'), ('H', 'HC_TTREE_LJDIR_PB'), ('H', 'HC_STRONG'), ('H', 'H2_STRONG_LJDIR_DP'), ('H', 'H1_STRONG_LJDIR'),
           ('P', 'P3_JR_B60'), ('P', 'PC_TEMP_P2'), ('P', 'PC_TEMP_P3'), ('P', 'PC_TWIN_B40'), ('O', 'O1_SEXBAL'), ('O', 'OC_TCAL'),
           ('CMB', 'CMB_P3JR_O1SEXBAL')]


def clause(m, task, ref, metric):
    for c in m['utility']['clauses']:
        if c.get('task') == task and c.get('reference') == ref and c.get('metric') == metric:
            return c
    return None


def main():
    out = {}
    for fam, cfg in CONFIGS:
        for s in (0, 1):
            p = RUNS / fam / cfg / f'S{s}' / 'EVAL_MS.json'
            if not p.is_file():
                continue
            m = json.loads(p.read_text())
            cells = m['proxy']['cells']
            pair = sum(v['pair']['max_incl_single'] for v in cells.values()) / len(cells)
            inc = sum(v['income']['view_max'] for v in cells.values()) / len(cells)
            b = clause(m, 'income', 'Ucal', 'brier_excess')
            g = [x for x in m['utility'].get('guards', [])] if isinstance(m['utility'].get('guards'), list) else m['utility'].get('guards')
            out[f'{fam}/{cfg}/S{s}'] = {'pair_proxy_mean4': pair, 'income_view_mean4': inc, 'n_cells': len(cells),
                                        'income_brier_vs_Ucal': None if b is None else b.get('value', b.get('point')),
                                        'guards': g}
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
