"""Coordinator's quick view of a grid output (cell-mean SCORE AUC per member and view, per available version).

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/peek_grid.py BANK
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
bank = sys.argv[1]
for v in (0, 1):
    p = ROOT / '.private' / 'pcrl_dual_design_release_v1' / 'runs' / 'GRID' / bank / f'S{v}' / 'GRID.json'
    if not p.is_file():
        print(f'V{v}: not yet'); continue
    g = json.loads(p.read_text())
    cells = [c for c in g['cells'] if '@' not in c]
    reg = [c for c in cells if c[2:4] in ('R1', 'R2') and c[-2:] in ('k0', 'k1')]
    print(f'== V{v}: {len(reg)} registered cells of {len(cells)}')
    rows = []
    for m in g['cells'][reg[0]]:
        a = {vw: float(np.mean([g['cells'][c][m][vw]['score_auc'] for c in reg])) for vw in ('pair', 'income', 'occupation')}
        nl = [g['cells'][c][m]['pair'].get('extras', {}).get('nolookup', {}).get('score_auc') for c in reg]
        a['nolookup_pair'] = float(np.mean(nl)) if all(x is not None for x in nl) else None
        rows.append((a['pair'], m, a))
    for _, m, a in sorted(rows):
        print(f"   {m:14s} pair {a['pair']:.4f}  inc {a['income']:.4f}  occ {a['occupation']:.4f}  nolookup-pair "
              f"{a['nolookup_pair'] if a['nolookup_pair'] is None else round(a['nolookup_pair'], 4)}")
