"""Verifier check 7 (DRF1; label-free): the recovery table of RESEARCH_DECISION.md (registered-cell SCORE AUCs, version means over the
four registered cells of each version, pair / income / occupation) recomputed from the grids' per-cell records, and a few quoted
numbers (optimistic diagnostic) against my check-3 outputs. 4-decimal rounding of the document is matched within 5e-5.

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check7_document_numbers.py --out FILE
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

RES = REPO / 'results' / 'pcrl_direct_recovery_frontier_v1'
PRIV = REPO / '.private' / 'pcrl_direct_recovery_frontier_v1'
ROWS = {'DR_AUC_R1': 'DRF1_R1', 'DR_MI_R1': 'DRF1_R1', 'DR_CTRL_HI_R1': 'DRF1_R1', 'DR_AUC_R2': 'DRF1_R2', 'DR_MI_R2': 'DRF1_R2',
        'TWB_C64': 'DRF1_R1', 'SUCC': 'DRF1_R1', 'RETIRED_B3CTL': 'DRF1_R1', 'H_LJ': 'DRF1_R1', 'NEURAL_RAW': 'DRF1_R1'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    items, info = [], {'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}
    doc = (RES / 'RESEARCH_DECISION.md').read_text()
    info['document_sha256'] = V.sha256_file(RES / 'RESEARCH_DECISION.md')
    grids = {}
    for bank in set(ROWS.values()):
        cells = {}
        for v in (0, 1):
            cells.update(json.loads((PRIV / 'runs' / 'GRID' / bank / f'S{v}' / 'GRID.json').read_text())['cells'])
        grids[bank] = cells
    for row in doc.splitlines():
        if not row.startswith('| ') or row.startswith('| Release') or row.startswith('|---'):
            continue
        name = row.split('|')[1].strip().split(' ')[0]
        if name not in ROWS:
            continue
        nums = [float(x) for x in re.findall(r'0\.\d{4}', row)]
        cells = grids[ROWS[name]]
        mine = []
        for view in ('pair', 'income', 'occupation'):
            for v in (0, 1):
                cs = [c for c in V.REGISTERED if c.startswith(f'S{v}')]
                mine.append(sum(cells[c][name][view]['score_auc'] for c in cs) / len(cs))
        for i, (d_, m_) in enumerate(zip(nums, mine)):
            items.append(V.cmp_num(f'table:{name}:{["pair", "income", "occupation"][i // 2]}:V{i % 2}', m_, d_, 5.0001e-5))
        items.append(V.cmp_eq(f'table:{name}:six_numbers', len(nums), 6))
    out = {'schema': 'DRF1-verifier-check7-v1', 'info': info, 'overall': V.summarize(items), 'summary': {'table': V.summarize(items)},
           'items': items}
    if a.out:
        V.dump(a.out, out)
    print(json.dumps(V.jsonable({'overall': out['overall']}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
