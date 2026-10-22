"""Beside comparisons on the registered cells (coordinator; descriptive, never a decision input): each family CAND against its
ablations, the dual-design beside references (DDR_TWIN = CTL_TWIN bitwise, DDR_PTC_R1, DDR_ZTILT_R1) and the other family's CAND; and
each ablation against the shared twin. Every entry is `nrm.rules.comparison` (aggregate screen and recipient gate) of the grid's
paired cells, benefit = comparator AUC - candidate AUC.

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/beside.py --bank IPR1_MAIN
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402
from ipr import claims as CL         # noqa: E402
from nrm import rules as NR          # noqa: E402

PAIRS = [('IA_SFG', c) for c in ('IA_BLIND', 'DDR_TWIN', 'DDR_PTC_R1', 'DDR_ZTILT_R1', 'IB_CSF')] + \
        [('IB_CSF', c) for c in ('IB_CSFT', 'IB_IND', 'DDR_TWIN', 'DDR_PTC_R1', 'DDR_ZTILT_R1')] + \
        [(a, 'CTL_TWIN') for a in ('IA_BLIND', 'IB_CSFT', 'IB_IND')] + [('IB_CSFT', 'IB_IND')]
PAIR_SETS = {'IPR1_MAIN': PAIRS,
             # IA revision R1 (card IA_OSG_R1): against the registered IA candidate, its ablation and the dual-design references
             'IPR1_R1': [('IA_OSG', c) for c in ('IA_SFG', 'IA_OSG_BLIND', 'DDR_TWIN', 'DDR_PTC_R1', 'DDR_ZTILT_R1')]
                        + [('IA_OSG_BLIND', 'CTL_TWIN')]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='IPR1_MAIN')
    a = ap.parse_args()
    g = CL.load_grid(a.bank)
    out = {}
    for cand, comp in PAIR_SETS[a.bank]:
        e = (g['paired'].get(cand) or {}).get(comp)
        if not e:
            out[f'{cand}__vs__{comp}'] = {'missing': True}
            continue
        c = NR.comparison(e)
        agg = c['benefit'].get('aggregate') or {}
        pv = agg.get('per_version') or {}
        vm = {str(v): (pv.get(v) or pv.get(str(v)) or {}).get('mean') for v in (0, 1)}
        out[f'{cand}__vs__{comp}'] = {'cand': cand, 'comparator': comp, 'label': c['benefit']['label'], 'mean': agg.get('mean'),
                                      'se_agg': agg.get('se_agg'), 'lower': agg.get('lower'), 'version_means': vm,
                                      'recipient': {w: c['recipient'][w]['label'] for w in NR.SINGLE_VIEWS},
                                      'recipient_gate_pass': c['recipient_gate_pass']}
        print(cand, comp, c['benefit']['label'], agg.get('mean'), agg.get('se_agg'))
    C.write_json(C.RESULTS / 'claims' / f'BESIDE_{a.bank}.json', {'schema': 'IPR1-beside-v1', 'utc': C.now(), 'bank': a.bank,
                                                                  'note': 'descriptive; registered cells; never a decision input',
                                                                  'comparisons': out})


if __name__ == '__main__':
    main()
