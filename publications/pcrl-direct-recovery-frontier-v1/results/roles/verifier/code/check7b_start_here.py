"""Verifier check 7b (DRF1; label-free): numbers quoted in START_HERE.md, formatted from my own outputs (checks 2, 3, 3b, 3d, 3e) at
the document's precision, must appear in the document.

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check7b_start_here.py --out FILE
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

OUT = HERE.parent / 'out'
RES = HERE.parents[2]


def load(name):
    return json.loads((OUT / name).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    doc = (RES / 'START_HERE.md').read_text()
    r1 = load('OUTOFQUEUE_20261025T0402Z_check3_DRF_R1_vs_decisions.json')
    r2 = load('OUTOFQUEUE_20261025T0434Z_check3_DRF_R2_vs_decisions.json')
    cd = load('OUTOFQUEUE_20261025T0407Z_check3d_confidence_reserve_DRF_R1.json')
    e1 = load('OUTOFQUEUE_20261025T0431Z_check3e_d2r_DRF_R1_vs_record_detail.json')
    c = r1['comparisons']
    succ = r1['frontier']   # not used for SUCC costs; SUCC costs from check 2 below
    c2 = load('DRF_VER_C2_V1_check2.json')
    sp = c2['per_subject']['DR_AUC_R1']
    d_succ = [sp[str(v)]['paired_total_nll']['REF:SUCC'] for v in (0, 1)]
    pooled = (sum(x[0] for x in d_succ) / 2, sum(x[1] for x in d_succ) / 2)
    opt = r1['optimistic']['per_construction']['OPT_L']
    want = {
        'OPT_L in-sample gain range': f"{min(opt['in_sample']['by_version'][v]['gain_pair'] for v in ('0', '1')):.3f}-"
                                      f"{max(opt['in_sample']['by_version'][v]['gain_pair'] for v in ('0', '1')):.3f}",
        'OPT_L nested gap': f"{-opt['nested']['by_version']['0']['gain_pair']:.3f} / {-opt['nested']['by_version']['1']['gain_pair']:.3f}",
        'confidence passes': f"{cd['mine']['cand']['confidence']['n_pass']} of 28",
        'CTRL bracket benefit': f"+{c['DR_CTRL_HI_R1']['pair']['x']:.3f}\n    (se {c['DR_CTRL_HI_R1']['pair']['se_agg']:.3f})",
        'OBJ benefit R1': f"+{c['DR_MI_R1']['pair']['x']:.4f} ({c['DR_MI_R1']['pair']['se_agg']:.4f})",
        'OBJ benefit R2': f"+{r2['comparisons']['DR_MI_R2']['pair']['x']:.4f} ({r2['comparisons']['DR_MI_R2']['pair']['se_agg']:.4f})",
        'SUCC benefit': f"+{c['SUCC']['pair']['x']:.4f} (se {c['SUCC']['pair']['se_agg']:.4f})",
        'SUCC total NLL spend': f"+{pooled[0]:.4f}, se {pooled[1]:.4f}",
        'NEURAL_RAW benefit': f"+{c['NEURAL_RAW']['pair']['x']:.3f}, se {c['NEURAL_RAW']['pair']['se_agg']:.3f}",
        'SUCC sign change': f"+{e1['per_comparator']['SUCC']['registered']['x']:.4f} to {e1['per_comparator']['SUCC']['robustness']['x']:.4f}",
    }
    items = []
    flat = ' '.join(doc.split())
    for k, s in want.items():
        items.append(V.cmp_eq(f'start_here:{k}:{s}', ' '.join(s.split()) in flat, True))
    out = {'schema': 'DRF1-verifier-check7b-v1', 'info': {'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                                                          'document_sha256': V.sha256_file(RES / 'START_HERE.md')},
           'expected_strings': want, 'overall': V.summarize(items), 'summary': {'start_here': V.summarize(items)}, 'items': items}
    if a.out:
        V.dump(a.out, out)
    print(json.dumps(V.jsonable({'overall': out['overall'], 'expected': want}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
