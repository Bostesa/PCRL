"""Investigator A, R1 design: men-minus-women MODEL_FIT occupation Brier contrasts from the FA_TFR_v1 fit records (no label read).
CAND contrast per rho = men - women of the descriptive out-of-fold spend over SUCC; the lambda = 0 LO point's per-group spend = CAND's
minus the recorded guard difference (all are means over the same MODEL_FIT people).
Run: python3 results/pcrl_strong_anchor_release_v1/roles/investigator_a/r1_contrasts.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
out = {}
for v in (0, 1):
    r = json.loads((ROOT / 'results' / 'pcrl_strong_anchor_release_v1' / 'fa' / f'FIT_V{v}.json').read_text())
    s = r['selection']
    mf = s['mf_spend_over_SUCC_descriptive_biased']
    cand = {k: {'men': d['occ_brier_men']['estimate'], 'women': d['occ_brier_women']['estimate'],
                'contrast': d['occ_brier_men']['estimate'] - d['occ_brier_women']['estimate']} for k, d in mf.items()}
    row = s['per_rho']['0.3']
    g = row['mf_guard_spend_over_reference']
    lo_men = mf['0.3']['occ_brier_men']['estimate'] - g['occ_brier_men']['estimate']
    lo_women = mf['0.3']['occ_brier_women']['estimate'] - g['occ_brier_women']['estimate']
    out[f'V{v}'] = {'cand_by_rho': cand, 'lo_s': row['guard_reference_s'],
                    'lo_point': {'men': lo_men, 'women': lo_women, 'contrast': lo_men - lo_women}}
p = Path(__file__).with_name('r1_contrasts.json')
p.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
for v, d in out.items():
    print(v, 'LO s', d['lo_s'], {k: round(x, 5) for k, x in d['lo_point'].items()})
    for k in sorted(d['cand_by_rho'], key=float):
        print('   rho', k, {kk: round(x, 5) for kk, x in d['cand_by_rho'][k].items()})
