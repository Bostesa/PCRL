"""Label-free (investigator B, FS stop memo): ALT / CTRL bracket spends of FS_CPI version 1 (CALIBRATION, committed fitting record) and
a linear interpolation of the ALT path's MODEL_FIT descriptive pair drop at CAND's CALIBRATION spend. Records only; python3."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
r = json.loads((ROOT / 'results/pcrl_strong_anchor_release_v1/fs/FIT_CPI_V1.json').read_text())
c = r['cand']['spend_over_start']['value']
alt, ctrl = r['paths']['ALT'], r['paths']['CTRL']
for b in ('0.25', '0.3', '0.35', '0.4', '0.45'):
    g = alt['grid'].get(b)
    if g:
        print('ALT beta', b, 'spend', round(g['spend_over_start'], 5), 'se_diff_vs_cand', round(g['se_diff_vs_cand'], 5))
h = ctrl['grid'][str(ctrl['hi'])]
print('CTRL hi', ctrl['hi'], 'spend', round(h['spend_over_start'], 5), 'cand spend', round(c, 5))
auc = r['mf_reader_check']['auc']
s_lo, s_hi = alt['grid']['0.3']['spend_over_start'], alt['grid']['0.4']['spend_over_start']
w = (c - s_lo) / (s_hi - s_lo)
out = {}
for rd in ('LOG_LR_pair', 'HGB15_pair'):
    base = auc['SUCC'][rd]
    lo, hi, cand = auc['FS_CPI_BETA_LO'][rd] - base, auc['FS_CPI_BETA_HI'][rd] - base, auc['FS_CPI'][rd] - base
    out[rd] = {'cand_drop': round(cand, 4), 'alt_interpolated_drop': round(lo + w * (hi - lo), 4), 'weight': round(w, 3),
               'dq_hi_drop': round(auc['FS_CPI_DQ_HI'][rd] - base, 4)}
print(json.dumps(out))
