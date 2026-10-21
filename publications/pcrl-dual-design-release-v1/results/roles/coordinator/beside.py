"""Descriptive pair comparisons of a candidate in a saved bank (coordinator; aggregate grid outputs only): the inherited aggregate
screen `nrm.rules.benefit_screen` (benefit = comparator AUC - candidate AUC on the registered cells; se_agg; version means) and the
per-recipient differences, for every comparator of the candidate. Writes claims/BESIDE_<bank>.json.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/beside.py --bank DDR1_R1_BESIDE --cand DB_PTC_R1
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ddr import common as C          # noqa: E402
from ddr import claims as CL         # noqa: E402
from nrm import rules as NR          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--bank', required=True)
ap.add_argument('--cand', required=True)
a = ap.parse_args()
g = CL.load_grid(a.bank)
out = {}
for comp, e in sorted((g['paired'].get(a.cand) or {}).items()):
    b = NR.benefit_screen(e.get('pair'))
    ag = b['aggregate']
    rec = {'label': b['label'], 'mean': ag.get('mean'), 'se_agg': ag.get('se_agg'),
           'version_means': {str(v): ag['per_version'][v]['mean'] for v in C.VERSIONS} if ag.get('per_version') else None}
    for view in ('income', 'occupation'):            # candidate - comparator (ddr.devreaders.paired); negative favours the candidate
        cells = {c: x for c, x in ((e.get(view) or {}).get('cells') or {}).items() if c in NR.REGISTERED}   # the pair fields' 8 cells
        rec[f'{view}_cand_minus_comp'] = sum(cells.values()) / len(cells) if cells else None
    rec['cells'] = 'registered (8: R1 / R2 x k0 / k1 x both versions) for every field'
    out[comp] = rec
    print(f"{a.cand} vs {comp:14s} {rec['label']:24s} {rec['mean']:+.4f} se {rec['se_agg']:.4f} V {rec['version_means']} "
          f"inc {rec['income_cand_minus_comp']:+.4f} occ {rec['occupation_cand_minus_comp']:+.4f}")
C.write_json(C.RESULTS / 'claims' / f'BESIDE_{a.bank}.json', {'schema': 'DDR1-beside-v1', 'utc': C.now(), 'bank': a.bank,
                                                              'cand': a.cand, 'comparisons': out,
                                                              'note': 'descriptive; never a decision input'})
