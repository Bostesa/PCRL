"""D2R plan of a reading (coordinator; mechanical): the reading's claims plan plus the bracketing controls its D2 frontier comparisons
used (`frontier.bracketing_control` per path in DECISIONS.json), which `ddr.claims d2r` needs as `brackets` / `bracket_ids`.
Written once per reading as claims/PLAN_<reading>_D2R.json; the D2R reading is descriptive and never changes D2.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/write_d2r_plan.py --reading DA_REG
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ddr import common as C          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--reading', required=True)
ap.add_argument('--ablations', nargs='*', default=None, help="ablation config ids (default: the plan's own)")
a = ap.parse_args()
plan = json.loads((C.RESULTS / 'claims' / f'PLAN_{a.reading}.json').read_text())
dec = json.loads((C.RESULTS / 'claims' / a.reading / 'DECISIONS.json').read_text())['decisions']
ids = {}
for key, path in (('SBC_FRONTIER', 'SBC'), ('SPEND_FRONTIER', 'SPEND')):
    comp = (dec.get('D2') or {}).get('comparisons', {}).get(key) or {}
    ids[path] = (comp.get('frontier') or {}).get('bracketing_control') or comp.get('point')
if a.ablations is not None:
    plan['ablations'] = list(a.ablations)
plan.update({'plan_id': f"{plan['plan_id']}_D2R", 'brackets': [b for b in ids.values() if b], 'bracket_ids': ids,
             'd2r_note': 'bracketing controls copied from the reading\'s D2 frontier comparisons (mechanical); descriptive reading',
             'written_utc': C.now()})
out = C.RESULTS / 'claims' / f'PLAN_{a.reading}_D2R.json'
C.write_json_x(out, plan)
print(json.dumps({'plan': str(out.relative_to(C.ROOT)), 'bracket_ids': ids, 'ablations': plan.get('ablations')}))
