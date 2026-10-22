"""D2R plan of a reading (coordinator; mechanical): the reading's claims plan plus the bracketing controls its D2 frontier comparisons
used (`frontier.bracketing_control` per path in DECISIONS.json), which `ipr.claims d2r` needs as `brackets` / `bracket_ids` (all three
paths recorded; an explicit None where the family has no such path, as for a SEX-blind operation's SBC and RO). Written once per
reading as claims/PLAN_<reading>_D2R.json; the D2R reading is descriptive and never changes D2.

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/write_d2r_plan.py --reading IA_REG
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--reading', required=True)
a = ap.parse_args()
plan = json.loads((C.RESULTS / 'claims' / f'PLAN_{a.reading}.json').read_text())
dec = json.loads((C.RESULTS / 'claims' / a.reading / 'DECISIONS.json').read_text())['decisions']
ids = {}
for key, path in (('SBC_FRONTIER', 'SBC'), ('SPEND_FRONTIER', 'SPEND'), ('RO_FRONTIER', 'RO')):
    if plan['paths'].get(path) is None:
        ids[path] = None
        continue
    comp = (dec.get('D2') or {}).get('comparisons', {}).get(key) or {}
    ids[path] = (comp.get('frontier') or {}).get('bracketing_control') or comp.get('point')
plan.update({'plan_id': f"{plan['plan_id']}_D2R", 'brackets': [b for b in ids.values() if b], 'bracket_ids': ids,
             'd2r_note': "bracketing controls copied from the reading's D2 frontier comparisons (mechanical); descriptive reading",
             'written_utc': C.now()})
out = C.RESULTS / 'claims' / f'PLAN_{a.reading}_D2R.json'
C.write_json_x(out, plan)
print(json.dumps({'plan': str(out.relative_to(C.ROOT)), 'bracket_ids': ids, 'ablations': plan.get('ablations')}))
