"""DB_PTC_v1 section 6, R2 trigger and the family stop rule, computed mechanically from a reading's saved grid (label-free: aggregate
grid outputs only; written before the DB R1 grid output exists).

- Stop (the family closes without a further revision): the pair benefit over the SBC bracketing control is not positive in both
  versions, or its aggregate mean is below 1 se_agg (the reading's D2 record).
- R2 trigger (both versions): the pair benefit over the SBC bracketing control is below half the larger single-view benefit over it
  (registered-cell version means; benefit = comparator AUC - candidate AUC in every view).

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/db_r2_trigger.py --reading DB_R1 --bank DDR1_R1
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ddr import common as C          # noqa: E402
from ddr import claims as CL         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--reading', required=True)
ap.add_argument('--bank', required=True)
a = ap.parse_args()
dec = json.loads((C.RESULTS / 'claims' / a.reading / 'DECISIONS.json').read_text())['decisions']
plan = json.loads((C.RESULTS / 'claims' / f'PLAN_{a.reading}.json').read_text())
sbc = (dec['D2']['comparisons'].get('SBC_FRONTIER') or {})
q = (sbc.get('frontier') or {}).get('bracketing_control') or sbc.get('point')
g = CL.load_grid(a.bank)
e = g['paired'][plan['cand']][q]
# ddr.devreaders.paired: the pair cells are comparator - candidate, the recipient-view cells candidate - comparator; every view is
# put on the benefit scale (comparator - candidate) here.
sign = {'pair': 1.0, 'income': -1.0, 'occupation': -1.0}
vm = {view: {v: sign[view] * sum(x for c, x in d['cells'].items() if c.startswith(f'S{v}')) /
             max(1, sum(1 for c in d['cells'] if c.startswith(f'S{v}'))) for v in C.VERSIONS} for view, d in e.items()}
trig = {v: vm['pair'][v] < 0.5 * max(vm['income'][v], vm['occupation'][v]) for v in C.VERSIONS}
stop_versions = not all(vm['pair'][v] > 0 for v in C.VERSIONS)
mean, se = sbc.get('benefit_mean'), sbc.get('benefit_se')
stop_se = mean is None or se is None or mean < se
out = {'reading': a.reading, 'cand': plan['cand'], 'sbc_bracketing_control': q, 'version_means': vm,
       'aggregate': {'mean': mean, 'se_agg': se},
       'stop': {'not_positive_in_both_versions': stop_versions, 'aggregate_below_1se': stop_se, 'family_closes': stop_versions or stop_se},
       'r2_trigger': {'per_version': trig, 'holds': all(trig.values())}, 'utc': C.now(),
       'note': 'DB_PTC_v1 section 6; MODEL_SELECT outcome (a revision triggered here is ADAPTIVE)'}
C.write_json(C.RESULTS / 'claims' / f'DB_R2_TRIGGER_{a.reading}.json', out)
print(json.dumps(out, indent=1))
