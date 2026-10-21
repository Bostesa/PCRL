"""Robustness-refit bank for a revision reading whose CAND is its family's strongest feasible nonzero release (RULES.md 8; PROTOCOL
6.3). Same design as DDR1_ROB (roles/coordinator/write_rob_bank.py): the release, the shared TWIN, the release's own bracketing
controls (its reading's D2 frontier comparisons) and NEURAL_RAW / H_LJ on repetitions R1-R4 x reader seeds k0-k3 (the registered
cells R1 / R2 x k0 / k1 are cache hits of the revision grid). Label-free; written before any output of the new bank.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/write_rob_revision.py --reading DB_R1
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ddr import common as C          # noqa: E402
from ddr import devreaders as DR     # noqa: E402
from ddr import banks as BK          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--reading', required=True)
ap.add_argument('--ablations', nargs='*', default=[], help="member specs FAMILY/CONFIG of the release's component ablations")
a = ap.parse_args()
plan = json.loads((C.RESULTS / 'claims' / f'PLAN_{a.reading}.json').read_text())
dec = json.loads((C.RESULTS / 'claims' / a.reading / 'DECISIONS.json').read_text())['decisions']
brk = []
for key in ('SBC_FRONTIER', 'SPEND_FRONTIER'):
    comp = (dec.get('D2') or {}).get('comparisons', {}).get(key) or {}
    q = (comp.get('frontier') or {}).get('bracketing_control') or comp.get('point')
    if q and q != 'TWIN':
        brk.append(q)
names = [plan['twin'], plan['cand']] + brk
mem = {k: BK.family_member(plan['members'][k]) for k in names}
abl = [m.split('/')[1] for m in a.ablations]
for m in a.ablations:
    mem[m.split('/')[1]] = BK.family_member(m)
mem['NEURAL_RAW'] = BK.REFERENCE_MEMBERS['NEURAL_RAW']
mem['H_LJ'] = BK.REFERENCE_MEMBERS['H_LJ']
bank = f'DDR1_ROB_{a.reading}'
spec = {'bank_id': bank,
        'purpose': f'robustness to classifier refitting (D2R, descriptive) of {plan["cand"]} ({a.reading}), the strongest feasible nonzero '
                   'release of its family: new household splits R3 / R4 and reader seeds k2 / k3 for the release, its TWIN, its '
                   'bracketing controls, its component ablations (' + (', '.join(abl) or 'none') + ') and NEURAL_RAW / H_LJ; '
                   'registered cells included (cache hits, except for the ablations) so that entries merge',
        'members': mem, 'candidates': [plan['cand']] + abl, 'comparators': list(mem), 'reps': ['R1', 'R2', 'R3', 'R4'],
        'reader_seeds': [0, 1, 2, 3], 'written_utc': C.now(), 'benchmark_references': ['NEURAL_RAW', 'H_LJ'], 'diagnostics': []}
DR.validate_spec(spec)
out = C.RESULTS / 'grid_specs' / f'{bank}.json'
if out.exists():
    raise SystemExit(f'{out.name} exists (write-once)')
C.write_json(out, spec)
print(json.dumps({'spec': str(out.relative_to(C.ROOT)), 'members': sorted(mem), 'brackets': brk, 'ablations': abl}))
