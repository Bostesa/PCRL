"""Label-free extras of a revision reading (coordinator), written before any grid output of the revision's bank is read:
- the beside bank `grid_specs/<bank>_BESIDE.json`: the revision's release against the family's beside controls (DB: ZTILT, SHUF) and
  the registered release of the same family, on the registered cells (all members are cache hits of the revision and main grids);
- the data-rich reader spec `rich/RICH_SPEC_<bank>.json`: every member of the revision plan scored by the readers trained on its own
  MODEL_FIT out-of-fold release (revision members by the revision fit's readers 'FAMILY:R<n>', reused members by the registered fit's
  readers), with the revision's release compared against each of them.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/write_revision_extras.py --plan claims/PLAN_DB_R1.json --bank DDR1_R1
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ddr import common as C          # noqa: E402
from ddr import banks as BK          # noqa: E402
from ddr import plan as PL           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--plan', required=True)
ap.add_argument('--bank', required=True)
a = ap.parse_args()
plan = json.loads((C.RESULTS / a.plan).read_text())
fam, rev, cand = plan['family'], f"R{plan['revision']}", plan['cand']
if not cand:
    raise SystemExit('no candidate saved in both versions: nothing to compare')
reg = PL.FAMILY_PLAN[fam]
sv = PL.saved()
out = {}
beside = [c for c in reg['beside'] + (reg['cand'],) if PL._both(sv, fam, c)]
if beside:
    members = {'DA_TWIN': 'DA/DA_TWIN', cand: f'{fam}/{cand}', **{c: f'{fam}/{c}' for c in beside}}
    spec = BK.write(f'{a.bank}_BESIDE', f'revision reading {fam} {rev}: {cand} against the family\'s beside controls and its registered '
                                        f'release on the registered cells (descriptive; written before any output of {a.bank})',
                    members, [cand])
    out['beside_bank'] = str(spec.relative_to(C.ROOT))
rev_members = {cand, *plan['paths']['SBC']}
members = {}
for name, m in plan['members'].items():
    f2, cfg = m.split('/')
    members[name] = {'family': f2, 'config': cfg, 'reader_key': cfg,
                     'readers_of': f'{fam}:{rev}' if cfg in rev_members else (f2 if cfg != 'DA_TWIN' else 'DA')}
for c in beside:
    members[c] = {'family': fam, 'config': c, 'reader_key': c, 'readers_of': fam}
rspec = {'id': f'{a.bank}_RICH', 'members': members, 'comparisons': [[cand, c] for c in members if c != cand],
         'note': f'revision {fam} {rev}: revision members scored with the revision fit\'s readers ({fam}:{rev}); reused members with the '
                 'registered fit\'s readers; DA_TWIN with DA-trained readers. Written before any output of the revision grid.',
         'written_utc': C.now()}
rp = C.RESULTS / 'rich' / f'RICH_SPEC_{a.bank}.json'
C.write_json_x(rp, rspec)
out['rich_spec'] = str(rp.relative_to(C.ROOT))
print(json.dumps(out, indent=1))
