"""Rebuild the ten packages of lanes/DEPLOY_V1.json against the current code (coordinator; label-free; PROTOCOL_DEVIATIONS item 12).

The packages built at 04:05Z pin the hashes of code files that commit 221a91f21 changed afterwards, so ipr.deploy refuses them from a
restored copy. This script moves those package folders (unchanged) from the private deploy/ folder to deploy_stale_v1/, and writes
lanes/DEPLOY_V2.json: a fresh build of each into its original folder (the stored releases under runs/STORED are reused; they are
recomputed bitwise by the check), then one check over all fourteen packages (the ten rebuilt and the four of lanes/DEPLOY_R1.json),
written to deploy/DEPLOYMENT_CHECK_V2.json.

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/write_deploy_v2_lane.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402

v1 = json.loads((C.RESULTS / 'lanes' / 'DEPLOY_V1.json').read_text())
builds = [u for u in v1 if u['argv'][3] == 'build']
stale = C.PRIV / 'deploy_stale_v1'
stale.mkdir(exist_ok=False)
units, moved = [], []
for u in builds:
    out = ROOT / u['argv'][u['argv'].index('--out') + 1]
    shutil.move(str(out), str(stale / out.name))
    moved.append(out.name)
    units.append({**u, 'unit': f'IPR_DEPLOY_BUILD2_{out.name}'})
pkgs = sorted(str((C.PRIV / 'deploy' / n).relative_to(ROOT)) for n in moved) + \
    [str((C.PRIV / 'deploy' / n).relative_to(ROOT)) for n in ('IA_OSG_V0', 'IA_OSG_V1', 'IA_OSG_RO_HI_V0', 'IA_OSG_RO_HI_V1')]
units.append({'unit': 'IPR_DEPLOY_CHECK_V2', 'stage': 'deployment_check', 'rss_gib': 2.0,
              'argv': ['PY', '-m', 'ipr.deploy', 'check', '--packages', *pkgs, '--report',
                       'results/pcrl_input_and_pair_release_v1/deploy/DEPLOYMENT_CHECK_V2.json']})
p = C.RESULTS / 'lanes' / 'DEPLOY_V2.json'
if p.exists():
    raise SystemExit(f'{p.name} exists (lanes are write-once)')
p.write_text(json.dumps(units, indent=1) + '\n')
print(json.dumps({'lane': str(p.relative_to(ROOT)), 'moved_to_deploy_stale_v1': moved, 'units': [u['unit'] for u in units]}))
