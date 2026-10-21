"""Deployment lane (coordinator; label-free): store, build and check packages for each family's strongest feasible nonzero release
(claims/STRONGEST.json: DA_STD, DB_PTC_R1), the shared twin DA_TWIN and each release's decisive control (the bracketing control its
D2 frontier did not separate from: DA_SPD_HI for DA_REG, DB_SBC_R1_HI for DB_R1), both versions.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/write_deploy_lane.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ddr import common as C          # noqa: E402

NAMES = ['DA/DA_TWIN', 'DA/DA_STD', 'DA/DA_SPD_HI', 'DB/DB_PTC_R1', 'DB/DB_SBC_R1_HI']
units, pkgs = [], []
for n in NAMES:
    cfg = n.split('/')[1]
    for v in C.VERSIONS:
        units.append({'unit': f'DDR_DEPLOY_STORE_{cfg}_V{v}_V1', 'stage': 'deployment_check', 'rss_gib': 1.5,
                      'argv': ['PY', '-m', 'ddr.deploy', 'store', '--name', n, '--version', str(v)]})
for n in NAMES:
    cfg = n.split('/')[1]
    for v in C.VERSIONS:
        out = f'.private/pcrl_dual_design_release_v1/deploy/{cfg}_V{v}'
        pkgs.append(out)
        units.append({'unit': f'DDR_DEPLOY_BUILD_{cfg}_V{v}_V1', 'stage': 'deployment_check', 'rss_gib': 1.5,
                      'argv': ['PY', '-m', 'ddr.deploy', 'build', '--name', n, '--version', str(v), '--out', out]})
units.append({'unit': 'DDR_DEPLOY_CHECK_V1', 'stage': 'deployment_check', 'rss_gib': 2.0,
              'argv': ['PY', '-m', 'ddr.deploy', 'check', '--packages', *pkgs, '--report',
                       'results/pcrl_dual_design_release_v1/deploy/DEPLOYMENT_CHECK.json']})
p = C.RESULTS / 'lanes' / 'DEPLOY_V1.json'
p.write_text(json.dumps(units, indent=1) + '\n')
print(json.dumps({'lane': str(p.relative_to(C.ROOT)), 'units': len(units), 'packages': pkgs}))
