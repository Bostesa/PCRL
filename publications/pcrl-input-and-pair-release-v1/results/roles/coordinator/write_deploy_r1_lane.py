"""Deployment lane of IA's strongest release since the revision (coordinator; label-free): store, build and check packages for IA_OSG
(claims/STRONGEST.json) and IA_OSG_RO_HI, the occupation-restricted release-only tilt it is at the point more recoverable than (its RO
bracketing control; it fails G1 in version 1, which the check does not test), both versions. The registered packages are in
lanes/DEPLOY_V1.json (deploy/DEPLOYMENT_CHECK.json); this lane writes deploy/DEPLOYMENT_CHECK_R1.json.

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/write_deploy_r1_lane.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402

NAMES = {'IA_OSG': 'IA/IA_OSG', 'IA_OSG_RO_HI': 'IA/IA_OSG_RO_HI'}
units, pkgs = [], []
for label, name in NAMES.items():
    for v in C.VERSIONS:
        units.append({'unit': f'IPR_DEPLOY_STORE_{label}_V{v}_R1', 'stage': 'deployment_check', 'rss_gib': 1.5,
                      'argv': ['PY', '-m', 'ipr.deploy', 'store', '--name', name, '--version', str(v)]})
for label, name in NAMES.items():
    for v in C.VERSIONS:
        out = f'.private/pcrl_input_and_pair_release_v1/deploy/{label}_V{v}'
        pkgs.append(out)
        units.append({'unit': f'IPR_DEPLOY_BUILD_{label}_V{v}_R1', 'stage': 'deployment_check', 'rss_gib': 1.5,
                      'argv': ['PY', '-m', 'ipr.deploy', 'build', '--name', name, '--version', str(v), '--out', out]})
units.append({'unit': 'IPR_DEPLOY_CHECK_R1', 'stage': 'deployment_check', 'rss_gib': 2.0,
              'argv': ['PY', '-m', 'ipr.deploy', 'check', '--packages', *pkgs, '--report',
                       'results/pcrl_input_and_pair_release_v1/deploy/DEPLOYMENT_CHECK_R1.json']})
p = C.RESULTS / 'lanes' / 'DEPLOY_R1.json'
if p.exists():
    raise SystemExit(f'{p.name} exists (lanes are write-once)')
p.write_text(json.dumps(units, indent=1) + '\n')
print(json.dumps({'lane': str(p.relative_to(C.ROOT)), 'units': len(units), 'packages': pkgs}))
