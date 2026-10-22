"""Deployment lane (coordinator; label-free): store, build and check packages for each family's strongest feasible nonzero release
(claims/STRONGEST.json: IA_SFG, IB_CSF), the shared twin CTL_TWIN and each release's decisive control, both versions. No release met
D2, so "decisive" is the simplest alternative explanation each reading was not separated from: its SEX-blind spending bracket SPEND
(Q* of the D2 frontier; a shared CTL point named per version by its role member, paths/SHARED_ROLES.json).

    python3 results/pcrl_input_and_pair_release_v1/roles/coordinator/write_deploy_lane.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ipr import common as C          # noqa: E402
from ipr import roles as RO          # noqa: E402

PER_VERSION = {}
for name in ('CTL/CTL_TWIN', 'IA/IA_SFG', 'IB/IB_CSF'):
    PER_VERSION[name.split('/')[1]] = {v: name for v in C.VERSIONS}
for role in ('IA_SPD_HI', 'IB_SPD_HI'):
    PER_VERSION[role] = {v: RO.resolve(f'ROLE:{role}', v) for v in C.VERSIONS}
units, pkgs = [], []
for label, per in PER_VERSION.items():
    for v in C.VERSIONS:
        units.append({'unit': f'IPR_DEPLOY_STORE_{label}_V{v}_V1', 'stage': 'deployment_check', 'rss_gib': 1.5,
                      'argv': ['PY', '-m', 'ipr.deploy', 'store', '--name', per[v], '--version', str(v)]})
for label, per in PER_VERSION.items():
    for v in C.VERSIONS:
        out = f'.private/pcrl_input_and_pair_release_v1/deploy/{label}_V{v}'
        pkgs.append(out)
        units.append({'unit': f'IPR_DEPLOY_BUILD_{label}_V{v}_V1', 'stage': 'deployment_check', 'rss_gib': 1.5,
                      'argv': ['PY', '-m', 'ipr.deploy', 'build', '--name', per[v], '--version', str(v), '--out', out]})
units.append({'unit': 'IPR_DEPLOY_CHECK_V1', 'stage': 'deployment_check', 'rss_gib': 2.0,
              'argv': ['PY', '-m', 'ipr.deploy', 'check', '--packages', *pkgs, '--report',
                       'results/pcrl_input_and_pair_release_v1/deploy/DEPLOYMENT_CHECK.json']})
p = C.RESULTS / 'lanes' / 'DEPLOY_V1.json'
p.write_text(json.dumps(units, indent=1) + '\n')
print(json.dumps({'lane': str(p.relative_to(C.ROOT)), 'units': len(units), 'packages': pkgs,
                  'names': {k: {str(v): n for v, n in per.items()} for k, per in PER_VERSION.items()}}))
