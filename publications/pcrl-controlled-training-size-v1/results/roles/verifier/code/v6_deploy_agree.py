"""Verifier stage V6 (procedural, label-free): the verifier's deployment record of the final packages (records/V_DEPLOY__<tag>.json)
vs the coordinator's deploy/DEPLOYMENT_CHECKS.json (per package: bitwise and refusals), and the package set vs PROTOCOL section 11.

PCRL controlled training-size campaign (TSZ1).

    python3 .../v6_deploy_agree.py --deploy-tag LIVE1 --coord deploy/DEPLOYMENT_CHECKS.json --tag V1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

RES = V.RESULTS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--deploy-tag', required=True); ap.add_argument('--coord', required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    mp = V.RECORDS / f'V_DEPLOY__{a.deploy_tag}.json'
    cp = RES / a.coord
    mine, theirs = json.loads(mp.read_text())['packages'], json.loads(cp.read_text())
    per, bad = {}, []
    for pk, m in mine.items():
        t = theirs['packages'].get(pk)
        if t is None:
            per[pk] = {'in_coordinator_record': False}; bad.append(f'{pk} missing in the coordinator record'); continue
        t_bit = all(all(x.values()) for x in t['bitwise_equal_to_stored_DEV_REPORT_wire'].values())
        t_ref = all(str(x).startswith('refused') for x in t['refusals'].values())
        per[pk] = {'name': m['name'], 'version': m['version'], 'verifier_bitwise': m['bitwise'], 'coordinator_bitwise': t_bit,
                   'verifier_refusals': m['refusals_all_pass'], 'coordinator_refusals': t_ref, 'verifier_income': m['income_all_bitwise'],
                   'verifier_own_pipeline_within_1e-12': m.get('own_pipeline_within_1e-12')}
        if m['bitwise'] != t_bit or m['refusals_all_pass'] != t_ref or [t['name'], t['version']] != [m['name'], m['version']]:
            bad.append(f'{pk} disagrees')
    required = {('A1:A1__S100__V0', 0), ('A1:A1__S100__V1', 1), ('A1:A1__S50_O0__V0', 0)}
    sel = {v: json.loads((RES / 'conditional' / f'SELECTION_V{v}.json').read_text())['selected'] for v in (0, 1)}
    for v in (0, 1):
        for arm in ('LOC', 'PAIR'):
            if float(sel[v][arm]) > 0:
                required.add((f'CORR:{arm}@s{float(sel[v][arm])}', v))
                required.add((f'CORR:CONT@s{float(sel[v][arm])}', v))
    have = {(m['name'], m['version']) for m in mine.values()}
    ctrl = any(n.startswith('CTRL:') and n.endswith(f'__V{v}') for n, v in have for _ in [0])
    missing = sorted(f'{n} v{v}' for n, v in required - have)
    ok = not bad and not missing and all(x.get('verifier_bitwise') and x.get('verifier_refusals') for x in per.values())
    chk = {'id': f'V6.DEPLOY.FINAL_PACKAGES_AGREE__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
           'what': 'final deployment packages (PROTOCOL section 11): the required set is present (W100_v, the S100 TASK_REFIT control, the '
                   'representative A1_PO_16 S50_O0 v0 fit, and every selected nonzero correction with its matched continuation); the verifier\'s '
                   'deployment results agree with the coordinator\'s DEPLOYMENT_CHECKS.json per package (bitwise and refusals)',
           'inputs': V.inputs_sha([mp, cp]), 'expected': 'agreement; required set present',
           'observed': {'per_package': per, 'problems': bad, 'required_missing': missing, 'control_package_present': ctrl,
                        'coordinator_all_bitwise': theirs.get('all_bitwise'), 'coordinator_all_refused': theirs.get('all_refused')},
           'status': 'PASS' if ok else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}
    V.write_record(f'V6_DEPLOY_AGREE__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'problems': bad, 'missing': missing}))


if __name__ == '__main__':
    main()
