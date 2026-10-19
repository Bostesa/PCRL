"""Verifier stage V2 (procedural, label-free): agreement of the two W100 deployment records (RULES.md section 7, T3: the verifier's
gating DEPLOY_W100.json and the coordinator's non-gating deploy/DEPLOYMENT_CHECKS_W100.json; a disagreement blocks the trigger).

PCRL controlled training-size campaign (TSZ1).

    python3 .../v2_deploy_agree.py --tag V1
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
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    mine_p = V.VER_DIR / 'DEPLOY_W100.json'
    theirs_p = RES / 'deploy' / 'DEPLOYMENT_CHECKS_W100.json'
    mine, theirs = json.loads(mine_p.read_text()), json.loads(theirs_p.read_text())
    per = {}
    for v in ('0', '1'):
        m = mine[v]
        pk = Path(m['package_dir']).name
        t = theirs['packages'].get(pk)
        if t is None:
            per[v] = {'package': pk, 'in_coordinator_record': False}
            continue
        t_bit = all(all(x.values()) for x in t['bitwise_equal_to_stored_DEV_REPORT_wire'].values())
        t_ref = all(str(x).startswith('refused') for x in t['refusals'].values())
        per[v] = {'package': pk, 'in_coordinator_record': True, 'verifier_bitwise': m['bitwise'], 'coordinator_bitwise': t_bit,
                  'verifier_refusals_all_pass': m['refusals_all_pass'], 'coordinator_all_refused': t_ref,
                  'coordinator_refusals_checked': sorted(t['refusals']), 'same_name_version': [t['name'], t['version']] == [m['name'], int(v)]}
    agree = all(x.get('in_coordinator_record') and x['verifier_bitwise'] == x['coordinator_bitwise']
                and x['verifier_refusals_all_pass'] == x['coordinator_all_refused'] and x['same_name_version'] for x in per.values())
    chk = {'id': f'V2.TRIGGER.DEPLOY_RECORDS_AGREE__{a.tag}', 'kind': 'PROCEDURAL_COMPLIANCE',
           'what': 'T3: the verifier\'s gating DEPLOY_W100.json and the coordinator\'s non-gating DEPLOYMENT_CHECKS_W100.json agree per version '
                   '(bitwise and refusals) for the same packages (a disagreement would block the trigger)',
           'inputs': V.inputs_sha([mine_p, theirs_p]), 'expected': 'agreement',
           'observed': {'per_version': per, 'coordinator_all_bitwise': theirs.get('all_bitwise'), 'coordinator_all_refused': theirs.get('all_refused')},
           'status': 'PASS' if agree else 'FAIL', 'code_sha256': V.code_hashes(__file__, V.__file__)}
    V.write_record(f'V2_DEPLOY_AGREE__{a.tag}.json', chk)
    print(json.dumps(V.append_checks([chk])))
    print(json.dumps({chk['id']: chk['status'], 'per': per}, default=str)[:1500])


if __name__ == '__main__':
    main()
