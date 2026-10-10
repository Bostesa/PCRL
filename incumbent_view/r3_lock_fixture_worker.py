"""Queue-only data-free dual-lock bridge checks."""
import argparse
import hashlib
import io
import json
import unittest
from pathlib import Path
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--unit-id',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();context=assert_active_from_env(args.unit_id,'admission')
    from incumbent_view import r3_lock_bridge_tests,r3_recovery_tests
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module)
                             for module in (r3_lock_bridge_tests,r3_recovery_tests)])
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt={'schema':'pcrl-incumbent-r3-lock-bridge-fixture-v1','status':'PASS' if result.wasSuccessful() else 'FAIL',
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'queue':context,'log':stream.getvalue(),'real_data_reads':False}
    path=Path(args.output)
    with path.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'status':receipt['status'],'tests_run':result.testsRun,'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
