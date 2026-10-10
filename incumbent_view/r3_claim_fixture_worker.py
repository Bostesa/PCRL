"""Queue-only data-free A/B/C fixture runner."""
import argparse,hashlib,io,json,unittest
from pathlib import Path
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env

def main():
    p=argparse.ArgumentParser();p.add_argument('--unit-id',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    context=assert_active_from_env(a.unit_id,'admission')
    from incumbent_view import r3_test_claims
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(r3_test_claims))
    receipt={'schema':'pcrl-incumbent-r3-data-free-claim-fixture-v1','status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'log':stream.getvalue(),'queue':context,'real_data_reads':False}
    path=Path(a.output)
    with path.open('x') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({'status':receipt['status'],'tests_run':receipt['tests_run'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
