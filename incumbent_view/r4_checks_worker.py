"""Queue-only tiny synthetic codec checks; no Adult loader or model fit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import unittest
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--fixture-lock",required=True)
    parser.add_argument("--fixture-lock-sha256",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    active=assert_active_from_env(stage="verification")
    lock=Path(args.fixture_lock)
    if hashlib.sha256(lock.read_bytes()).hexdigest()!=args.fixture_lock_sha256:
        raise PermissionError("fixture lock changed")
    from incumbent_view import r4_test_views
    result=unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(r4_test_views))
    assert_active_from_env(unit_id=active["unit_id"],stage="verification")
    payload={"schema":"pcrl-incumbent-r4-queued-codec-check-v1",
             "unit_id":active["unit_id"],"fixture_lock_sha256":args.fixture_lock_sha256,
             "status":"PASS" if result.wasSuccessful() else "FAIL",
             "tests_run":result.testsRun,"failures":len(result.failures),
             "errors":len(result.errors),"scientific_rows_read":0,"model_fits":0,
             "scope":"synthetic array and exact public-wire codec engineering checks"}
    with os.fdopen(os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644),"w") as out:
        json.dump(payload,out,sort_keys=True,indent=2);out.write("\n")
    print(json.dumps(payload,sort_keys=True))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=="__main__":main()
