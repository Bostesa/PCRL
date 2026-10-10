"""Queued synthetic final-attacker admission checks; no scientific source read."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import unittest
from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--fixture-lock",required=True)
    p.add_argument("--fixture-lock-sha256",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    active=assert_active_from_env(stage="verification")
    lock=Path(a.fixture_lock)
    if hashlib.sha256(lock.read_bytes()).hexdigest()!=a.fixture_lock_sha256:
        raise PermissionError("fixture lock changed")
    from incumbent_view import r4_test_views,r4_test_worker
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module)
                              for module in (r4_test_views,r4_test_worker)])
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    assert_active_from_env(unit_id=active["unit_id"],stage="verification")
    receipt={"schema":"pcrl-incumbent-r4-queued-admission-check-v1", "unit_id":active["unit_id"],
             "fixture_lock_sha256":a.fixture_lock_sha256,
             "tests_run":outcome.testsRun,"failures":len(outcome.failures),"errors":len(outcome.errors),
             "status":"PASS" if outcome.wasSuccessful() else "FAIL",
             "real_person_rows_read":0,"scientific_reader_fits":0,
             "scope":"exact public wire, label allowlist preopen, positive sampled support, original INNER gate semantics"}
    with os.fdopen(os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644),"w") as f:
        json.dump(receipt,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps(receipt,sort_keys=True))
    if not outcome.wasSuccessful():raise SystemExit(1)


if __name__=="__main__":main()
