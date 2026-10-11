"""Queued package-dispatch/source/cache sampler engineering extension."""
import argparse,unittest
from .scheduler import assert_active_from_env
from .source import write_once

def run(output):
    active=assert_active_from_env(stage='admission')
    suite=unittest.defaultTestLoader.loadTestsFromNames(['conditional_release.tests.test_source_deployment','conditional_release.tests.test_cached_person_wire','conditional_release.tests.test_sampler_contract','conditional_release.tests.test_package_dispatch'])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_once(output,{'schema':'CBR1-package-engineering-v1','unit_id':active['unit_id'],'tests_run':result.testsRun,'status':'PASS' if result.wasSuccessful() else 'FAIL',
        'failures':len(result.failures),'errors':len(result.errors),'scientific_arrays_read':False,'scope':'predeclared family package dispatch and unchanged sampling law'},False)
    if not result.wasSuccessful():raise RuntimeError('package fixture failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
