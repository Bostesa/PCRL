"""Queued source/cache engineering fixture receipt."""
import argparse,json,unittest
from .scheduler import assert_active_from_env
from .source import write_once

def run(output):
    active=assert_active_from_env(stage='admission')
    suite=unittest.defaultTestLoader.loadTestsFromNames(['conditional_release.tests.test_source_deployment','conditional_release.tests.test_cached_person_wire'])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'schema':'CBR1-r1-synthetic-engineering-v1','unit_id':active['unit_id'],'status':'PASS' if result.wasSuccessful() else 'FAIL',
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'scientific_arrays_read':False}
    write_once(output,receipt,False)
    if not result.wasSuccessful():raise RuntimeError('source deployment fixture failure')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
