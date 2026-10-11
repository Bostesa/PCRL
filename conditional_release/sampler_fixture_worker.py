"""Queue-only sampler engineering tests before the science lock."""
import argparse,unittest
from .scheduler import assert_active_from_env
from .source import write_once

def run(output):
    active=assert_active_from_env(stage='admission')
    suite=unittest.defaultTestLoader.loadTestsFromNames(['conditional_release.tests.test_source_deployment','conditional_release.tests.test_cached_person_wire','conditional_release.tests.test_sampler_contract'])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_once(output,{'schema':'CBR1-explicit-sampler-engineering-v1','unit_id':active['unit_id'],'tests_run':result.testsRun,
        'status':'PASS' if result.wasSuccessful() else 'FAIL','failures':len(result.failures),'errors':len(result.errors),
        'scientific_arrays_read':False,'contract':'joint or two recipient-domain independent coins; sequential conditional draws; duplicate person cache; no new wire fields'},False)
    if not result.wasSuccessful():raise RuntimeError('sampler engineering failure')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
