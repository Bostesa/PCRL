"""Queue-only data-free reader/metric contract tests."""
import argparse,json,unittest
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    from .scheduler import assert_active_from_env
    assert_active_from_env(stage='admission')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('task_conditioned_representation.test_attacks'))
    receipt={'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'successful':result.wasSuccessful()}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(receipt,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
