"""Queue entry point: run the engineering tests (synthetic only) and write a receipt."""
import argparse, json, sys
from pathlib import Path


def run(spec_path):
    import pytest
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'verification')
    rc = pytest.main(['-q', '-p', 'no:cacheprovider', str(Path(__file__).with_name('test_runner.py'))])
    with Path(cfg['output']).open('x') as f:
        json.dump({'queue_unit_id': active['unit_id'], 'pytest_returncode': int(rc), 'passed': int(rc) == 0}, f)
    if rc != 0:
        raise AssertionError('tests failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
