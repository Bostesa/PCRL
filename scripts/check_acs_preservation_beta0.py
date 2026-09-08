"""Artificial beta-zero comparison with reviewed source; never fits ACS data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from threadpoolctl import threadpool_limits
from tests.test_acs_pca16_preservation_training import data, statistics
from experiments.acs_bottleneck_training import train_preservation_unit, tree_digest


def run():
    reviewed = 'f197933a9ff57b5702b3547d0b8788f15c80189e'
    with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=1):
        directory = Path(temporary)
        source = subprocess.check_output(['git', 'show', reviewed+':experiments/acs_bottleneck_training.py'])
        path = directory/'reviewed_training.py'
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location('reviewed_training', path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        torch.set_num_threads(1)
        x, y, attributes = data()
        validation, validation_y, _ = data(23, 20)
        old.train_pair(x, y, attributes, validation, validation_y, 2, directory/'original',
                       miniature=True, initialization='pca16')
        result = train_preservation_unit(x, y, attributes, validation, validation_y, 2,
            directory/'new', miniature=True, beta=0., fitting_statistics=statistics(x))
        digest = lambda p: tree_digest(torch.load(p, weights_only=True))
        checks = {stage: digest(directory/'original'/stage) == digest(directory/'new'/stage)
                  for stage in ('initialization.pt', 'warm_base.pt', 'warm_adversary.pt')}
        for arm in result['arms']:
            old_arm = 'C_bottleneck' if arm.startswith('C_') else 'D_protected'
            checks[arm] = digest(directory/'original'/old_arm/'final.pt') == digest(directory/'new'/arm/'final.pt')
        assert all(checks.values())
        return {'reviewed_commit': reviewed, 'reviewed_module_sha256': hashlib.sha256(source).hexdigest(),
                'fixture': '37 fitting/23 validation artificial examples,1/1/2 epochs; beta0 only',
                'checks': checks, 'passed': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Optional fresh report path; otherwise stdout')
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError('Use a fresh report path; preserve historical evidence')
    rendered = json.dumps(run(), indent=2, allow_nan=False)+'\n'
    if args.report:
        with args.report.open('x') as handle:
            handle.write(rendered)
    else:
        print(rendered, end='')
