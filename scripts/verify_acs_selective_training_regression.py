"""Artificial exact-update regression against the reviewed e169195 training file.

Uses git show to obtain the immutable historical source; never loads ACS or any
historical scientific fitted model. All disposable checkpoints remain temporary.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import acs_bottleneck_training as training

REVIEWED_COMMIT = "e1691955c330d4782ccd600d171ae9571f23fdb5"


def artificial_data(n, seed):
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n, 32)) * np.linspace(.2, 3., 32) + np.linspace(-3., 5., 32)).astype(np.float32)
    x[:, 12] = 7.
    source = {k: (np.arange(n) % 2).astype(np.int64) for k in training.SOURCE_SCHEMA}
    source["income_binary"][::9] = -1
    attributes = {"SEX": np.arange(n) % 2, "RAC1P": np.arange(n) % 9}
    attributes["RAC1P"][::11] = -1
    return x, source, attributes


def fitting_statistics(x):
    raw = x.astype(np.float64)
    std = raw.std(0)
    return {"mean": raw.mean(0).tolist(), "scale": np.where(std > 1e-12, std, 1.).tolist()}


def run():
    started = time.perf_counter()
    historical_source = subprocess.check_output(
        ["git", "show", f"{REVIEWED_COMMIT}:experiments/acs_bottleneck_training.py"], cwd=ROOT)
    x, source, attributes = artificial_data(37, 43)
    xv, source_val, _ = artificial_data(23, 44)
    checks = {}
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="acs_selective_regression_") as temporary, threadpool_limits(limits=1):
            directory = Path(temporary)
            module_path = directory / "reviewed_training.py"
            module_path.write_bytes(historical_source)
            spec = importlib.util.spec_from_file_location("reviewed_acs_preservation_training", module_path)
            historical = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(historical)
            old = historical.train_preservation_unit(x, source, attributes, xv, source_val, 2, directory / "old",
                beta=1., miniature=True, fitting_statistics=fitting_statistics(x))
            current = training.train_selective_unit(x, source, attributes, xv, source_val, 2, directory / "new",
                teacher_targets=x[:, :16], teacher_name="R", reconstruction_weight=.1,
                fitting_statistics=fitting_statistics(x), miniature=True)
            paths = [(name, name) for name in ("initialization.pt", "warm_base.pt", "warm_adversary.pt")]
            paths += [(f"{arm}_persistent/{stage}.pt", f"{arm}/{stage}.pt") for arm in ("C", "D") for stage in ("fork", "final")]
            for before, after in paths:
                old_state = torch.load(directory / "old" / before, weights_only=True)
                new_state = torch.load(directory / "new" / after, weights_only=True)
                old_hash, new_hash = training.tree_digest(old_state), training.tree_digest(new_state)
                assert old_hash == new_hash, (before, after, "model/adversaries/full Adam/counters differ")
                checks[after] = {"historical_path": before, "historical_sha256": old_hash, "new_sha256": new_hash, "bitwise_exact": True}
            assert old["metadata"]["config"] == current["metadata"]["config"]
            assert old["metadata"]["schedules"] == current["metadata"]["schedules"]
    finally:
        torch.set_num_threads(original_threads)
    return {"passed": True, "scope": "miniature artificial inputs only; no historical scientific fitting",
            "reviewed_commit": REVIEWED_COMMIT,
            "reviewed_module_source_sha256": hashlib.sha256(historical_source).hexdigest(),
            "current_module_source_sha256": hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "comparison": "new R/rho=.1 versus historical beta=1 persistent; all model, observer, Adam and counter state",
            "checkpoints": checks, "historical_config_and_schedules_exact": True,
            "runtime_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Fresh JSON report; stdout if omitted")
    args = parser.parse_args()
    if args.report and args.report.exists():
        raise FileExistsError("Preserve existing regression evidence")
    rendered = json.dumps(run(), indent=2, allow_nan=False) + "\n"
    if args.report:
        with args.report.open("x") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
