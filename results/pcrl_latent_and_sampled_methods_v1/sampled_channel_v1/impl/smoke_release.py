"""Role-sized data-free cached sampled-release timing for R03 admission.

This fixture generates synthetic score/cell arrays, one draw per synthetic
unit and two recipient projections. It opens no role export or Adult source.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import stat
import time

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env

from .bank import FixedBank
from .channel import GateWeights, JointPolicy
from .materialize import materialize_role
from .release import CachedJointRelease
from .source_bridge import sha_array


def _bank(width: int, counts: tuple[int, ...]) -> FixedBank:
    rows = []
    for predicted, count in enumerate(counts):
        for index in range(count):
            row = np.full(width, (0.15 + index * 0.0001) / (width - 1), np.float64)
            row[predicted] = 0.85 - index * 0.0001
            rows.append(row)
    p = np.stack(rows)
    p /= p.sum(axis=1, keepdims=True)
    return FixedBank.from_source(p, np.arange(len(p), dtype=np.int64),
                                 np.argmax(p, axis=1).astype(np.int64))


def _actions(bank: FixedBank, classes: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(i) for i in np.flatnonzero(bank.decision == c)[:4])
                 for c in classes)


ROLE_SHAPES = {"ATTACK_FIT_NEW": (4064, 4061),
               "INNER_SELECTION": (2235, 2234)}


def run(output_dir: Path, *, unit_id: str) -> dict:
    assert_active_from_env(unit_id=unit_id, stage="a")
    started_cpu, started_wall = time.process_time(), time.monotonic()
    supplied = Path(output_dir)
    root = supplied.resolve(strict=False)
    if (supplied.is_symlink() or supplied.parent.is_symlink() or
            root.exists() or root.parent.name != ".private" or
            not root.parent.is_dir() or root.parent.is_symlink() or
            stat.S_IMODE(root.parent.stat().st_mode) & 0o077):
        raise ValueError("synthetic smoke output must be fresh and owner-only under .private")
    root.mkdir(mode=0o700)
    bank1 = _bank(2, (8, 8))
    bank2 = _bank(6, (64, 64, 64, 64, 64, 1))
    cls1 = (np.arange(64, dtype=np.int64) // 32)
    cls2 = np.r_[np.repeat(np.arange(5, dtype=np.int64), 128), np.array([5], np.int64)]
    policy = JointPolicy(bank1, bank2, _actions(bank1, cls1),
                         _actions(bank2, cls2), GateWeights.zeros(input_width=12))
    summaries = {}
    for role, (n, expected_groups) in ROLE_SHAPES.items():
        destination = root / role
        destination.mkdir(mode=0o700)
        unit = np.arange(n, dtype=np.int64)
        duplicates = n - expected_groups
        unit[-duplicates:] = unit[:duplicates]
        if len(np.unique(unit)) != expected_groups:
            raise AssertionError("synthetic role group count differs")
        cell1 = unit % len(cls1)
        cell2 = unit % len(cls2)
        score1 = np.stack([bank1.probability[policy.actions1[c][0]] for c in cell1])
        score2 = np.stack([bank2.probability[policy.actions2[c][0]] for c in cell2])
        row_id = np.arange(n, dtype=np.int64)
        pair_sha = sha_array(np.column_stack((row_id, unit)))
        service = CachedJointRelease(policy, cache_path=destination / "cache.sqlite3",
                                     release_id="SCV1-synthetic-smoke-v1", source_seed=0,
                                     role=role, realization=0,
                                     source_binding_sha256="a" * 64,
                                     private_root=b"s" * 32, offline=True)
        role_cpu, role_wall = time.process_time(), time.monotonic()
        receipt = materialize_role(service, row_id=row_id, unit=unit,
                                   score1=score1, score2=score2,
                                   cell1=cell1, cell2=cell2,
                                   output_dir=destination,
                                   expected_row_unit_sha256=pair_sha,
                                   source_identity={
                                       "tstar_bank_admission_receipt_sha256": "b" * 64,
                                       "tstar_role_tokens_file_sha256": "c" * 64,
                                       "reference_file_sha256": "d" * 64})
        summaries[role] = {"rows": n, "groups": expected_groups,
                           "status": receipt["status"],
                           "cpu_seconds": time.process_time() - role_cpu,
                           "wall_seconds": time.monotonic() - role_wall,
                           "receipt_sha256": __import__("hashlib").sha256(
                               (destination / "RELEASE_RECEIPT.json").read_bytes()).hexdigest()}
    result = {"schema": "SCV1-synthetic-role-release-smoke-v1",
              "scope": "DATA_FREE_SYNTHETIC_ONLY", "roles": summaries,
              "total_cpu_seconds": time.process_time() - started_cpu,
              "total_wall_seconds": time.monotonic() - started_wall,
              "maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "python_threads": {name: os.environ.get(name) for name in
                                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    target = root / "SMOKE_RECEIPT.json"
    target.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unit-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(Path(args.output_dir), unit_id=args.unit_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
