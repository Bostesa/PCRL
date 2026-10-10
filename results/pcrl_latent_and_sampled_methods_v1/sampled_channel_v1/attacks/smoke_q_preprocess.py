"""R03-only data-free full-grid timing for two public Q laws.

This unit contains no Adult rows, labels, role store, or historical bank read.
It measures the 64 by 641 compile and two-file artifact write for a
selected-like law and a deterministic fixed-map law.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import stat
import time
from pathlib import Path

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env
from . import q_aware, q_artifact, smoke_attack, wire


UNIT_ID = "A-Q-PREPROCESS-PUBLIC-GRID-SMOKE-001"
OUTPUT_NAMES = ["SMOKE_RECEIPT.json", "selected/Q_TABLE_RECEIPT.json",
                "selected/q_table.npz", "fixed/Q_TABLE_RECEIPT.json",
                "fixed/q_table.npz"]


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _admit(args: argparse.Namespace) -> Path:
    active = assert_active_from_env(unit_id=args.unit_id, stage="a")
    if args.unit_id != UNIT_ID:
        raise PermissionError("Q smoke unit ID differs")
    original = Path(args.original_lock).resolve(strict=True)
    amendment = Path(args.amendment).resolve(strict=True)
    if (_sha(original) != active["seal_sha256"] or
            _sha(amendment) != args.amendment_sha256):
        raise PermissionError("Q smoke source seal differs")
    lock = json.loads(original.read_text())
    dated = json.loads(amendment.read_text())
    if (lock.get("status") != "LOCALLY_SEALED_D_STAGE" or
            dated.get("schema") != "pcrl-latent-sampled-dated-amendment-v1" or
            dated.get("status") != "LOCALLY_SEALED_SYNTHETIC_A" or
            dated.get("input_scope") != "SYNTHETIC_ONLY" or
            dated.get("stage_scope") != ["a"] or
            dated.get("authorized_unit_id") != UNIT_ID or
            dated.get("original_lock_sha256") != active["seal_sha256"] or
            dated.get("implementation_source_sha256") != _sha(Path(__file__))):
        raise PermissionError("Q smoke dated synthetic-only authorization differs")
    queue_root = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((queue_root / "state.json").read_text())
    latest = [row for row in state["amendments"] if "a" in row.get("stages", [])]
    spec = state["units"][UNIT_ID]["spec"]
    output = Path(args.private_root)
    root = output.resolve(strict=False)
    expected = [str(root / name) for name in OUTPUT_NAMES]
    if (not latest or latest[-1]["sha256"] != args.amendment_sha256 or
            Path(latest[-1]["path"]).resolve() != amendment or
            spec.get("amendment_sha256") != args.amendment_sha256 or
            spec.get("stage") != "a" or
            spec.get("data_access") != "SYNTHETIC_ONLY" or
            spec.get("deps") != ["R04-A-PREFIT-SOURCE-001"] or
            spec.get("outputs") != expected or root.exists() or
            output.is_symlink() or ".private" not in root.parts):
        raise PermissionError("Q smoke queue spec/output differs")
    private = Path(*root.parts[:root.parts.index(".private") + 1])
    for path in (p for p in root.parents if p == private or private in p.parents):
        if (path.is_symlink() or not path.is_dir() or
                path.stat().st_uid != os.getuid() or
                stat.S_IMODE(path.stat().st_mode) & 0o077):
            raise PermissionError("Q smoke custody ancestor is not owner-only")
    required = (original, amendment, *(Path(module.__file__).resolve(strict=True)
                 for module in (q_aware, q_artifact, smoke_attack, wire)),
                Path(__file__).resolve(strict=True))
    declared = {Path(row["path"]).resolve(): row["sha256"]
                for row in spec.get("sources", [])}
    if any(declared.get(path) != _sha(path) for path in required):
        raise PermissionError("Q smoke module/source bytes differ from queued spec")
    return root


def _law(bank1: wire.PublicBank, bank2: wire.PublicBank, fixed: bool):
    choices = [{label: np.flatnonzero(np.argmax(bank.probability, axis=1) == label)
                for label in range(bank.probability.shape[1])}
               for bank in (bank1, bank2)]

    def sparse(score1: np.ndarray, score2: np.ndarray, i: int, j: int):
        left = choices[0][int(np.argmax(score1))]
        right = choices[1][int(np.argmax(score2))]
        if fixed:
            return ((int(left[0]), int(right[0]), 1.0),)
        return tuple((int(left[(i + k) % len(left)]),
                      int(right[(j + k) % len(right)]), 1.0 / 16.0)
                     for k in range(16))

    return sparse


def run(args: argparse.Namespace) -> dict:
    root = _admit(args)
    guard = lambda: assert_active_from_env(unit_id=UNIT_ID, stage="a")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    p1, c1 = smoke_attack._scores(64, 2, 16)
    p2, c2 = smoke_attack._scores(641, 6, 321)
    banks = (smoke_attack._bank(1, p1), smoke_attack._bank(2, p2))
    root.mkdir(mode=0o700)
    cases = {}
    for case in ("selected", "fixed"):
        guard()
        t0, c0 = time.monotonic(), time.process_time()
        table = q_aware.compile_uniform_prototype_features(
            banks[0], banks[1], c1, c2,
            _law(*banks, fixed=case == "fixed"),
            public_centroids_sha256="f" * 64)
        compile_wall = time.monotonic() - t0
        child = root / case
        child.mkdir(mode=0o700)
        artifact = q_artifact.save_q_table_artifact(
            table, child, policy_receipt_sha256=("a" if case == "selected" else "b") * 64)
        cases[case] = {"prototype_count": table.prototype_count,
                       "compile_wall_s": compile_wall,
                       "compile_and_save_wall_s": time.monotonic() - t0,
                       "compile_and_save_cpu_s": time.process_time() - c0,
                       "table_content_sha256": table.table_sha256,
                       "q_receipt_sha256": artifact["receipt_sha256"],
                       "q_file_bytes": (child / "q_table.npz").stat().st_size}
    receipt = {"schema": "r17-public-Q-full-grid-synthetic-smoke-v1",
               "status": "COMPLETE", "unit_id": UNIT_ID,
               "synthetic_only": True, "adult_role_values_read": False,
               "fine_cells": [64, 641], "bank_codes": [16, 321],
               "cases": cases, "wall_s": time.monotonic() - start_wall,
               "cpu_s": time.process_time() - start_cpu,
               "ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "source_sha256": _sha(Path(__file__).resolve())}
    target = root / "SMOKE_RECEIPT.json"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    for name in ("unit-id", "original-lock", "amendment", "amendment-sha256",
                 "private-root"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(run(parser.parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
