"""Synthetic-only role-sized timing of fixed diagnostic codes and wire writer.

No source/Adult file is opened. R03 must admit this numerical smoke under a
dated amendment; its JSON output contains only timing and fixed dimensions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
import time

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env

from ..impl.bank import FixedBank, build_actions
from ..impl.reference_materialize import materialize_reference_role
from ..impl.source_bridge import sha_array
from .diagnostic_codes import assert_group_coherent, select_codes


ROLE_SHAPES = {"ATTACK_FIT_NEW": (4064, 4061),
               "INNER_SELECTION": (2235, 2234)}


def _bank(width: int, per_class: tuple[int, ...]) -> tuple[FixedBank, np.ndarray]:
    vectors, classes = [], []
    for label, count in enumerate(per_class):
        for index in range(count):
            winner = .51 + .39 * (index + 1) / (count + 1)
            p = np.full(width, (1. - winner) / (width - 1), np.float64)
            p[label] = winner
            vectors.append(p)
            classes.append(label)
    probabilities = np.stack(vectors)
    classes = np.asarray(classes, np.int64)
    bank = FixedBank.from_source(probabilities,
                                 np.arange(len(probabilities), dtype=np.int64), classes)
    return bank, classes


def run(*, unit_id: str, arm_id: str, role: str, rows: int,
        output_path: Path) -> dict:
    assert_active_from_env(unit_id=unit_id, stage="a")
    if (arm_id not in ("A_IDENTITY", "A_CLASS_ONLY") or
            role not in ROLE_SHAPES or rows != ROLE_SHAPES[role][0]):
        raise ValueError("synthetic arm/size differs")
    start_cpu, start_wall = time.process_time(), time.monotonic()
    bank1, old_class1 = _bank(2, (8, 8))
    bank2, old_class2 = _bank(6, (64, 64, 64, 64, 64, 1))
    fine_to_old1 = np.tile(np.arange(16, dtype=np.int64), 4)
    fine_to_old2 = np.concatenate((np.tile(np.arange(320, dtype=np.int64), 2),
                                    np.asarray([320], np.int64)))
    actions1 = build_actions(bank1, input_class=old_class1[fine_to_old1],
                             fine_to_old_token=fine_to_old1)
    actions2 = build_actions(bank2, input_class=old_class2[fine_to_old2],
                             fine_to_old_token=fine_to_old2)
    row_id = np.arange(rows, dtype=np.int64)
    unit = row_id.copy()
    duplicates = rows - ROLE_SHAPES[role][1]
    if duplicates:
        unit[-duplicates:] = unit[:duplicates]
    if len(np.unique(unit)) != ROLE_SHAPES[role][1]:
        raise AssertionError("synthetic role group count differs")
    cell1 = unit % len(actions1)
    cell2 = unit % len(actions2)
    code1 = select_codes(bank1, actions1, cell1, arm_id=arm_id)
    code2 = select_codes(bank2, actions2, cell2, arm_id=arm_id)
    assert_group_coherent(unit, code1, code2)
    with TemporaryDirectory(dir=Path(gettempdir()).resolve()) as temporary:
        private = Path(temporary) / ".private"
        private.mkdir(mode=0o700)
        receipt = materialize_reference_role(
            row_id=row_id, unit=unit,
            prob1=bank1.probability[code1], prob2=bank2.probability[code2],
            decision1=bank1.decision[code1], decision2=bank2.decision[code2],
            code1=code1, code2=code2, arm_id=arm_id, seed=0,
            role=role, release_id=f"{arm_id}-s0-v1",
            source_binding_sha256="a" * 64,
            source_identity={"role_source_file_sha256": "b" * 64,
                             "reference_file_sha256": "c" * 64,
                             "bank_receipt_sha256": "d" * 64,
                             "source_audit_sha256": "e" * 64},
            interface_kind="finite",
            bank_sha256_by_recipient={"r1": sha_array(bank1.probability),
                                      "r2": sha_array(bank2.probability)},
            expected_row_unit_sha256=sha_array(np.column_stack((row_id, unit))),
            output_dir=private / "release",
            bank1=bank1.probability, bank2=bank2.probability)
        if receipt["rows"] != rows or receipt["status"] != "FROZEN_DETERMINISTIC":
            raise AssertionError("synthetic writer did not complete")
    summary = {"schema": "SCV1-synthetic-diagnostic-release-timing-v1",
               "unit_id": unit_id, "arm_id": arm_id, "role": role,
               "rows": rows, "groups": ROLE_SHAPES[role][1],
               "bank_sizes": [16, 321],
               "fine_cells": [64, 641], "role_source": "synthetic_only",
               "cpu_seconds": time.process_time() - start_cpu,
               "wall_seconds": time.monotonic() - start_wall,
               "complete": True}
    payload = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
    target = Path(output_path)
    if target.exists() or not target.parent.is_dir():
        raise FileExistsError("synthetic timing receipt must be new")
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--arm-id", choices=("A_IDENTITY", "A_CLASS_ONLY"), required=True)
    parser.add_argument("--role", choices=("ATTACK_FIT_NEW", "INNER_SELECTION"), required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(unit_id=args.unit_id, arm_id=args.arm_id,
                 role=args.role, rows=args.rows,
                 output_path=args.output)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
