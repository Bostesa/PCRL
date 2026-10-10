"""One-recipient deployment of a selected bank-anchored policy package.

The provider computes both frozen task routes privately. Output is exactly
`code`, `prob`, `decision` for the requested recipient. Stochastic policies
require a private, precommitted per-row draw file; it is never exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

import numpy as np

from code_witness.bank_deploy import load_bank_service
from dpc.deploy import load_permitted_input, schema_names, schema_sha256
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_package import load_policy

from .r1_runtime import release_pair_from_raw


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _draws(path: Path, expected_sha256: str, rows: int,
           arm_id: str) -> np.ndarray:
    if (path.is_symlink() or ".private" not in path.parts or
            stat.S_IMODE(path.stat().st_mode) & 0o077 or
            _sha(path) != expected_sha256):
        raise PermissionError("private frozen draw file is not pinned or owner-only")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"uniforms"}:
            raise ValueError("draw file cannot carry labels or extra fields")
        value = np.array(archive["uniforms"], copy=True)
    expected_shape = (rows, 2) if arm_id in ("P", "L") else (rows,)
    if (value.dtype != np.float64 or value.shape != expected_shape or
            not np.isfinite(value).all() or np.any((value < 0) | (value >= 1))):
        raise ValueError("frozen draw dimensions or probabilities differ")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bank_anchored.r1_deploy",
                                     allow_abbrev=False)
    for name in ("source-root", "source-sha256", "qpc-root", "public-table",
                 "public-table-sha256", "schema", "schema-sha256",
                 "policy-dir", "policy-receipt-sha256",
                 "source-binding-sha256", "input", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--purpose", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--draws")
    parser.add_argument("--draws-sha256")
    args = parser.parse_args(argv)
    names = tuple(schema_names(args.schema))
    if schema_sha256(names) != args.schema_sha256:
        raise ValueError("ordered 83-column schema hash differs")
    policy, receipt = load_policy(Path(args.policy_dir),
                                  receipt_sha256=args.policy_receipt_sha256)
    if (receipt["source_seed"] != args.seed or
            receipt["source_binding_sha256"] != args.source_binding_sha256):
        raise ValueError("selected policy source/seed binding differs")
    service = load_bank_service(
        source_root=Path(args.source_root), qpc_root=Path(args.qpc_root),
        table_path=Path(args.public_table),
        expected_table_sha256=args.public_table_sha256,
        expected_source_sha256=args.source_sha256,
        expected_schema_sha256=args.schema_sha256,
        ordered_names=names, bank="W_TSTAR", seed=args.seed)
    for purpose, (bank, part) in enumerate(((policy.bank1, service.policy_pair.p1),
                                             (policy.bank2, service.policy_pair.p2))):
        if (not np.array_equal(bank.probability, service.tables[purpose]) or
                not np.array_equal(bank.decision, part.token_class) or
                tuple(bank.old_to_global) != tuple(range(len(bank.probability))) or
                any(actions[0] != int(part.cell_token[cell])
                    for cell, actions in enumerate((policy.actions1,
                                                    policy.actions2)[purpose]))):
            raise ValueError("selected policy bank or identity action differs from source wire")
    raw = load_permitted_input(args.input, list(names))
    if receipt["arm_id"] in ("B0", "D"):
        if args.draws is not None or args.draws_sha256 is not None:
            raise ValueError("deterministic release cannot read sampling coins")
        # D inherits the one-uniform sampler API but its law is one-hot. A
        # fixed selector is a call convention, not a sampled release coin.
        uniforms = (None if receipt["arm_id"] == "B0" else
                    np.full(len(raw), 0.5, dtype=np.float64))
    else:
        if args.draws is None or args.draws_sha256 is None:
            raise ValueError("selected nonidentity policy needs pinned private draws")
        uniforms = _draws(Path(args.draws), args.draws_sha256, len(raw),
                          receipt["arm_id"])
    wire = release_pair_from_raw(service, policy, raw, names,
                                 uniforms)[args.purpose]
    output = Path(args.output)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **wire)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": "pcrl-bank-anchored-deployment-v1",
                      "arm_id": receipt["arm_id"], "seed": args.seed,
                      "purpose": args.purpose, "rows": len(raw),
                      "policy_receipt_sha256": args.policy_receipt_sha256,
                      "public_table_sha256": args.public_table_sha256,
                      "release_sha256": _sha(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
