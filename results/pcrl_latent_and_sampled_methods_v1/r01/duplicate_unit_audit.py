"""Aggregate-only duplicate-unit check on admitted score and fine-cell files.

Usage: python3 duplicate_unit_audit.py REPAIR_WORKTREE PORTFOLIO_WORKTREE
Requires SOURCE_AUDIT.json PASS. No labels or raw features are opened.
"""

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_sources import member, sha_file


def main():
    repair, portfolio = map(Path, sys.argv[1:3])
    rpackage = repair / "results/pcrl_utility_reference_repair_v1"
    ppackage = portfolio / "results/pcrl_method_portfolio_v1"
    manifest = json.loads((rpackage / "m1/DATA_ACCESS_MANIFEST.json").read_text())
    out = {}
    for seed in (0, 1):
        root = ppackage / f"shared/.private/tstar_score_bank_s{seed}_v1"
        receipt = json.loads((root / "BANK_ADMISSION_RECEIPT.json").read_text())
        per_role = {}
        for role in ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION"):
            ref_path = rpackage / manifest["role_file_relative_path"][f"{role}:references_s{seed}"]
            tok_path = root / f"tokens_{role}.npz"
            assert sha_file(ref_path) == manifest["role_file_sha256"][f"{role}:references_s{seed}"]
            assert sha_file(tok_path) == receipt["private_output_sha256"][tok_path.name]
            with ZipFile(tok_path) as archive:
                assert set(archive.namelist()) == {f"{name}.npy" for name in
                    ("row_id", "unit", "cell_1", "cell_2", "token_1", "token_2")}
            ref_ids, ref_groups = member(ref_path, "row_id"), member(ref_path, "unit")
            tok_ids, tok_groups = member(tok_path, "row_id"), member(tok_path, "unit")
            assert np.array_equal(ref_ids, tok_ids) and np.array_equal(ref_groups, tok_groups)
            values = {name: member(path, name) for name, path in
                      (("U0_1", ref_path), ("U0_2", ref_path),
                       ("cell_1", tok_path), ("cell_2", tok_path))}
            groups, inverse, counts = np.unique(ref_groups, return_inverse=True, return_counts=True)
            duplicate = np.flatnonzero(counts > 1)
            mismatch = {name: 0 for name in values}
            joint_mismatch = 0
            for group_index in duplicate:
                positions = np.flatnonzero(inverse == group_index)
                failed = [name for name, value in values.items()
                          if not np.array_equal(value[positions],
                                                np.broadcast_to(value[positions[0]], value[positions].shape))]
                for name in failed:
                    mismatch[name] += 1
                joint_mismatch += bool(failed)
            per_role[role] = {"duplicate_groups": len(duplicate),
                              "duplicate_rows": int(counts[duplicate].sum()),
                              "groups_with_any_mismatch": joint_mismatch,
                              "groups_with_mismatch_by_field": mismatch}
        out[str(seed)] = per_role
    print(json.dumps({"schema": "pcrl-r01-duplicate-unit-score-cell-audit-v1",
                      "status": "AGGREGATE_ONLY", "seeds": out,
                      "scope": "U0 score and T* fine-cell equality within duplicate units; no X, labels, or identifiers output"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
