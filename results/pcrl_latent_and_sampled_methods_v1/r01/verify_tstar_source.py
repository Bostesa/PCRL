"""Emit aggregate verification of R01 T* role-only source loader.

Usage: python -m results.pcrl_latent_and_sampled_methods_v1.r01.verify_tstar_source
       PORTFOLIO_PACKAGE REPAIR_MANIFEST SOURCE_AUDIT
No labels, raw X, source mixed archives, or excluded roles are opened.
"""

import json
import sys
from pathlib import Path

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_file
from results.pcrl_latent_and_sampled_methods_v1.r01.tstar_source import (
    AccessRefused, TStarSourceStore,
)


def main():
    portfolio, repair, audit = map(Path, sys.argv[1:4])
    out = {}
    for seed in (0, 1):
        store = TStarSourceStore(portfolio, repair, audit, seed)
        banks = {}
        for recipient in (1, 2):
            bank = store.read_bank(recipient)
            banks[f"r{recipient}"] = {
                "input_fine_cells": len(bank["input_class"]),
                "canonical_actions": len(bank["canonical_action_id"]),
                "class_order": list(range(bank["probability"].shape[1])),
                "all_actions_reachable_by_identity_map": bool(
                    np.array_equal(np.unique(bank["fine_to_hcal_token"]), bank["canonical_action_id"])),
                "bank_file_sha256": store.receipt["private_output_sha256"][f"bank_r{recipient}.npz"],
            }
        roles = {}
        for role in ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION"):
            values = store.read_role_cells(role)
            roles[role] = {
                "rows": len(values["row_id"]),
                "groups": len(np.unique(values["unit"])),
                "ordered_row_unit_pair_sha256": store.receipt["role_receipts"][role]["row_unit_pair_sha256"],
                "source_reference_file_sha256": store.receipt["role_receipts"][role]["source_reference_sha256"],
                "token_file_sha256": store.receipt["private_output_sha256"][f"tokens_{role}.npz"],
            }
        for forbidden in ("CALIBRATION_HELDOUT", "HEAD_VALIDATION", "OSF_DEVELOPMENT_ASSESSMENT"):
            try:
                store.read_role_cells(forbidden)
            except AccessRefused:
                pass
            else:
                raise AssertionError("excluded T* role accepted")
        out[str(seed)] = {"receipt_sha256": sha_file(store.root / "BANK_ADMISSION_RECEIPT.json"),
                          "source_binding_sha256": store.source_binding_sha256,
                          "banks": banks, "roles": roles}
    result = {"schema": "pcrl-r01-tstar-source-loader-verification-v1",
              "status": "HASH_ROLE_CLASS_MAP_PASS",
              "scope": "FIT/ATT/INNER role-only fine cells and fixed T* actions; no labels/X/mixed archive read; no fit",
              "independent_decoder_parity": "PENDING_R04",
              "source_audit_sha256": sha_file(audit), "seeds": out}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
