"""Aggregate-only exact-output deduplication of hash-admitted fixed T* banks.

Usage: python3 bank_identity_audit.py PORTFOLIO_WORKTREE
No per-person token files, source arrays, task labels, or SEX labels are read.
"""

import json
import sys
from pathlib import Path

import numpy as np

from audit_sources import member, sha_array, sha_file


def main():
    package = Path(sys.argv[1]) / "results/pcrl_method_portfolio_v1"
    output = {}
    for seed in (0, 1):
        root = package / f"shared/.private/tstar_score_bank_s{seed}_v1"
        receipt = json.loads((root / "BANK_ADMISSION_RECEIPT.json").read_text())
        seed_out = {}
        for recipient in (1, 2):
            name = f"r{recipient}"
            bank_path = root / f"bank_{name}.npz"
            assert sha_file(bank_path) == receipt["private_output_sha256"][bank_path.name]
            bank = member(bank_path, "probability")
            token_class = member(bank_path, "token_class")
            token_id = member(bank_path, "token_id")
            input_class = member(bank_path, "input_class")
            fine_to_token = member(bank_path, "fine_to_hcal_token")
            expected = receipt["source_binding"]["recipients"][name]
            for key, value in (("bank_probability_sha256", bank),
                               ("token_class_sha256", token_class),
                               ("input_class_sha256", input_class),
                               ("fine_to_hcal_token_sha256", fine_to_token)):
                assert sha_array(value) == expected[key]
            classes = 2 if recipient == 1 else 6
            assert bank.shape == (len(token_class), classes)
            assert np.array_equal(token_id, np.arange(len(bank)))
            assert input_class.shape == fine_to_token.shape
            assert np.all((fine_to_token >= 0) & (fine_to_token < len(bank)))
            decision = np.argmax(bank, axis=1)
            assert np.array_equal(decision, token_class)
            assert np.array_equal(input_class, token_class[fine_to_token])
            # Exact float64 bytes plus actual argmax decision define an output.
            keys = [(row.tobytes(), int(decision[index])) for index, row in enumerate(bank)]
            used = np.bincount(fine_to_token, minlength=len(bank))
            per_class = {}
            for cls in range(classes):
                positions = np.flatnonzero(token_class == cls)
                per_class[str(cls)] = {
                    "bank_token_ids": len(positions),
                    "canonical_outputs": len({keys[index] for index in positions}),
                    "input_fine_cells": int(np.count_nonzero(input_class == cls)),
                    "identity_map_tokens_used": int(np.count_nonzero(used[positions])),
                }
                assert per_class[str(cls)]["canonical_outputs"] >= 1
                assert per_class[str(cls)]["input_fine_cells"] >= 1
            seed_out[name] = {
                "input_fine_cells": len(input_class),
                "output_token_ids": len(bank),
                "canonical_output_vector_decision_pairs": len(set(keys)),
                "redundant_token_ids_after_exact_dedup": len(bank) - len(set(keys)),
                "identity_map_distinct_tokens_used": int(np.count_nonzero(used)),
                "identity_map_fine_cells_beyond_first_per_token": int(len(input_class) - np.count_nonzero(used)),
                "identity_map_tokens_with_multiple_fine_cells": int(np.count_nonzero(used >= 2)),
                "identity_map_max_fine_cells_per_token": int(used.max()),
                "per_class": per_class,
            }
        output[str(seed)] = seed_out
    print(json.dumps({"schema": "pcrl-r01-tstar-bank-identity-audit-v1",
                      "status": "HASH_ADMITTED_FIXED_TABLES_AGGREGATE_ONLY",
                      "independent_bank_decoder_parity": "PENDING",
                      "seeds": output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
