"""Hash-only parity of the T* public source, canonical actions, and Q maps.

This verifier reads fixed public bank/map exports and clean public centroids.
It never opens role tokens, person rows, FIT counts, labels, or assessment data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .role_store import sha_array, sha_file
from .tstar_source import TStarSourceStore
from ..sampled_channel_v1.impl.bank import FixedBank, build_actions
from ..sampled_channel_v1.impl.release import _bank_hash
from ..sampled_channel_v1.impl.source_bridge import action_map_sha256
from ..sampled_channel_v1.controls.diagnostic_codes import select_codes
from ..sampled_channel_v1.attacks.run_q_preprocess import build_fixed_public_policy


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
FIELDS = frozenset({"bank_r1", "bank_r2", "fine_to_code_r1", "fine_to_code_r2"})
CENTROID_FIELDS = frozenset({"centroid_r1", "centroid_r2", "cell_class_r1", "cell_class_r2"})


def _exact_npz(path: Path, fields: frozenset[str]) -> dict:
    with ZipFile(path) as archive:
        if {member.filename for member in archive.infolist()} != {
                name + ".npy" for name in fields}:
            raise ValueError("public archive member set differs")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in fields}


def _mismatches(left, right) -> int:
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        raise ValueError("public parity arrays have different shapes")
    return int(np.count_nonzero(a != b))


def verify(*, portfolio: Path, repair_manifest: Path) -> dict:
    source_audit = ROOT / "SOURCE_AUDIT.json"
    historical_path = ROOT / "A_HISTORICAL_SCORE_SOURCE_AUDIT.json"
    historical = json.loads(historical_path.read_text())
    output = {"schema": "pcrl-r01-a-diagnostic-public-parity-v1",
              "status": "PUBLIC_PARITY_PENDING",
              "scope": "fixed T* banks, clean public centroids, R13 canonical actions, R17 deterministic Q maps; no role/person/label/fit/assessment read",
              "source_audit_sha256": sha_file(source_audit),
              "historical_source_audit_sha256": sha_file(historical_path),
              "implementation_sha256": {
                  "r13_bank": sha_file(CAMPAIGN / "sampled_channel_v1/impl/bank.py"),
                  "r13_source_bridge": sha_file(CAMPAIGN / "sampled_channel_v1/impl/source_bridge.py"),
                  "r13_diagnostic_codes": sha_file(CAMPAIGN / "sampled_channel_v1/controls/diagnostic_codes.py"),
                  "r17_q_preprocess": sha_file(CAMPAIGN / "sampled_channel_v1/attacks/run_q_preprocess.py"),
                  "r01_tstar_source": sha_file(ROOT / "tstar_source.py"),
              },
              "seeds": {}}
    total_mismatch = 0
    for seed in (0, 1):
        source = TStarSourceStore(portfolio, repair_manifest, source_audit, seed)
        map_path = ROOT / f"A_FIXED_MAP_T_STAR_S{seed}.npz"
        map_receipt_path = map_path.with_suffix(".json")
        map_receipt = json.loads(map_receipt_path.read_text())
        cent_path = ROOT / f"A_PUBLIC_FINE_CENTROIDS_S{seed}.npz"
        cent_bound = historical["public_centroids"][f"s{seed}"]
        if (map_receipt["npz_sha256"] != sha_file(map_path) or
                map_receipt["source_binding_sha256"] != source.source_binding_sha256 or
                map_receipt["original_bank_receipt_sha256"] !=
                sha_file(source.root / "BANK_ADMISSION_RECEIPT.json") or
                map_receipt["source_audit_sha256"] != sha_file(source_audit) or
                cent_bound["clean_relative_path"] != f"r01/{cent_path.name}" or
                cent_bound["clean_sha256"] != sha_file(cent_path)):
            raise ValueError("T* public map/centroid source identity differs")
        public_map = _exact_npz(map_path, FIELDS)
        centroid = _exact_npz(cent_path, CENTROID_FIELDS)
        for name, value in public_map.items():
            if map_receipt["array_sha256"][name] != sha_array(value):
                raise ValueError("fixed public map array SHA differs")
        banks = (source.read_bank(1), source.read_bank(2))
        records = []
        computed = []
        seed_out = {"source_binding_sha256": source.source_binding_sha256,
                    "fixed_map_receipt_sha256": sha_file(map_receipt_path),
                    "fixed_map_npz_sha256": sha_file(map_path),
                    "centroid_npz_sha256": sha_file(cent_path),
                    "recipients": {}}
        for recipient, bank in ((1, banks[0]), (2, banks[1])):
            source_probability = bank["probability"]
            source_fine = bank["fine_to_hcal_token"]
            source_class = bank["input_class"]
            clean_probability = public_map[f"bank_r{recipient}"]
            clean_fine = public_map[f"fine_to_code_r{recipient}"]
            clean_class = centroid[f"cell_class_r{recipient}"]
            inferred_class = np.argmax(centroid[f"centroid_r{recipient}"], axis=1).astype(np.int64)
            canonical = FixedBank.from_source(source_probability,
                                              bank["token_id"], bank["token_class"])
            actions = build_actions(canonical, input_class=source_class,
                                    fine_to_old_token=source_fine)
            cell_ids = np.arange(len(source_fine), dtype=np.int64)
            expected_identity = np.asarray(canonical.old_to_global,
                                           dtype=np.int64)[source_fine]
            selected_identity = select_codes(canonical, actions, cell_ids,
                                             arm_id="A_IDENTITY")
            selected_class = select_codes(canonical, actions, cell_ids,
                                          arm_id="A_CLASS_ONLY")
            mismatches = {
                "source_vs_clean_bank_elements": _mismatches(source_probability, clean_probability),
                "source_vs_clean_fine_to_old_code": _mismatches(source_fine, clean_fine),
                "source_vs_centroid_class": _mismatches(source_class, clean_class),
                "source_vs_inferred_centroid_class": _mismatches(source_class, inferred_class),
                "old_to_canonical_identity_vs_r13_action": _mismatches(expected_identity, selected_identity),
            }
            report = {
                "fine_cells": int(len(source_fine)),
                "old_bank_rows": int(len(source_probability)),
                "canonical_bank_rows": int(len(canonical.probability)),
                "source_bank_array_sha256": sha_array(source_probability),
                "source_fine_to_old_code_sha256": sha_array(source_fine),
                "source_input_class_sha256": sha_array(source_class),
                "centroid_class_sha256": sha_array(clean_class),
                "old_to_global_sha256": sha_array(np.asarray(canonical.old_to_global, dtype=np.int64)),
                "canonical_bank_sha256": _bank_hash(canonical),
                "canonical_bank_array_sha256": sha_array(canonical.probability),
                "r13_action_map_sha256": action_map_sha256(actions),
                "r13_identity_code_sha256": sha_array(selected_identity),
                "r13_class_only_code_sha256": sha_array(selected_class),
                "mismatch_counts": mismatches,
            }
            expected = historical["tstar"][f"s{seed}"]["recipients"][f"r{recipient}"]
            if (report["canonical_bank_sha256"] != expected["canonical_bank_sha256"] or
                    report["canonical_bank_array_sha256"] != expected["canonical_probability_array_sha256"] or
                    report["r13_action_map_sha256"] != expected["action_map_sha256"]):
                raise ValueError("R13 canonical bank/action differs from R01 source audit")
            seed_out["recipients"][f"r{recipient}"] = report
            total_mismatch += sum(mismatches.values())
            records.append(bank)
            computed.append((canonical, actions))
        for arm in ("A_IDENTITY", "A_CLASS_ONLY"):
            _, _, law, descriptor = build_fixed_public_policy(
                arm_id=arm, seed=seed, source_kind="same_bank_diagnostic",
                records=tuple(records), release_id=f"{arm}-s{seed}-v1",
                source_binding_sha256=source.source_binding_sha256,
                source_receipt_sha256=sha_file(source.root / "BANK_ADMISSION_RECEIPT.json"))
            q_codes = []
            for recipient in (1, 2):
                size = len(banks[recipient - 1]["fine_to_hcal_token"])
                selected = np.empty(size, dtype=np.int64)
                for cell in range(size):
                    pair = law(None, None, cell if recipient == 1 else 0,
                               cell if recipient == 2 else 0)
                    if len(pair) != 1 or pair[0][2] != 1.0:
                        raise ValueError("R17 diagnostic public law is not deterministic")
                    selected[cell] = pair[0][recipient - 1]
                canonical, actions = computed[recipient - 1]
                expected_codes = select_codes(canonical, actions,
                                              np.arange(size, dtype=np.int64), arm_id=arm)
                count = _mismatches(selected, expected_codes)
                seed_out["recipients"][f"r{recipient}"]["mismatch_counts"][
                    f"r17_q_{arm.lower()}_vs_r13_codes"] = count
                seed_out["recipients"][f"r{recipient}"][f"r17_q_{arm.lower()}_code_sha256"] = sha_array(selected)
                total_mismatch += count
                q_codes.append(selected)
            seed_out[arm] = {
                "q_code_sha256_by_recipient": {"r1": sha_array(q_codes[0]),
                                                "r2": sha_array(q_codes[1])},
                "descriptor_mechanism_sha256": descriptor["mechanism_sha256"]}
        output["seeds"][f"s{seed}"] = seed_out
    output["total_mismatch_count"] = total_mismatch
    output["status"] = ("PUBLIC_TSTAR_R13_R17_DIAGNOSTIC_PARITY_PASS"
                        if total_mismatch == 0 else
                        "PUBLIC_TSTAR_R13_R17_DIAGNOSTIC_PARITY_FAIL")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    args = parser.parse_args()
    destination = ROOT / "A_DIAGNOSTIC_PUBLIC_PARITY_RECEIPT.json"
    if destination.exists():
        raise FileExistsError("diagnostic parity receipt is write-once")
    value = verify(portfolio=args.portfolio, repair_manifest=args.repair_manifest)
    destination.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": value["status"],
                      "total_mismatch_count": value["total_mismatch_count"],
                      "receipt_sha256": sha_file(destination)}, sort_keys=True))
    if value["total_mismatch_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
