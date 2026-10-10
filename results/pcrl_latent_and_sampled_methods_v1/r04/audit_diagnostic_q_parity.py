"""Public-only T* diagnostic/Q fine-cell parity, independently reconstructed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .independent_a_source import sha_file


FLOOR = 1e-12


def _archive(path: Path, expected_sha: str, keys: set[str]) -> dict[str, np.ndarray]:
    if sha_file(path) != expected_sha:
        raise ValueError(f"source SHA differs: {path.name}")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != keys:
            raise ValueError(f"source members differ: {path.name}")
        return {key: np.array(npz[key], copy=True) for key in npz.files}


def _canonical(probability: np.ndarray, token_id: np.ndarray,
               token_class: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (probability.dtype != np.float64 or probability.ndim != 2 or
            not np.isfinite(probability).all() or np.any(probability < 0) or
            not np.allclose(probability.sum(axis=1), 1, atol=1e-12, rtol=0) or
            not np.array_equal(token_id, np.arange(len(probability))) or
            not np.array_equal(token_class, np.argmax(probability, axis=1))):
        raise ValueError("source bank probability, token IDs or classes differ")
    q = np.array(probability, dtype="<f8", order="C", copy=True)
    q[q == 0] = 0.0
    classes = np.argmax(q, axis=1)
    if np.any(np.sum(q == q.max(axis=1, keepdims=True), axis=1) != 1):
        raise ValueError("non-unique source bank decision")
    keys = [(int(classes[i]), q[i].tobytes()) for i in range(len(q))]
    unique = sorted(set(keys))
    lookup = {key: i for i, key in enumerate(unique)}
    bank = np.stack([np.frombuffer(key[1], dtype="<f8").copy() for key in unique])
    decision = np.array([key[0] for key in unique], dtype=np.int64)
    old_to_global = np.array([lookup[key] for key in keys], dtype=np.int64)
    return bank, decision, old_to_global


def _distance(p: np.ndarray, q: np.ndarray) -> float:
    left, right = np.maximum(p, FLOOR), np.maximum(q, FLOOR)
    left, right = left / left.sum(), right / right.sum()
    return float(0.5 * (np.sum(left * np.log(left / right)) +
                        np.sum(right * np.log(right / left))))


def _actions(bank: np.ndarray, decision: np.ndarray, old_to_global: np.ndarray,
             input_class: np.ndarray, fine_to_old: np.ndarray) -> tuple[tuple[int, ...], ...]:
    if (input_class.dtype != np.int64 or fine_to_old.dtype != np.int64 or
            input_class.shape != fine_to_old.shape or
            np.any((fine_to_old < 0) | (fine_to_old >= len(old_to_global)))):
        raise ValueError("fine-cell class or old-token map differs")
    medoid = {}
    for label in np.unique(input_class):
        ids = np.flatnonzero(decision == label)
        medoid[int(label)] = min((sum(_distance(bank[i], bank[j]) for j in ids), int(i))
                                 for i in ids)[1]
    rows = []
    for label, old in zip(input_class, fine_to_old):
        identity = int(old_to_global[int(old)])
        if decision[identity] != label:
            raise ValueError("fine-cell identity code changes class")
        neighbors = [int(i) for i in np.flatnonzero(decision == label) if i != identity]
        neighbors.sort(key=lambda i: (_distance(bank[identity], bank[i]), i))
        rows.append(tuple(dict.fromkeys((identity, *neighbors[:2], medoid[int(label)]))))
    return tuple(rows)


def _digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    campaign = Path(args.campaign).resolve(strict=True)
    portfolio = Path(args.portfolio).resolve(strict=True)
    audit_path = campaign / "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"
    audit = json.loads(audit_path.read_text())
    source_audit_path = campaign / "r01/SOURCE_AUDIT.json"
    source_audit_sha = sha_file(source_audit_path)
    rows = []
    for seed in (0, 1):
        fixed_path = campaign / f"r01/A_FIXED_MAP_T_STAR_S{seed}.json"
        fixed = json.loads(fixed_path.read_text())
        if (fixed["source_audit_sha256"] != source_audit_sha or
                fixed["npz_sha256"] != sha_file(
                    campaign / f"r01/A_FIXED_MAP_T_STAR_S{seed}.npz")):
            raise ValueError("T* fixed public map source receipt differs")
        fixed_arrays = _archive(
            campaign / f"r01/A_FIXED_MAP_T_STAR_S{seed}.npz",
            fixed["npz_sha256"],
            {"bank_r1", "bank_r2", "fine_to_code_r1", "fine_to_code_r2"})
        clean_info = audit["public_centroids"][f"s{seed}"]
        clean = _archive(campaign / clean_info["clean_relative_path"],
                         clean_info["clean_sha256"],
                         {"centroid_r1", "centroid_r2", "cell_class_r1", "cell_class_r2"})
        bank_root = portfolio / f"shared/.private/tstar_score_bank_s{seed}_v1"
        if sha_file(bank_root / "BANK_ADMISSION_RECEIPT.json") != fixed["original_bank_receipt_sha256"]:
            raise ValueError("T* original bank receipt differs")
        for recipient in ("r1", "r2"):
            original = _archive(
                bank_root / f"bank_{recipient}.npz",
                fixed["original_bank_file_sha256_by_recipient"][recipient],
                {"probability", "token_class", "token_id", "input_class",
                 "fine_to_hcal_token"})
            centroid = clean[f"centroid_{recipient}"]
            if (not np.array_equal(original["probability"], fixed_arrays[f"bank_{recipient}"])
                    or not np.array_equal(original["fine_to_hcal_token"],
                                          fixed_arrays[f"fine_to_code_{recipient}"])
                    or not np.array_equal(clean[f"cell_class_{recipient}"],
                                          np.argmax(centroid, axis=1))
                    or not np.array_equal(original["input_class"],
                                          clean[f"cell_class_{recipient}"])):
                raise ValueError(f"T* bank/Q centroid cross-export differs: s{seed} {recipient}")
            bank, decision, old_to_global = _canonical(
                original["probability"], original["token_id"], original["token_class"])
            source_actions = _actions(bank, decision, old_to_global,
                                      original["input_class"],
                                      original["fine_to_hcal_token"])
            clean_actions = _actions(bank, decision, old_to_global,
                                     clean[f"cell_class_{recipient}"],
                                     fixed_arrays[f"fine_to_code_{recipient}"])
            if source_actions != clean_actions:
                raise ValueError("T* source/Q global canonical action IDs differ")
            identity = np.array([row[0] for row in source_actions], dtype=np.int64)
            source_old = original["fine_to_hcal_token"]
            if (not np.array_equal(identity, old_to_global[source_old]) or
                    not np.array_equal(bank[identity], original["probability"][source_old]) or
                    not np.array_equal(decision[identity], original["input_class"])):
                raise ValueError("T* canonical identity wire/probability/decision differs")
            class_medoid = {}
            for label in np.unique(decision):
                ids = np.flatnonzero(decision == label)
                class_medoid[int(label)] = min(
                    (sum(_distance(bank[i], bank[j]) for j in ids), int(i))
                    for i in ids)[1]
            class_only = np.array([class_medoid[int(decision[code])]
                                   for code in identity], dtype=np.int64)
            if any(int(code) not in row for code, row in zip(class_only, source_actions)):
                raise ValueError("T* diagnostic class-only code not in allowed action row")
            if (not np.array_equal(decision[class_only], original["input_class"]) or
                    not np.array_equal(np.argmax(bank[class_only], axis=1),
                                       original["input_class"])):
                raise ValueError("T* class-only wire/probability/decision differs")
            rows.append({"seed": seed, "recipient": recipient,
                         "fine_cells": len(identity), "canonical_bank_codes": len(bank),
                         "input_class_parity_mismatches": 0,
                         "source_to_clean_action_row_mismatches": 0,
                         "old_to_global_nonidentity_count":
                             int(np.count_nonzero(old_to_global != np.arange(len(old_to_global)))),
                         "identity_fine_old_vs_global_count":
                             int(np.count_nonzero(identity != source_old)),
                         "old_to_global_sha256": _digest(old_to_global),
                         "canonical_bank_probability_sha256": _digest(bank),
                         "identity_code_sha256": _digest(identity),
                         "class_only_code_sha256": _digest(class_only),
                         "source_bank_sha256": sha_file(bank_root / f"bank_{recipient}.npz"),
                         "fixed_map_sha256": fixed["npz_sha256"],
                         "clean_centroid_sha256": clean_info["clean_sha256"]})
    return {"schema": "pcrl-r04-public-diagnostic-q-parity-v1",
            "status": "PUBLIC_TSTAR_DIAGNOSTIC_Q_PARITY_PASS",
            "historical_source_audit_sha256": sha_file(audit_path),
            "source_audit_sha256": source_audit_sha,
            "scope": "public bank and fine-cell exports only; no person rows or labels",
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--public-receipt", required=True)
    args = parser.parse_args()
    output = Path(args.public_receipt)
    if output.exists():
        raise FileExistsError("diagnostic Q parity receipt already exists")
    result = run(args)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
