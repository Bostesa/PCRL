"""Read-only A historical score-source audit; never opens labels or assessments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .role_store import RoleOnlyStore, sha_array, sha_file
from .tstar_source import TStarSourceStore
from ..sampled_channel_v1.impl.bank import FixedBank, build_actions
from ..sampled_channel_v1.impl.release import _bank_hash
from ..sampled_channel_v1.impl.source_bridge import action_map_sha256


RELEASE = "U|JOINT|i8o64|l0.1|H-CLASS-TEMP"
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
BANK_FIELDS = {"probability", "token_class", "token_id", "input_class",
               "fine_to_hcal_token"}
TOKEN_FIELDS = {"row_id", "unit", "cell_1", "cell_2", "token_1", "token_2"}


def _archive(path: Path, fields: set[str]) -> dict:
    with ZipFile(path) as zip_file:
        if {member.filename for member in zip_file.infolist()} != {name + ".npy" for name in fields}:
            raise ValueError("score source member schema differs")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: np.array(arrays[name], copy=True) for name in fields}


def _binding_sha(binding: dict) -> str:
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def pstar_seed(portfolio: Path, cache: Path, store: RoleOnlyStore, seed: int) -> dict:
    root = portfolio / "shared/.private" / f"pstar_score_bank_s{seed}_v1"
    receipt_path = root / "BANK_ADMISSION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "pcrl-portfolio-a-score-bank-admission-v1"
            or receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED"
            or receipt.get("release_id") != RELEASE or receipt.get("seed") != seed
            or receipt.get("preprocessing_only") is not True
            or receipt.get("source_binding_sha256") != _binding_sha(receipt["source_binding"])):
        raise ValueError("P* source receipt identity differs")
    source = receipt["source_json_sha256"]
    paths = {
        "model.pt": cache / f"hcal_v1/admitted/rel__s{seed}__U/model.pt",
        "fine.json": cache / f"hcal_v1/admitted/units/fine__s{seed}/fine.json",
        "policy.json": cache / f"hcal_v1/admitted/units/pol__s{seed}__U_JOINT_i8o64_l0.1/policy.json",
        "decoder.json": cache / f"hcal_v1/package/decoder__s{seed}__U_JOINT_i8o64_l0.1_H-CLASS-TEMP.json",
    }
    expected = {"model.pt": source["teacher_model_sha256"],
                "fine.json": source["fine_sha256"],
                "policy.json": source["policy_sha256"],
                "decoder.json": source["decoder_file_sha256"]}
    if (receipt["source_binding"]["feature_names_sha256_newline_join"]
            != store.manifest["source_schema_sha256"]):
        raise ValueError("P* feature schema differs")
    if any(sha_file(path) != expected[name] for name, path in paths.items()):
        raise ValueError("P* checkpoint, partition, policy or decoder bytes differ")
    if expected["model.pt"] != ("5061a0074af74d21b53aca80b96d8b83b85a3e43cdb9e6758b2551be9045b9a7"
                                 if seed == 0 else
                                 "f52bef54ef77faf51cb6f55273e54f75ea094c4379bbf73b57a3941ce2f7a74c"):
        raise ValueError("P* source is not repaired U0")
    banks, bank_report = {}, {}
    for recipient, width in ((1, 2), (2, 6)):
        name = f"bank_r{recipient}.npz"
        path = root / name
        if sha_file(path) != receipt["private_output_sha256"][name]:
            raise ValueError("P* bank bytes differ")
        bank = _archive(path, BANK_FIELDS)
        p, d, ids = bank["probability"], bank["token_class"], bank["token_id"]
        cell_class, mapping = bank["input_class"], bank["fine_to_hcal_token"]
        bound = receipt["source_binding"]["recipients"][f"r{recipient}"]
        if (p.dtype != np.float64 or p.shape != (len(ids), width)
                or any(x.dtype != np.int64 or x.ndim != 1 for x in (d, ids, cell_class, mapping))
                or d.shape != ids.shape or mapping.shape != cell_class.shape
                or not np.array_equal(ids, np.arange(len(ids)))
                or not np.isfinite(p).all() or np.any(p < 0)
                or np.max(np.abs(p.sum(axis=1) - 1), initial=0) > 1e-8
                or not np.array_equal(np.argmax(p, axis=1), d)
                or np.any((mapping < 0) | (mapping >= len(ids)))
                or not np.array_equal(d[mapping], cell_class)
                or len({(row.tobytes(), int(dec)) for row, dec in zip(p, d)}) != len(ids)
                or sha_array(p) != bound["bank_probability_sha256"]
                or sha_array(d) != bound["token_class_sha256"]
                or sha_array(cell_class) != bound["input_class_sha256"]
                or sha_array(mapping) != bound["fine_to_hcal_token_sha256"]):
            raise ValueError("P* bank, class or source binding differs")
        banks[recipient] = bank
        bank_report[f"r{recipient}"] = {"path": str(path), "sha256": sha_file(path),
                                        "tokens": len(ids), "classes": width,
                                        "input_fine_cells": len(cell_class),
                                        "per_class_tokens": np.bincount(d, minlength=width).tolist()}
    role_report = {}
    for role in ROLES:
        name = f"tokens_{role}.npz"
        path = root / name
        if sha_file(path) != receipt["private_output_sha256"][name]:
            raise ValueError("P* role token bytes differ")
        rec = _archive(path, TOKEN_FIELDS)
        ref = store.read(role, f"references_s{seed}")
        bound = receipt["role_receipts"][role]
        pair = sha_array(np.column_stack((rec["row_id"], rec["unit"])))
        if (any(rec[key].dtype != np.int64 or rec[key].shape != ref["row_id"].shape
                for key in TOKEN_FIELDS)
                or not np.array_equal(rec["row_id"], ref["row_id"])
                or not np.array_equal(rec["unit"], ref["unit"])
                or pair != store.manifest["roles"][role]["row_unit_pair_sha256"]
                or pair != bound["row_unit_pair_sha256"]
                or bound["source_reference_sha256"] !=
                   store.manifest["role_file_sha256"][f"{role}:references_s{seed}"]):
            raise ValueError("P* role person/reference identity differs")
        for recipient in (1, 2):
            cell, token = rec[f"cell_{recipient}"], rec[f"token_{recipient}"]
            bank = banks[recipient]
            if (np.any((cell < 0) | (cell >= len(bank["input_class"])))
                    or np.any((token < 0) | (token >= len(bank["token_id"])))
                    or not np.array_equal(token, bank["fine_to_hcal_token"][cell])
                    or sha_array(cell) != bound[f"cell_{recipient}_sha256"]
                    or sha_array(token) != bound[f"token_{recipient}_sha256"]):
                raise ValueError("P* cell/token mapping differs")
        role_report[role] = {"path": str(path), "sha256": sha_file(path),
                             "rows": len(rec["row_id"]), "row_unit_pair_sha256": pair,
                             "source_reference_sha256": bound["source_reference_sha256"]}
    return {"receipt_path": str(receipt_path), "receipt_sha256": sha_file(receipt_path),
            "source_binding_sha256": receipt["source_binding_sha256"],
            "source_paths_sha256": {name: {"path": str(path), "sha256": expected[name]}
                                     for name, path in paths.items()},
            "banks": bank_report, "roles": role_report,
            "decoder_internal_sha256": receipt["decoder_sha256"],
            "policy_fingerprint": receipt["policy_fingerprint"]}


def tstar_bank_actions(portfolio: Path, manifest: Path, audit: Path) -> dict:
    out = {}
    for seed in (0, 1):
        source = TStarSourceStore(portfolio, manifest, audit, seed)
        recipients = {}
        for recipient in (1, 2):
            rec = source.read_bank(recipient)
            bank = FixedBank.from_source(rec["probability"], rec["token_id"],
                                         rec["token_class"])
            actions = build_actions(bank, input_class=rec["input_class"],
                                    fine_to_old_token=rec["fine_to_hcal_token"])
            recipients[f"r{recipient}"] = {
                "canonical_bank_sha256": _bank_hash(bank),
                "canonical_probability_array_sha256": sha_array(bank.probability),
                "action_map_sha256": action_map_sha256(actions),
                "canonical_outputs": len(bank.probability),
                "fine_action_rows": len(actions)}
        out[f"s{seed}"] = {"source_binding_sha256": source.source_binding_sha256,
                          "recipients": recipients}
    return out


D0_RELEASES = {
    "hcal_local_d0": ("U|LOCAL|i8o64|l0.1", "hcal_v1/admitted/units", "U_LOCAL_i8o64_l0.1"),
    "hcal_seq12_d0": ("U|SEQ-12|i8o64|l0.1", "hcal_v1/admitted/units", "U_SEQ-12_i8o64_l0.1"),
    "lra_seq21": ("U|SEQ-21|i8o64|l0.1", "lra_v1/run/units", "U_SEQ-21_i8o64_l0.1"),
}


def d0_seed(portfolio: Path, cache: Path, store: RoleOnlyStore,
            release_key: str, seed: int) -> dict:
    release_id, policy_parent, policy_stem = D0_RELEASES[release_key]
    root = portfolio / "shared/.private" / f"{release_key}_score_bank_s{seed}_v1"
    receipt_path = root / "BANK_ADMISSION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "pcrl-portfolio-a-score-bank-admission-v1"
            or receipt.get("status") != "ROLE_ONLY_BANK_ADMITTED"
            or receipt.get("release_id") != release_id or receipt.get("seed") != seed
            or receipt.get("preprocessing_only") is not True
            or receipt.get("source_binding_sha256") != _binding_sha(receipt["source_binding"])
            or receipt["source_json_sha256"]["decoder_file_sha256"] is not None):
        raise ValueError("D0 role-only source receipt differs")
    source = receipt["source_json_sha256"]
    paths = {
        "model.pt": cache / f"hcal_v1/admitted/rel__s{seed}__U/model.pt",
        "fine.json": cache / f"hcal_v1/admitted/units/fine__s{seed}/fine.json",
        "policy.json": cache / policy_parent / f"pol__s{seed}__{policy_stem}/policy.json",
    }
    expected = {"model.pt": source["teacher_model_sha256"],
                "fine.json": source["fine_sha256"],
                "policy.json": source["policy_sha256"]}
    if (receipt["source_binding"]["feature_names_sha256_newline_join"]
            != store.manifest["source_schema_sha256"]
            or any(sha_file(path) != expected[name] for name, path in paths.items())):
        raise ValueError("D0 model, partition, policy or predictor source differs")
    bank_report = {}
    for recipient, width in ((1, 2), (2, 6)):
        name = f"bank_r{recipient}.npz"
        path = root / name
        if sha_file(path) != receipt["private_output_sha256"][name]:
            raise ValueError("D0 bank archive differs")
        bank = _archive(path, BANK_FIELDS)
        p, d, ids = bank["probability"], bank["token_class"], bank["token_id"]
        mapping, cell_class = bank["fine_to_hcal_token"], bank["input_class"]
        bound = receipt["source_binding"]["recipients"][f"r{recipient}"]
        if (p.dtype != np.float64 or p.shape != (len(ids), width)
                or any(x.dtype != np.int64 or x.ndim != 1 for x in (d, ids, mapping, cell_class))
                or d.shape != ids.shape or mapping.shape != cell_class.shape
                or not np.array_equal(ids, np.arange(len(ids)))
                or not np.isfinite(p).all() or np.any(p < 0)
                or np.max(np.abs(p.sum(axis=1) - 1), initial=0) > 1e-8
                or not np.array_equal(np.argmax(p, axis=1), d)
                or np.any((mapping < 0) | (mapping >= len(ids)))
                or not np.array_equal(d[mapping], cell_class)
                or len({(row.tobytes(), int(dec)) for row, dec in zip(p, d)}) != len(ids)
                or sha_array(p) != bound["bank_probability_sha256"]
                or sha_array(d) != bound["token_class_sha256"]
                or sha_array(mapping) != bound["fine_to_hcal_token_sha256"]):
            raise ValueError("D0 bank, class or receipt array binding differs")
        bank_report[f"r{recipient}"] = {"path": str(path), "sha256": sha_file(path),
                                        "tokens": len(ids), "classes": width,
                                        "input_fine_cells": len(cell_class)}
    role_report = {}
    for role in ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION"):
        name = f"tokens_{role}.npz"
        path = root / name
        bound = receipt["role_receipts"][role]
        if (sha_file(path) != receipt["private_output_sha256"][name]
                or bound["row_unit_pair_sha256"] !=
                   store.manifest["roles"][role]["row_unit_pair_sha256"]
                or bound["source_reference_sha256"] !=
                   store.manifest["role_file_sha256"][f"{role}:references_s{seed}"]):
            raise ValueError("D0 role token/reference file binding differs")
        role_report[role] = {"path": str(path), "sha256": sha_file(path),
                             "rows": bound["rows"],
                             "row_unit_pair_sha256": bound["row_unit_pair_sha256"],
                             "source_reference_sha256": bound["source_reference_sha256"]}
    return {"release_id": release_id, "receipt_path": str(receipt_path),
            "receipt_sha256": sha_file(receipt_path),
            "source_binding_sha256": receipt["source_binding_sha256"],
            "source_paths_sha256": {name: {"path": str(path), "sha256": expected[name]}
                                     for name, path in paths.items()},
            "banks": bank_report, "roles": role_report,
            "calibration": "D0 original smoothed prototypes; no H-GLOBAL decoder"}


def public_centroids(cache: Path) -> dict:
    """Prove clean public-only centroid artifacts match pinned source partitions."""
    out = {}
    for seed in (0, 1):
        source = cache / f"hcal_v1/admitted/units/fine__s{seed}/fine.json"
        fine = json.loads(source.read_text())
        destination = Path(__file__).resolve().parent / f"A_PUBLIC_FINE_CENTROIDS_S{seed}.npz"
        fields = {"centroid_r1", "centroid_r2", "cell_class_r1", "cell_class_r2"}
        clean = _archive(destination, fields)
        for recipient, width, count in ((1, 2, 64), (2, 6, 641)):
            centroid = clean[f"centroid_r{recipient}"]
            classes = clean[f"cell_class_r{recipient}"]
            source_section = fine[f"fine{recipient}"]
            if (centroid.dtype != np.float64 or centroid.shape != (count, width)
                    or classes.dtype != np.int64 or classes.shape != (count,)
                    or not np.array_equal(centroid, np.asarray(source_section["centroid"],
                                                            dtype=np.float64))
                    or not np.array_equal(classes, np.asarray(source_section["cell_class"],
                                                           dtype=np.int64))
                    or not np.isfinite(centroid).all() or np.any(centroid < 0)
                    or np.max(np.abs(centroid.sum(axis=1) - 1)) > 1e-12
                    or not np.array_equal(np.argmax(centroid, axis=1), classes)):
                raise ValueError("clean public centroid source or schema differs")
        digest = hashlib.sha256()
        for recipient in (1, 2):
            value = np.ascontiguousarray(clean[f"centroid_r{recipient}"], dtype="<f8")
            digest.update(json.dumps(list(value.shape), sort_keys=True,
                                     separators=(",", ":")).encode() + b"\n")
            digest.update(value.tobytes())
        out[f"s{seed}"] = {
            "clean_path": str(destination), "clean_sha256": sha_file(destination),
            "source_fine_path": str(source), "source_fine_sha256": sha_file(source),
            "prototype_array_pair_sha256": digest.hexdigest(),
            "centroid_r1_sha256": sha_array(clean["centroid_r1"]),
            "centroid_r2_sha256": sha_array(clean["centroid_r2"]),
            "schema": {"centroid_r1": [64, 2], "centroid_r2": [641, 6],
                       "cell_class_r1": [64], "cell_class_r2": [641]},
            "excluded_from_clean_export": "FIT counts, means, assignment rows, labels and any person identity"}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--r01-source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    portfolio = args.portfolio.resolve(strict=True)
    store = RoleOnlyStore(args.repair_manifest)
    report = {"schema": "pcrl-r01-a-historical-score-source-audit-v1",
              "status": "ROLE_ONLY_HISTORICAL_SCORE_SOURCES_VERIFIED",
              "scope": "P* bank and ATT/INNER token/reference arrays; T* bank/action tables only; no X, labels, assessment or fit",
              "pstar_release_id": RELEASE,
              "pstar_complete_wire": "per recipient coarse token ID, fixed H-CLASS-TEMP probability vector and strict-argmax decision; fine cell remains private",
              "pstar": {f"s{seed}": pstar_seed(portfolio, args.cache.resolve(strict=True), store, seed)
                        for seed in (0, 1)},
              "tstar": tstar_bank_actions(portfolio, args.repair_manifest,
                                          args.r01_source_audit),
              "d0": {key: {f"s{seed}": d0_seed(portfolio, args.cache.resolve(strict=True),
                                               store, key, seed)
                            for seed in (0, 1)} for key in D0_RELEASES},
              "public_centroids": public_centroids(args.cache.resolve(strict=True)),
              "historical_control_status": {
                  "hcal_local_d0": "two-seed role-only bank/token exports exist; different uncalibrated D0 bank",
                  "hcal_seq12_d0": "two-seed role-only bank/token exports exist; different uncalibrated D0 bank",
                  "lra_seq21_d0": "two-seed role-only bank/token exports exist; different uncalibrated D0 bank",
                  "hcal_h_global_local_seq": "no LOCAL/SEQ H-GLOBAL decoder package for both seeds",
                  "prior_adult_protected_stochastic_a": "not fitted in predecessor; no historical release to replay"},
              "independent_pstar_qpc_decoder_parity": "PENDING_R04"}
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2,
                                      allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output_sha256": sha_file(args.output)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
